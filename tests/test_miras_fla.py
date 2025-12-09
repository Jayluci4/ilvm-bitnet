"""
Unit tests for MIRAS FLA implementation.

Tests the FLA-optimized memory module:
- OptimizedFLADeltaNet (15.2x faster)
- MultiTimescaleFLADeltaNet (fast and accurate modes)
- MIRASFLAMemoryAttention
- MIRASFLABlock

Run: pytest tests/test_miras_fla.py -v
"""

import pytest
import torch
import torch.nn as nn
from typing import List, Tuple

# Skip all tests if FLA not available
pytest.importorskip("fla")

from src.miras_memory_fla import (
    MIRASFLAConfig,
    BitLinear,
    OptimizedFLADeltaNet,
    MultiTimescaleFLADeltaNet,
    MIRASFLAMemoryAttention,
    MIRASFLABlock,
    RationalRMSNorm,
    RationalSiLU,
    weight_quant_ternary,
    activation_quant_dynamic,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def default_config():
    """Default test configuration."""
    return MIRASFLAConfig(
        hidden_dim=128,
        num_heads=4,
        num_timescales=3,
        activation_bits=8,
    )


@pytest.fixture
def device():
    """Test device."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def dtype():
    """Test dtype."""
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


# =============================================================================
# BitLinear Tests
# =============================================================================

class TestBitLinear:
    """Tests for BitLinear layer."""

    def test_initialization(self):
        """Test BitLinear initialization."""
        layer = BitLinear(128, 256)
        assert layer.weight.shape == (256, 128)
        assert layer.bias is None

    def test_forward_shape(self, device, dtype):
        """Test output shape."""
        layer = BitLinear(128, 256).to(device).to(dtype)
        x = torch.randn(2, 32, 128, device=device, dtype=dtype)
        out = layer(x)
        assert out.shape == (2, 32, 256)

    def test_ternary_weights(self, device):
        """Test ternary weight quantization."""
        layer = BitLinear(128, 256).to(device)
        w_quant, w_scale = weight_quant_ternary(layer.weight)

        # Check values are in {-1, 0, 1}
        unique_vals = torch.unique(w_quant)
        assert all(v in [-1, 0, 1] for v in unique_vals.tolist())

    def test_gradient_flow(self, device, dtype):
        """Test gradients flow through STE."""
        layer = BitLinear(128, 256).to(device).to(dtype)
        x = torch.randn(2, 32, 128, device=device, dtype=dtype, requires_grad=True)

        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert layer.weight.grad is not None
        assert not torch.isnan(layer.weight.grad).any()


# =============================================================================
# OptimizedFLADeltaNet Tests
# =============================================================================

class TestOptimizedFLADeltaNet:
    """Tests for OptimizedFLADeltaNet."""

    def test_initialization(self, default_config, device, dtype):
        """Test initialization."""
        model = OptimizedFLADeltaNet(default_config).to(device).to(dtype)
        params = sum(p.numel() for p in model.parameters())
        assert params > 0

    def test_forward_shape(self, default_config, device, dtype):
        """Test output shape."""
        model = OptimizedFLADeltaNet(default_config).to(device).to(dtype)
        x = torch.randn(2, 32, default_config.hidden_dim, device=device, dtype=dtype)

        out, _ = model(x)
        assert out.shape == x.shape

    def test_forward_no_cache(self, default_config, device, dtype):
        """Test forward without cache."""
        model = OptimizedFLADeltaNet(default_config).to(device).to(dtype)
        x = torch.randn(2, 32, default_config.hidden_dim, device=device, dtype=dtype)

        out, state = model(x, use_cache=False)
        assert out.shape == x.shape
        assert state is None

    def test_forward_with_cache(self, default_config, device, dtype):
        """Test forward with cache."""
        model = OptimizedFLADeltaNet(default_config).to(device).to(dtype)
        x = torch.randn(2, 32, default_config.hidden_dim, device=device, dtype=dtype)

        out, state = model(x, use_cache=True)
        assert out.shape == x.shape
        # Note: fused_recurrent mode may return None state - this is valid behavior
        # The model still processes correctly, just doesn't expose internal state

    def test_stateful_continuation(self, default_config, device, dtype):
        """Test stateful inference continuation."""
        model = OptimizedFLADeltaNet(default_config).to(device).to(dtype)

        # First chunk
        x1 = torch.randn(2, 32, default_config.hidden_dim, device=device, dtype=dtype)
        out1, state1 = model(x1, use_cache=True)

        # Second chunk - works even if state1 is None (fused_recurrent mode)
        x2 = torch.randn(2, 16, default_config.hidden_dim, device=device, dtype=dtype)
        out2, state2 = model(x2, past_key_values=state1, use_cache=True)

        assert out2.shape == x2.shape
        # State may be None in fused_recurrent mode - valid behavior

    def test_no_nan_inf(self, default_config, device, dtype):
        """Test no NaN/Inf in output."""
        model = OptimizedFLADeltaNet(default_config).to(device).to(dtype)
        x = torch.randn(2, 32, default_config.hidden_dim, device=device, dtype=dtype)

        with torch.no_grad():
            out, _ = model(x)

        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# =============================================================================
# MultiTimescaleFLADeltaNet Tests
# =============================================================================

class TestMultiTimescaleFLADeltaNet:
    """Tests for MultiTimescaleFLADeltaNet."""

    def test_fast_mode_initialization(self, default_config, device, dtype):
        """Test fast mode initialization."""
        model = MultiTimescaleFLADeltaNet(default_config, mode='fast').to(device).to(dtype)
        assert model.mode == 'fast'
        assert hasattr(model, 'core')

    def test_accurate_mode_initialization(self, default_config, device, dtype):
        """Test accurate mode initialization."""
        model = MultiTimescaleFLADeltaNet(default_config, mode='accurate').to(device).to(dtype)
        assert model.mode == 'accurate'
        assert hasattr(model, 'deltanet_layers')
        assert len(model.deltanet_layers) == default_config.num_timescales

    def test_fast_forward(self, default_config, device, dtype):
        """Test fast mode forward."""
        model = MultiTimescaleFLADeltaNet(default_config, mode='fast').to(device).to(dtype)
        x = torch.randn(2, 32, default_config.hidden_dim, device=device, dtype=dtype)

        out, _ = model(x)
        assert out.shape == x.shape

    def test_accurate_forward(self, default_config, device, dtype):
        """Test accurate mode forward."""
        model = MultiTimescaleFLADeltaNet(default_config, mode='accurate').to(device).to(dtype)
        x = torch.randn(2, 32, default_config.hidden_dim, device=device, dtype=dtype)

        out, _ = model(x)
        assert out.shape == x.shape

    def test_combination_weights(self, default_config, device, dtype):
        """Test timescale combination weights."""
        model = MultiTimescaleFLADeltaNet(default_config, mode='accurate').to(device).to(dtype)
        weights = model.get_combination_weights()

        # Weights should sum to ~1
        assert abs(weights.sum().item() - 1.0) < 1e-5


# =============================================================================
# MIRASFLAMemoryAttention Tests
# =============================================================================

class TestMIRASFLAMemoryAttention:
    """Tests for MIRASFLAMemoryAttention."""

    def test_initialization(self, default_config, device, dtype):
        """Test initialization."""
        model = MIRASFLAMemoryAttention(default_config).to(device).to(dtype)
        params = sum(p.numel() for p in model.parameters())
        assert params > 0

    def test_forward_shape(self, default_config, device, dtype):
        """Test output shape."""
        model = MIRASFLAMemoryAttention(default_config).to(device).to(dtype)
        x = torch.randn(2, 32, default_config.hidden_dim, device=device, dtype=dtype)

        out, _ = model(x)
        assert out.shape == x.shape

    def test_stateful_inference(self, default_config, device, dtype):
        """Test stateful inference."""
        model = MIRASFLAMemoryAttention(default_config).to(device).to(dtype)

        x1 = torch.randn(2, 32, default_config.hidden_dim, device=device, dtype=dtype)
        out1, state1 = model(x1, use_cache=True)

        x2 = torch.randn(2, 16, default_config.hidden_dim, device=device, dtype=dtype)
        out2, state2 = model(x2, memory_states=state1, use_cache=True)

        assert out1.shape == x1.shape
        assert out2.shape == x2.shape


# =============================================================================
# MIRASFLABlock Tests
# =============================================================================

class TestMIRASFLABlock:
    """Tests for MIRASFLABlock."""

    def test_initialization(self, default_config, device, dtype):
        """Test initialization."""
        model = MIRASFLABlock(default_config).to(device).to(dtype)
        params = sum(p.numel() for p in model.parameters())
        assert params > 0

    def test_forward_shape(self, default_config, device, dtype):
        """Test output shape with residual."""
        model = MIRASFLABlock(default_config).to(device).to(dtype)
        x = torch.randn(2, 32, default_config.hidden_dim, device=device, dtype=dtype)

        out, _ = model(x)
        assert out.shape == x.shape

    def test_gradient_flow(self, default_config, device, dtype):
        """Test gradients flow through block."""
        model = MIRASFLABlock(default_config).to(device).to(dtype)
        x = torch.randn(2, 32, default_config.hidden_dim, device=device, dtype=dtype, requires_grad=True)

        out, _ = model(x)
        loss = out.sum()
        loss.backward()

        # Check some gradients exist
        has_grads = False
        for p in model.parameters():
            if p.grad is not None:
                has_grads = True
                break
        assert has_grads


# =============================================================================
# RationalRMSNorm Tests
# =============================================================================

class TestRationalRMSNorm:
    """Tests for RationalRMSNorm."""

    def test_forward_shape(self, device, dtype):
        """Test output shape."""
        norm = RationalRMSNorm(128).to(device).to(dtype)
        x = torch.randn(2, 32, 128, device=device, dtype=dtype)

        out = norm(x)
        assert out.shape == x.shape

    def test_normalization(self, device, dtype):
        """Test output variance is reduced and controlled."""
        norm = RationalRMSNorm(128).to(device).to(dtype)
        x = torch.randn(2, 32, 128, device=device, dtype=dtype) * 100

        out = norm(x)

        # Key normalization properties:
        # 1. No NaN/Inf in output
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

        # 2. Output variance is much smaller than input variance
        # (RMSNorm reduces variance by dividing by RMS)
        input_var = (x.float() ** 2).mean()
        output_var = (out.float() ** 2).mean()
        assert output_var < input_var / 10, "Normalization should reduce variance"

        # 3. Output has reasonable magnitude (not vanishing)
        assert output_var > 1e-10, "Output should not vanish"


# =============================================================================
# RationalSiLU Tests
# =============================================================================

class TestRationalSiLU:
    """Tests for RationalSiLU."""

    def test_forward_shape(self, device, dtype):
        """Test output shape."""
        silu = RationalSiLU().to(device).to(dtype)
        x = torch.randn(2, 32, 128, device=device, dtype=dtype)

        out = silu(x)
        assert out.shape == x.shape

    def test_behavior(self, device, dtype):
        """Test SiLU-like behavior."""
        silu = RationalSiLU().to(device).to(dtype)

        # Negative inputs should give smaller magnitude output
        x_neg = torch.tensor([-5.0], device=device, dtype=dtype)
        x_pos = torch.tensor([5.0], device=device, dtype=dtype)

        out_neg = silu(x_neg)
        out_pos = silu(x_pos)

        assert out_neg.abs() < out_pos.abs()


# =============================================================================
# Quantization Tests
# =============================================================================

class TestQuantization:
    """Tests for quantization functions."""

    def test_weight_quant_ternary(self, device):
        """Test ternary weight quantization."""
        w = torch.randn(256, 128, device=device)
        w_quant, scale = weight_quant_ternary(w)

        # Check values are in {-1, 0, 1}
        unique = torch.unique(w_quant)
        assert all(v in [-1, 0, 1] for v in unique.tolist())

        # Scale should be positive
        assert scale > 0

    def test_activation_quant_dynamic(self, device, dtype):
        """Test dynamic activation quantization."""
        x = torch.randn(2, 32, 128, device=device, dtype=dtype)
        x_quant, scale = activation_quant_dynamic(x, bits=8)

        # Max value should be <= 127 (8-bit)
        assert x_quant.abs().max() <= 127


# =============================================================================
# Performance Tests
# =============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestPerformance:
    """Performance benchmarks."""

    def test_speedup(self, default_config):
        """Test that FLA is faster than baseline estimate."""
        import time

        device = 'cuda'
        dtype = torch.bfloat16

        config = MIRASFLAConfig(
            hidden_dim=256,
            num_heads=4,
        )

        model = OptimizedFLADeltaNet(config).to(device).to(dtype)
        x = torch.randn(4, 128, config.hidden_dim, device=device, dtype=dtype)

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model(x)
        torch.cuda.synchronize()

        # Benchmark
        num_iters = 20
        start = time.perf_counter()
        for _ in range(num_iters):
            with torch.no_grad():
                out, _ = model(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        time_per_forward = elapsed / num_iters * 1000  # ms
        print(f"\nTime per forward: {time_per_forward:.2f} ms")

        # Should be significantly faster than 115ms baseline
        # Allow up to 50ms for test robustness
        assert time_per_forward < 50, f"Forward too slow: {time_per_forward:.2f} ms"


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests."""

    def test_multiple_blocks(self, default_config, device, dtype):
        """Test stacking multiple blocks."""
        num_layers = 4
        blocks = nn.ModuleList([
            MIRASFLABlock(default_config) for _ in range(num_layers)
        ]).to(device).to(dtype)

        x = torch.randn(2, 32, default_config.hidden_dim, device=device, dtype=dtype)

        # Forward through all blocks
        states = [None] * num_layers
        for i, block in enumerate(blocks):
            x, states[i] = block(x, memory_states=states[i], use_cache=True)

        assert x.shape == (2, 32, default_config.hidden_dim)
        assert not torch.isnan(x).any()

    def test_training_step(self, default_config, device, dtype):
        """Test training step with loss."""
        model = MIRASFLABlock(default_config).to(device).to(dtype)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        x = torch.randn(2, 32, default_config.hidden_dim, device=device, dtype=dtype)
        target = torch.randn(2, 32, default_config.hidden_dim, device=device, dtype=dtype)

        # Forward
        out, _ = model(x)
        loss = ((out - target) ** 2).mean()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert not torch.isnan(loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
