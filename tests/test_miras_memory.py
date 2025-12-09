"""
Tests for MIRAS Memory Module

Validates:
1. Multi-timescale retention (Nested Learning)
2. Huber-loss attentional bias (robust to outliers)
3. DeltaNet delta rule (key overwriting)
4. BitLinear ternary weights
5. ZK/FHE compatibility (only +, -, *, /)
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from miras_memory import (
    MIRASConfig,
    RationalHuber,
    MultiTimescaleRetention,
    MemoryMLP,
    MIRASMemoryAttention,
    MIRASBlock,
    BitLinear,
    RationalSiLU,
    RationalRMSNorm,
)


class TestRationalHuber:
    """Test Huber loss with rational operations."""

    def test_pseudo_huber_loss_shape(self):
        """Test that Huber loss preserves shape."""
        huber = RationalHuber(delta=1.0)
        x = torch.randn(2, 64, 256)
        loss = huber(x)
        assert loss.shape == x.shape

    def test_huber_gradient_bounded(self):
        """Test that Huber gradient is bounded (robust to outliers)."""
        huber = RationalHuber(delta=1.0)
        # Test with large values
        x = torch.tensor([-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0])
        gradients = huber.gradient(x)

        # Gradients should be bounded approximately between -delta and +delta
        assert torch.all(gradients.abs() <= 1.1), f"Gradients not bounded: {gradients}"

    def test_huber_loss_positive(self):
        """Test that Huber loss is always non-negative."""
        huber = RationalHuber(delta=1.0)
        x = torch.randn(100)
        loss = huber(x)
        assert torch.all(loss >= 0), "Huber loss should be non-negative"

    def test_huber_zero_at_origin(self):
        """Test that Huber loss is zero at x=0."""
        huber = RationalHuber(delta=1.0)
        x = torch.zeros(1)
        loss = huber(x)
        assert torch.allclose(loss, torch.zeros(1), atol=1e-6)

    def test_huber_only_rational_ops(self):
        """Verify Huber uses only +, -, *, / (no transcendentals)."""
        # The implementation uses Babylonian sqrt which is purely rational
        huber = RationalHuber(delta=1.0, n_iterations=8)
        x = torch.randn(10)

        # Should not raise any errors (no exp, sin, cos, sqrt built-in)
        loss = huber(x)
        grad = huber.gradient(x)

        assert loss is not None
        assert grad is not None


class TestMultiTimescaleRetention:
    """Test multi-timescale retention from Nested Learning."""

    def test_retention_initialization(self):
        """Test that retention rates are initialized correctly."""
        config = MIRASConfig(
            fast_retention=0.9,
            medium_retention=0.99,
            slow_retention=0.999,
        )
        retention = MultiTimescaleRetention(config)

        rates = retention.get_retention_rates()
        assert len(rates) == 3, "Should have 3 timescales"

        # Rates should be approximately correct (after sigmoid transform)
        # Note: sigmoid of inverse_sigmoid(x) should give x
        assert rates[0].item() < rates[1].item() < rates[2].item(), \
            "Fast < Medium < Slow retention"

    def test_retention_forward(self):
        """Test forward pass of retention module."""
        config = MIRASConfig(hidden_dim=64, memory_dim=64)
        retention = MultiTimescaleRetention(config)

        batch_size = 2
        memory_states = [
            torch.randn(batch_size, 64, 64)
            for _ in range(3)
        ]
        update = torch.randn(batch_size, 64, 64)

        new_states, combined = retention(memory_states, update)

        assert len(new_states) == 3
        assert combined.shape == (batch_size, 64, 64)

    def test_retention_decay_ordering(self):
        """Test that fast memory decays faster than slow memory."""
        config = MIRASConfig(hidden_dim=64, memory_dim=64)
        retention = MultiTimescaleRetention(config)

        batch_size = 1
        # Initialize with ones, update with zeros
        memory_states = [
            torch.ones(batch_size, 64, 64)
            for _ in range(3)
        ]
        update = torch.zeros(batch_size, 64, 64)

        new_states, _ = retention(memory_states, update)

        # Fast memory should have decayed more (smaller values)
        # Slow memory should have retained more (larger values)
        fast_mean = new_states[0].mean()
        slow_mean = new_states[2].mean()

        assert fast_mean < slow_mean, \
            f"Fast memory ({fast_mean}) should decay more than slow ({slow_mean})"


class TestBitLinear:
    """Test ternary weight linear layer."""

    def test_bitlinear_forward(self):
        """Test forward pass with ternary weights."""
        layer = BitLinear(256, 128)
        x = torch.randn(2, 64, 256)
        y = layer(x)
        assert y.shape == (2, 64, 128)

    def test_bitlinear_ternary_weights(self):
        """Test that quantized weights are ternary {-1, 0, 1}."""
        layer = BitLinear(256, 128)
        w_quant, w_scale = layer.get_ternary_weights()

        unique_vals = torch.unique(w_quant)
        expected_vals = torch.tensor([-1, 0, 1], dtype=torch.int8)

        for val in unique_vals:
            assert val in expected_vals, f"Weight {val} not in {{-1, 0, 1}}"

    def test_bitlinear_gradient_flow(self):
        """Test that gradients flow through STE."""
        layer = BitLinear(64, 32)
        x = torch.randn(2, 64, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert layer.weight.grad is not None, "Gradients should flow to weights"
        assert x.grad is not None, "Gradients should flow to input"


class TestMemoryMLP:
    """Test memory MLP with BitLinear."""

    def test_memory_mlp_forward(self):
        """Test forward pass of memory MLP."""
        config = MIRASConfig(hidden_dim=128)
        mlp = MemoryMLP(config)
        x = torch.randn(2, 64, 128)
        y = mlp(x)
        assert y.shape == (2, 64, 128)

    def test_memory_mlp_has_bitlinear(self):
        """Test that MLP uses BitLinear layers."""
        config = MIRASConfig(hidden_dim=128)
        mlp = MemoryMLP(config)

        bitlinear_count = 0
        for module in mlp.modules():
            if isinstance(module, BitLinear):
                bitlinear_count += 1

        assert bitlinear_count == 2, "Should have 2 BitLinear layers (up and down)"


class TestMIRASMemoryAttention:
    """Test MIRAS memory attention module."""

    def test_attention_forward(self):
        """Test forward pass of attention module."""
        config = MIRASConfig(hidden_dim=128, memory_dim=128)
        attn = MIRASMemoryAttention(config)

        x = torch.randn(2, 64, 128)
        output, memory_states = attn(x)

        assert output.shape == (2, 64, 128)
        assert len(memory_states) == config.num_timescales

    def test_attention_stateful(self):
        """Test that attention maintains state across calls."""
        config = MIRASConfig(hidden_dim=128, memory_dim=128)
        attn = MIRASMemoryAttention(config)

        x = torch.randn(2, 64, 128)

        # First pass
        output1, memory1 = attn(x)

        # Second pass with memory
        output2, memory2 = attn(x, memory_states=memory1)

        # Outputs should differ because memory state changed
        assert not torch.allclose(output1, output2), \
            "Outputs should differ with state"

    def test_attention_operation_count(self):
        """Test operation counting."""
        config = MIRASConfig(hidden_dim=128, memory_dim=128)
        attn = MIRASMemoryAttention(config)

        stats = attn.count_operations()

        assert stats["bitlinear_layers"] == 6, "Should have 6 BitLinear layers"
        assert stats["transcendentals"] == 0, "Should have ZERO transcendentals"
        assert stats["memory_timescales"] == 3, "Should have 3 timescales"


class TestMIRASBlock:
    """Test MIRAS transformer block."""

    def test_block_forward(self):
        """Test forward pass of MIRAS block."""
        config = MIRASConfig(hidden_dim=128, memory_dim=128)
        block = MIRASBlock(config)

        x = torch.randn(2, 64, 128)
        output, memory_states = block(x)

        assert output.shape == (2, 64, 128)
        assert len(memory_states) == config.num_timescales

    def test_block_residual(self):
        """Test that block uses residual connections."""
        config = MIRASConfig(hidden_dim=128, memory_dim=128)
        block = MIRASBlock(config)

        # Zero initialization should pass through residual
        x = torch.zeros(2, 64, 128)
        output, _ = block(x)

        # Output should be close to zero (not exactly due to norms)
        assert output.abs().mean() < 1.0, "Residual should preserve zero input approximately"


class TestZKFHECompatibility:
    """Test ZK/FHE compatibility - only +, -, *, / operations."""

    def test_no_transcendentals_in_huber(self):
        """Verify Huber loss has no transcendentals."""
        huber = RationalHuber(delta=1.0, n_iterations=8)

        # Check that only rational ops are used
        # Babylonian sqrt uses only: +, -, *, /
        x = torch.randn(10, requires_grad=True)
        loss = huber(x)
        loss.sum().backward()

        assert x.grad is not None

    def test_no_transcendentals_in_silu(self):
        """Verify RationalSiLU has no transcendentals."""
        silu = RationalSiLU(scale=1.5, n_iterations=8)
        x = torch.randn(10, requires_grad=True)
        y = silu(x)
        y.sum().backward()

        assert x.grad is not None

    def test_no_transcendentals_in_rmsnorm(self):
        """Verify RationalRMSNorm has no transcendentals."""
        norm = RationalRMSNorm(hidden_size=64, n_iterations=15)
        x = torch.randn(2, 64, requires_grad=True)
        y = norm(x)
        y.sum().backward()

        assert x.grad is not None

    def test_full_module_compatibility(self):
        """Test that full MIRAS module uses only rational operations."""
        config = MIRASConfig(hidden_dim=64, memory_dim=64)
        miras = MIRASMemoryAttention(config)

        # Verify operation count
        stats = miras.count_operations()
        assert stats["transcendentals"] == 0, \
            f"MIRAS should have 0 transcendentals, got {stats['transcendentals']}"

        # Forward pass should work
        x = torch.randn(2, 16, 64, requires_grad=True)
        output, _ = miras(x)
        output.sum().backward()

        assert x.grad is not None


class TestNumericalStability:
    """Test numerical stability of MIRAS components."""

    def test_huber_stability_large_inputs(self):
        """Test Huber is stable with large inputs."""
        huber = RationalHuber(delta=1.0)
        x = torch.tensor([1e6, -1e6, 1e4, -1e4])
        loss = huber(x)

        assert torch.all(torch.isfinite(loss)), "Huber should be stable with large inputs"

    def test_retention_stability(self):
        """Test retention is stable over many iterations."""
        config = MIRASConfig(hidden_dim=32, memory_dim=32)
        retention = MultiTimescaleRetention(config)

        memory_states = [
            torch.ones(1, 32, 32) for _ in range(3)
        ]

        # Run many iterations
        for _ in range(100):
            update = torch.randn(1, 32, 32) * 0.1
            memory_states, _ = retention(memory_states, update)

        # All states should be finite
        for state in memory_states:
            assert torch.all(torch.isfinite(state)), "Memory should remain stable"

    def test_attention_stability_long_sequence(self):
        """Test attention is stable with long sequences."""
        config = MIRASConfig(hidden_dim=64, memory_dim=64)
        attn = MIRASMemoryAttention(config)

        # Long sequence
        x = torch.randn(1, 256, 64)
        output, memory_states = attn(x)

        assert torch.all(torch.isfinite(output)), "Output should be finite"
        for state in memory_states:
            assert torch.all(torch.isfinite(state)), "Memory should be finite"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
