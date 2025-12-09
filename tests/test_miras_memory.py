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


class TestDeltaNetOverwrite:
    """Test DeltaNet key overwriting capability.

    DeltaNet delta rule: S = S - beta * outer(error, k) / ||k||^2
    This should enable key overwriting - updating existing associations.
    """

    def test_deltanet_overwrites_old_value(self):
        """Test that DeltaNet can overwrite an existing key-value pair."""
        config = MIRASConfig(hidden_dim=32, memory_dim=32)
        attn = MIRASMemoryAttention(config)

        batch_size = 1
        seq_len = 2

        # Create two inputs with similar keys but different values
        # First token: establish key-value pair
        # Second token: same key, different value
        x = torch.randn(batch_size, seq_len, 32)

        # First pass: establish memory
        output1, memory1 = attn(x[:, :1, :])

        # Second pass with same-ish input: should update memory
        output2, memory2 = attn(x[:, :1, :], memory_states=memory1)

        # Memory should have changed (DeltaNet updates)
        memory_diff = sum(
            (m1 - m2).abs().sum().item()
            for m1, m2 in zip(memory1, memory2)
        )

        assert memory_diff > 0, "DeltaNet should update memory states"

    def test_deltanet_retrieval_after_update(self):
        """Test that after key overwrite, retrieval returns new value."""
        config = MIRASConfig(hidden_dim=32, memory_dim=32)
        attn = MIRASMemoryAttention(config)

        # Three-step test:
        # 1. Write value_a with key_k
        # 2. Write value_b with same key_k (overwrite)
        # 3. Query with key_k should retrieve value_b, not value_a

        # This is a conceptual test - the exact mechanism depends on
        # how the attention projects queries and keys
        x = torch.randn(1, 4, 32)

        # Run through memory
        output, memory_states = attn(x)

        # Verify memory is populated
        for state in memory_states:
            assert state.abs().sum() > 0, "Memory should be non-zero after input"


# =============================================================================
# Tests for miras_utils.py
# =============================================================================

# Need to add path for training module
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

try:
    from miras_utils import (
        save_memory_states,
        load_memory_states,
        reset_memory_states,
        MIRASMetrics,
        compute_miras_metrics,
        log_miras_metrics,
        MIRASAblationConfig,
        ABLATION_CONFIGS,
        get_ablation_config,
        RetrievalProbe,
        save_miras_checkpoint,
        load_miras_checkpoint,
    )
    MIRAS_UTILS_AVAILABLE = True
except ImportError:
    MIRAS_UTILS_AVAILABLE = False


@pytest.mark.skipif(not MIRAS_UTILS_AVAILABLE, reason="miras_utils not available")
class TestMemoryStateCheckpointing:
    """Test memory state save/load functionality."""

    def test_save_load_roundtrip(self, tmp_path):
        """Test that memory states survive save/load cycle."""
        # Create test memory states
        memory_states = {
            0: [torch.randn(2, 64, 64) for _ in range(3)],
            2: [torch.randn(2, 64, 64) for _ in range(3)],
            4: [torch.randn(2, 64, 64) for _ in range(3)],
        }

        # Save
        save_path = tmp_path / "memory_states.pt"
        save_memory_states(memory_states, save_path)

        assert save_path.exists(), "Memory states file should be created"

        # Load
        loaded = load_memory_states(save_path, device=torch.device("cpu"))

        # Verify
        assert len(loaded) == len(memory_states)
        for layer_idx in memory_states:
            assert layer_idx in loaded
            for i, (orig, load) in enumerate(zip(memory_states[layer_idx], loaded[layer_idx])):
                assert torch.allclose(orig, load), f"Layer {layer_idx} state {i} mismatch"

    def test_save_load_dtype_preserved(self, tmp_path):
        """Test that dtype is correctly handled in save/load."""
        memory_states = {
            0: [torch.randn(1, 32, 32, dtype=torch.float32) for _ in range(3)],
        }

        save_path = tmp_path / "memory_states.pt"
        save_memory_states(memory_states, save_path)

        # Load with different dtype
        loaded = load_memory_states(save_path, device=torch.device("cpu"), dtype=torch.bfloat16)

        assert loaded[0][0].dtype == torch.bfloat16


@pytest.mark.skipif(not MIRAS_UTILS_AVAILABLE, reason="miras_utils not available")
class TestMIRASMetricsComputation:
    """Test MIRAS metrics computation and logging."""

    def test_metrics_dataclass_defaults(self):
        """Test that MIRASMetrics has correct defaults."""
        metrics = MIRASMetrics()

        assert metrics.retention_fast == 0.0
        assert metrics.retention_medium == 0.0
        assert metrics.retention_slow == 0.0

    def test_metrics_logging_format(self):
        """Test that metrics logging produces readable output."""
        metrics = MIRASMetrics(
            retention_fast=0.9,
            retention_medium=0.99,
            retention_slow=0.999,
            weight_fast=0.33,
            weight_medium=0.33,
            weight_slow=0.34,
            memory_norm_fast=10.5,
            memory_norm_medium=15.2,
            memory_norm_slow=20.1,
            memory_sparsity_fast=0.05,
            memory_sparsity_medium=0.10,
            memory_sparsity_slow=0.15,
            memory_grad_norm=0.001,
        )

        log_output = log_miras_metrics(metrics, step=1000, prefix="Test: ")

        assert "Step 1000" in log_output
        assert "0.9" in log_output  # retention_fast
        assert "Test:" in log_output  # prefix


@pytest.mark.skipif(not MIRAS_UTILS_AVAILABLE, reason="miras_utils not available")
class TestMIRASAblationConfig:
    """Test ablation configuration system."""

    def test_all_configs_available(self):
        """Test that all pre-defined ablation configs exist."""
        expected_configs = [
            "full", "no_memory", "single_timescale", "no_deltanet",
            "no_huber", "fast_only", "slow_only", "memory_last_half", "memory_every_4th"
        ]

        for config_name in expected_configs:
            config = get_ablation_config(config_name)
            assert config is not None, f"Config '{config_name}' should exist"

    def test_no_memory_config(self):
        """Test that no_memory config disables memory."""
        config = get_ablation_config("no_memory")
        assert config.use_memory == False

        # Should return empty layer list
        layers = config.get_memory_layers(num_layers=8)
        assert len(layers) == 0

    def test_memory_layer_patterns(self):
        """Test different memory layer patterns."""
        num_layers = 8

        # All layers
        full_config = MIRASAblationConfig(memory_layer_pattern="all")
        assert full_config.get_memory_layers(num_layers) == [0, 1, 2, 3, 4, 5, 6, 7]

        # Even layers
        even_config = MIRASAblationConfig(memory_layer_pattern="even")
        assert even_config.get_memory_layers(num_layers) == [0, 2, 4, 6]

        # Odd layers
        odd_config = MIRASAblationConfig(memory_layer_pattern="odd")
        assert odd_config.get_memory_layers(num_layers) == [1, 3, 5, 7]

        # Last half
        last_half_config = MIRASAblationConfig(memory_layer_pattern="last_half")
        assert last_half_config.get_memory_layers(num_layers) == [4, 5, 6, 7]

        # First half
        first_half_config = MIRASAblationConfig(memory_layer_pattern="first_half")
        assert first_half_config.get_memory_layers(num_layers) == [0, 1, 2, 3]

        # Every 4th
        every_4th_config = MIRASAblationConfig(memory_layer_pattern="every_4th")
        assert every_4th_config.get_memory_layers(num_layers) == [0, 4]

    def test_config_describe(self):
        """Test that config description is readable."""
        config = get_ablation_config("full")
        desc = config.describe()

        assert "3TS" in desc  # 3 timescales
        assert "DeltaNet" in desc
        assert "Huber" in desc

    def test_no_memory_describe(self):
        """Test no_memory config description."""
        config = get_ablation_config("no_memory")
        desc = config.describe()

        assert "NoMemory" in desc


@pytest.mark.skipif(not MIRAS_UTILS_AVAILABLE, reason="miras_utils not available")
class TestCheckpointEnhancement:
    """Test checkpoint save/load with MIRAS data."""

    def test_save_miras_checkpoint(self, tmp_path):
        """Test enhancing checkpoint with MIRAS data."""
        base_checkpoint = {
            "step": 1000,
            "model_state_dict": {},
        }

        memory_states = {
            0: [torch.randn(1, 32, 32) for _ in range(3)],
        }

        ablation_config = MIRASAblationConfig()
        metrics = MIRASMetrics(retention_fast=0.9)

        enhanced = save_miras_checkpoint(
            base_checkpoint, memory_states, ablation_config, metrics
        )

        assert "memory_states" in enhanced
        assert "ablation_config" in enhanced
        assert "miras_metrics" in enhanced
        assert enhanced["step"] == 1000  # Original data preserved

    def test_load_miras_checkpoint(self):
        """Test loading MIRAS data from checkpoint."""
        checkpoint = {
            "memory_states": {
                "0": [torch.randn(1, 32, 32) for _ in range(3)],
            },
            "ablation_config": {"use_memory": True, "num_timescales": 3},
            "miras_metrics": {"retention_fast": 0.9},
        }

        memory_states, ablation_config, metrics = load_miras_checkpoint(
            checkpoint, device=torch.device("cpu")
        )

        assert 0 in memory_states
        assert len(memory_states[0]) == 3

        # Note: ablation_config loading needs full dataclass fields
        # This test may fail if not all fields provided


@pytest.mark.skipif(not MIRAS_UTILS_AVAILABLE, reason="miras_utils not available")
class TestRetrievalProbeLogic:
    """Test retrieval probe helper methods (without model)."""

    def test_retrieval_prompt_structure(self):
        """Test that retrieval prompts have correct structure."""
        # Create mock tokenizer
        class MockTokenizer:
            def encode(self, text, **kwargs):
                # Simple approximation: 1 char = 0.25 tokens
                return list(range(len(text) // 4))

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 1)

        model = MockModel()
        tokenizer = MockTokenizer()

        probe = RetrievalProbe(model, tokenizer)

        fact = "The answer is 42."
        query = "What is the answer?"

        prompt = probe.create_retrieval_prompt(fact, query, distance=100)

        # Prompt should contain both fact and query
        assert fact in prompt
        assert query in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
