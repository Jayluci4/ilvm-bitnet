"""
T4-Optimized Training Configuration

Memory budget for Tesla T4 (15.6GB):
- Model weights: ~500MB for 125M params in FP32
- Gradients: ~500MB (same as weights)
- Optimizer states (8-bit Adam): ~250MB (0.5x weights)
- Activations: Variable based on batch size and seq_len
- CUDA kernels/overhead: ~1GB

With gradient checkpointing, we can train 125M-350M models on T4.
"""

import sys
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class T4Config:
    """Training configuration optimized for Tesla T4 GPU (15.6GB VRAM)."""

    # Model size presets (choose based on experiment)
    model_size: str = "125M"  # Options: "50M", "125M", "350M"

    # Training hyperparameters
    learning_rate: float = 3e-5  # Rational networks need lower LR for stability
    weight_decay: float = 0.01
    warmup_steps: int = 2000  # Longer warmup for rational operators
    max_steps: int = 100000
    gradient_clip: float = 0.5  # Stricter clipping for rational network stability

    # Memory optimization
    gradient_accumulation_steps: int = 8
    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True  # FP16 for forward/backward
    use_8bit_adam: bool = True

    # Batch sizes (per GPU, before accumulation)
    batch_size: int = 4
    eval_batch_size: int = 8

    # Sequence length
    max_seq_len: int = 512

    # Data
    dataset_name: str = "c4"  # HuggingFace dataset
    dataset_config: str = "en"
    streaming: bool = True  # Stream to avoid memory issues

    # Logging
    log_every: int = 100
    eval_every: int = 1000
    save_every: int = 5000
    output_dir: str = "checkpoints"

    # Reproducibility
    seed: int = 42

    # OOM handling
    oom_retry_batch_size: int = 2  # Fallback batch size on OOM


@dataclass
class ModelConfig:
    """Model configuration for RationalBitNet."""
    vocab_size: int = 50257  # GPT-2 vocab size for fair comparison
    hidden_dim: int = 512
    intermediate_dim: int = 1536
    num_heads: int = 8
    num_layers: int = 6
    max_seq_len: int = 2048
    rope_base: float = 10000.0
    rms_norm_eps: float = 1e-6
    activation_bits: int = 8


def get_model_config(size: str = "125M") -> ModelConfig:
    """Get model configuration for a given size.

    Sizes designed to fit in T4 memory with training:
    - 50M: Very small, fast iteration, development (~36M actual params)
    - 125M: Small, fits comfortably, good for POC (~60M actual params)
    - 350M: Medium, tight fit, needs all optimizations (~167M actual params)
    - GPT2: True GPT-2 small equivalent (~162M actual params) - fits T4!
    """
    # GPT-2 vocab size (50257) for fair comparison with GPT-2 124M baseline
    # Using GPT-2 BPE tokenizer enables: consistent vocabulary, fair perplexity comparison
    GPT2_VOCAB_SIZE = 50257

    configs = {
        "50M": ModelConfig(
            vocab_size=GPT2_VOCAB_SIZE,
            hidden_dim=384,
            intermediate_dim=1152,
            num_heads=6,
            num_layers=6,
            max_seq_len=512,
        ),
        "125M": ModelConfig(
            vocab_size=GPT2_VOCAB_SIZE,
            hidden_dim=512,
            intermediate_dim=1536,
            num_heads=8,
            num_layers=8,
            max_seq_len=512,
        ),
        "350M": ModelConfig(
            vocab_size=GPT2_VOCAB_SIZE,
            hidden_dim=768,
            intermediate_dim=2304,
            num_heads=12,
            num_layers=12,
            max_seq_len=512,
        ),
        # True GPT-2 small equivalent - fits T4 with ~2.3GB training memory!
        "GPT2": ModelConfig(
            vocab_size=GPT2_VOCAB_SIZE,
            hidden_dim=768,
            intermediate_dim=3072,  # 4x hidden like original GPT-2
            num_heads=12,
            num_layers=12,
            max_seq_len=1024,
        ),
    }

    if size not in configs:
        raise ValueError(f"Unknown model size: {size}. Choose from {list(configs.keys())}")

    return configs[size]


def estimate_memory(config: ModelConfig) -> dict:
    """Estimate memory usage for a given configuration.

    Returns dict with memory estimates in GB.
    """
    # Parameter count
    embed_params = config.vocab_size * config.hidden_dim  # Embeddings
    lm_head_params = config.hidden_dim * config.vocab_size  # LM head (ternary)

    # Per layer
    qkv_params = 3 * config.hidden_dim * config.hidden_dim  # Q, K, V
    o_params = config.hidden_dim * config.hidden_dim  # Output projection
    mlp_params = 3 * config.hidden_dim * config.intermediate_dim  # gate, up, down
    norm_params = 2 * config.hidden_dim  # 2 RMSNorm per layer

    layer_params = qkv_params + o_params + mlp_params + norm_params
    total_params = embed_params + lm_head_params + config.num_layers * layer_params

    # Memory estimates (in GB)
    fp32_size = total_params * 4 / 1e9  # FP32 weights
    fp16_size = total_params * 2 / 1e9  # FP16 weights

    # Training memory (rough estimates)
    # With gradient checkpointing and 8-bit Adam:
    # - Weights: FP32 master weights
    # - Gradients: FP32
    # - Optimizer: ~0.5x weights (8-bit Adam)
    # - Activations: ~0.5-1x weights (with checkpointing)
    training_memory = fp32_size * 3.5  # Conservative estimate

    return {
        "total_params": total_params,
        "fp32_gb": fp32_size,
        "fp16_gb": fp16_size,
        "training_gb": training_memory,
        "fits_t4": training_memory < 14.0,  # Leave 1.6GB headroom
    }


def print_config_summary(train_config: T4Config, model_config: ModelConfig):
    """Print configuration summary."""
    memory = estimate_memory(model_config)

    print("=" * 70)
    print("I-LVM TRAINING CONFIGURATION")
    print("=" * 70)
    print()
    print(f"Model Size: {train_config.model_size}")
    print(f"  Parameters: {memory['total_params']:,}")
    print(f"  FP32 Size: {memory['fp32_gb']:.2f} GB")
    print(f"  Training Memory: {memory['training_gb']:.2f} GB")
    print(f"  Fits T4: {'Yes' if memory['fits_t4'] else 'NO - Consider smaller model'}")
    print()
    print("Model Architecture:")
    print(f"  Hidden dim: {model_config.hidden_dim}")
    print(f"  Intermediate dim: {model_config.intermediate_dim}")
    print(f"  Num heads: {model_config.num_heads}")
    print(f"  Num layers: {model_config.num_layers}")
    print(f"  Max seq len: {model_config.max_seq_len}")
    print()
    print("Training Settings:")
    print(f"  Learning rate: {train_config.learning_rate}")
    print(f"  Batch size: {train_config.batch_size}")
    print(f"  Gradient accumulation: {train_config.gradient_accumulation_steps}")
    print(f"  Effective batch: {train_config.batch_size * train_config.gradient_accumulation_steps}")
    print(f"  Gradient checkpointing: {train_config.use_gradient_checkpointing}")
    print(f"  Mixed precision: {train_config.use_mixed_precision}")
    print(f"  8-bit Adam: {train_config.use_8bit_adam}")
    print("=" * 70)


if __name__ == "__main__":
    # Test configuration
    train_config = T4Config()
    model_config = get_model_config(train_config.model_size)
    print_config_summary(train_config, model_config)
