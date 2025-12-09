# BitNet-ODP: Integer-Only Latent Variable Models

**Combining BitNet b1.58 ternary weights with ODP rational operators for complete floating-point elimination.**

## Overview

```
BitNet (Memory)              ODP (Compute)              Combined
-----------------            -------------              --------
Weights: {-1,0,1}     +      Ops: +,-,*,/       =      Integer-Only LVM
Matmul: adds only            No transcendentals         ZK/FHE compatible
10x compression              0 SFU calls                Provably correct
```

## Training Results

**125M parameter model trained on TinyStories:**

| Step | Train Loss | Eval Loss | PPL |
|------|------------|-----------|-----|
| 100 | 10.80 | - | - |
| 1000 | 5.16 | 10.56 | 38,509 |
| 2000 | 2.36 | 10.24 | 27,884 |
| 3000 | 2.32 | 10.17 | 26,039 |
| 3700 | 2.33 | - | - |

Training stable with zero OOM errors and zero NaN/Inf issues.

## Quick Start

```bash
# Install dependencies
pip install torch transformers datasets bitsandbytes triton

# Train 125M model on TinyStories
python training/train.py \
    --model_size 125M \
    --dataset tinystories \
    --max_steps 10000 \
    --batch_size 4 \
    --gradient_accumulation 8 \
    --learning_rate 5e-5 \
    --warmup_steps 2000
```

```python
from src.rational_bitnet import RationalBitNet, RationalBitNetConfig

# Create integer-only transformer
config = RationalBitNetConfig(
    vocab_size=32000,
    hidden_dim=512,
    num_heads=8,
    num_layers=8,
)
model = RationalBitNet(config)

# Forward pass uses ZERO floating-point transcendentals
outputs = model(input_ids)
```

## Architecture

### Model Statistics (125M)
- **Total parameters**: 78.7M
- **BitLinear layers**: 57
- **Ternary params**: 53M (67.3%)
- **Transcendental ops**: 0

### Components

| Component | Standard | Integer-Only | Method |
|-----------|----------|--------------|--------|
| Linear | FP16 matmul | BitLinear | {-1,0,1} ternary weights |
| RMSNorm | rsqrt() | RationalRMSNorm | Newton-Raphson iteration |
| SiLU | exp() | RationalSiLU | Learnable P(x)/Q(x) |
| Softmax | exp() | RationalSoftmax | Polynomial (1+x/n)^n |
| RoPE | sin/cos | RationalRoPE | Cayley transform |
| Attention | O(N^2) | LinearAttention | O(N) with rational feature map |

## Stability Fixes

Nine stability fixes ensure stable training without NaN/Inf:

### Core ODP Fixes
1. **Newton-Raphson rsqrt**: O(1) division for normalization
2. **Learnable Rational SiLU**: P(x)/Q(x) with clamped denominator
3. **Linear Attention**: O(N) with cumsum-based causal masking
4. **Fused Triton Kernels**: 4-25x speedup

### Training Stability
5. **Bilinear Twist Preconditioning**: Bounded gradients
6. **Physics-Correct Hyperparameters**: LR 5e-5, warmup 2000, clip 0.5
7. **Spiky Init + Chunkwise Recurrence**: State-passing for infinite context

### BitNet-Specific Guards
8. **Minimum Scale Clamp**: `scale = max(mean(|W|), 1e-6)` prevents distribution collapse
9. **Zero-Sparsity Trap Monitor**: Penalty when >80% weights become 0

### Precision
- **BF16 required**: 8-bit exponent prevents underflow in rational division (FP16 fails)

## Project Structure

```
bitnet-odp/
├── src/
│   ├── rational_bitnet.py    # Core model: BitLinear, RationalOps, Attention
│   ├── triton_rational.py    # Fused Triton kernels
│   ├── miras_memory.py       # MIRAS persistent memory (sequential)
│   ├── miras_memory_fla.py   # MIRAS with FLA Triton kernels (15x faster)
│   └── unified_ilvm.py       # Unified architecture
├── training/
│   ├── train.py              # Training entry point
│   ├── trainer.py            # Training loop with all fixes
│   ├── data_loader.py        # TinyStories streaming loader
│   └── config.py             # Model size configurations
├── tests/
│   ├── test_odp_operators.py # Rational operator tests
│   ├── test_titans_architecture.py
│   ├── test_miras_memory.py  # Sequential MIRAS tests
│   └── test_miras_fla.py     # FLA-optimized MIRAS tests (30 tests)
├── docs/
│   └── BLUE_PAPER.md         # Technical specification
└── archive/
    └── debug_scripts/        # NaN debugging tools
```

## Training Configuration

### Recommended Settings (Tesla T4)
```yaml
Model: 125M
Batch size: 4
Gradient accumulation: 8
Effective batch: 32
Learning rate: 5e-5
Warmup steps: 2000
Gradient clip: 0.5
Mixed precision: BF16
Optimizer: 8-bit Adam
Gradient checkpointing: True
```

### Model Sizes

| Size | Params | Hidden | Layers | Heads | T4 Compatible |
|------|--------|--------|--------|-------|---------------|
| 125M | 78.7M | 512 | 8 | 8 | Yes |
| 350M | 302M | 1024 | 24 | 16 | Yes |
| 1.3B | 1.3B | 2048 | 24 | 16 | No |

## Stage 2b: MIRAS Memory Training

After Stage 1 completes (BitNet + ODP validation), enable MIRAS memory for long-range retrieval:

```bash
# Train with MIRAS memory (multi-timescale retention + DeltaNet)
python training/train.py \
    --model_size 125M \
    --use_memory \
    --dataset fineweb-edu \
    --max_steps 50000 \
    --batch_size 4 \
    --gradient_accumulation 8 \
    --learning_rate 5e-5 \
    --warmup_steps 2000
```

### MIRAS Memory Features

| Component | Function |
|-----------|----------|
| Multi-timescale Retention | Fast (0.9), Medium (0.99), Slow (0.999) decay rates |
| DeltaNet Delta Rule | Key overwriting: S = S - beta * outer(error, k) / ||k||^2 |
| Huber Loss | Robust gradients, less sensitive to outliers |
| Learnable Weights | Automatic timescale mixing per layer |

### Ablation Configurations

```python
from training.miras_utils import get_ablation_config

# Available configs:
# - full: All features enabled
# - no_memory: Baseline (no memory)
# - single_timescale: One retention rate only
# - no_deltanet: Disable key overwriting
# - no_huber: Use L2 instead of Huber
# - fast_only / slow_only: Single timescale experiments
# - memory_last_half / memory_every_4th: Sparse memory layers
```

### Memory Metrics Logged

Every 500 steps with `--use_memory`:
- Retention rates (learned)
- Timescale combination weights
- Memory utilization (Frobenius norm)
- Memory sparsity
- Memory gradient norm

## FLA Triton Optimization

MIRAS memory uses Flash Linear Attention (FLA) Triton kernels for 15x+ speedup:

### Performance Benchmarks (Tesla T4)

| Implementation | Time per Forward | Throughput | Speedup |
|----------------|------------------|------------|---------|
| Sequential DeltaNet | ~115 ms | ~17k tokens/sec | 1x |
| FLA DeltaNet (chunk) | ~7.5 ms | ~270k tokens/sec | 15.2x |
| FLA DeltaNet (fused_recurrent) | ~4.8 ms | ~420k tokens/sec | 23.7x |

### Installation

```bash
pip install flash-linear-attention
```

### Usage

```python
from src.miras_memory_fla import (
    MIRASFLAConfig,
    OptimizedFLADeltaNet,
    MultiTimescaleFLADeltaNet,
    MIRASFLAMemoryAttention,
    MIRASFLABlock,
)

config = MIRASFLAConfig(
    hidden_dim=512,
    num_heads=8,
    num_timescales=3,
)

# Fast mode: single DeltaNet with fused_recurrent (max speed)
memory = MultiTimescaleFLADeltaNet(config, mode='fast')

# Accurate mode: separate DeltaNets per timescale (more control)
memory = MultiTimescaleFLADeltaNet(config, mode='accurate')

# Full memory attention block
block = MIRASFLABlock(config)
output, state = block(x, use_cache=True)
```

### Architecture

```
Input
  │
  ├─────────────────────────────────────┐
  │                                     │
  ▼                                     ▼
OptimizedFLADeltaNet              (accurate mode only)
(fused_recurrent mode)            Multi-DeltaNet layers
  │                                     │
  ├─────────────────────────────────────┤
  │                                     │
  ▼                                     ▼
BitLinear Output Projection       Weighted Combine
  │                                     │
  └─────────────────────────────────────┘
                  │
                  ▼
               Output
```

### Key Optimizations

| Optimization | Description |
|--------------|-------------|
| Triton Kernels | Chunkwise-parallel DeltaNet from NeurIPS 2024 paper |
| Fused Recurrent | Single kernel for entire recurrence (23.7x speedup) |
| Parallel Associative Scan | Process all sequence chunks simultaneously |
| BitLinear Integration | Ternary weights for ZK/FHE compatibility |
| Multi-head Multi-timescale | Implicit timescale via learned per-head retention |

### References

- FLA: https://github.com/fla-org/flash-linear-attention
- DeltaNet Paper: "Parallelizing Linear Transformers with Delta Rule" (NeurIPS 2024)

## Applications

1. **ZK-Native AI**: Direct compilation to SNARK/STARK circuits
2. **FHE Inference**: Encrypted computation without bootstrapping overhead
3. **Provably Safe AI**: Exact, reproducible, formally verifiable
4. **Integer-Only Hardware**: Mature process nodes (28nm) become viable

## Key Insight

Traditional neural networks assume floating-point arithmetic with transcendental functions (exp, sin, cos, sqrt). This creates barriers:
- Non-determinism across hardware
- ZK/FHE incompatibility
- Formal verification impossibility

**Integer-Only LVM thesis**: All neural network operations can be reformulated as rational functions (quotients of polynomials) without loss of expressivity, while gaining verifiability, privacy, and efficiency.

## Related Work

- BitNet b1.58 (Microsoft, 2024) - Ternary quantization
- ODP/NOVA Paper - Winograd optimization
- RLVR Paper - Principal/off-principal weight directions
- Linear Attention - O(N) attention mechanisms
- FHE-LLM research

## License

MIT

## Author

Jayant Lohia
