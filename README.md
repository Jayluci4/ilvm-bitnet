# BitNet-ODP: Integer-Only Latent Variable Models

**Combining BitNet's ternary weights with ODP's rational operators for complete floating-point elimination.**

## Overview

```
BitNet (Memory)              ODP (Compute)              Combined
─────────────────            ─────────────              ────────
Weights: {-1,0,1}     +      Ops: +,-,*,/       =      Integer-Only LVM
Matmul: adds only            No transcendentals         ZK/FHE compatible
10x compression              0 SFU calls                 Provably correct
```

## What is a Latent Variable Model (LVM)?

Any model where:
1. **Inputs are mapped into a latent space** (encoding)
2. **Computations happen in that latent space** (processing)
3. **Outputs are generated from it** (decoding)

This includes all transformers, autoencoders, diffusion models, and language models.

## Quick Start

```python
from src.rational_bitnet import (
    RationalBitNet,
    RationalBitNetConfig,
)

# Create an integer-only transformer
config = RationalBitNetConfig(
    vocab_size=32000,
    hidden_dim=512,
    num_heads=8,
    num_layers=6,
)
model = RationalBitNet(config)

# Forward pass uses ZERO floating-point transcendentals
outputs = model(input_ids)
```

## Components

| Component | Standard | Integer-Only |
|-----------|----------|--------------|
| Linear | FP16 matmul | BitLinear (ternary weights) |
| RMSNorm | rsqrt() | Babylonian iteration |
| SiLU | exp() | Algebraic sigmoid |
| Softmax | exp() | Polynomial (1+x/4)^4 |
| RoPE | sin/cos | Cayley transform |

## Applications

1. **ZK-Native AI**: Direct compilation to SNARK/STARK circuits
2. **FHE Inference**: Encrypted computation without bootstrapping overhead
3. **Provably Safe AI**: Exact, reproducible, formally verifiable
4. **Integer-Only Hardware**: Mature process nodes (28nm) become viable

## Documentation

- [Blue Paper](docs/BLUE_PAPER.md) - Complete technical specification

## Project Structure

```
bitnet-odp/
├── src/
│   └── rational_bitnet.py    # Core implementation
├── docs/
│   └── BLUE_PAPER.md         # Technical specification
├── tests/                     # Test suite
└── examples/                  # Usage examples
```

## Key Insight

Traditional neural networks assume floating-point arithmetic. This creates barriers:
- Non-determinism across hardware
- ZK/FHE incompatibility
- Formal verification impossibility

**Integer-Only LVM thesis**: All neural network operations can be reformulated as rational functions (quotients of polynomials) without loss of expressivity, while gaining verifiability, privacy, and efficiency.

## Related Work

- BitNet b1.58 (Microsoft, 2024)
- ODP/NOVA Paper (Winograd optimization)
- FHE-LLM research

## Author

Jayant Lohia
