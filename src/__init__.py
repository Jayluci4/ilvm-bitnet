"""
BitNet-ODP: Integer-Only Latent Variable Models

Combines:
- BitNet b1.58: {-1, 0, 1} ternary weights (additions only)
- ODP Rational: All activations/norms/softmax use only +, -, *, /

Result: A neural network that uses ZERO floating-point transcendental operations.
"""

from .rational_bitnet import (
    # Core components
    BitLinear,
    RationalRMSNorm,
    RationalSiLU,
    RationalSoftmax,
    RationalRoPE,

    # Complete model
    RationalBitNet,
    RationalBitNetConfig,
    RationalBitNetAttention,
    RationalBitNetMLP,
    RationalBitNetBlock,

    # Utilities
    ste_round,
    weight_quant_ternary,
    activation_quant_dynamic,
)

__version__ = "0.1.0"
__author__ = "Jayant Lohia"

__all__ = [
    # Core components
    "BitLinear",
    "RationalRMSNorm",
    "RationalSiLU",
    "RationalSoftmax",
    "RationalRoPE",

    # Complete model
    "RationalBitNet",
    "RationalBitNetConfig",
    "RationalBitNetAttention",
    "RationalBitNetMLP",
    "RationalBitNetBlock",

    # Utilities
    "ste_round",
    "weight_quant_ternary",
    "activation_quant_dynamic",
]
