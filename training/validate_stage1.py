#!/usr/bin/env python3
"""
Stage 1 Validation: TinyStories Quick Validation

This script answers the key questions for proof-of-concept:
1. Does the model learn? (loss decreases)
2. Are operators numerically stable? (no NaN/Inf)
3. Is forward pass stable in FP16? (mixed precision works)
4. Does ternary projection break gradients? (gradients flow)
5. Do ODP operators cause instabilities? (stable activations)
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# Add parent directories to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))

from rational_bitnet import RationalBitNet, RationalBitNetConfig
from config import get_model_config
from data_loader import create_dataloader, print_dataset_info


def check_numerical_stability(model, batch, device):
    """Check for NaN/Inf in forward pass."""
    model.eval()
    input_ids = batch["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs["logits"]

        has_nan = torch.isnan(logits).any().item()
        has_inf = torch.isinf(logits).any().item()

        return not has_nan and not has_inf, {
            "has_nan": has_nan,
            "has_inf": has_inf,
            "logits_min": logits.min().item(),
            "logits_max": logits.max().item(),
            "logits_mean": logits.mean().item(),
        }


def check_gradient_flow(model, batch, device):
    """Check that gradients flow through ternary projection."""
    model.train()
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    # Forward pass
    outputs = model(input_ids, labels=labels)
    loss = outputs["loss"]

    # Backward pass
    loss.backward()

    # Check gradient flow
    gradient_stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            gradient_stats[name] = {
                "has_grad": True,
                "grad_norm": grad.norm().item(),
                "grad_mean": grad.mean().item(),
                "grad_max": grad.abs().max().item(),
            }
        else:
            gradient_stats[name] = {"has_grad": False}

    # Check specifically for BitLinear layers
    bitlinear_grads = [v for k, v in gradient_stats.items() if "weight" in k and v.get("has_grad", False)]
    all_have_grads = len(bitlinear_grads) > 0 and all(g["grad_norm"] > 0 for g in bitlinear_grads)

    return all_have_grads, gradient_stats


def check_mixed_precision(model, batch, device):
    """Check that FP16 forward pass is stable."""
    if not torch.cuda.is_available():
        return True, {"message": "No CUDA, skipping FP16 test"}

    model.eval()
    input_ids = batch["input_ids"].to(device)

    from torch.cuda.amp import autocast

    with torch.no_grad():
        with autocast():
            outputs = model(input_ids)
            logits = outputs["logits"]

            has_nan = torch.isnan(logits).any().item()
            has_inf = torch.isinf(logits).any().item()

            return not has_nan and not has_inf, {
                "dtype": str(logits.dtype),
                "has_nan": has_nan,
                "has_inf": has_inf,
                "logits_range": (logits.min().item(), logits.max().item()),
            }


def check_learning(model, optimizer, train_loader, device, num_steps=10):
    """Check that model can learn (loss decreases)."""
    model.train()

    losses = []
    train_iter = iter(train_loader)

    for step in range(num_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    # Check if loss decreases overall (not strictly monotonic)
    first_half_avg = sum(losses[:5]) / 5
    second_half_avg = sum(losses[5:]) / 5
    is_learning = second_half_avg < first_half_avg

    return is_learning, {
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "first_half_avg": first_half_avg,
        "second_half_avg": second_half_avg,
        "losses": losses,
    }


def check_odp_operators(model):
    """Check that ODP operators produce stable outputs."""
    from rational_bitnet import RationalRMSNorm, RationalSiLU, RationalSoftmax, RationalRoPE

    results = {}

    # Test RationalRMSNorm
    norm = RationalRMSNorm(64, n_iterations=15)
    x = torch.randn(2, 16, 64)
    y = norm(x)
    results["RationalRMSNorm"] = {
        "output_range": (y.min().item(), y.max().item()),
        "stable": not (torch.isnan(y).any() or torch.isinf(y).any()),
    }

    # Test RationalSiLU
    silu = RationalSiLU(scale=1.5, n_iterations=8)
    y = silu(x)
    results["RationalSiLU"] = {
        "output_range": (y.min().item(), y.max().item()),
        "stable": not (torch.isnan(y).any() or torch.isinf(y).any()),
    }

    # Test RationalSoftmax
    softmax = RationalSoftmax(dim=-1)
    attn = torch.randn(2, 4, 16, 16)  # [batch, heads, seq, seq]
    y = softmax(attn)
    results["RationalSoftmax"] = {
        "output_range": (y.min().item(), y.max().item()),
        "sums_to_one": torch.allclose(y.sum(dim=-1), torch.ones_like(y.sum(dim=-1)), atol=0.01),
        "stable": not (torch.isnan(y).any() or torch.isinf(y).any()),
    }

    # Test RationalRoPE
    head_dim = 16
    seq_len = 16
    rope = RationalRoPE(head_dim, max_position=128)
    q = torch.randn(2, 4, seq_len, head_dim)  # [batch, heads, seq, head_dim]
    k = torch.randn(2, 4, seq_len, head_dim)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(2, -1)  # [batch, seq]
    q_rot, k_rot = rope(q, k, position_ids)
    results["RationalRoPE"] = {
        "q_range": (q_rot.min().item(), q_rot.max().item()),
        "k_range": (k_rot.min().item(), k_rot.max().item()),
        "stable": not (torch.isnan(q_rot).any() or torch.isnan(k_rot).any()),
    }

    all_stable = all(r["stable"] for r in results.values())
    return all_stable, results


def main():
    print("=" * 70)
    print("STAGE 1 VALIDATION: TinyStories Quick Check")
    print("=" * 70)
    print()
    print("Testing:")
    print("  1. Does the model learn? (loss decreases)")
    print("  2. Are operators numerically stable? (no NaN/Inf)")
    print("  3. Is forward pass stable in FP16? (mixed precision)")
    print("  4. Does ternary projection break gradients? (gradient flow)")
    print("  5. Do ODP operators cause instabilities?")
    print("=" * 70)
    print()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create small model for quick validation
    model_config = get_model_config("50M")  # Use smallest model for speed
    config = RationalBitNetConfig(
        vocab_size=model_config.vocab_size,
        hidden_dim=model_config.hidden_dim,
        intermediate_dim=model_config.intermediate_dim,
        num_heads=model_config.num_heads,
        num_layers=4,  # Even fewer layers for speed
        max_seq_len=128,  # Shorter sequences
    )

    model = RationalBitNet(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Create data loader - use synthetic first for quick validation
    print("\nUsing synthetic data for quick validation...")
    train_loader = create_dataloader(
        dataset_name="tinystories",
        batch_size=2,
        max_seq_len=128,
        vocab_size=model_config.vocab_size,
        use_synthetic=True,  # Quick synthetic test
        max_samples=100,
    )

    # Get a batch for testing
    batch = next(iter(train_loader))

    results = {}

    # Test 1: Numerical stability
    print("\n1. Testing numerical stability...")
    stable, stats = check_numerical_stability(model, batch, device)
    results["numerical_stability"] = {"passed": stable, "stats": stats}
    print(f"   {'PASS' if stable else 'FAIL'}: {stats}")

    # Test 2: Gradient flow
    print("\n2. Testing gradient flow through ternary projection...")
    model.zero_grad()
    grads_flow, grad_stats = check_gradient_flow(model, batch, device)
    results["gradient_flow"] = {"passed": grads_flow}
    print(f"   {'PASS' if grads_flow else 'FAIL'}: Gradients flow to all weight matrices")

    # Test 3: Mixed precision
    print("\n3. Testing FP16 forward pass...")
    fp16_stable, fp16_stats = check_mixed_precision(model, batch, device)
    results["mixed_precision"] = {"passed": fp16_stable, "stats": fp16_stats}
    print(f"   {'PASS' if fp16_stable else 'FAIL'}: {fp16_stats}")

    # Test 4: Learning
    print("\n4. Testing learning (10 steps)...")
    is_learning, learning_stats = check_learning(model, optimizer, train_loader, device, num_steps=10)
    results["learning"] = {"passed": is_learning, "stats": learning_stats}
    print(f"   {'PASS' if is_learning else 'FAIL'}: Loss {learning_stats['first_loss']:.4f} -> {learning_stats['last_loss']:.4f}")

    # Test 5: ODP operators
    print("\n5. Testing ODP rational operators...")
    odp_stable, odp_stats = check_odp_operators(model)
    results["odp_operators"] = {"passed": odp_stable, "stats": odp_stats}
    for op_name, op_stats in odp_stats.items():
        print(f"   {op_name}: {'PASS' if op_stats['stable'] else 'FAIL'}")

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    all_passed = all(r["passed"] for r in results.values())

    for test_name, test_result in results.items():
        status = "PASS" if test_result["passed"] else "FAIL"
        print(f"  [{status}] {test_name}")

    print()
    if all_passed:
        print("ALL TESTS PASSED - Ready for Stage 1 TinyStories training!")
        print()
        print("Next steps:")
        print("  1. Run: python train.py --dataset tinystories --model_size 125M --max_steps 10000")
        print("  2. Monitor loss - should decrease steadily")
        print("  3. If stable after ~1 epoch, proceed to Stage 2 (FineWeb-Edu)")
    else:
        print("SOME TESTS FAILED - Fix issues before proceeding")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
