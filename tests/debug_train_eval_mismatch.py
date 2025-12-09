"""
Diagnostic script to debug train/eval loss mismatch in BitNet-ODP.

Key diagnostics:
1. Check weight sparsity in all BitLinear layers
2. Test train vs eval mode produces same loss on identical input
3. Check if STE behavior differs between modes
4. Verify scale computation in weight_quant_ternary

Run: python tests/debug_train_eval_mismatch.py

"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import json


def load_model_from_checkpoint(checkpoint_path: str):
    """Load model from checkpoint without needing TrainingConfig class."""
    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load with weights_only=False to get all data
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Get model config from checkpoint
    model_config_dict = checkpoint.get('model_config', {})
    print(f"Model config: {model_config_dict}")

    # Import and create model
    from rational_bitnet import RationalBitNet, RationalBitNetConfig

    # Handle different config formats
    if hasattr(model_config_dict, '__dict__'):
        model_config_dict = model_config_dict.__dict__

    config = RationalBitNetConfig(
        vocab_size=model_config_dict.get('vocab_size', 32000),
        hidden_dim=model_config_dict.get('hidden_dim', 512),
        intermediate_dim=model_config_dict.get('intermediate_dim', 1536),
        num_heads=model_config_dict.get('num_heads', 8),
        num_layers=model_config_dict.get('num_layers', 8),
        max_seq_len=model_config_dict.get('max_seq_len', 512),
    )

    model = RationalBitNet(config)

    # Load state dict - handle wrapped model
    state_dict = checkpoint['model_state_dict']

    # Remove 'model.' prefix if present (from GradientCheckpointWrapper)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v  # Remove 'model.' prefix
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)

    return model, config


def analyze_bitlinear_sparsity(model: nn.Module) -> Dict[str, Dict]:
    """Analyze sparsity of all BitLinear layers."""
    from rational_bitnet import BitLinear, weight_quant_ternary

    results = {}
    total_ternary_params = 0
    total_zeros = 0
    total_ones = 0
    total_neg_ones = 0

    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            with torch.no_grad():
                w = module.weight
                w_quant, scale = weight_quant_ternary(w)

                # Count each value
                num_zeros = (w_quant == 0).sum().item()
                num_ones = (w_quant == 1).sum().item()
                num_neg_ones = (w_quant == -1).sum().item()
                total = w_quant.numel()

                sparsity = num_zeros / total

                results[name] = {
                    'shape': list(w.shape),
                    'scale': scale.item(),
                    'zeros': num_zeros,
                    'ones': num_ones,
                    'neg_ones': num_neg_ones,
                    'total': total,
                    'sparsity': sparsity,
                    'distribution': {
                        '-1': num_neg_ones / total,
                        '0': num_zeros / total,
                        '1': num_ones / total,
                    }
                }

                total_ternary_params += total
                total_zeros += num_zeros
                total_ones += num_ones
                total_neg_ones += num_neg_ones

    overall_sparsity = total_zeros / total_ternary_params if total_ternary_params > 0 else 0

    return {
        'layers': results,
        'summary': {
            'total_ternary_params': total_ternary_params,
            'total_zeros': total_zeros,
            'total_ones': total_ones,
            'total_neg_ones': total_neg_ones,
            'overall_sparsity': overall_sparsity,
            'overall_distribution': {
                '-1': total_neg_ones / total_ternary_params if total_ternary_params > 0 else 0,
                '0': total_zeros / total_ternary_params if total_ternary_params > 0 else 0,
                '1': total_ones / total_ternary_params if total_ternary_params > 0 else 0,
            }
        }
    }


def test_train_eval_consistency(model: nn.Module, device: str = 'cuda') -> Dict:
    """Test if model produces same output in train vs eval mode."""
    model = model.to(device)

    # Create fixed test input
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 64
    vocab_size = model.config.vocab_size

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()

    results = {}

    # Test in train mode
    model.train()
    with torch.no_grad():  # No gradients for fair comparison
        train_outputs = model(input_ids, labels=labels)
        train_loss = train_outputs['loss'].item()
        train_logits = train_outputs['logits'].clone()

    # Test in eval mode
    model.eval()
    with torch.no_grad():
        eval_outputs = model(input_ids, labels=labels)
        eval_loss = eval_outputs['loss'].item()
        eval_logits = eval_outputs['logits'].clone()

    # Compare
    logit_diff = (train_logits - eval_logits).abs()
    logit_diff_mean = logit_diff.mean().item()
    logit_diff_max = logit_diff.max().item()

    results['train_loss'] = train_loss
    results['eval_loss'] = eval_loss
    results['loss_diff'] = abs(train_loss - eval_loss)
    results['logit_diff_mean'] = logit_diff_mean
    results['logit_diff_max'] = logit_diff_max
    results['is_consistent'] = results['loss_diff'] < 0.01  # Should be nearly identical

    return results


def test_bitlinear_train_eval_behavior(model: nn.Module, device: str = 'cuda') -> Dict:
    """Test individual BitLinear layer behavior in train vs eval mode."""
    from rational_bitnet import BitLinear, weight_quant_ternary

    results = {}
    model = model.to(device)

    # Create test input
    torch.manual_seed(42)
    test_input = torch.randn(2, 64, model.config.hidden_dim, device=device, dtype=torch.float32)

    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            # Test in train mode
            module.train()
            with torch.no_grad():
                train_out = module(test_input.clone())

            # Test in eval mode
            module.eval()
            with torch.no_grad():
                eval_out = module(test_input.clone())

            # Compare
            diff = (train_out - eval_out).abs()
            diff_mean = diff.mean().item()
            diff_max = diff.max().item()

            results[name] = {
                'train_out_mean': train_out.mean().item(),
                'eval_out_mean': eval_out.mean().item(),
                'diff_mean': diff_mean,
                'diff_max': diff_max,
                'is_consistent': diff_max < 1e-5,
            }

            # Only test a few layers
            if len(results) >= 5:
                break

    return results


def analyze_weight_scale_distribution(model: nn.Module) -> Dict:
    """Analyze the distribution of weight scales across layers."""
    from rational_bitnet import BitLinear, weight_quant_ternary

    scales = []
    layer_info = []

    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            with torch.no_grad():
                w = module.weight
                _, scale = weight_quant_ternary(w)
                scales.append(scale.item())

                # Also check raw weight statistics
                layer_info.append({
                    'name': name,
                    'scale': scale.item(),
                    'weight_mean': w.abs().mean().item(),
                    'weight_std': w.std().item(),
                    'weight_min': w.min().item(),
                    'weight_max': w.max().item(),
                })

    return {
        'scales': scales,
        'scale_min': min(scales) if scales else 0,
        'scale_max': max(scales) if scales else 0,
        'scale_mean': sum(scales) / len(scales) if scales else 0,
        'layers': layer_info,
    }


def check_gradient_flow(model: nn.Module, device: str = 'cuda') -> Dict:
    """Check if gradients flow properly through BitLinear layers."""
    from rational_bitnet import BitLinear

    model = model.to(device)
    model.train()

    # Create test input
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 32
    vocab_size = model.config.vocab_size

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()

    # Forward pass
    outputs = model(input_ids, labels=labels)
    loss = outputs['loss']

    # Backward pass
    loss.backward()

    # Check gradients
    results = {}
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            if module.weight.grad is not None:
                grad = module.weight.grad
                results[name] = {
                    'has_grad': True,
                    'grad_mean': grad.mean().item(),
                    'grad_std': grad.std().item(),
                    'grad_max': grad.abs().max().item(),
                    'has_nan': torch.isnan(grad).any().item(),
                    'has_inf': torch.isinf(grad).any().item(),
                }
            else:
                results[name] = {
                    'has_grad': False,
                }

            # Only check a few layers
            if len(results) >= 5:
                break

    # Zero gradients
    model.zero_grad()

    return results


def create_fresh_model_and_compare():
    """Create a fresh model and compare train/eval behavior."""
    from rational_bitnet import RationalBitNet, RationalBitNetConfig

    config = RationalBitNetConfig(
        vocab_size=1000,  # Small vocab for testing
        hidden_dim=128,
        intermediate_dim=384,
        num_heads=4,
        num_layers=2,
        max_seq_len=64,
    )

    model = RationalBitNet(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Test consistency
    results = test_train_eval_consistency(model, device)

    return results


def main():
    print("=" * 70)
    print("BITNET-ODP TRAIN/EVAL MISMATCH DIAGNOSTIC")
    print("=" * 70)
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # First test with fresh model
    print("\n" + "=" * 50)
    print("TEST 1: Fresh Model Train/Eval Consistency")
    print("=" * 50)

    fresh_results = create_fresh_model_and_compare()
    print(f"Train Loss: {fresh_results['train_loss']:.6f}")
    print(f"Eval Loss: {fresh_results['eval_loss']:.6f}")
    print(f"Loss Diff: {fresh_results['loss_diff']:.6f}")
    print(f"Logit Diff Mean: {fresh_results['logit_diff_mean']:.6f}")
    print(f"Logit Diff Max: {fresh_results['logit_diff_max']:.6f}")
    print(f"Consistent: {fresh_results['is_consistent']}")

    if not fresh_results['is_consistent']:
        print("\nWARNING: Fresh model is NOT consistent between train/eval modes!")
        print("This indicates a fundamental issue with BitLinear or other layers.")

    # Try to load checkpoint
    checkpoint_paths = [
        '/home/jayantlohia16/experiment/gemma-intelligent/conv/src/bitnet-odp/checkpoints/checkpoint_best.pt',
        '/home/jayantlohia16/experiment/gemma-intelligent/conv/src/bitnet-odp/checkpoints/checkpoint_latest.pt',
        '/home/jayantlohia16/experiment/gemma-intelligent/conv/src/checkpoints/test/best_model.pt',
    ]

    checkpoint_path = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint_path = path
            break

    if checkpoint_path:
        print("\n" + "=" * 50)
        print("TEST 2: Trained Model Weight Sparsity Analysis")
        print("=" * 50)

        try:
            model, config = load_model_from_checkpoint(checkpoint_path)
            model = model.to(device)

            # Analyze sparsity
            sparsity_results = analyze_bitlinear_sparsity(model)

            print(f"\nOverall Sparsity: {sparsity_results['summary']['overall_sparsity']*100:.2f}%")
            print(f"Distribution: {sparsity_results['summary']['overall_distribution']}")

            if sparsity_results['summary']['overall_sparsity'] > 0.9:
                print("\nCRITICAL: Model has >90% zero weights - likely collapsed!")
            elif sparsity_results['summary']['overall_sparsity'] > 0.8:
                print("\nWARNING: Model has >80% zero weights - approaching collapse!")

            # Print per-layer sparsity (first 10 layers)
            print("\nPer-layer sparsity (first 10):")
            for i, (name, info) in enumerate(list(sparsity_results['layers'].items())[:10]):
                print(f"  {name}: {info['sparsity']*100:.1f}% zeros, scale={info['scale']:.6f}")

            # Test train/eval consistency
            print("\n" + "=" * 50)
            print("TEST 3: Trained Model Train/Eval Consistency")
            print("=" * 50)

            consistency_results = test_train_eval_consistency(model, device)
            print(f"Train Loss: {consistency_results['train_loss']:.6f}")
            print(f"Eval Loss: {consistency_results['eval_loss']:.6f}")
            print(f"Loss Diff: {consistency_results['loss_diff']:.6f}")
            print(f"Consistent: {consistency_results['is_consistent']}")

            if not consistency_results['is_consistent']:
                print("\nINVESTIGATING: Testing individual BitLinear layers...")
                bitlinear_results = test_bitlinear_train_eval_behavior(model, device)
                for name, info in bitlinear_results.items():
                    print(f"  {name}: train={info['train_out_mean']:.6f}, eval={info['eval_out_mean']:.6f}, diff={info['diff_max']:.6f}")

            # Analyze weight scales
            print("\n" + "=" * 50)
            print("TEST 4: Weight Scale Distribution")
            print("=" * 50)

            scale_results = analyze_weight_scale_distribution(model)
            print(f"Scale range: [{scale_results['scale_min']:.6f}, {scale_results['scale_max']:.6f}]")
            print(f"Scale mean: {scale_results['scale_mean']:.6f}")

            if scale_results['scale_min'] < 1e-5:
                print("\nWARNING: Some scales are very small - may cause quantization issues!")

            # Check gradient flow
            print("\n" + "=" * 50)
            print("TEST 5: Gradient Flow Check")
            print("=" * 50)

            grad_results = check_gradient_flow(model, device)
            for name, info in grad_results.items():
                if info['has_grad']:
                    status = "OK" if not info['has_nan'] and not info['has_inf'] else "PROBLEM"
                    print(f"  {name}: {status}, grad_mean={info['grad_mean']:.6f}, grad_max={info['grad_max']:.6f}")
                else:
                    print(f"  {name}: NO GRADIENT")

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nNo checkpoint found. Run training first.")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)

    # Summary
    print("\nSUMMARY:")
    print("- If fresh model is inconsistent: Bug in BitLinear/RationalOps forward pass")
    print("- If trained model has >90% zeros: Model collapsed (scale clamp issue)")
    print("- If scales are very small: Weight distribution collapsed")
    print("- If train/eval differ on same input: STE or mode-dependent behavior")


if __name__ == "__main__":
    main()
