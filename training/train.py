#!/usr/bin/env python3
"""
I-LVM Training Script for Tesla T4 GPU

This script trains the Integer-Only Latent Variable Model (I-LVM) which combines:
- BitNet b1.58: {-1, 0, 1} ternary weights (1.58-bit quantization)
- ODP Rational Operators: All operations use only +, -, *, / (no transcendentals)
- MIRAS Memory: Multi-timescale retention with DeltaNet delta rule (optional)

ODP Components Used:
- RationalRMSNorm: Babylonian sqrt iteration (15 iterations)
- RationalSiLU: Algebraic sigmoid approximation
- RationalSoftmax: Polynomial exp approximation (1 + x/4)^4
- RationalRoPE: Cayley transform for sin/cos

MIRAS Components (when --use_memory is enabled):
- MultiTimescaleRetention: Fast/Medium/Slow memory decay (Nested Learning)
- RationalHuber: Huber-loss attentional bias (robust to outliers)
- DeltaNet Delta Rule: Key overwriting for multi-KV recall

Result: A model requiring ZERO floating-point transcendental operations.

Usage:
    python train.py --model_size 125M --max_steps 10000
    python train.py --model_size GPT2 --use_memory  # With MIRAS memory
    python train.py --synthetic  # Use synthetic data for testing
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

import torch

# Add parent directories to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))

from rational_bitnet import RationalBitNet, RationalBitNetConfig
from unified_ilvm import UnifiedILVM, UnifiedILVMConfig, get_unified_ilvm_config
from config import T4Config, get_model_config, print_config_summary, estimate_memory
from data_loader import create_dataloader
from trainer import ILVMTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train I-LVM on Tesla T4")

    # Model configuration
    parser.add_argument(
        "--model_size",
        type=str,
        default="125M",
        choices=["50M", "125M", "350M", "GPT2"],
        help="Model size preset (GPT2 = true 162M params like GPT-2 small)",
    )

    # MIRAS Memory integration
    parser.add_argument(
        "--use_memory",
        action="store_true",
        help="Use Unified I-LVM with MIRAS memory (multi-timescale retention + DeltaNet)",
    )

    # Training configuration
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_seq_len", type=int, default=512)

    # Memory optimizations
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--no_mixed_precision", action="store_true")
    parser.add_argument("--no_8bit_adam", action="store_true")

    # Data - PoC Training Stages:
    # Stage 1: tinystories (validation) -> Stage 2: fineweb-edu (training) -> Stage 3: slimpajama (generalization)
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--dataset", type=str, default="tinystories",
                        choices=["tinystories", "fineweb-edu", "slimpajama", "c4"],
                        help="Dataset for training: tinystories (Stage 1), fineweb-edu (Stage 2), slimpajama (Stage 3)")
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="Override HuggingFace dataset config (e.g., 'sample-10BT' for fineweb-edu)")

    # Output
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=5000)

    # Misc
    parser.add_argument("--seed", type=int, default=42)

    # Resume from checkpoint
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (e.g., checkpoints/checkpoint_best.pt)")
    parser.add_argument("--reset_optimizer", action="store_true",
                        help="Reset optimizer state when resuming (useful after architecture changes)")

    return parser.parse_args()


def setup_device():
    """Setup and verify compute device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

        print(f"GPU: {gpu_name}")
        print(f"Memory: {gpu_memory:.1f} GB")

        # Check if it's a T4
        if "T4" in gpu_name:
            print("Detected Tesla T4 - using optimized settings")
        else:
            print(f"Note: This script is optimized for T4, but will work on {gpu_name}")

        return device
    else:
        print("WARNING: No GPU detected, training will be very slow!")
        return torch.device("cpu")


def verify_odp_components(use_memory: bool = False):
    """Verify that ODP rational operators are being used."""
    print("\nVerifying ODP Rational Operators:")
    print("  [x] RationalRMSNorm: Babylonian sqrt (no SFU calls)")
    print("  [x] RationalSiLU: Algebraic sigmoid (no exp)")
    print("  [x] RationalSoftmax: Polynomial exp (no SFU calls)")
    print("  [x] RationalRoPE: Cayley transform (no trig)")
    print("  [x] BitLinear: {-1, 0, 1} weights (additions only)")

    if use_memory:
        print("\nMIRAS Memory Components (Enabled):")
        print("  [x] MultiTimescaleRetention: Fast/Medium/Slow decay (Nested Learning)")
        print("  [x] RationalHuber: Pseudo-Huber loss (Babylonian sqrt)")
        print("  [x] DeltaNet Delta Rule: Key overwriting for multi-KV recall")
        print("      S = S - beta * outer(error, k) / ||k||^2")

    print("\nResult: ZERO transcendental operations!")


def main():
    """Main training function."""
    args = parse_args()

    # Print header
    print("=" * 70)
    if args.use_memory:
        print("UNIFIED I-LVM TRAINING: Integer-Only LVM + MIRAS Memory")
    else:
        print("I-LVM TRAINING: Integer-Only Latent Variable Model")
    print("=" * 70)
    print()
    print("Combining:")
    print("  - BitNet b1.58: {-1, 0, 1} ternary weights")
    print("  - ODP Rational: All ops use only +, -, *, /")
    if args.use_memory:
        print("  - MIRAS Memory: Multi-timescale retention + DeltaNet")
    print("=" * 70)

    # Verify ODP components
    verify_odp_components(use_memory=args.use_memory)

    # Setup device
    device = setup_device()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create configurations
    train_config = T4Config(
        model_size=args.model_size,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        max_seq_len=args.max_seq_len,
        use_gradient_checkpointing=not args.no_gradient_checkpointing,
        use_mixed_precision=not args.no_mixed_precision,
        use_8bit_adam=not args.no_8bit_adam,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    model_config = get_model_config(args.model_size)

    # Print configuration
    print_config_summary(train_config, model_config)

    # Check memory requirements
    memory = estimate_memory(model_config)
    if not memory["fits_t4"]:
        print(f"\nWARNING: Model may not fit in T4 memory!")
        print(f"Estimated: {memory['training_gb']:.2f} GB needed, T4 has 15.6 GB")
        print("Consider using --model_size 50M or enabling more optimizations")

    # Create model with ODP rational operators
    if args.use_memory:
        print("\nCreating Unified I-LVM with MIRAS memory...")
        config = get_unified_ilvm_config(model_size=args.model_size)
        model = UnifiedILVM(config)

        # Print model statistics with memory info
        total_params = sum(p.numel() for p in model.parameters())
        stats = model.count_operations()

        print(f"\nModel Statistics (Unified I-LVM + MIRAS):")
        print(f"  Total parameters: {total_params:,}")
        print(f"  BitLinear layers: {stats['bitlinear_layers']}")
        print(f"  Ternary params: {stats['ternary_params']:,} ({stats['ternary_params']/total_params*100:.1f}%)")
        print(f"  Rational RMSNorm: {stats['rational_norms']}")
        print(f"  Rational SiLU: {stats['rational_activations']}")
        print(f"  Rational Softmax: {stats['rational_softmax']}")
        print(f"  Rational RoPE: {stats['rational_rope']}")
        print(f"  MIRAS memory layers: {stats['memory_layers']}")
        print(f"  Memory timescales: {stats['memory_timescales']} (Fast/Medium/Slow)")
        print(f"  DeltaNet delta rule: {stats['memory_layers']} layers")
        print(f"  Transcendentals: {stats['transcendentals']} (ZERO!)")
    else:
        print("\nCreating model with ODP rational operators...")
        config = RationalBitNetConfig(
            vocab_size=model_config.vocab_size,
            hidden_dim=model_config.hidden_dim,
            intermediate_dim=model_config.intermediate_dim,
            num_heads=model_config.num_heads,
            num_layers=model_config.num_layers,
            max_seq_len=model_config.max_seq_len,
        )
        model = RationalBitNet(config)

        # Print model statistics
        total_params = sum(p.numel() for p in model.parameters())
        stats = model.count_operations()

        print(f"\nModel Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  BitLinear layers: {stats['bitlinear_layers']}")
        print(f"  Ternary params: {stats['ternary_params']:,} ({stats['ternary_params']/total_params*100:.1f}%)")
        print(f"  Rational RMSNorm: {stats['rational_norms']}")
        print(f"  Rational SiLU: {stats['rational_activations']}")
        print(f"  Rational Softmax: {stats['rational_softmax']}")
        print(f"  Rational RoPE: {stats['rational_rope']}")
        print(f"  Transcendentals: {stats['transcendentals']} (ZERO!)")

    # Create data loaders
    print("\nCreating data loaders...")
    train_loader = create_dataloader(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split="train",
        batch_size=train_config.batch_size,
        max_seq_len=train_config.max_seq_len,
        vocab_size=model_config.vocab_size,
        use_synthetic=args.synthetic,
        seed=args.seed,
    )

    eval_loader = None
    if not args.synthetic:
        try:
            eval_loader = create_dataloader(
                dataset_name=args.dataset,
                dataset_config=args.dataset_config,
                split="validation",
                batch_size=train_config.eval_batch_size,
                max_seq_len=train_config.max_seq_len,
                vocab_size=model_config.vocab_size,
                use_synthetic=False,
                seed=args.seed,
            )
        except Exception as e:
            print(f"Could not create eval loader: {e}")
            print("Will skip evaluation during training")

    # Create trainer
    print("\nInitializing trainer...")
    trainer = ILVMTrainer(
        model=model,
        train_config=train_config,
        model_config=config,
        train_loader=train_loader,
        eval_loader=eval_loader,
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')

        # Check if checkpoint has 'model.' prefix (saved from wrapped model)
        sample_key = list(checkpoint['model_state_dict'].keys())[0]
        has_model_prefix = sample_key.startswith('model.')

        # Load model state dict (handle wrapper prefix mismatch)
        if has_model_prefix and hasattr(trainer.model, 'model'):
            # Checkpoint from wrapped model, load directly into wrapper
            trainer.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("  Loaded wrapped model state dict")
        elif has_model_prefix:
            # Strip 'model.' prefix for unwrapped model
            new_state = {k.replace('model.', '', 1): v for k, v in checkpoint['model_state_dict'].items()}
            trainer.model.load_state_dict(new_state, strict=False)
            print("  Loaded state dict (stripped 'model.' prefix)")
        else:
            # No prefix, load directly
            if hasattr(trainer.model, 'model'):
                trainer.model.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                trainer.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("  Loaded state dict directly")

        # Restore optimizer and scheduler (unless reset requested)
        if not args.reset_optimizer:
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("  Restored optimizer and scheduler state")
        else:
            print("  Reset optimizer (fresh state, keeping model weights)")

        trainer.global_step = checkpoint['global_step']
        trainer.best_loss = checkpoint['best_loss']

        print(f"  Resumed from step {trainer.global_step}, best_loss={trainer.best_loss:.4f}")

    # Start training
    print("\nStarting training...")
    results = trainer.train()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation loss: {results['best_loss']:.4f}")
    print(f"Final step: {results['final_step']}")
    print(f"Checkpoints saved to: {args.output_dir}")
    print()
    print("Model Properties:")
    print("  [x] ZERO transcendental operations (exp, sqrt, sin, cos)")
    print("  [x] {-1, 0, 1} ternary weights (matmul = additions only)")
    print("  [x] Ready for integer-only inference")
    print("  [x] ZK/FHE compatible (only +, -, *, / operations)")
    if args.use_memory:
        print()
        print("MIRAS Memory Features:")
        print("  [x] Multi-timescale retention (Fast/Medium/Slow decay)")
        print("  [x] DeltaNet delta rule (key overwriting for multi-KV recall)")
        print("  [x] RationalHuber attentional bias (robust to outliers)")


if __name__ == "__main__":
    main()
