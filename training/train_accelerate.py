"""
Accelerate-based Multi-GPU Training for BitNet-ODP

Uses HuggingFace Accelerate to wrap the existing trainer with minimal changes.
This is the fastest way to enable 8x L4 GPU training without rewriting.

Setup:
    pip install accelerate
    accelerate config  # Configure for multi-GPU

Usage:
    accelerate launch training/train_accelerate.py --model_size 1B --use_memory
"""

import os
import sys
import argparse
import time
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from config import L4Config, get_model_config, get_l4_config, estimate_memory
from data_loader import create_dataloader


def setup_l4_optimizations():
    """Apply L4 Ada architecture optimizations."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def create_model(config: L4Config, model_config, use_memory: bool = False):
    """Create model for training."""
    if use_memory:
        from unified_ilvm import UnifiedILVM, get_unified_ilvm_config
        ilvm_config = get_unified_ilvm_config(config.model_size)
        model = UnifiedILVM(ilvm_config)
    else:
        from rational_bitnet import RationalBitNet, RationalBitNetConfig
        bitnet_config = RationalBitNetConfig(
            vocab_size=model_config.vocab_size,
            hidden_dim=model_config.hidden_dim,
            intermediate_dim=model_config.intermediate_dim,
            num_heads=model_config.num_heads,
            num_layers=model_config.num_layers,
            max_seq_len=model_config.max_seq_len,
        )
        model = RationalBitNet(bitnet_config)

    # Enable gradient checkpointing
    if config.use_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()

    return model


def create_optimizer(model: nn.Module, config: L4Config):
    """Create optimizer with 8-bit Adam if available."""
    if config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            return bnb.optim.Adam8bit(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                betas=(0.9, 0.95),
            )
        except ImportError:
            pass

    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
        fused=config.use_fused_adam,
    )


def create_scheduler(optimizer, config: L4Config, num_training_steps: int):
    """Create cosine scheduler with warmup."""
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        progress = (step - config.warmup_steps) / (num_training_steps - config.warmup_steps)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


class AccelerateTrainer:
    """Trainer using HuggingFace Accelerate for multi-GPU."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        train_dataloader,
        eval_dataloader,
        config: L4Config,
        accelerator: Accelerator,
    ):
        self.config = config
        self.accelerator = accelerator

        # Prepare model, optimizer, scheduler for distributed training
        # NOTE: For streaming datasets that are already sharded via .shard(),
        # we DON'T let Accelerate wrap the dataloader (it would try to add
        # DistributedSampler which doesn't work with IterableDatasets)
        self.model, self.optimizer, self.scheduler = accelerator.prepare(
            model, optimizer, scheduler
        )

        # For streaming datasets, prepare dataloader with proper settings
        # dispatch_batches=False: don't try to dispatch batches across processes
        # split_batches=False: each process gets its own complete batches
        from torch.utils.data import IterableDataset
        if isinstance(train_dataloader.dataset, IterableDataset):
            accelerator.print("Detected streaming/iterable dataset - using manual sharding")
            # Don't wrap - dataset is already sharded via .shard() in data_loader.py
            self.train_dataloader = train_dataloader
        else:
            self.train_dataloader = accelerator.prepare(train_dataloader)

        if eval_dataloader is not None:
            if isinstance(eval_dataloader.dataset, IterableDataset):
                self.eval_dataloader = eval_dataloader
            else:
                self.eval_dataloader = accelerator.prepare(eval_dataloader)
        else:
            self.eval_dataloader = None

        self.global_step = 0
        self.best_loss = float('inf')

    def train_step(self, batch):
        """Single training step."""
        self.model.train()

        # Move batch to device (needed when not using Accelerate dataloader wrapping)
        input_ids = batch['input_ids'].to(self.accelerator.device)
        labels = batch.get('labels', input_ids[:, 1:])
        if isinstance(labels, torch.Tensor):
            labels = labels.to(self.accelerator.device)

        # Forward with mixed precision (handled by Accelerate)
        outputs = self.model(input_ids)
        # Handle different output formats: dict (UnifiedILVM), tuple, or tensor
        if isinstance(outputs, dict):
            logits = outputs["logits"]
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        # Compute loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels.contiguous()

        if shift_logits.size(1) != shift_labels.size(1):
            min_len = min(shift_logits.size(1), shift_labels.size(1))
            shift_logits = shift_logits[:, :min_len, :]
            shift_labels = shift_labels[:, :min_len]

        # Use reshape instead of view for non-contiguous tensors
        loss = nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
        )

        # Backward (Accelerate handles gradient accumulation)
        self.accelerator.backward(loss)

        return loss.item()

    def optimizer_step(self):
        """Optimizer step with gradient clipping."""
        # Clip gradients
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)

        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

    @torch.no_grad()
    def evaluate(self):
        """Evaluate the model."""
        if self.eval_dataloader is None:
            return float('inf')

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.eval_dataloader:
            # Move batch to device (needed when not using Accelerate dataloader wrapping)
            input_ids = batch['input_ids'].to(self.accelerator.device)
            labels = batch.get('labels', input_ids[:, 1:])
            if isinstance(labels, torch.Tensor):
                labels = labels.to(self.accelerator.device)

            outputs = self.model(input_ids)
            # Handle different output formats: dict (UnifiedILVM), tuple, or tensor
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels.contiguous()

            if shift_logits.size(1) != shift_labels.size(1):
                min_len = min(shift_logits.size(1), shift_labels.size(1))
                shift_logits = shift_logits[:, :min_len, :]
                shift_labels = shift_labels[:, :min_len]

            # Use reshape instead of view for non-contiguous tensors
            loss = nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )
            total_loss += loss.item()
            num_batches += 1

            if num_batches >= 50:
                break

        # Gather across processes
        total_loss = self.accelerator.gather(torch.tensor([total_loss])).mean().item()
        return total_loss / num_batches if num_batches > 0 else float('inf')

    def save_checkpoint(self, path: str):
        """Save checkpoint."""
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        self.accelerator.save({
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_loss': self.best_loss,
        }, path)

    def train(self):
        """Main training loop."""
        self.accelerator.print("=" * 70)
        self.accelerator.print("ACCELERATE MULTI-GPU TRAINING")
        self.accelerator.print("=" * 70)
        self.accelerator.print(f"Num processes: {self.accelerator.num_processes}")
        self.accelerator.print(f"Batch size per device: {self.config.batch_size}")
        self.accelerator.print(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        effective_batch = (
            self.config.batch_size *
            self.config.gradient_accumulation_steps *
            self.accelerator.num_processes
        )
        self.accelerator.print(f"Effective batch size: {effective_batch}")
        self.accelerator.print(f"Mixed precision: {self.accelerator.mixed_precision}")
        self.accelerator.print("=" * 70)

        accumulation_loss = 0.0
        step_times = []

        self.accelerator.print("Starting training loop...")
        self.accelerator.print("Waiting for first batch from dataloader...")

        for batch_idx, batch in enumerate(self.train_dataloader):
            if batch_idx == 0:
                self.accelerator.print(f"First batch received! Shape: {batch['input_ids'].shape}")
            if self.global_step >= self.config.max_steps:
                break

            step_start = time.perf_counter()

            # Training step with gradient accumulation context
            with self.accelerator.accumulate(self.model):
                loss = self.train_step(batch)
                accumulation_loss += loss

                # Only step optimizer when gradients are synced
                if self.accelerator.sync_gradients:
                    self.optimizer_step()
                    self.global_step += 1

                    step_time = time.perf_counter() - step_start
                    step_times.append(step_time)

                    # Logging
                    if self.global_step % self.config.log_every == 0:
                        avg_loss = accumulation_loss / self.config.gradient_accumulation_steps
                        lr = self.scheduler.get_last_lr()[0]
                        avg_step_time = sum(step_times[-10:]) / len(step_times[-10:])
                        tokens_per_sec = (
                            self.config.batch_size *
                            self.config.max_seq_len *
                            self.config.gradient_accumulation_steps *
                            self.accelerator.num_processes
                        ) / avg_step_time

                        self.accelerator.print(
                            f"Step {self.global_step:6d} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"LR: {lr:.2e} | "
                            f"Tok/s: {tokens_per_sec:,.0f}"
                        )

                    accumulation_loss = 0.0

                    # Evaluation
                    if self.global_step % self.config.eval_every == 0:
                        eval_loss = self.evaluate()
                        self.accelerator.print(f"  Eval loss: {eval_loss:.4f}")

                        if eval_loss < self.best_loss:
                            self.best_loss = eval_loss
                            self.save_checkpoint(f"{self.config.output_dir}/best_model.pt")

                    # Save checkpoint
                    if self.global_step % self.config.save_every == 0:
                        self.save_checkpoint(
                            f"{self.config.output_dir}/checkpoint_step{self.global_step}.pt"
                        )

        # Final save
        self.save_checkpoint(f"{self.config.output_dir}/final_model.pt")
        self.accelerator.print("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Accelerate Training for BitNet-ODP")
    parser.add_argument("--model_size", type=str, default="125M",
                        choices=["125M", "350M", "1B", "3B", "7B"])
    parser.add_argument("--dataset", type=str, default="c4")
    parser.add_argument("--max_steps", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=2500)
    parser.add_argument("--use_memory", action="store_true",
                        help="Use MIRAS memory (UnifiedILVM)")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # L4 optimizations
    setup_l4_optimizations()

    # Initialize Accelerator with BF16 mixed precision
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation,
        mixed_precision="bf16",  # Native BF16 on L4
        log_with="tensorboard",
        project_dir=args.output_dir,
    )

    set_seed(args.seed)

    # Configuration
    config = get_l4_config(num_gpus=accelerator.num_processes)
    config.model_size = args.model_size
    config.batch_size = args.batch_size
    config.gradient_accumulation_steps = args.gradient_accumulation
    config.learning_rate = args.learning_rate
    config.warmup_steps = args.warmup_steps
    config.gradient_clip = args.gradient_clip
    config.max_seq_len = args.max_seq_len
    config.max_steps = args.max_steps
    config.log_every = args.log_every
    config.eval_every = args.eval_every
    config.save_every = args.save_every
    config.output_dir = args.output_dir
    config.dataset_name = args.dataset

    # Model config
    model_config = get_model_config(args.model_size)

    # Print config
    if accelerator.is_main_process:
        memory = estimate_memory(model_config, num_gpus=accelerator.num_processes)
        print("=" * 70)
        print("ACCELERATE TRAINING CONFIGURATION")
        print("=" * 70)
        print(f"Model: {args.model_size} ({memory['total_params']:,} params)")
        print(f"Memory per GPU: {memory['per_gpu_gb']:.2f} GB")
        print(f"Fits L4: {memory['fits_l4']}")
        print(f"Num processes: {accelerator.num_processes}")
        effective_batch = config.batch_size * config.gradient_accumulation_steps * accelerator.num_processes
        print(f"Effective batch: {effective_batch}")
        print(f"Mixed precision: {accelerator.mixed_precision}")
        print(f"MIRAS memory: {args.use_memory}")
        print("=" * 70)

        os.makedirs(config.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    # Create model
    model = create_model(config, model_config, args.use_memory)
    accelerator.print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, config.max_steps)

    # Create data loaders with distributed sharding
    # Pass rank/world_size to ensure each GPU gets different data
    accelerator.print("Creating data loaders...")
    train_dataloader = create_dataloader(
        dataset_name=config.dataset_name,
        batch_size=config.batch_size,
        max_seq_len=config.max_seq_len,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )

    eval_dataloader = create_dataloader(
        dataset_name=config.dataset_name,
        batch_size=config.eval_batch_size,
        max_seq_len=config.max_seq_len,
        split="validation",
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )

    # Create trainer
    trainer = AccelerateTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=config,
        accelerator=accelerator,
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
