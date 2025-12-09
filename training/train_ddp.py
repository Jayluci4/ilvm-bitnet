"""
DDP Multi-GPU Training Script for BitNet-ODP

Optimized for 8x NVIDIA L4 GPUs (Ada Architecture):
- Native BF16 with 4th gen Tensor Cores
- NCCL backend for gradient synchronization
- TF32 matmul acceleration
- Fused optimizer kernels
- torch.compile for graph optimization

Usage:
    torchrun --nproc_per_node=8 training/train_ddp.py --model_size 1B --use_memory
"""

import os
import sys
import argparse
import time
import math
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from config import L4Config, get_model_config, get_l4_config, estimate_memory
from data_loader import create_dataloader
from rational_bitnet import RationalBitNet, RationalBitNetConfig


def setup_distributed():
    """Initialize distributed training environment."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = 0
        local_rank = 0
        world_size = 1

    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )

    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def setup_l4_optimizations():
    """Apply L4 Ada architecture optimizations."""
    # Enable TF32 for Tensor Cores (8x faster matmul, slight precision loss)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Enable cudnn benchmark for consistent input sizes
    torch.backends.cudnn.benchmark = True

    # Set BF16 as default dtype for autocast
    torch.set_float32_matmul_precision("high")


def is_main_process(rank: int) -> bool:
    """Check if this is the main process."""
    return rank == 0


def print_rank0(msg: str, rank: int):
    """Print only on rank 0."""
    if is_main_process(rank):
        print(msg)


def create_model(config: L4Config, model_config, use_memory: bool = False, rank: int = 0):
    """Create and configure the model for DDP training."""
    if use_memory:
        from unified_ilvm import UnifiedILVM, get_unified_ilvm_config
        ilvm_config = get_unified_ilvm_config(config.model_size)
        model = UnifiedILVM(ilvm_config)
        print_rank0(f"Created UnifiedILVM with MIRAS memory", rank)
    else:
        bitnet_config = RationalBitNetConfig(
            vocab_size=model_config.vocab_size,
            hidden_dim=model_config.hidden_dim,
            intermediate_dim=model_config.intermediate_dim,
            num_heads=model_config.num_heads,
            num_layers=model_config.num_layers,
            max_seq_len=model_config.max_seq_len,
        )
        model = RationalBitNet(bitnet_config)
        print_rank0(f"Created RationalBitNet", rank)

    # Enable gradient checkpointing for memory efficiency
    if config.use_gradient_checkpointing:
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        print_rank0("Gradient checkpointing enabled", rank)

    return model


def create_optimizer(model: nn.Module, config: L4Config, rank: int):
    """Create optimizer with L4 optimizations."""
    # Separate parameter groups for muP
    param_groups = []

    # Embedding parameters
    embed_params = [p for n, p in model.named_parameters() if 'embed' in n.lower()]
    if embed_params:
        param_groups.append({
            'params': embed_params,
            'lr': config.learning_rate,
            'name': 'embedding'
        })

    # Output projection parameters
    output_params = [p for n, p in model.named_parameters() if 'lm_head' in n.lower()]
    if output_params:
        param_groups.append({
            'params': output_params,
            'lr': config.learning_rate,
            'name': 'output'
        })

    # Hidden layer parameters
    hidden_params = [p for n, p in model.named_parameters()
                     if 'embed' not in n.lower() and 'lm_head' not in n.lower()]
    if hidden_params:
        param_groups.append({
            'params': hidden_params,
            'lr': config.learning_rate,
            'name': 'hidden'
        })

    # Try to use 8-bit Adam with fused kernels
    if config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.Adam8bit(
                param_groups,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                betas=(0.9, 0.95),
            )
            print_rank0("Using 8-bit Adam optimizer", rank)
            return optimizer
        except ImportError:
            print_rank0("bitsandbytes not available, using standard AdamW", rank)

    # Fall back to fused AdamW
    if config.use_fused_adam:
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
            fused=True,  # Use fused CUDA kernels
        )
        print_rank0("Using fused AdamW optimizer", rank)
    else:
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )
        print_rank0("Using standard AdamW optimizer", rank)

    return optimizer


def create_scheduler(optimizer, config: L4Config, total_steps: int):
    """Create learning rate scheduler with warmup and cosine decay."""
    def lr_lambda(step):
        if step < config.warmup_steps:
            # Linear warmup
            return step / config.warmup_steps
        else:
            # Cosine decay
            progress = (step - config.warmup_steps) / (total_steps - config.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class DDPTrainer:
    """Distributed Data Parallel trainer for BitNet-ODP."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        config: L4Config,
        rank: int,
        local_rank: int,
        world_size: int,
    ):
        self.config = config
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{local_rank}")

        # Move model to device
        model = model.to(self.device)

        # Optionally compile the model
        if config.compile_model and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode="reduce-overhead")
                print_rank0("Model compiled with torch.compile", rank)
            except Exception as e:
                print_rank0(f"torch.compile failed: {e}", rank)

        # Wrap in DDP
        if world_size > 1:
            self.model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                bucket_cap_mb=config.ddp_bucket_cap_mb,
                find_unused_parameters=config.ddp_find_unused_parameters,
            )
        else:
            self.model = model

        self.optimizer = optimizer
        self.scheduler = scheduler

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')

        # Setup mixed precision
        self.use_bf16 = config.use_bf16 and torch.cuda.is_bf16_supported()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Execute a single training step."""
        self.model.train()

        input_ids = batch['input_ids'].to(self.device)
        labels = batch.get('labels', input_ids[:, 1:]).to(self.device)

        # Forward pass with autocast
        with torch.amp.autocast('cuda', dtype=torch.bfloat16 if self.use_bf16 else torch.float16):
            outputs = self.model(input_ids)

            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            # Shift for language modeling
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels.contiguous()

            if shift_logits.size(1) != shift_labels.size(1):
                min_len = min(shift_logits.size(1), shift_labels.size(1))
                shift_logits = shift_logits[:, :min_len, :]
                shift_labels = shift_labels[:, :min_len]

            # Compute loss
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        loss.backward()

        return loss.item() * self.config.gradient_accumulation_steps

    def optimizer_step(self) -> float:
        """Execute optimizer step with gradient clipping."""
        # Gradient clipping
        if self.world_size > 1:
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.module.parameters(),
                self.config.gradient_clip
            )
        else:
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip
            )

        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        return grad_norm.item()

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        if not is_main_process(self.rank):
            return

        # Get underlying model (unwrap DDP)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'config': self.config,
        }

        torch.save(checkpoint, path)
        print_rank0(f"Checkpoint saved to {path}", self.rank)

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        if not os.path.exists(path):
            print_rank0(f"No checkpoint found at {path}", self.rank)
            return

        # Load on CPU first, then move to device
        checkpoint = torch.load(path, map_location='cpu')

        # Get underlying model
        model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint.get('best_loss', float('inf'))

        print_rank0(f"Checkpoint loaded from {path}, step {self.global_step}", self.rank)

    def train(self, train_loader: DataLoader, eval_loader: Optional[DataLoader] = None):
        """Main training loop."""
        print_rank0("=" * 70, self.rank)
        print_rank0("STARTING DDP TRAINING", self.rank)
        print_rank0("=" * 70, self.rank)
        print_rank0(f"World size: {self.world_size}", self.rank)
        print_rank0(f"Batch size per GPU: {self.config.batch_size}", self.rank)
        print_rank0(f"Gradient accumulation: {self.config.gradient_accumulation_steps}", self.rank)
        effective_batch = self.config.batch_size * self.config.gradient_accumulation_steps * self.world_size
        print_rank0(f"Effective batch size: {effective_batch}", self.rank)
        print_rank0(f"Mixed precision: BF16={self.use_bf16}", self.rank)
        print_rank0("=" * 70, self.rank)

        accumulation_loss = 0.0
        step_times = []

        for batch_idx, batch in enumerate(train_loader):
            if self.global_step >= self.config.max_steps:
                break

            step_start = time.perf_counter()

            # Training step
            loss = self.train_step(batch)
            accumulation_loss += loss

            # Optimizer step after accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                grad_norm = self.optimizer_step()
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
                        self.world_size
                    ) / avg_step_time

                    print_rank0(
                        f"Step {self.global_step:6d} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Grad: {grad_norm:.2f} | "
                        f"Tok/s: {tokens_per_sec:,.0f}",
                        self.rank
                    )

                accumulation_loss = 0.0

                # Evaluation
                if eval_loader and self.global_step % self.config.eval_every == 0:
                    eval_loss = self.evaluate(eval_loader)
                    print_rank0(f"  Eval loss: {eval_loss:.4f}", self.rank)

                    if eval_loss < self.best_loss:
                        self.best_loss = eval_loss
                        self.save_checkpoint(f"{self.config.output_dir}/best_model.pt")

                # Checkpoint
                if self.global_step % self.config.save_every == 0:
                    self.save_checkpoint(
                        f"{self.config.output_dir}/checkpoint_step{self.global_step}.pt"
                    )

        # Final save
        self.save_checkpoint(f"{self.config.output_dir}/final_model.pt")
        print_rank0("Training complete!", self.rank)

    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader) -> float:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in eval_loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch.get('labels', input_ids[:, 1:]).to(self.device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16 if self.use_bf16 else torch.float16):
                outputs = self.model(input_ids)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels.contiguous()

                if shift_logits.size(1) != shift_labels.size(1):
                    min_len = min(shift_logits.size(1), shift_labels.size(1))
                    shift_logits = shift_logits[:, :min_len, :]
                    shift_labels = shift_labels[:, :min_len]

                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

            total_loss += loss.item()
            num_batches += 1

            if num_batches >= 50:  # Limit eval batches
                break

        # Sync loss across GPUs
        if self.world_size > 1:
            loss_tensor = torch.tensor([total_loss, num_batches], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            total_loss = loss_tensor[0].item()
            num_batches = int(loss_tensor[1].item())

        return total_loss / num_batches if num_batches > 0 else float('inf')


def main():
    parser = argparse.ArgumentParser(description="DDP Training for BitNet-ODP")
    parser.add_argument("--model_size", type=str, default="1B",
                        choices=["350M", "1B", "3B", "7B"])
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
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # Setup distributed
    rank, local_rank, world_size = setup_distributed()

    # L4 optimizations
    setup_l4_optimizations()

    # Configuration
    config = get_l4_config(num_gpus=world_size)
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

    # Print config on rank 0
    if is_main_process(rank):
        memory = estimate_memory(model_config, num_gpus=world_size)
        print("=" * 70)
        print("DDP TRAINING CONFIGURATION")
        print("=" * 70)
        print(f"Model: {args.model_size} ({memory['total_params']:,} params)")
        print(f"Memory per GPU: {memory['per_gpu_gb']:.2f} GB")
        print(f"Fits L4: {memory['fits_l4']}")
        print(f"World size: {world_size}")
        print(f"Effective batch: {config.batch_size * config.gradient_accumulation_steps * world_size}")
        print(f"Max sequence length: {config.max_seq_len}")
        print(f"MIRAS memory: {args.use_memory}")
        print("=" * 70)

    # Create output directory
    if is_main_process(rank):
        os.makedirs(config.output_dir, exist_ok=True)

    # Barrier to ensure directory exists
    if world_size > 1:
        dist.barrier()

    # Create model
    model = create_model(config, model_config, args.use_memory, rank)

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config, rank)
    scheduler = create_scheduler(optimizer, config, config.max_steps)

    # Create trainer
    trainer = DDPTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )

    # Resume from checkpoint
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Create data loaders
    print_rank0("Creating data loaders...", rank)
    train_loader = create_dataloader(
        dataset_name=config.dataset_name,
        batch_size=config.batch_size,
        max_seq_len=config.max_seq_len,
        streaming=True,
    )

    eval_loader = create_dataloader(
        dataset_name=config.dataset_name,
        batch_size=config.eval_batch_size,
        max_seq_len=config.max_seq_len,
        streaming=True,
        split="validation",
    )

    # Train
    try:
        trainer.train(train_loader, eval_loader)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
