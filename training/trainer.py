"""
I-LVM Trainer with T4 Memory Optimizations

Features:
- Gradient checkpointing for memory efficiency
- Mixed precision training (FP16)
- 8-bit Adam optimizer
- Gradient accumulation
- OOM recovery with batch size reduction
- Learning rate warmup for rational networks
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.amp import autocast  # Use torch.amp for BF16 support

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rational_bitnet import RationalBitNet, RationalBitNetConfig


class GradientCheckpointWrapper(nn.Module):
    """Wrapper to add gradient checkpointing to model."""

    def __init__(self, model: RationalBitNet):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward with gradient checkpointing on transformer blocks."""
        if self.training and torch.is_grad_enabled():
            # Embed tokens
            hidden_states = self.model.embed_tokens(input_ids)
            batch_size, seq_len = input_ids.shape

            # Create causal mask
            if attention_mask is None:
                causal_mask = torch.triu(
                    torch.full((seq_len, seq_len), float('-inf'), device=input_ids.device),
                    diagonal=1
                )
                attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)

            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

            # Checkpointed forward through layers
            for layer in self.model.layers:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    use_reentrant=False,
                )

            # Final norm and LM head
            hidden_states = self.model.norm(hidden_states)
            logits = self.model.lm_head(hidden_states)

            # Compute loss
            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, self.model.config.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

            return {"logits": logits, "loss": loss}
        else:
            return self.model(input_ids, attention_mask, labels)


class UnifiedGradientCheckpointWrapper(nn.Module):
    """Wrapper to add gradient checkpointing to UnifiedILVM model.

    Handles tuple returns from UnifiedILVMBlock: (hidden_states, memory_states)
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward with gradient checkpointing on transformer blocks."""
        if self.training and torch.is_grad_enabled():
            # Embed tokens
            hidden_states = self.model.embed_tokens(input_ids)
            batch_size, seq_len = input_ids.shape

            # Create causal mask
            if attention_mask is None:
                causal_mask = torch.triu(
                    torch.full((seq_len, seq_len), float('-inf'), device=input_ids.device),
                    diagonal=1
                )
                attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)

            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

            # Memory states for MIRAS (initialized to None for each layer)
            memory_states = None

            # Checkpointed forward through layers
            # UnifiedILVMBlock returns (hidden_states, memory_states) tuple
            for layer in self.model.layers:
                # Use a wrapper function that handles tuple unpacking
                result = torch.utils.checkpoint.checkpoint(
                    self._layer_forward,
                    layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    memory_states,
                    use_reentrant=False,
                )
                # Unpack the result tuple
                hidden_states, memory_states = result

            # Final norm and LM head
            hidden_states = self.model.norm(hidden_states)
            logits = self.model.lm_head(hidden_states)

            # Compute loss
            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, self.model.config.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

            return {"logits": logits, "loss": loss}
        else:
            return self.model(input_ids, attention_mask, labels)

    @staticmethod
    def _layer_forward(layer, hidden_states, attention_mask, position_ids, memory_states):
        """Wrapper for layer forward that returns tuple properly."""
        return layer(hidden_states, attention_mask, position_ids, memory_states)


def get_8bit_optimizer(model: nn.Module, lr: float, weight_decay: float = 0.01):
    """Get 8-bit Adam optimizer if bitsandbytes is available."""
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.Adam8bit(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        print("Using 8-bit Adam optimizer (bitsandbytes)")
        return optimizer
    except ImportError:
        print("bitsandbytes not available, using standard AdamW")
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )


def get_lr_scheduler(optimizer, warmup_steps: int, max_steps: int):
    """Get learning rate scheduler with warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item()))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class ILVMTrainer:
    """Trainer for Integer-Only LVM with T4 optimizations."""

    def _create_mup_param_groups(
        self,
        base_lr: float,
        weight_decay: float,
        hidden_dim: int,
    ) -> list:
        """Create µP-style parameter groups with different learning rates.

        µP (Maximal Update Parametrization) ensures:
        - Embedding: lr * sqrt(hidden_dim) for proper gradient scaling
        - Hidden layers: base_lr
        - Output layer (LM head): lr / sqrt(hidden_dim) for stability

        This prevents mid-training collapse in binary/ternary models.
        """
        # Access the inner model if wrapped
        model = self.model.model if hasattr(self.model, 'model') else self.model

        embed_params = []
        hidden_params = []
        output_params = []
        other_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if 'embed_tokens' in name:
                embed_params.append(param)
            elif 'lm_head' in name:
                output_params.append(param)
            elif any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                          'gate_proj', 'up_proj', 'down_proj']):
                hidden_params.append(param)
            else:
                other_params.append(param)

        # µP learning rate scaling
        import math
        width_scale = math.sqrt(hidden_dim / 512)  # Normalize to base width of 512

        param_groups = []

        if embed_params:
            param_groups.append({
                'params': embed_params,
                'lr': base_lr * width_scale,  # Scale up for embeddings
                'weight_decay': 0.0,  # No weight decay on embeddings
                'name': 'embedding',
            })

        if hidden_params:
            param_groups.append({
                'params': hidden_params,
                'lr': base_lr,
                'weight_decay': weight_decay,
                'name': 'hidden',
            })

        if output_params:
            param_groups.append({
                'params': output_params,
                'lr': base_lr / width_scale,  # Scale down for output
                'weight_decay': weight_decay,
                'name': 'output',
            })

        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': base_lr,
                'weight_decay': weight_decay,
                'name': 'other',
            })

        print(f"µP parameter groups:")
        for pg in param_groups:
            print(f"  {pg['name']}: {len(pg['params'])} params, lr={pg['lr']:.2e}")

        return param_groups

    def __init__(
        self,
        model: RationalBitNet,
        train_config: Any,
        model_config: RationalBitNetConfig,
        train_loader: Any,
        eval_loader: Optional[Any] = None,
    ):
        self.train_config = train_config
        self.model_config = model_config
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Wrap model with gradient checkpointing if enabled
        if train_config.use_gradient_checkpointing:
            # Detect if this is UnifiedILVM (has MIRAS memory) or RationalBitNet
            # UnifiedILVMBlock returns tuple (hidden_states, memory_states)
            is_unified = hasattr(model, 'layers') and len(model.layers) > 0 and \
                         hasattr(model.layers[0], 'use_memory')
            if is_unified:
                self.model = UnifiedGradientCheckpointWrapper(model)
                print("Gradient checkpointing enabled (UnifiedILVM with MIRAS)")
            else:
                self.model = GradientCheckpointWrapper(model)
                print("Gradient checkpointing enabled")
        else:
            self.model = model

        self.model = self.model.to(self.device)

        # Initialize optimizer with µP-style parameter groups
        # Different LRs for embedding, hidden, and output layers
        param_groups = self._create_mup_param_groups(
            train_config.learning_rate,
            train_config.weight_decay,
            model_config.hidden_dim if hasattr(model_config, 'hidden_dim') else 512,
        )

        if train_config.use_8bit_adam:
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.Adam8bit(param_groups)
                print("Using 8-bit Adam optimizer with µP parameter groups")
            except ImportError:
                self.optimizer = torch.optim.AdamW(param_groups)
                print("Using AdamW with µP parameter groups (bitsandbytes not available)")
        else:
            self.optimizer = torch.optim.AdamW(param_groups)
            print("Using AdamW with µP parameter groups")

        # Learning rate scheduler
        self.scheduler = get_lr_scheduler(
            self.optimizer,
            train_config.warmup_steps,
            train_config.max_steps,
        )

        # Mixed precision - use BF16 for rational operators (better dynamic range than FP16)
        self.use_bf16 = train_config.use_mixed_precision and torch.cuda.is_bf16_supported()
        if self.use_bf16:
            # BF16 doesn't need GradScaler (same exponent range as FP32)
            self.scaler = None
            print("Using BF16 mixed precision (no GradScaler needed)")
        elif train_config.use_mixed_precision:
            # Fallback to FP16 with GradScaler
            self.scaler = GradScaler()
            print("Using FP16 mixed precision with GradScaler")
        else:
            self.scaler = None

        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        self.current_batch_size = train_config.batch_size
        self.oom_count = 0

        # Output directory
        self.output_dir = Path(train_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _handle_oom(self):
        """Handle OOM by reducing batch size."""
        self.oom_count += 1
        torch.cuda.empty_cache()

        if self.current_batch_size > self.train_config.oom_retry_batch_size:
            old_bs = self.current_batch_size
            self.current_batch_size = max(
                self.train_config.oom_retry_batch_size,
                self.current_batch_size // 2,
            )
            print(f"OOM! Reducing batch size: {old_bs} -> {self.current_batch_size}")
            return True
        else:
            print(f"OOM! Already at minimum batch size ({self.current_batch_size})")
            return False

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, bool]:
        """Execute single training step with OOM and NaN handling."""
        self.model.train()

        # Move to device
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Truncate batch if needed due to OOM
        if input_ids.size(0) > self.current_batch_size:
            input_ids = input_ids[:self.current_batch_size]
            labels = labels[:self.current_batch_size]

        try:
            # Determine autocast dtype
            if self.use_bf16:
                amp_dtype = torch.bfloat16
            elif self.scaler is not None:
                amp_dtype = torch.float16
            else:
                amp_dtype = None

            # Forward pass
            if amp_dtype is not None:
                with autocast(device_type='cuda', dtype=amp_dtype):
                    outputs = self.model(input_ids, labels=labels)
                    loss = outputs["loss"]

                    # NaN detection
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.nan_count = getattr(self, 'nan_count', 0) + 1
                        print(f"WARNING: NaN/Inf loss detected (count: {self.nan_count})")
                        if self.nan_count > 10:
                            print("ERROR: Too many NaN losses. Stopping training.")
                            raise RuntimeError("Training diverged - too many NaN losses")
                        self.optimizer.zero_grad()
                        return 0.0, False

                    # Reset NaN counter on good loss
                    self.nan_count = 0

                    # Scale loss for gradient accumulation
                    loss = loss / self.train_config.gradient_accumulation_steps

                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()  # BF16 doesn't need scaler
            else:
                outputs = self.model(input_ids, labels=labels)
                loss = outputs["loss"]

                # NaN detection
                if torch.isnan(loss) or torch.isinf(loss):
                    self.nan_count = getattr(self, 'nan_count', 0) + 1
                    print(f"WARNING: NaN/Inf loss detected (count: {self.nan_count})")
                    if self.nan_count > 10:
                        print("ERROR: Too many NaN losses. Stopping training.")
                        raise RuntimeError("Training diverged - too many NaN losses")
                    self.optimizer.zero_grad()
                    return 0.0, False

                self.nan_count = 0
                loss = loss / self.train_config.gradient_accumulation_steps
                loss.backward()

            return loss.item() * self.train_config.gradient_accumulation_steps, True

        except RuntimeError as e:
            if "out of memory" in str(e):
                self._handle_oom()
                return 0.0, False
            else:
                raise e

    def optimizer_step(self):
        """Execute optimizer step with NaN-safe gradient clipping."""
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        # Check for NaN gradients before clipping
        has_nan_grad = False
        for param in self.model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    has_nan_grad = True
                    break

        if has_nan_grad:
            print("WARNING: NaN/Inf gradient detected, skipping optimizer step")
            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.update()
            return

        # Aggressive gradient clipping (0.5 instead of 1.0 for rational networks)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.train_config.gradient_clip,
        )

        # Skip if grad norm is too large (indicates instability)
        if grad_norm > self.train_config.gradient_clip * 100:
            print(f"WARNING: Gradient norm {grad_norm:.2f} too large, skipping step")
            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.update()
            return

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.scheduler.step()
        self.optimizer.zero_grad()

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set."""
        if self.eval_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        max_eval_batches = 100  # Limit eval for speed

        # Determine autocast dtype
        if self.use_bf16:
            amp_dtype = torch.bfloat16
        elif self.scaler is not None:
            amp_dtype = torch.float16
        else:
            amp_dtype = None

        for batch in self.eval_loader:
            if num_batches >= max_eval_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            if amp_dtype is not None:
                with autocast(device_type='cuda', dtype=amp_dtype):
                    outputs = self.model(input_ids, labels=labels)
                    total_loss += outputs["loss"].item()
            else:
                outputs = self.model(input_ids, labels=labels)
                total_loss += outputs["loss"].item()

            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return {"val_loss": avg_loss, "val_perplexity": perplexity}

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "model_config": asdict(self.model_config) if hasattr(self.model_config, '__dataclass_fields__') else self.model_config.__dict__,
        }

        # Save latest
        torch.save(checkpoint, self.output_dir / "checkpoint_latest.pt")

        # Save best
        if is_best:
            torch.save(checkpoint, self.output_dir / "checkpoint_best.pt")

        # Save by step
        if self.global_step % (self.train_config.save_every * 5) == 0:
            torch.save(checkpoint, self.output_dir / f"checkpoint_step_{self.global_step}.pt")

    def train(self):
        """Main training loop."""
        print("=" * 70)
        print("I-LVM TRAINING")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Batch size: {self.train_config.batch_size}")
        print(f"Gradient accumulation: {self.train_config.gradient_accumulation_steps}")
        print(f"Effective batch: {self.train_config.batch_size * self.train_config.gradient_accumulation_steps}")
        print(f"Max steps: {self.train_config.max_steps}")
        print("=" * 70)

        train_iter = iter(self.train_loader)
        accumulated_loss = 0.0
        accumulated_steps = 0
        start_time = time.time()

        while self.global_step < self.train_config.max_steps:
            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            # Train step
            loss, success = self.train_step(batch)

            if not success:
                # OOM occurred, retry with smaller batch
                continue

            accumulated_loss += loss
            accumulated_steps += 1

            # Optimizer step after accumulation
            if accumulated_steps >= self.train_config.gradient_accumulation_steps:
                self.optimizer_step()
                self.global_step += 1

                avg_loss = accumulated_loss / accumulated_steps

                # Logging
                if self.global_step % self.train_config.log_every == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = self.global_step / elapsed
                    lr = self.scheduler.get_last_lr()[0]

                    print(f"Step {self.global_step:6d} | "
                          f"Loss: {avg_loss:.4f} | "
                          f"LR: {lr:.2e} | "
                          f"Steps/s: {steps_per_sec:.2f} | "
                          f"OOM: {self.oom_count}")

                # Evaluation
                if self.global_step % self.train_config.eval_every == 0:
                    eval_metrics = self.evaluate()
                    if eval_metrics:
                        print(f"  Eval - Loss: {eval_metrics['val_loss']:.4f} | "
                              f"PPL: {eval_metrics['val_perplexity']:.2f}")

                        # Save best
                        if eval_metrics['val_loss'] < self.best_loss:
                            self.best_loss = eval_metrics['val_loss']
                            self.save_checkpoint(is_best=True)
                            print("  Saved best checkpoint!")

                # Save checkpoint
                if self.global_step % self.train_config.save_every == 0:
                    self.save_checkpoint()
                    print(f"  Saved checkpoint at step {self.global_step}")

                # Reset accumulation
                accumulated_loss = 0.0
                accumulated_steps = 0

        # Final save
        self.save_checkpoint()
        print("\nTraining complete!")
        print(f"Best validation loss: {self.best_loss:.4f}")

        return {"best_loss": self.best_loss, "final_step": self.global_step}


if __name__ == "__main__":
    # Quick test
    print("Testing trainer components...")

    from config import T4Config, get_model_config, print_config_summary

    train_config = T4Config(model_size="50M", max_steps=10)
    model_config = get_model_config(train_config.model_size)
    print_config_summary(train_config, model_config)

    # Create model
    config = RationalBitNetConfig(
        vocab_size=model_config.vocab_size,
        hidden_dim=model_config.hidden_dim,
        intermediate_dim=model_config.intermediate_dim,
        num_heads=model_config.num_heads,
        num_layers=model_config.num_layers,
        max_seq_len=model_config.max_seq_len,
    )
    model = RationalBitNet(config)

    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print("Trainer test complete!")
