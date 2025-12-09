# I-LVM Project Status

**Date**: December 9, 2025
**Status**: In Development - Stability Fixes Applied

---

## What We Are Building

**I-LVM (Integer-Only Latent Variable Model)** - A language model that requires ZERO floating-point transcendental operations, enabling:

- Pure integer inference on specialized hardware
- ZK/FHE-compatible AI (only +, -, *, / operations)
- 10x memory reduction via 1.58-bit weights

### Core Architecture

| Component | Implementation | Benefit |
|-----------|---------------|---------|
| **BitNet b1.58** | {-1, 0, 1} ternary weights | Matmul becomes additions only |
| **RationalRMSNorm** | Babylonian sqrt iteration | No SFU calls |
| **RationalSiLU** | Algebraic sigmoid approximation | No exp() |
| **RationalSoftmax** | Polynomial (1+x/8)^8 | No exp() |
| **RationalRoPE** | Cayley transform | No sin/cos |

### Model Configuration (125M)

```
Parameters: 78.7M total (67.3% ternary)
Hidden dim: 512
Layers: 8
Heads: 8
Vocab: 50,257 (GPT-2 tokenizer for fair comparison)
```

---

## How We Build the Model

### Source: `src/rational_bitnet.py`

#### 1. BitLinear Layer (Lines 86-170)

Ternary weight quantization using AbsMean scaling with Straight-Through Estimator (STE):

```python
def weight_quant_ternary(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # AbsMean scaling
    scale = w.abs().mean().clamp(min=1e-8)
    # Normalize and round to {-1, 0, 1}
    w_normalized = w / scale
    w_quant = torch.clamp(ste_round(w_normalized), min=-1, max=1)
    return w_quant, scale
```

**Key Innovation**: MatMul with ternary weights becomes:
- W[i,j] = 1: add x[j]
- W[i,j] = -1: subtract x[j]
- W[i,j] = 0: skip

**muP Initialization** (Lines 120-127):
```python
init_std = 1.0 / math.sqrt(in_features)  # Standard muP
if is_output_layer:
    init_std = init_std / math.sqrt(in_features)  # Extra scaling for LM head
```

#### 2. RationalRMSNorm (Lines 165-205)

Babylonian method for 1/sqrt(x) using only +, -, *, /:

```python
def _babylonian_rsqrt(self, x: torch.Tensor) -> torch.Tensor:
    x_safe = torch.clamp(x.float(), min=1e-6, max=1e6)  # FP32 + clamp
    y = torch.ones_like(x_safe)
    for _ in range(15):  # 15 iterations
        y_clamped = torch.clamp(y, min=1e-6)
        y = (y + x_safe / y_clamped) * 0.5
        y = torch.clamp(y, min=1e-6, max=1e6)
    return 1.0 / torch.clamp(y, min=1e-6)
```

#### 3. RationalSiLU (Lines 208-247)

Algebraic sigmoid: σ(x) ≈ 0.5 * (1 + x/√(1+x²))

```python
def forward(self, x):
    x_fp32 = x.float()
    x_scaled = x_fp32 / 1.5  # Scale factor
    rsqrt_val = self._babylonian_rsqrt(1.0 + x_scaled * x_scaled)
    x_normalized = torch.clamp(x_scaled * rsqrt_val, min=-1.0, max=1.0)
    sigmoid_approx = 0.5 * (1.0 + x_normalized)
    return (x_fp32 * sigmoid_approx).to(input_dtype)
```

#### 4. RationalSoftmax (Lines 250-290)

Polynomial exp approximation: exp(x) ≈ (1 + x/8)^8

```python
def forward(self, x, mask=None):
    x = x.float()  # Always FP32
    x_shifted = torch.clamp(x - x.max(dim=-1, keepdim=True).values, min=-6.0, max=0.0)
    t = torch.clamp(1.0 + x_shifted * 0.125, min=0.1)  # (1 + x/8)
    t2 = t * t; t4 = t2 * t2; weights = t4 * t4  # t^8
    sum_weights = torch.clamp(weights.sum(dim=-1, keepdim=True), min=1e-8)
    return torch.clamp(weights / sum_weights, min=0.0, max=1.0).to(input_dtype)
```

#### 5. RationalRoPE (Lines 293-355)

Cayley transform: cos(θ) = (1-t²)/(1+t²), sin(θ) = 2t/(1+t²)

```python
def _cayley_rotation(self, t):
    t = t.float()
    t_sq = t * t
    denom = torch.clamp(1.0 + t_sq, min=1e-6)
    cos_val = torch.clamp((1.0 - t_sq) / denom, min=-1.0, max=1.0)
    sin_val = torch.clamp((2.0 * t) / denom, min=-1.0, max=1.0)
    return cos_val, sin_val
```

#### 6. Model Assembly (Lines 513-560)

```python
class RationalBitNet(nn.Module):
    def __init__(self, config):
        self.output_scale = 1.0 / math.sqrt(config.hidden_dim)  # muP
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)
        nn.init.normal_(self.embed_tokens.weight, std=1.0)  # muP init

        self.layers = nn.ModuleList([RationalBitNetBlock(config) for _ in range(config.num_layers)])
        self.norm = RationalRMSNorm(config.hidden_dim)
        self.lm_head = BitLinear(config.hidden_dim, config.vocab_size, is_output_layer=True)

    def forward(self, input_ids, labels=None):
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask, position_ids)
        logits = self.lm_head(self.norm(hidden)) * self.output_scale  # muP scaling
        loss = F.cross_entropy(logits, labels) if labels else None
        return {"logits": logits, "loss": loss}
```

---

## How We Train the Model

### Source: `training/train.py` (Entry Point)

**Command**:
```bash
python training/train.py \
    --model_size 125M \
    --dataset tinystories \
    --max_steps 10000 \
    --batch_size 4 \
    --gradient_accumulation 8 \
    --learning_rate 3e-5
```

### Source: `training/trainer.py` (Training Loop)

#### 1. muP Parameter Groups (Lines 191-270)

Different learning rates for embedding, hidden, and output layers:

```python
def _create_mup_param_groups(self, base_lr, weight_decay, hidden_dim):
    width_scale = math.sqrt(hidden_dim / 512)  # Normalize to base width

    param_groups = [
        {'params': embed_params, 'lr': base_lr * width_scale, 'weight_decay': 0.0},
        {'params': hidden_params, 'lr': base_lr, 'weight_decay': weight_decay},
        {'params': output_params, 'lr': base_lr / width_scale, 'weight_decay': weight_decay},
    ]
    return param_groups
```

#### 2. BF16 Mixed Precision (Lines 330-350)

BF16 for speed, FP32 for dangerous math:

```python
# In __init__
self.use_bf16 = train_config.use_mixed_precision and torch.cuda.is_bf16_supported()

# In train_step
if self.use_bf16:
    amp_dtype = torch.bfloat16
    with autocast(device_type='cuda', dtype=amp_dtype):
        outputs = self.model(input_ids, labels=labels)
        loss = outputs["loss"]
    loss.backward()  # No GradScaler needed for BF16
```

#### 3. NaN-Safe Training (Lines 312-350)

```python
# NaN detection in forward
if torch.isnan(loss) or torch.isinf(loss):
    self.nan_count += 1
    if self.nan_count > 10:
        raise RuntimeError("Training diverged")
    self.optimizer.zero_grad()
    return 0.0, False  # Skip batch

# NaN-safe gradient clipping
for param in self.model.parameters():
    if param.grad is not None and torch.isnan(param.grad).any():
        print("WARNING: NaN gradient detected, skipping step")
        self.optimizer.zero_grad()
        return
```

#### 4. Gradient Checkpointing (Lines 31-83)

Memory-efficient training via recomputation:

```python
class GradientCheckpointWrapper(nn.Module):
    def forward(self, input_ids, attention_mask=None, labels=None):
        for layer in self.model.layers:
            hidden_states = torch.utils.checkpoint.checkpoint(
                layer, hidden_states, attention_mask, position_ids,
                use_reentrant=False,
            )
```

### Source: `training/config.py` (Configurations)

```python
@dataclass
class T4Config:
    model_size: str = "125M"
    learning_rate: float = 3e-5       # Lower for rational networks
    gradient_clip: float = 0.5        # Aggressive clipping
    warmup_steps: int = 2000          # Longer warmup
    gradient_accumulation_steps: int = 8
    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True  # BF16
    use_8bit_adam: bool = True
```

### Source: `training/data_loader.py` (Dataset Streaming)

```python
DATASET_CONFIGS = {
    "tinystories": DatasetConfig(
        hf_path="roneneldan/TinyStories",
        text_field="text",
    ),
    "fineweb-edu": DatasetConfig(
        hf_path="HuggingFaceFW/fineweb-edu",
        hf_config="sample-10BT",  # 10B token sample
    ),
    "slimpajama": DatasetConfig(
        hf_path="cerebras/SlimPajama-627B",
    ),
}

# GPT-2 tokenizer for fair comparison
tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

---

## Training Stages

| Stage | Dataset | Size | Purpose |
|-------|---------|------|---------|
| 1 | TinyStories | ~500MB | Validation, stability testing |
| 2 | FineWeb-Edu | 10B tokens | Main training |
| 3 | SlimPajama | 627B tokens | Generalization |

---

## Memory Budget (T4 15.6GB)

```
Model weights (FP32):     ~300MB
Gradients (FP32):         ~300MB
Optimizer (8-bit Adam):   ~150MB
Activations (checkpointed): ~500MB
CUDA overhead:            ~1GB
---------------------------------
Total:                    ~2.3GB (fits comfortably)
```

---

## What Went Right

### 1. Architecture Design
- Successfully implemented all rational operators using only +, -, *, /
- BitNet ternary quantization working correctly with STE
- Achieved ZERO transcendental operations in forward pass

### 2. Dataset Streaming
- TinyStories, FineWeb-Edu (10B), SlimPajama all stream correctly
- GPT-2 tokenizer integration working for fair perplexity comparison

### 3. Initial Training Progress
- Training ran successfully for ~4500 steps
- Loss decreased from ~10 to ~1.2 (good convergence)
- Ternary weight quantization stable during training

### 4. Infrastructure
- T4 GPU memory management working (gradient checkpointing, 8-bit Adam)
- Checkpoint save/resume implemented
- Logging and monitoring functional

---

## What Went Wrong

### 1. FP16 Incompatibility with Rational Operators

**Root Cause**: FP16 has limited dynamic range (5-bit exponent) which causes:
- Underflow in Babylonian iterations (small denominators become 0)
- Overflow in backward pass through divisions
- NaN cascade after ~4500 training steps

**Symptom**:
```
Step 4300 | Loss: 1.48
Step 4500 | Loss: 2.09  <- spike
Step 4600 | Loss: nan   <- explosion
```

### 2. Gradient Explosion in Rational Backward Pass

The backward pass through:
- `1/y` in Babylonian rsqrt
- `x/sum` in RationalSoftmax
- Cayley transform divisions

...produces very large gradients when denominators are small, which overflow in FP16.

### 3. Checkpoint Resume Issues

State dict keys had `model.` prefix from GradientCheckpointWrapper, causing silent load failures with `strict=False`.

---

## Fixes Applied

### 1. BF16 Instead of FP16
- BF16 has 8-bit exponent (same as FP32) with better dynamic range
- Eliminates underflow in rational operators
- No GradScaler needed (direct backward compatible)

```python
torch.autocast(device_type='cuda', dtype=torch.bfloat16)
```

### 2. FP32 Internals for All Rational Operators
- All Babylonian iterations compute in FP32
- All divisions have denominator clamping (min=1e-6)
- Output cast back to input dtype

```python
def forward(self, x):
    input_dtype = x.dtype
    x_fp32 = x.float()  # FP32 for computation
    # ... rational computation ...
    return result.to(input_dtype)  # Cast back
```

### 3. Denominator Clamping
Every division operation now has explicit clamping:

```python
y_clamped = torch.clamp(y, min=1e-6)
result = x / y_clamped
```

### 4. muP Scaling for Stability
Implemented Maximal Update Parametrization:

- **Embedding LR**: base_lr * sqrt(width/512)
- **Hidden LR**: base_lr
- **Output LR**: base_lr / sqrt(width/512)
- **Output logit scaling**: logits * (1/sqrt(hidden_dim))

This prevents mid-training collapse in ternary models.

### 5. NaN-Safe Gradient Clipping
- Check for NaN/Inf gradients before optimizer step
- Skip step if NaN detected (up to 10 retries)
- Aggressive gradient clipping (0.5)

---

## Current Training Configuration

```python
# Precision
dtype = torch.bfloat16       # BF16 for speed + stability
rational_ops = torch.float32  # FP32 for dangerous math

# Optimization
learning_rate = 3e-5
gradient_clip = 0.5
warmup_steps = 2000
batch_size = 4 * 8 = 32 effective

# muP Scaling
embed_lr = 3e-5 * 1.0 = 3e-5
hidden_lr = 3e-5
output_lr = 3e-5 / 1.0 = 3e-5
```

---

## Project Structure

```
bitnet-odp/
├── src/
│   ├── rational_bitnet.py    # Core model (BitNet + ODP rational ops)
│   └── unified_ilvm.py       # With MIRAS memory (optional)
├── training/
│   ├── train.py              # Main training script
│   ├── trainer.py            # ILVMTrainer with BF16 + muP
│   ├── config.py             # T4-optimized configurations
│   └── data_loader.py        # Streaming datasets
├── tests/                    # Unit tests
├── docs/                     # Documentation
├── examples/                 # Usage examples
├── checkpoints/              # Model checkpoints
└── archive/                  # Old logs and debug scripts
```

---

## Next Steps

1. **Verify BF16 + muP Training Stability**
   - Run 10k steps on TinyStories
   - Confirm no NaN/Inf issues
   - Target loss < 1.5

2. **Stage 2: FineWeb-Edu (10B)**
   - Scale to larger dataset
   - Monitor loss stability across dataset switch

3. **Stage 3: SlimPajama**
   - Full generalization training
   - Compare perplexity with GPT-2 baseline

4. **Inference Optimization**
   - Export ternary weights to int8
   - Benchmark integer-only inference
   - Measure speedup vs FP16 baseline

---

## Key Learnings

1. **FP16 is incompatible with division-heavy rational operators** - always use BF16 or FP32
2. **Clamp ALL denominators to min 1e-6** - prevents NaN cascade
3. **muP is essential for ternary models** - prevents mid-training collapse
4. **Checkpoint loading needs prefix handling** - watch for wrapper prefixes
5. **Babylonian iterations need careful FP32 handling** - small values explode in low precision

---

## References

- BitNet b1.58: "The Era of 1-bit LLMs" (Microsoft, 2024)
- ODP Rational Operators: Operator Discovery Platform approximations
- muP: "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer"
- fix.md in archive/ for detailed stability analysis
