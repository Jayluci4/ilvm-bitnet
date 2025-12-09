# Appendix C: Failure Mode Catalog

This appendix catalogs known failure modes in the Integer-Only LVM architecture, their root causes, symptoms, and mitigations. Understanding these failure modes is essential for building robust systems.

---

## The Physics of the Problem

### C.1 Two Fundamental Bottlenecks

Neural network inference is constrained by two physical bottlenecks:

```
┌─────────────────────────────────────────────────────────────────┐
│                    NEURAL NETWORK BOTTLENECKS                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. MEMORY BOTTLENECK                                           │
│     ─────────────────                                           │
│     Problem: Weights don't fit in cache                         │
│     Physics: Memory bandwidth << Compute throughput             │
│     Metric:  bytes_moved / ops_performed (arithmetic intensity) │
│                                                                 │
│     Transformer reality:                                        │
│       - 7B model = 14GB weights (FP16)                          │
│       - L2 cache = 64MB                                         │
│       - Ratio: 218:1 (must stream from RAM)                     │
│                                                                 │
│  2. COMPUTE BOTTLENECK                                          │
│     ─────────────────                                           │
│     Problem: Transcendentals are slow/complex                   │
│     Physics: SFU (Special Function Unit) throughput << ALU      │
│     Examples: exp(), sqrt(), sin(), cos(), rsqrt()              │
│                                                                 │
│     GPU reality:                                                │
│       - FP32 FLOPS: 30 TFLOPS                                   │
│       - SFU throughput: ~1/4 of FLOPS                           │
│       - exp() latency: ~20 cycles vs 4 for add                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### C.2 How BitNet + ODP Solve These

```
BITNET (Memory Solution)
────────────────────────
Before: W ∈ FP16^{n×m}     → 16 bits per weight
After:  W ∈ {-1,0,1}^{n×m} → 1.58 bits per weight (10x compression)

Matmul transformation:
  y = W @ x  →  y = sum(W[i,j] * x[j])  →  y = ±x[j] additions only

ODP (Compute Solution)
──────────────────────
Before: softmax = exp(x) / sum(exp(x))  → SFU call per element
After:  softmax ≈ (1+x/4)^4 / sum(...)  → polynomial, no SFU

Before: RMSNorm = x / sqrt(mean(x²))    → SFU rsqrt call
After:  RMSNorm = x / babylonian(...)   → iterative +,-,*,/ only
```

---

## Category 1: Weight Quantization Failures

### C.1.1 Distribution Collapse

**Symptom**: All weights cluster around 0, model outputs become constant.

**Root Cause**: AbsMean scale becomes too small when weights are already near-zero.

```python
# Problem scenario
W = torch.randn(512, 512) * 0.001  # Very small weights
scale = W.abs().mean()             # scale ≈ 0.0008
W_normalized = W / scale           # Values around ±1.25
W_ternary = round(W_normalized)    # All become -1, 0, or 1
# But effective weights = W_ternary * scale ≈ 0.0008 → vanishing signal
```

**Detection**:
- Monitor `scale` values during training
- Alert if `scale < 1e-4`

**Mitigation**:
```python
def safe_scale(W, min_scale=1e-6):
    scale = W.abs().mean()
    return max(scale, min_scale)
```

### C.1.2 Gradient Explosion in STE

**Symptom**: NaN or Inf during training, loss spikes.

**Root Cause**: Straight-Through Estimator passes gradients unchanged. Large pre-quantization weights receive large gradients.

**Physics**: The STE gradient approximation $\frac{\partial L}{\partial W} \approx \frac{\partial L}{\partial W_q}$ ignores the discretization effect, causing gradient scale mismatch.

**Detection**:
- Monitor gradient norms
- Check for NaN in any tensor

**Mitigation**:
```python
# Gradient clipping specific to BitLinear
max_grad_norm = 1.0
torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

# Weight regularization to keep pre-quant weights bounded
loss = loss + 0.01 * (W ** 2).mean()
```

### C.1.3 Zero-Sparsity Trap

**Symptom**: Most weights become exactly 0, destroying model capacity.

**Root Cause**: AbsMean scaling maps |W| < scale/2 to 0. If weights are initially small, most round to 0.

**Detection**:
- Monitor sparsity ratio: `(W_ternary == 0).float().mean()`
- Alert if sparsity > 0.8

**Mitigation**:
```python
# Initialize with larger weights
nn.init.normal_(W, std=0.5)  # Not std=0.02

# Or use sparsity penalty that discourages excessive zeros
sparsity = (W_ternary == 0).float().mean()
if sparsity > 0.6:
    loss = loss - 0.1 * sparsity  # Encourage non-zero
```

---

## Category 2: Rational Operator Failures

### C.2.1 Newton-Raphson Divergence

**Symptom**: rsqrt returns Inf, causes NaN propagation.

**Root Cause**: Newton-Raphson for 1/sqrt(x) diverges for x near 0 or very large x.

**Physics**: The iteration $y_{n+1} = y_n(3 - xy_n^2)/2$ requires $y_0^2 < 3/x$ for convergence.

**Detection**:
```python
def safe_babylonian_sqrt(x, n_iterations=15):
    if x <= 0:
        return 0.0
    if x > 1e12:  # Detect problematic input
        logging.warning(f"Large input to sqrt: {x}")
```

**Mitigation**:
```python
def robust_rsqrt(variance, eps=1e-6, max_variance=1e6):
    """RMSNorm-safe inverse square root"""
    variance = variance.clamp(eps, max_variance)
    y = 1.0 / variance.sqrt()  # Initial guess using hardware sqrt
    # Then refine with Newton-Raphson if needed
    for _ in range(3):
        y = y * (3 - variance * y * y) / 2
    return y
```

### C.2.2 Polynomial Softmax Underflow

**Symptom**: All attention weights become zero, attention becomes uniform.

**Root Cause**: When logits are very negative (< -10), polynomial exp (1+x/4)^4 ≈ 0.

**Physics**: The polynomial approximation has limited dynamic range:
- At x = -4: (0)^4 = 0
- At x = -8: (-1)^4 = 1 (wrong sign in intermediate)

**Detection**:
```python
x_shifted = x - x.max()
if x_shifted.min() < -10:
    logging.warning("Logits too spread for polynomial softmax")
```

**Mitigation**:
```python
def stable_poly_softmax(x, min_val=-10):
    x_shifted = x - x.max(dim=-1, keepdim=True).values
    x_clamped = x_shifted.clamp(min=min_val, max=0)

    # (1 + x/4)^4 with positivity guarantee
    t = 1 + x_clamped / 4
    t = t.clamp(min=0.01)  # Prevent negative/zero
    weights = t * t * t * t

    return weights / weights.sum(dim=-1, keepdim=True)
```

### C.2.3 Cayley Transform Singularity

**Symptom**: RoPE produces NaN for certain positions.

**Root Cause**: Cayley transform is undefined at θ = ±π (t = ±∞).

**Physics**: $\cos(\theta) = (1-t^2)/(1+t^2)$ where $t = \tan(\theta/2)$ is undefined at $\theta = \pi$.

**Detection**:
```python
# Check if any t values are extreme
if (t.abs() > 1e3).any():
    logging.warning("RoPE t values approaching singularity")
```

**Mitigation**:
```python
def safe_cayley(t, max_t=100.0):
    """Cayley transform with singularity protection"""
    t = t.clamp(-max_t, max_t)
    t_sq = t * t
    denom = 1 + t_sq + 1e-8  # Epsilon for safety
    cos_theta = (1 - t_sq) / denom
    sin_theta = (2 * t) / denom
    return cos_theta, sin_theta
```

---

## Category 3: Linear Attention Failures (Ouroboros Research)

These failure modes were discovered during Ouroboros kernel evolution experiments.

### C.3.1 DC Offset Drowning

**Symptom**: Model achieves high 1-KV recall but fails on 2-KV.

**Root Cause**: Feature map φ(x) has large constant term (base), causing all keys to have similar projections.

**Physics**: If φ(x) = base + f(x) where base >> f(x), then:
```
φ(k_A) ≈ base + small_variation
φ(k_B) ≈ base + small_variation
cosine_similarity(φ(k_A), φ(k_B)) ≈ 1  (indistinguishable!)
```

**Detection**:
```python
# Check feature map output statistics
features = kernel_fn(keys)
base_ratio = features.mean() / features.std()
if base_ratio > 10:
    logging.warning("DC offset drowning: base too large")
```

**Mitigation**:
```python
# "Spiky" initialization with low base
init_params = {
    'base': 0.01,   # Small base (not 0.5)
    'scale': 10.0,  # High scale for signal
    ...
}
```

### C.3.2 Brick Wall Saturation

**Symptom**: All feature map outputs saturate to max value (e.g., 100.0).

**Root Cause**: Hard clamping (clamp(x, 0, 100)) causes all large inputs to map to identical outputs.

**Physics**: When φ(K_A) = φ(K_B) = [100, 100, ...], keys become indistinguishable.

**Detection**:
```python
features = kernel_fn(keys)
saturation_ratio = (features >= 99.9).float().mean()
if saturation_ratio > 0.1:
    logging.warning(f"Brick wall saturation: {saturation_ratio:.1%}")
```

**Mitigation**:
```python
# Remove hard clamps, add soft penalty
def soft_bounded_kernel(x, target_scale=10.0):
    features = kernel_fn(x)  # No clamp

    # Soft magnitude penalty
    magnitude = features.abs().mean()
    penalty = 0.01 * F.relu(magnitude - target_scale)

    return features, penalty
```

### C.3.3 Linear Attention Multi-KV Barrier

**Symptom**: 1-KV recall works (100%) but 2-KV fails (~50%).

**Root Cause**: Pure linear attention with additive updates CANNOT distinguish multiple keys. This is a theoretical limitation proven in the "Based" paper.

**Physics**: Linear attention state update is:
```
S = S + outer(k, v)  # Additive

When querying with k_A:
  S @ k_A = (outer(k_A, v_A) + outer(k_B, v_B)) @ k_A
          = v_A * (k_A @ k_A) + v_B * (k_B @ k_A)
          ≠ v_A (contaminated by k_B contribution)
```

**Fundamental Limit**: Additive updates cannot overwrite previous associations.

**Solution**: DeltaNet delta rule (Schlag et al., 2021):
```python
def deltanet_update(S, k, v, beta=1.0):
    """DeltaNet enables key overwriting via gradient descent update"""
    v_old = S @ k                           # What we currently predict
    error = v_old - v                       # Prediction error
    S = S - beta * outer(error, k) / (k @ k + 1e-8)  # Gradient update
    return S

# When beta=1.0, this exactly replaces old value with new value
# Still ZK/FHE compatible: only uses +, -, *, /
```

### C.3.4 Orthogonality Collapse

**Symptom**: Keys that should be distinguishable become indistinguishable.

**Root Cause**: Feature map outputs only positive values, limiting direction diversity.

**Physics**: If φ(x) >= 0 for all x, then all feature vectors lie in the positive orthant. Maximum angle between any two vectors is 90° (not 180°).

**Detection**:
```python
features = kernel_fn(keys)
if features.min() >= 0:
    logging.warning("Feature map is all-positive: orthogonality limited")

# Check cosine similarity
cos_sim = F.cosine_similarity(features[0], features[1], dim=-1)
if cos_sim > 0.5:
    logging.warning(f"Keys not orthogonal: cos_sim={cos_sim:.3f}")
```

**Mitigation**:
```python
# Allow negative feature values (zero-centered)
def orthogonal_kernel(x, params):
    """Feature map that allows negative outputs"""
    shift, scale, a, b, c, d, e, f = params

    numerator = a*x + b*x**2 + e*x**3  # Cubic for asymmetry
    denominator = 1 + c*x**2 + d*x**4 + f*x**6

    # Shift=0 allows negative outputs!
    return shift + scale * numerator / denominator
```

---

## Category 4: Training Stability Failures

### C.4.1 Loss Spikes During Evolution

**Symptom**: Fitness drops suddenly during CMA-ES kernel evolution.

**Root Cause**: Population samples extreme parameter values that cause numerical instability.

**Mitigation**:
```python
# Stability penalty in fitness function
def fitness_with_stability(params):
    accuracy = evaluate_recall(params)

    # Penalty for extreme parameters
    param_magnitude = np.abs(params).max()
    stability_penalty = 0.1 * max(0, param_magnitude - 50)

    return accuracy - stability_penalty
```

### C.4.2 Learning Rate Sensitivity

**Symptom**: Training diverges with standard transformer LR (3e-4).

**Root Cause**: Rational operators have sharper gradients due to polynomial terms.

**Physics**: For polynomial y = x^n, gradient scales as n*x^(n-1). Higher-order terms amplify gradients.

**Mitigation**:
```python
# Use ~0.1x standard transformer LR
optimizer = AdamW(model.parameters(), lr=5e-5)

# Extended warmup for stability
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=2000,  # 4x typical
    num_training_steps=total_steps
)
```

### C.4.3 NaN Propagation Chain

**Symptom**: Single NaN in one layer propagates to entire network.

**Root Cause**: Division by zero or sqrt of negative in one operation corrupts all downstream.

**Mitigation**:
```python
def nan_guard_forward(x, name="layer"):
    """Check and handle NaN in forward pass"""
    if torch.isnan(x).any():
        logging.error(f"NaN detected in {name}")
        # Option 1: Replace with zeros
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        # Option 2: Use previous valid state
        # Option 3: Raise exception
    return x
```

---

## Category 5: Deployment Failures

### C.5.1 Precision Mismatch

**Symptom**: Model works in training but fails at deployment with different precision.

**Root Cause**: Trained in FP32, deployed in INT8. Rational operators may have tighter precision requirements.

**Detection**:
```python
# Test with target deployment precision during validation
with torch.cuda.amp.autocast():  # or explicit int8 quantization
    val_loss = model(val_data)
```

### C.5.2 Hardware Reproducibility

**Symptom**: Different outputs on different hardware despite integer-only design.

**Root Cause**:
- Different rounding modes in integer division
- Different order of operations due to parallelism

**Mitigation**:
```python
# Force consistent rounding
def integer_div(a, b):
    """Floor division for consistency"""
    return a // b  # Not a / b which uses FP division

# Or use fixed-point with explicit scaling
SCALE = 2**16
a_fixed = (a * SCALE).long()
b_fixed = (b * SCALE).long()
result = a_fixed // b_fixed  # Deterministic
```

---

## Summary Table

| ID | Failure Mode | Severity | Detection | Mitigation |
|----|--------------|----------|-----------|------------|
| C.1.1 | Distribution Collapse | High | scale < 1e-4 | Min scale clamping |
| C.1.2 | STE Gradient Explosion | Critical | NaN in grads | Gradient clipping |
| C.1.3 | Zero-Sparsity Trap | Medium | sparsity > 0.8 | Larger init, penalty |
| C.2.1 | Newton Divergence | Critical | Inf output | Input clamping |
| C.2.2 | Softmax Underflow | High | All weights ~0 | Logit clamping |
| C.2.3 | Cayley Singularity | Medium | Large t values | t value clamping |
| C.3.1 | DC Offset Drowning | High | base_ratio > 10 | Spiky init (base=0.01) |
| C.3.2 | Brick Wall Saturation | High | saturation > 10% | Remove hard clamps |
| C.3.3 | Multi-KV Barrier | Fundamental | 2KV fails | DeltaNet delta rule |
| C.3.4 | Orthogonality Collapse | High | cos_sim > 0.5 | Zero-centered features |
| C.4.1 | Evolution Spikes | Medium | Fitness drops | Stability penalty |
| C.4.2 | LR Sensitivity | High | Training diverges | 0.1x LR, long warmup |
| C.4.3 | NaN Chain | Critical | Any NaN | Guard functions |
| C.5.1 | Precision Mismatch | Medium | Deploy test fails | Test at target precision |
| C.5.2 | HW Reproducibility | Low | Cross-HW diff | Fixed-point ops |

---

## Monitoring Checklist

```python
class ILVMHealthCheck:
    """Runtime monitoring for Integer-Only LVM"""

    def check_weights(self, model):
        for name, param in model.named_parameters():
            if 'weight' in name:
                scale = param.abs().mean()
                sparsity = (param == 0).float().mean()

                assert scale > 1e-6, f"{name}: scale too small"
                assert sparsity < 0.9, f"{name}: too sparse"

    def check_activations(self, x, name):
        assert not torch.isnan(x).any(), f"{name}: NaN detected"
        assert not torch.isinf(x).any(), f"{name}: Inf detected"
        assert x.abs().max() < 1e6, f"{name}: activation explosion"

    def check_attention(self, attn_weights):
        # Should sum to 1
        assert torch.allclose(attn_weights.sum(-1),
                              torch.ones_like(attn_weights.sum(-1)),
                              rtol=0.01), "Attention weights don't sum to 1"

        # Should be non-negative
        assert (attn_weights >= 0).all(), "Negative attention weights"
```

---

*This catalog is continuously updated based on experimental findings.*
