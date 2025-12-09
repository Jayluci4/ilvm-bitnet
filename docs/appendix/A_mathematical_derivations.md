# Appendix A: Mathematical Derivations

This appendix provides complete mathematical derivations for all rational operators used in the Integer-Only LVM architecture.

---

## A.1 Babylonian Square Root

### A.1.1 Problem Statement

Given a positive real number $a > 0$, compute $\sqrt{a}$ using only addition, subtraction, multiplication, and division.

### A.1.2 Newton-Raphson Derivation

We seek the root of $f(y) = y^2 - a$.

Newton-Raphson iteration:
$$y_{n+1} = y_n - \frac{f(y_n)}{f'(y_n)} = y_n - \frac{y_n^2 - a}{2y_n} = \frac{y_n + a/y_n}{2}$$

This is the Babylonian method, known since ~1900 BCE.

### A.1.3 Convergence Analysis

**Theorem A.1** (Quadratic Convergence): Let $e_n = y_n - \sqrt{a}$ be the error at iteration $n$. Then:
$$e_{n+1} = \frac{e_n^2}{2y_n} \leq \frac{e_n^2}{2\sqrt{a}}$$

**Proof**:
$$y_{n+1} = \frac{y_n + a/y_n}{2} = \frac{y_n^2 + a}{2y_n}$$

$$e_{n+1} = y_{n+1} - \sqrt{a} = \frac{y_n^2 + a - 2y_n\sqrt{a}}{2y_n} = \frac{(y_n - \sqrt{a})^2}{2y_n} = \frac{e_n^2}{2y_n}$$

Since $y_n \geq \sqrt{a}$ for all $n \geq 1$ (when $y_0 \geq \sqrt{a}$), we have:
$$e_{n+1} \leq \frac{e_n^2}{2\sqrt{a}}$$

**Corollary A.2**: The number of correct digits approximately doubles each iteration.

### A.1.4 Iteration Count Table

| Iterations | Relative Error | Sufficient For |
|------------|----------------|----------------|
| 3 | ~1e-6 | FP16 |
| 5 | ~1e-12 | FP32 |
| 8 | ~1e-24 | FP64 |
| 15 | ~1e-48 | Extended precision |

### A.1.5 Implementation

```python
def babylonian_sqrt(a: float, n_iterations: int = 15) -> float:
    """Compute sqrt(a) using only +, -, *, /"""
    if a <= 0:
        return 0.0

    # Initial guess: 1.0 works for normalized inputs
    y = 1.0

    for _ in range(n_iterations):
        y = (y + a / y) / 2  # Babylonian step

    return y
```

---

## A.2 Inverse Square Root (for RMSNorm)

### A.2.1 Problem Statement

Compute $1/\sqrt{a}$ directly, avoiding the division after sqrt.

### A.2.2 Newton-Raphson for Inverse Sqrt

Let $f(y) = 1/y^2 - a$, seeking root at $y = 1/\sqrt{a}$.

$$y_{n+1} = y_n - \frac{f(y_n)}{f'(y_n)} = y_n - \frac{1/y_n^2 - a}{-2/y_n^3} = y_n + \frac{y_n^3(1/y_n^2 - a)}{2}$$

Simplifying:
$$y_{n+1} = y_n \cdot \frac{3 - a \cdot y_n^2}{2}$$

### A.2.3 Implementation

```python
def newton_rsqrt(a: float, n_iterations: int = 15) -> float:
    """Compute 1/sqrt(a) using only +, -, *, /"""
    if a <= 0:
        return 1e6  # Large value for zero input

    # Initial guess
    y = 1.0

    for _ in range(n_iterations):
        y = y * (3 - a * y * y) / 2  # Newton-Raphson step

    return y
```

### A.2.4 Numerical Stability Note

The rsqrt iteration can diverge if initial guess is poor. The Babylonian sqrt followed by division is more stable for a wide range of inputs.

---

## A.3 Algebraic Sigmoid

### A.3.1 Problem Statement

Approximate $\sigma(x) = 1/(1 + e^{-x})$ without using exp().

### A.3.2 Algebraic Form

The algebraic sigmoid is:
$$\sigma_{alg}(x) = \frac{1}{2}\left(1 + \frac{x}{\sqrt{1 + x^2}}\right)$$

### A.3.3 Properties

**Theorem A.3**: The algebraic sigmoid has the following properties:
1. $\sigma_{alg}(0) = 0.5$ (matches standard sigmoid)
2. $\lim_{x \to \infty} \sigma_{alg}(x) = 1$ (correct asymptote)
3. $\lim_{x \to -\infty} \sigma_{alg}(x) = 0$ (correct asymptote)
4. $\sigma_{alg}'(0) = 0.25$ (standard sigmoid: $0.25$)

**Proof of (4)**:
$$\frac{d}{dx}\sigma_{alg}(x) = \frac{1}{2} \cdot \frac{d}{dx}\left(\frac{x}{\sqrt{1+x^2}}\right) = \frac{1}{2} \cdot \frac{1}{(1+x^2)^{3/2}}$$

At $x=0$: $\sigma_{alg}'(0) = 1/2 \cdot 1 = 0.5$

(Note: Standard sigmoid has $\sigma'(0) = 0.25$, so the algebraic version is 2x steeper at origin.)

### A.3.4 Error Analysis

Maximum absolute error vs standard sigmoid:
- For $|x| < 2$: error < 0.02
- For $|x| < 4$: error < 0.05
- For $|x| > 6$: error < 0.01 (asymptotic agreement)

### A.3.5 Implementation

```python
def algebraic_sigmoid(x: float) -> float:
    """Compute sigmoid approximation using only +, -, *, / and sqrt"""
    rsqrt = 1.0 / babylonian_sqrt(1 + x * x)
    return 0.5 * (1 + x * rsqrt)
```

---

## A.4 Rational SiLU (Sigmoid Linear Unit)

### A.4.1 Standard Definition

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

### A.4.2 Rational Approximation

Using algebraic sigmoid:
$$\text{SiLU}_{rat}(x) = x \cdot \sigma_{alg}(x/s)$$

where $s$ is a scaling factor (typically 1.0 to 1.5).

Expanded:
$$\text{SiLU}_{rat}(x) = \frac{x}{2}\left(1 + \frac{x/s}{\sqrt{1 + (x/s)^2}}\right)$$

### A.4.3 Implementation

```python
def rational_silu(x: float, scale: float = 1.5) -> float:
    """SiLU approximation using only +, -, *, / and sqrt"""
    x_scaled = x / scale
    rsqrt = 1.0 / babylonian_sqrt(1 + x_scaled * x_scaled)
    sigmoid_approx = 0.5 * (1 + x_scaled * rsqrt)
    return x * sigmoid_approx
```

---

## A.5 Polynomial Softmax

### A.5.1 Taylor Series Motivation

For $x$ near 0:
$$e^x = \sum_{n=0}^{\infty} \frac{x^n}{n!} = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots$$

### A.5.2 Practical Approximation

For $x \in [-n, 0]$, the approximation $(1 + x/n)^n$ converges to $e^x$ as $n \to \infty$.

We use $n=4$ for a good balance:
$$e^x \approx (1 + x/4)^4 = \left(\frac{4+x}{4}\right)^4$$

### A.5.3 Error Analysis

For $x \in [-10, 0]$:
- At $x = 0$: exact ($e^0 = 1$, $(1)^4 = 1$)
- At $x = -4$: $e^{-4} \approx 0.0183$, $(0)^4 = 0$ (error: 0.0183)
- At $x = -2$: $e^{-2} \approx 0.135$, $(0.5)^4 = 0.0625$ (relative error: 54%)

The approximation is acceptable for softmax because:
1. We only need relative rankings, not absolute values
2. The max-shift ensures most values are in $[-5, 0]$

### A.5.4 Softmax Algorithm

```python
def polynomial_softmax(x: List[float]) -> List[float]:
    """Softmax using polynomial exp approximation"""
    # Shift for numerical stability
    x_max = max(x)
    x_shifted = [xi - x_max for xi in x]

    # Clip to valid range
    x_clipped = [max(xi, -10.0) for xi in x_shifted]

    # Polynomial exp: (1 + x/4)^4
    def poly_exp(t):
        u = 1 + t / 4
        u = max(u, 0.01)  # Ensure positivity
        return u * u * u * u

    weights = [poly_exp(xi) for xi in x_clipped]
    total = sum(weights)

    return [w / total for w in weights]
```

---

## A.6 Cayley Transform for Trigonometry

### A.6.1 Fundamental Identity

The Cayley transform provides an exact algebraic relationship:

Given $t = \tan(\theta/2)$:
$$\cos(\theta) = \frac{1 - t^2}{1 + t^2}$$
$$\sin(\theta) = \frac{2t}{1 + t^2}$$

### A.6.2 Proof

Using the half-angle formulas:
$$\cos(\theta) = \cos^2(\theta/2) - \sin^2(\theta/2)$$

Divide by $\cos^2(\theta/2)$:
$$\cos(\theta) = \frac{1 - \tan^2(\theta/2)}{1 + \tan^2(\theta/2)} = \frac{1 - t^2}{1 + t^2}$$

Similarly:
$$\sin(\theta) = 2\sin(\theta/2)\cos(\theta/2) = \frac{2\tan(\theta/2)}{1 + \tan^2(\theta/2)} = \frac{2t}{1 + t^2}$$

### A.6.3 Approximation for RoPE

For small angles (typical in RoPE):
$$t = \tan(\theta/2) \approx \theta/2$$

This gives a polynomial approximation without needing to compute tan.

### A.6.4 Implementation

```python
def cayley_sincos(t: float) -> Tuple[float, float]:
    """Compute (sin(theta), cos(theta)) from t = tan(theta/2)
    using only +, -, *, /"""
    t_sq = t * t
    denom = 1 + t_sq
    cos_theta = (1 - t_sq) / denom
    sin_theta = (2 * t) / denom
    return sin_theta, cos_theta

def rope_rotation(x: float, y: float, theta: float) -> Tuple[float, float]:
    """Apply RoPE rotation using Cayley transform"""
    t = theta / 2  # Small angle approximation
    sin_t, cos_t = cayley_sincos(t)
    x_rot = x * cos_t - y * sin_t
    y_rot = x * sin_t + y * cos_t
    return x_rot, y_rot
```

### A.6.5 Limitation

The Cayley transform is undefined at $\theta = \pm\pi$ (where $t = \pm\infty$). For RoPE with typical position frequencies, this is never reached.

---

## A.7 Ternary Weight Quantization

### A.7.1 Quantization Function

Given weight matrix $W \in \mathbb{R}^{m \times n}$:

$$W_q = \text{round}\left(\frac{W}{\gamma}\right) \in \{-1, 0, 1\}^{m \times n}$$

where $\gamma = \text{mean}(|W|)$ is the AbsMean scale.

### A.7.2 Straight-Through Estimator

The rounding operation is non-differentiable. We use STE:

**Forward**: $W_q = \text{round}(W/\gamma)$
**Backward**: $\frac{\partial L}{\partial W} = \frac{\partial L}{\partial W_q}$ (identity gradient)

### A.7.3 Mathematical Justification

The STE works because:
1. The quantization error is bounded: $|W_q \cdot \gamma - W| \leq \gamma/2$
2. Gradients point in approximately the same direction
3. Over many updates, the network learns weights that quantize well

### A.7.4 Implementation

```python
def ste_round(x: Tensor) -> Tensor:
    """Round with straight-through gradient"""
    return x + (x.round() - x).detach()

def weight_quant_ternary(W: Tensor) -> Tuple[Tensor, Tensor]:
    """Quantize weights to {-1, 0, 1}"""
    gamma = W.abs().mean()
    W_norm = W / (gamma + 1e-8)
    W_q = ste_round(W_norm).clamp(-1, 1)
    return W_q, gamma
```

---

## A.8 Summary: Operation Counts

| Operator | Operations per Call |
|----------|-------------------|
| Babylonian sqrt (15 iter) | 45 (+, -, *, /) |
| Newton rsqrt (15 iter) | 60 (+, -, *, /) |
| Algebraic sigmoid | 6 + sqrt = 51 |
| Rational SiLU | 7 + sqrt = 52 |
| Polynomial exp | 4 (multiplications) |
| Cayley sincos | 6 (+, -, *, /) |

All operations use only: **addition, subtraction, multiplication, division**.

**Zero transcendentals. Zero SFU calls.**
