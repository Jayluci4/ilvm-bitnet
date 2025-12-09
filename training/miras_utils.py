"""
MIRAS Memory Utilities for Stage 2b Training

Provides:
1. Memory state checkpointing (save/load multi-timescale states)
2. Memory-specific metrics (retention rates, utilization, sparsity)
3. Ablation configuration helpers
4. Retrieval probes for capability testing

Reference:
- MIRAS: "Memory Is (Really) All You Need" (Google, NeurIPS 2025)
- Nested Learning: Multi-timescale memory systems
- DeltaNet: Key overwriting for associative memory
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
import json
from pathlib import Path


# =============================================================================
# Memory State Checkpointing
# =============================================================================

def save_memory_states(
    memory_states: Dict[int, List[torch.Tensor]],
    path: Path,
) -> None:
    """Save memory states to disk.

    Memory states structure:
    {
        layer_idx: [fast_state, medium_state, slow_state],
        ...
    }

    Each state is [batch, hidden_dim, hidden_dim] tensor.
    """
    # Convert to serializable format
    serialized = {}
    for layer_idx, states in memory_states.items():
        serialized[str(layer_idx)] = [s.cpu() for s in states]

    torch.save(serialized, path)


def load_memory_states(
    path: Path,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Dict[int, List[torch.Tensor]]:
    """Load memory states from disk."""
    serialized = torch.load(path, map_location='cpu')

    memory_states = {}
    for layer_idx_str, states in serialized.items():
        layer_idx = int(layer_idx_str)
        memory_states[layer_idx] = [s.to(device=device, dtype=dtype) for s in states]

    return memory_states


def reset_memory_states(
    model: nn.Module,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Dict[int, List[torch.Tensor]]:
    """Initialize fresh memory states for all memory layers.

    Returns empty memory states dict matching model's memory layer structure.
    """
    memory_states = {}

    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'use_memory') and layer.use_memory:
            hidden_dim = model.config.hidden_dim
            num_timescales = model.config.num_timescales

            memory_states[i] = [
                torch.zeros(batch_size, hidden_dim, hidden_dim, device=device, dtype=dtype)
                for _ in range(num_timescales)
            ]

    return memory_states


# =============================================================================
# Memory Metrics
# =============================================================================

@dataclass
class MIRASMetrics:
    """Container for MIRAS-specific metrics."""
    # Retention rates (learned)
    retention_fast: float = 0.0
    retention_medium: float = 0.0
    retention_slow: float = 0.0

    # Timescale combination weights
    weight_fast: float = 0.0
    weight_medium: float = 0.0
    weight_slow: float = 0.0

    # Memory utilization (Frobenius norm)
    memory_norm_fast: float = 0.0
    memory_norm_medium: float = 0.0
    memory_norm_slow: float = 0.0

    # Memory sparsity (fraction near zero)
    memory_sparsity_fast: float = 0.0
    memory_sparsity_medium: float = 0.0
    memory_sparsity_slow: float = 0.0

    # Gradient flow
    memory_grad_norm: float = 0.0


def compute_miras_metrics(
    model: nn.Module,
    memory_states: Optional[Dict[int, List[torch.Tensor]]] = None,
) -> MIRASMetrics:
    """Compute MIRAS-specific metrics from model and memory states.

    Args:
        model: UnifiedILVM model
        memory_states: Current memory states (optional)

    Returns:
        MIRASMetrics dataclass with all metrics
    """
    metrics = MIRASMetrics()

    # Find first memory layer to get retention rates
    for layer in model.layers:
        if hasattr(layer, 'memory_attn'):
            retention = layer.memory_attn.retention

            # Get retention rates
            rates = retention.get_retention_rates()
            metrics.retention_fast = rates[0].item()
            metrics.retention_medium = rates[1].item()
            metrics.retention_slow = rates[2].item()

            # Get timescale weights (normalized)
            weights = torch.abs(retention.timescale_weights)
            weights = weights / (weights.sum() + 1e-8)
            metrics.weight_fast = weights[0].item()
            metrics.weight_medium = weights[1].item()
            metrics.weight_slow = weights[2].item()

            break

    # Compute memory state metrics if available
    if memory_states:
        # Aggregate across all layers
        total_norm = [0.0, 0.0, 0.0]
        total_sparsity = [0.0, 0.0, 0.0]
        num_layers = 0

        for layer_idx, states in memory_states.items():
            num_layers += 1
            for i, state in enumerate(states):
                # Frobenius norm (memory utilization)
                total_norm[i] += state.norm().item()

                # Sparsity (fraction of values < 0.01)
                total_sparsity[i] += (state.abs() < 0.01).float().mean().item()

        if num_layers > 0:
            metrics.memory_norm_fast = total_norm[0] / num_layers
            metrics.memory_norm_medium = total_norm[1] / num_layers
            metrics.memory_norm_slow = total_norm[2] / num_layers

            metrics.memory_sparsity_fast = total_sparsity[0] / num_layers
            metrics.memory_sparsity_medium = total_sparsity[1] / num_layers
            metrics.memory_sparsity_slow = total_sparsity[2] / num_layers

    # Compute gradient norm for memory parameters
    total_grad_norm = 0.0
    num_params = 0
    for name, param in model.named_parameters():
        if 'memory' in name and param.grad is not None:
            total_grad_norm += param.grad.norm().item() ** 2
            num_params += 1

    if num_params > 0:
        metrics.memory_grad_norm = (total_grad_norm ** 0.5) / num_params

    return metrics


def log_miras_metrics(metrics: MIRASMetrics, step: int, prefix: str = "") -> str:
    """Format MIRAS metrics for logging."""
    lines = [
        f"{prefix}MIRAS Metrics @ Step {step}:",
        f"  Retention - Fast: {metrics.retention_fast:.4f}, Med: {metrics.retention_medium:.4f}, Slow: {metrics.retention_slow:.4f}",
        f"  Weights   - Fast: {metrics.weight_fast:.4f}, Med: {metrics.weight_medium:.4f}, Slow: {metrics.weight_slow:.4f}",
        f"  Mem Norm  - Fast: {metrics.memory_norm_fast:.2f}, Med: {metrics.memory_norm_medium:.2f}, Slow: {metrics.memory_norm_slow:.2f}",
        f"  Sparsity  - Fast: {metrics.memory_sparsity_fast:.2%}, Med: {metrics.memory_sparsity_medium:.2%}, Slow: {metrics.memory_sparsity_slow:.2%}",
        f"  Grad Norm - Memory: {metrics.memory_grad_norm:.4f}",
    ]
    return "\n".join(lines)


# =============================================================================
# Ablation Configuration
# =============================================================================

@dataclass
class MIRASAblationConfig:
    """Configuration for MIRAS ablation experiments.

    Allows selectively disabling components to measure their contribution.
    """
    # Memory architecture
    use_memory: bool = True

    # Multi-timescale retention
    use_multi_timescale: bool = True
    num_timescales: int = 3  # 1 = single timescale (no Nested Learning)

    # DeltaNet delta rule
    use_deltanet_rule: bool = True
    delta_beta: float = 0.8  # 0 = no update

    # Huber loss (robustness)
    use_huber_loss: bool = True
    huber_delta: float = 1.0  # Large delta = closer to L2

    # Memory layer pattern
    memory_layer_pattern: str = "all"  # "all", "even", "odd", "last_half", "first_half", "every_4th"

    # Retention rate initialization
    fast_retention: float = 0.9
    medium_retention: float = 0.99
    slow_retention: float = 0.999

    # Learnable retention (can freeze)
    learnable_retention: bool = True

    def get_memory_layers(self, num_layers: int) -> List[int]:
        """Get list of layer indices that should have memory."""
        if not self.use_memory:
            return []

        if self.memory_layer_pattern == "all":
            return list(range(num_layers))
        elif self.memory_layer_pattern == "even":
            return [i for i in range(num_layers) if i % 2 == 0]
        elif self.memory_layer_pattern == "odd":
            return [i for i in range(num_layers) if i % 2 == 1]
        elif self.memory_layer_pattern == "last_half":
            return list(range(num_layers // 2, num_layers))
        elif self.memory_layer_pattern == "first_half":
            return list(range(num_layers // 2))
        elif self.memory_layer_pattern == "every_4th":
            return [i for i in range(num_layers) if i % 4 == 0]
        else:
            raise ValueError(f"Unknown memory_layer_pattern: {self.memory_layer_pattern}")

    def describe(self) -> str:
        """Return human-readable description of ablation config."""
        parts = []

        if not self.use_memory:
            return "NoMemory (baseline)"

        if self.use_multi_timescale:
            parts.append(f"{self.num_timescales}TS")
        else:
            parts.append("1TS")

        if self.use_deltanet_rule:
            parts.append(f"DeltaNet(b={self.delta_beta})")
        else:
            parts.append("NoDeltatNet")

        if self.use_huber_loss:
            parts.append(f"Huber(d={self.huber_delta})")
        else:
            parts.append("L2")

        parts.append(f"Layers:{self.memory_layer_pattern}")

        return "_".join(parts)


# Standard ablation configurations
ABLATION_CONFIGS = {
    "full": MIRASAblationConfig(),  # All features enabled

    "no_memory": MIRASAblationConfig(use_memory=False),

    "single_timescale": MIRASAblationConfig(
        use_multi_timescale=False,
        num_timescales=1,
    ),

    "no_deltanet": MIRASAblationConfig(
        use_deltanet_rule=False,
        delta_beta=0.0,
    ),

    "no_huber": MIRASAblationConfig(
        use_huber_loss=False,
    ),

    "fast_only": MIRASAblationConfig(
        num_timescales=1,
        fast_retention=0.9,
    ),

    "slow_only": MIRASAblationConfig(
        num_timescales=1,
        fast_retention=0.999,  # Use slow retention for single timescale
    ),

    "memory_last_half": MIRASAblationConfig(
        memory_layer_pattern="last_half",
    ),

    "memory_every_4th": MIRASAblationConfig(
        memory_layer_pattern="every_4th",
    ),
}


def get_ablation_config(name: str) -> MIRASAblationConfig:
    """Get pre-defined ablation configuration by name."""
    if name not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation: {name}. Available: {list(ABLATION_CONFIGS.keys())}")
    return ABLATION_CONFIGS[name]


# =============================================================================
# Retrieval Probes
# =============================================================================

class RetrievalProbe:
    """Test model's ability to retrieve information from N tokens ago.

    This probes the key capability that MIRAS memory should enable:
    long-range information retrieval without softmax attention's O(N^2) cost.
    """

    def __init__(self, model: nn.Module, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    def create_retrieval_prompt(
        self,
        fact: str,
        query: str,
        distance: int,
        filler: str = " The weather is nice.",
    ) -> str:
        """Create a prompt with fact, filler, and query separated by `distance` tokens.

        Structure:
        [fact] [filler repeated to reach distance] [query]

        Example:
        "The secret code is 42. The weather is nice. (x100) What is the secret code?"
        """
        # Encode fact and query to get their token counts
        fact_tokens = len(self.tokenizer.encode(fact))
        query_tokens = len(self.tokenizer.encode(query))
        filler_tokens = len(self.tokenizer.encode(filler))

        # Calculate how many filler repetitions we need
        needed_filler_tokens = distance - fact_tokens - query_tokens
        if needed_filler_tokens <= 0:
            num_fillers = 0
        else:
            num_fillers = needed_filler_tokens // filler_tokens + 1

        prompt = fact + (filler * num_fillers) + " " + query
        return prompt

    @torch.no_grad()
    def test_retrieval(
        self,
        fact: str,
        query: str,
        expected_answer: str,
        distance: int,
        max_new_tokens: int = 20,
    ) -> Tuple[bool, str]:
        """Test if model can retrieve fact from distance tokens ago.

        Args:
            fact: The fact to encode (e.g., "The secret code is 42.")
            query: The query to ask (e.g., "What is the secret code?")
            expected_answer: Expected substring in response (e.g., "42")
            distance: Number of tokens between fact and query
            max_new_tokens: Maximum tokens to generate

        Returns:
            Tuple of (success, generated_text)
        """
        self.model.eval()

        prompt = self.create_retrieval_prompt(fact, query, distance)

        # Encode
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Generate
        if hasattr(self.model, 'generate'):
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.1,  # Low temperature for determinism
            )
        else:
            # Manual generation for models without generate()
            output_ids = input_ids
            memory_states = None

            for _ in range(max_new_tokens):
                outputs = self.model(output_ids, memory_states=memory_states)
                logits = outputs["logits"][:, -1, :]
                memory_states = outputs.get("memory_states", None)

                next_token = logits.argmax(dim=-1, keepdim=True)
                output_ids = torch.cat([output_ids, next_token], dim=1)

                # Stop at EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        # Decode and check
        generated = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = generated[len(prompt):]  # Remove prompt

        success = expected_answer.lower() in response.lower()

        return success, response

    def sweep_distances(
        self,
        fact: str = "The secret code is 42.",
        query: str = "What is the secret code?",
        expected: str = "42",
        distances: List[int] = [10, 50, 100, 200, 500, 1000],
    ) -> Dict[int, Tuple[bool, str]]:
        """Test retrieval at multiple distances.

        Returns dict mapping distance -> (success, response)
        """
        results = {}

        for d in distances:
            success, response = self.test_retrieval(fact, query, expected, d)
            results[d] = (success, response)
            print(f"  Distance {d:4d}: {'PASS' if success else 'FAIL'} | '{response[:50]}...'")

        return results

    def multi_fact_recall(
        self,
        facts: List[Tuple[str, str, str]],  # List of (fact, query, expected)
        total_distance: int = 500,
    ) -> Dict[str, bool]:
        """Test recall of multiple facts interleaved.

        This tests DeltaNet's key overwriting capability.

        Args:
            facts: List of (fact, query, expected_answer) tuples
            total_distance: Total tokens in prompt

        Returns:
            Dict mapping fact index to success
        """
        # Build interleaved prompt
        prompt_parts = []
        for fact, _, _ in facts:
            prompt_parts.append(fact)

        # Add filler
        filler = " This is filler text."
        filler_needed = total_distance - sum(len(self.tokenizer.encode(p)) for p in prompt_parts)
        num_fillers = max(0, filler_needed // len(self.tokenizer.encode(filler)))
        prompt_parts.append(filler * num_fillers)

        # Add queries
        for _, query, _ in facts:
            prompt_parts.append(query)

        prompt = " ".join(prompt_parts)

        # Generate and check each fact
        results = {}
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        if hasattr(self.model, 'generate'):
            output_ids = self.model.generate(input_ids, max_new_tokens=50)
        else:
            # Simple generation
            output_ids = input_ids
            for _ in range(50):
                outputs = self.model(output_ids)
                logits = outputs["logits"][:, -1, :]
                next_token = logits.argmax(dim=-1, keepdim=True)
                output_ids = torch.cat([output_ids, next_token], dim=1)

        generated = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = generated[len(prompt):]

        for i, (_, _, expected) in enumerate(facts):
            results[f"fact_{i}"] = expected.lower() in response.lower()

        return results


# =============================================================================
# Checkpoint Enhancement
# =============================================================================

def save_miras_checkpoint(
    checkpoint: Dict[str, Any],
    memory_states: Optional[Dict[int, List[torch.Tensor]]],
    ablation_config: Optional[MIRASAblationConfig],
    metrics: Optional[MIRASMetrics],
) -> Dict[str, Any]:
    """Enhance checkpoint with MIRAS-specific data.

    Adds:
    - memory_states: Multi-timescale memory states
    - ablation_config: Ablation configuration used
    - miras_metrics: Latest MIRAS metrics
    """
    enhanced = checkpoint.copy()

    if memory_states is not None:
        # Serialize memory states
        serialized_states = {}
        for layer_idx, states in memory_states.items():
            serialized_states[str(layer_idx)] = [s.cpu() for s in states]
        enhanced["memory_states"] = serialized_states

    if ablation_config is not None:
        enhanced["ablation_config"] = asdict(ablation_config)

    if metrics is not None:
        enhanced["miras_metrics"] = asdict(metrics)

    return enhanced


def load_miras_checkpoint(
    checkpoint: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Dict[int, List[torch.Tensor]], Optional[MIRASAblationConfig], Optional[MIRASMetrics]]:
    """Load MIRAS-specific data from checkpoint.

    Returns:
        Tuple of (memory_states, ablation_config, metrics)
    """
    memory_states = {}
    ablation_config = None
    metrics = None

    if "memory_states" in checkpoint:
        for layer_idx_str, states in checkpoint["memory_states"].items():
            layer_idx = int(layer_idx_str)
            memory_states[layer_idx] = [s.to(device=device, dtype=dtype) for s in states]

    if "ablation_config" in checkpoint:
        ablation_config = MIRASAblationConfig(**checkpoint["ablation_config"])

    if "miras_metrics" in checkpoint:
        metrics = MIRASMetrics(**checkpoint["miras_metrics"])

    return memory_states, ablation_config, metrics


# =============================================================================
# Demo / Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MIRAS UTILITIES TEST")
    print("=" * 70)

    # Test ablation configs
    print("\nAblation Configurations:")
    for name, config in ABLATION_CONFIGS.items():
        print(f"  {name}: {config.describe()}")

    # Test metrics
    print("\nMetrics structure:")
    metrics = MIRASMetrics(
        retention_fast=0.9,
        retention_medium=0.99,
        retention_slow=0.999,
        weight_fast=0.33,
        weight_medium=0.33,
        weight_slow=0.34,
    )
    print(log_miras_metrics(metrics, step=1000))

    print("\n" + "=" * 70)
    print("MIRAS UTILITIES TEST COMPLETE")
    print("=" * 70)
