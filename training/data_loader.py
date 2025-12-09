"""
Memory-Efficient Data Loader for I-LVM Training

Supports three training stages for 125M proof-of-concept:
1. TinyStories (1-2 epochs): Quick validation of learning + stability
2. FineWeb-Edu (10B tokens): Real training, measure perplexity
3. SlimPajama (2-6B tokens): Generalization testing

Uses streaming datasets to avoid loading entire dataset into memory.
Implements tokenization on-the-fly and dynamic batching.
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
from typing import Optional, Iterator, Dict, Any, Tuple
import random
from dataclasses import dataclass


# ============================================================================
# Dataset Configurations for PoC Training Stages
# ============================================================================

@dataclass
class DatasetConfig:
    """Configuration for a training dataset."""
    name: str
    hf_path: str
    hf_config: Optional[str]
    split: str
    text_field: str
    estimated_tokens: int  # Approximate token count
    download_size_gb: float
    description: str


# Stage 1: TinyStories - Quick validation
TINYSTORIES_CONFIG = DatasetConfig(
    name="tinystories",
    hf_path="roneneldan/TinyStories",
    hf_config=None,
    split="train",
    text_field="text",
    estimated_tokens=500_000_000,  # ~500M tokens
    download_size_gb=0.5,
    description="Stage 1: Quick validation - Does model learn? Are operators stable?"
)

# Stage 2: FineWeb-Edu - Real training
FINEWEB_EDU_CONFIG = DatasetConfig(
    name="fineweb-edu",
    hf_path="HuggingFaceFW/fineweb-edu",
    hf_config="sample-10BT",  # 10B token sample (~10GB)
    split="train",
    text_field="text",
    estimated_tokens=10_000_000_000,  # 10B tokens
    download_size_gb=10.0,
    description="Stage 2: Real training - Measure perplexity, compare vs GPT-2 124M"
)

# Stage 3: SlimPajama - Generalization
SLIMPAJAMA_CONFIG = DatasetConfig(
    name="slimpajama",
    hf_path="cerebras/SlimPajama-627B",
    hf_config=None,
    split="train",
    text_field="text",
    estimated_tokens=627_000_000_000,  # 627B total, we use 2-6B slice
    download_size_gb=900.0,  # Full dataset, streaming recommended
    description="Stage 3: Generalization - Test diverse text, validate expressive completeness"
)

# C4 fallback
C4_CONFIG = DatasetConfig(
    name="c4",
    hf_path="allenai/c4",
    hf_config="en",
    split="train",
    text_field="text",
    estimated_tokens=365_000_000_000,
    download_size_gb=300.0,
    description="Fallback: C4 English dataset"
)

DATASET_REGISTRY = {
    "tinystories": TINYSTORIES_CONFIG,
    "fineweb-edu": FINEWEB_EDU_CONFIG,
    "slimpajama": SLIMPAJAMA_CONFIG,
    "c4": C4_CONFIG,
}


def get_dataset_config(name: str) -> DatasetConfig:
    """Get dataset configuration by name."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name]


def print_dataset_info(name: str):
    """Print information about a dataset."""
    config = get_dataset_config(name)
    print(f"\n{'='*70}")
    print(f"DATASET: {config.name}")
    print(f"{'='*70}")
    print(f"  Path: {config.hf_path}")
    if config.hf_config:
        print(f"  Config: {config.hf_config}")
    print(f"  Tokens: {config.estimated_tokens:,}")
    print(f"  Download: ~{config.download_size_gb:.1f} GB")
    print(f"  Purpose: {config.description}")
    print(f"{'='*70}\n")


# ============================================================================
# Tokenizer Wrapper
# ============================================================================

class SimpleTokenizer:
    """Simple byte-level tokenizer for testing without HuggingFace tokenizers."""

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        # Reserve special tokens
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.unk_token_id = 3

    def encode(self, text: str, add_special_tokens: bool = False) -> list:
        """Encode text to token IDs using byte encoding."""
        tokens = [self.bos_token_id] if add_special_tokens else []
        for char in text:
            # Map bytes to vocab range (skip special tokens)
            token_id = (ord(char) % (self.vocab_size - 4)) + 4
            tokens.append(token_id)
        if add_special_tokens:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, tokens: list) -> str:
        """Decode token IDs back to text."""
        chars = []
        for t in tokens:
            if t < 4:  # Skip special tokens
                continue
            chars.append(chr((t - 4) % 256))
        return ''.join(chars)


def get_tokenizer(tokenizer_name: Optional[str] = None, vocab_size: int = 50257):
    """Get tokenizer - uses GPT-2 by default for fair comparison with GPT-2 124M baseline.

    GPT-2 tokenizer benefits:
    - Consistent vocabulary (50,257 tokens)
    - Fair perplexity comparison with GPT-2 124M
    - Well-tested BPE tokenization
    """
    # Default to GPT-2 tokenizer for fair comparison
    if tokenizer_name is None:
        tokenizer_name = "gpt2"

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        print(f"Using HuggingFace tokenizer: {tokenizer_name} (vocab_size={len(tokenizer)})")
        return tokenizer
    except Exception as e:
        print(f"Failed to load {tokenizer_name}: {e}")
        print(f"Falling back to SimpleTokenizer (vocab_size={vocab_size})")
        return SimpleTokenizer(vocab_size=vocab_size)


# ============================================================================
# Streaming Dataset
# ============================================================================

class StreamingTextDataset(IterableDataset):
    """Streaming dataset that yields tokenized text samples.

    Features:
    - Streams data from HuggingFace datasets
    - Tokenizes on-the-fly to save memory
    - Packs multiple short sequences for efficiency
    - Handles sequence length properly
    - Supports distributed training (shards data across GPUs)
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        max_seq_len: int = 512,
        tokenizer: Optional[Any] = None,
        vocab_size: int = 32000,
        seed: int = 42,
        max_samples: Optional[int] = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.dataset_config = dataset_config
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer or SimpleTokenizer(vocab_size)
        self.vocab_size = vocab_size
        self.seed = seed
        self.max_samples = max_samples
        self.rank = rank
        self.world_size = world_size
        self._dataset = None

    def _get_dataset(self):
        """Lazy load dataset with proper distributed sharding."""
        if self._dataset is None:
            try:
                from datasets import load_dataset

                load_kwargs = {
                    "path": self.dataset_config.hf_path,
                    "split": self.dataset_config.split,
                    "streaming": True,
                }
                if self.dataset_config.hf_config:
                    load_kwargs["name"] = self.dataset_config.hf_config

                print(f"Loading {self.dataset_config.name} (streaming)...")
                self._dataset = load_dataset(**load_kwargs)

                # Apply distributed sharding at dataset level (proper HuggingFace way)
                # This ensures each GPU gets a unique shard of the data
                if self.world_size > 1:
                    print(f"  Sharding for distributed training: rank {self.rank}/{self.world_size}")
                    self._dataset = self._dataset.shard(
                        num_shards=self.world_size,
                        index=self.rank
                    )

                self._dataset = self._dataset.shuffle(seed=self.seed + self.rank, buffer_size=10000)
                print(f"Dataset {self.dataset_config.name} loaded successfully!")

            except Exception as e:
                print(f"Failed to load {self.dataset_config.name}: {e}")
                print("Falling back to synthetic data for testing...")
                self._dataset = None

        return self._dataset

    def _tokenize(self, text: str) -> list:
        """Tokenize text using the configured tokenizer."""
        if hasattr(self.tokenizer, 'encode'):
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
        else:
            # Fallback: simple character encoding
            tokens = [ord(c) % self.vocab_size for c in text]
        return tokens

    def _generate_synthetic_sample(self) -> Dict[str, torch.Tensor]:
        """Generate synthetic sample for testing without real data."""
        input_ids = torch.randint(0, self.vocab_size, (self.max_seq_len,))
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": torch.ones(self.max_seq_len),
        }

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over tokenized samples.

        Note: Distributed sharding is handled at the dataset level via .shard()
        in _get_dataset(), so no manual sharding is needed here.
        """
        print(f"[Rank {self.rank}] Starting __iter__...")
        dataset = self._get_dataset()
        print(f"[Rank {self.rank}] Dataset obtained, starting iteration...")

        if dataset is None:
            # Synthetic data fallback - apply sharding for synthetic data
            sample_count = 0
            global_count = 0
            while True:
                if self.max_samples and sample_count >= self.max_samples:
                    break
                # Shard synthetic data: only yield if global_count % world_size == rank
                if global_count % self.world_size == self.rank:
                    yield self._generate_synthetic_sample()
                    sample_count += 1
                global_count += 1
            return

        # Buffer for packing sequences
        token_buffer = []
        sample_count = 0
        text_field = self.dataset_config.text_field
        example_count = 0

        print(f"[Rank {self.rank}] Starting to iterate over dataset...")
        for example in dataset:
            if example_count == 0:
                print(f"[Rank {self.rank}] First example received!")
            example_count += 1
            if self.max_samples and sample_count >= self.max_samples:
                break

            # Get text from example
            if isinstance(example, dict):
                text = example.get(text_field, "")
            else:
                text = str(example)

            if not text:
                continue

            # Tokenize
            tokens = self._tokenize(text)
            token_buffer.extend(tokens)

            # Yield complete sequences (dataset already sharded via .shard())
            while len(token_buffer) >= self.max_seq_len:
                sequence = token_buffer[:self.max_seq_len]
                token_buffer = token_buffer[self.max_seq_len:]

                input_ids = torch.tensor(sequence, dtype=torch.long)

                if sample_count == 0:
                    print(f"[Rank {self.rank}] Yielding first full sequence!")

                yield {
                    "input_ids": input_ids,
                    "labels": input_ids.clone(),
                    "attention_mask": torch.ones(self.max_seq_len),
                }
                sample_count += 1

                if self.max_samples and sample_count >= self.max_samples:
                    break


class SyntheticDataset(IterableDataset):
    """Synthetic dataset for testing without external dependencies."""

    def __init__(
        self,
        vocab_size: int = 32000,
        max_seq_len: int = 512,
        num_samples: int = 100000,
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_samples = num_samples

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Generate synthetic samples."""
        for _ in range(self.num_samples):
            input_ids = torch.randint(0, self.vocab_size, (self.max_seq_len,))
            yield {
                "input_ids": input_ids,
                "labels": input_ids.clone(),
                "attention_mask": torch.ones(self.max_seq_len),
            }


# ============================================================================
# Data Loader Factory
# ============================================================================

def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """Collate batch of samples into tensors."""
    input_ids = torch.stack([x["input_ids"] for x in batch])
    labels = torch.stack([x["labels"] for x in batch])
    attention_mask = torch.stack([x["attention_mask"] for x in batch])

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def create_dataloader(
    dataset_name: str = "tinystories",
    dataset_config: Optional[str] = None,  # Override for custom HF config
    split: str = "train",
    batch_size: int = 4,
    max_seq_len: int = 512,
    vocab_size: int = 32000,
    num_workers: int = 0,
    tokenizer: Optional[Any] = None,
    tokenizer_name: Optional[str] = None,
    use_synthetic: bool = False,
    seed: int = 42,
    max_samples: Optional[int] = None,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """Create a memory-efficient data loader.

    Args:
        dataset_name: Dataset name (tinystories, fineweb-edu, slimpajama, c4)
        dataset_config: Override HuggingFace config
        split: Dataset split (train, validation, test)
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        vocab_size: Vocabulary size
        num_workers: Number of data loading workers
        tokenizer: Optional tokenizer instance
        tokenizer_name: HuggingFace tokenizer name to load
        use_synthetic: Use synthetic data for testing
        seed: Random seed
        max_samples: Maximum number of samples to yield

    Returns:
        DataLoader instance
    """
    if use_synthetic:
        dataset = SyntheticDataset(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            num_samples=max_samples or 100000,
        )
    else:
        # Get dataset configuration
        config = get_dataset_config(dataset_name)

        # Override config if provided
        if dataset_config:
            config.hf_config = dataset_config
        if split:
            config.split = split

        # Get tokenizer
        if tokenizer is None and tokenizer_name:
            tokenizer = get_tokenizer(tokenizer_name, vocab_size)
        elif tokenizer is None:
            tokenizer = SimpleTokenizer(vocab_size)

        dataset = StreamingTextDataset(
            dataset_config=config,
            max_seq_len=max_seq_len,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            seed=seed,
            max_samples=max_samples,
            rank=rank,
            world_size=world_size,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )


def create_poc_dataloaders(
    stage: str = "tinystories",
    batch_size: int = 4,
    max_seq_len: int = 512,
    vocab_size: int = 32000,
    tokenizer_name: Optional[str] = None,
    seed: int = 42,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create train and eval dataloaders for a PoC training stage.

    Args:
        stage: Training stage (tinystories, fineweb-edu, slimpajama)
        batch_size: Training batch size
        max_seq_len: Maximum sequence length
        vocab_size: Vocabulary size
        tokenizer_name: HuggingFace tokenizer to use
        seed: Random seed

    Returns:
        Tuple of (train_loader, eval_loader)
    """
    print_dataset_info(stage)

    # Get tokenizer (shared between train and eval)
    tokenizer = get_tokenizer(tokenizer_name, vocab_size)

    # Create train loader
    train_loader = create_dataloader(
        dataset_name=stage,
        split="train",
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        tokenizer=tokenizer,
        seed=seed,
    )

    # Create eval loader (if validation split exists)
    eval_loader = None
    try:
        eval_loader = create_dataloader(
            dataset_name=stage,
            split="validation",
            batch_size=batch_size * 2,  # Larger batch for eval
            max_seq_len=max_seq_len,
            vocab_size=vocab_size,
            tokenizer=tokenizer,
            seed=seed,
            max_samples=1000,  # Limit eval samples
        )
    except Exception as e:
        print(f"No validation split available: {e}")

    return train_loader, eval_loader


# ============================================================================
# CLI Test
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test data loaders")
    parser.add_argument("--dataset", type=str, default="tinystories",
                        choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--num_batches", type=int, default=5)
    args = parser.parse_args()

    print("="*70)
    print("I-LVM DATA LOADER TEST")
    print("="*70)

    # Print available datasets
    print("\nAvailable datasets:")
    for name, config in DATASET_REGISTRY.items():
        print(f"  - {name}: {config.description[:50]}...")

    if args.synthetic:
        print("\n1. Testing SYNTHETIC data loader:")
        loader = create_dataloader(
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            use_synthetic=True,
        )
    else:
        print(f"\n1. Testing {args.dataset.upper()} data loader:")
        loader = create_dataloader(
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            use_synthetic=False,
        )

    # Test loading batches
    print(f"\nLoading {args.num_batches} batches...")
    for i, batch in enumerate(loader):
        if i >= args.num_batches:
            break
        print(f"  Batch {i+1}: input_ids={batch['input_ids'].shape}, "
              f"labels={batch['labels'].shape}")

    print("\nData loader test complete!")
