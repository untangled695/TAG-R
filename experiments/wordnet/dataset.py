"""
WordNet Hierarchy Dataset

Task: Predict ancestors in the WordNet IS-A hierarchy

This is the key validation task for testing whether hyperbolic geometry
provides better hierarchical representation than Euclidean space.

Example:
    Input:  "collie.n.01" (leaf)
    Target: ["dog.n.01", "mammal.n.01", "animal.n.01"] (ancestors)
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict

try:
    from nltk.corpus import wordnet as wn
except ImportError:
    print("WARNING: NLTK not installed. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "nltk"])
    import nltk
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    from nltk.corpus import wordnet as wn


class WordNetHierarchyDataset(Dataset):
    """
    WordNet hierarchy reconstruction dataset.

    Each sample contains:
    - leaf: A leaf synset (e.g., "collie.n.01")
    - ancestors: List of ancestor synsets in the IS-A hierarchy

    The task is to predict which synsets are ancestors of a given leaf.
    """

    def __init__(
        self,
        max_depth: int = 8,
        min_depth: int = 2,
        max_samples: int = 10000,
        pos: str = 'n',
        seed: int = 42,
    ):
        """
        Args:
            max_depth: Maximum path depth to include
            min_depth: Minimum path depth to include
            max_samples: Maximum number of samples (for faster experiments)
            pos: Part of speech ('n' = noun, 'v' = verb, etc.)
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)

        self.samples = []
        self.synset_to_idx = {}
        self.idx_to_synset = {}

        # Build vocabulary from all synsets
        print("Building WordNet vocabulary...")
        all_synsets = list(wn.all_synsets(pos))
        idx_counter = 0

        for synset in all_synsets:
            name = synset.name()
            if name not in self.synset_to_idx:
                self.synset_to_idx[name] = idx_counter
                self.idx_to_synset[idx_counter] = name
                idx_counter += 1

        self.vocab_size = len(self.synset_to_idx)
        print(f"  Vocabulary size: {self.vocab_size} synsets")

        # Build training samples from hypernym paths
        print("Building hierarchy samples...")
        for synset in all_synsets:
            paths = synset.hypernym_paths()

            if not paths:
                continue

            # Take the longest path (most informative)
            path = max(paths, key=len)

            if min_depth <= len(path) <= max_depth:
                leaf_name = synset.name()
                leaf_idx = self.synset_to_idx[leaf_name]

                # All nodes in path except the leaf are ancestors
                ancestor_names = [node.name() for node in path[:-1]]
                ancestor_indices = [self.synset_to_idx[name] for name in ancestor_names]

                if ancestor_indices:  # Must have at least one ancestor
                    self.samples.append({
                        'leaf': leaf_idx,
                        'ancestors': ancestor_indices,
                        'depth': len(path) - 1,
                    })

        # Subsample if needed
        if len(self.samples) > max_samples:
            np.random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

        print(f"  Total samples: {len(self.samples)}")
        print(f"  Average depth: {np.mean([s['depth'] for s in self.samples]):.2f}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        depths = [s['depth'] for s in self.samples]
        num_ancestors = [len(s['ancestors']) for s in self.samples]

        return {
            'num_samples': len(self.samples),
            'vocab_size': self.vocab_size,
            'mean_depth': np.mean(depths),
            'std_depth': np.std(depths),
            'min_depth': np.min(depths),
            'max_depth': np.max(depths),
            'mean_ancestors': np.mean(num_ancestors),
            'max_ancestors': np.max(num_ancestors),
        }


def collate_hierarchy_batch(batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for batching hierarchy samples.

    Args:
        batch: List of samples from WordNetHierarchyDataset

    Returns:
        leaves: Tensor of leaf indices [batch_size]
        ancestors: Padded tensor of ancestor indices [batch_size, max_ancestors]
                  Padded with -1 for varying lengths
    """
    leaves = torch.tensor([sample['leaf'] for sample in batch], dtype=torch.long)

    # Find max number of ancestors in this batch
    max_ancestors = max(len(sample['ancestors']) for sample in batch)

    # Pad ancestor lists to same length
    ancestors = torch.full((len(batch), max_ancestors), -1, dtype=torch.long)

    for i, sample in enumerate(batch):
        ancestor_list = sample['ancestors']
        ancestors[i, :len(ancestor_list)] = torch.tensor(ancestor_list, dtype=torch.long)

    return leaves, ancestors


def create_wordnet_splits(
    dataset: WordNetHierarchyDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split dataset into train/val/test.

    Args:
        dataset: WordNetHierarchyDataset instance
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        seed: Random seed

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)

    return torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )


if __name__ == "__main__":
    # Test dataset creation
    print("\n" + "="*60)
    print("WordNet Hierarchy Dataset Test")
    print("="*60 + "\n")

    dataset = WordNetHierarchyDataset(max_samples=1000)
    stats = dataset.get_statistics()

    print("\nDataset Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    print("\nExample samples:")
    for i in range(3):
        sample = dataset[i]
        leaf_name = dataset.idx_to_synset[sample['leaf']]
        ancestor_names = [dataset.idx_to_synset[idx] for idx in sample['ancestors']]

        print(f"\n  Sample {i+1}:")
        print(f"    Leaf: {leaf_name}")
        print(f"    Ancestors: {ancestor_names[:5]}...")  # Show first 5
        print(f"    Depth: {sample['depth']}")

    # Test collation
    print("\nTesting batch collation...")
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_hierarchy_batch,
    )

    leaves, ancestors = next(iter(loader))
    print(f"  Leaves shape: {leaves.shape}")
    print(f"  Ancestors shape: {ancestors.shape}")
    print(f"  Leaves: {leaves}")
    print(f"  Ancestors:\n{ancestors}")

    print("\n✅ WordNet Dataset Test Complete")
