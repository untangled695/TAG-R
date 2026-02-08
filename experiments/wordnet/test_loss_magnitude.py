"""Test that initial loss magnitude is reasonable for vocab task."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math

from dataset import WordNetHierarchyDataset, collate_hierarchy_batch, create_wordnet_splits
from models import HierarchyPredictor


def test_initial_loss():
    """Verify initial loss is appropriate for vocabulary task."""

    print("="*60)
    print("TESTING INITIAL LOSS MAGNITUDE")
    print("="*60)
    print()

    # Create dataset
    dataset = WordNetHierarchyDataset(max_samples=200, min_depth=2, max_depth=8)
    train_split, _, _ = create_wordnet_splits(dataset)

    loader = DataLoader(
        train_split,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_hierarchy_batch,
    )

    vocab_size = dataset.vocab_size
    print(f"Vocabulary size: {vocab_size:,}")
    print(f"Expected random loss (uniform): {math.log(vocab_size):.4f}")
    print()

    # Test all three attention types
    for attention_type in ['euclidean', 'hyperbolic', 'adaptive']:
        print(f"\n{'-'*60}")
        print(f"Testing: {attention_type.upper()}")
        print(f"{'-'*60}")

        model = HierarchyPredictor(
            vocab_size=vocab_size,
            d_model=32,
            attention_type=attention_type,
            n_heads=4,
        )

        model.eval()
        criterion = nn.BCEWithLogitsLoss(reduction='mean')

        # Get one batch
        leaves, ancestors = next(iter(loader))

        with torch.no_grad():
            logits = model(leaves)

        # Create targets
        targets = torch.zeros_like(logits)
        for i in range(ancestors.shape[0]):
            valid_ancestors = ancestors[i][ancestors[i] != -1]
            targets[i, valid_ancestors] = 1.0

        loss = criterion(logits, targets)

        print(f"  Initial loss: {loss.item():.4f}")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Logits stats: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")
        print(f"  Logits std: {logits.std():.4f}")

        # Check if loss is degenerate
        if loss.item() < 0.1:
            print(f"  ❌ DEGENERATE: Loss too low! Model collapsed.")
        elif 0.6 < loss.item() < 0.8:
            print(f"  ⚠️  SUSPICIOUS: Loss ≈ 0.69 = -ln(0.5) (binary classification?)")
        elif loss.item() > 0.8:
            print(f"  ✓ HEALTHY: Loss indicates proper vocab-scale task")
        else:
            print(f"  ? UNCLEAR: Loss in unusual range")

    print()
    print("="*60)


if __name__ == "__main__":
    test_initial_loss()
