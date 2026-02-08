"""Quick test to verify NaN fixes work."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import WordNetHierarchyDataset, collate_hierarchy_batch, create_wordnet_splits
from models import HierarchyPredictor


def test_nan_fix():
    """Test if the 3 critical fixes eliminate NaN."""

    print("="*60)
    print("TESTING NaN FIX - ALL 3 FIXES APPLIED")
    print("="*60)
    print()
    print("Fixes applied:")
    print("  1. SafeAtanh with gradient clamping (max=100.0)")
    print("  2. SGD optimizer (no momentum staleness)")
    print("  3. Safe norm with eps inside sqrt")
    print()

    # Create tiny dataset
    dataset = WordNetHierarchyDataset(max_samples=200, min_depth=2, max_depth=8)
    train_split, _, _ = create_wordnet_splits(dataset)

    loader = DataLoader(
        train_split,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_hierarchy_batch,
    )

    print(f"Dataset: {len(train_split)} samples")
    print()

    # Test hyperbolic model
    print("-"*60)
    print("Testing HYPERBOLIC model")
    print("-"*60)

    model = HierarchyPredictor(
        vocab_size=dataset.vocab_size,
        d_model=32,
        attention_type='hyperbolic',
        n_heads=4,
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Run 10 iterations
    model.train()
    success = True

    for step in range(10):
        # Get batch
        leaves, ancestors = next(iter(loader))

        optimizer.zero_grad()

        # Forward
        logits = model(leaves)

        # Create targets
        targets = torch.zeros_like(logits)
        for i in range(ancestors.shape[0]):
            valid_ancestors = ancestors[i][ancestors[i] != -1]
            targets[i, valid_ancestors] = 1.0

        # Loss
        loss = criterion(logits, targets)

        # Check for NaN
        if torch.isnan(loss).any():
            print(f"❌ Iteration {step+1}: NaN LOSS detected!")
            print(f"   Loss value: {loss.item()}")
            success = False
            break

        # Backward
        loss.backward()

        # Check gradients
        has_nan_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"❌ Iteration {step+1}: NaN gradient in {name}")
                has_nan_grad = True
                break

        if has_nan_grad:
            success = False
            break

        # Clip and update
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        # Project parameters
        with torch.no_grad():
            for param in model.parameters():
                param.data = torch.clamp(param.data, -1.0, 1.0)

        # Check parameters
        has_nan_param = False
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"❌ Iteration {step+1}: NaN in {name} after optimizer step!")
                has_nan_param = True
                break

        if has_nan_param:
            success = False
            break

        # Print progress
        print(f"✓ Iteration {step+1}: Loss={loss.item():.6f}, grad_norm={torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')):.4f}")

    print()
    if success:
        print("="*60)
        print("🎉 SUCCESS! All 10 iterations completed without NaN!")
        print("="*60)
    else:
        print("="*60)
        print("❌ FAILURE: NaN still appears despite fixes")
        print("="*60)

    print()
    return success


if __name__ == "__main__":
    test_nan_fix()
