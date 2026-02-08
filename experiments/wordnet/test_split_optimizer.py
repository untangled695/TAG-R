"""Test that split optimizer allows learning (loss should decrease)."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import WordNetHierarchyDataset, collate_hierarchy_batch, create_wordnet_splits
from models import HierarchyPredictor


def test_learning():
    """Verify model can learn with split optimizer (loss should decrease)."""

    print("="*60)
    print("TESTING SPLIT OPTIMIZER - DOES LOSS DECREASE?")
    print("="*60)
    print()

    # Create dataset
    dataset = WordNetHierarchyDataset(max_samples=500, min_depth=2, max_depth=8)
    train_split, _, _ = create_wordnet_splits(dataset)

    loader = DataLoader(
        train_split,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_hierarchy_batch,
    )

    print(f"Dataset: {len(train_split)} samples")
    print()

    # Test hyperbolic model (most critical)
    attention_type = 'hyperbolic'
    print(f"Testing: {attention_type.upper()}")
    print("-"*60)

    model = HierarchyPredictor(
        vocab_size=dataset.vocab_size,
        d_model=32,
        attention_type=attention_type,
        n_heads=4,
    )

    # Split parameters
    hyperbolic_params = []
    euclidean_params = []

    for name, param in model.named_parameters():
        if 'embedding' in name:
            hyperbolic_params.append(param)
        else:
            euclidean_params.append(param)

    # Dual optimizers
    opt_hyperbolic = optim.SGD(hyperbolic_params, lr=1e-2)
    opt_euclidean = optim.AdamW(euclidean_params, lr=1e-3)

    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    model.train()

    print(f"Split: {len(hyperbolic_params)} hyperbolic, {len(euclidean_params)} euclidean")
    print()

    # Train for 3 epochs
    for epoch in range(3):
        total_loss = 0
        num_batches = 0

        for leaves, ancestors in tqdm(loader, desc=f"Epoch {epoch+1}", leave=False):
            # Zero both optimizers
            opt_hyperbolic.zero_grad()
            opt_euclidean.zero_grad()

            # Forward
            logits = model(leaves)

            # Create targets
            targets = torch.zeros_like(logits)
            for i in range(ancestors.shape[0]):
                valid_ancestors = ancestors[i][ancestors[i] != -1]
                targets[i, valid_ancestors] = 1.0

            # Loss
            loss = criterion(logits, targets)

            # Backward
            loss.backward()

            # Clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Step both
            opt_hyperbolic.step()
            opt_euclidean.step()

            # Selective projection (only embeddings)
            with torch.no_grad():
                for param in hyperbolic_params:
                    if param.dim() == 2:  # Embedding matrix
                        norm = torch.sqrt(torch.sum(param ** 2, dim=-1, keepdim=True) + 1e-10)
                        scaling = torch.where(norm > 0.95, 0.95 / norm, torch.ones_like(norm))
                        param.data.mul_(scaling)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}: Loss={avg_loss:.6f}")

    print()
    print("="*60)
    if avg_loss < 0.69:
        print("✅ SUCCESS! Loss decreased - model is learning!")
    else:
        print("❌ FAILURE: Loss did not decrease - still stuck")
    print("="*60)


if __name__ == "__main__":
    test_learning()
