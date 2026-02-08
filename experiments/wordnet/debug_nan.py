"""Debug script to identify where NaN originates in hyperbolic training."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import WordNetHierarchyDataset, collate_hierarchy_batch, create_wordnet_splits
from models import HierarchyPredictor


def debug_forward_pass():
    """Run a single forward pass with detailed NaN checking."""

    print("="*60)
    print("NaN DEBUGGING - HYPERBOLIC ATTENTION")
    print("="*60)

    # Create tiny dataset
    dataset = WordNetHierarchyDataset(max_samples=100, min_depth=2, max_depth=7)
    train_split, _, _ = create_wordnet_splits(dataset)

    loader = DataLoader(
        train_split,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_hierarchy_batch,
    )

    # Create hyperbolic model
    model = HierarchyPredictor(
        vocab_size=dataset.vocab_size,
        d_model=16,
        attention_type='hyperbolic',
        n_heads=4,
    )

    # Get one batch
    leaves, ancestors = next(iter(loader))

    print(f"\nInput:")
    print(f"  Leaves shape: {leaves.shape}")
    print(f"  Leaves values: {leaves}")
    print(f"  Vocab size: {dataset.vocab_size}")

    # Step 1: Check embedding
    print(f"\n{'='*60}")
    print("STEP 1: EMBEDDING LAYER")
    print(f"{'='*60}")
    embeddings = model.embedding(leaves).unsqueeze(1)
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Embeddings norm: {embeddings.norm(dim=-1).mean():.6f}")
    print(f"  Has NaN: {torch.isnan(embeddings).any()}")
    print(f"  Stats: min={embeddings.min():.6f}, max={embeddings.max():.6f}, mean={embeddings.mean():.6f}")

    # Step 2: Check pre-normalization
    print(f"\n{'='*60}")
    print("STEP 2: PRE-LAYER NORMALIZATION")
    print(f"{'='*60}")
    normalized = embeddings / (embeddings.norm(dim=-1, keepdim=True) + 1e-8)
    scaled = normalized * 0.1
    print(f"  After normalization shape: {scaled.shape}")
    print(f"  After normalization norm: {scaled.norm(dim=-1).mean():.6f}")
    print(f"  Has NaN: {torch.isnan(scaled).any()}")
    print(f"  Stats: min={scaled.min():.6f}, max={scaled.max():.6f}, mean={scaled.mean():.6f}")

    # Step 3: Check attention forward (this is where NaN likely occurs)
    print(f"\n{'='*60}")
    print("STEP 3: HYPERBOLIC ATTENTION")
    print(f"{'='*60}")

    try:
        # Hook to check intermediate values
        def check_nan_hook(module, input, output):
            if isinstance(output, torch.Tensor):
                if torch.isnan(output).any():
                    print(f"  ⚠️  NaN detected in {module.__class__.__name__} output!")
                    print(f"      Output shape: {output.shape}")
                    print(f"      Stats: min={output[~torch.isnan(output)].min() if (~torch.isnan(output)).any() else float('nan'):.6f}")

        # Register hooks
        for name, module in model.named_modules():
            if hasattr(module, 'forward'):
                module.register_forward_hook(check_nan_hook)

        attended = model.attention(scaled)
        print(f"  Attention output shape: {attended.shape}")
        print(f"  Has NaN: {torch.isnan(attended).any()}")
        if torch.isnan(attended).any():
            print(f"  ❌ NaN FOUND IN ATTENTION OUTPUT")
        else:
            print(f"  ✓ No NaN in attention output")
            print(f"  Stats: min={attended.min():.6f}, max={attended.max():.6f}, mean={attended.mean():.6f}")

    except Exception as e:
        print(f"  ❌ EXCEPTION in attention: {e}")
        import traceback
        traceback.print_exc()

    # Step 4: Check final prediction
    print(f"\n{'='*60}")
    print("STEP 4: FULL FORWARD PASS")
    print(f"{'='*60}")

    try:
        logits = model(leaves)
        print(f"  Logits shape: {logits.shape}")
        print(f"  Has NaN: {torch.isnan(logits).any()}")
        if torch.isnan(logits).any():
            print(f"  ❌ NaN FOUND IN LOGITS")
            nan_count = torch.isnan(logits).sum().item()
            total = logits.numel()
            print(f"  NaN count: {nan_count}/{total} ({100*nan_count/total:.2f}%)")
        else:
            print(f"  ✓ No NaN in logits")
            print(f"  Stats: min={logits.min():.6f}, max={logits.max():.6f}, mean={logits.mean():.6f}")

    except Exception as e:
        print(f"  ❌ EXCEPTION in forward pass: {e}")
        import traceback
        traceback.print_exc()

    # Step 5: Check backward pass
    print(f"\n{'='*60}")
    print("STEP 5: BACKWARD PASS (GRADIENT COMPUTATION)")
    print(f"{'='*60}")

    try:
        # Create loss
        targets = torch.zeros_like(logits)
        for i in range(ancestors.shape[0]):
            valid_ancestors = ancestors[i][ancestors[i] != -1]
            targets[i, valid_ancestors] = 1.0

        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        loss = criterion(logits, targets)

        print(f"  Loss value: {loss.item():.6f}")
        print(f"  Loss has NaN: {torch.isnan(loss).any()}")

        # Backward
        loss.backward()

        # Check gradients
        print(f"\n  Checking gradients...")
        has_nan_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"    ❌ NaN in {name} gradient!")
                    has_nan_grad = True
                else:
                    grad_norm = param.grad.norm().item()
                    if grad_norm > 100:
                        print(f"    ⚠️  Large gradient in {name}: {grad_norm:.2f}")

        if not has_nan_grad:
            print(f"    ✓ All gradients are finite")

    except Exception as e:
        print(f"  ❌ EXCEPTION in backward pass: {e}")
        import traceback
        traceback.print_exc()

    # Step 6: Test multiple training steps
    print(f"\n{'='*60}")
    print("STEP 6: MULTIPLE TRAINING ITERATIONS")
    print(f"{'='*60}")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    for step in range(5):
        optimizer.zero_grad()

        # Forward
        logits = model(leaves)

        # Loss
        targets = torch.zeros_like(logits)
        for i in range(ancestors.shape[0]):
            valid_ancestors = ancestors[i][ancestors[i] != -1]
            targets[i, valid_ancestors] = 1.0

        loss = criterion(logits, targets)

        print(f"  Step {step+1}: Loss={loss.item():.6f}, Logits NaN={torch.isnan(logits).any()}, Loss NaN={torch.isnan(loss).any()}")

        if torch.isnan(loss).any():
            print(f"    ❌ NaN DETECTED AT STEP {step+1}")
            break

        # Backward
        loss.backward()

        # Check for NaN gradients
        has_nan_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"    ❌ NaN gradient in {name}")
                has_nan_grad = True
                break

        if has_nan_grad:
            print(f"    ❌ NaN DETECTED IN GRADIENTS AT STEP {step+1}")
            break

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        # Update
        optimizer.step()

    print(f"\n{'='*60}")
    print("DEBUGGING COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    debug_forward_pass()
