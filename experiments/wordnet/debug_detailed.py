"""Detailed debugging with hooks on every operation."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import WordNetHierarchyDataset, collate_hierarchy_batch, create_wordnet_splits
from models import HierarchyPredictor


def debug_with_hooks():
    """Add hooks to every tensor to find NaN source."""

    print("="*60)
    print("DETAILED NaN DEBUGGING WITH HOOKS")
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

    # Manually trace forward pass with grad checking
    print("\nManual forward pass with intermediate checks:")

    # Set requires_grad
    model.train()
    leaves_input = leaves.clone()

    # Step 1: Embedding
    embeddings = model.embedding(leaves_input).unsqueeze(1)
    embeddings.retain_grad()
    print(f"\n1. Embeddings: shape={embeddings.shape}, has_nan={torch.isnan(embeddings).any()}")

    # Step 2: Normalization
    norm = embeddings.norm(dim=-1, keepdim=True) + 1e-8
    norm.retain_grad()
    normalized = embeddings / norm
    normalized.retain_grad()
    scaled = normalized * 0.1
    scaled.retain_grad()
    print(f"2. Scaled input: shape={scaled.shape}, has_nan={torch.isnan(scaled).any()}")

    # Step 3: Linear projections
    Q = model.attention.W_q(scaled)
    Q.retain_grad()
    K = model.attention.W_k(scaled)
    K.retain_grad()
    V = model.attention.W_v(scaled)
    V.retain_grad()
    print(f"3. Q/K/V projected: Q_nan={torch.isnan(Q).any()}, K_nan={torch.isnan(K).any()}, V_nan={torch.isnan(V).any()}")

    # Reshape for attention
    batch, seq, _ = scaled.shape
    Q = Q.view(batch, seq, model.attention.n_heads, model.attention.d_head).permute(0, 2, 1, 3)
    K = K.view(batch, seq, model.attention.n_heads, model.attention.d_head).permute(0, 2, 1, 3)
    V = V.view(batch, seq, model.attention.n_heads, model.attention.d_head).permute(0, 2, 1, 3)

    # Step 4: Exp map (Tangent → Manifold)
    Q_hyp = model.attention.exp_map_zero(Q)
    Q_hyp.retain_grad()
    K_hyp = model.attention.exp_map_zero(K)
    K_hyp.retain_grad()
    print(f"4. Q_hyp/K_hyp: Q_hyp_nan={torch.isnan(Q_hyp).any()}, K_hyp_nan={torch.isnan(K_hyp).any()}")

    # Step 5: Distance computation
    dist = model.attention.pairwise_poincare_distance(Q_hyp, K_hyp)
    dist.retain_grad()
    print(f"5. Hyperbolic distance: shape={dist.shape}, has_nan={torch.isnan(dist).any()}")
    print(f"   Distance stats: min={dist.min().item():.4f}, max={dist.max().item():.4f}, mean={dist.mean().item():.4f}")

    # Step 6: Attention scores
    scores = -dist / model.attention.scale
    scores.retain_grad()
    print(f"6. Attention scores: has_nan={torch.isnan(scores).any()}")
    print(f"   Scores stats: min={scores.min().item():.4f}, max={scores.max().item():.4f}, mean={scores.mean().item():.4f}")

    # Step 7: Softmax
    attn_weights = torch.softmax(scores, dim=-1)
    attn_weights.retain_grad()
    print(f"7. Attention weights: has_nan={torch.isnan(attn_weights).any()}")

    # Step 8: Value processing
    V_hyp = model.attention.exp_map_zero(V)
    V_hyp.retain_grad()
    V_tan = model.attention.log_map_zero(V_hyp)
    V_tan.retain_grad()
    print(f"8. V processing: V_hyp_nan={torch.isnan(V_hyp).any()}, V_tan_nan={torch.isnan(V_tan).any()}")

    # Step 9: Weighted sum
    out_tan = torch.matmul(attn_weights, V_tan)
    out_tan.retain_grad()
    print(f"9. Weighted sum: has_nan={torch.isnan(out_tan).any()}")

    # Step 10: Map back to manifold and tangent
    out_hyp = model.attention.exp_map_zero(out_tan)
    out_hyp.retain_grad()
    out_final_tan = model.attention.log_map_zero(out_hyp)
    out_final_tan.retain_grad()
    print(f"10. Final mappings: out_hyp_nan={torch.isnan(out_hyp).any()}, out_final_tan_nan={torch.isnan(out_final_tan).any()}")

    # Step 11: Output projection
    out_final_tan_reshaped = out_final_tan.permute(0, 2, 1, 3).contiguous().view(batch, seq, model.attention.d_model)
    attended = model.attention.W_o(out_final_tan_reshaped)
    attended.retain_grad()
    print(f"11. Attended output: has_nan={torch.isnan(attended).any()}")

    # Step 12: Predictor
    pooled = attended.squeeze(1)
    logits = model.predictor(pooled)
    logits.retain_grad()
    print(f"12. Logits: has_nan={torch.isnan(logits).any()}")

    # Create loss
    targets = torch.zeros_like(logits)
    for i in range(ancestors.shape[0]):
        valid_ancestors = ancestors[i][ancestors[i] != -1]
        targets[i, valid_ancestors] = 1.0

    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    loss = criterion(logits, targets)
    print(f"\n13. Loss: value={loss.item():.6f}, has_nan={torch.isnan(loss).any()}")

    # Backward pass
    print("\n" + "="*60)
    print("BACKWARD PASS - CHECKING EACH GRADIENT")
    print("="*60)

    loss.backward()

    tensors_to_check = [
        ("logits", logits),
        ("attended", attended),
        ("out_final_tan", out_final_tan),
        ("out_hyp", out_hyp),
        ("out_tan", out_tan),
        ("V_tan", V_tan),
        ("V_hyp", V_hyp),
        ("attn_weights", attn_weights),
        ("scores", scores),
        ("dist", dist),
        ("K_hyp", K_hyp),
        ("Q_hyp", Q_hyp),
        ("V", V),
        ("K", K),
        ("Q", Q),
        ("scaled", scaled),
        ("normalized", normalized),
        ("norm", norm),
        ("embeddings", embeddings),
    ]

    for name, tensor in tensors_to_check:
        if tensor.grad is not None:
            has_nan = torch.isnan(tensor.grad).any()
            if has_nan:
                print(f"❌ NaN in {name}.grad")
                # This is where NaN first appears
                break
            else:
                grad_norm = tensor.grad.norm().item()
                print(f"✓ {name}.grad OK (norm={grad_norm:.4f})")
        else:
            print(f"  {name}: no gradient")

    # Test multiple iterations with optimizer
    print("\n" + "="*60)
    print("TESTING MULTIPLE ITERATIONS WITH OPTIMIZER")
    print("="*60)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for step in range(3):
        print(f"\nIteration {step+1}:")

        optimizer.zero_grad()

        # Forward
        logits = model(leaves)
        print(f"  Forward: logits_nan={torch.isnan(logits).any()}")

        # Loss
        targets = torch.zeros_like(logits)
        for i in range(ancestors.shape[0]):
            valid_ancestors = ancestors[i][ancestors[i] != -1]
            targets[i, valid_ancestors] = 1.0

        loss = criterion(logits, targets)
        print(f"  Loss: value={loss.item():.6f}, nan={torch.isnan(loss).any()}")

        if torch.isnan(loss).any():
            print(f"  ❌ NaN LOSS at iteration {step+1}")
            break

        # Backward
        loss.backward()

        # Check gradients
        has_nan = False
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"  ❌ NaN gradient in {name}")
                has_nan = True
                break

        if has_nan:
            print(f"  ❌ NaN GRADIENTS at iteration {step+1}")
            break
        else:
            print(f"  ✓ All gradients finite")

        # Clip and update
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        # Check if parameters went bad
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"  ❌ NaN in {name} AFTER optimizer step!")
                break

    print("="*60)


if __name__ == "__main__":
    debug_with_hooks()
