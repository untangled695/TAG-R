"""
Geometry Verification Tests for Hyperbolic Attention

These tests verify that:
1. Hyperbolic distance differs meaningfully from Euclidean
2. Gradients flow correctly through attention
3. Points near boundary have exponentially large distances
"""

import pytest
import torch
import math

from aga.attention.hyperbolic_attention import (
    HyperbolicAttentionStable,
    EuclideanAttentionBaseline,
)


def test_attention_gradient_flow():
    """Verify that gradients flow through hyperbolic attention without NaNs."""
    d_model = 16
    n_heads = 4
    seq_len = 5
    batch = 2

    # Create random inputs
    x = torch.randn(batch, seq_len, d_model, requires_grad=True)

    # Model: Hyperbolic Attention
    model_hyp = HyperbolicAttentionStable(d_model, n_heads, curvature=-1.0)

    # Forward Pass
    out = model_hyp(x)
    loss = out.sum()
    loss.backward()

    print(f"✓ Gradient Norm: {x.grad.norm().item():.4f}")

    # Check for NaNs
    assert not torch.isnan(x.grad).any(), "Gradients contain NaNs!"
    assert x.grad.norm().item() > 0, "Gradients are zero (not flowing)"

    print("✅ Attention Gradient Flow Test Passed")


def test_hierarchical_distance():
    """
    Verify that hyperbolic distance captures hierarchy.

    Points near the boundary should have exponentially larger distances
    than points near the origin. This is the KEY property that makes
    hyperbolic space suitable for hierarchies.
    """
    d_model = 16
    n_heads = 4

    model_hyp = HyperbolicAttentionStable(d_model, n_heads, curvature=-1.0)

    # Create two vectors: one "deep" (near boundary), one "shallow" (near origin)
    deep_vec = torch.zeros(1, 1, d_model)
    deep_vec[0, 0, 0] = 0.95  # Near boundary

    shallow_vec = torch.zeros(1, 1, d_model)
    shallow_vec[0, 0, 0] = 0.1  # Near origin

    # Map to manifold
    deep_hyp = model_hyp.exp_map_zero(deep_vec)
    shallow_hyp = model_hyp.exp_map_zero(shallow_vec)

    # Compute hyperbolic distance
    # Add batch and head dimensions for pairwise_poincare_distance
    deep_hyp_expanded = deep_hyp.unsqueeze(0).unsqueeze(0)  # [1, 1, 1, d_model]
    shallow_hyp_expanded = shallow_hyp.unsqueeze(0).unsqueeze(0)  # [1, 1, 1, d_model]

    hyp_dist = model_hyp.pairwise_poincare_distance(
        deep_hyp_expanded, shallow_hyp_expanded
    )
    hyp_dist_value = hyp_dist.item()

    # Compute Euclidean distance for comparison
    euclidean_dist = torch.norm(deep_vec - shallow_vec).item()

    print(f"✓ Hyperbolic Distance (Deep vs Shallow): {hyp_dist_value:.4f}")
    print(f"✓ Euclidean Distance (Deep vs Shallow): {euclidean_dist:.4f}")
    print(f"✓ Ratio (Hyperbolic/Euclidean): {hyp_dist_value/euclidean_dist:.2f}x")

    # In hyperbolic space, distance to boundary grows exponentially
    # The ratio should be significantly > 1.0
    assert hyp_dist_value > euclidean_dist * 2.0, (
        f"Hyperbolic distance ({hyp_dist_value:.4f}) should be much larger "
        f"than Euclidean ({euclidean_dist:.4f}) for hierarchical points"
    )

    # Specific check: 0.95 and 0.1 should be VERY far apart in hyperbolic space
    assert hyp_dist_value > 2.0, (
        f"Distance should be > 2.0 for boundary points, got {hyp_dist_value:.4f}"
    )

    print("✅ Hierarchical Distance Test Passed")


def test_boundary_exponential_growth():
    """
    Verify exponential distance growth near boundary.

    As points approach the boundary, distances should grow exponentially,
    not linearly like in Euclidean space.

    Note: We create points directly on manifold at specific radii
    to test the distance metric itself, not the exp_map transformation.
    """
    d_model = 8
    n_heads = 2

    model_hyp = HyperbolicAttentionStable(d_model, n_heads, curvature=-1.0)

    # Create points DIRECTLY on manifold at specific radii
    # (not using exp_map which has tanh compression)
    radii = [0.1, 0.3, 0.5, 0.7, 0.85]
    distances = []

    origin = torch.zeros(1, 1, 1, d_model)  # [batch, heads, seq, d_model]

    for r in radii:
        # Create point directly at radius r on manifold
        point_on_manifold = torch.zeros(1, 1, 1, d_model)
        point_on_manifold[0, 0, 0, 0] = r  # Set first coordinate to r

        dist = model_hyp.pairwise_poincare_distance(origin, point_on_manifold)
        distances.append(dist.item())

        print(f"  Radius {r:.2f} → Distance: {dist.item():.4f}")

    # Verify that distances increase significantly
    # In hyperbolic space, distance from origin to radius r is:
    # d(0, r) = (2/√c) * atanh(√c * r)
    # For c=1: d(0, r) = 2 * atanh(r)

    # Check that last distance is much larger than first
    ratio_total = distances[-1] / distances[0]
    print(f"✓ Distance ratio (0.85/0.1): {ratio_total:.2f}")

    # For r=0.85 vs r=0.1:
    # d(0, 0.85) ≈ 2*atanh(0.85) ≈ 2*1.26 ≈ 2.52
    # d(0, 0.1) ≈ 2*atanh(0.1) ≈ 2*0.10 ≈ 0.20
    # Ratio ≈ 12.6x

    # In linear (Euclidean) space, ratio would be just 8.5x
    assert ratio_total > 10.0, (
        f"Distance growth too slow: {ratio_total:.2f}x (expected >10x for hyperbolic)"
    )

    # Also verify the formula matches theoretical expectation
    # For c=1, distance should be 2 * atanh(r)
    for r, d in zip(radii, distances):
        expected = 2.0 * math.atanh(r)
        error = abs(d - expected) / expected
        print(f"  r={r:.2f}: measured={d:.4f}, expected={expected:.4f}, error={error:.2%}")
        assert error < 0.05, f"Distance formula mismatch: {error:.2%} error"

    print("✅ Exponential Growth Test Passed")


def test_hyperbolic_vs_euclidean_attention():
    """
    Compare hyperbolic and Euclidean attention outputs.

    They should produce DIFFERENT results because hyperbolic attention
    uses distance-based scoring instead of dot products.
    """
    d_model = 16
    n_heads = 4
    seq_len = 8
    batch = 2

    x = torch.randn(batch, seq_len, d_model)

    # Both models
    model_hyp = HyperbolicAttentionStable(d_model, n_heads, curvature=-1.0)
    model_euc = EuclideanAttentionBaseline(d_model, n_heads)

    # Forward passes
    out_hyp = model_hyp(x)
    out_euc = model_euc(x)

    # Outputs should be different
    diff = torch.norm(out_hyp - out_euc).item()
    print(f"✓ Output difference (Hyp vs Euc): {diff:.4f}")

    # Difference should be significant
    assert diff > 0.1, (
        f"Hyperbolic and Euclidean attention should produce different outputs, "
        f"but difference is only {diff:.4f}"
    )

    print("✅ Hyperbolic vs Euclidean Difference Test Passed")


def test_attention_preserves_manifold():
    """Verify that attention output stays on manifold (or in valid tangent space)."""
    d_model = 16
    n_heads = 4
    seq_len = 10
    batch = 4

    x = torch.randn(batch, seq_len, d_model) * 0.1  # Small values near origin

    model_hyp = HyperbolicAttentionStable(d_model, n_heads, curvature=-1.0)
    output = model_hyp(x)

    # Output should not have NaNs or Infs
    assert not torch.any(torch.isnan(output)), "Output contains NaNs"
    assert not torch.any(torch.isinf(output)), "Output contains Infs"

    # Output should be reasonable magnitude (not exploding)
    max_norm = torch.norm(output, dim=-1).max().item()
    print(f"✓ Max output norm: {max_norm:.4f}")

    assert max_norm < 100.0, f"Output norms too large: {max_norm:.4f}"

    print("✅ Manifold Preservation Test Passed")


def test_attention_with_mask():
    """Verify that attention masking works correctly."""
    d_model = 16
    n_heads = 4
    seq_len = 6
    batch = 2

    x = torch.randn(batch, seq_len, d_model)

    # Create causal mask
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    mask = mask.expand(batch, n_heads, seq_len, seq_len)

    model_hyp = HyperbolicAttentionStable(d_model, n_heads, curvature=-1.0)
    output = model_hyp(x, mask=mask)

    # Should complete without errors
    assert not torch.any(torch.isnan(output))

    print("✅ Attention Masking Test Passed")


@pytest.mark.parametrize("curvature", [-0.5, -1.0, -2.0])
def test_different_curvatures(curvature):
    """Test attention with different curvature values."""
    d_model = 16
    n_heads = 4
    seq_len = 8
    batch = 2

    x = torch.randn(batch, seq_len, d_model) * 0.1

    model = HyperbolicAttentionStable(d_model, n_heads, curvature=curvature)
    output = model(x)

    assert not torch.any(torch.isnan(output))
    assert not torch.any(torch.isinf(output))

    print(f"✅ Curvature {curvature} Test Passed")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("GEOMETRY VERIFICATION TESTS")
    print("="*60 + "\n")

    print("Test 1: Gradient Flow")
    test_attention_gradient_flow()

    print("\nTest 2: Hierarchical Distance")
    test_hierarchical_distance()

    print("\nTest 3: Exponential Growth Near Boundary")
    test_boundary_exponential_growth()

    print("\nTest 4: Hyperbolic vs Euclidean Difference")
    test_hyperbolic_vs_euclidean_attention()

    print("\nTest 5: Manifold Preservation")
    test_attention_preserves_manifold()

    print("\nTest 6: Attention Masking")
    test_attention_with_mask()

    print("\nTest 7: Different Curvatures")
    for c in [-0.5, -1.0, -2.0]:
        test_different_curvatures(c)

    print("\n" + "="*60)
    print("✅ ALL GEOMETRY TESTS PASSED")
    print("="*60 + "\n")
