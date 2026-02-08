"""
Unit tests for hyperbolic operations.

These tests are CRITICAL for ensuring numerical stability.
All tests must pass before proceeding with the rest of the implementation.
"""

import pytest
import torch
import math
from aga.manifolds.hyperbolic import (
    project_onto_ball,
    mobius_add,
    exp_map,
    log_map,
    exp_map_zero,
    log_map_zero,
    poincare_distance,
    pairwise_poincare_distance,
    parallel_transport,
    random_ball_points,
    check_bounds,
)


@pytest.fixture
def random_points():
    """Generate random points for testing."""
    torch.manual_seed(42)
    return {
        'x': random_ball_points(100, 32, c=1.0),
        'y': random_ball_points(100, 32, c=1.0),
        'v': torch.randn(100, 32) * 0.1,
    }


class TestProjection:
    """Test projection onto Poincaré ball."""

    def test_projection_keeps_points_in_ball(self):
        """Projected points should be within ball."""
        c = 1.0
        x = torch.randn(1000, 32) * 100  # Extreme values
        x_proj = project_onto_ball(x, c, eps=1e-3)

        max_norm = 1.0 / math.sqrt(c) - 1e-3
        norms = torch.norm(x_proj, dim=-1)

        # Allow small floating point tolerance
        assert torch.all(norms <= max_norm + 1e-6), f"Max norm: {norms.max():.6f}, allowed: {max_norm:.6f}"

    def test_projection_identity_for_interior_points(self):
        """Points already in ball should be unchanged."""
        c = 1.0
        x = torch.randn(100, 32) * 0.01  # Small values, definitely in ball
        x_proj = project_onto_ball(x, c, eps=1e-3)

        assert torch.allclose(x, x_proj, atol=1e-6)

    @pytest.mark.parametrize("c", [0.5, 1.0, 2.0, 5.0])
    def test_projection_across_curvatures(self, c):
        """Projection should work for different curvatures."""
        x = torch.randn(100, 32)
        x_proj = project_onto_ball(x, c, eps=1e-3)

        max_norm = 1.0 / math.sqrt(c) - 1e-3
        norms = torch.norm(x_proj, dim=-1)

        # Allow small floating point tolerance
        assert torch.all(norms <= max_norm + 1e-6)
        assert not torch.any(torch.isnan(x_proj))


class TestMobiusAddition:
    """Test Möbius addition."""

    def test_mobius_add_identity(self):
        """x ⊕ 0 = x"""
        c = 1.0
        x = random_ball_points(100, 32, c=c)
        zero = torch.zeros_like(x)

        result = mobius_add(x, zero, c)

        assert torch.allclose(result, x, atol=1e-4)

    def test_mobius_add_inverse(self):
        """x ⊕ (-x) ≈ 0"""
        c = 1.0
        x = random_ball_points(100, 32, c=c) * 0.1  # Small to avoid numerical issues

        result = mobius_add(x, -x, c)
        norms = torch.norm(result, dim=-1)

        assert torch.all(norms < 1e-3), f"Max norm: {norms.max():.6f}"

    def test_mobius_add_stays_in_ball(self):
        """Result of Möbius addition should be in ball."""
        c = 1.0
        x = random_ball_points(100, 32, c=c)
        y = random_ball_points(100, 32, c=c)

        result = mobius_add(x, y, c)
        norms = torch.norm(result, dim=-1)

        max_norm = 1.0 / math.sqrt(c)
        assert torch.all(norms < max_norm), f"Max norm: {norms.max():.6f}"

    @pytest.mark.parametrize("c", [0.5, 1.0, 2.0, 5.0])
    def test_mobius_add_no_nans(self, c):
        """Möbius addition should not produce NaNs."""
        x = random_ball_points(100, 32, c=c)
        y = random_ball_points(100, 32, c=c)

        result = mobius_add(x, y, c)

        assert not torch.any(torch.isnan(result))
        assert not torch.any(torch.isinf(result))


class TestExpLogMaps:
    """Test exponential and logarithmic maps."""

    def test_exp_log_identity(self):
        """exp(log(y, x), x) ≈ y"""
        c = 1.0
        x = random_ball_points(100, 32, c=c)
        y = random_ball_points(100, 32, c=c)

        v = log_map(y, x, c)
        y_rec = exp_map(v, x, c)

        error = torch.norm(y - y_rec, dim=-1).max().item()
        assert error < 1e-3, f"Exp-log identity error: {error:.6f}"

    def test_log_exp_identity(self):
        """log(exp(v, x), x) ≈ v"""
        c = 1.0
        x = random_ball_points(100, 32, c=c)
        v = torch.randn(100, 32) * 0.1  # Small tangent vectors

        y = exp_map(v, x, c)
        v_rec = log_map(y, x, c)

        error = torch.norm(v - v_rec, dim=-1).max().item()
        assert error < 1e-3, f"Log-exp identity error: {error:.6f}"

    def test_exp_map_stays_in_ball(self):
        """Exponential map should produce points in ball."""
        c = 1.0
        x = random_ball_points(100, 32, c=c)
        v = torch.randn(100, 32) * 0.5

        y = exp_map(v, x, c)
        norms = torch.norm(y, dim=-1)

        max_norm = 1.0 / math.sqrt(c)
        assert torch.all(norms < max_norm), f"Max norm: {norms.max():.6f}"

    @pytest.mark.parametrize("c", [0.5, 1.0, 2.0, 5.0])
    def test_exp_log_no_nans_across_curvatures(self, c):
        """Exp/log maps should not produce NaNs."""
        x = random_ball_points(100, 32, c=c)
        y = random_ball_points(100, 32, c=c)

        v = log_map(y, x, c)
        assert not torch.any(torch.isnan(v))

        y_rec = exp_map(v, x, c)
        assert not torch.any(torch.isnan(y_rec))


class TestExpLogZero:
    """Test simplified exponential/logarithmic maps at origin."""

    def test_exp_log_zero_identity(self):
        """exp_0(log_0(x)) ≈ x"""
        c = 1.0
        x = random_ball_points(100, 32, c=c)

        v = log_map_zero(x, c)
        x_rec = exp_map_zero(v, c)

        error = torch.norm(x - x_rec, dim=-1).max().item()
        assert error < 1e-3, f"Exp-log-zero identity error: {error:.6f}"

    def test_log_exp_zero_identity(self):
        """log_0(exp_0(v)) ≈ v"""
        c = 1.0
        v = torch.randn(100, 32) * 0.1

        x = exp_map_zero(v, c)
        v_rec = log_map_zero(x, c)

        error = torch.norm(v - v_rec, dim=-1).max().item()
        assert error < 1e-3, f"Log-exp-zero identity error: {error:.6f}"

    def test_exp_zero_equivalent_to_exp(self):
        """exp_0(v) should equal exp(v, 0)"""
        c = 1.0
        v = torch.randn(100, 32) * 0.1
        zero = torch.zeros(100, 32)

        result1 = exp_map_zero(v, c)
        result2 = exp_map(v, zero, c)

        assert torch.allclose(result1, result2, atol=1e-4)


class TestPoincareDistance:
    """Test hyperbolic distance."""

    def test_distance_symmetry(self):
        """d(x, y) = d(y, x)"""
        c = 1.0
        x = random_ball_points(100, 32, c=c)
        y = random_ball_points(100, 32, c=c)

        d_xy = poincare_distance(x, y, c)
        d_yx = poincare_distance(y, x, c)

        assert torch.allclose(d_xy, d_yx, atol=1e-5)

    def test_distance_identity(self):
        """d(x, x) = 0"""
        c = 1.0
        x = random_ball_points(100, 32, c=c)

        d = poincare_distance(x, x, c)

        assert torch.all(d < 1e-5), f"Max distance: {d.max():.6f}"

    def test_distance_positivity(self):
        """d(x, y) > 0 for x ≠ y"""
        c = 1.0
        x = random_ball_points(100, 32, c=c)
        y = random_ball_points(100, 32, c=c)

        d = poincare_distance(x, y, c)

        assert torch.all(d >= 0)

    def test_triangle_inequality(self):
        """d(x, z) ≤ d(x, y) + d(y, z)"""
        c = 1.0
        x = random_ball_points(50, 32, c=c)
        y = random_ball_points(50, 32, c=c)
        z = random_ball_points(50, 32, c=c)

        d_xz = poincare_distance(x, z, c)
        d_xy = poincare_distance(x, y, c)
        d_yz = poincare_distance(y, z, c)

        # Allow small numerical tolerance
        assert torch.all(d_xz <= d_xy + d_yz + 1e-3)


class TestPairwiseDistance:
    """Test pairwise distance computation."""

    def test_pairwise_shape(self):
        """Pairwise distance should have correct shape."""
        c = 1.0
        x = random_ball_points(10, 8, 32, c=c)  # [10, 8, 32]
        y = random_ball_points(10, 12, 32, c=c)  # [10, 12, 32]

        dists = pairwise_poincare_distance(x, y, c)

        assert dists.shape == (10, 8, 12)

    def test_pairwise_matches_individual(self):
        """Pairwise should match individual distance calls."""
        c = 1.0
        x = random_ball_points(5, 3, 32, c=c)
        y = random_ball_points(5, 4, 32, c=c)

        dists_pairwise = pairwise_poincare_distance(x, y, c)

        # Check a few random entries
        for b in range(5):
            for i in range(3):
                for j in range(4):
                    d_individual = poincare_distance(x[b, i], y[b, j], c)
                    d_pairwise = dists_pairwise[b, i, j]
                    assert torch.allclose(d_individual, d_pairwise, atol=1e-5)


class TestStabilityStress:
    """Stress tests for numerical stability."""

    def test_exp_log_identity_stress_10k(self):
        """Run exp-log identity test 10,000 times."""
        c = 1.0
        max_error = 0.0

        for _ in range(100):  # 100 batches of 100 = 10,000 total
            x = random_ball_points(100, 32, c=c)
            y = random_ball_points(100, 32, c=c)

            v = log_map(y, x, c)
            y_rec = exp_map(v, x, c)

            error = torch.norm(y - y_rec, dim=-1).max().item()
            max_error = max(max_error, error)

            assert error < 1e-3, f"Exp-log identity failed: {error:.6f}"
            assert not torch.any(torch.isnan(y_rec))

        print(f"✅ Exp-log identity stress test passed. Max error: {max_error:.6f}")

    def test_training_simulation_1000_steps(self):
        """Simulate 1000 training steps without NaNs."""
        c = 1.0
        embeddings = torch.nn.Parameter(random_ball_points(1000, 32, c=c))

        optimizer = torch.optim.Adam([embeddings], lr=0.01)

        for step in range(1000):
            # Simulate forward pass
            x = embeddings[:100]
            y = embeddings[100:200]

            # Compute some loss using hyperbolic operations
            v = log_map(y, x, c)
            y_pred = exp_map(v * 0.9, x, c)  # Simulate prediction

            loss = torch.sum((y - y_pred) ** 2)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Check gradients
            assert not torch.any(torch.isnan(embeddings.grad)), f"NaN gradient at step {step}"

            optimizer.step()

            # Project back onto ball after update
            with torch.no_grad():
                embeddings.data = project_onto_ball(embeddings.data, c, eps=1e-3)

            # Check embeddings
            diagnostics = check_bounds(embeddings, c)
            assert diagnostics['violations'] == 0, f"Boundary violation at step {step}"
            assert diagnostics['nan_count'] == 0, f"NaN in embeddings at step {step}"

        print(f"✅ Training simulation passed 1000 steps without issues")

    def test_extreme_values(self):
        """Test operations with extreme values."""
        c = 1.0

        # Very small values
        x_small = torch.randn(10, 32) * 1e-8
        y_small = torch.randn(10, 32) * 1e-8

        result = mobius_add(x_small, y_small, c)
        assert not torch.any(torch.isnan(result))

        # Values near boundary
        x_boundary = random_ball_points(10, 32, c=c)
        x_boundary = project_onto_ball(x_boundary * 0.99, c, eps=1e-4)

        v = torch.randn(10, 32) * 0.01
        result = exp_map(v, x_boundary, c)
        assert not torch.any(torch.isnan(result))

        diagnostics = check_bounds(result, c)
        assert diagnostics['violations'] == 0


class TestUtilities:
    """Test utility functions."""

    def test_random_ball_points(self):
        """Random points should be in ball."""
        c = 1.0
        points = random_ball_points(100, 32, c=c)

        diagnostics = check_bounds(points, c)
        assert diagnostics['violations'] == 0
        assert diagnostics['nan_count'] == 0

    def test_check_bounds(self):
        """Check bounds should return correct diagnostics."""
        c = 1.0
        x = random_ball_points(100, 32, c=c)

        diagnostics = check_bounds(x, c)

        assert 'max_norm' in diagnostics
        assert 'min_norm' in diagnostics
        assert 'violations' in diagnostics
        assert diagnostics['max_norm'] < 1.0 / math.sqrt(c)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
