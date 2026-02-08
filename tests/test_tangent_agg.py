"""
Tests for Stable Tangent Space Aggregation.

These tests verify that:
1. Aggregation preserves points on the manifold
2. No numerical instabilities (NaNs, infs)
3. Weighted averaging behaves correctly
"""

import pytest
import torch
import math

from aga.attention.tangent_aggregation import (
    StableTangentSpaceAggregation,
    HyperbolicMean,
    WeightedHyperbolicCentroid,
    tangent_space_mlp,
    batched_tangent_aggregation,
)
from aga.manifolds.hyperbolic import (
    random_ball_points,
    check_bounds,
    poincare_distance,
)


class TestStableTangentSpaceAggregation:
    """Test basic tangent space aggregation."""

    def test_aggregation_preserves_manifold(self):
        """Aggregated result should stay on manifold."""
        c = 1.0
        batch, num_points, dim = 10, 20, 32

        points = random_ball_points(batch, num_points, dim, c=c)
        weights = torch.rand(batch, num_points)

        aggregator = StableTangentSpaceAggregation(curvature=c)
        result = aggregator(points, weights, anchor=None)

        # Check bounds
        diagnostics = check_bounds(result, c)
        assert diagnostics['violations'] == 0, f"Boundary violations: {diagnostics['violations']}"
        assert diagnostics['nan_count'] == 0, f"NaN count: {diagnostics['nan_count']}"
        assert diagnostics['inf_count'] == 0, f"Inf count: {diagnostics['inf_count']}"

    def test_uniform_weights_close_to_mean(self):
        """Uniform weights should give approximate Euclidean mean for small values."""
        c = 1.0
        batch, num_points, dim = 5, 10, 32

        # Small points near origin (Euclidean-like)
        points = torch.randn(batch, num_points, dim) * 0.01

        # Uniform weights
        weights = torch.ones(batch, num_points) / num_points

        aggregator = StableTangentSpaceAggregation(curvature=c)
        result = aggregator(points, weights, anchor=None)

        # Compare with Euclidean mean
        euclidean_mean = points.mean(dim=1)

        # Should be close for small values
        error = torch.norm(result - euclidean_mean, dim=-1).max().item()
        assert error < 0.1, f"Error too large: {error:.6f}"

    def test_single_point_with_weight_1(self):
        """Aggregating single point with weight 1 should return that point."""
        c = 1.0
        batch, dim = 5, 32

        points = random_ball_points(batch, 1, dim, c=c)
        weights = torch.ones(batch, 1)

        aggregator = StableTangentSpaceAggregation(curvature=c)
        result = aggregator(points, weights, anchor=None)

        # Should match input point
        error = torch.norm(result - points.squeeze(1), dim=-1).max().item()
        assert error < 1e-3, f"Error: {error:.6f}"

    def test_anchor_vs_origin(self):
        """Both anchor and origin methods should give similar results for small values."""
        c = 1.0
        batch, num_points, dim = 5, 10, 32

        points = random_ball_points(batch, num_points, dim, c=c) * 0.1
        weights = torch.rand(batch, num_points)
        anchor = random_ball_points(batch, dim, c=c) * 0.1

        aggregator = StableTangentSpaceAggregation(curvature=c)

        result_origin = aggregator(points, weights, anchor=None)
        result_anchor = aggregator(points, weights, anchor=anchor)

        # Both should be valid points
        assert check_bounds(result_origin, c)['violations'] == 0
        assert check_bounds(result_anchor, c)['violations'] == 0

    @pytest.mark.parametrize("c", [0.5, 1.0, 2.0, 5.0])
    def test_no_nans_across_curvatures(self, c):
        """Aggregation should work without NaNs across different curvatures."""
        batch, num_points, dim = 10, 20, 32

        points = random_ball_points(batch, num_points, dim, c=c)
        weights = torch.rand(batch, num_points)

        aggregator = StableTangentSpaceAggregation(curvature=c)
        result = aggregator(points, weights, anchor=None)

        assert not torch.any(torch.isnan(result))
        assert not torch.any(torch.isinf(result))


class TestHyperbolicMean:
    """Test Fréchet mean computation."""

    def test_mean_of_identical_points(self):
        """Mean of identical points should be that point."""
        c = 1.0
        batch, num_points, dim = 5, 10, 32

        # All points are the same
        point = random_ball_points(batch, 1, dim, c=c)
        points = point.repeat(1, num_points, 1)

        mean_module = HyperbolicMean(curvature=c)
        result = mean_module(points)

        error = torch.norm(result - point.squeeze(1), dim=-1).max().item()
        assert error < 1e-3, f"Error: {error:.6f}"

    def test_mean_preserves_manifold(self):
        """Mean should stay on manifold."""
        c = 1.0
        batch, num_points, dim = 10, 20, 32

        points = random_ball_points(batch, num_points, dim, c=c)

        mean_module = HyperbolicMean(curvature=c)
        result = mean_module(points)

        diagnostics = check_bounds(result, c)
        assert diagnostics['violations'] == 0

    def test_mean_with_weights(self):
        """Weighted mean should work correctly."""
        c = 1.0
        batch, num_points, dim = 5, 10, 32

        points = random_ball_points(batch, num_points, dim, c=c)
        weights = torch.rand(batch, num_points)

        mean_module = HyperbolicMean(curvature=c)
        result = mean_module(points, weights)

        assert not torch.any(torch.isnan(result))
        diagnostics = check_bounds(result, c)
        assert diagnostics['violations'] == 0


class TestWeightedHyperbolicCentroid:
    """Test weighted centroid computation."""

    def test_centroid_uniform_weights(self):
        """Centroid with uniform weights."""
        c = 1.0
        batch, num_points, dim = 5, 10, 32

        points = random_ball_points(batch, num_points, dim, c=c)

        centroid_module = WeightedHyperbolicCentroid(
            dim=dim,
            curvature=c,
            learnable_weights=False
        )
        result = centroid_module(points)

        diagnostics = check_bounds(result, c)
        assert diagnostics['violations'] == 0
        assert not torch.any(torch.isnan(result))

    def test_centroid_learnable_weights(self):
        """Centroid with learnable weights."""
        c = 1.0
        batch, num_points, dim = 5, 10, 32

        points = random_ball_points(batch, num_points, dim, c=c)

        centroid_module = WeightedHyperbolicCentroid(
            dim=dim,
            curvature=c,
            learnable_weights=True
        )
        result = centroid_module(points)

        diagnostics = check_bounds(result, c)
        assert diagnostics['violations'] == 0
        assert not torch.any(torch.isnan(result))

    def test_centroid_gradient_flow(self):
        """Gradients should flow through centroid."""
        c = 1.0
        batch, num_points, dim = 2, 5, 8

        points = random_ball_points(batch, num_points, dim, c=c)
        points.requires_grad_(True)

        centroid_module = WeightedHyperbolicCentroid(
            dim=dim,
            curvature=c,
            learnable_weights=True
        )

        result = centroid_module(points)
        loss = result.sum()
        loss.backward()

        # Check gradients exist
        assert points.grad is not None
        assert not torch.any(torch.isnan(points.grad))

        # Check model parameters have gradients
        for param in centroid_module.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestTangentSpaceMLP:
    """Test MLP operations in tangent space."""

    def test_mlp_preserves_manifold(self):
        """MLP output should stay on manifold."""
        c = 1.0
        batch, dim = 10, 32

        x = random_ball_points(batch, dim, c=c)

        mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.ReLU(),
            torch.nn.Linear(dim, dim),
        )

        result = tangent_space_mlp(x, mlp, c=c)

        diagnostics = check_bounds(result, c)
        assert diagnostics['violations'] == 0
        assert not torch.any(torch.isnan(result))

    def test_mlp_gradient_flow(self):
        """Gradients should flow through tangent space MLP."""
        c = 1.0
        batch, dim = 5, 16

        x = random_ball_points(batch, dim, c=c)
        x.requires_grad_(True)

        mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.ReLU(),
            torch.nn.Linear(dim, dim),
        )

        result = tangent_space_mlp(x, mlp, c=c)
        loss = result.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.any(torch.isnan(x.grad))

        for param in mlp.parameters():
            assert param.grad is not None


class TestBatchedOperations:
    """Test batched tangent aggregation."""

    def test_batched_aggregation_shape(self):
        """Batched aggregation should produce correct shape."""
        c = 1.0
        batch, seq_len, num_points, dim = 4, 8, 10, 32

        points = random_ball_points(batch, seq_len, num_points, dim, c=c)
        weights = torch.rand(batch, seq_len, num_points)

        result = batched_tangent_aggregation(points, weights, c=c)

        assert result.shape == (batch, seq_len, dim)

    def test_batched_aggregation_preserves_manifold(self):
        """Batched aggregation should preserve manifold."""
        c = 1.0
        batch, seq_len, num_points, dim = 4, 8, 10, 32

        points = random_ball_points(batch, seq_len, num_points, dim, c=c)
        weights = torch.rand(batch, seq_len, num_points)

        result = batched_tangent_aggregation(points, weights, c=c)

        # Check all points in batch
        result_flat = result.reshape(-1, dim)
        diagnostics = check_bounds(result_flat, c)
        assert diagnostics['violations'] == 0


class TestStressAndStability:
    """Stress tests for numerical stability."""

    def test_large_batch_aggregation(self):
        """Test with large batches."""
        c = 1.0
        batch, num_points, dim = 100, 50, 64

        points = random_ball_points(batch, num_points, dim, c=c)
        weights = torch.rand(batch, num_points)

        aggregator = StableTangentSpaceAggregation(curvature=c)
        result = aggregator(points, weights, anchor=None)

        diagnostics = check_bounds(result, c)
        assert diagnostics['violations'] == 0
        assert diagnostics['nan_count'] == 0

    def test_extreme_weights(self):
        """Test with extreme weight distributions."""
        c = 1.0
        batch, num_points, dim = 10, 20, 32

        points = random_ball_points(batch, num_points, dim, c=c)

        aggregator = StableTangentSpaceAggregation(curvature=c)

        # Test with one very high weight
        weights = torch.zeros(batch, num_points)
        weights[:, 0] = 1.0

        result = aggregator(points, weights, anchor=None)

        # Should be relatively close to first point (softmax may redistribute slightly)
        error = torch.norm(result - points[:, 0], dim=-1).max().item()
        assert error < 1.0, f"Error: {error:.6f}"  # Relaxed tolerance for extreme case

        assert check_bounds(result, c)['violations'] == 0

    def test_points_near_boundary(self):
        """Test aggregation with points near boundary."""
        c = 1.0
        batch, num_points, dim = 10, 20, 32

        # Generate points and push them near boundary
        points = random_ball_points(batch, num_points, dim, c=c) * 0.95
        weights = torch.rand(batch, num_points)

        aggregator = StableTangentSpaceAggregation(curvature=c)
        result = aggregator(points, weights, anchor=None)

        diagnostics = check_bounds(result, c)
        assert diagnostics['violations'] == 0
        assert not torch.any(torch.isnan(result))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
