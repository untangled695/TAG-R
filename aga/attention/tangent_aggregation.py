"""
Stable Tangent Space Aggregation for Hyperbolic Attention

This module implements the key innovation for averaging in hyperbolic space:
Map to tangent space → average (Euclidean) → map back to manifold

This preserves the hyperbolic geometry while enabling stable weighted aggregation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from aga.manifolds.hyperbolic import (
    exp_map,
    log_map,
    exp_map_zero,
    log_map_zero,
    project_onto_ball,
)


class StableTangentSpaceAggregation(nn.Module):
    """
    Aggregate attention-weighted values while preserving hyperbolic structure.

    Strategy: V lives on manifold → log to tangent → weighted average → exp back

    This is geometrically correct and numerically stable because:
    1. Tangent space at any point is isomorphic to R^d (Euclidean)
    2. Weighted averaging is well-defined in Euclidean space
    3. Exp map correctly "bends" the result back onto the manifold
    """

    def __init__(self, curvature: float = 1.0):
        super().__init__()
        self.c = curvature

    def forward(
        self,
        points: torch.Tensor,
        weights: torch.Tensor,
        anchor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Aggregate points using weighted average in tangent space.

        Args:
            points: Points on Poincaré ball [batch, num_points, dim]
            weights: Attention weights [batch, num_points]
            anchor: Base point for tangent space [batch, dim]
                   If None, uses origin (more efficient)

        Returns:
            Aggregated point on Poincaré ball [batch, dim]
        """
        batch, num_points, dim = points.shape

        # Normalize weights (should already be normalized, but ensure it)
        weights_norm = F.softmax(weights, dim=-1).unsqueeze(-1)  # [batch, num_points, 1]

        if anchor is None:
            # Use origin as anchor (more efficient, simplified formulas)
            return self._aggregate_at_origin(points, weights_norm)
        else:
            # Use specified anchor point
            return self._aggregate_at_point(points, weights_norm, anchor)

    def _aggregate_at_origin(
        self,
        points: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate using origin as anchor (simplified version).

        Args:
            points: [batch, num_points, dim]
            weights: [batch, num_points, 1]

        Returns:
            [batch, dim]
        """
        # Step 1: Map all points to tangent space at origin
        tangent_vecs = log_map_zero(points, self.c)  # [batch, num_points, dim]

        # Step 2: Weighted average in tangent space (Euclidean operation - SAFE!)
        aggregated_tangent = torch.sum(weights * tangent_vecs, dim=1)  # [batch, dim]

        # Step 3: Map back to manifold
        aggregated_point = exp_map_zero(aggregated_tangent, self.c)  # [batch, dim]

        return aggregated_point

    def _aggregate_at_point(
        self,
        points: torch.Tensor,
        weights: torch.Tensor,
        anchor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate using specified anchor point.

        Args:
            points: [batch, num_points, dim]
            weights: [batch, num_points, 1]
            anchor: [batch, dim]

        Returns:
            [batch, dim]
        """
        # Step 1: Map all points to tangent space at anchor
        anchor_expanded = anchor.unsqueeze(1)  # [batch, 1, dim]
        tangent_vecs = log_map(points, anchor_expanded, self.c)  # [batch, num_points, dim]

        # Step 2: Weighted average in tangent space
        aggregated_tangent = torch.sum(weights * tangent_vecs, dim=1)  # [batch, dim]

        # Step 3: Map back to manifold from anchor
        aggregated_point = exp_map(aggregated_tangent, anchor, self.c)  # [batch, dim]

        return aggregated_point


class HyperbolicMean(nn.Module):
    """
    Compute Fréchet mean (hyperbolic average) of points.

    The Fréchet mean is the point that minimizes the sum of squared
    hyperbolic distances to all input points.

    For computational efficiency, we use tangent space aggregation
    as an approximation.
    """

    def __init__(self, curvature: float = 1.0, max_iter: int = 10):
        super().__init__()
        self.c = curvature
        self.max_iter = max_iter
        self.aggregator = StableTangentSpaceAggregation(curvature)

    def forward(
        self,
        points: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute weighted Fréchet mean.

        Args:
            points: Points on Poincaré ball [batch, num_points, dim]
            weights: Optional weights [batch, num_points]
                    If None, use uniform weights

        Returns:
            Fréchet mean [batch, dim]
        """
        batch, num_points, dim = points.shape

        if weights is None:
            weights = torch.ones(batch, num_points, device=points.device) / num_points

        # Use tangent space aggregation as approximation
        # This is efficient and numerically stable
        mean = self.aggregator(points, weights, anchor=None)

        return mean


class WeightedHyperbolicCentroid(nn.Module):
    """
    Compute weighted centroid in hyperbolic space.

    This is a learnable module that can be used as a pooling operation
    for hyperbolic neural networks.
    """

    def __init__(
        self,
        dim: int,
        curvature: float = 1.0,
        learnable_weights: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.c = curvature
        self.aggregator = StableTangentSpaceAggregation(curvature)

        if learnable_weights:
            # Learn attention weights for pooling
            self.weight_net = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, 1),
            )
        else:
            self.weight_net = None

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted centroid.

        Args:
            points: Points on Poincaré ball [batch, num_points, dim]

        Returns:
            Centroid [batch, dim]
        """
        batch, num_points, dim = points.shape

        if self.weight_net is not None:
            # Learned weights
            logits = self.weight_net(points).squeeze(-1)  # [batch, num_points]
            weights = F.softmax(logits, dim=-1)
        else:
            # Uniform weights
            weights = torch.ones(batch, num_points, device=points.device) / num_points

        centroid = self.aggregator(points, weights, anchor=None)

        return centroid


def tangent_space_mlp(
    x: torch.Tensor,
    mlp: nn.Module,
    c: float = 1.0,
) -> torch.Tensor:
    """
    Apply MLP in tangent space at origin.

    This allows using standard Euclidean MLPs with hyperbolic inputs.

    Args:
        x: Points on Poincaré ball [*, dim]
        mlp: PyTorch MLP module
        c: Curvature

    Returns:
        Transformed points on Poincaré ball [*, dim]
    """
    # Map to tangent space
    x_tangent = log_map_zero(x, c)

    # Apply MLP (Euclidean operation)
    y_tangent = mlp(x_tangent)

    # Map back to manifold
    y = exp_map_zero(y_tangent, c)

    return y


# Utility functions for batched operations

def batched_tangent_aggregation(
    points: torch.Tensor,
    weights: torch.Tensor,
    c: float = 1.0,
) -> torch.Tensor:
    """
    Convenience function for batched aggregation at origin.

    Args:
        points: [batch, seq_len, num_points, dim]
        weights: [batch, seq_len, num_points]
        c: Curvature

    Returns:
        [batch, seq_len, dim]
    """
    batch, seq_len, num_points, dim = points.shape

    # Reshape for processing
    points_flat = points.reshape(batch * seq_len, num_points, dim)
    weights_flat = weights.reshape(batch * seq_len, num_points)

    # Aggregate
    aggregator = StableTangentSpaceAggregation(curvature=c)
    result_flat = aggregator(points_flat, weights_flat, anchor=None)

    # Reshape back
    result = result_flat.reshape(batch, seq_len, dim)

    return result
