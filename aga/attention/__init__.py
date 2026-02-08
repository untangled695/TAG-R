"""Attention mechanisms for Adaptive Geometric Attention."""

from aga.attention.hyperbolic_attention import (
    HyperbolicAttentionStable,
    EuclideanAttentionBaseline,
)
from aga.attention.spherical_attention import SphericalAttentionStable
from aga.attention.tangent_aggregation import (
    StableTangentSpaceAggregation,
    HyperbolicMean,
    WeightedHyperbolicCentroid,
    tangent_space_mlp,
    batched_tangent_aggregation,
)

__all__ = [
    "HyperbolicAttentionStable",
    "EuclideanAttentionBaseline",
    "SphericalAttentionStable",
    "StableTangentSpaceAggregation",
    "HyperbolicMean",
    "WeightedHyperbolicCentroid",
    "tangent_space_mlp",
    "batched_tangent_aggregation",
]
