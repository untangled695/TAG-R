"""
Curvature-Aware Router for Adaptive Geometric Attention

The router analyzes input structure and predicts which geometry
(Euclidean, Hyperbolic, or Spherical) is most appropriate for each token.

Key Innovation: Soft routing allows mixing geometries, enabling
representation of "mixed curvature" structures (e.g., trees with cycles).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CurvatureAwareRouter(nn.Module):
    """
    Routes tokens to appropriate geometric attention mechanisms.

    The router learns to recognize structural patterns:
    - Tree-like structure → Hyperbolic (hierarchical)
    - Cycle-like structure → Spherical (periodic/symmetric)
    - Flat structure → Euclidean (standard)

    Uses soft routing (mixing outputs) rather than hard routing (selecting one)
    to support mixed-curvature representations.
    """

    def __init__(
        self,
        d_model: int,
        num_geometries: int = 3,
        hidden_dim: int = None,
        temperature: float = 1.0,
    ):
        """
        Args:
            d_model: Input dimension
            num_geometries: Number of geometric experts (3: Euclidean, Hyperbolic, Spherical)
            hidden_dim: Hidden dimension for MLP (default: d_model // 2)
            temperature: Temperature for softmax (lower = sharper routing)
        """
        super().__init__()
        self.num_geometries = num_geometries
        self.temperature = temperature

        if hidden_dim is None:
            hidden_dim = d_model // 2

        # Curvature probe: projects state to geometry logits
        # 2-layer MLP captures non-linear structural cues
        self.probe = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_geometries),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict geometry weights for each token.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            weights: Softmax probabilities [batch, seq_len, num_geometries]
            logits: Raw scores [batch, seq_len, num_geometries] (for load balancing)
        """
        # Project to geometry logits
        logits = self.probe(x)  # [batch, seq_len, num_geometries]

        # Apply temperature and softmax for soft routing
        # Temperature < 1.0 makes routing sharper (more decisive)
        # Temperature > 1.0 makes routing softer (more mixing)
        weights = F.softmax(logits / self.temperature, dim=-1)

        return weights, logits

    def get_top_k_routing(
        self, x: torch.Tensor, k: int = 2
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get top-k geometry routing (for sparse computation).

        This can be used for efficiency: only compute attention for top-k geometries.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            k: Number of geometries to select per token

        Returns:
            top_k_weights: Renormalized weights for top-k [batch, seq_len, k]
            top_k_indices: Indices of selected geometries [batch, seq_len, k]
            logits: Raw scores [batch, seq_len, num_geometries]
        """
        weights, logits = self.forward(x)

        # Get top-k geometries
        top_k_weights, top_k_indices = torch.topk(weights, k, dim=-1)

        # Renormalize so top-k weights sum to 1
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        return top_k_weights, top_k_indices, logits


class GeometryFeatureExtractor(nn.Module):
    """
    Extract structural features that indicate geometry type.

    This module can optionally be used before the router to provide
    explicit geometric features (e.g., local tree-ness, cycle-ness).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def compute_local_structure_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute local structural features from embeddings.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            features: Structural features [batch, seq_len, num_features]
        """
        batch, seq_len, d_model = x.shape

        # Compute pairwise distances (Euclidean)
        # x_expanded: [batch, seq_len, 1, d_model]
        # x_expanded_t: [batch, 1, seq_len, d_model]
        x_expanded = x.unsqueeze(2)
        x_expanded_t = x.unsqueeze(1)
        dists = torch.norm(x_expanded - x_expanded_t, dim=-1)  # [batch, seq_len, seq_len]

        # Feature 1: Tree-ness
        # High ratio of max/min neighbor distances suggests hierarchical structure
        # Add small epsilon to avoid division by zero
        neighbor_dists, _ = torch.topk(dists, k=min(5, seq_len), dim=-1, largest=False)
        max_neighbor = neighbor_dists[:, :, -1]
        min_neighbor = neighbor_dists[:, :, 1].clamp(min=1e-6)  # Skip self (index 0)
        tree_ness = (max_neighbor / min_neighbor).unsqueeze(-1)

        # Feature 2: Cycle-ness
        # Low variance in neighbor distances suggests regularity/symmetry
        cycle_ness = (1.0 / (torch.var(neighbor_dists, dim=-1) + 1e-6)).unsqueeze(-1)

        # Feature 3: Flatness
        # Low overall distance variance suggests Euclidean structure
        flat_ness = (1.0 / (torch.var(dists, dim=-1) + 1e-6)).unsqueeze(-1)

        # Combine features
        features = torch.cat([tree_ness, cycle_ness, flat_ness], dim=-1)

        # Normalize features to [0, 1] range using sigmoid
        features = torch.sigmoid(features / 10.0)  # Scale down to avoid saturation

        return features


# Load Balancing Loss Functions

def load_balancing_loss(router_logits: torch.Tensor, importance_factor: float = 0.01) -> torch.Tensor:
    """
    Encourages router to use all geometries equally across batch.

    Prevents "collapse mode" where router only uses one geometry (typically Euclidean).

    This is critical! Without load balancing, the router will converge to
    using only Euclidean attention because it's easiest to optimize initially.

    Args:
        router_logits: Raw routing scores [batch, seq_len, num_geometries]
        importance_factor: Weight for load balance loss (0.01 is typical)

    Returns:
        Load balance loss (scalar)
    """
    # Convert logits to probabilities
    probs = F.softmax(router_logits, dim=-1)  # [batch, seq_len, num_geometries]

    # Calculate usage of each geometry across batch and sequence
    usage = torch.mean(probs, dim=(0, 1))  # [num_geometries]

    # Ideal: each geometry used 1/num_geometries of the time
    num_geometries = probs.shape[-1]
    target_usage = torch.ones_like(usage) / num_geometries

    # MSE loss between actual and target usage
    balance_loss = F.mse_loss(usage, target_usage)

    return balance_loss * importance_factor


def load_balancing_loss_cv(router_logits: torch.Tensor, importance_factor: float = 0.01) -> torch.Tensor:
    """
    Alternative load balancing using coefficient of variation.

    Minimizes variance in geometry usage relative to mean.

    Args:
        router_logits: Raw routing scores [batch, seq_len, num_geometries]
        importance_factor: Weight for load balance loss

    Returns:
        Load balance loss (scalar)
    """
    # Convert logits to probabilities
    probs = F.softmax(router_logits, dim=-1)  # [batch, seq_len, num_geometries]

    # Calculate usage of each geometry
    usage = torch.mean(probs, dim=(0, 1))  # [num_geometries]

    # Coefficient of variation squared: var(usage) / mean(usage)^2
    variance = torch.var(usage)
    mean_usage = torch.mean(usage)

    cv_squared = variance / (mean_usage ** 2 + 1e-10)

    return cv_squared * importance_factor


def router_entropy_loss(router_logits: torch.Tensor, importance_factor: float = 0.01) -> torch.Tensor:
    """
    Encourages high entropy in routing decisions.

    This prevents the router from being too confident (always routing to one geometry).

    Args:
        router_logits: Raw routing scores [batch, seq_len, num_geometries]
        importance_factor: Weight for entropy loss

    Returns:
        Negative entropy loss (scalar) - we want to maximize entropy, so minimize negative
    """
    # Convert to probabilities
    probs = F.softmax(router_logits, dim=-1) + 1e-10  # Add epsilon for log stability

    # Compute entropy: -sum(p * log(p))
    entropy = -torch.sum(probs * torch.log(probs), dim=-1)  # [batch, seq_len]

    # Average over batch and sequence
    mean_entropy = torch.mean(entropy)

    # Maximum entropy for n geometries is log(n)
    num_geometries = probs.shape[-1]
    max_entropy = torch.log(torch.tensor(float(num_geometries)))

    # We want high entropy, so minimize negative entropy
    # Normalize by max entropy so loss is in [0, 1] range
    normalized_entropy = mean_entropy / max_entropy

    # Return negative entropy (minimize negative = maximize positive)
    return -(normalized_entropy - 0.5) * importance_factor  # Center around 0.5


def combined_routing_loss(
    router_logits: torch.Tensor,
    balance_weight: float = 0.01,
    entropy_weight: float = 0.001,
) -> torch.Tensor:
    """
    Combine load balancing and entropy losses.

    Args:
        router_logits: Raw routing scores [batch, seq_len, num_geometries]
        balance_weight: Weight for load balance loss
        entropy_weight: Weight for entropy loss

    Returns:
        Combined routing loss (scalar)
    """
    balance_loss = load_balancing_loss(router_logits, balance_weight)
    entropy_loss = router_entropy_loss(router_logits, entropy_weight)

    return balance_loss + entropy_loss
