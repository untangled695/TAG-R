"""
Hyperbolic Attention with Stable Tangent Space Aggregation

This module implements attention that respects hyperbolic geometry:
- Attention scores based on hyperbolic distance (not dot product)
- Values aggregated via tangent space operations
- All operations maintain numerical stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from aga.utils.safe_math import safe_atanh, safe_tanh


class HyperbolicAttentionStable(nn.Module):
    """
    Geometry-Native Attention with Tangent Space Aggregation.

    Key Mechanism:
    - Attention Scores = -distance_hyperbolic(q, k)
    - Aggregation = ExpMap(WeightedAvg(LogMap(v)))

    This differs from standard attention in fundamental ways:
    1. Distance-based scores capture hierarchical relationships
    2. Tangent space aggregation preserves geometric properties
    3. Points near boundary (deep in hierarchy) have exponentially large distances
    """

    def __init__(self, d_model: int, n_heads: int, curvature: float = -1.0, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            curvature: Negative curvature (negative -> hyperbolic, -1.0 is standard)
            dropout: Dropout probability
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # We use c = -curvature (if curvature is -1.0, c=1.0)
        self.c = abs(curvature)
        self.scale = self.d_head ** 0.5

        # Projections operate in Tangent Space (effectively Euclidean)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, H: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with hyperbolic geometry.

        Args:
            H: Input tensor [batch, seq, d_model]
            mask: Optional attention mask [batch, 1, seq, seq]

        Returns:
            Output tensor [batch, seq, d_model]
        """
        batch, seq, _ = H.shape

        # 1. Linear Projections (Tangent Space)
        # We assume H comes in already "safe" or is mapped to tangent space first.
        # Ideally, H is the output of a previous layer's LogMap.
        # For this layer, we treat input H as living in tangent space of origin.
        Q = self.W_q(H).view(batch, seq, self.n_heads, self.d_head)
        K = self.W_k(H).view(batch, seq, self.n_heads, self.d_head)
        V = self.W_v(H).view(batch, seq, self.n_heads, self.d_head)

        # Transpose for attention: [batch, heads, seq, d_head]
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # 2. Map Q and K to Manifold for Scoring
        # We need the ACTUAL distance on the ball to measure "hierarchical" similarity.
        Q_hyp = self.exp_map_zero(Q)
        K_hyp = self.exp_map_zero(K)

        # 3. Compute Attention Scores via Hyperbolic Distance
        # dist matrix: [batch, heads, seq_q, seq_k]
        hyp_dist = self.pairwise_poincare_distance(Q_hyp, K_hyp)

        # Score is negative distance (closer = higher attention)
        scores = -hyp_dist / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 4. Tangent Space Aggregation (The Stability Key)
        # We map V to manifold, then back to tangent for averaging.
        # This ensures geometric correctness.
        V_hyp = self.exp_map_zero(V)  # Values live on manifold
        V_tan = self.log_map_zero(V_hyp)  # Map to tangent for averaging

        # Weighted sum in tangent space
        # attn_weights: [batch, heads, seq_q, seq_k]
        # V_tan:        [batch, heads, seq_k, d_head]
        out_tan = torch.matmul(attn_weights, V_tan)  # [batch, heads, seq_q, d_head]

        # Map aggregated result back to manifold
        out_hyp = self.exp_map_zero(out_tan)

        # 5. Output Projection
        # To feed into next layer (residual stream), we need to be in tangent space again.
        out_final_tan = self.log_map_zero(out_hyp)

        # Reshape and project
        out_final_tan = out_final_tan.permute(0, 2, 1, 3).contiguous().view(batch, seq, self.d_model)
        output = self.W_o(out_final_tan)

        return output

    # --- Geometric Primitives (Verified safe versions) ---

    def exp_map_zero(self, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map at origin: Tangent space → Poincaré ball

        Simplified formula when base point is origin:
        exp_0(v) = tanh(√c ||v||) * v / (√c ||v||)

        Args:
            v: Tangent vectors [*, d]

        Returns:
            Points on Poincaré ball [*, d]
        """
        # --- FIX 9: Safe Norm (eps inside sqrt prevents gradient explosion at zero) ---
        norm = torch.sqrt(torch.sum(v * v, dim=-1, keepdim=True) + 1e-8)
        sqrt_c = math.sqrt(self.c)

        # --- FIX 3: Tighter Clamping + Safe Gradients ---
        # Use safe_tanh with bounded gradients to prevent NaN in backprop
        target_norm = safe_tanh(sqrt_c * norm).clamp(max=0.9) / sqrt_c
        result = v * (target_norm / norm)

        # Safety check
        result_norm = torch.sqrt(torch.sum(result * result, dim=-1, keepdim=True) + 1e-8)
        max_norm = (1.0 / sqrt_c) - 1e-5
        result = torch.where(
            result_norm > max_norm,
            result * (max_norm / result_norm),
            result
        )

        return result

    def log_map_zero(self, y: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map at origin: Poincaré ball → Tangent space

        Simplified formula when base point is origin:
        log_0(y) = atanh(√c ||y||) * y / (√c ||y||)

        Args:
            y: Points on Poincaré ball [*, d]

        Returns:
            Tangent vectors [*, d]
        """
        sqrt_c = math.sqrt(self.c)
        # --- FIX 9: Safe Norm (eps inside sqrt) ---
        norm = torch.sqrt(torch.sum(y * y, dim=-1, keepdim=True) + 1e-8)
        norm = norm.clamp(max=1.0/sqrt_c - 1e-5)  # Keep max clamp for boundary safety

        # Use safe_atanh with bounded gradients to prevent NaN in backprop
        atanh_arg = (sqrt_c * norm).clamp(min=-0.99, max=0.99)
        target_norm = safe_atanh(atanh_arg) / sqrt_c
        result = y * (target_norm / norm)

        return result

    def pairwise_poincare_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise hyperbolic distances between sequences.

        The hyperbolic distance captures hierarchical relationships:
        - Points near boundary are exponentially far apart
        - Points near origin behave like Euclidean space

        Args:
            x: Points [batch, heads, seq_x, d]
            y: Points [batch, heads, seq_y, d]

        Returns:
            Distances [batch, heads, seq_x, seq_y]
        """
        # Expand dimensions for broadcasting
        # x: [batch, heads, seq_x, 1, d]
        # y: [batch, heads, 1, seq_y, d]
        x_expanded = x.unsqueeze(-2)
        y_expanded = y.unsqueeze(-3)

        # Compute Möbius addition: (-x) ⊕ y
        # This gives the "difference" in hyperbolic space
        x2 = torch.sum(x_expanded * x_expanded, dim=-1, keepdim=True)
        y2 = torch.sum(y_expanded * y_expanded, dim=-1, keepdim=True)
        xy = torch.sum(x_expanded * y_expanded, dim=-1, keepdim=True)

        c = self.c

        # Möbius addition formula: (a ⊕ b) = (num / denom)
        num = (1 + 2*c*xy + c*y2) * (-x_expanded) + (1 - c*x2) * y_expanded
        denom = 1 + 2*c*(-xy) + c**2 * x2 * y2
        denom = denom.clamp(min=1e-10)

        mob_add = num / denom

        # --- FIX 4: Safety Clamp + Safe Gradients ---
        # Use safe_atanh with bounded gradients to prevent NaN in backprop
        # --- FIX 9: Safe Norm (eps inside sqrt) ---
        norm_mob = torch.sqrt(torch.sum(mob_add * mob_add, dim=-1) + 1e-8)
        norm_mob = norm_mob.clamp(max=0.95)  # Keep max clamp for boundary safety
        sqrt_c = math.sqrt(c)

        atanh_arg = (sqrt_c * norm_mob).clamp(min=-0.99, max=0.99)
        dist = (2.0 / sqrt_c) * safe_atanh(atanh_arg)

        return dist


class EuclideanAttentionBaseline(nn.Module):
    """
    Standard scaled dot-product attention for comparison.

    This serves as a baseline to verify that hyperbolic geometry
    actually provides different (and better) behavior.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.scale = self.d_head ** 0.5

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, H: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Standard attention forward pass."""
        batch, seq, _ = H.shape

        # Linear projections
        Q = self.W_q(H).view(batch, seq, self.n_heads, self.d_head)
        K = self.W_k(H).view(batch, seq, self.n_heads, self.d_head)
        V = self.W_v(H).view(batch, seq, self.n_heads, self.d_head)

        # Transpose: [batch, heads, seq, d_head]
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, V)

        # Reshape and project
        out = out.permute(0, 2, 1, 3).contiguous().view(batch, seq, self.d_model)
        output = self.W_o(out)

        return output
