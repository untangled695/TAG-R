"""
Adaptive Geometric Attention (AGA) Layer

This module combines multiple geometric attention mechanisms (Euclidean, Hyperbolic, Spherical)
with dynamic routing based on input structure.

The key innovation: Let the data decide which geometry to use, rather than
imposing a single geometric prior on all data.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from aga.attention.hyperbolic_attention import HyperbolicAttentionStable, EuclideanAttentionBaseline
from aga.attention.spherical_attention import SphericalAttentionStable
from aga.routing.curvature_router import CurvatureAwareRouter


class AGALayer(nn.Module):
    """
    Adaptive Geometric Attention Layer.

    Routes each token to appropriate geometric attention mechanism:
    - Euclidean: Flat, standard structure
    - Hyperbolic: Hierarchical, tree-like structure
    - Spherical: Cyclic, periodic structure

    Uses soft routing (mixing) rather than hard routing (selecting)
    to support mixed-curvature representations.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        hyperbolic_curvature: float = -1.0,
        router_temperature: float = 1.0,
        use_residual: bool = True,
        use_layernorm: bool = True,
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
            hyperbolic_curvature: Curvature for hyperbolic attention (negative)
            router_temperature: Temperature for routing softmax (lower = sharper)
            use_residual: Whether to use residual connections
            use_layernorm: Whether to use layer normalization
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.use_residual = use_residual

        # 1. Curvature-Aware Router
        self.router = CurvatureAwareRouter(
            d_model=d_model,
            num_geometries=3,
            temperature=router_temperature,
        )

        # 2. Geometric Attention Experts
        self.euclidean_attn = EuclideanAttentionBaseline(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        self.hyperbolic_attn = HyperbolicAttentionStable(
            d_model=d_model,
            n_heads=n_heads,
            curvature=hyperbolic_curvature,
            dropout=dropout,
        )

        self.spherical_attn = SphericalAttentionStable(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        # 3. Output projection and normalization
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        if use_layernorm:
            self.norm = nn.LayerNorm(d_model)
        else:
            self.norm = None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_router_info: bool = False,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Forward pass with adaptive geometric attention.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask [batch, 1, seq_len, seq_len]
            return_router_info: Whether to return routing diagnostics

        Returns:
            output: Attention output [batch, seq_len, d_model]
            router_info: Optional dict with routing information
        """
        residual = x

        # --- Step 1: Determine Geometry Distribution ---
        # weights: [batch, seq_len, 3] where index 0=Euc, 1=Hyp, 2=Sph
        router_weights, router_logits = self.router(x)

        # --- Step 2: Execute All Geometric Experts in Parallel ---
        # Note: For efficiency, could implement top-k sparse execution
        # For now, compute all and mix (Mixture of Experts with soft routing)

        out_euclidean = self.euclidean_attn(x, mask)
        out_hyperbolic = self.hyperbolic_attn(x, mask)
        out_spherical = self.spherical_attn(x, mask)

        # --- Step 3: Weighted Aggregation ---
        # Expand weights for broadcasting: [batch, seq_len, 3] -> [batch, seq_len, 3, 1]
        # Then select each geometry's weight
        w_euc = router_weights[..., 0].unsqueeze(-1)  # [batch, seq_len, 1]
        w_hyp = router_weights[..., 1].unsqueeze(-1)  # [batch, seq_len, 1]
        w_sph = router_weights[..., 2].unsqueeze(-1)  # [batch, seq_len, 1]

        # Mix outputs: Combine Euclidean + Hierarchical + Cyclic structure
        combined_out = (
            w_euc * out_euclidean +
            w_hyp * out_hyperbolic +
            w_sph * out_spherical
        )

        # --- Step 4: Output Projection ---
        output = self.output_proj(combined_out)
        output = self.dropout(output)

        # --- Step 5: Residual Connection & Normalization ---
        if self.use_residual:
            output = output + residual

        if self.norm is not None:
            output = self.norm(output)

        # Prepare router info for diagnostics/loss
        router_info = None
        if return_router_info:
            router_info = {
                'weights': router_weights,          # [batch, seq_len, 3]
                'logits': router_logits,            # [batch, seq_len, 3]
                'mean_weights': router_weights.mean(dim=(0, 1)),  # [3] - avg usage per geometry
            }

        return output, router_info

    def get_geometry_usage_stats(self, x: torch.Tensor) -> dict:
        """
        Get statistics on which geometries the router selects.

        Useful for debugging router behavior.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            dict with geometry usage statistics
        """
        with torch.no_grad():
            weights, logits = self.router(x)

            # Mean usage across batch and sequence
            mean_weights = weights.mean(dim=(0, 1))  # [3]

            # Entropy of routing distribution
            probs = weights + 1e-10
            entropy = -torch.sum(probs * torch.log(probs), dim=-1)
            mean_entropy = entropy.mean().item()

            # Max geometry per token (hard routing decision)
            max_geometry = torch.argmax(weights, dim=-1)  # [batch, seq_len]

            # Count of tokens assigned to each geometry (hard routing)
            geometry_counts = torch.bincount(
                max_geometry.flatten(),
                minlength=3
            ).float()

            return {
                'mean_weights': mean_weights.cpu().numpy(),
                'mean_entropy': mean_entropy,
                'hard_routing_counts': geometry_counts.cpu().numpy(),
                'geometry_names': ['Euclidean', 'Hyperbolic', 'Spherical'],
            }


class AGABlock(nn.Module):
    """
    Full Transformer block with AGA attention and feed-forward.

    This is the building block for AGA-based models.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int = None,
        dropout: float = 0.1,
        hyperbolic_curvature: float = -1.0,
        router_temperature: float = 1.0,
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension (default: 4 * d_model)
            dropout: Dropout probability
            hyperbolic_curvature: Curvature for hyperbolic attention
            router_temperature: Temperature for routing
        """
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_model

        # 1. Adaptive Geometric Attention
        self.attention = AGALayer(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            hyperbolic_curvature=hyperbolic_curvature,
            router_temperature=router_temperature,
            use_residual=True,
            use_layernorm=True,
        )

        # 2. Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_router_info: bool = False,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Forward pass through AGA block.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask
            return_router_info: Whether to return routing diagnostics

        Returns:
            output: Block output [batch, seq_len, d_model]
            router_info: Optional routing information
        """
        # Attention sub-layer (includes residual + norm)
        attn_out, router_info = self.attention(x, mask, return_router_info)

        # Feed-forward sub-layer with residual + norm
        ffn_out = self.ffn(attn_out)
        output = self.norm_ffn(attn_out + ffn_out)

        return output, router_info
