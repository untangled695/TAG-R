"""
Task-specific model wrappers for WordNet hierarchy prediction.

These models wrap our geometric attention mechanisms for the
hierarchy reconstruction task.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

from aga.models.aga_layer import AGALayer, AGABlock
from aga.attention.hyperbolic_attention import HyperbolicAttentionStable, EuclideanAttentionBaseline


class HierarchyPredictor(nn.Module):
    """
    Base model for hierarchy prediction task.

    Architecture:
    1. Embedding layer (maps synset indices to dense vectors)
    2. Attention layer (Euclidean/Hyperbolic/Adaptive)
    3. Prediction head (maps to all possible ancestors)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        attention_type: str = 'euclidean',
        n_heads: int = 4,
        dropout: float = 0.1,
        hyperbolic_curvature: float = -1.0,
    ):
        """
        Args:
            vocab_size: Number of synsets in vocabulary
            d_model: Model dimension
            attention_type: 'euclidean', 'hyperbolic', or 'adaptive'
            n_heads: Number of attention heads
            dropout: Dropout probability
            hyperbolic_curvature: Curvature for hyperbolic attention
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.attention_type = attention_type

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # --- FIX 1: Initialize Embeddings Near Origin ---
        # Standard init creates vectors with norm > 1. We need norm << 1 for Poincaré ball.
        nn.init.normal_(self.embedding.weight, mean=0, std=1e-3)

        # Attention layer (different types)
        if attention_type == 'euclidean':
            # Standard Transformer encoder layer
            self.attention = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
            )

        elif attention_type == 'hyperbolic':
            # Pure hyperbolic attention
            self.attention = HyperbolicAttentionStable(
                d_model=d_model,
                n_heads=n_heads,
                curvature=hyperbolic_curvature,
                dropout=dropout,
            )

        elif attention_type == 'adaptive':
            # Full AGA system
            self.attention = AGABlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_model * 4,
                dropout=dropout,
                hyperbolic_curvature=hyperbolic_curvature,
            )

        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

        # Prediction head
        self.predictor = nn.Linear(d_model, vocab_size)

        # Store routing info (for adaptive only)
        self.last_router_info = None

    def forward(self, leaf_indices: torch.Tensor) -> torch.Tensor:
        """
        Predict ancestors for given leaf synsets.

        Args:
            leaf_indices: Leaf synset indices [batch_size]

        Returns:
            logits: Prediction logits [batch_size, vocab_size]
        """
        # Embed: [batch_size] -> [batch_size, 1, d_model]
        embeddings = self.embedding(leaf_indices).unsqueeze(1)

        # Embeddings are already initialized near origin (std=1e-3), no need to normalize
        # Apply attention
        if self.attention_type == 'adaptive':
            # AGA returns (output, router_info)
            attended, router_info = self.attention(embeddings, return_router_info=True)
            self.last_router_info = router_info
        else:
            attended = self.attention(embeddings)

        # Pool: [batch_size, 1, d_model] -> [batch_size, d_model]
        pooled = attended.squeeze(1)

        # Predict: [batch_size, d_model] -> [batch_size, vocab_size]
        logits = self.predictor(pooled)

        return logits

    def get_router_statistics(self) -> Optional[Dict]:
        """Get router statistics (for adaptive models only)."""
        if self.attention_type != 'adaptive' or self.last_router_info is None:
            return None

        return {
            'mean_weights': self.last_router_info['mean_weights'].detach().cpu().numpy(),
            'weights': self.last_router_info['weights'].detach().cpu(),
            'logits': self.last_router_info['logits'].detach().cpu(),
        }


class MultiScaleHierarchyPredictor(nn.Module):
    """
    Multi-scale model with multiple attention layers.

    This can capture both shallow and deep hierarchical patterns.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int = 2,
        attention_type: str = 'adaptive',
        n_heads: int = 4,
        dropout: float = 0.1,
        hyperbolic_curvature: float = -1.0,
    ):
        """
        Args:
            vocab_size: Number of synsets
            d_model: Model dimension
            num_layers: Number of attention layers
            attention_type: Type of attention
            n_heads: Number of heads
            dropout: Dropout probability
            hyperbolic_curvature: Curvature for hyperbolic attention
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.attention_type = attention_type

        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # --- FIX 1: Initialize Embeddings Near Origin ---
        nn.init.normal_(self.embedding.weight, mean=0, std=1e-3)

        # Stack of attention layers
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            if attention_type == 'adaptive':
                layer = AGABlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_model * 4,
                    dropout=dropout,
                    hyperbolic_curvature=hyperbolic_curvature,
                )
            elif attention_type == 'hyperbolic':
                layer = HyperbolicAttentionStable(
                    d_model=d_model,
                    n_heads=n_heads,
                    curvature=hyperbolic_curvature,
                    dropout=dropout,
                )
            else:  # euclidean
                layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    batch_first=True,
                )

            self.layers.append(layer)

        # Prediction head
        self.predictor = nn.Linear(d_model, vocab_size)

        # Store routing info for all layers
        self.layer_router_info = []

    def forward(self, leaf_indices: torch.Tensor) -> torch.Tensor:
        """Forward pass through all layers."""
        # Embed
        x = self.embedding(leaf_indices).unsqueeze(1)

        # Apply layers
        self.layer_router_info = []

        for layer in self.layers:
            if self.attention_type == 'adaptive':
                x, router_info = layer(x, return_router_info=True)
                self.layer_router_info.append(router_info)
            else:
                x = layer(x)

        # Pool and predict
        pooled = x.squeeze(1)
        logits = self.predictor(pooled)

        return logits
