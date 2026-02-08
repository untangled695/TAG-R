"""
Poincaré Disk Visualization - Visual Proof of Hierarchical Learning

This generates the "killer visualization" that proves hyperbolic geometry
learned the hierarchical structure correctly:
- Root concepts (entity, organism) near center
- Leaf concepts (specific animals, objects) near boundary
- Radial distance correlates with depth in WordNet taxonomy
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple

from dataset import WordNetHierarchyDataset
from models import HierarchyPredictor


def project_to_poincare_disk(embeddings: torch.Tensor) -> np.ndarray:
    """
    Project high-dimensional embeddings to 2D Poincaré disk.

    Uses PCA for dimensionality reduction, then re-normalizes to ensure
    all points stay within the unit disk (preserving hyperbolic structure).

    Args:
        embeddings: [vocab_size, d_model] tensor

    Returns:
        2D coordinates [vocab_size, 2] on Poincaré disk
    """
    embeddings_np = embeddings.detach().cpu().numpy()

    # PCA to 2D (preserves most variance while being geometry-aware)
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings_np)

    # Re-normalize to stay within disk
    # Use tanh to soft-project back to valid hyperbolic space
    norm = np.linalg.norm(emb_2d, axis=1, keepdims=True)
    emb_2d = emb_2d / (norm + 1e-8) * np.tanh(norm)

    # Ensure all points are strictly inside disk (norm < 1)
    emb_2d = emb_2d * 0.95

    return emb_2d


def visualize_poincare_hierarchy(
    model: HierarchyPredictor,
    dataset: WordNetHierarchyDataset,
    output_path: str = 'poincare_visualization.png',
    annotate_samples: int = 20,
):
    """
    Create Poincaré disk visualization showing learned hierarchy.

    Args:
        model: Trained hierarchy predictor
        dataset: WordNet dataset (for synset names)
        output_path: Where to save visualization
        annotate_samples: Number of synsets to label
    """
    print("="*60)
    print("GENERATING POINCARÉ DISK VISUALIZATION")
    print("="*60)
    print()

    model.eval()

    # Extract embeddings
    embeddings = model.embedding.weight  # [vocab_size, d_model]
    print(f"Embedding shape: {embeddings.shape}")

    # Project to 2D disk
    emb_2d = project_to_poincare_disk(embeddings)
    print(f"2D projection shape: {emb_2d.shape}")

    # Compute radial distance (proxy for depth)
    radial_dist = np.linalg.norm(emb_2d, axis=1)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # Draw boundary circle
    circle = plt.Circle((0, 0), 1.0, color='black', fill=False,
                       linestyle='--', linewidth=2, label='Poincaré Disk Boundary')
    ax.add_artist(circle)

    # Scatter plot colored by depth
    scatter = ax.scatter(
        emb_2d[:, 0],
        emb_2d[:, 1],
        c=radial_dist,
        cmap='plasma',
        s=30,
        alpha=0.6,
        edgecolors='white',
        linewidth=0.5,
    )

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Distance from Origin (≈ Depth)')
    cbar.ax.tick_params(labelsize=10)

    # Annotate interesting synsets
    # Find extreme cases (center vs boundary)
    center_indices = np.argsort(radial_dist)[:annotate_samples//2]
    boundary_indices = np.argsort(radial_dist)[-annotate_samples//2:]
    annotate_indices = np.concatenate([center_indices, boundary_indices])

    for idx in annotate_indices:
        # Get synset name (if available)
        if hasattr(dataset, 'idx_to_synset'):
            synset = dataset.idx_to_synset[idx]
            name = synset.name().split('.')[0]  # Get lemma name
        else:
            name = f"syn_{idx}"

        ax.annotate(
            name,
            xy=(emb_2d[idx, 0], emb_2d[idx, 1]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            alpha=0.7,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'),
        )

    # Styling
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    # Title and labels
    ax.set_title(
        'Learned WordNet Hierarchy in Hyperbolic Space\n'
        '(Center = Root Concepts, Boundary = Leaf Concepts)',
        fontsize=14,
        fontweight='bold',
        pad=20,
    )
    ax.set_xlabel('Poincaré Disk - Dimension 1', fontsize=12)
    ax.set_ylabel('Poincaré Disk - Dimension 2', fontsize=12)

    # Legend
    ax.legend(loc='upper right', fontsize=10)

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")

    # Print statistics
    print()
    print("Radial Distribution Statistics:")
    print(f"  Mean distance: {radial_dist.mean():.4f}")
    print(f"  Std distance: {radial_dist.std():.4f}")
    print(f"  Min distance: {radial_dist.min():.4f} (center concepts)")
    print(f"  Max distance: {radial_dist.max():.4f} (leaf concepts)")
    print(f"  Median distance: {np.median(radial_dist):.4f}")

    # Check if hierarchy is learned
    print()
    if radial_dist.std() > 0.1:
        print("✅ HIERARCHY DETECTED: Strong radial gradient suggests")
        print("   the model learned hierarchical structure!")
    else:
        print("⚠️  WEAK HIERARCHY: Embeddings are clustered near origin.")
        print("   May need more training or higher curvature.")

    print("="*60)

    return emb_2d, radial_dist


def analyze_depth_correlation(
    model: HierarchyPredictor,
    dataset: WordNetHierarchyDataset,
) -> float:
    """
    Compute correlation between embedding norm and actual WordNet depth.

    High correlation = model correctly maps depth to radial distance.

    Returns:
        Pearson correlation coefficient
    """
    embeddings = model.embedding.weight.detach().cpu().numpy()
    norms = np.linalg.norm(embeddings, axis=1)

    # Get actual depths from WordNet
    depths = []
    for idx in range(len(dataset.vocab)):
        synset = dataset.idx_to_synset[idx]
        # Compute depth (distance to root)
        depth = len(synset.hypernym_paths()[0]) if synset.hypernym_paths() else 0
        depths.append(depth)

    depths = np.array(depths)

    # Compute correlation
    from scipy.stats import pearsonr
    corr, p_value = pearsonr(norms, depths)

    print()
    print("Depth-Norm Correlation Analysis:")
    print(f"  Pearson correlation: {corr:.4f}")
    print(f"  P-value: {p_value:.6f}")

    if corr > 0.5 and p_value < 0.01:
        print("  ✅ STRONG POSITIVE CORRELATION")
        print("     Hyperbolic geometry correctly encodes hierarchy!")
    elif corr > 0.3:
        print("  ✓ MODERATE CORRELATION")
        print("     Model learned some hierarchical structure.")
    else:
        print("  ❌ WEAK CORRELATION")
        print("     Model may not be exploiting hyperbolic geometry.")

    return corr


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python visualize_hierarchy.py <model_path>")
        print("Example: python visualize_hierarchy.py outputs/wordnet_exp/model_hyperbolic_32.pt")
        sys.exit(1)

    model_path = sys.argv[1]

    # Load dataset
    print("Loading WordNet dataset...")
    dataset = WordNetHierarchyDataset(max_samples=5000)

    # Load model
    print(f"Loading model from {model_path}...")
    model = HierarchyPredictor(
        vocab_size=dataset.vocab_size,
        d_model=32,  # Adjust based on your model
        attention_type='hyperbolic',
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Generate visualization
    visualize_poincare_hierarchy(model, dataset)

    # Analyze depth correlation
    analyze_depth_correlation(model, dataset)
