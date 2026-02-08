"""
Router Visualization and Analysis Tools

Analyzes where the AGA router sends tokens:
- Distribution of routing decisions
- Routing patterns for different hierarchy depths
- Heatmaps showing which structures route to which geometries
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List
from torch.utils.data import DataLoader

from dataset import WordNetHierarchyDataset, collate_hierarchy_batch
from models import HierarchyPredictor


def collect_router_statistics(
    model: HierarchyPredictor,
    data_loader: DataLoader,
    device: str = 'cuda',
) -> Dict:
    """
    Collect routing statistics from model.

    Args:
        model: Trained HierarchyPredictor with adaptive attention
        data_loader: DataLoader for evaluation
        device: Device to run on

    Returns:
        Dictionary with routing statistics
    """
    assert model.attention_type == 'adaptive', "Model must use adaptive attention"

    model.eval()
    model.to(device)

    all_weights = []
    all_depths = []
    all_num_ancestors = []

    with torch.no_grad():
        for leaves, ancestors in data_loader:
            leaves = leaves.to(device)

            # Forward pass (stores router info)
            _ = model(leaves)

            # Get router statistics
            router_stats = model.get_router_statistics()

            if router_stats is not None:
                # Weights: [batch_size, 1, 3]
                weights = router_stats['weights'].squeeze(1).numpy()  # [batch_size, 3]
                all_weights.append(weights)

                # Get metadata for this batch
                for i in range(ancestors.shape[0]):
                    valid_ancestors = ancestors[i][ancestors[i] != -1]
                    all_depths.append(len(valid_ancestors))
                    all_num_ancestors.append(len(valid_ancestors))

    # Concatenate all batches
    all_weights = np.concatenate(all_weights, axis=0)  # [total_samples, 3]
    all_depths = np.array(all_depths)

    return {
        'weights': all_weights,  # [N, 3] - routing weights for each sample
        'depths': all_depths,  # [N] - hierarchy depth for each sample
        'num_ancestors': np.array(all_num_ancestors),  # [N] - number of ancestors
        'mean_weights': all_weights.mean(axis=0),  # [3] - average across all samples
        'std_weights': all_weights.std(axis=0),  # [3] - std across all samples
    }


def plot_router_distribution(
    stats: Dict,
    output_path: str = './router_distribution.png',
):
    """
    Plot distribution of routing decisions.

    Shows box plots and violin plots for each geometry.
    """
    weights = stats['weights']  # [N, 3]

    geometry_names = ['Euclidean', 'Hyperbolic', 'Spherical']

    # Create DataFrame for seaborn
    df = pd.DataFrame(weights, columns=geometry_names)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot
    sns.boxplot(data=df, ax=axes[0])
    axes[0].set_title('Router Weight Distribution', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Routing Weight', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Violin plot
    sns.violinplot(data=df, ax=axes[1])
    axes[1].set_title('Router Weight Density', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Routing Weight', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"  Router distribution saved to {output_path}")


def plot_router_by_depth(
    stats: Dict,
    output_path: str = './router_by_depth.png',
):
    """
    Plot routing patterns vs hierarchy depth.

    Shows how router decisions change based on tree depth.
    """
    weights = stats['weights']  # [N, 3]
    depths = stats['depths']  # [N]

    geometry_names = ['Euclidean', 'Hyperbolic', 'Spherical']

    # Group by depth
    unique_depths = sorted(np.unique(depths))
    mean_weights_by_depth = []

    for depth in unique_depths:
        mask = depths == depth
        mean_weights = weights[mask].mean(axis=0)
        mean_weights_by_depth.append(mean_weights)

    mean_weights_by_depth = np.array(mean_weights_by_depth)  # [num_depths, 3]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, name in enumerate(geometry_names):
        ax.plot(unique_depths, mean_weights_by_depth[:, i], marker='o', linewidth=2, label=name)

    ax.set_xlabel('Hierarchy Depth', fontsize=12)
    ax.set_ylabel('Mean Routing Weight', fontsize=12)
    ax.set_title('Router Behavior vs Hierarchy Depth', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"  Router by depth saved to {output_path}")


def plot_router_heatmap(
    stats: Dict,
    output_path: str = './router_heatmap.png',
):
    """
    Create heatmap showing routing patterns.

    Rows: Depth bins
    Columns: Geometries
    """
    weights = stats['weights']
    depths = stats['depths']

    geometry_names = ['Euclidean', 'Hyperbolic', 'Spherical']

    # Create depth bins
    depth_bins = [0, 2, 4, 6, 8, 100]
    depth_labels = ['0-2', '2-4', '4-6', '6-8', '8+']

    # Bin depths
    binned_depths = np.digitize(depths, depth_bins[:-1]) - 1

    # Compute mean weights for each bin
    heatmap_data = []
    for bin_idx in range(len(depth_labels)):
        mask = binned_depths == bin_idx
        if mask.sum() > 0:
            mean_weights = weights[mask].mean(axis=0)
        else:
            mean_weights = np.array([0.0, 0.0, 0.0])
        heatmap_data.append(mean_weights)

    heatmap_data = np.array(heatmap_data)  # [num_bins, 3]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        xticklabels=geometry_names,
        yticklabels=depth_labels,
        ax=ax,
        cbar_kws={'label': 'Mean Routing Weight'},
    )

    ax.set_xlabel('Geometry Type', fontsize=12)
    ax.set_ylabel('Hierarchy Depth', fontsize=12)
    ax.set_title('Router Heatmap: Depth vs Geometry', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"  Router heatmap saved to {output_path}")


def analyze_router_specialization(stats: Dict):
    """
    Analyze how specialized the router is.

    Measures:
    - Entropy (lower = more specialized)
    - Max weight (higher = more decisive)
    - Dominant geometry (which geometry is used most)
    """
    weights = stats['weights']  # [N, 3]

    # Compute entropy for each sample
    eps = 1e-10
    entropies = -np.sum(weights * np.log(weights + eps), axis=1)  # [N]

    # Max weight per sample
    max_weights = weights.max(axis=1)  # [N]

    # Dominant geometry (hard routing)
    dominant_geometries = weights.argmax(axis=1)  # [N]
    geometry_counts = np.bincount(dominant_geometries, minlength=3)

    geometry_names = ['Euclidean', 'Hyperbolic', 'Spherical']

    print("\n" + "="*60)
    print("ROUTER SPECIALIZATION ANALYSIS")
    print("="*60)

    print(f"\nMean Routing Weights:")
    for i, name in enumerate(geometry_names):
        print(f"  {name}: {stats['mean_weights'][i]:.4f} ± {stats['std_weights'][i]:.4f}")

    print(f"\nEntropy Statistics:")
    print(f"  Mean entropy: {entropies.mean():.4f}")
    print(f"  Max entropy (uniform): {np.log(3):.4f}")
    print(f"  Specialization: {1 - entropies.mean() / np.log(3):.2%}")

    print(f"\nMax Weight Statistics:")
    print(f"  Mean max weight: {max_weights.mean():.4f}")
    print(f"  Decisiveness: {(max_weights > 0.5).mean():.2%} of samples have dominant geometry")

    print(f"\nHard Routing Counts:")
    for i, name in enumerate(geometry_names):
        print(f"  {name}: {geometry_counts[i]} ({geometry_counts[i] / len(weights) * 100:.1f}%)")

    print("="*60 + "\n")


def visualize_all(
    model: HierarchyPredictor,
    data_loader: DataLoader,
    output_dir: str = './outputs/router_analysis',
    device: str = 'cuda',
):
    """
    Run all router visualizations.

    Args:
        model: Trained adaptive model
        data_loader: DataLoader for analysis
        output_dir: Output directory for plots
        device: Device to run on
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("ROUTER VISUALIZATION AND ANALYSIS")
    print("="*60 + "\n")

    # Collect statistics
    print("Collecting router statistics...")
    stats = collect_router_statistics(model, data_loader, device)
    print(f"  Analyzed {len(stats['weights'])} samples")

    # Analyze specialization
    analyze_router_specialization(stats)

    # Create plots
    print("\nGenerating plots...")
    plot_router_distribution(stats, os.path.join(output_dir, 'router_distribution.png'))
    plot_router_by_depth(stats, os.path.join(output_dir, 'router_by_depth.png'))
    plot_router_heatmap(stats, os.path.join(output_dir, 'router_heatmap.png'))

    print(f"\nAll visualizations saved to {output_dir}")
    print("="*60 + "\n")

    return stats


if __name__ == "__main__":
    # Example usage (requires trained model)
    print("Router visualization tools loaded.")
    print("Usage:")
    print("  from visualize_router import visualize_all")
    print("  stats = visualize_all(trained_model, data_loader)")
