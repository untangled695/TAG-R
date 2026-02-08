"""
Master Experiment Runner for WordNet Hierarchy Reconstruction

This script validates the core hypothesis:
Hyperbolic-32 achieves what Euclidean-256 needs for hierarchical reasoning.

Runs:
1. Euclidean baseline across dimensions [16, 32, 64, 128, 256]
2. Hyperbolic attention across same dimensions
3. Adaptive (AGA) across same dimensions

Generates:
- Efficiency plots (Recall@K vs Dimension)
- Router distribution analysis
- Performance comparison table
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import os
import json
from typing import Dict, List, Tuple

from dataset import WordNetHierarchyDataset, collate_hierarchy_batch, create_wordnet_splits
from models import HierarchyPredictor
from aga.routing.curvature_router import load_balancing_loss, combined_routing_loss


class HierarchyTrainer:
    """Trainer for hierarchy prediction models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        load_balance_weight: float = 0.01,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # --- FIX 10: Split Optimization (Hyperbolic vs Euclidean parameters) ---
        # Group A: Manifold parameters (embeddings) - Use SGD + Projection
        # Group B: Euclidean parameters (attention, predictor) - Use AdamW + NO projection
        hyperbolic_params = []
        euclidean_params = []

        for name, param in model.named_parameters():
            if 'embedding' in name:
                # Embeddings live on Poincaré ball - need SGD + projection
                hyperbolic_params.append(param)
            else:
                # Everything else (attention matrices, predictor) - standard Euclidean
                euclidean_params.append(param)

        # Create dual optimizers
        self.opt_hyperbolic = optim.SGD(hyperbolic_params, lr=1e-2)
        self.opt_euclidean = optim.AdamW(euclidean_params, lr=1e-3, weight_decay=weight_decay)
        self.hyperbolic_params = hyperbolic_params

        print(f"  Split Optimization: {len(hyperbolic_params)} hyperbolic params (SGD), "
              f"{len(euclidean_params)} euclidean params (AdamW)")

        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.load_balance_weight = load_balance_weight

        self.train_losses = []
        self.val_metrics = []

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for leaves, ancestors in tqdm(self.train_loader, desc="Training", leave=False):
            leaves = leaves.to(self.device)
            ancestors = ancestors.to(self.device)

            # Zero both optimizers
            self.opt_hyperbolic.zero_grad()
            self.opt_euclidean.zero_grad()

            # Forward pass
            logits = self.model(leaves)  # [batch_size, vocab_size]

            # --- DEBUG: Check for NaN ---
            if torch.isnan(logits).any():
                print(f"WARNING: NaN detected in logits! Batch {num_batches}")
                print(f"  Leaves: {leaves[:5]}")
                print(f"  Logits stats: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")
                # Continue anyway to see if loss computation helps identify issue

            # Multi-label BCE loss
            # Create target tensor: 1 for ancestors, 0 for non-ancestors
            targets = torch.zeros_like(logits)

            for i in range(ancestors.shape[0]):
                valid_ancestors = ancestors[i][ancestors[i] != -1]
                targets[i, valid_ancestors] = 1.0

            # Compute loss
            loss = self.criterion(logits, targets).mean()

            # Add load balancing loss for adaptive models
            if hasattr(self.model, 'attention_type') and self.model.attention_type == 'adaptive':
                router_info = self.model.last_router_info
                if router_info is not None:
                    lb_loss = load_balancing_loss(
                        router_info['logits'],
                        importance_factor=self.load_balance_weight
                    )
                    loss = loss + lb_loss

            loss.backward()

            # --- FIX 6: Aggressive Gradient Clipping ---
            # This prevents exploding gradients that push embeddings to the boundary
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Step both optimizers
            self.opt_hyperbolic.step()
            self.opt_euclidean.step()

            # --- FIX 10: SELECTIVE Projection (ONLY hyperbolic parameters) ---
            # Only project embeddings to Poincaré ball - leave Euclidean weights free!
            if hasattr(self.model, 'attention_type') and self.model.attention_type in ['hyperbolic', 'adaptive']:
                with torch.no_grad():
                    for param in self.hyperbolic_params:
                        # Project embedding vectors to stay within Poincaré ball
                        # Compute norm along embedding dimension
                        if param.dim() == 2:  # Embedding matrix [vocab_size, d_model]
                            norm = torch.sqrt(torch.sum(param ** 2, dim=-1, keepdim=True) + 1e-10)
                            # Clamp norm to max 0.95 (stay away from boundary)
                            scaling = torch.where(norm > 0.95, 0.95 / norm, torch.ones_like(norm))
                            param.data.mul_(scaling)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)

        return avg_loss

    def evaluate(self, k: int = 10) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()

        total_recall = 0
        total_precision = 0
        total_f1 = 0
        num_samples = 0

        with torch.no_grad():
            for leaves, ancestors in self.val_loader:
                leaves = leaves.to(self.device)
                ancestors = ancestors.to(self.device)

                logits = self.model(leaves)

                # Get top-k predictions
                _, top_k_indices = torch.topk(logits, k=k, dim=-1)

                # Compute metrics
                for i in range(leaves.shape[0]):
                    valid_ancestors = ancestors[i][ancestors[i] != -1]

                    if len(valid_ancestors) == 0:
                        continue

                    # Convert to sets for intersection
                    predicted = set(top_k_indices[i].cpu().numpy())
                    ground_truth = set(valid_ancestors.cpu().numpy())

                    # Compute metrics
                    intersection = predicted & ground_truth
                    recall = len(intersection) / len(ground_truth) if len(ground_truth) > 0 else 0
                    precision = len(intersection) / k if k > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                    total_recall += recall
                    total_precision += precision
                    total_f1 += f1
                    num_samples += 1

        metrics = {
            f'recall@{k}': total_recall / max(num_samples, 1),
            f'precision@{k}': total_precision / max(num_samples, 1),
            f'f1@{k}': total_f1 / max(num_samples, 1),
        }

        self.val_metrics.append(metrics)

        return metrics

    def train(self, num_epochs: int, eval_every: int = 1) -> Dict:
        """Train for multiple epochs."""
        print(f"\nTraining for {num_epochs} epochs...")

        best_recall = 0
        best_epoch = 0

        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch()

            # Evaluate
            if (epoch + 1) % eval_every == 0:
                metrics = self.evaluate(k=10)

                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"Loss={train_loss:.4f}, "
                      f"Recall@10={metrics['recall@10']:.4f}, "
                      f"F1@10={metrics['f1@10']:.4f}")

                # Track best model
                if metrics['recall@10'] > best_recall:
                    best_recall = metrics['recall@10']
                    best_epoch = epoch + 1

        print(f"Best Recall@10: {best_recall:.4f} at epoch {best_epoch}")

        return {
            'best_recall': best_recall,
            'best_epoch': best_epoch,
            'final_loss': self.train_losses[-1],
        }


def run_single_experiment(
    attention_type: str,
    d_model: int,
    dataset: WordNetHierarchyDataset,
    train_split,
    val_split,
    device: str,
    num_epochs: int = 10,
    batch_size: int = 32,
    output_dir: str = './outputs/wordnet_exp',
) -> Dict:
    """Run a single experiment configuration."""

    print(f"\n{'='*60}")
    print(f"Running: {attention_type.upper()} with d_model={d_model}")
    print(f"{'='*60}")

    # Create data loaders
    train_loader = DataLoader(
        train_split,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_hierarchy_batch,
    )

    val_loader = DataLoader(
        val_split,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_hierarchy_batch,
    )

    # Create model
    model = HierarchyPredictor(
        vocab_size=dataset.vocab_size,
        d_model=d_model,
        attention_type=attention_type,
        n_heads=4,
        dropout=0.1,
        hyperbolic_curvature=-1.0,
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Train
    trainer = HierarchyTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=1e-4,  # --- FIX 5: Lower LR (1e-3 too aggressive for hyperbolic)
        weight_decay=1e-4,
        load_balance_weight=0.01,
    )

    results = trainer.train(num_epochs=num_epochs, eval_every=1)

    # Final evaluation
    final_metrics = trainer.evaluate(k=10)

    # Combine results
    results.update(final_metrics)
    results['num_params'] = num_params
    results['d_model'] = d_model
    results['attention_type'] = attention_type

    # Save model checkpoint for later visualization
    model_filename = f'model_{attention_type}_d{d_model}.pt'
    model_path = os.path.join(output_dir, model_filename)
    torch.save(model.state_dict(), model_path)
    print(f"  Model saved to: {model_path}")

    return results, model


def run_full_experiment(
    dims: List[int] = [16, 32, 64, 128],
    max_samples: int = 5000,
    num_epochs: int = 10,
    output_dir: str = './outputs',
    device: str = None,
):
    """Run complete experiment across all dimensions and attention types."""

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n{'='*60}")
    print(f"WORDNET HIERARCHY RECONSTRUCTION EXPERIMENT")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Dimensions to test: {dims}")
    print(f"Epochs per config: {num_epochs}")
    print(f"Max samples: {max_samples}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    print("\nLoading WordNet dataset...")
    dataset = WordNetHierarchyDataset(
        max_depth=8,
        min_depth=2,
        max_samples=max_samples,
        pos='n',
    )

    stats = dataset.get_statistics()
    print(f"\nDataset statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # Create splits
    train_split, val_split, test_split = create_wordnet_splits(
        dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_split)}")
    print(f"  Val: {len(val_split)}")
    print(f"  Test: {len(test_split)}")

    # Run experiments
    all_results = []
    models_by_config = {}

    attention_types = ['euclidean', 'hyperbolic', 'adaptive']

    for attention_type in attention_types:
        for d_model in dims:
            config_key = f"{attention_type}_{d_model}"

            results, model = run_single_experiment(
                attention_type=attention_type,
                d_model=d_model,
                dataset=dataset,
                train_split=train_split,
                val_split=val_split,
                device=device,
                num_epochs=num_epochs,
                batch_size=32,
                output_dir=output_dir,
            )

            all_results.append(results)
            models_by_config[config_key] = model

            # Save results incrementally
            df = pd.DataFrame(all_results)
            df.to_csv(os.path.join(output_dir, 'results.csv'), index=False)

    # Create visualizations
    print("\nGenerating plots...")
    create_plots(all_results, output_dir)

    # Save final results
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}\n")

    return all_results, models_by_config


def create_plots(results: List[Dict], output_dir: str):
    """Create visualization plots."""

    df = pd.DataFrame(results)

    # Plot 1: Recall@10 vs Dimension
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x='d_model',
        y='recall@10',
        hue='attention_type',
        marker='o',
        linewidth=2.5,
        markersize=8,
    )
    plt.title('WordNet Hierarchy Reconstruction: Efficiency at Scale', fontsize=14, fontweight='bold')
    plt.xlabel('Embedding Dimension', fontsize=12)
    plt.ylabel('Recall@10', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Attention Type', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'recall_vs_dimension.png'), dpi=300)
    plt.close()

    # Plot 2: Parameters vs Performance
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x='num_params',
        y='recall@10',
        hue='attention_type',
        size='d_model',
        sizes=(100, 400),
        alpha=0.7,
    )
    plt.title('Parameter Efficiency', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Parameters', fontsize=12)
    plt.ylabel('Recall@10', fontsize=12)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Configuration', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'params_vs_recall.png'), dpi=300)
    plt.close()

    # Plot 3: Comparison table
    pivot = df.pivot_table(
        values='recall@10',
        index='d_model',
        columns='attention_type',
        aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(
        cellText=np.round(pivot.values, 3),
        rowLabels=pivot.index,
        colLabels=pivot.columns,
        cellLoc='center',
        loc='center',
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    plt.title('Recall@10 by Dimension and Attention Type', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(os.path.join(output_dir, 'comparison_table.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Plots saved to {output_dir}")


if __name__ == "__main__":
    # Run quick test
    results, models = run_full_experiment(
        dims=[16, 32, 64],  # Add 128, 256 for full run
        max_samples=3000,  # Increase to 10000 for full run
        num_epochs=5,  # Increase to 20 for full run
        output_dir='./outputs/wordnet_exp',
    )

    # Print summary
    df = pd.DataFrame(results)
    print("\nFinal Results Summary:")
    print(df[['attention_type', 'd_model', 'recall@10', 'num_params']].to_string(index=False))
