"""
Full-Scale WordNet Experiment - The Stress Test

This experiment is designed to break the tie and reveal geometric advantages:
1. Higher dimensions [16, 32, 64, 128, 256] to test capacity scaling
2. Full WordNet dataset (10K samples) to saturate the models
3. Longer training (20 epochs) to ensure convergence
4. Depth-stratified evaluation to show hierarchical reasoning
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

# Now we can use local imports
if __name__ == "__main__":
    # Import the main experiment function
    from run_experiment import run_full_experiment

    print("="*70)
    print("FULL-SCALE WORDNET HIERARCHY EXPERIMENT")
    print("="*70)
    print()
    print("Goal: Stress-test geometric capacity to reveal efficiency gaps")
    print("Expected: Hyperbolic-32 matches Euclidean-256 performance")
    print()

    results, models = run_full_experiment(
        dims=[16, 32, 64, 128, 256],  # Full dimension sweep
        max_samples=10000,  # Full dataset (saturate capacity)
        num_epochs=20,  # Longer training for convergence
        output_dir='./outputs/wordnet_full_scale',
    )

    print()
    print("="*70)
    print("FULL-SCALE EXPERIMENT COMPLETE")
    print("="*70)
    print()
    print("Check outputs/wordnet_full_scale/ for:")
    print("  - results.csv: Raw performance data")
    print("  - recall_vs_dimension.png: The efficiency curve")
    print("  - comparison_table.png: Direct performance matrix")
