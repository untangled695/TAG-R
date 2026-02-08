#!/bin/bash
# Robust experiment launcher - runs in foreground with proper logging

cd /Volumes/Documents/Research/geometric-attention/geometric-attention
export PYTHONPATH=/Volumes/Documents/Research/geometric-attention/geometric-attention:$PYTHONPATH

echo "========================================================"
echo "FULL-SCALE WORDNET EXPERIMENT - ROBUST MODE"
echo "========================================================"
echo ""
echo "This will run for approximately 2-3 hours."
echo "Started at: $(date)"
echo ""
echo "To monitor progress:"
echo "  tail -f outputs/wordnet_full_scale/results.csv"
echo ""
echo "Press Ctrl+C to stop (NOT recommended - will lose progress)"
echo "========================================================"
echo ""

# Run with unbuffered output
python -u experiments/wordnet/run_full_scale.py 2>&1 | tee outputs/experiment_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "========================================================"
echo "Experiment completed at: $(date)"
echo "Results saved to: outputs/wordnet_full_scale/"
echo "========================================================"
