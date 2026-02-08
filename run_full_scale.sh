#!/bin/bash
# Full-Scale Experiment Launcher with proper environment setup

cd /Volumes/Documents/Research/geometric-attention/geometric-attention
export PYTHONPATH=/Volumes/Documents/Research/geometric-attention/geometric-attention:$PYTHONPATH

echo "Starting full-scale experiment..."
echo "PYTHONPATH=$PYTHONPATH"
echo "Working directory: $(pwd)"
echo ""

python experiments/wordnet/run_full_scale.py 2>&1 | tee outputs/full_scale.log

echo ""
echo "Experiment complete! Check outputs/full_scale.log for details."
