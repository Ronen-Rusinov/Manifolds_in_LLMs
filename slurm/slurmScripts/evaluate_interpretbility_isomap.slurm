#!/bin/bash
#SBATCH --job-name=evaluate_interpretability_isomap
#SBATCH --output=./slurm/logs/evaluate_interpretability_isomap/evaluate_interpretability_%j/.log
#SBATCH --error=./slurm/logs/evaluate_interpretability_isomap/evaluate_interpretability_%j/.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100GB
#SBATCH --partition=studentbatch
    

# Run the interpretability evaluation script
# Uses:
# - Layer 18 (main layer for activation extraction)
# - 50 components (primary dimensionality)
# - Isomap embedding (non-linear dimensionality reduction)
# - Auto-select prototype centroid (picks centroid with most activations)
python -u scripts/evaluate_interpretability.py \
    --layer 18 \
    --n-components 50 \
    --embedding-type isomap \
    --use-llm

