#!/bin/bash
#SBATCH --job-name=evaluate_interpretability_autoencoder
#SBATCH --output=./slurm/logs/evaluate_interpretability_autoencoder/evaluate_interpretability_%j/.log
#SBATCH --error=./slurm/logs/evaluate_interpretability_autoencoder/evaluate_interpretability_%j/.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100GB
#SBATCH --partition=studentbatch
#SBATCH --gres=gpu:1
#SBATCH --constraint="geforce_rtx_2080"


# Run the interpretability evaluation script
# Uses:
# - Layer 18 (main layer for activation extraction)
# - 50 components (primary dimensionality)
# - Autoencoder embedding (more complex but potentially more accurate)
# - Auto-select prototype centroid (picks centroid with most activations)
python -u scripts/evaluate_interpretability.py \
    --layer 18 \
    --n-components 50 \
    --embedding-type autoencoder \
    --use-llm

