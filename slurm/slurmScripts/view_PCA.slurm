#!/bin/bash
#SBATCH --job-name=view_PCA
#SBATCH --output=./slurm/logs/view_PCA/view_PCA_%j/.log
#SBATCH --error=./slurm/logs/view_PCA/view_PCA_%j/.err
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80GB
#SBATCH --partition=studentbatch

python -u $LLM_PROJ_PATH/experiments/PCA_interpretation.py
