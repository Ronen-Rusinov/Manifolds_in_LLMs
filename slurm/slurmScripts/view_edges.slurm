#!/bin/bash
#SBATCH --job-name=view_edges
#SBATCH --output=./slurm/logs/logs/view_edges/view_edges_%j.log
#SBATCH --error=./slurm/logs/errs/view_edges/view_edges_%j.err
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50GB
#SBATCH --partition=studentbatch

python -u $LLM_PROJ_PATH/experiments/check_neighborhoods_overlap.py
