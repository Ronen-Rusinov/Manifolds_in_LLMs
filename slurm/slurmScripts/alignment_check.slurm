#!/bin/bash
#SBATCH --job-name=alignment_check
#SBATCH --output=./slurm/logs/mapping_alignment/mapping_alignment_%j/.log
#SBATCH --error=./slurm/logs/mapping_alignment/mapping_alignment_%j/.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100GB
#SBATCH --partition=studentbatch

python -u $LLM_PROJ_PATH/experiments/mapping_alignment.py