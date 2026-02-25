#!/bin/bash
#SBATCH --job-name=Isomap_12D
#SBATCH --output=./slurm/logs/isomap_12D/Isomap_12D_%j/.log
#SBATCH --error=./slurm/logs/isomap_12D/Isomap_12D_%j/.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200GB
#SBATCH --partition=studentbatch

python -u $LLM_PROJ_PATH/scripts/train_and_save_isomap_12D.py
    