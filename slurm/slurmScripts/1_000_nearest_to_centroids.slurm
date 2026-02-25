#!/bin/bash
#SBATCH --job-name=1_000_nearest_to_centroids
#SBATCH --output=./slurm/logs/1_000_nearest_to_centroids/1_000_nearest_to_centroids_%j/.log
#SBATCH --error=./slurm/logs/1_000_nearest_to_centroids/1_000_nearest_to_centroids_%j/.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-2
#SBATCH --mem=80GB
#SBATCH --partition=studentbatch


python -u $LLM_PROJ_PATH/scripts/obtain_1000_nearest_to_centroids.py $SLURM_ARRAY_TASK_ID
