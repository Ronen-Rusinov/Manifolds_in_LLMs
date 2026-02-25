#!/bin/bash
#SBATCH --job-name=K_nearest_to_centroids
#SBATCH --output=./slurm/logs/K_nearest_to_centroids/K_nearest_to_centroids_%j/.log
#SBATCH --error=./slurm/logs/K_nearest_to_centroids/K_nearest_to_centroids_%j/.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80GB
#SBATCH --partition=studentbatch


python -u $LLM_PROJ_PATH/scripts/obtain_K_nearest_to_centroids.py \
  --config $LLM_PROJ_PATH/config/profile_production.yaml
