#!/bin/bash
#SBATCH --job-name=KMeans
#SBATCH --output=./slurm/logs/kmeans/KMeans_%j/.log
#SBATCH --error=./slurm/logs/kmeans/KMeans_%j/.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200GB
#SBATCH --partition=studentbatch

echo "Running KMeans Clustering"

python -u $LLM_PROJ_PATH/scripts/kmeans.py \
  --max_iter 300 \
  --n_init 20 \
  
  --config $LLM_PROJ_PATH/config/profile_production.yaml
