#!/bin/bash
#SBATCH --job-name=MiniBatchKMeans
#SBATCH --output=./slurm/logs/minibatch_kmeans/MiniBatchKMeans_%j/.log
#SBATCH --error=./slurm/logs/minibatch_kmeans/MiniBatchKMeans_%j/.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=200GB
#SBATCH --partition=studentbatch

echo "Running MiniBatch KMeans Clustering"
echo "python -u $LLM_PROJ_PATH/scripts/minibatch_kmeans.py \
  --config $LLM_PROJ_PATH/config/profile_production.yaml"

python -u $LLM_PROJ_PATH/scripts/minibatch_kmeans.py \
  --config $LLM_PROJ_PATH/config/profile_production.yaml
