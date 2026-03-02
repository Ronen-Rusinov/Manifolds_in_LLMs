#!/bin/bash
#SBATCH --job-name=sort_embeddings_by_components
#SBATCH --output=./slurm/logs/sort_embeddings_by_components/sort_embeddings_by_components_%j/.log
#SBATCH --error=./slurm/logs/sort_embeddings_by_components/sort_embeddings_by_components_%j/.err
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100GB
#SBATCH --partition=studentbatch

python -u $LLM_PROJ_PATH/scripts/preprocess_embeddings.py \
  --config $LLM_PROJ_PATH/config/profile_production.yaml
