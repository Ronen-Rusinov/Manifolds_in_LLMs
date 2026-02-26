#!/bin/bash
#SBATCH --job-name=Locales-embeddings
#SBATCH --output=./slurm/logs/locales-embeddings/Locales_embeddings_%j/.log
#SBATCH --error=./slurm/logs/locales_embeddings/Locales_embeddings_%j/.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80GB
#SBATCH --partition=studentbatch

python -u $LLM_PROJ_PATH/experiments/check_locales_embeddings.py \
  --config $LLM_PROJ_PATH/config/profile_production.yaml
