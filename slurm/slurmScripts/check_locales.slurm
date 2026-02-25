#!/bin/bash
#SBATCH --job-name=Locales
#SBATCH --output=./slurm/logs/locales/Locales_%j/.log
#SBATCH --error=./slurm/logs/locales/Locales_%j/.err
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=50GB
#SBATCH --partition=studentbatch

python -u $LLM_PROJ_PATH/experiments/check_locales_test.py \
  --config $LLM_PROJ_PATH/config/profile_production.yaml
