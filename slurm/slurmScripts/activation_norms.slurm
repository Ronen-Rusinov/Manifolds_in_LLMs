#!/bin/bash
#SBATCH --job-name=Activation_Norms
#SBATCH --output=./slurm/logs/activation_norms/Activation_Norms_%j/.log
#SBATCH --error=./slurm/logs/activation_norms/Activation_Norms_%j/.err
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30GB
#SBATCH --partition=studentbatch

python -u $LLM_PROJ_PATH/experiments/activation_norms.py \
  --config $LLM_PROJ_PATH/config/profile_production.yaml
