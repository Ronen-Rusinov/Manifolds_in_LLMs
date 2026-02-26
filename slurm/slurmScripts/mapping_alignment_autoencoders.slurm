#!/bin/bash
#SBATCH --job-name=mapping_alignment_autoencoders
#SBATCH --output=./slurm/logs/mapping_alignment_autoencoders/mapping_alignment_autoencoders_%j/.log
#SBATCH --error=./slurm/logs/mapping_alignment_autoencoders/mapping_alignment_autoencoders_%j/.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100GB
#SBATCH --partition=studentbatch

python -u $LLM_PROJ_PATH/experiments/mapping_alignment_autoencoders.py \
  --n-components 50 \
  --config $LLM_PROJ_PATH/config/profile_production.yaml
