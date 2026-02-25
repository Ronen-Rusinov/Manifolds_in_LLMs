#!/bin/bash
#SBATCH --job-name=train_autoencoder
#SBATCH --output=./slurm/logs/train_autoencoder/train_autoencoder_%j/.log
#SBATCH --error=./slurm/logs/train_autoencoder/train_autoencoder_%j/.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=50GB
#SBATCH --partition=studentbatch
#SBATCH --gres=gpu:1
#SBATCH --constraint="geforce_rtx_2080"

python -u $LLM_PROJ_PATH/scripts/train_autoencoder.py \
  --config $LLM_PROJ_PATH/config/profile_production.yaml