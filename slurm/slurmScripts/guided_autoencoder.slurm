#!/bin/bash
#SBATCH --job-name=GuidedAutoencoder
#SBATCH --output=./slurm/logs/GuidedAutoencoder/GuidedAutoencoder_%j/.log
#SBATCH --error=./slurm/logs/GuidedAutoencoder/GuidedAutoencoder_%j/.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=70GB
#SBATCH --partition=studentbatch
#SBATCH --gres=gpu:1
#SBATCH --constraint="geforce_rtx_2080"

python -u $LLM_PROJ_PATH/experiments/autoencoder_with_guidance.py