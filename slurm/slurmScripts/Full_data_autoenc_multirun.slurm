#!/bin/bash
#SBATCH --job-name=Full_data_autoenc_multirun
#SBATCH --output=./slurm/logs/Full_data_autoenc_multirun/Full_data_autoenc_multirun_%j/.log
#SBATCH --error=./slurm/logs/Full_data_autoenc_multirun/Full_data_autoenc_multirun_%j/.err
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100GB
#SBATCH --partition=studentbatch
#SBATCH --gres=gpu:1
#SBATCH --constraint="geforce_rtx_2080"

python -u scripts/train_tied_autoencoder_multiple_runs.py \
  --config config/default_config.yaml \
  --layer 18 \
  --chunk-size 100000 \
  --epochs 100 \
  --learning-rate 2e-2 \
  --patience 30 \
  --num-runs 20