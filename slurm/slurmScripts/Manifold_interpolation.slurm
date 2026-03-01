#!/bin/bash
#SBATCH --job-name=Manifold_interpolation
#SBATCH --output=./slurm/logs/Manifold_interpolation/Manifold_interpolation_%j/.log
#SBATCH --error=./slurm/logs/Manifold_interpolation/Manifold_interpolation_%j/.err
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100GB
#SBATCH --partition=studentbatch
#IGNORE --gres=gpu:1
#IGNORE --constraint="geforce_rtx_2080"
#SBATCH --array=[0]


#If slurm jobid == 0, add option --method local_pca
#If slurm jobid == 1, add option --method pca
#If slurm jobid == 2, add option --method autoencoder

# Set method based on SLURM_ARRAY_TASK_ID
case $SLURM_ARRAY_TASK_ID in
  0)
    METHOD="--method local_pca"
    ;;
  1)
    METHOD="--method pca"
    ;;
  2)
    METHOD="--method autoencoder"
    ;;
  *)
    METHOD=""
    ;;
esac




python -u experiments/Manifold_interpolation.py \
  --layer 18 \
 $METHOD \

