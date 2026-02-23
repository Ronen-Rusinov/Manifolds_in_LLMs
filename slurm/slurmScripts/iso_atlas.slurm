#!/bin/bash
#SBATCH --job-name=iso_atlas
#SBATCH --output=./slurm/logs/logs/iso_atlas/iso_atlas_%j.log
#SBATCH --error=./slurm/logs/errs/iso_atlas/iso_atlas_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=100GB
#SBATCH --partition=studentbatch
#SBATCH --gres=gpu:1
#SBATCH --constraint="geforce_rtx_2080"
#SBATCH --array=[0,40,80,120,160]

echo "Running Isomap for centroids in range: ${SLURM_ARRAY_TASK_ID} to $((${SLURM_ARRAY_TASK_ID} + 40))"
python -u $LLM_PROJ_PATH/scripts/isomap_for_each_centroid.py ${SLURM_ARRAY_TASK_ID} 1