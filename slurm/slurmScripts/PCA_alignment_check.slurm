#!/bin/bash
#SBATCH --job-name=PCA_alignment_check
#SBATCH --output=./slurm/logs/PCA_mapping_alignment/PCA_mapping_alignment_%j/.log
#SBATCH --error=./slurm/logs/PCA_mapping_alignment/PCA_mapping_alignment_%j/.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100GB
#SBATCH --partition=studentbatch

python -u experiments/mapping_alignment_pca.py --n-centroids 200 --n-components 50 --k-nearest-large 10000
