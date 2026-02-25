#!/bin/bash
#SBATCH --job-name=produce_balltree
#SBATCH --output=./slurm/logs/produce_balltree/produce_balltree_%j/.log
#SBATCH --error=./slurm/logs/produce_balltree/produce_balltree_%j/.err
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80GB
#SBATCH --partition=studentbatch

echo "Running BallTree Production"
echo "python -u $LLM_PROJ_PATH/scripts/produce_balltree.py \
  --config $LLM_PROJ_PATH/config/profile_production.yaml"
  
python -u $LLM_PROJ_PATH/scripts/produce_balltree.py \
  --config $LLM_PROJ_PATH/config/profile_production.yaml
