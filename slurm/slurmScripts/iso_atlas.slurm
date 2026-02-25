#!/bin/bash
#SBATCH --job-name=iso_atlas
#SBATCH --output=./slurm/logs/iso_atlas/iso_atlas_%j/.log
#SBATCH --error=./slurm/logs/iso_atlas/iso_atlas_%j/.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --partition=studentbatch
#SBATCH --array=[0,40,80,120,160]

N_COMPONENTS=${N_COMPONENTS:-50}
DISABLE_3D=${DISABLE_3D:-1}
DISABLE_4D=${DISABLE_4D:-0}

ARGS=("--n-components" "${N_COMPONENTS}")
ARGS+=("--offset" "${SLURM_ARRAY_TASK_ID}")
ARGS+=("--count" "40")
if [[ "${DISABLE_3D}" == "1" ]]; then
	ARGS+=("--no-3d")
fi
if [[ "${DISABLE_4D}" == "1" ]]; then
	ARGS+=("--no-4d")
fi

ARGS+=("--config" "$LLM_PROJ_PATH/config/profile_production.yaml")

echo "Running Isomap for centroids in range: ${SLURM_ARRAY_TASK_ID} to $((${SLURM_ARRAY_TASK_ID} + 40))"
echo "Args: ${ARGS[*]}"
python -u $LLM_PROJ_PATH/scripts/isomap_for_each_centroid.py "${ARGS[@]}"