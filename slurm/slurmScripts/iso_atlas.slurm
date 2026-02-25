#!/bin/bash
#SBATCH --job-name=iso_atlas
#SBATCH --output=./slurm/logs/logs/iso_atlas/iso_atlas_%j.log
#SBATCH --error=./slurm/logs/errs/iso_atlas/iso_atlas_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --partition=studentbatch
#SBATCH --array=[0,40,80,120,160]

N_COMPONENTS=${N_COMPONENTS:-50}
DISABLE_3D=${DISABLE_3D:-1}
DISABLE_4D=${DISABLE_4D:-1}

ARGS=("${SLURM_ARRAY_TASK_ID}" 40 "--n-components" "${N_COMPONENTS}")
if [[ "${DISABLE_3D}" == "1" ]]; then
	ARGS+=("--no-3d")
fi
if [[ "${DISABLE_4D}" == "1" ]]; then
	ARGS+=("--no-4d")
fi

echo "Running Isomap for centroids in range: ${SLURM_ARRAY_TASK_ID} to $((${SLURM_ARRAY_TASK_ID} + 40))"
echo "Args: ${ARGS[*]}"
python -u $LLM_PROJ_PATH/scripts/isomap_for_each_centroid.py "${ARGS[@]}"