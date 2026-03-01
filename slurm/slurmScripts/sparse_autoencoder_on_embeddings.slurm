#!/bin/bash
#SBATCH --job-name=sparse_autoencoder_on_embeddings
#SBATCH --output=./slurm/logs/sparse_autoencoder_on_embeddings/sparse_autoencoder_on_embeddings_%j/.log
#SBATCH --error=./slurm/logs/sparse_autoencoder_on_embeddings/sparse_autoencoder_on_embeddings_%j/.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --partition=studentbatch
#SBATCH --gres=gpu:1
#SBATCH --constraint="geforce_rtx_2080"

#SBATCH --array=[0,50,100,150]

N_COMPONENTS=${N_COMPONENTS:-50}
DISABLE_3D=${DISABLE_3D:-0}
DISABLE_4D=${DISABLE_4D:-0}

ARGS=("--n-components" "${N_COMPONENTS}")
ARGS+=("--offset" "${SLURM_ARRAY_TASK_ID}")
ARGS+=("--count" "50")
if [[ "${DISABLE_3D}" == "1" ]]; then
	ARGS+=("--no-3d")
fi
if [[ "${DISABLE_4D}" == "1" ]]; then
	ARGS+=("--no-4d")
fi

ARGS+=("--config" "$LLM_PROJ_PATH/config/profile_production.yaml" "--source" "isomap" \
 "--n-components" "50")

echo "Running sparse autoencoder for centroids in range: ${SLURM_ARRAY_TASK_ID} to $((${SLURM_ARRAY_TASK_ID} + 40))"
echo "Args: ${ARGS[*]}"
python -u $LLM_PROJ_PATH/scripts/train_sparse_autoencoder_on_embeddings.py "${ARGS[@]}"