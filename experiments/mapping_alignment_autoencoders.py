import os
import sys
from datetime import datetime
from pathlib import Path

import argparse
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from config_manager import load_config, add_config_argument
from utils import common
from rigid_procrustes import impose_X_on_Y

# Load configuration with CLI argument overrides
parser = argparse.ArgumentParser(
	description="Check if overlapping neighborhoods can be rigidly aligned (autoencoder embeddings)"
)

# Mapping alignment parameters
parser.add_argument("--n-centroids", type=int, help="Number of centroids")
parser.add_argument("--n-components", type=int, help="Number of components for autoencoder")
parser.add_argument("--k-nearest-large", type=int, help="Number of nearest neighbors")

add_config_argument(parser)
args = parser.parse_args()
config = load_config(args.config)

# Override config with CLI arguments
if args.n_centroids is not None:
	config.clustering.n_centroids = args.n_centroids
if args.n_components is not None:
	config.dimensionality.n_components = args.n_components
if args.k_nearest_large is not None:
	config.clustering.k_nearest_large = args.k_nearest_large


def main():
	result_dir = (
		Path(__file__).parent.parent
		/ "results"
		/ f"mapping_alignment_autoencoders_{config.clustering.n_centroids}_{config.dimensionality.n_components}_{config.clustering.k_nearest_large}"
	)
	result_dir.mkdir(parents=True, exist_ok=True)

	alignment_distances = np.full(
		(config.clustering.n_centroids, config.clustering.n_centroids),
		config.numerical.sentinel_value,
		dtype=np.float32,
	)
	alignment_distances_before_alignment = np.full(
		(config.clustering.n_centroids, config.clustering.n_centroids),
		config.numerical.sentinel_value,
		dtype=np.float32,
	)

	# Load all embeddings into memory to avoid repeated disk access
	all_embeddings = common.batch_load_autoencoder_embeddings(
		config.clustering.n_centroids,
		config.dimensionality.n_components,
	)

	# Load required data using shared utilities
	centroids = common.load_centroids(f"minibatch_kmeans_{config.clustering.n_centroids}")
	neighbor_indices = common.load_neighbor_indices(
		f"nearest_{config.clustering.k_nearest_large}_neighbors_indices_layer_{config.model.layer_for_activation}_n_centroids_{config.clustering.n_centroids}.npy"
	)
	activations = common.load_activations(config=config)

	# Validate data consistency
	common.validate_data_consistency(centroids, neighbor_indices, activations)

	for i in range(config.clustering.n_centroids):
		for j in range(i + 1, config.clustering.n_centroids):
			print(f"[{datetime.now()}] Processing centroid pair ({i}, {j})...", flush=True)

			indices_i = neighbor_indices[i]
			indices_j = neighbor_indices[j]

			embeddings_i = all_embeddings[i]
			embeddings_j = all_embeddings[j]

			indices_i_set = set(indices_i)
			indices_j_set = set(indices_j)
			common_indices = list(indices_i_set.intersection(indices_j_set))

			if len(common_indices) <= config.dimensionality.n_components//2:
				print(
					f"[{datetime.now()}] Skipping centroid pair ({i}, {j}) due to insufficient common neighbors ({len(common_indices)}).",
					flush=True,
				)
				continue

			local_indices_i = [np.where(indices_i == common_index)[0][0] for common_index in common_indices]
			local_indices_j = [np.where(indices_j == common_index)[0][0] for common_index in common_indices]

			common_embeddings_i = embeddings_i[local_indices_i]
			common_embeddings_j = embeddings_j[local_indices_j]

			print(
				f"[{datetime.now()}] Common embeddings shape for centroid {i}: {common_embeddings_i.shape}",
				flush=True,
			)
			print(
				f"[{datetime.now()}] Common embeddings shape for centroid {j}: {common_embeddings_j.shape}",
				flush=True,
			)

			aligned_embeddings_j = impose_X_on_Y(common_embeddings_i.T, common_embeddings_j.T).T

			distances = np.linalg.norm(aligned_embeddings_j - common_embeddings_j, axis=1)
			unaligned_distances = np.linalg.norm(common_embeddings_i - common_embeddings_j, axis=1)

			print(
				f"[{datetime.now()}] Centroid pair ({i}, {j}) - Mean distance after alignment: {distances.mean():.4f}, "
				f"Mean distance before alignment: {unaligned_distances.mean():.4f}",
				flush=True,
			)
			alignment_distances[i, j] = distances.mean()
			alignment_distances_before_alignment[i, j] = unaligned_distances.mean()
			alignment_distances[j, i] = distances.mean()
			alignment_distances_before_alignment[j, i] = unaligned_distances.mean()

	alignment_distances_path = (
		result_dir
		/ f"alignment_distances_{config.dimensionality.n_components}D_n_clusters{config.clustering.n_centroids}.npy"
	)
	print(
		f"[{datetime.now()}] Saving alignment distances matrix to {alignment_distances_path}...",
		flush=True,
	)
	np.save(alignment_distances_path, alignment_distances)
	print(f"[{datetime.now()}] Alignment distances matrix saved.", flush=True)

	alignment_distances_before_alignment_path = (
		result_dir
		/ f"alignment_distances_before_alignment_{config.dimensionality.n_components}D_n_clusters{config.clustering.n_centroids}.npy"
	)
	print(
		f"[{datetime.now()}] Saving alignment distances matrix before alignment to {alignment_distances_before_alignment_path}...",
		flush=True,
	)
	np.save(alignment_distances_before_alignment_path, alignment_distances_before_alignment)
	print(f"[{datetime.now()}] Alignment distances matrix before alignment saved.", flush=True)

	import matplotlib.pyplot as plt

	# Determine the common colorbar range across both matrices
	vmin = min(alignment_distances_before_alignment[alignment_distances_before_alignment != config.numerical.sentinel_value].min(),
				alignment_distances[alignment_distances != config.numerical.sentinel_value].min())
	vmax = max(alignment_distances_before_alignment[alignment_distances_before_alignment != config.numerical.sentinel_value].max(),
				alignment_distances[alignment_distances != config.numerical.sentinel_value].max())

	fig, axes = plt.subplots(1, 2, figsize=(config.visualization.fig_width_standard, config.visualization.fig_height_standard))

	im1 = axes[0].imshow(alignment_distances_before_alignment, cmap='viridis', vmin=vmin, vmax=vmax)
	axes[0].set_title("Mean Distance Before Alignment")

	im2 = axes[1].imshow(alignment_distances, cmap='viridis', vmin=vmin, vmax=vmax)
	axes[1].set_title("Mean Distance After Alignment")

	fig.colorbar(im2, ax=axes)

	heatmap_path = (
		result_dir
		/ f"alignment_distances_heatmap_{config.dimensionality.n_components}D_n_clusters{config.clustering.n_centroids}.png"
	)
	print(f"[{datetime.now()}] Saving alignment distances heatmap to {heatmap_path}...", flush=True)
	plt.savefig(heatmap_path)
	print(f"[{datetime.now()}] Alignment distances heatmap saved.", flush=True)


if __name__ == "__main__":
	main()
