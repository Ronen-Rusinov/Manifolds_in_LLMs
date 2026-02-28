import os
import sys
from datetime import datetime
from pathlib import Path

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from config_manager import load_config, add_config_argument

# Load configuration with CLI argument overrides
parser = argparse.ArgumentParser(
	description="Compare alignment distances across Isomap, Autoencoder, and PCA methods"
)

# Mapping alignment parameters
parser.add_argument("--n-centroids", type=int, help="Number of centroids")
parser.add_argument("--n-components", type=int, help="Number of components")
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
	base_path = Path(__file__).parent.parent / "results"
	
	# Define paths for the three methods
	isomap_dir = base_path / f"mapping_alignment_{config.clustering.n_centroids}_{config.dimensionality.n_components}_{config.clustering.k_nearest_large}"
	autoencoder_dir = base_path / f"mapping_alignment_autoencoders_{config.clustering.n_centroids}_{config.dimensionality.n_components}_{config.clustering.k_nearest_large}"
	pca_dir = base_path / f"mapping_alignment_pca_{config.clustering.n_centroids}_{config.dimensionality.n_components}_{config.clustering.k_nearest_large}"
	
	# Define file paths for alignment distances (after alignment)
	isomap_path = isomap_dir / f"alignment_distances_{config.dimensionality.n_components}D_n_clusters{config.clustering.n_centroids}.npy"
	autoencoder_path = autoencoder_dir / f"alignment_distances_{config.dimensionality.n_components}D_n_clusters{config.clustering.n_centroids}.npy"
	pca_path = pca_dir / f"alignment_distances_{config.dimensionality.n_components}D_n_clusters{config.clustering.n_centroids}.npy"
	
	# Check if all files exist
	missing_files = []
	for path, name in [(isomap_path, "Isomap"), (autoencoder_path, "Autoencoder"), (pca_path, "PCA")]:
		if not path.exists():
			missing_files.append((name, path))
	
	if missing_files:
		print(f"[{datetime.now()}] Error: Missing alignment distance files:", flush=True)
		for name, path in missing_files:
			print(f"  - {name}: {path}", flush=True)
		print("\nPlease run the corresponding mapping alignment scripts first:", flush=True)
		print("  - mapping_alignment.py (for Isomap)", flush=True)
		print("  - mapping_alignment_autoencoders.py (for Autoencoder)", flush=True)
		print("  - mapping_alignment_pca.py (for PCA)", flush=True)
		return
	
	# Load all three alignment distance matrices
	print(f"[{datetime.now()}] Loading alignment distance matrices...", flush=True)
	alignment_distances_isomap = np.load(isomap_path)
	alignment_distances_autoencoder = np.load(autoencoder_path)
	alignment_distances_pca = np.load(pca_path)
	print(f"[{datetime.now()}] All matrices loaded successfully.", flush=True)
	
	# Load MDS 1D embedding to reorder centroids by similarity
	mds_embedding_path = base_path / "overlaps" / f"mds_1d_embedding_{config.clustering.n_centroids}_{config.clustering.k_nearest_large}_{config.model.layer_for_activation}.npy"
	if mds_embedding_path.exists():
		print(f"[{datetime.now()}] Loading MDS 1D embedding for centroid reordering...", flush=True)
		mds_embedding = np.load(mds_embedding_path)
		centroid_order = np.argsort(mds_embedding)
		print(f"[{datetime.now()}] Reordering centroids by MDS embedding...", flush=True)
		
		# Reorder all alignment distance matrices
		alignment_distances_isomap = alignment_distances_isomap[centroid_order][:, centroid_order]
		alignment_distances_autoencoder = alignment_distances_autoencoder[centroid_order][:, centroid_order]
		alignment_distances_pca = alignment_distances_pca[centroid_order][:, centroid_order]
		print(f"[{datetime.now()}] Matrices reordered by centroid similarity.", flush=True)
	else:
		print(f"[{datetime.now()}] Warning: MDS embedding not found at {mds_embedding_path}.", flush=True)
		print(f"[{datetime.now()}] Run check_neighborhoods_overlap.py first to generate the embedding.", flush=True)
		print(f"[{datetime.now()}] Proceeding with original centroid ordering...", flush=True)
	
	# Determine the common colorbar range across all three matrices
	# Exclude sentinel values from the range calculation
	valid_isomap = alignment_distances_isomap[alignment_distances_isomap != config.numerical.sentinel_value]
	valid_autoencoder = alignment_distances_autoencoder[alignment_distances_autoencoder != config.numerical.sentinel_value]
	valid_pca = alignment_distances_pca[alignment_distances_pca != config.numerical.sentinel_value]
	
	vmin = min(valid_isomap.min(), valid_autoencoder.min(), valid_pca.min())
	vmax = max(valid_isomap.max(), valid_autoencoder.max(), valid_pca.max())
	
	print(f"[{datetime.now()}] Colorbar range: [{vmin:.4f}, {vmax:.4f}]", flush=True)
	

	#sentinal valued squares should appear to be white in the heatmap, so we set the colormap to have white for the sentinel value
	cmap = plt.cm.viridis
	cmap.set_bad(color='white')
	#set lal sentinels to NaN so they will be colored white in the heatmap
	alignment_distances_isomap[alignment_distances_isomap == config.numerical.sentinel_value] = np.nan
	alignment_distances_autoencoder[alignment_distances_autoencoder == config.numerical.sentinel_value] = np.nan
	alignment_distances_pca[alignment_distances_pca == config.numerical.sentinel_value] = np.nan

	# Create a compact layout with heatmaps on top and histogram below
	fig = plt.figure(figsize=(15, 8))
	gs = GridSpec(2, 4, figure=fig, height_ratios=[1, 0.6], width_ratios=[1, 1, 1, 0.05], 
	              hspace=0.35, wspace=0.15)
	axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
	cax = fig.add_subplot(gs[0, 3])
	hist_ax = fig.add_subplot(gs[1, :3])
	
	# Plot Isomap
	im1 = axes[0].imshow(alignment_distances_isomap, cmap=cmap, vmin=vmin, vmax=vmax)
	axes[0].set_title("Isomap")
	ylabel_text = "Centroid (MDS ordered)" if mds_embedding_path.exists() else "Centroid"
	axes[0].set_ylabel(ylabel_text)
	
	# Plot Autoencoder
	im2 = axes[1].imshow(alignment_distances_autoencoder, cmap=cmap, vmin=vmin, vmax=vmax)
	axes[1].set_title("Autoencoder")
	xlabel_text = "Centroid (MDS ordered)" if mds_embedding_path.exists() else "Centroid"
	axes[1].set_xlabel(xlabel_text)
	
	# Plot PCA
	im3 = axes[2].imshow(alignment_distances_pca, cmap=cmap, vmin=vmin, vmax=vmax)
	axes[2].set_title("PCA")
	
	# Add a shared colorbar
	fig.colorbar(im3, cax=cax, label="Mean Distance After Alignment")
	
	# Plot error distribution histogram with shared bins across all methods
	hist_min = min(valid_isomap.min(), valid_autoencoder.min(), valid_pca.min())
	hist_max = max(valid_isomap.max(), valid_autoencoder.max(), valid_pca.max())
	shared_bins = np.linspace(hist_min, hist_max, 500)
	
	hist_ax.hist(valid_isomap, bins=shared_bins, alpha=0.6, label='Isomap', color='#1f77b4', edgecolor='black', linewidth=0.5)
	hist_ax.hist(valid_autoencoder, bins=shared_bins, alpha=0.6, label='Autoencoder', color='#ff7f0e', edgecolor='black', linewidth=0.5)
	hist_ax.hist(valid_pca, bins=shared_bins, alpha=0.6, label='PCA', color='#2ca02c', edgecolor='black', linewidth=0.5)
	hist_ax.set_xlabel("Alignment Distance")
	hist_ax.set_ylabel("Frequency")
	hist_ax.set_title("Distribution of Alignment Errors")
	hist_ax.legend(loc='upper right')
	
	# Add overall title
	title_suffix = " (MDS ordered)" if mds_embedding_path.exists() else ""
	fig.suptitle(
		f"Alignment Distance Comparison ({config.dimensionality.n_components}D, {config.clustering.n_centroids} clusters){title_suffix}",
		fontsize=14
	)
	
	# Save the comparison figure
	output_dir = base_path / "mapping_alignment_comparison"
	output_dir.mkdir(parents=True, exist_ok=True)
	
	output_path = output_dir / f"alignment_comparison_{config.dimensionality.n_components}D_n_clusters{config.clustering.n_centroids}.png"
	print(f"[{datetime.now()}] Saving comparison heatmap to {output_path}...", flush=True)
	plt.savefig(output_path, dpi=300, bbox_inches='tight')
	print(f"[{datetime.now()}] Comparison heatmap saved successfully.", flush=True)
	
	# Print summary statistics
	print(f"\n[{datetime.now()}] Summary Statistics:", flush=True)
	print(f"  Isomap - Mean: {valid_isomap.mean():.4f}, Std: {valid_isomap.std():.4f}, Min: {valid_isomap.min():.4f}, Max: {valid_isomap.max():.4f}", flush=True)
	print(f"  Autoencoder - Mean: {valid_autoencoder.mean():.4f}, Std: {valid_autoencoder.std():.4f}, Min: {valid_autoencoder.min():.4f}, Max: {valid_autoencoder.max():.4f}", flush=True)
	print(f"  PCA - Mean: {valid_pca.mean():.4f}, Std: {valid_pca.std():.4f}, Min: {valid_pca.min():.4f}, Max: {valid_pca.max():.4f}", flush=True)


if __name__ == "__main__":
	main()
