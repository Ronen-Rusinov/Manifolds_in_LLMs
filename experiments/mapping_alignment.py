import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from config_manager import load_config, add_config_argument
import argparse
from utils import common
import numpy as np
from rigid_procrustes import impose_X_on_Y, procrustes

# Load configuration with CLI argument overrides
parser = argparse.ArgumentParser(description="Check if overlapping neighborhoods can be rigidly aligned")

# Mapping alignment parameters
parser.add_argument("--n-centroids", type=int, help="Number of centroids")
parser.add_argument("--n-components", type=int, help="Number of components for Isomap")
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

    # Initialize result matrices and directories
    os.makedirs(Path(__file__).parent.parent / "results" / f"mapping_alignment_{config.clustering.n_centroids}_{config.dimensionality.n_components}_{config.clustering.k_nearest_large}", exist_ok=True)
    alignment_distances = np.full((config.clustering.n_centroids, config.clustering.n_centroids), config.numerical.sentinel_value, dtype=np.float32)
    alignment_distances_before_alignment = np.full((config.clustering.n_centroids, config.clustering.n_centroids), config.numerical.sentinel_value, dtype=np.float32)

    # Load all embeddings into memory to avoid repeated disk access
    all_embeddings = common.batch_load_isomap_embeddings(
        config.clustering.n_centroids,
        config.dimensionality.n_components
    )
    
    # Load required data using shared utilities
    centroids = common.load_centroids(f"minibatch_kmeans_{config.clustering.n_centroids}")
    neighbor_indices = common.load_neighbor_indices(f"nearest_{config.clustering.k_nearest_large}_neighbors_indices_layer_{config.model.layer_for_activation}_n_centroids_{config.clustering.n_centroids}.npy")
    activations = common.load_activations(config=config)
    
    # Validate data consistency
    common.validate_data_consistency(centroids, neighbor_indices, activations)
    

    for i in range(config.clustering.n_centroids):
        for j in range(i+1, config.clustering.n_centroids):
            print(f"[{datetime.now()}] Processing centroid pair ({i}, {j})...", flush=True)
            #Get the nearest neighbor indices for centroids i and j
            indices_i = neighbor_indices[i]
            indices_j = neighbor_indices[j]

            #Get the corresponding Isomap embeddings
            embeddings_i = all_embeddings[i]
            embeddings_j = all_embeddings[j]

            #get the global index of the matching activations within the neighborhoods of the centroids
            indices_i_set = set(indices_i)
            indices_j_set = set(indices_j)
            common_indices = list(indices_i_set.intersection(indices_j_set))

            if len(common_indices) <= config.dimensionality.n_components: 
                print(f"[{datetime.now()}] Skipping centroid pair ({i}, {j}) due to insufficient common neighbors ({len(common_indices)}).", flush=True)
                continue
            
            """
            Now, there is a certain intricacy.
            The indexing in both the embedding spaces are different, as they are both ordered according to the nearest neighbor indices of their respective centroids.

            Specifically, the code block responsible is:

            neighbor_idx = neighbor_indices[centroid_idx]
            neighborhood_activations = activations[neighbor_idx]
            print(f"[{datetime.now()}] Neighborhood activations shape: {neighborhood_activations.shape}", flush=True)
            print(f"[{datetime.now()}] Applying Isomap to 12D...", flush=True)
            embeddings_12d, isomap_12d = apply_isomap_to_neighborhood(
                neighborhood_activations, 
                neighbor_idx, 
                N_COMPONENTS_12D, 
                N_NEIGHBORS
            )

            So, the common indices are in the global indexing of the activations, but we need to find their corresponding indices in the local indexing of the embeddings.
            The local indexing is just the range [0, 9_999], so we can map the global indices to local indices by finding their positions in the neighbor_idx array.
            """

            local_indices_i = [np.where(indices_i == common_index)[0][0] for common_index in common_indices]
            local_indices_j = [np.where(indices_j == common_index)[0][0] for common_index in common_indices]

            #Now we can get the corresponding embeddings for the common indices
            common_embeddings_i = embeddings_i[local_indices_i]
            common_embeddings_j = embeddings_j[local_indices_j]

            #print embedding shapes
            print(f"[{datetime.now()}] Common embeddings shape for centroid {i}: {common_embeddings_i.shape}", flush=True)
            print(f"[{datetime.now()}] Common embeddings shape for centroid {j}: {common_embeddings_j.shape}", flush=True)

            #Now we can apply the procrustes analysis to find the optimal rigid transformation that aligns the two sets of embeddings
            aligned_embeddings_j = impose_X_on_Y(common_embeddings_i.T, common_embeddings_j.T).T
            
            #Now we can compute the distances between the aligned embeddings and the original embeddings
            distances = np.linalg.norm(aligned_embeddings_j - common_embeddings_j, axis=1)
            
            #For reference compute the distances without the alignment
            unaligned_distances = np.linalg.norm(common_embeddings_i - common_embeddings_j, axis=1)

            print(f"[{datetime.now()}] Centroid pair ({i}, {j}) - Mean distance after alignment: {distances.mean():.4f}, Mean distance before alignment: {unaligned_distances.mean():.4f}", flush=True)
            alignment_distances[i, j] = distances.mean()
            alignment_distances_before_alignment[i, j] = unaligned_distances.mean()
            alignment_distances[j, i] = distances.mean()
            alignment_distances_before_alignment[j, i] = unaligned_distances.mean()

    #Save the alignment distances matrix
    alignment_distances_path = Path(__file__).parent.parent / "results" / f"mapping_alignment_{config.clustering.n_centroids}_{config.dimensionality.n_components}_{config.clustering.k_nearest_large}" / f"alignment_distances_{config.dimensionality.n_components}D_n_clusters{config.clustering.n_centroids}.npy"
    print(f"[{datetime.now()}] Saving alignment distances matrix to {alignment_distances_path}...", flush=True)
    np.save(alignment_distances_path, alignment_distances)
    print(f"[{datetime.now()}] Alignment distances matrix saved.", flush=True)

    #Save the alignment distances matrix before alignment
    alignment_distances_before_alignment_path = Path(__file__).parent.parent / "results" / f"mapping_alignment_{config.clustering.n_centroids}_{config.dimensionality.n_components}_{config.clustering.k_nearest_large}" / f"alignment_distances_before_alignment_{config.dimensionality.n_components}D_n_clusters{config.clustering.n_centroids}.npy"
    print(f"[{datetime.now()}] Saving alignment distances matrix before alignment to {alignment_distances_before_alignment_path}...", flush=True)
    np.save(alignment_distances_before_alignment_path, alignment_distances_before_alignment)
    print(f"[{datetime.now()}] Alignment distances matrix before alignment saved.", flush=True)

    #save a heatmap of the alignment distances matrix before and after alignment for visual comparison
    import matplotlib.pyplot as plt
    plt.figure(figsize=(config.visualization.fig_width_standard, config.visualization.fig_height_standard))
    plt.subplot(1, 2, 1)
    plt.imshow(alignment_distances_before_alignment, cmap='viridis')
    plt.colorbar()
    plt.title("Mean Distance Before Alignment")

    plt.subplot(1, 2, 2)
    plt.imshow(alignment_distances, cmap='viridis')
    plt.colorbar()
    plt.title("Mean Distance After Alignment")

    heatmap_path = Path(__file__).parent.parent / "results" / f"mapping_alignment_{config.clustering.n_centroids}_{config.dimensionality.n_components}_{config.clustering.k_nearest_large}" / f"alignment_distances_heatmap_{config.dimensionality.n_components}D_n_clusters{config.clustering.n_centroids}.png"
    print(f"[{datetime.now()}] Saving alignment distances heatmap to {heatmap_path}...", flush=True)
    plt.savefig(heatmap_path)
    print(f"[{datetime.now()}] Alignment distances heatmap saved.", flush=True)

if __name__ == "__main__":
    main()




