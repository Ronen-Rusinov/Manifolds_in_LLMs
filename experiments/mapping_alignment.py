#This script check to confirm wether or not the overlaps in the embeddings are
#Indeed isometries. I.E, if the overlap between centroid A and centroid B, 
#Corresponds to embeddings which can be rigidly mapped to each other in 12D.

#The script's prerequisits are the outputs of the script obtain_10000_nearest_to_centroids.py
#And subsequently relies on the output of minibatch_kmeans.py as well as
#on the output of produce_balltree.py, as well as the outputs of isomap_for_each_centroid.py

#Their outputs are stored in 
#/results/Balltree/nearest_neighbors_indices_1.npy
#/results/minibatch_kmeans/centroids.npy
#/results/Balltree/balltree_layer_18_all_parquets.pkl
#and /results/iso_atlas/12D/centroid_[0000-0199]_embeddings_12D.npy respectively.

import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils.load_data import load_all_parquets
import numpy as np
from rigid_procrustes import impose_X_on_Y, procrustes

def load_centroids():
    """Load centroids from minibatch_kmeans."""
    centroids_path = Path(__file__).parent.parent / "results" / "minibatch_kmeans" / "centroids.npy"
    print(f"[{datetime.now()}] Loading centroids from {centroids_path}...", flush=True)
    centroids = np.load(centroids_path)
    print(f"[{datetime.now()}] Centroids loaded. Shape: {centroids.shape}", flush=True)
    return centroids

def load_neighbor_indices(indices_file="nearest_neighbors_indices_1.npy"):
    """Load precomputed nearest neighbor indices."""
    indices_path = Path(__file__).parent.parent / "results" / "Balltree" / indices_file
    print(f"[{datetime.now()}] Loading nearest neighbor indices from {indices_path}...", flush=True)
    neighbor_indices = np.load(indices_path)
    print(f"[{datetime.now()}] Neighbor indices loaded. Shape: {neighbor_indices.shape}", flush=True)
    return neighbor_indices

def load_activations():
    """Load all activation vectors and corresponding prompts."""
    print(f"[{datetime.now()}] Loading all activations and prompts...", flush=True)
    df = load_all_parquets(timing=True)
    activations = np.array(df['activation_layer_18'].tolist(), dtype=np.float32)
    print(f"[{datetime.now()}] Activations loaded. Shape: {activations.shape}", flush=True)
    return activations

def load_isomap_embeddings(centroid_index):
    """Load Isomap embeddings for a specific centroid."""
    #pretty format the centroid index to be 4 digits with leading zeros
    centroid_index_str = f"{centroid_index:04d}"
    embeddings_path = Path(__file__).parent.parent / "results" / "iso_atlas" / "12D" / f"centroid_{centroid_index_str}_embeddings_12D.npy"
    print(f"[{datetime.now()}] Loading Isomap embeddings for centroid {centroid_index} from {embeddings_path}...", flush=True)
    embeddings = np.load(embeddings_path)
    print(f"[{datetime.now()}] Isomap embeddings loaded for centroid {centroid_index}. Shape: {embeddings.shape}", flush=True)
    return embeddings

def main():
    centroids = load_centroids()
    neighbor_indices = load_neighbor_indices()
    activations = load_activations()

    #have -1 indicate a skipped pair due to insufficient common neighbors, and initialize the alignment distances matrix
    os.makedirs(Path(__file__).parent.parent / "results" / "mapping_alignment", exist_ok=True)
    alignment_distances = np.full((200, 200), -1.0, dtype=np.float32)
    alignment_distances_before_alignment = np.full((200, 200), -1.0, dtype=np.float32)

    #preload all embeddings into memory to avoid repeated disk access during the pairwise comparisons
    all_embeddings = {}
    for i in range(200):
        all_embeddings[i] = load_isomap_embeddings(i)

    for i in range(200):
        for j in range(i+1, 200):
            print(f"[{datetime.now()}] Processing centroid pair ({i}, {j})...", flush=True)
            #Get the nearest neighbor indices for centroids i and j
            indices_i = neighbor_indices[i]
            indices_j = neighbor_indices[j]

            #Get the corresponding activation vectors
            activations_i = activations[indices_i]
            activations_j = activations[indices_j]

            #Get the corresponding Isomap embeddings
            embeddings_i = all_embeddings[i]
            embeddings_j = all_embeddings[j]

            #get the global index of the matching activations within the neighborhoods of the centroids
            indices_i_set = set(indices_i)
            indices_j_set = set(indices_j)
            common_indices = list(indices_i_set.intersection(indices_j_set))

            if len(common_indices) <= 20:
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
    alignment_distances_path = Path(__file__).parent.parent / "results" / "mapping_alignment" / "alignment_distances.npy"
    print(f"[{datetime.now()}] Saving alignment distances matrix to {alignment_distances_path}...", flush=True)
    np.save(alignment_distances_path, alignment_distances)
    print(f"[{datetime.now()}] Alignment distances matrix saved.", flush=True)

    #Save the alignment distances matrix before alignment
    alignment_distances_before_alignment_path = Path(__file__).parent.parent / "results" / "mapping_alignment" / "alignment_distances_before_alignment.npy"
    print(f"[{datetime.now()}] Saving alignment distances matrix before alignment to {alignment_distances_before_alignment_path}...", flush=True)
    np.save(alignment_distances_before_alignment_path, alignment_distances_before_alignment)
    print(f"[{datetime.now()}] Alignment distances matrix before alignment saved.", flush=True)

    #save a heatmap of the alignment distances matrix before and after alignment for visual comparison
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(alignment_distances_before_alignment, cmap='viridis')
    plt.colorbar()
    plt.title("Mean Distance Before Alignment")

    #mark with red the pairs that were skipped due to insufficient common neighbors
    skipped_pairs = np.where(alignment_distances_before_alignment == -1)
    plt.scatter(skipped_pairs[1], skipped_pairs[0], color='red', label='Skipped Pairs', s=1)

    plt.subplot(1, 2, 2)
    plt.imshow(alignment_distances, cmap='viridis')
    plt.colorbar()
    plt.title("Mean Distance After Alignment")

    #mark with red the pairs that were skipped due to insufficient common neighbors
    skipped_pairs = np.where(alignment_distances == -1)
    plt.scatter(skipped_pairs[1], skipped_pairs[0], color='red', label='Skipped Pairs', s=1)

    heatmap_path = Path(__file__).parent.parent / "results" / "mapping_alignment" / "alignment_distances_heatmap.png"
    print(f"[{datetime.now()}] Saving alignment distances heatmap to {heatmap_path}...", flush=True)
    plt.savefig(heatmap_path)
    print(f"[{datetime.now()}] Alignment distances heatmap saved.", flush=True)

if __name__ == "__main__":
    main()




