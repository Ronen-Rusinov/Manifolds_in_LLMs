import numpy as np
import os
import pickle
from tqdm import tqdm
from src.config_manager import load_config, add_config_argument
import argparse

# Load configuration with CLI argument overrides
parser = argparse.ArgumentParser(description="Check overlap of nearest neighbor neighborhoods between centroids")

# Neighborhood parameters
parser.add_argument("--n_centroids", type=int, help="Number of centroids")
parser.add_argument("--k_nearest_10000", type=int, help="Number of nearest neighbors")
parser.add_argument("--layer_for_activation", type=int, help="Layer index for activation extraction")

add_config_argument(parser)
args = parser.parse_args()
config = load_config(args.config)

# Override config with CLI arguments
if args.n_centroids is not None:
    config.clustering.n_centroids = args.n_centroids
if args.k_nearest_10000 is not None:
    config.clustering.k_nearest_10000 = args.k_nearest_10000
if args.layer_for_activation is not None:
    config.model.layer_for_activation = args.layer_for_activation

#load centroids
print(f"Loading centroids...")
centroids_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results','minibatch_kmeans' ,'centroids.npy'))
with open(centroids_path, "rb") as f:
    centroids = np.load(f)
print(f"Centroids loaded from {centroids_path}.")
print(f"Centroids shape: {centroids.shape}")

#load nearest neighbors indices
print(f"Loading nearest neighbors indices...")
neighbors_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results','Balltree' ,'nearest_neighbors_indices_1.npy'))
with open(neighbors_path, "rb") as f:
    neighbors_indices = np.load(f)
print(f"Nearest neighbors indices loaded from {neighbors_path}.")
print(f"Nearest neighbors indices shape: {neighbors_indices.shape}")

total_seen_indices = set()
for i in range(len(centroids)):
    total_seen_indices.update(neighbors_indices[i])
print(f"Total unique nearest neighbor indices across all centroids: {len(total_seen_indices)}")

#if overlaps list and matrix exist, load them and skip the computation
overlaps_list_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results','overlaps' ,'overlaps_list.npy'))
overlaps_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results','overlaps' ,'overlaps_matrix.npy'))
if os.path.exists(overlaps_list_path) and os.path.exists(overlaps_path):
    print(f"Overlaps list and matrix already exist, loading them...")
    with open(overlaps_list_path, "rb") as f:
        overlaps_list = np.load(f)
    with open(overlaps_path, "rb") as f:
        overlaps = np.load(f)
    print(f"Overlaps list loaded from {overlaps_list_path}.")
    print(f"Overlaps matrix loaded from {overlaps_path}.")
    print(f"Overlaps list shape: {overlaps_list.shape}")
    print(f"Overlaps matrix shape: {overlaps.shape}")

else:
    print(f"Overlaps list and matrix do not exist, computing them...")
    overlaps = np.zeros((len(centroids), len(centroids)), dtype=int)

    #List of vectors, in R^3, first index is the first centroid, second index is the second centroid, third index is the number of shared neighbors between the two centroids
    overlaps_list = np.zeros((len(centroids)*(len(centroids)-1)//2,3), dtype=int)

    total_ind = 0
    for i in tqdm(range(len(centroids))):
        for j in range(i+1, len(centroids)):
            overlap = len(set(neighbors_indices[i]) & set(neighbors_indices[j]))
            overlaps[i, j] = overlap
            overlaps[j, i] = overlap
            overlaps_list[total_ind, 0] = i
            overlaps_list[total_ind, 1] = j
            overlaps_list[total_ind, 2] = overlap
            total_ind += 1

    #order the overlaps list by the number of shared neighbors
    overlaps_list = overlaps_list[overlaps_list[:, 2].argsort()[::-1]]

    #save the overlaps list to a file
    overlaps_list_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results','overlaps' ,'overlaps_list.npy'))
    np.save(overlaps_list_path, overlaps_list)
    print(f"Overlaps list saved to {overlaps_list_path}.")

    #save the overlaps matrix to a file
    overlaps_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results','overlaps' ,'overlaps_matrix.npy'))
    np.save(overlaps_path, overlaps)
    print(f"Overlaps matrix saved to {overlaps_path}.")


#make a nice heatmap of the overlaps
import matplotlib.pyplot as plt
plt.figure(figsize=(config.visualization.fig_width_compact, config.visualization.fig_height_compact))
plt.imshow(overlaps, cmap='magma')
plt.colorbar(label='Number of shared neighbors')
plt.title('Overlap of Nearest Neighbors between Centroids')
plt.xlabel('Centroid Index')
plt.ylabel('Centroid Index')
heatmap_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results','overlaps' ,'overlaps_heatmap.png'))
plt.savefig(heatmap_path)
print(f"Overlap heatmap saved to {heatmap_path}.")

#For each row, make a histogram with N bins of the number of shared neighbors between the centroid and the other centroids
#plot them all together in a single heatmap, where the x axis is the number of shared neighbors and the y axis is the centroid index

hist_arr = np.zeros((len(centroids), config.visualization.histogram_bins), dtype=int)
for i in range(len(centroids)):
    hist = np.histogram(overlaps[i], bins=config.visualization.histogram_bins, range=(0, config.clustering.k_nearest_10000))
    hist_arr[i] = hist[0]
plt.figure(figsize=(config.visualization.fig_width_compact, config.visualization.fig_height_compact))
plt.imshow(hist_arr, aspect='auto', cmap='viridis')
plt.colorbar(label='Frequency of shared neighbors')
plt.title('Histogram of Shared Neighbors for Each Centroid')
plt.xlabel('Number of Shared Neighbors (binned)')
plt.ylabel('Centroid Index')
histogram_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results','overlaps' ,'overlaps_histogram_heatmap.png'))
plt.savefig(histogram_path)
print(f"Overlap histogram heatmap saved to {histogram_path}.")

#Make a histogram of the overlaps
plt.figure(figsize=(config.visualization.fig_width_compact, config.visualization.fig_height_standard))
plt.hist(overlaps_list[:, 2], bins=config.visualization.histogram_bins, color='blue', alpha=0.7)
plt.title('Distribution of Shared Neighbors between Centroids')
plt.xlabel('Number of Shared Neighbors')
plt.ylabel('Frequency')
histogram_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results','overlaps' ,'overlaps_histogram.png'))
plt.savefig(histogram_path)
print(f"Overlap histogram saved to {histogram_path}.")


#Treat the overlaps list as a graph, where the centroids are the nodes and the number of shared neighbors is the weight of the edge between two nodes.
#Apply a physics-based layout algorithm to the graph, where the nodes repel each other and the edges attract the nodes based on their weights.
import networkx as nx
G = nx.Graph()
for i in range(len(centroids)):
    G.add_node(i)
for i in range(len(centroids)):
    for j in range(i+1, len(centroids)):
        G.add_edge(i, j, weight=overlaps[i, j]/1000) #normalize the weight by the maximum number of shared neighbors (10000) to avoid very large weights
pos = nx.spring_layout(G, weight='weight')
#poss is a list with entries of the form [0-199] : [x,y]
#From here we can use matplotlib to graph
#Width and opacity of conncetions between nodes is proportional to the weight of the edge

#ROtation counterclockwise for aesthetics (I like when the big clump is on the right :D)
for key in pos:
    x, y = pos[key]
    pos[key] = [-y, x]

plt.figure(figsize=(config.visualization.fig_width_large, config.visualization.fig_height_large))
for i in range(len(centroids)):
    for j in range(i+1, len(centroids)):
        weight = overlaps[i, j]/10000
        if weight > 0: #only plot edges with non-zero weight
            plt.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color='blue', alpha=weight, linewidth=weight*5)
plt.scatter([pos[i][0] for i in range(len(centroids))], [pos[i][1] for i in range(len(centroids))], color='red', s=100)
#add labels to the nodes
for i in range(len(centroids)):
    plt.text(pos[i][0], pos[i][1], str(i), fontsize=6, ha='center', va='center', color='white')


plt.title('Graph of Centroids based on Shared Neighbors')
graph_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results','overlaps' ,'overlaps_graph.png'))
plt.savefig(graph_path, dpi=300)
print(f"Overlap graph saved to {graph_path}.")

