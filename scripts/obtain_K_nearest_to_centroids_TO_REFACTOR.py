import os
import sys
import time
from datetime import datetime

#add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from utils import load_data
from config_manager import load_config, add_config_argument
import argparse
from sklearn.neighbors import BallTree
import numpy
import pickle

parser = argparse.ArgumentParser(description="Obtain K nearest neighbors for each centroid")
parser.add_argument("mode", type=int, choices=[0, 1, 2],
                    help="Mode: 0=first centroid only, 1=all centroids, 2=all centroids with dual tree")

# K-nearest and clustering parameters
parser.add_argument("--k", type=int, help="Number of nearest neighbors to retrieve")
parser.add_argument("--balltree_leaf_size", type=int, help="Leaf size for BallTree")
parser.add_argument("--layer_for_activation", type=int, help="Layer index for activation extraction")

add_config_argument(parser)
args = parser.parse_args()
config = load_config(args.config)

# Override config with CLI arguments
k_nearest = args.k if args.k is not None else config.clustering.k_nearest_10000
if args.balltree_leaf_size is not None:
    config.model.balltree_leaf_size = args.balltree_leaf_size
if args.layer_for_activation is not None:
    config.model.layer_for_activation = args.layer_for_activation

#load centroids
print(f"[{datetime.now()}] loading centroids...")
centroids_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results','minibatch_kmeans' ,'centroids.npy'))
with open(centroids_path, "rb") as f:
    centroids = numpy.load(f)
print(f"[{datetime.now()}] centroids loaded from {centroids_path}.")
print(f"[{datetime.now()}] centroids shape: {centroids.shape}")


#load ball tree
print(f"[{datetime.now()}] loading BallTree...")
ball_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results','Balltree' ,'balltree_layer_18_all_parquets.pkl'))

with open(ball_path, "rb") as f:
    tree = pickle.load(f)
print(f"[{datetime.now()}] BallTree loaded from {ball_path}.")

print(f"[{datetime.now()}] Using K={k_nearest} nearest neighbors")
print(f"[{datetime.now()}] BallTree leaf size: {config.model.balltree_leaf_size}")

# Process based on mode argument
if args.mode == 0:
    print(f"[{datetime.now()}] Querying first centroid for {k_nearest} nearest neighbors...")
    dist, ind = tree.query(centroids[0].reshape(1, -1), k=k_nearest)
    print(f"[{datetime.now()}] Nearest neighbors for first centroid obtained.")

     #save the indices of the nearest neighbors
    neighbors_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results','Balltree' ,f'nearest_neighbors_indices_0_k{k_nearest}.npy'))
    with open(neighbors_path, "wb") as f:
        numpy.save(f, ind)
    print(f"[{datetime.now()}] Nearest neighbors indices saved to {neighbors_path}.")

elif args.mode == 1:
    print(f"[{datetime.now()}] Querying all centroids for {k_nearest} nearest neighbors...")
    dist, ind = tree.query(centroids, k=k_nearest)
    print(f"[{datetime.now()}] Nearest neighbors for all centroids obtained.")

    #save the indices of the nearest neighbors
    neighbors_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results','Balltree' ,f'nearest_neighbors_indices_1_k{k_nearest}.npy'))
    with open(neighbors_path, "wb") as f:
        numpy.save(f, ind)
    print(f"[{datetime.now()}] Nearest neighbors indices saved to {neighbors_path}.")

elif args.mode == 2:
    print(f"[{datetime.now()}] Querying all centroids for {k_nearest} nearest neighbors with dual tree algorithm...")
    dist, ind = tree.query(centroids, k=k_nearest, dualtree=True)
    print(f"[{datetime.now()}] Nearest neighbors for all centroids obtained with dual tree algorithm.")

    #save the indices of the nearest neighbors
    neighbors_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results','Balltree' ,f'nearest_neighbors_indices_2_k{k_nearest}.npy'))
    with open(neighbors_path, "wb") as f:
        numpy.save(f, ind)
    print(f"[{datetime.now()}] Nearest neighbors indices saved to {neighbors_path}.")

