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

# K-nearest and clustering parameters
parser.add_argument("--k", type=int, help="Number of nearest neighbors to retrieve")
parser.add_argument("--balltree_leaf_size", type=int, help="Leaf size for BallTree")
parser.add_argument("--layer_for_activation", type=int, help="Layer index for activation extraction")
parser.add_argument("--use-dual-tree", action="store_true", help="Use dual-tree algorithm for BallTree queries")

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

if args.use_dual_tree:
    print(f"[{datetime.now()}] Using dual-tree algorithm for BallTree queries")

# Query all centroids at once
print(f"[{datetime.now()}] Querying all centroids for {k_nearest} nearest neighbors...")
dist, ind = tree.query(centroids, k=k_nearest, dualtree=args.use_dual_tree)
print(f"[{datetime.now()}] Nearest neighbors for all centroids obtained.")

# Save nearest neighbors indices
neighbors_save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results','Balltree' ,f'nearest_{k_nearest}_neighbors_indices_layer_{config.model.layer_for_activation}.npy'))
with open(neighbors_save_path, "wb") as f:
    numpy.save(f, ind)
print(f"[{datetime.now()}] Nearest neighbors indices saved to {neighbors_save_path}.")