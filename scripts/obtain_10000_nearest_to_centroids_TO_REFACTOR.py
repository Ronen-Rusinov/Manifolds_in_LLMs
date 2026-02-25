import os
import sys
import time
from datetime import datetime

#add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from utils import load_data
from config_manager import load_config_with_args
from sklearn.neighbors import BallTree
import numpy
import pickle

config = load_config_with_args(description="Obtain 10000 nearest neighbors for each centroid")

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

#If argv[1] = 0, test only on first centroid
#If argv[1] = 1, test on all centroids
#If argv[1] = 2, test on all centroids and use the dual tree algorithm setting

if len(sys.argv) < 2:
    print("Please provide a test number (0, 1, or 2).")
    sys.exit(1)
test_number = int(sys.argv[1])
if test_number == 0:
    print(f"[{datetime.now()}] Testing BallTree on first centroid...")
    dist, ind = tree.query(centroids[0].reshape(1, -1), k=config.clustering.k_nearest_10000)
    print(f"[{datetime.now()}] Nearest neighbors for first centroid obtained.")

     #save the indices of the nearest neighbors for all centroids
    neighbors_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results','Balltree' ,'nearest_neighbors_indices_0.npy'))
    with open(neighbors_path, "wb") as f:
        numpy.save(f, ind)
    print(f"[{datetime.now()}] Nearest neighbors indices saved to {neighbors_path}.")


elif test_number == 1:
    print(f"[{datetime.now()}] Testing BallTree on all centroids...")
    dist, ind = tree.query(centroids, k=config.clustering.k_nearest_10000)
    print(f"[{datetime.now()}] Nearest neighbors for all centroids obtained.")

    #save the indices of the nearest neighbors for all centroids
    neighbors_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results','Balltree' ,'nearest_neighbors_indices_1.npy'))
    with open(neighbors_path, "wb") as f:
        numpy.save(f, ind)
    print(f"[{datetime.now()}] Nearest neighbors indices saved to {neighbors_path}.")

elif test_number == 2:
    print(f"[{datetime.now()}] Testing BallTree on all centroids with dual tree algorithm...")
    dist, ind = tree.query(centroids, k=config.clustering.k_nearest_10000, dualtree=True)
    print(f"[{datetime.now()}] Nearest neighbors for all centroids obtained with dual tree algorithm.")

    #save the indices of the nearest neighbors for all centroids
    neighbors_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results','Balltree' ,'nearest_neighbors_indices_2.npy'))
    with open(neighbors_path, "wb") as f:
        numpy.save(f, ind)
    print(f"[{datetime.now()}] Nearest neighbors indices saved to {neighbors_path}.")
else:
    print("Invalid test number. Please provide 0, 1, or 2.")
    sys.exit(1)

