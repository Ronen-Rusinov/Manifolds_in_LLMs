import os
import sys
import time
from datetime import datetime

#add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from utils import load_data
from config_manager import load_config_with_args
from sklearn.neighbors import BallTree
import numpy as np
import pickle

# Load configuration with CLI argument overrides
config = load_config_with_args(
    description="Build and save BallTree for efficient nearest neighbor queries"
)

def test_ball_tree_first_parquet(layer, save_path=None):
    #load first parquet
    print(f"[{datetime.now()}] loading first parquet...")
    df = load_data.load_first_parquet(timing=False)
    #get activations
    print(f"[{datetime.now()}] Obtaining activations from layer {layer}...")
    activations = np.array(df[f"activation_layer_{layer}"].tolist(), dtype=np.float16)

    print(f"[{datetime.now()}] Building BallTree with leaf_size={config.model.balltree_leaf_size}...")
    start_time = time.time()
    tree = BallTree(activations, leaf_size=config.model.balltree_leaf_size)
    build_time = time.time() - start_time
    print(f"[{datetime.now()}] BallTree built in {build_time:.2f} seconds.")

    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(tree, f)
        print(f"[{datetime.now()}] BallTree saved to {save_path}.")
            

def test_ball_tree_all_parquets(layer, save_path=None):
    #load all parquets
    print(f"[{datetime.now()}] loading all parquets...")
    df = load_data.load_all_parquets(timing=False)
    #get activations
    print(f"[{datetime.now()}] Obtaining activations from layer {layer}...")
    activations = np.array(df[f"activation_layer_{layer}"].tolist(), dtype=np.float16)

    print(f"[{datetime.now()}] Building BallTree with leaf_size={config.model.balltree_leaf_size}...")
    start_time = time.time()
    tree = BallTree(activations, leaf_size=config.model.balltree_leaf_size)
    build_time = time.time() - start_time
    print(f"[{datetime.now()}] BallTree built in {build_time:.2f} seconds.")

    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(tree, f)
        print(f"[{datetime.now()}] BallTree saved to {save_path}.")
            
if __name__ == "__main__":
    #if the number 1 is passed as an arg, test layers from config on the first parquet
    #if the number 2 is passed as an arg, test primary layer on all parquets
    #if the number 3 is passed as an arg, test alternative layer on all parquets
    if len(sys.argv) < 2:
        print("Please provide a test number (1, 2, or 3).")
        sys.exit(1)
    test_number = int(sys.argv[1])
    if test_number == 1:
        print(f"Testing BallTree on first parquet for layers {config.model.layer_alternative} and {config.model.layer_for_activation}...")
        test_ball_tree_first_parquet(layer=config.model.layer_alternative, save_path=f"balltree_layer_{config.model.layer_alternative}.pkl")
        test_ball_tree_first_parquet(layer=config.model.layer_for_activation, save_path=f"balltree_layer_{config.model.layer_for_activation}.pkl")
    elif test_number == 2:
        print(f"Testing BallTree on all parquets for layer {config.model.layer_for_activation}...")
        test_ball_tree_all_parquets(layer=config.model.layer_for_activation, save_path=f"balltree_layer_{config.model.layer_for_activation}_all_parquets.pkl")
    elif test_number == 3:
        print(f"Testing BallTree on all parquets for layer {config.model.layer_alternative}...")
        test_ball_tree_all_parquets(layer=config.model.layer_alternative, save_path=f"balltree_layer_{config.model.layer_alternative}_all_parquets.pkl")
    else:
        print("Invalid test number. Please provide 1, 2, or 3.")
