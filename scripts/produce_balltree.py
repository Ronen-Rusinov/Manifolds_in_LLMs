import os
import sys
import time
import argparse
from datetime import datetime

#add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from utils import load_data
from config_manager import load_config, add_config_argument
from sklearn.neighbors import BallTree
import numpy as np
import pickle

def build_ball_tree_first_parquet(config, layer, save_path=None, use_dual_tree=False):
    #load first parquet
    print(f"[{datetime.now()}] loading first parquet...")
    df = load_data.load_first_parquet(timing=False)
    #get activations
    print(f"[{datetime.now()}] Obtaining activations from layer {layer}...")
    activations = np.array(df[f"activation_layer_{layer}"].tolist(), dtype=np.float32)

    print(f"[{datetime.now()}] Building BallTree with leaf_size={config.model.balltree_leaf_size}...")
    start_time = time.time()
    tree = BallTree(activations, leaf_size=config.model.balltree_leaf_size)
    build_time = time.time() - start_time
    print(f"[{datetime.now()}] BallTree built in {build_time:.2f} seconds.")

    if save_path is not None:
        #Make sure the save path exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(tree, f)
        print(f"[{datetime.now()}] BallTree saved to {save_path}.")
            

def build_ball_tree_all_parquets(config, layer, save_path=None):
    #load all parquets
    print(f"[{datetime.now()}] loading all parquets...")
    df = load_data.load_all_parquets(timing=False)
    #get activations
    print(f"[{datetime.now()}] Obtaining activations from layer {layer}...")
    activations = np.array(df[f"activation_layer_{layer}"].tolist(), dtype=np.float32)

    print(f"[{datetime.now()}] Building BallTree with leaf_size={config.model.balltree_leaf_size}...")
    start_time = time.time()
    tree = BallTree(activations, leaf_size=config.model.balltree_leaf_size)
    build_time = time.time() - start_time
    print(f"[{datetime.now()}] BallTree built in {build_time:.2f} seconds.")

    if save_path is not None:
        #Make sure the save path exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(tree, f)
        print(f"[{datetime.now()}] BallTree saved to {save_path}.")
            
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Build BallTree for efficient nearest neighbor queries")
    
    # BallTree parameters
    parser.add_argument("--balltree_leaf_size", type=int, help="Leaf size for BallTree")
    parser.add_argument("--layer_for_activation", type=int, help="Layer index for primary activation extraction")
    parser.add_argument("--layer_alternative", type=int, help="Layer index for alternative activation extraction")
    
    add_config_argument(parser)
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    # Override config with CLI arguments
    if args.balltree_leaf_size is not None:
        config.model.balltree_leaf_size = args.balltree_leaf_size
    if args.layer_for_activation is not None:
        config.model.layer_for_activation = args.layer_for_activation
    if args.layer_alternative is not None:
        config.model.layer_alternative = args.layer_alternative

    build_ball_tree_all_parquets(
        config=config,
        layer=config.model.layer_for_activation,
        save_path=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results','Balltree' ,f'balltree_layer_{config.model.layer_for_activation}_all_parquets.pkl'))
    )