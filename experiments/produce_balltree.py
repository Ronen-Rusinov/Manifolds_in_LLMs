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
    activations = np.array(df[f"activation_layer_{layer}"].tolist(), dtype=np.float16)

    print(f"[{datetime.now()}] Building BallTree with leaf_size={config.model.balltree_leaf_size}, dual_tree={use_dual_tree}...")
    start_time = time.time()
    tree = BallTree(activations, leaf_size=config.model.balltree_leaf_size)
    build_time = time.time() - start_time
    print(f"[{datetime.now()}] BallTree built in {build_time:.2f} seconds.")

    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(tree, f)
        print(f"[{datetime.now()}] BallTree saved to {save_path}.")
            

def build_ball_tree_all_parquets(config, layer, save_path=None, use_dual_tree=False):
    #load all parquets
    print(f"[{datetime.now()}] loading all parquets...")
    df = load_data.load_all_parquets(timing=False)
    #get activations
    print(f"[{datetime.now()}] Obtaining activations from layer {layer}...")
    activations = np.array(df[f"activation_layer_{layer}"].tolist(), dtype=np.float16)

    print(f"[{datetime.now()}] Building BallTree with leaf_size={config.model.balltree_leaf_size}, dual_tree={use_dual_tree}...")
    start_time = time.time()
    tree = BallTree(activations, leaf_size=config.model.balltree_leaf_size)
    build_time = time.time() - start_time
    print(f"[{datetime.now()}] BallTree built in {build_time:.2f} seconds.")

    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(tree, f)
        print(f"[{datetime.now()}] BallTree saved to {save_path}.")
            
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Build BallTree for efficient nearest neighbor queries")
    parser.add_argument("mode", type=int, choices=[1, 2, 3],
                        help="Build mode: 1=first parquet (both layers), 2=all parquets (primary layer), 3=all parquets (alternative layer)")
    parser.add_argument("--use-dual-tree", action="store_true",
                        help="Use dual-tree algorithm for BallTree queries")
    
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
    
    if args.mode == 1:
        print(f"Building BallTree on first parquet for layers {config.model.layer_alternative} and {config.model.layer_for_activation}...")
        build_ball_tree_first_parquet(config, layer=config.model.layer_alternative, 
                                       save_path=f"balltree_layer_{config.model.layer_alternative}.pkl",
                                       use_dual_tree=args.use_dual_tree)
        build_ball_tree_first_parquet(config, layer=config.model.layer_for_activation, 
                                       save_path=f"balltree_layer_{config.model.layer_for_activation}.pkl",
                                       use_dual_tree=args.use_dual_tree)
    elif args.mode == 2:
        print(f"Building BallTree on all parquets for layer {config.model.layer_for_activation}...")
        build_ball_tree_all_parquets(config, layer=config.model.layer_for_activation, 
                                      save_path=f"balltree_layer_{config.model.layer_for_activation}_all_parquets.pkl",
                                      use_dual_tree=args.use_dual_tree)
    elif args.mode == 3:
        print(f"Building BallTree on all parquets for layer {config.model.layer_alternative}...")
        build_ball_tree_all_parquets(config, layer=config.model.layer_alternative, 
                                      save_path=f"balltree_layer_{config.model.layer_alternative}_all_parquets.pkl",
                                      use_dual_tree=args.use_dual_tree)