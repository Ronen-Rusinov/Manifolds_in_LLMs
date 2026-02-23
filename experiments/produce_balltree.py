import os
import sys
import time
from datetime import datetime

#add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from utils import load_data
from sklearn.neighbors import BallTree
import numpy as np
import pickle

def test_ball_tree_first_parquet(layer=18, save_path=None):
    #load first parquet
    print(f"[{datetime.now()}] loading first parquet...")
    df = load_data.load_first_parquet(timing=False)
    #get activations
    print(f"[{datetime.now()}] Obtaining activations...")
    activations = np.array(df[f"activation_layer_{layer}"].tolist(), dtype=np.float16)

    print(f"[{datetime.now()}] Building BallTree...")
    start_time = time.time()
    tree = BallTree(activations, leaf_size=40)
    build_time = time.time() - start_time
    print(f"[{datetime.now()}] BallTree built in {build_time:.2f} seconds.")

    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(tree, f)
        print(f"[{datetime.now()}] BallTree saved to {save_path}.")
            

def test_ball_tree_all_parquets(layer=18, save_path=None):
    #load all parquets
    print(f"[{datetime.now()}] loading all parquets...")
    df = load_data.load_all_parquets(timing=False)
    #get activations
    print(f"[{datetime.now()}] Obtaining activations...")
    activations = np.array(df[f"activation_layer_{layer}"].tolist(), dtype=np.float16)

    print(f"[{datetime.now()}] Building BallTree...")
    start_time = time.time()
    tree = BallTree(activations, leaf_size=40)
    build_time = time.time() - start_time
    print(f"[{datetime.now()}] BallTree built in {build_time:.2f} seconds.")

    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(tree, f)
        print(f"[{datetime.now()}] BallTree saved to {save_path}.")
            
if __name__ == "__main__":
    #if the number 1 is passed as an arg, test layers 6 and 18 on the first parquet
    #if the number 2 is passed as an arg, test layer 18 on all parquets
    #if the number 3 is passed as an arg, test layers 6 on all parquets
    if len(sys.argv) < 2:
        print("Please provide a test number (1, 2, or 3).")
        sys.exit(1)
    test_number = int(sys.argv[1])
    if test_number == 1:
        print("Testing BallTree on first parquet for layers 6 and 18...")
        test_ball_tree_first_parquet(layer=6, save_path=f"balltree_layer_6.pkl")
        test_ball_tree_first_parquet(layer=18, save_path=f"balltree_layer_18.pkl")
    elif test_number == 2:
        print("Testing BallTree on all parquets for layer 18...")
        test_ball_tree_all_parquets(layer=18, save_path=f"balltree_layer_18_all_parquets.pkl")
    elif test_number == 3:
        print("Testing BallTree on all parquets for layer 6...")
        test_ball_tree_all_parquets(layer=6, save_path=f"balltree_layer_6_all_parquets.pkl")
    else:
        print("Invalid test number. Please provide 1, 2, or 3.")
