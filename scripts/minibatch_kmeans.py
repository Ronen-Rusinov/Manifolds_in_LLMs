"""
Script to load all activations, run MiniBatchKMeans clustering, and save the model.
"""

import sys
import os
from pathlib import Path
import numpy as np
import joblib
import time
from sklearn.cluster import MiniBatchKMeans

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.load_data import load_all_parquets

def main():
    print("Loading all activations...")
    start_time = time.time()
    df = load_all_parquets(timing=True)
    print(f"Total time to load: {time.time() - start_time:.2f}s")
    print(f"DataFrame shape: {df.shape}")
    
    # Extract activations from layer 18
    print("\nExtracting activation vectors...")
    activations = np.array(df['activation_layer_18'].tolist(), dtype=np.float32)
    print(f"Activations shape: {activations.shape}")
    
    # Run MiniBatchKMeans
    print("\nRunning MiniBatchKMeans clustering...")
    print("Parameters:")
    print(f"  - batch_size: 150000")
    print(f"  - n_clusters: 200")
    print(f"  - random_state: 42")
    
    start_time = time.time()
    kmeans = MiniBatchKMeans(
        n_clusters=200,
        batch_size=150_000,
        random_state=42,
        verbose=10,
        n_init=10,
        max_iter=100
    )
    kmeans.fit(activations)
    fit_time = time.time() - start_time
    print(f"\nFitting completed in {fit_time:.2f}s")
    print(f"Inertia: {kmeans.inertia_:.4f}")
    
    #save centroids
    centroids_path = Path(__file__).parent.parent / "results" / "minibatch_kmeans" / "centroids.npy"
    #if directory doesn't exist, create it
    centroids_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(centroids_path, kmeans.cluster_centers_)
    print(f"Centroids saved to {centroids_path}")

    params = kmeans.get_params()
    print("\nModel Parameters:")
    for param, value in params.items():
        print(f"  - {param}: {value}")
    
    #save params
    output_dir = Path(__file__).parent.parent / "results" / "minibatch_kmeans"
    output_dir.mkdir(parents=True, exist_ok=True)
    params_path = output_dir / "kmeans_params.txt"
    with open(params_path, "w") as f:
        f.write("MiniBatchKMeans Parameters\n")
        f.write("=" * 30 + "\n")
        for param, value in params.items():
            f.write(f"{param}: {value}\n")
    print(f"Parameters saved to {params_path}")
    
    # Create a summary file
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("MiniBatchKMeans Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset Shape: {df.shape}\n")
        f.write(f"Activation Vector Shape: {activations.shape}\n")
        f.write(f"Activation Vector Dtype: {activations.dtype}\n\n")
        f.write("KMeans Parameters:\n")
        f.write(f"  - n_clusters: {kmeans.n_clusters}\n")
        f.write(f"  - batch_size: {kmeans.batch_size}\n")
        f.write(f"  - random_state: 42\n\n")
        f.write("Results:\n")
        f.write(f"  - Inertia: {kmeans.inertia_:.4f}\n")
        f.write(f"  - Fitting Time: {fit_time:.2f}s\n")
        f.write(f"  - Cluster Centers Shape: {kmeans.cluster_centers_.shape}\n")
    
    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
