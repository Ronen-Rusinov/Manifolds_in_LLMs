"""Example usage of AtlasLoader for embedding and reconstruction operations.

This script demonstrates:
1. Loading centroids
2. Finding K nearest centroids to a query point
3. Embedding points using trained models (PCA, Isomap, Autoencoder)
4. Reconstructing original representations from embeddings
"""

import sys
import os
from pathlib import Path
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from atlas_loader import AtlasLoader
from config_manager import load_config
from utils.common import load_activations
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate AtlasLoader functionality"
    )
    parser.add_argument(
        "--centroid-idx",
        type=int,
        default=0,
        help="Index of centroid to use for embedding/reconstruction (default: 0)"
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=50,
        help="Number of components for dimensionality reduction (default: 50)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["pca", "isomap", "autoencoder"],
        default="pca",
        help="Embedding method to use (default: pca)"
    )
    parser.add_argument(
        "--n-test-points",
        type=int,
        default=10,
        help="Number of random test points to use (default: 10)"
    )
    parser.add_argument(
        "--k-nearest",
        type=int,
        default=5,
        help="Number of nearest centroids to find (default: 5)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for autoencoder inference (default: cpu)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file"
    )
    
    args = parser.parse_args()
    
    print(f"\n[{datetime.now()}] ========================================")
    print(f"[{datetime.now()}] AtlasLoader Example")
    print(f"[{datetime.now()}] ========================================\n")
    
    # Load config
    config = load_config(args.config)
    
    # Initialize loader
    print(f"[{datetime.now()}] Initializing AtlasLoader...")
    loader = AtlasLoader(config, device=args.device)
    
    # ========================================================================
    # 1. Load centroids and demonstrate nearest centroid finding
    # ========================================================================
    print(f"\n[{datetime.now()}] Step 1: Loading centroids and finding K nearest")
    print(f"[{datetime.now()}] " + "="*60)
    
    centroids = loader.load_centroids()
    print(f"[{datetime.now()}] Loaded {centroids.shape[0]} centroids with dimension {centroids.shape[1]}")
    
    # Test centroid finding with multiple query points
    query_centroid = centroids[args.centroid_idx]
    distances, indices = loader.get_nearest_centroids(query_centroid, k=args.k_nearest)
    
    print(f"\n[{datetime.now()}] Query point: Centroid {args.centroid_idx}")
    print(f"[{datetime.now()}] {args.k_nearest} Nearest centroids:")
    for i, (idx, dist) in enumerate(zip(indices, distances), 1):
        print(f"  {i}. Centroid {idx}: distance = {dist:.6f}")
    
    # ========================================================================
    # 2. Load test activation data and select random samples
    # ========================================================================
    print(f"\n[{datetime.now()}] Step 2: Loading test activation data")
    print(f"[{datetime.now()}] " + "="*60)
    
    try:
        activations = load_activations(config=config)
        print(f"[{datetime.now()}] Loaded {activations.shape[0]} activation vectors with dimension {activations.shape[1]}")
        
        # Select random test points
        n_test = min(args.n_test_points, activations.shape[0])
        test_indices = np.random.choice(activations.shape[0], n_test, replace=False)
        test_points = activations[test_indices]
        
        print(f"[{datetime.now()}] Selected {n_test} random test points")
        
    except Exception as e:
        print(f"[{datetime.now()}] Warning: Could not load activations ({e})")
        print(f"[{datetime.now()}] Creating synthetic test data instead...")
        
        # Create synthetic test data
        input_dim = centroids.shape[1]
        n_test = min(args.n_test_points, 100)
        test_points = np.random.randn(n_test, input_dim).astype(np.float32)
        print(f"[{datetime.now()}] Created {n_test} synthetic test points with dimension {input_dim}")
    
    # ========================================================================
    # 3. Demonstrate embedding and reconstruction
    # ========================================================================
    method = args.method.lower()
    n_comp = args.n_components
    centroid_idx = args.centroid_idx
    
    print(f"\n[{datetime.now()}] Step 3: Demonstrating {method.upper()} embedding and reconstruction")
    print(f"[{datetime.now()}] " + "="*60)
    print(f"[{datetime.now()}] Method: {method}")
    print(f"[{datetime.now()}] N Components: {n_comp}")
    print(f"[{datetime.now()}] Centroid Index: {centroid_idx}")
    print(f"[{datetime.now()}] Test Points Shape: {test_points.shape}")
    
    try:
        # Embedding phase
        print(f"\n[{datetime.now()}] Embedding {n_test} points using {method}...")
        embeddings = loader.embed(test_points, centroid_idx, method=method, n_components=n_comp)
        print(f"[{datetime.now()}] Embeddings shape: {embeddings.shape}")
        print(f"[{datetime.now()}] Embeddings stats - Min: {embeddings.min():.6f}, Max: {embeddings.max():.6f}, Mean: {embeddings.mean():.6f}")
        
        # Reconstruction phase
        print(f"\n[{datetime.now()}] Reconstructing from {method} embeddings...")
        
        # For Isomap, we might need training points for better reconstruction
        training_points = None
        if method == "isomap":
            print(f"[{datetime.now()}] Note: Using test points as 'training' reference for Isomap reconstruction")
            training_points = test_points
        
        reconstructed = loader.reconstruct(
            embeddings,
            centroid_idx,
            method=method,
            n_components=n_comp,
            training_points=training_points
        )
        print(f"[{datetime.now()}] Reconstructed shape: {reconstructed.shape}")
        print(f"[{datetime.now()}] Reconstructed stats - Min: {reconstructed.min():.6f}, Max: {reconstructed.max():.6f}, Mean: {reconstructed.mean():.6f}")
        
        # Compute reconstruction error
        reconstruction_error = np.mean(np.linalg.norm(test_points - reconstructed, axis=1))
        print(f"\n[{datetime.now()}] Reconstruction Error (Mean L2 distance): {reconstruction_error:.6f}")
        
        # Show per-sample statistics
        print(f"\n[{datetime.now()}] Per-sample reconstruction errors:")
        per_sample_errors = np.linalg.norm(test_points - reconstructed, axis=1)
        print(f"  Mean: {per_sample_errors.mean():.6f}")
        print(f"  Std:  {per_sample_errors.std():.6f}")
        print(f"  Min:  {per_sample_errors.min():.6f}")
        print(f"  Max:  {per_sample_errors.max():.6f}")
        
    except FileNotFoundError as e:
        print(f"\n[{datetime.now()}] ERROR: {e}")
        print(f"[{datetime.now()}] Make sure you've trained models for:")
        print(f"  - Method: {method}")
        print(f"  - N Components: {n_comp}")
        print(f"  - Centroid Index: {centroid_idx}")
    
    # ========================================================================
    # 4. Demonstrate loading models directly
    # ========================================================================
    print(f"\n[{datetime.now()}] Step 4: Loading models directly")
    print(f"[{datetime.now()}] " + "="*60)
    
    try:
        if method == "pca":
            model = loader.load_pca_model(centroid_idx, n_comp)
            print(f"[{datetime.now()}] Successfully loaded PCA model for centroid {centroid_idx}")
            print(f"[{datetime.now()}] Model type: {type(model)}")
            print(f"[{datetime.now()}] Explained variance ratio: {model.explained_variance_ratio_.sum():.6f}")
        
        elif method == "isomap":
            model = loader.load_isomap_model(centroid_idx, n_comp)
            print(f"[{datetime.now()}] Successfully loaded Isomap model for centroid {centroid_idx}")
            print(f"[{datetime.now()}] Model type: {type(model)}")
        
        elif method == "autoencoder":
            model = loader.load_autoencoder_model(centroid_idx, n_comp)
            print(f"[{datetime.now()}] Successfully loaded Autoencoder model for centroid {centroid_idx}")
            print(f"[{datetime.now()}] Model type: {type(model)}")
            print(f"[{datetime.now()}] Model device: {next(model.parameters()).device}")
    
    except FileNotFoundError as e:
        print(f"[{datetime.now()}] Warning: {e}")
    
    # ========================================================================
    # 5. Summary
    # ========================================================================
    print(f"\n[{datetime.now()}] ========================================")
    print(f"[{datetime.now()}] Example completed successfully!")
    print(f"[{datetime.now()}] ========================================\n")


if __name__ == "__main__":
    main()
