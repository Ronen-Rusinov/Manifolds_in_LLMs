"""Transfer latent embeddings between centroid neighborhoods.

This module allows for the transfer of latent embeddings from the surrounding
of one centroid to the surrounding of another centroid, assuming that both
embeddings are isometric and there is sufficient overlap between train points.

It solves the orthogonal procrustes problem to find the optimal transformation,
allowing both points and directions to be transferred between neighborhoods.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from config_manager import load_config
from utils import common
from rigid_procrustes import impose_X_on_Y, euclidean_procrustes

# Valid embedding types
VALID_EMBEDDING_TYPES = {"isomap", "pca", "autoencoder"}


def load_embeddings_by_type(
    centroid_index: int,
    n_components: int,
    embedding_type: str = "isomap",
) -> np.ndarray:
    """Load embeddings for a specific centroid by type.
    
    Args:
        centroid_index: Index of centroid
        n_components: Number of dimensions
        embedding_type: Type of embeddings ("isomap", "pca", "autoencoder")
    
    Returns:
        Array of embeddings
    
    Raises:
        ValueError: If embedding_type is not recognized
    """
    embedding_type = embedding_type.lower()
    if embedding_type not in VALID_EMBEDDING_TYPES:
        raise ValueError(
            f"Unknown embedding_type '{embedding_type}'. "
            f"Must be one of: {', '.join(VALID_EMBEDDING_TYPES)}"
        )
    
    if embedding_type == "isomap":
        return common.load_isomap_embeddings(centroid_index, n_components)
    elif embedding_type == "pca":
        return common.load_pca_embeddings(centroid_index, n_components)
    elif embedding_type == "autoencoder":
        return common.load_autoencoder_embeddings(centroid_index, n_components)


def batch_load_embeddings_by_type(
    n_centroids: int,
    n_components: int,
    embedding_type: str = "isomap",
) -> dict:
    """Load embeddings for all centroids by type.
    
    Args:
        n_centroids: Total number of centroids
        n_components: Number of dimensions
        embedding_type: Type of embeddings ("isomap", "pca", "autoencoder")
    
    Returns:
        Dictionary mapping centroid_index -> embeddings array
    
    Raises:
        ValueError: If embedding_type is not recognized
    """
    embedding_type = embedding_type.lower()
    if embedding_type not in VALID_EMBEDDING_TYPES:
        raise ValueError(
            f"Unknown embedding_type '{embedding_type}'. "
            f"Must be one of: {', '.join(VALID_EMBEDDING_TYPES)}"
        )
    
    if embedding_type == "isomap":
        return common.batch_load_isomap_embeddings(n_centroids, n_components)
    elif embedding_type == "pca":
        return common.batch_load_pca_embeddings(n_centroids, n_components)
    elif embedding_type == "autoencoder":
        return common.batch_load_autoencoder_embeddings(n_centroids, n_components)


def transfer_embedding_between_centroids(
    centroid_i: int,
    centroid_j: int,
    embedding_point: np.ndarray,
    neighbor_indices: np.ndarray,
    embeddings_i: np.ndarray,
    embeddings_j: np.ndarray,
    direction_vector: Optional[np.ndarray] = None,
    embedding_type: str = "isomap",
    config=None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Transfer an embedding point from one centroid's neighborhood to another's.
    
    Given an embedding point in the neighborhood of centroid_i, find the optimal
    transformation to map it to the corresponding neighborhood of centroid_j using
    procrustes analysis on common neighbor points.
    
    Args:
        centroid_i: Index of source centroid
        centroid_j: Index of target centroid
        embedding_point: The embedding point to transfer (shape: n_components,)
        neighbor_indices: Array of shape (n_centroids, k) with neighbor indices
        embeddings_i: Embeddings for centroid_i (shape: k, n_components)
        embeddings_j: Embeddings for centroid_j (shape: k, n_components)
        direction_vector: Optional direction vector to transform (shape: n_components,)
        embedding_type: Type of embeddings ("isomap", "pca", "autoencoder")
        config: Configuration object (optional, used for validation thresholds)
    
    Returns:
        Tuple of (transferred_point, transferred_direction)
        - transferred_point: The point transformed to centroid_j's space
        - transferred_direction: The direction transformed (None if not provided)
    
    Raises:
        ValueError: If overlap is insufficient for reliable transformation or invalid embedding_type
    """
    if config is None:
        config = load_config()
    
    # Validate embedding type
    embedding_type = embedding_type.lower()
    if embedding_type not in VALID_EMBEDDING_TYPES:
        raise ValueError(
            f"Unknown embedding_type '{embedding_type}'. "
            f"Must be one of: {', '.join(VALID_EMBEDDING_TYPES)}"
        )
    
    n_components = embedding_point.shape[0]
    
    # Get neighbor indices for both centroids
    indices_i = neighbor_indices[centroid_i]
    indices_j = neighbor_indices[centroid_j]
    
    # Find common indices in global space
    indices_i_set = set(indices_i)
    indices_j_set = set(indices_j)
    common_indices = list(indices_i_set.intersection(indices_j_set))
    
    # Check if overlap is sufficient
    min_required_overlap = max(n_components // 2, 5)  # At least n_components/2 or 5, whichever is larger
    if len(common_indices) <= min_required_overlap:
        raise ValueError(
            f"Insufficient overlap between centroids {centroid_i} and {centroid_j}. "
            f"Found {len(common_indices)} common neighbors, but need at least {min_required_overlap}."
        )
    
    print(
        f"[{datetime.now()}] Transfer {centroid_i} -> {centroid_j} ({embedding_type}): "
        f"Found {len(common_indices)} common neighbors (need >{min_required_overlap})",
        flush=True
    )
    
    # Map global indices to local indices in each neighborhood
    local_indices_i = [np.where(indices_i == common_idx)[0][0] for common_idx in common_indices]
    local_indices_j = [np.where(indices_j == common_idx)[0][0] for common_idx in common_indices]
    
    # Extract embeddings of common points
    common_embeddings_i = embeddings_i[local_indices_i]
    common_embeddings_j = embeddings_j[local_indices_j]
    
    # Apply procrustes analysis to find the optimal transformation
    # impose_X_on_Y finds the transformation that aligns X (source) to Y (target)
    aligned_embeddings_j = impose_X_on_Y(
        common_embeddings_i.T,  # shape: (n_components, n_common)
        common_embeddings_j.T   # shape: (n_components, n_common)
    )
    
    # Compute the rotation matrix and translation from euclidean procrustes
    common_embeddings_i_T = common_embeddings_i.T
    common_embeddings_j_T = common_embeddings_j.T
    R, t = euclidean_procrustes(common_embeddings_i_T, common_embeddings_j_T)
    
    # Transform the embedding point from centroid_i's space to centroid_j's space
    transferred_point = R @ embedding_point + t
    
    # Transform the direction vector if provided
    transferred_direction = None
    if direction_vector is not None:
        # For direction vectors, we only apply the rotation (not translation)
        transferred_direction = R @ direction_vector
    
    # Validation: check alignment quality
    alignment_error = np.linalg.norm(aligned_embeddings_j - common_embeddings_j_T, axis=0).mean()
    print(
        f"[{datetime.now()}] Transfer {centroid_i} -> {centroid_j} ({embedding_type}): "
        f"Mean alignment error: {alignment_error:.6f}",
        flush=True
    )
    
    if alignment_error > 50.0:  # Warn if alignment error is very high. 50 does not seem low, but given as most of the time those are 50 dimensional
    #embeddings, it's fine.
        print(
            f"[{datetime.now()}] WARNING: High alignment error ({alignment_error:.6f}) "
            f"for centroids {centroid_i} -> {centroid_j} ({embedding_type}). Transfer may be unreliable.",
            flush=True
        )
    
    return transferred_point, transferred_direction


def batch_transfer_embeddings(
    source_centroid: int,
    target_centroid: int,
    source_embeddings: np.ndarray,
    neighbor_indices: np.ndarray,
    embeddings_i: np.ndarray,
    embeddings_j: np.ndarray,
    source_directions: Optional[np.ndarray] = None,
    embedding_type: str = "isomap",
    config=None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Transfer multiple embedding points from one centroid to another.
    
    Args:
        source_centroid: Index of source centroid
        target_centroid: Index of target centroid
        source_embeddings: Array of embedding points (shape: n_points, n_components)
        neighbor_indices: Array of shape (n_centroids, k) with neighbor indices
        embeddings_i: Embeddings for source centroid
        embeddings_j: Embeddings for target centroid
        source_directions: Optional array of direction vectors
        embedding_type: Type of embeddings ("isomap", "pca", "autoencoder")
        config: Configuration object
    
    Returns:
        Tuple of (transferred_embeddings, transferred_directions)
    """
    n_points = source_embeddings.shape[0]
    n_components = source_embeddings.shape[1]
    
    transferred_embeddings = np.zeros_like(source_embeddings)
    transferred_directions = None
    
    if source_directions is not None:
        transferred_directions = np.zeros_like(source_directions)
    
    for point_idx in range(n_points):
        point, direction = transfer_embedding_between_centroids(
            source_centroid,
            target_centroid,
            source_embeddings[point_idx],
            neighbor_indices,
            embeddings_i,
            embeddings_j,
            direction_vector=source_directions[point_idx] if source_directions is not None else None,
            embedding_type=embedding_type,
            config=config,
        )
        
        transferred_embeddings[point_idx] = point
        if transferred_directions is not None:
            transferred_directions[point_idx] = direction
    
    return transferred_embeddings, transferred_directions


def main():
    """Example usage of transfer_latent_embeddings functionality."""
    import argparse
    
    config = load_config()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Transfer embeddings between centroids")
    parser.add_argument(
        "--embedding-type",
        type=str,
        default="isomap",
        choices=list(VALID_EMBEDDING_TYPES),
        help="Type of embeddings to use (default: isomap)"
    )
    parser.add_argument(
        "--source-centroid",
        type=int,
        default=0,
        help="Index of source centroid (default: 0)"
    )
    parser.add_argument(
        "--target-centroid",
        type=int,
        default=1,
        help="Index of target centroid (default: 1)"
    )
    args = parser.parse_args()
    
    embedding_type = args.embedding_type
    centroid_i = args.source_centroid
    centroid_j = args.target_centroid
    
    # Load necessary data
    print(f"[{datetime.now()}] Loading data...", flush=True)
    print(f"[{datetime.now()}] Using embedding type: {embedding_type}", flush=True)
    
    centroids = common.load_centroids(f"minibatch_kmeans_{config.clustering.n_centroids}")
    neighbor_indices = common.load_neighbor_indices(
        f'nearest_{config.clustering.k_nearest_large}_neighbors_indices_layer_{config.model.layer_for_activation}_n_centroids_{config.clustering.n_centroids}.npy'
    )
    
    # Load embeddings based on type
    try:
        all_embeddings = batch_load_embeddings_by_type(
            config.clustering.n_centroids,
            config.dimensionality.n_components,
            embedding_type=embedding_type
        )
    except ValueError as e:
        print(f"[{datetime.now()}] Failed to load embeddings: {e}", flush=True)
        return
    
    print(f"[{datetime.now()}] Transferring embeddings from centroid {centroid_i} to {centroid_j}...", flush=True)
    
    embeddings_i = all_embeddings[centroid_i]
    embeddings_j = all_embeddings[centroid_j]
    
    # Take a sample point from the first centroid's neighborhood
    sample_point = embeddings_i[0:1]  # Keep as 2D array
    sample_direction = np.random.randn(config.dimensionality.n_components)
    
    try:
        transferred_point, transferred_direction = transfer_embedding_between_centroids(
            centroid_i,
            centroid_j,
            sample_point[0],
            neighbor_indices,
            embeddings_i,
            embeddings_j,
            direction_vector=sample_direction,
            embedding_type=embedding_type,
            config=config,
        )
        
        print(f"[{datetime.now()}] Transfer successful!", flush=True)
        print(f"[{datetime.now()}] Embedding type: {embedding_type}", flush=True)
        print(f"[{datetime.now()}] Original point shape: {sample_point[0].shape}", flush=True)
        print(f"[{datetime.now()}] Transferred point shape: {transferred_point.shape}", flush=True)
        if transferred_direction is not None:
            print(f"[{datetime.now()}] Transferred direction shape: {transferred_direction.shape}", flush=True)
    
    except ValueError as e:
        print(f"[{datetime.now()}] Transfer failed: {e}", flush=True)


if __name__ == "__main__":
    main()