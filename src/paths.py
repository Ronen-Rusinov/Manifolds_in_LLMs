"""Centralized path management for Manifolds in LLMs project.

This module provides consistent path resolution across all scripts,
handling relative/absolute paths and environment variable overrides.
"""

import os
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory.
    
    Returns:
        Path to project root (parent of src directory)
    """
    return Path(__file__).parent.parent


def get_results_dir() -> Path:
    """Get the results directory.
    
    Returns:
        Path to results directory
    """
    return get_project_root() / "results"


def get_data_dir() -> Path:
    """Get the data directory.
    
    Returns:
        Path to data directory
    """
    project_root = get_project_root()
    return project_root / "data" / "activations_data"


def get_centroids_path(centroid_dir: str = "minibatch_kmeans_200") -> Path:
    """Get path to centroids file.
    
    Args:
        centroid_dir: Directory name within results (default: minibatch_kmeans_200)
    
    Returns:
        Path to centroids.npy file
    """
    
    centroid_count = ''.join(filter(str.isdigit, centroid_dir))
        
    return get_results_dir() / centroid_dir / f"centroids_{centroid_count}.npy"


def get_neighbor_indices_path(indices_file: str = "nearest_10000_neighbors_indices_layer_18_n_centroids_200.npy") -> Path:
    """Get path to nearest neighbor indices file.
    
    Args:
        indices_file: Filename of indices f'nearest_{k_nearest}_neighbors_indices_layer_{config.model.layer_for_activation}_n_centroids_{config.clustering.n_centroids}.npy

    Returns:
        Path to neighbor indices file
    """

    return get_results_dir() / "Balltree" / indices_file


def get_isomap_embeddings_path(centroid_index: int, n_components: int) -> Path:
    """Get path to Isomap embeddings for a specific centroid.
    
    Args:
        centroid_index: Index of centroid (0-based)
        n_components: Number of dimensions for embeddings
    
    Returns:
        Path to embeddings npy file
    """
    centroid_index_str = f"{centroid_index:04d}"
    return (
        get_results_dir() / f"iso_atlas_{n_components}D" / f"{n_components}D" /
        f"centroid_{centroid_index_str}_embeddings_{n_components}D.npy"
    )


def get_isomap_3d_embeddings_path(centroid_index: int) -> Path:
    """Get path to 3D Isomap embeddings for visualization.
    
    Args:
        centroid_index: Index of centroid (0-based)
    
    Returns:
        Path to 3D embeddings npy file
    """
    centroid_index_str = f"{centroid_index:04d}"
    return (
        get_results_dir() / f"iso_atlas_3D" / "3D" /
        f"centroid_{centroid_index_str}_embeddings_3D.npy"
    )


def get_isomap_4d_embeddings_path(centroid_index: int) -> Path:
    """Get path to 4D Isomap embeddings for visualization.
    
    Args:
        centroid_index: Index of centroid (0-based)
    
    Returns:
        Path to 4D embeddings npy file
    """
    centroid_index_str = f"{centroid_index:04d}"
    return (
        get_results_dir() / f"iso_atlas_4D" / "4D" /
        f"centroid_{centroid_index_str}_embeddings_4D.npy"
    )


def get_mapping_alignment_dir() -> Path:
    """Get directory for mapping alignment results.
    
    Returns:
        Path to mapping_alignment directory
    """
    return get_results_dir() / "mapping_alignment"


def get_autoencoder_dir() -> Path:
    """Get directory for autoencoder results.
    
    Returns:
        Path to autoencoder directory
    """
    return get_results_dir() / "autoencoder"


def get_isomap_12d_dir() -> Path:
    """Get directory for 12D Isomap results.
    
    Returns:
        Path to isomap_12D directory
    """
    return get_results_dir() / "isomap_12D"
