"""Shared utility functions for common data loading operations.

This module centralizes loading functions used across multiple scripts
to reduce code duplication and ensure consistency.
"""

from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import joblib

from utils.load_data import load_all_parquets
from config_manager import load_config
import paths


def load_centroids(centroid_dir: str = "minibatch_kmeans") -> np.ndarray:
    """Load centroids from minibatch_kmeans results.
    
    Args:
        centroid_dir: Directory name within results containing centroids (default: minibatch_kmeans)
    
    Returns:
        Array of centroid vectors with shape (n_centroids, n_features)
    
    Raises:
        FileNotFoundError: If centroids file does not exist
    """
    centroids_path = paths.get_centroids_path(centroid_dir)
    if not centroids_path.exists():
        raise FileNotFoundError(f"Centroids file not found at {centroids_path}")
    
    print(f"[{datetime.now()}] Loading centroids from {centroids_path}...", flush=True)
    centroids = np.load(centroids_path)
    print(f"[{datetime.now()}] Centroids loaded. Shape: {centroids.shape}", flush=True)
    return centroids


def load_neighbor_indices(indices_file: str = "nearest_neighbors_indices_1.npy") -> np.ndarray:
    """Load precomputed nearest neighbor indices.
    
    Args:
        indices_file: Filename of indices (default: nearest_neighbors_indices_1.npy)
    
    Returns:
        Array of neighbor indices with shape (n_centroids, k_neighbors)
    
    Raises:
        FileNotFoundError: If indices file does not exist
    """
    indices_path = paths.get_neighbor_indices_path(indices_file)
    if not indices_path.exists():
        raise FileNotFoundError(f"Neighbor indices file not found at {indices_path}")
    
    print(f"[{datetime.now()}] Loading nearest neighbor indices from {indices_path}...", flush=True)
    neighbor_indices = np.load(indices_path)
    print(f"[{datetime.now()}] Neighbor indices loaded. Shape: {neighbor_indices.shape}", flush=True)
    return neighbor_indices


def load_activations(config=None, layer: Optional[int] = None) -> np.ndarray:
    """Load all activation vectors for specified layer.
    
    Args:
        config: Config object. If None, loads default config. Used to get layer_for_activation.
        layer: Override layer index. If provided, uses this instead of config value.
    
    Returns:
        Array of activations with shape (n_samples, activation_dim)
    
    Raises:
        ValueError: If layer is not specified and config doesn't have it
    """
    if config is None:
        config = load_config()
    
    if layer is None:
        layer = config.model.layer_for_activation
    
    print(f"[{datetime.now()}] Loading all activations from layer {layer}...", flush=True)
    df = load_all_parquets(timing=True)
    
    column_name = f'activation_layer_{layer}'
    if column_name not in df.columns:
        raise ValueError(
            f"Column '{column_name}' not found in data. "
            f"Available columns: {list(df.columns)}"
        )
    
    activations = np.array(df[column_name].tolist(), dtype=np.float32)
    print(f"[{datetime.now()}] Activations loaded. Shape: {activations.shape}, dtype: {activations.dtype}", flush=True)
    return activations


def load_activations_with_prompts(
    config=None, 
    layer: Optional[int] = None,
    prompt_column: str = 'text_prefix'
) -> Tuple[np.ndarray, List[str]]:
    """Load activation vectors and corresponding prompts.
    
    Args:
        config: Config object. If None, loads default config. Used to get layer_for_activation.
        layer: Override layer index. If provided, uses this instead of config value.
        prompt_column: Name of column containing prompts (default: text_prefix)
    
    Returns:
        Tuple of (activations array, prompts list)
    
    Raises:
        ValueError: If layer is not specified and config doesn't have it
    """
    if config is None:
        config = load_config()
    
    if layer is None:
        layer = config.model.layer_for_activation
    
    print(f"[{datetime.now()}] Loading activations and prompts from layer {layer}...", flush=True)
    df = load_all_parquets(timing=True)
    
    column_name = f'activation_layer_{layer}'
    if column_name not in df.columns:
        raise ValueError(
            f"Column '{column_name}' not found in data. "
            f"Available columns: {list(df.columns)}"
        )
    
    activations = np.array(df[column_name].tolist(), dtype=np.float32)
    prompts = df[prompt_column].tolist() if prompt_column in df.columns else [None] * len(df)
    
    print(f"[{datetime.now()}] Activations loaded. Shape: {activations.shape}, dtype: {activations.dtype}", flush=True)
    print(f"[{datetime.now()}] Prompts loaded. Count: {len(prompts)}", flush=True)
    return activations, prompts


def load_isomap_embeddings(centroid_index: int, n_components: int) -> np.ndarray:
    """Load Isomap embeddings for a specific centroid.
    
    Args:
        centroid_index: Index of centroid (0-based)
        n_components: Number of dimensions for embeddings
    
    Returns:
        Array of embeddings with shape (n_neighborhood_samples, n_components)
    
    Raises:
        FileNotFoundError: If embeddings file does not exist
    """
    embeddings_path = paths.get_isomap_embeddings_path(centroid_index, n_components)
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Isomap embeddings file not found at {embeddings_path}")
    
    print(f"[{datetime.now()}] Loading Isomap embeddings for centroid {centroid_index} from {embeddings_path}...", flush=True)
    embeddings = np.load(embeddings_path)
    print(f"[{datetime.now()}] Isomap embeddings loaded. Shape: {embeddings.shape}", flush=True)
    return embeddings


def batch_load_isomap_embeddings(n_centroids: int, n_components: int) -> dict:
    """Load Isomap embeddings for all centroids.
    
    Useful for avoiding repeated disk access during pairwise comparisons.
    
    Args:
        n_centroids: Total number of centroids to load
        n_components: Number of dimensions for embeddings
    
    Returns:
        Dictionary mapping centroid_index -> embeddings array
    
    Raises:
        FileNotFoundError: If any embeddings file does not exist
    """
    print(f"[{datetime.now()}] Preloading all {n_centroids} embeddings into memory...", flush=True)
    all_embeddings = {}
    
    for i in range(n_centroids):
        try:
            embeddings_path = paths.get_isomap_embeddings_path(i, n_components)
            all_embeddings[i] = np.load(embeddings_path)
            if (i + 1) % 50 == 0:
                print(f"[{datetime.now()}] Loaded {i + 1}/{n_centroids} embeddings", flush=True)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Missing embeddings for centroid {i} at {embeddings_path}. "
                f"Run isomap_for_each_centroid script for all centroids first."
            )
    
    print(f"[{datetime.now()}] All embeddings preloaded successfully", flush=True)
    return all_embeddings


def validate_data_consistency(
    centroids: np.ndarray,
    neighbor_indices: np.ndarray,
    activations: np.ndarray,
    prompts: Optional[List[str]] = None
) -> bool:
    """Validate consistency between loaded data arrays.
    
    Args:
        centroids: Centroids array
        neighbor_indices: Neighbor indices array
        activations: Activations array
        prompts: Optional list of prompts
    
    Returns:
        True if all data is consistent
    
    Raises:
        ValueError: If any consistency check fails
    """
    print(f"[{datetime.now()}] Validating data consistency...", flush=True)
    
    # Check centroid-indices consistency
    if centroids.shape[0] != neighbor_indices.shape[0]:
        raise ValueError(
            f"Number of centroids ({centroids.shape[0]}) does not match "
            f"number of neighbor indices ({neighbor_indices.shape[0]})"
        )
    
    # Check activations-prompts consistency
    if prompts is not None and activations.shape[0] != len(prompts):
        raise ValueError(
            f"Number of activations ({activations.shape[0]}) does not match "
            f"number of prompts ({len(prompts)})"
        )
    
    print(f"[{datetime.now()}] ✓ Centroids shape: {centroids.shape}", flush=True)
    print(f"[{datetime.now()}] ✓ Neighbor indices shape: {neighbor_indices.shape}", flush=True)
    print(f"[{datetime.now()}] ✓ Activations shape: {activations.shape}", flush=True)
    if prompts is not None:
        print(f"[{datetime.now()}] ✓ Prompts count: {len(prompts)}", flush=True)
    
    return True
