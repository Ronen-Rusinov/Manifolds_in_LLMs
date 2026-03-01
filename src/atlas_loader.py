"""Atlas Loader: Load trained dimensionality reduction models and apply embeddings.

This module provides functionality to:
1. Load trained PCA, Isomap, and Autoencoder models from disk
2. Find K nearest centroids to a query point
3. Apply embeddings using trained models
4. Reconstruct ambient representations from embeddings
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Union, List
import warnings

import numpy as np
import torch
import joblib
from sklearn.neighbors import KDTree

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config_manager import Config, load_config
import paths


class AtlasLoader:
    """Load and manage trained dimensionality reduction models for data atlases."""
    
    def __init__(self, config: Optional[Config] = None, device: str = 'cpu'):
        """Initialize the AtlasLoader.
        
        Args:
            config: Config object. If None, loads default config.
            device: Device for autoencoder inference ('cpu' or 'cuda')
        """
        if config is None:
            config = load_config()
        
        self.config = config
        self.device = device
        self._centroids = None
        self._kdtree = None
        self._model_cache = {}  # Cache loaded models to avoid reloading
        
    def load_centroids(self, n_centroids: Optional[int] = None) -> np.ndarray:
        """Load centroid vectors into memory.
        
        Args:
            n_centroids: Number of centroids. If None, uses config value.
        
        Returns:
            Array of centroids with shape (n_centroids, embedding_dim)
        """
        if n_centroids is None:
            n_centroids = self.config.clustering.n_centroids
        
        if self._centroids is None:
            centroid_dir = f"minibatch_kmeans_{n_centroids}"
            centroids_path = paths.get_centroids_path(centroid_dir)
            
            if not centroids_path.exists():
                raise FileNotFoundError(f"Centroids not found at {centroids_path}")
            
            print(f"[{datetime.now()}] Loading centroids from {centroids_path}...", flush=True)
            self._centroids = np.load(centroids_path)
            print(f"[{datetime.now()}] Loaded centroids with shape {self._centroids.shape}", flush=True)
        
        return self._centroids
    
    def _build_kdtree(self) -> KDTree:
        """Build KDTree for nearest neighbor queries.
        
        Returns:
            KDTree built on centroids
        """
        if self._kdtree is None:
            centroids = self.load_centroids()
            print(f"[{datetime.now()}] Building KDTree on {centroids.shape[0]} centroids...", flush=True)
            self._kdtree = KDTree(centroids, leaf_size=self.config.model.balltree_leaf_size)
            print(f"[{datetime.now()}] KDTree built successfully", flush=True)
        
        return self._kdtree
    
    def get_nearest_centroids(
        self, 
        query_point: np.ndarray, 
        k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find K nearest centroids to a query point.
        
        Args:
            query_point: Query vector with shape (embedding_dim,) or (1, embedding_dim)
            k: Number of nearest centroids to find
        
        Returns:
            Tuple of (distances, indices) where:
                - distances: Array of shape (k,) with distances to nearest centroids
                - indices: Array of shape (k,) with indices of nearest centroids
        """
        if query_point.ndim == 1:
            query_point = query_point.reshape(1, -1)
        
        kdtree = self._build_kdtree()
        
        print(f"[{datetime.now()}] Finding {k} nearest centroids to query point...", flush=True)
        distances, indices = kdtree.query(query_point, k=k)
        
        # KDTree returns arrays with shape (n_queries, k), we want (k,) for single query
        distances = distances[0] if distances.ndim > 1 else distances
        indices = indices[0] if indices.ndim > 1 else indices
        
        print(f"[{datetime.now()}] Found nearest centroids with distances: {distances}", flush=True)
        
        return distances, indices
    
    def _get_model_cache_key(self, method: str, centroid_idx: int, n_components: int) -> str:
        """Generate cache key for model storage."""
        return f"{method}_{centroid_idx:04d}_{n_components}D"
    
    # ============================================================================
    # PCA Methods
    # ============================================================================
    
    def load_pca_model(self, centroid_idx: int, n_components: int) -> object:
        """Load a trained PCA model for a specific centroid.
        
        Args:
            centroid_idx: Index of the centroid (0-based)
            n_components: Dimensionality of the PCA model
        
        Returns:
            Trained PCA model (sklearn.decomposition.PCA)
        
        Raises:
            FileNotFoundError: If model file does not exist
        """
        cache_key = self._get_model_cache_key("pca", centroid_idx, n_components)
        
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        model_dir = paths.get_results_dir() / f"pca_atlas_{n_components}D" / f"{n_components}D"
        model_path = model_dir / f"centroid_{centroid_idx:04d}_pca_{n_components}D.joblib"
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"PCA model not found at {model_path}. "
                f"Make sure you've run pca_for_each_centroid.py with n_components={n_components}"
            )
        
        print(f"[{datetime.now()}] Loading PCA model from {model_path}...", flush=True)
        model = joblib.load(model_path)
        self._model_cache[cache_key] = model
        
        return model
    
    def embed_pca(
        self,
        points: np.ndarray,
        centroid_idx: int,
        n_components: int
    ) -> np.ndarray:
        """Apply PCA embedding to points using a trained model.
        
        Args:
            points: Input points with shape (n_points, input_dim)
            centroid_idx: Index of the centroid whose model to use
            n_components: Dimensionality of the PCA model
        
        Returns:
            Embedded points with shape (n_points, n_components)
        """
        model = self.load_pca_model(centroid_idx, n_components)
        
        print(f"[{datetime.now()}] Applying PCA embedding (centroid {centroid_idx}, {n_components}D)...", flush=True)
        embeddings = model.transform(points)
        
        print(f"[{datetime.now()}] PCA embeddings computed. Shape: {embeddings.shape}", flush=True)
        return embeddings
    
    def reconstruct_pca(
        self,
        embeddings: np.ndarray,
        centroid_idx: int,
        n_components: int
    ) -> np.ndarray:
        """Reconstruct ambient representation from PCA embeddings.
        
        Args:
            embeddings: Embedded points with shape (n_points, n_components)
            centroid_idx: Index of the centroid whose model to use
            n_components: Dimensionality of the PCA model
        
        Returns:
            Reconstructed points in ambient space with shape (n_points, input_dim)
        """
        model = self.load_pca_model(centroid_idx, n_components)
        
        print(f"[{datetime.now()}] Reconstructing from PCA embeddings (centroid {centroid_idx})...", flush=True)
        reconstructed = model.inverse_transform(embeddings)
        
        print(f"[{datetime.now()}] Reconstruction complete. Shape: {reconstructed.shape}", flush=True)
        return reconstructed
    
    # ============================================================================
    # Isomap Methods
    # ============================================================================
    
    def load_isomap_model(self, centroid_idx: int, n_components: int) -> object:
        """Load a trained Isomap model for a specific centroid.
        
        Args:
            centroid_idx: Index of the centroid (0-based)
            n_components: Dimensionality of the Isomap model
        
        Returns:
            Trained Isomap model (sklearn.manifold.Isomap)
        
        Raises:
            FileNotFoundError: If model file does not exist
        """
        cache_key = self._get_model_cache_key("isomap", centroid_idx, n_components)
        
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        model_dir = paths.get_results_dir() / f"iso_atlas_{n_components}D" / f"{n_components}D"
        model_path = model_dir / f"centroid_{centroid_idx:04d}_isomap_{n_components}D.joblib"
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Isomap model not found at {model_path}. "
                f"Make sure you've run isomap_for_each_centroid.py with n_components={n_components}"
            )
        
        print(f"[{datetime.now()}] Loading Isomap model from {model_path}...", flush=True)
        model = joblib.load(model_path)
        self._model_cache[cache_key] = model
        
        return model
    
    def embed_isomap(
        self,
        points: np.ndarray,
        centroid_idx: int,
        n_components: int
    ) -> np.ndarray:
        """Apply Isomap embedding to points using a trained model.
        
        Args:
            points: Input points with shape (n_points, input_dim)
            centroid_idx: Index of the centroid whose model to use
            n_components: Dimensionality of the Isomap model
        
        Returns:
            Embedded points with shape (n_points, n_components)
        """
        model = self.load_isomap_model(centroid_idx, n_components)
        
        print(f"[{datetime.now()}] Applying Isomap embedding (centroid {centroid_idx}, {n_components}D)...", flush=True)
        embeddings = model.transform(points)
        
        print(f"[{datetime.now()}] Isomap embeddings computed. Shape: {embeddings.shape}", flush=True)
        return embeddings
    
    def reconstruct_isomap(
        self,
        embeddings: np.ndarray,
        centroid_idx: int,
        n_components: int,
        training_points: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Reconstruct ambient representation from Isomap embeddings.
        
        Note: Isomap does not have an inverse transform like PCA. This function
        uses embedding interpolation based on training points if provided.
        
        Args:
            embeddings: Embedded points with shape (n_points, n_components)
            centroid_idx: Index of the centroid whose model to use
            n_components: Dimensionality of the Isomap model
            training_points: Original training points used to fit the model.
                            If provided, uses nearest neighbor interpolation for reconstruction.
                            If None, raises a warning and returns embeddings as-is.
        
        Returns:
            Reconstructed points in ambient space with shape (n_points, input_dim)
        """
        model = self.load_isomap_model(centroid_idx, n_components)
        
        if training_points is None:
            warnings.warn(
                "Isomap does not have a built-in inverse transform. "
                "To reconstruct, provide original training_points for interpolation. "
                "Returning embeddings as-is.",
                UserWarning
            )
            return embeddings
        
        print(f"[{datetime.now()}] Reconstructing from Isomap embeddings using interpolation...", flush=True)
        
        # Get training embeddings
        training_embeddings = model.transform(training_points)
        
        # Use KDTree to find nearest neighbors in embedding space
        kdtree_emb = KDTree(training_embeddings, leaf_size=30)
        distances, indices = kdtree_emb.query(embeddings, k=5)
        
        # Inverse distance weighting for reconstruction
        reconstructed = np.zeros((embeddings.shape[0], training_points.shape[1]))
        
        for i in range(embeddings.shape[0]):
            # Weight by inverse distance (add small epsilon to avoid division by zero)
            weights = 1.0 / (distances[i] + 1e-6)
            weights /= weights.sum()
            
            # Weighted combination of nearest training points
            reconstructed[i] = np.average(training_points[indices[i]], axis=0, weights=weights)
        
        print(f"[{datetime.now()}] Reconstruction complete. Shape: {reconstructed.shape}", flush=True)
        return reconstructed
    
    # ============================================================================
    # Autoencoder Methods
    # ============================================================================
    
    def load_autoencoder_model(self, centroid_idx: int, n_components: int) -> object:
        """Load a trained isometric autoencoder model for a specific centroid.
        
        Args:
            centroid_idx: Index of the centroid (0-based)
            n_components: Dimensionality (latent dim) of the autoencoder
        
        Returns:
            Trained autoencoder model (TiedWeightAutoencoder)
        
        Raises:
            FileNotFoundError: If model file does not exist
            ImportError: If TiedWeightAutoencoder cannot be imported
        """
        cache_key = self._get_model_cache_key("autoencoder", centroid_idx, n_components)
        
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        # Import autoencoder class
        try:
            from TiedWeightAutoencoder import TiedWeightAutoencoder
        except ImportError:
            raise ImportError(
                "Could not import TiedWeightAutoencoder. "
                "Make sure it's in the src directory."
            )
        
        model_dir = paths.get_results_dir() / f"autoencoder_atlas_{n_components}D" / f"{n_components}D"
        model_path = model_dir / f"centroid_{centroid_idx:04d}_autoencoder_{n_components}D.pt"
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Autoencoder model not found at {model_path}. "
                f"Make sure you've run isometric_autoencoder_for_each_centroid.py with n_components={n_components}"
            )
        
        print(f"[{datetime.now()}] Loading autoencoder model from {model_path}...", flush=True)
        
        # Load state dict to determine input dimension
        state_dict = torch.load(model_path, map_location=self.device)
        
        # Infer input dimension from encoder weights
        # encoder_mat_1 is stored as (input_dim, hidden_dim) custom matrix, not PyTorch Linear
        input_dim = state_dict['encoder_mat_1'].shape[0]
        
        # Create model instance
        model = TiedWeightAutoencoder(
            input_dim=input_dim,
            latent_dim=n_components,
            device=self.device
        )
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        self._model_cache[cache_key] = model
        
        return model
    
    def embed_autoencoder(
        self,
        points: np.ndarray,
        centroid_idx: int,
        n_components: int
    ) -> np.ndarray:
        """Apply autoencoder embedding to points using a trained model.
        
        Args:
            points: Input points with shape (n_points, input_dim)
            centroid_idx: Index of the centroid whose model to use
            n_components: Latent dimension of the autoencoder
        
        Returns:
            Embedded points (latent codes) with shape (n_points, n_components)
        """
        model = self.load_autoencoder_model(centroid_idx, n_components)
        
        print(f"[{datetime.now()}] Applying autoencoder embedding (centroid {centroid_idx}, {n_components}D)...", flush=True)
        
        # Convert to tensor
        points_tensor = torch.from_numpy(points).float().to(self.device)
        
        # Encode
        with torch.no_grad():
            embeddings = model.encode(points_tensor)
        
        embeddings_np = embeddings.cpu().numpy()
        
        print(f"[{datetime.now()}] Autoencoder embeddings computed. Shape: {embeddings_np.shape}", flush=True)
        return embeddings_np
    
    def reconstruct_autoencoder(
        self,
        embeddings: np.ndarray,
        centroid_idx: int,
        n_components: int
    ) -> np.ndarray:
        """Reconstruct ambient representation from autoencoder embeddings.
        
        Args:
            embeddings: Embedded points (latent codes) with shape (n_points, n_components)
            centroid_idx: Index of the centroid whose model to use
            n_components: Latent dimension of the autoencoder
        
        Returns:
            Reconstructed points in ambient space with shape (n_points, input_dim)
        """
        model = self.load_autoencoder_model(centroid_idx, n_components)
        
        print(f"[{datetime.now()}] Reconstructing from autoencoder embeddings (centroid {centroid_idx})...", flush=True)
        
        # Convert to tensor
        embeddings_tensor = torch.from_numpy(embeddings).float().to(self.device)
        
        # Decode
        with torch.no_grad():
            reconstructed = model.decode(embeddings_tensor)
        
        reconstructed_np = reconstructed.cpu().numpy()
        
        print(f"[{datetime.now()}] Reconstruction complete. Shape: {reconstructed_np.shape}", flush=True)
        return reconstructed_np
    
    # ============================================================================
    # Unified convenience methods
    # ============================================================================
    
    def embed(
        self,
        points: np.ndarray,
        centroid_idx: int,
        method: str = "pca",
        n_components: int = 50
    ) -> np.ndarray:
        """Generic embedding function supporting all three methods.
        
        Args:
            points: Input points with shape (n_points, input_dim)
            centroid_idx: Index of the centroid whose model to use
            method: Embedding method ("pca", "isomap", or "autoencoder")
            n_components: Target dimensionality
        
        Returns:
            Embedded points with shape (n_points, n_components)
        
        Raises:
            ValueError: If method is not recognized
        """
        if method.lower() == "pca":
            return self.embed_pca(points, centroid_idx, n_components)
        elif method.lower() == "isomap":
            return self.embed_isomap(points, centroid_idx, n_components)
        elif method.lower() == "autoencoder":
            return self.embed_autoencoder(points, centroid_idx, n_components)
        else:
            raise ValueError(
                f"Unknown embedding method: {method}. "
                f"Must be one of: 'pca', 'isomap', 'autoencoder'"
            )
    
    def reconstruct(
        self,
        embeddings: np.ndarray,
        centroid_idx: int,
        method: str = "pca",
        n_components: int = 50,
        training_points: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Generic reconstruction function supporting all three methods.
        
        Args:
            embeddings: Embedded points with shape (n_points, n_components)
            centroid_idx: Index of the centroid whose model to use
            method: Embedding method ("pca", "isomap", or "autoencoder")
            n_components: Latent dimensionality
            training_points: Required for Isomap reconstruction. 
                            Original training points for interpolation.
        
        Returns:
            Reconstructed points in ambient space
        
        Raises:
            ValueError: If method is not recognized
        """
        if method.lower() == "pca":
            return self.reconstruct_pca(embeddings, centroid_idx, n_components)
        elif method.lower() == "isomap":
            return self.reconstruct_isomap(embeddings, centroid_idx, n_components, training_points)
        elif method.lower() == "autoencoder":
            return self.reconstruct_autoencoder(embeddings, centroid_idx, n_components)
        else:
            raise ValueError(
                f"Unknown embedding method: {method}. "
                f"Must be one of: 'pca', 'isomap', 'autoencoder'"
            )
    
    def clear_cache(self):
        """Clear the model cache to free memory."""
        self._model_cache.clear()
        print(f"[{datetime.now()}] Model cache cleared", flush=True)


# ============================================================================
# Convenience functions for direct use without class
# ============================================================================

def load_pca_model(centroid_idx: int, n_components: int, config: Optional[Config] = None) -> object:
    """Load a PCA model directly.
    
    Args:
        centroid_idx: Index of the centroid
        n_components: Dimensionality of the model
        config: Config object. If None, uses default.
    
    Returns:
        Trained PCA model
    """
    loader = AtlasLoader(config)
    return loader.load_pca_model(centroid_idx, n_components)


def load_isomap_model(centroid_idx: int, n_components: int, config: Optional[Config] = None) -> object:
    """Load an Isomap model directly.
    
    Args:
        centroid_idx: Index of the centroid
        n_components: Dimensionality of the model
        config: Config object. If None, uses default.
    
    Returns:
        Trained Isomap model
    """
    loader = AtlasLoader(config)
    return loader.load_isomap_model(centroid_idx, n_components)


def load_autoencoder_model(centroid_idx: int, n_components: int, config: Optional[Config] = None, device: str = 'cpu') -> object:
    """Load an autoencoder model directly.
    
    Args:
        centroid_idx: Index of the centroid
        n_components: Latent dimension of the model
        config: Config object. If None, uses default.
        device: Device for inference ('cpu' or 'cuda')
    
    Returns:
        Trained autoencoder model
    """
    loader = AtlasLoader(config, device=device)
    return loader.load_autoencoder_model(centroid_idx, n_components)


if __name__ == "__main__":
    # Example usage
    config = load_config()
    loader = AtlasLoader(config)
    
    # Load centroids
    centroids = loader.load_centroids()
    print(f"Centroids shape: {centroids.shape}")
    
    # Find K nearest centroids to first centroid
    distances, indices = loader.get_nearest_centroids(centroids[0], k=5)
    print(f"Nearest centroids: {indices}")
    print(f"Distances: {distances}")
