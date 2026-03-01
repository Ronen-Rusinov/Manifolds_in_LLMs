# AtlasLoader Module Documentation

## Overview

The `AtlasLoader` module provides a unified interface for working with trained dimensionality reduction models across your three main approaches:
- **PCA** (Principal Component Analysis)
- **Isomap** (Isometric Mapping)
- **Isometric Autoencoders** (TiedWeightAutoencoder)

The module handles:
1. Loading pre-trained models from disk
2. Finding K nearest centroids to query points
3. Embedding points into low-dimensional spaces
4. Reconstructing original ambient representations from embeddings

## Quick Start

### Basic Usage

```python
from atlas_loader import AtlasLoader
from config_manager import load_config

# Initialize the loader
config = load_config()
loader = AtlasLoader(config, device='cpu')  # use 'cuda' for GPU

# Load centroids
centroids = loader.load_centroids()

# Find 5 nearest centroids to a query point
distances, indices = loader.get_nearest_centroids(query_point, k=5)

# Embed points using PCA
embeddings = loader.embed_pca(points, centroid_idx=0, n_components=50)

# Reconstruct original representation from embeddings
reconstructed = loader.reconstruct_pca(embeddings, centroid_idx=0, n_components=50)
```

### Using Unified Interface

```python
# Generic method that works with any embedding type
embeddings = loader.embed(points, centroid_idx=0, method='pca', n_components=50)
reconstructed = loader.reconstruct(embeddings, centroid_idx=0, method='pca', n_components=50)
```

## API Reference

### Class: AtlasLoader

#### Constructor

```python
AtlasLoader(config: Optional[Config] = None, device: str = 'cpu')
```

**Parameters:**
- `config`: Configuration object. If None, loads default config.
- `device`: Device for autoencoder inference ('cpu' or 'cuda'). Defaults to 'cpu'.

#### Core Methods

##### 1. Loading Centroids

```python
load_centroids(n_centroids: Optional[int] = None) -> np.ndarray
```

Loads centroid vectors from disk. Results are cached for efficient repeated access.

**Parameters:**
- `n_centroids`: Number of centroids. If None, uses config value.

**Returns:**
- Centroid array of shape `(n_centroids, embedding_dim)`

**Example:**
```python
centroids = loader.load_centroids(n_centroids=200)
print(centroids.shape)  # (200, 4096) for example
```

##### 2. Finding Nearest Centroids

```python
get_nearest_centroids(
    query_point: np.ndarray, 
    k: int = 5
) -> Tuple[np.ndarray, np.ndarray]
```

Finds K nearest centroids to a query point using KDTree for efficient search.

**Parameters:**
- `query_point`: Query vector of shape `(embedding_dim,)` or `(1, embedding_dim)`
- `k`: Number of nearest centroids to find

**Returns:**
- Tuple of (distances, indices) where both have shape `(k,)`

**Example:**
```python
distances, indices = loader.get_nearest_centroids(activations[0], k=5)
print(f"Nearest centroids: {indices}")
print(f"Distances: {distances}")
```

### PCA Methods

#### Load PCA Model

```python
load_pca_model(centroid_idx: int, n_components: int) -> sklearn.decomposition.PCA
```

Loads a trained PCA model for a specific centroid.

**Parameters:**
- `centroid_idx`: Index of the centroid (0-based)
- `n_components`: Dimensionality of the PCA model

**Returns:**
- Trained PCA model

**Raises:**
- `FileNotFoundError`: If model file doesn't exist

**Example:**
```python
pca_model = loader.load_pca_model(centroid_idx=0, n_components=50)
print(f"Explained variance ratio: {pca_model.explained_variance_ratio_.sum():.4f}")
```

#### Embed with PCA

```python
embed_pca(
    points: np.ndarray,
    centroid_idx: int,
    n_components: int
) -> np.ndarray
```

Embeds points into PCA space.

**Parameters:**
- `points`: Input points of shape `(n_points, input_dim)`
- `centroid_idx`: Index of centroid whose model to use
- `n_components`: PCA dimensionality

**Returns:**
- Embedded points of shape `(n_points, n_components)`

**Example:**
```python
# Get first 10 neighbors of centroid 5
neighbor_indices = loader.get_nearest_centroids(centroids[5], k=10)[1]
neighbors = activations[neighbor_indices]

# Embed them
embeddings = loader.embed_pca(neighbors, centroid_idx=5, n_components=50)
print(embeddings.shape)  # (10, 50)
```

#### Reconstruct from PCA

```python
reconstruct_pca(
    embeddings: np.ndarray,
    centroid_idx: int,
    n_components: int
) -> np.ndarray
```

Reconstructs original representation from PCA embeddings.

**Parameters:**
- `embeddings`: Embedded points of shape `(n_points, n_components)`
- `centroid_idx`: Index of centroid whose model to use
- `n_components`: PCA dimensionality

**Returns:**
- Reconstructed points in ambient space of shape `(n_points, input_dim)`

**Example:**
```python
# Embed and reconstruct
embeddings = loader.embed_pca(points, centroid_idx=5, n_components=50)
reconstructed = loader.reconstruct_pca(embeddings, centroid_idx=5, n_components=50)

# Compute error
error = np.mean(np.linalg.norm(points - reconstructed, axis=1))
print(f"Reconstruction error: {error:.6f}")
```

### Isomap Methods

#### Load Isomap Model

```python
load_isomap_model(centroid_idx: int, n_components: int) -> sklearn.manifold.Isomap
```

Loads a trained Isomap model.

**Parameters:**
- `centroid_idx`: Index of the centroid (0-based)
- `n_components`: Dimensionality of the Isomap model

**Returns:**
- Trained Isomap model

**Raises:**
- `FileNotFoundError`: If model file doesn't exist

#### Embed with Isomap

```python
embed_isomap(
    points: np.ndarray,
    centroid_idx: int,
    n_components: int
) -> np.ndarray
```

Embeds points into Isomap space.

**Parameters:**
- `points`: Input points of shape `(n_points, input_dim)`
- `centroid_idx`: Index of centroid whose model to use
- `n_components`: Isomap dimensionality

**Returns:**
- Embedded points of shape `(n_points, n_components)`

#### Reconstruct from Isomap

```python
reconstruct_isomap(
    embeddings: np.ndarray,
    centroid_idx: int,
    n_components: int,
    training_points: Optional[np.ndarray] = None
) -> np.ndarray
```

Reconstructs original representation from Isomap embeddings.

**Important Note:** Isomap does not have a built-in inverse transform like PCA. The reconstruction uses interpolation:
- If `training_points` is provided, uses KDTree-based nearest neighbor interpolation in the embedding space
- If `training_points` is None, returns embeddings with a warning

**Parameters:**
- `embeddings`: Embedded points of shape `(n_points, n_components)`
- `centroid_idx`: Index of centroid whose model to use
- `n_components`: Isomap dimensionality
- `training_points`: Original training data for interpolation (optional)

**Returns:**
- Reconstructed points of shape `(n_points, input_dim)`

**Example:**
```python
# Embed points
embeddings = loader.embed_isomap(test_points, centroid_idx=5, n_components=50)

# Reconstruct using training points for interpolation
training_points = activations[neighbor_indices]
reconstructed = loader.reconstruct_isomap(
    embeddings, 
    centroid_idx=5, 
    n_components=50,
    training_points=training_points
)
```

### Autoencoder Methods

#### Load Autoencoder Model

```python
load_autoencoder_model(centroid_idx: int, n_components: int) -> TiedWeightAutoencoder
```

Loads a trained autoencoder model.

**Parameters:**
- `centroid_idx`: Index of the centroid (0-based)
- `n_components`: Latent dimension of the autoencoder

**Returns:**
- Trained TiedWeightAutoencoder in evaluation mode

**Raises:**
- `FileNotFoundError`: If model file doesn't exist
- `ImportError`: If TiedWeightAutoencoder cannot be imported

**Example:**
```python
ae_model = loader.load_autoencoder_model(centroid_idx=0, n_components=50)
ae_model.eval()  # Already in eval mode, but good practice
```

#### Embed with Autoencoder

```python
embed_autoencoder(
    points: np.ndarray,
    centroid_idx: int,
    n_components: int
) -> np.ndarray
```

Embeds points into autoencoder latent space.

**Parameters:**
- `points`: Input points of shape `(n_points, input_dim)`
- `centroid_idx`: Index of centroid whose model to use
- `n_components`: Latent dimension

**Returns:**
- Latent codes of shape `(n_points, n_components)`

**Example:**
```python
latent = loader.embed_autoencoder(points, centroid_idx=0, n_components=50)
print(latent.shape)  # (n_points, 50)
```

#### Reconstruct from Autoencoder

```python
reconstruct_autoencoder(
    embeddings: np.ndarray,
    centroid_idx: int,
    n_components: int
) -> np.ndarray
```

Reconstructs original representation from autoencoder latent codes.

**Parameters:**
- `embeddings`: Latent codes of shape `(n_points, n_components)`
- `centroid_idx`: Index of centroid whose model to use
- `n_components`: Latent dimension

**Returns:**
- Reconstructed points in ambient space of shape `(n_points, input_dim)`

**Example:**
```python
# Full cycle: embed and reconstruct
latent = loader.embed_autoencoder(points, centroid_idx=0, n_components=50)
reconstructed = loader.reconstruct_autoencoder(latent, centroid_idx=0, n_components=50)

# Compute MSE error
mse_error = np.mean((points - reconstructed)**2)
print(f"MSE: {mse_error:.6f}")
```

### Unified Interface Methods

#### Generic Embed

```python
embed(
    points: np.ndarray,
    centroid_idx: int,
    method: str = "pca",
    n_components: int = 50
) -> np.ndarray
```

Generic embedding function supporting all three methods.

**Parameters:**
- `points`: Input points of shape `(n_points, input_dim)`
- `centroid_idx`: Index of centroid whose model to use
- `method`: Embedding method ('pca', 'isomap', or 'autoencoder')
- `n_components`: Target dimensionality

**Returns:**
- Embedded points

**Example:**
```python
# Same code works for all methods
for method in ['pca', 'isomap', 'autoencoder']:
    embeddings = loader.embed(points, centroid_idx=0, method=method, n_components=50)
    print(f"{method}: {embeddings.shape}")
```

#### Generic Reconstruct

```python
reconstruct(
    embeddings: np.ndarray,
    centroid_idx: int,
    method: str = "pca",
    n_components: int = 50,
    training_points: Optional[np.ndarray] = None
) -> np.ndarray
```

Generic reconstruction function supporting all three methods.

**Parameters:**
- `embeddings`: Embedded points
- `centroid_idx`: Index of centroid whose model to use
- `method`: Embedding method ('pca', 'isomap', or 'autoencoder')
- `n_components`: Latent dimensionality
- `training_points`: Required for Isomap reconstruction

**Returns:**
- Reconstructed ambient representation

#### Clear Cache

```python
clear_cache()
```

Clears the in-memory model cache to free memory.

**Example:**
```python
# Process many centroids
for i in range(200):
    embeddings = loader.embed(points, centroid_idx=i, method='pca', n_components=50)
    # ... do something with embeddings ...
    if i % 10 == 0:
        loader.clear_cache()  # Free memory every 10 iterations
```

## Convenience Functions

For quick one-off operations without creating an AtlasLoader instance:

```python
from atlas_loader import load_pca_model, load_isomap_model, load_autoencoder_model

# Load models directly
pca = load_pca_model(centroid_idx=0, n_components=50)
isomap = load_isomap_model(centroid_idx=0, n_components=50)
autoencoder = load_autoencoder_model(centroid_idx=0, n_components=50, device='cuda')
```

## Path Resolution

The module resolves model paths automatically using the config. Helper functions in `paths.py` provide direct access:

```python
from paths import (
    get_pca_model_path,
    get_isomap_model_path,
    get_autoencoder_model_path,
    get_autoencoder_history_path
)

# Get paths directly
pca_path = get_pca_model_path(centroid_idx=0, n_components=50)
ae_path = get_autoencoder_model_path(centroid_idx=0, n_components=50)
history_path = get_autoencoder_history_path(centroid_idx=0, n_components=50)
```

## Configuration

The module respects the following config parameters:

- `config.clustering.n_centroids`: Default number of centroids to load
- `config.model.balltree_leaf_size`: Leaf size for KDTree in nearest centroid search
- `config.training.random_seed`: Random seed (indirectly affects model paths)

## File Structure

```
results/
├── minibatch_kmeans_200/
│   └── centroids_200.npy
├── pca_atlas_50D/
│   └── 50D/
│       ├── centroid_0000_pca_50D.joblib
│       ├── centroid_0000_embeddings_50D.npy
│       └── ...
├── iso_atlas_50D/
│   └── 50D/
│       ├── centroid_0000_isomap_50D.joblib
│       ├── centroid_0000_embeddings_50D.npy
│       └── ...
└── autoencoder_atlas_50D/
    └── 50D/
        ├── centroid_0000_autoencoder_50D.pt
        ├── centroid_0000_history_50D.joblib
        ├── centroid_0000_embeddings_50D.npy
        └── ...
```

## Common Usage Patterns

### Pattern 1: Batch Embedding and Reconstruction

```python
loader = AtlasLoader(config)

# Process multiple centroids
for centroid_idx in range(10):
    # Embed
    embeddings = loader.embed(test_points, centroid_idx, method='pca', n_components=50)
    
    # Reconstruct
    reconstructed = loader.reconstruct(embeddings, centroid_idx, method='pca', n_components=50)
    
    # Compute error
    error = np.mean(np.linalg.norm(test_points - reconstructed, axis=1))
    print(f"Centroid {centroid_idx}: error = {error:.6f}")
```

### Pattern 2: Finding Nearby Centroids and Embedding

```python
loader = AtlasLoader(config)
centroids = loader.load_centroids()

# Find 5 nearest centroids to a query point
distances, indices = loader.get_nearest_centroids(query_point, k=5)

# Embed point using each nearby centroid
embeddings = []
for centroid_idx in indices:
    emb = loader.embed(query_point.reshape(1, -1), centroid_idx, method='pca')
    embeddings.append(emb)

# Take average embedding
avg_embedding = np.mean(embeddings, axis=0)
```

### Pattern 3: Cross-Modal Comparison

```python
loader = AtlasLoader(config)

# Embed same points with all three methods
pca_emb = loader.embed_pca(points, centroid_idx=0, n_components=50)
iso_emb = loader.embed_isomap(points, centroid_idx=0, n_components=50)
ae_emb = loader.embed_autoencoder(points, centroid_idx=0, n_components=50)

# Compare embeddings
print(pca_emb.shape, iso_emb.shape, ae_emb.shape)
```

## Error Handling

```python
try:
    embeddings = loader.embed_pca(points, centroid_idx=0, n_components=50)
except FileNotFoundError as e:
    print(f"Model not found: {e}")
    print("Make sure to train models first!")

try:
    ae_model = loader.load_autoencoder_model(centroid_idx=0, n_components=50)
except ImportError as e:
    print(f"Cannot import autoencoder: {e}")
```

## Performance Tips

1. **Model Caching**: Models are automatically cached in memory. Use `clear_cache()` if processing many centroids to avoid memory issues.

2. **GPU Acceleration**: Use `device='cuda'` for autoencoder inference if CUDA is available.

3. **Batch Operations**: Process multiple points at once rather than one-by-one for better efficiency.

4. **KDTree Reuse**: The KDTree for nearest centroid search is built once and reused.

## Testing

Run the included example script:

```bash
cd examples
python atlas_loader_example.py --method pca --n-components 50 --centroid-idx 0
python atlas_loader_example.py --method isomap --n-components 50 --centroid-idx 5
python atlas_loader_example.py --method autoencoder --n-components 50 --device cuda
```

## Troubleshooting

### Models not found
**Error:** `FileNotFoundError: ... model not found at ...`

**Solution:** Make sure you've run the corresponding training script (pca_for_each_centroid.py, isomap_for_each_centroid.py, isometric_autoencoder_for_each_centroid.py) with the correct parameters.

### Import errors for TiedWeightAutoencoder
**Error:** `ImportError: Could not import TiedWeightAutoencoder`

**Solution:** Ensure TiedWeightAutoencoder.py is in the src directory and the src path is properly added to sys.path.

### CUDA out of memory
**Error:** `RuntimeError: CUDA out of memory`

**Solution:** 
- Use `device='cpu'` instead of 'cuda'
- Or use `loader.clear_cache()` to free memory
- Or process fewer points at a time
