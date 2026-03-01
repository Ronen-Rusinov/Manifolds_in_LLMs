# AtlasLoader Implementation Summary

## Overview

I've implemented a comprehensive `AtlasLoader` module that provides unified access to your trained dimensionality reduction models (PCA, Isomap, and Isometric Autoencoders). The module handles model loading, centroid management, point embedding, and reconstruction across all three methods.

## Files Created/Modified

### 1. **New File: `/src/atlas_loader.py`** (540+ lines)
   
**Core Class: `AtlasLoader`**
   - Manages loading and caching of all three model types
   - Provides unified interface for embedding and reconstruction
   - Handles centroids and KDTree for nearest centroid search
   - Memory-efficient with caching and optional cache clearing

**Key Features:**
   - **Model Caching**: Automatically caches loaded models to avoid redundant disk I/O
   - **Flexible Initialization**: Works with or without a provided config
   - **GPU Support**: Autoencoder inference supports both CPU and GPU
   - **Error Handling**: Clear, informative error messages for missing models or files

### 2. **Enhanced File: `/src/paths.py`**
   
Added four new helper functions:
   - `get_pca_model_path(centroid_index, n_components)` → Path to PCA model
   - `get_isomap_model_path(centroid_index, n_components)` → Path to Isomap model
   - `get_autoencoder_model_path(centroid_index, n_components)` → Path to AutoEncoder model
   - `get_autoencoder_history_path(centroid_index, n_components)` → Path to training history

These functions resolve paths directly from config, with automatic formatting of centroid indices.

### 3. **New File: `/examples/atlas_loader_example.py`** (220+ lines)

A comprehensive example script demonstrating:
   - Loading centroids
   - Finding K nearest centroids
   - Embedding points with all three methods
   - Reconstructing original representations
   - Computing reconstruction errors
   - Direct model loading

**Usage:**
```bash
python examples/atlas_loader_example.py --method pca --n-components 50
python examples/atlas_loader_example.py --method isomap --n-components 50 --k-nearest 10
python examples/atlas_loader_example.py --method autoencoder --device cuda
```

### 4. **New File: `/src/ATLAS_LOADER_README.md`** (600+ lines)

Complete API documentation including:
   - Quick start guide
   - Full API reference for all methods
   - Usage patterns and examples
   - File structure overview
   - Troubleshooting guide
   - Performance tips

## Implementation Details

### 1. Loading Models from Disk

**For PCA and Isomap (joblib format):**
```python
loader = AtlasLoader(config)
pca_model = loader.load_pca_model(centroid_idx=0, n_components=50)
isomap_model = loader.load_isomap_model(centroid_idx=0, n_components=50)
```

**For Autoencoder (PyTorch format):**
```python
ae_model = loader.load_autoencoder_model(centroid_idx=0, n_components=50, device='cuda')
```

**Implementation Details:**
- Models are loaded once and cached to avoid redundant disk I/O
- File paths are resolved automatically from config
- Clear error messages if models don't exist
- Autoencoder input dimension is automatically inferred from state dict

### 2. ✓ Finding K Closest Centroids

```python
distances, indices = loader.get_nearest_centroids(query_point, k=5)
```

**Implementation:**
- Builds KDTree on centroids for efficient nearest neighbor search
- Automatically handles 1D and 2D input arrays
- KDTree is built once and reused
- Returns sorted results by distance

**Complexity:** O(log n) query time after O(n log n) KDTree construction

### 3. ✓ Embedding Points (Three Methods)

**PCA:**
```python
embeddings = loader.embed_pca(points, centroid_idx=0, n_components=50)
```
- Uses sklearn's PCA.transform()
- Input: (n_points, input_dim) → Output: (n_points, n_components)

**Isomap:**
```python
embeddings = loader.embed_isomap(points, centroid_idx=0, n_components=50)
```
- Uses sklearn's Isomap.transform()
- Maintains geodesic manifold structure

**Autoencoder:**
```python
embeddings = loader.embed_autoencoder(points, centroid_idx=0, n_components=50)
```
- Uses trained encoder neural network
- Input: (n_points, input_dim) → Output: (n_points, latent_dim)
- Supports both CPU and GPU inference

### 4. ✓ Reconstruction from Embeddings

**PCA Reconstruction:**
```python
reconstructed = loader.reconstruct_pca(embeddings, centroid_idx=0, n_components=50)
```
- Uses sklearn's PCA.inverse_transform()
- Perfect reconstruction possible in theory (up to numerical precision)

**Isomap Reconstruction:**
```python
reconstructed = loader.reconstruct_isomap(embeddings, centroid_idx=0, n_components=50, training_points=X)
```
- Isomap has no built-in inverse transform
- Uses KDTree-based nearest neighbor interpolation in embedding space
- Weighted average of 5 nearest training points in embedding space
- Inverse distance weighting for smooth reconstruction

**Autoencoder Reconstruction:**
```python
reconstructed = loader.reconstruct_autoencoder(embeddings, centroid_idx=0, n_components=50)
```
- Uses trained decoder neural network
- (n_points, latent_dim) → Output: (n_points, input_dim)

## API Summary

### Centroid Operations
- `load_centroids()` - Load all centroid vectors
- `get_nearest_centroids(query_point, k)` - Find K nearest centroids

### PCA Operations
- `load_pca_model(centroid_idx, n_components)` - Load trained PCA model
- `embed_pca(points, centroid_idx, n_components)` - Embed to PCA space
- `reconstruct_pca(embeddings, centroid_idx, n_components)` - Reconstruct from PCA

### Isomap Operations
- `load_isomap_model(centroid_idx, n_components)` - Load trained Isomap model
- `embed_isomap(points, centroid_idx, n_components)` - Embed to Isomap space
- `reconstruct_isomap(embeddings, centroid_idx, n_components, training_points)` - Reconstruct from Isomap

### Autoencoder Operations
- `load_autoencoder_model(centroid_idx, n_components)` - Load trained autoencoder
- `embed_autoencoder(points, centroid_idx, n_components)` - Embed to latent space
- `reconstruct_autoencoder(embeddings, centroid_idx, n_components)` - Reconstruct from latent

### Unified Interface
- `embed(points, centroid_idx, method, n_components)` - Generic embedding
- `reconstruct(embeddings, centroid_idx, method, n_components, training_points)` - Generic reconstruction
- `clear_cache()` - Free memory

## Key Features

### 1. **Unified Interface**
```python
# Same code works for all methods
for method in ['pca', 'isomap', 'autoencoder']:
    embeddings = loader.embed(points, centroid_idx=0, method=method, n_components=50)
    reconstructed = loader.reconstruct(embeddings, centroid_idx=0, method=method, n_components=50)
```

### 2. **Model Caching**
- Automatically caches loaded models to avoid redundant disk I/O
- Clear cache manually when processing many centroids: `loader.clear_cache()`

### 3. **GPU Support**
```python
# Use CUDA for autoencoder inference
loader = AtlasLoader(config, device='cuda')
embeddings = loader.embed_autoencoder(points, centroid_idx=0, n_components=50)
```

### 4. **Error Handling**
- Clear error messages if models don't exist
- Informative warnings (e.g., for Isomap reconstruction without training_points)
- Automatic input dimension inference for autoencoders

### 5. **Verbose Logging**
All major operations print timestamps and progress information for debugging

## Integration with Existing Code

The module integrates seamlessly with your existing infrastructure:

1. **Config Management**: Works with your existing `config_manager.py`
2. **Path Resolution**: Uses functions from `paths.py` for consistent path handling
3. **Data Loading**: Compatible with `utils/common.py` utilities
4. **Model Training**: Loads models trained by existing scripts (pca_for_each_centroid.py, etc.)

## Example Workflows

### Workflow 1: Embedding and Reconstruction Pipeline

```python
from atlas_loader import AtlasLoader
from config_manager import load_config

config = load_config()
loader = AtlasLoader(config)

# Load data
activations = np.load('activations.npy')

# Find nearest centroid
centroids = loader.load_centroids()
distances, indices = loader.get_nearest_centroids(activations[0], k=1)
best_centroid = indices[0]

# Embed
embeddings = loader.embed(activations, centroid_idx=best_centroid, method='pca', n_components=50)

# Reconstruct
reconstructed = loader.reconstruct(embeddings, centroid_idx=best_centroid, method='pca', n_components=50)

# Evaluate
error = np.mean(np.linalg.norm(activations - reconstructed, axis=1))
```

### Workflow 2: Multi-Method Comparison

```python
for method in ['pca', 'isomap', 'autoencoder']:
    try:
        embeddings = loader.embed(test_points, 0, method=method, n_components=50)
        reconstructed = loader.reconstruct(embeddings, 0, method=method, n_components=50)
        error = np.mean(np.linalg.norm(test_points - reconstructed, axis=1))
        print(f"{method}: {error:.6f}")
    except FileNotFoundError as e:
        print(f"{method}: Not available - {e}")
```

### Workflow 3: Batch Processing

```python
# Process many centroids efficiently
for centroid_idx in range(200):
    embeddings = loader.embed(points, centroid_idx, method='pca', n_components=50)
    # ... process embeddings ...
    
    if (centroid_idx + 1) % 10 == 0:
        loader.clear_cache()  # Free memory every 10 iterations
```

## Testing

The example script provides comprehensive testing:

```bash
# Test PCA
python examples/atlas_loader_example.py --method pca --n-components 50 --centroid-idx 0

# Test Isomap
python examples/atlas_loader_example.py --method isomap --n-components 50

# Test Autoencoder with GPU
python examples/atlas_loader_example.py --method autoencoder --device cuda --n-test-points 20

# Test nearest centroid finding
python examples/atlas_loader_example.py --k-nearest 10
```

## Requirements

The module uses standard libraries plus your existing dependencies:
- `numpy` - Array operations
- `sklearn` (scikit-learn) - PCA and Isomap
- `torch` - Autoencoder models
- `joblib` - Model serialization

All are already in your requirements or used by existing code.

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Load centroids | O(n) | Once per session, cached |
| Build KDTree | O(n log n) | Once per session |
| Find K nearest | O(log n) | Query time |
| PCA embed | O(m·d) | m=points, d=components |
| Isomap embed | O(m·d) | Via trained model |
| AE embed | O(m·d) | Via neural network |
| Reconstruct | O(m·d) | All methods similar |

## Future Extensions

The module is designed to be extensible:

1. **Additional Methods**: Easy to add t-SNE, UMAP, etc.
2. **Batch Model Loading**: Load multiple models at once
3. **Model Ensembling**: Combine predictions from multiple methods
4. **Online Learning**: Support for incremental model updates
5. **Visualization**: Add built-in visualization methods

## Summary

✅ **Complete Implementation** of all requested features:

1. ✅ **Loading Models** from disk with automatic path resolution
2. ✅ **K Nearest Centroids** using efficient KDTree search
3. ✅ **Embedding Functions** for PCA, Isomap, and Autoencoder
4. ✅ **Reconstruction** with proper handling for each method (especially Isomap)

**Additional Features:**
- Unified generic interface for all methods
- Model caching for efficiency
- Comprehensive error handling
- GPU support for autoencoders
- Timestamped logging
- Complete documentation and examples
- Unit-testable API design
