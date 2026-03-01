# AtlasLoader Quick Reference

A 1-page cheat sheet for the AtlasLoader module.

## Installation & Setup

```python
from atlas_loader import AtlasLoader
from config_manager import load_config

# Initialize (uses default config if not provided)
config = load_config()
loader = AtlasLoader(config, device='cpu')  # or 'cuda' for GPU
```

## Loading Centroids & Finding Neighbors

```python
# Load all centroids
centroids = loader.load_centroids()  # shape: (n_centroids, embedding_dim)

# Find K nearest centroids to a query point
distances, indices = loader.get_nearest_centroids(query_point, k=5)
# distances: [d1, d2, d3, d4, d5]
# indices:   [idx1, idx2, idx3, idx4, idx5]

# Find nearest centroid to each point in batch
for i, point in enumerate(points):
    dist, idx = loader.get_nearest_centroids(point, k=1)
    print(f"Point {i} -> Centroid {idx[0]}")
```

## Embedding & Reconstruction

### PCA

```python
# Embed points to PCA space
embeddings = loader.embed_pca(points, centroid_idx=0, n_components=50)  # (n, 50)

# Reconstruct back to original space
reconstructed = loader.reconstruct_pca(embeddings, centroid_idx=0, n_components=50)  # (n, original_dim)

# Compute reconstruction error
error = np.mean(np.linalg.norm(points - reconstructed, axis=1))
```

### Isomap

```python
# Embed points
embeddings = loader.embed_isomap(points, centroid_idx=0, n_components=50)

# Reconstruct (requires training points for interpolation)
reconstructed = loader.reconstruct_isomap(
    embeddings, 
    centroid_idx=0, 
    n_components=50,
    training_points=training_data  # Original training points
)
```

### Autoencoder

```python
# Embed to latent space
latent = loader.embed_autoencoder(points, centroid_idx=0, n_components=50)

# Reconstruct from latent space
reconstructed = loader.reconstruct_autoencoder(latent, centroid_idx=0, n_components=50)
```

## Unified Interface (One API for All Methods)

```python
# Generic embed function
embeddings = loader.embed(
    points=points,
    centroid_idx=0,
    method='pca',  # or 'isomap', 'autoencoder'
    n_components=50
)

# Generic reconstruct function
reconstructed = loader.reconstruct(
    embeddings=embeddings,
    centroid_idx=0,
    method='pca',
    n_components=50,
    training_points=None  # Required for Isomap
)

# Loop over all methods
for method in ['pca', 'isomap', 'autoencoder']:
    try:
        emb = loader.embed(points, 0, method=method, n_components=50)
        rec = loader.reconstruct(emb, 0, method=method, n_components=50)
        error = np.mean(np.linalg.norm(points - rec, axis=1))
        print(f"{method}: {error:.6f}")
    except FileNotFoundError:
        print(f"{method}: Model not found")
```

## Direct Model Loading

```python
# Load models directly
pca_model = loader.load_pca_model(centroid_idx=0, n_components=50)
isomap_model = loader.load_isomap_model(centroid_idx=0, n_components=50)
ae_model = loader.load_autoencoder_model(centroid_idx=0, n_components=50)

# Use models directly
embeddings = pca_model.transform(points)
reconstructed = pca_model.inverse_transform(embeddings)
```

## Memory Management

```python
# Clear model cache to free memory
loader.clear_cache()

# Pattern: process many centroids, clear cache periodically
for i in range(200):
    embeddings = loader.embed(points, i, method='pca', n_components=50)
    # ... do something ...
    if (i + 1) % 10 == 0:
        loader.clear_cache()  # Free memory every 10 iterations
```

## Common Patterns

### Pattern 1: Simple Embedding Pipeline

```python
loader = AtlasLoader(load_config())

# Embed and reconstruct
embeddings = loader.embed_pca(points, centroid_idx=0, n_components=50)
reconstructed = loader.reconstruct_pca(embeddings, centroid_idx=0, n_components=50)
error = np.mean(np.linalg.norm(points - reconstructed, axis=1))
```

### Pattern 2: Find Best Centroid and Embed

```python
centroids = loader.load_centroids()

# For each point, find nearest centroid and embed
for point in points:
    _, [best_centroid] = loader.get_nearest_centroids(point, k=1)
    embedding = loader.embed_pca(point.reshape(1, -1), best_centroid, n_components=50)
```

### Pattern 3: Multi-Method Comparison

```python
# Compare all three methods
methods = ['pca', 'isomap', 'autoencoder']
results = {}

for method in methods:
    try:
        emb = loader.embed(points, centroid_idx=0, method=method, n_components=50)
        rec = loader.reconstruct(emb, centroid_idx=0, method=method, n_components=50)
        results[method] = np.mean(np.linalg.norm(points - rec, axis=1))
    except FileNotFoundError:
        results[method] = None

for method, error in results.items():
    print(f"{method}: {error}")
```

## GPU Acceleration

```python
# Use CUDA for autoencoder (faster)
loader = AtlasLoader(load_config(), device='cuda')
latent = loader.embed_autoencoder(points, centroid_idx=0, n_components=50)

# Or stick with CPU
loader = AtlasLoader(load_config(), device='cpu')
```

## Error Handling

```python
try:
    embeddings = loader.embed_pca(points, centroid_idx=0, n_components=50)
except FileNotFoundError as e:
    print(f"Model not found: {e}")
    print("Run: python scripts/pca_for_each_centroid.py --n-centroids 200 --n-components 50")
```

## File Structure

Models are stored in:
```
results/
├── pca_atlas_50D/50D/centroid_0000_pca_50D.joblib
├── iso_atlas_50D/50D/centroid_0000_isomap_50D.joblib
└── autoencoder_atlas_50D/50D/centroid_0000_autoencoder_50D.pt
```

## Configuration

Models automatically use config values for:
- `config.clustering.n_centroids` - Default number of centroids
- `config.model.balltree_leaf_size` - KDTree leaf size for efficiency

## API Summary Table

| Operation | Function | Input | Output |
|-----------|----------|-------|--------|
| Load centroids | `load_centroids()` | - | (n_centroids, dim) |
| Find K nearest | `get_nearest_centroids(point, k)` | (dim,) | (k,), (k,) |
| PCA embed | `embed_pca(points, idx, n_comp)` | (n, dim) | (n, n_comp) |
| PCA reconstruct | `reconstruct_pca(emb, idx, n_comp)` | (n, n_comp) | (n, dim) |
| Isomap embed | `embed_isomap(points, idx, n_comp)` | (n, dim) | (n, n_comp) |
| Isomap reconstruct | `reconstruct_isomap(emb, idx, n_comp, X)` | (n, n_comp) | (n, dim) |
| AE embed | `embed_autoencoder(points, idx, n_comp)` | (n, dim) | (n, n_comp) |
| AE reconstruct | `reconstruct_autoencoder(emb, idx, n_comp)` | (n, n_comp) | (n, dim) |
| Generic embed | `embed(points, idx, method, n_comp)` | (n, dim) | (n, n_comp) |
| Generic reconstruct | `reconstruct(emb, idx, method, n_comp)` | (n, n_comp) | (n, dim) |
| Load PCA | `load_pca_model(idx, n_comp)` | - | sklearn PCA |
| Load Isomap | `load_isomap_model(idx, n_comp)` | - | sklearn Isomap |
| Load AE | `load_autoencoder_model(idx, n_comp)` | - | TiedWeightAutoencoder |
| Clear cache | `clear_cache()` | - | - |

## Tips & Tricks

1. **Batch operations are faster** - process multiple points at once
2. **Cache models** - already done automatically
3. **Use GPU** - 2-10x speedup for autoencoders with device='cuda'
4. **Memory efficient** - call `loader.clear_cache()` when processing many centroids
5. **Logging** - all operations print timestamps, watch stdout for progress

## Example Command

```bash
# Run the full example
python examples/atlas_loader_example.py \
    --method pca \
    --n-components 50 \
    --centroid-idx 0 \
    --n-test-points 20 \
    --k-nearest 5
```

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| `FileNotFoundError: PCA model not found` | Run `scripts/pca_for_each_centroid.py` first |
| `ImportError: Could not import TiedWeightAutoencoder` | Check TiedWeightAutoencoder.py is in src/ |
| `CUDA out of memory` | Use `device='cpu'` or call `clear_cache()` |
| `ValueError: Isomap n_neighbors > n_samples` | Reduce n_neighbors in config or use fewer samples |

## For More Information

- Full documentation: `src/ATLAS_LOADER_README.md`
- Implementation details: `ATLAS_LOADER_IMPLEMENTATION.md`
- Example code: `examples/atlas_loader_example.py`
