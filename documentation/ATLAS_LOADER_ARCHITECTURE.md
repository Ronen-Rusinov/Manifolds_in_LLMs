# AtlasLoader Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      AtlasLoader Module                          │
│                   (src/atlas_loader.py)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                │             │             │
                ▼             ▼             ▼
        ┌─────────────┐  ┌─────────────┐  ┌──────────────┐
        │ PCA Module  │  │ Isomap      │  │ Autoencoder  │
        │ (sklearn)   │  │ (sklearn)   │  │ (PyTorch)    │
        └─────────────┘  └─────────────┘  └──────────────┘
                │             │             │
                └─────────────┼─────────────┘
                              │
                ┌─────────────▼─────────────┬──────────────┐
                │                          │              │
                ▼                          ▼              ▼
        ┌──────────────┐         ┌──────────────┐  ┌──────────┐
        │ Load Models  │         │ Embed Points │  │Reconstruct
        │ from Disk    │         │              │  │Original  │
        │ (caching)    │         │              │  │          │
        └──────────────┘         └──────────────┘  └──────────┘
```

## Data Flow

```
INPUT                    PROCESSING                    OUTPUT
─────────────────────────────────────────────────────────────────

centroid_idx ──┬──→ load_pca_model() ──→ PCA object
n_components ──┤                      └───→ cached
               │
               └──→ load_isomap_model() ──→ Isomap object
               │                        └───→ cached
               │
               └──→ load_autoencoder_model() ──→ AE object
                                             └───→ cached

──────────────────────────────────────────────────────────────────

points ──→ embed_pca()       ──→ PCA embeddings
          embed_isomap()     ──→ Isomap embeddings
          | embed_autoencoder() ──→ Latent codes

──────────────────────────────────────────────────────────────────

embeddings ──→ reconstruct_pca()       ──→ Ambient points
             │ reconstruct_isomap()     ──→ Reconstructed data
             └ reconstruct_autoencoder() ──→ Reconstructed data

──────────────────────────────────────────────────────────────────

query_point ──→ get_nearest_centroids() ──→ indices (sorted)
            └──→ KDTree search          └──→ distances (sorted)
```

## Method Comparison Matrix

```
┌─────────────────┬─────────────┬──────────────┬──────────────────┐
│ Feature         │ PCA         │ Isomap       │ Autoencoder      │
├─────────────────┼─────────────┼──────────────┼──────────────────┤
│ Model Format    │ .joblib     │ .joblib      │ .pt (PyTorch)    │
│ Embedding Type  │ Linear      │ Geodesic     │ Semi-linear      │
│ Reconstruction  │ Exact       │ Interpolated │ Decoder Network  │
│ GPU Support     │ No          │ No           │ Yes              │
│ Fit Method      │ SVD         │ Manifold     │ Gradient Descent │
│ Training Points │ Not needed  │ Needed (opt) │ Not needed       │
│ Speed           │ Fast        │ Medium       │ Medium-Fast      │
│ Memory Usage    │ Low         │ Medium       │ Medium-High      │
└─────────────────┴─────────────┴──────────────┴──────────────────┘
```

## Module Class Hierarchy

```
AtlasLoader (Main Class)
├── __init__(config, device)
│
├── Centroid Operations
│   ├── load_centroids()
│   └── get_nearest_centroids()
│
├── PCA Methods
│   ├── load_pca_model()
│   ├── embed_pca()
│   └── reconstruct_pca()
│
├── Isomap Methods
│   ├── load_isomap_model()
│   ├── embed_isomap()
│   └── reconstruct_isomap()
│
├── Autoencoder Methods
│   ├── load_autoencoder_model()
│   ├── embed_autoencoder()
│   └── reconstruct_autoencoder()
│
├── Unified Interface
│   ├── embed(method=...)
│   ├── reconstruct(method=...)
│   └── clear_cache()
│
└── Internal Utilities
    ├── _build_kdtree()
    ├── _get_model_cache_key()
    └── _model_cache (dict)
```

## File Organization

```
Manifolds_in_LLMs/
│
├── src/
│   ├── atlas_loader.py ................. Main module (540+ lines)
│   ├── ATLAS_LOADER_README.md ......... Full documentation
│   ├── paths.py (enhanced) ............ Path helpers (+4 functions)
│   └── config_manager.py .............. Configuration (unchanged)
│
├── examples/
│   └── atlas_loader_example.py ........ Example usage (220+ lines)
│
├── results/
│   ├── pca_atlas_50D/50D/
│   │   ├── centroid_0000_pca_50D.joblib
│   │   ├── centroid_0000_embeddings_50D.npy
│   │   └── ...
│   ├── iso_atlas_50D/50D/
│   │   ├── centroid_0000_isomap_50D.joblib
│   │   ├── centroid_0000_embeddings_50D.npy
│   │   └── ...
│   └── autoencoder_atlas_50D/50D/
│       ├── centroid_0000_autoencoder_50D.pt
│       ├── centroid_0000_history_50D.joblib
│       ├── centroid_0000_embeddings_50D.npy
│       └── ...
│
├── ATLAS_LOADER_COMPLETE.md ........... This summary
├── ATLAS_LOADER_IMPLEMENTATION.md ..... Implementation details
└── ATLAS_LOADER_QUICK_REFERENCE.md ... One-page cheat sheet
```

## API Usage Patterns

```
PATTERN 1: Basic Embedding
─────────────────────────────
loader = AtlasLoader(config)
embeddings = loader.embed_pca(points, 0, 50)
reconstructed = loader.reconstruct_pca(embeddings, 0, 50)

PATTERN 2: Unified Interface
─────────────────────────────
for method in ['pca', 'isomap', 'autoencoder']:
    emb = loader.embed(points, 0, method=method, n_components=50)
    rec = loader.reconstruct(emb, 0, method=method, n_components=50)

PATTERN 3: Nearest Centroid + Embed
────────────────────────────────────
distances, indices = loader.get_nearest_centroids(query_point, k=5)
best_centroid = indices[0]
embeddings = loader.embed(points, best_centroid, method='pca')

PATTERN 4: Batch Processing with Memory Management
──────────────────────────────────────────────────
for i in range(200):
    embeddings = loader.embed(points, i, method='pca')
    # ... process ...
    if (i + 1) % 10 == 0:
        loader.clear_cache()

PATTERN 5: Multi-Method Comparison
───────────────────────────────────
results = {}
for method in ['pca', 'isomap', 'autoencoder']:
    try:
        emb = loader.embed(test_points, 0, method=method, n_components=50)
        rec = loader.reconstruct(emb, 0, method=method, n_components=50)
        results[method] = compute_error(test_points, rec)
    except FileNotFoundError:
        results[method] = None
```

## Function Call Graph

```
User Code
   │
   ├─────────► AtlasLoader.__init__()
   │              └─► load_config() (if needed)
   │              └─► initialize cache {}
   │              └─► initialize KDTree (None)
   │
   ├─────────► load_centroids()
   │              └─► np.load() [once, then cached]
   │              └─► return centroids array
   │
   ├─────────► get_nearest_centroids(query, k)
   │              └─► _build_kdtree() [once, then reused]
   │              │     └─► KDTree(centroids)
   │              └─► kdtree.query(query, k)
   │              └─► return (distances, indices)
   │
   ├─────────► embed(method='pca', ...)
   │              └─► embed_pca() [or isomap/autoencoder]
   │                  └─► load_pca_model() [cache check]
   │                  │     └─► joblib.load() [first time]
   │                  │     └─► cache model
   │                  └─► model.transform() / encoder()
   │                  └─► return embeddings
   │
   ├─────────► reconstruct(method='pca', ...)
   │              └─► reconstruct_pca() [or isomap/autoencoder]
   │                  └─► load_pca_model() [from cache]
   │                  └─► model.inverse_transform() / decoder()
   │                  └─► return reconstructed
   │
   └─────────► clear_cache()
                  └─► _model_cache.clear()
```

## Data Shape Transformations

```
POINTS EMBEDDING JOURNEY
────────────────────────

Input Points
└─ Shape: (n_samples, input_dim)
   Example: (1000, 4096)

PCA Embedding
├─ loader.embed_pca(points, 0, 50)
├─ Shape: (1000, 50)
└─ Method: Linear transformation via sklearn PCA

Isomap Embedding
├─ loader.embed_isomap(points, 0, 50)
├─ Shape: (1000, 50)
└─ Method: Geodesic distance preservation

Autoencoder Embedding
├─ loader.embed_autoencoder(points, 0, 50)
├─ Shape: (1000, 50)
└─ Method: Neural network encoder


RECONSTRUCTION JOURNEY
──────────────────────

Low-D Embeddings
└─ Shape: (1000, 50)

PCA Reconstruction
├─ loader.reconstruct_pca(embeddings, 0, 50)
├─ Method: sklearn.inverse_transform()
└─ Shape: (1000, 4096)

Isomap Reconstruction
├─ loader.reconstruct_isomap(embeddings, 0, 50, training_data)
├─ Method: KDTree-based interpolation
└─ Shape: (1000, 4096)

Autoencoder Reconstruction
├─ loader.reconstruct_autoencoder(embeddings, 0, 50)
├─ Method: Neural network decoder
└─ Shape: (1000, 4096)
```

## Performance Characteristics

```
OPERATION                   TIME COMPLEXITY    SPACE COMPLEXITY
──────────────────────────────────────────────────────────────
load_centroids()            O(n)               O(n·d)
Build KDTree                O(n log n)         O(n)
Query K nearest             O(log n)           O(k)
embed_pca()                 O(m·d)             O(m·d)
embed_isomap()              O(m·d)             O(m·d)
embed_autoencoder()         O(m·d)             O(m·d)
reconstruct_pca()           O(m·d)             O(m·d)
reconstruct_isomap()        O(m·k) + O(m·d)    O(m·d)  [k=nearest]
reconstruct_autoencoder()   O(m·d)             O(m·d)

Legend: n=num_centroids, m=num_points, d=dimension, k=k_neighbors
```

## Memory Usage Profile

```
COMPONENT                    MEMORY (approx)
────────────────────────────────────────────
Centroids (200, 4096)        ~3.1 MB
KDTree overhead              ~1.5 MB
PCA Model (one)              ~5-10 MB
Isomap Model (one)           ~5-10 MB
Autoencoder Model (one)      ~50-100 MB (GPU: 50-100 MB VRAM)
Points batch (1000, 4096)    ~16 MB
Embeddings (1000, 50)        ~0.2 MB
─────────────────────────────────────────
Total (one method active)    ~30-50 MB
Total (all cached)           ~100-150 MB
```

## Quick Decision Tree

```
                    AtlasLoader
                        │
        ┌───────────────┼───────────────┬─────────────┐
        │               │               │             │
   Need to Load      Need to Find    Need to        Need to
   Centroids?        K-nearest?      Embed?         Reconstruct?
        │               │               │             │
        │               │               │             │
       YES             YES             YES            YES
        │               │               │             │
        ▼               ▼               ▼             ▼
   load_           get_nearest_      Which          Which
   centroids()     centroids()       method?        method?
                                  │   │   │      │   │   │
                              PCA ISO AE PCA ISO AE
                              │   │   │  │   │   │
                              ▼   ▼   ▼  ▼   ▼   ▼
                            embed_pca     reconstruct_pca
                            embed_isomap  reconstruct_isomap
                            embed_ae      reconstruct_ae
```

## Integration with Existing Code

```
Your Existing Code
├── pca_for_each_centroid.py ──→ Trains PCA models
├── isomap_for_each_centroid.py ─→ Trains Isomap models
└── isometric_autoencoder_for_each_centroid.py ──→ Trains AE models
                │
                └─── Results saved to disk
                        │
                        ▼
                   results/
                   ├── pca_atlas_*/
                   ├── iso_atlas_*/
                   └── autoencoder_atlas_*/
                        │
                        └─── NEW: atlas_loader.py reads these
                                 and provides unified interface
```

---

## Quick Start Command

```bash
# Run the comprehensive example
cd /home/ADV_2526a/ronenrusinov/LLM_manifolds/Manifolds_in_LLMs_refactoring/Manifolds_in_LLMs

python examples/atlas_loader_example.py \
    --method pca \
    --n-components 50 \
    --centroid-idx 0 \
    --n-test-points 10
```

## Documentation Hierarchy

```
Quick Reference (1 page)
    ▲
    │
    ├─── ATLAS_LOADER_QUICK_REFERENCE.md

Implementation Details (5-10 min read)
    ▲
    │
    ├─── ATLAS_LOADER_IMPLEMENTATION.md

Full API Reference (comprehensive)
    ▲
    │
    ├─── src/ATLAS_LOADER_README.md
    │
    └─── Code docstrings in atlas_loader.py

Examples
    ▲
    │
    └─── examples/atlas_loader_example.py
```

---

**Status: ✅ COMPLETE AND READY TO USE**
