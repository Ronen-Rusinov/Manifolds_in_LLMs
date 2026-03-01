# Training Sparse Autoencoders on Embeddings

## Overview

The `train_sparse_autoencoder_on_embeddings.py` script trains sparse autoencoders on the low-dimensional embeddings produced by the dimensionality reduction scripts:
- `pca_for_each_centroid.py`
- `isomap_for_each_centroid.py`
- `isometric_autoencoder_for_each_centroid.py`

This allows you to learn even sparser, more interpretable representations of the already-reduced embeddings.

## Usage

### Basic Usage

```bash
# Train on PCA embeddings with default settings
python scripts/train_sparse_autoencoder_on_embeddings.py \
    --source pca \
    --n-components 50

# Train on Isomap embeddings
python scripts/train_sparse_autoencoder_on_embeddings.py \
    --source isomap \
    --n-components 50

# Train on Autoencoder embeddings
python scripts/train_sparse_autoencoder_on_embeddings.py \
    --source autoencoder \
    --n-components 50
```

### Advanced Usage

#### Different Embedding Dimensionalities

```bash
# Train on 12D embeddings (from config default)
python scripts/train_sparse_autoencoder_on_embeddings.py \
    --source pca \
    --n-components 12

# Train on 100D embeddings
python scripts/train_sparse_autoencoder_on_embeddings.py \
    --source isomap \
    --n-components 100
```

#### Configurable Sparse Autoencoder Parameters

```bash
# Default: Heavy sparsity with L1 (latent_dim = 2x n_components, sparsity_weight = 0.5)
python scripts/train_sparse_autoencoder_on_embeddings.py \
    --source pca \
    --n-components 50
    # --latent-dim defaults to 100
    # --sparsity-weight defaults to 0.5

# Fine-tuned KL sparsity with specific target
python scripts/train_sparse_autoencoder_on_embeddings.py \
    --source pca \
    --n-components 50 \
    --latent-dim 150 \
    --sparsity-type kl \
    --target-sparsity 0.05 \
    --sparsity-weight 0.3

# Very heavy sparsity using Hoyer sparseness
python scripts/train_sparse_autoencoder_on_embeddings.py \
    --source isomap \
    --n-components 100 \
    --latent-dim 250 \
    --sparsity-type hoyer \
    --sparsity-weight 1.0
```

#### Batch Processing with Offset/Count

```bash
# Process only first 50 centroids
python scripts/train_sparse_autoencoder_on_embeddings.py \
    --source pca \
    --n-components 50 \
    --offset 0 \
    --count 50

# Process centroids 100-150
python scripts/train_sparse_autoencoder_on_embeddings.py \
    --source pca \
    --n-components 50 \
    --offset 100 \
    --count 50

# Process all centroids starting from 50
python scripts/train_sparse_autoencoder_on_embeddings.py \
    --source pca \
    --n-components 50 \
    --offset 50
    # --count defaults to remaining centroids
```

#### Training Hyperparameters

```bash
# Custom learning rate and early stopping
python scripts/train_sparse_autoencoder_on_embeddings.py \
    --source pca \
    --n-components 50 \
    --learning-rate 5e-4 \
    --epochs 500 \
    --patience 60

# Tied weight architecture (parameter efficient)
python scripts/train_sparse_autoencoder_on_embeddings.py \
    --source pca \
    --n-components 50 \
    --latent-dim 25 \
    --tied-weights
```

#### Combined Example

```bash
# Full-featured example with all custom parameters
python scripts/train_sparse_acustom parameters
python scripts/train_sparse_autoencoder_on_embeddings.py \
    --source isomap \
    --n-components 50 \
    --latent-dim 150 \
    --sparsity-type kl \
    --target-sparsity 0.05 \
    --sparsity-weight 0.\
    --epochs 400 \
    --patience 50 \
    --offset 0 \
    --count 100 \
    --tied-weights \
    --random-seed 42 \
    --config config/default_config.yaml
```

## Command-Line Arguments

### Embedding Source
- `--source` {pca, isomap, autoencoder}: Type of embeddings to load (default: pca)
- `--n-components`: Dimensionality of embeddings to load (required or from config)

### Model Architecture
- `--latent-dim`: Latent dimension of sparse autoencoder (default: 2x embedding dimension, ensuring overcomplete representation)
- `--tied-weights`: Use tied weight architecture (reduces parameters ~50%)

### Sparsity Regularization
- `--sparsity-type` {l1, kl, hoyer}: Type of sparsity penalty (default: l1)
- `--sparsity-weight`: Weight of sparsity loss in total loss (default: 2.0)
- `--target-sparsity`: Target activation probability for KL divergence (default: 0.05, only for KL)

### Training Configuration
- `--learning-rate`: Learning rate for Adam optimizer (default: 1e-3)
- `--epochs`: Maximum number of training epochs (default: 10000)
- `--patience`: Early stopping patience in epochs (default: 100)

### Data Processing
- `--offset`: Starting centroid index (default: 0)
- `--count`: Number of centroids to process (default: all remaining)

### Other
- `--random-seed`: Random seed for reproducibility
- `--config`: Path to YAML config file

## Output

The script creates a results directory with the following structure:

```
results/sparse_autoencoder_{source}_atlas_{n_components}D_latent_{latent_dim}D/
├── sparse_autoencoder_model.pt          # Trained model weights
├── training_history.npy                 # Training curves (pickle format)
├── evaluation_results.npy               # Evaluation metrics (pickle format)
├── config.txt                           # Configuration summary
├── training_history.png                 # Training loss plots
└── analysis.png                         # Evaluation analysis plots
```

### Files Explained

**sparse_autoencoder_model.pt**: PyTorch state dictionary of the trained model. Load with:
```python
from src.SparseAutoencoder import SparseAutoencoder
model = SparseAutoencoder(input_dim=50, latent_dim=20)
model.load_state_dict(torch.load('sparse_autoencoder_model.pt'))
```

**training_history.npy**: Dictionary containing:
- `train_loss`, `train_recon_loss`, `train_sparsity_loss`
- `val_loss`, `val_recon_loss`, `val_sparsity_loss`

Load with:
```python
history = np.load('training_history.npy', allow_pickle=True).item()
```

**evaluation_results.npy**: Dictionary containing:
- `reconstruction_errors`: Per-sample test MSE
- `mean_error`, `std_error`: Statistics
- `sparsity_level`: Fraction of near-zero activations
- `mean_activation_per_neuron`: Mean activation per latent unit

Load with:
```python
results = np.load('evaluation_results.npy', allow_pickle=True).item()
```

## Examples

### Example 1: Quick PCA-on-PCA Training

Train a sparse autoencoder on the first 50 centroids' PCA embeddings:

```bash
python scripts/train_sparse_autoencoder_on_embeddings.py \
    --source pca \
    --n-components 50 \
    --offset 0 \
    --count 50
```

Output: `results/sparse_autoencoder_pca_atlas_50D_latent_20D/`

### Example 2: Super-Sparse Features

Learn very sparse representations with Hoyer sparseness:

```bash
python scripts/train_sparse_autoencoder_on_embeddings.py \
    --source isomap \
    --n-components 100 \
    --latent-dim 30 \
    --sparsity-type hoyer \
    --sparsity-weight 2.0
```

### Example 3: Processing Multiple Batches

Process embeddings in batches (e.g., for memory efficiency on large datasets):

```bash
# Batch 1: Centroids 0-100
python scripts/train_sparse_autoencoder_on_embeddings.py \
    --source pca \
    --n-components 50 \
    --offset 0 \
    --count 100

# Batch 2: Centroids 100-200
python scripts/train_sparse_autoencoder_on_embeddings.py \
    --source pca \
    --n-components 50 \
    --offset 100 \
    --count 100

# Analyze both separately or combine results
```

### Example 4: Feature Extraction from Learned Representation

After training, extract sparse features:

```python
import torch
import numpy as np
from src.SparseAutoencoder import SparseAutoencoder

# Load model
model = SparseAutoencoder(input_dim=50, latent_dim=20)
model.load_state_dict(torch.load('results/.../sparse_autoencoder_model.pt'))
model.eval()

# Load an embedding
embedding = np.load('results/pca_atlas_50D/50D/centroid_0000_embeddings_50D.npy')

# Extract active features
features = model.extract_active_features(embedding[0])
print(f"Active features: {features['active_features']}")
print(f"Basis vectors of active features: {features['basis_vectors_all']}")

# Get feature contributions
contributions = model.get_feature_contribution(embedding[0])
print(f"Contribution sum: {contributions['feature_contributions'][0]['contribution_sum']}")
```

## Data Flow

```
PCA/Isomap/Autoencoder outputs
    ↓
pca_for_each_centroid.py / isomap_for_each_centroid.py / isometric_autoencoder_for_each_centroid.py
    ↓
results/{source}_atlas_{n_components}D/{n_components}D/centroid_{idx}_embeddings_{n_components}D.npy
    ↓
train_sparse_autoencoder_on_embeddings.py
    ├─ Load embeddings from multiple centroids
    ├─ Split into train/val/test
    ├─ Train sparse autoencoder
    ├─ Evaluate on test set
    └─ Save model & results
    ↓
results/sparse_autoencoder_{source}_atlas_{n_components}D_latent_{latent_dim}D/
```

## Implementation Details

### Data Loading
The script loads embeddings from subdirectories following the naming convention established by the for_each_centroid scripts:
- Pattern: `{source}_atlas_{n_components}D/{n_components}D/centroid_{index:04d}_embeddings_{n_components}D.npy`
- Handles missing embeddings gracefully with warnings
- Combines embeddings from specified centroid range

### Data Splitting
- Train: 70% (default)
- Validation: 15% (default)
- Test: 15% (default)
- Uses random shuffling for unbiased split

### Training
- Adam optimizer with configurable learning rate
- Validation-based early stopping
- Separate tracking of reconstruction and sparsity losses
- GPU support when available

### Sparsity Types

**L1**: Mean absolute activation
- Simplest to implement
- Minimal computational overhead
- Use when: You want a fast baseline

**KL Divergence** (recommended):
- Compares to target sparsity level
- More principled approach
- Better control over actual sparsity achieved
- Use when: You need predictable sparsity

**Hoyer Sparseness**: Scale-invariant measure (L1/L2 ratio)
- Most principled metric
- Scale-invariant
- Higher computational cost
- Use when: You need robust sparsity measurement

### Default Configuration Strategy

The script is configured by default for **overcomplete sparse autoencoders**:

- **Latent Dimension**: Automatically set to **2x the embedding dimension** (unless overridden)
  - E.g., 12D embeddings → 24D latent space
  - This creates an overcomplete basis where each dimension can be sparse
  - Allows the model to learn diverse sparse features

- **Sparsity Weight**: Default **0.5** (heavy sparsity enforcement)
  - Strong penalty on latent activations
  - Encourages most neurons to be inactive most of the time
  - Learned features are individually sparse and specialized

- **Sparsity Type**: Default **L1**
  - Simple, interpretable penalty
  - Computationally efficient
  - Encourages mean activations to be low across all features

This default configuration is ideal for learning **interpretable, sparse feature representations** from low-dimensional embeddings.

## Performance Considerations

### Memory Usage
- Embeddings loaded from disk (not all in memory simultaneously)
- Model: ~(input_dim + latent_dim) × hidden_dim parameters
- Tied weights reduce params ~50%
- Typical scenarios: <1GB RAM

### Training Time
- L1 sparsity: Fastest (~1x baseline autoencoder)
- KL sparsity: Moderate (~1.2x baseline)
- Hoyer sparseness: Slowest (~1.5x baseline)
- GPU: ~2-5 minutes per 10k embeddings (typical)
- CPU: ~10-20 minutes per 10k embeddings (typical)

### Optimization Tips
1. Use `--tied-weights` for parameter efficiency
2. Start with L1 or KL for initial experiments
3. Use early stopping (default patience=100) to avoid overfitting
4. Process in batches with --offset/--count if memory is limited
5. Lower sparsity_weight if reconstruction quality is poor
6. Increase sparsity_weight if not sparse enough

## Troubleshooting

### Problem: "No embeddings found"
- Check that source and n_components are correct
- Verify the embedding files exist: `results/{source}_atlas_{n_components}D/{n_components}D/`
- Ensure the for_each_centroid script was run first

### Problem: High reconstruction error
- Lower `--sparsity-weight` to prioritize reconstruction
- Increase `--latent-dim`
- Train for more epochs (increase `--epochs`)

### Problem: Not sparse enough
- Increase `--sparsity-weight`
- Lower `--target-sparsity` (for KL only)
- Try Hoyer sparseness instead of L1/KL

### Problem: Training too slow
- Use `--tied-weights` to reduce params
- Decrease `--latent-dim`
- Use L1 instead of KL or Hoyer
- Process smaller batches with `--count`

## See Also

- [SparseAutoencoder Documentation](../src/SPARSE_AUTOENCODER_README.md)
- Related scripts: `pca_for_each_centroid.py`, `isomap_for_each_centroid.py`, `isometric_autoencoder_for_each_centroid.py`
