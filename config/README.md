# Configuration System Documentation

## Overview

The Manifolds in LLMs project uses a centralized YAML-based configuration system to manage all hyperparameters and numerical constants. This replaces hardcoded values scattered throughout the codebase, enabling flexible experimentation and reproducible runs.

## Configuration Files

### Default Configuration
- **File**: `config/default_config.yaml`
- **Purpose**: Master configuration containing all~115 hyperparameters organized into 10 logical sections
- **Usage**: Base configuration for all scripts; can be overridden via profiles or CLI arguments

### Profile Configurations

#### Quick Test Profile
- **File**: `config/profile_quick_test.yaml`
- **Purpose**: Reduced parameters for fast development and debugging
- **Use When**: Testing code changes, debugging, or running quick validation
- **Key Changes**:
  - Reduced epochs (10 vs 300)
  - Smaller datasets (10% of training data)
  - Fewer clusters (50 vs 200)
  - Minimal gradient accumulation (2 steps)
  - Quick early stopping (5 patience)

#### Production Profile
- **File**: `config/profile_production.yaml`
- **Purpose**: Optimized parameters for full-scale production runs
- **Use When**: Running experiments, generating results for papers, or production deployment
- **Key Changes**: Uses standard defaults optimized for accuracy
  - Full epochs (300-1000)
  - Full datasets
  - All 200 centroids
  - Complete gradient accumulation (6 steps)
  - Extended early stopping (50 patience)

## Configuration Sections

### 1. Model & Dimensionality (`model`)
Controls LLM layer selection and model architecture parameters:
```yaml
model:
  latent_dim: 12              # Autoencoder/Isomap latent space dimension
  layer_for_activation: 18    # Primary LLM layer for extraction
  layer_alternative: 6        # Alternative layer for comparison
  balltree_leaf_size: 40      # BallTree spatial index parameter
```

### 2. Training & Optimization (`training`)
Hyperparameters for model training:
```yaml
training:
  epochs: 300                 # Standard training epochs
  epochs_extended: 1000       # Extended training for convergence
  learning_rate: 1.0e-3       # Adam optimizer learning rate
  patience: 20                # Early stopping patience
  accumulation_steps: 6       # Gradient accumulation for memory
  random_seed: 42             # Reproducibility seed
```

### 3. Clustering & Neighborhoods (`clustering`)
K-nearest neighbors and clustering parameters:
```yaml
clustering:
  n_centroids: 200            # Centroid set size
  n_clusters: 200             # MiniBatchKMeans clusters
  k_nearest_neighbors: 20     # Default for neighbor searches
  k_neighbors_isomap: 10      # Manifold learning neighbors
  k_nearest_10000: 10000      # Large-scale neighbor queries
```

### 4. Data Specifications (`data`)
Dataset loading and splitting parameters:
```yaml
data:
  batch_size: 200000          # KMeans batch size
  train_fraction: 0.5         # Training data percentage
  val_fraction: 0.2           # Validation data percentage
  n_samples_locality: 10      # Samples for locality checks
```

### 5. Dimensionality Reduction (`dimensionality`)
Manifold learning and dimensionality parameters:
```yaml
dimensionality:
  n_components: 12            # Isomap/PCA components (12D standard)
  n_neighbors: 50             # Manifold learning neighbors
  n_components_2d: 2          # 2D visualization
  n_components_3d: 3          # 3D visualization
```

### 6. Text Processing (`text`)
Context window and text extraction:
```yaml
text:
  first_n_words: 20           # Words before activation
  last_n_words: 20            # Words after activation
  first_n_tokens_isomap: 5    # Token context for Isomap
```

### 7. Visualization (`visualization`)
Figure sizes, display parameters:
```yaml
visualization:
  fig_width_standard: 12
  fig_height_standard: 6
  histogram_bins: 50          # Plot resolution
```

### 8. Numerical Constants (`numerical`)
Sentinel values, thresholds:
```yaml
numerical:
  zero_threshold: 1.0e-5      # Numerical tolerance
  sentinel_value: -1          # Missing data marker
  pca_max_samples: 100
```

### 9. Synthetic Data (`synthetic_data`)
Synthetic data generation parameters:
```yaml
synthetic_data:
  noise_level: 0.05           # Noise for synthetic data
  n_noise_dims: 2302          # Added noise dimensions
  n_total_dims: 2304          # Final dimension (2 + 2302)
```

### 10. Logging (`logging`)
Logging and verbosity settings:
```yaml
logging:
  log_interval: 1             # Logging frequency
  verbose_level: 10           # Verbosity level
```

## Usage Examples

### Basic Usage (Default Configuration)
```bash
# Run script with default configuration
python scripts/minibatch_kmeans.py

# Explicitly specify default config
python scripts/minibatch_kmeans.py --config config/default_config.yaml
```

### Using Profiles
```bash
# Run with quick test profile for development
python scripts/minibatch_kmeans.py --config config/profile_quick_test.yaml

# Run with production profile for full experiments
python scripts/minibatch_kmeans.py --config config/profile_production.yaml
```

### Override Individual Parameters
```bash
# Override specific parameters via CLI
python scripts/minibatch_kmeans.py \
  --config config/profile_production.yaml \
  --epochs 500 \
  --n_clusters 100 \
  --learning_rate 5e-4

# Multi-level override: profile + CLI args
python experiments/activation_norms.py \
  --config config/profile_quick_test.yaml \
  --histogram_bins 30 \
  --layer_for_activation 12
```

### SLURM Job Submission
SLURM scripts automatically use the production profile (default):
```bash
# Submit SLURM job (uses production profile)
sbatch slurm/slurmScripts/minibatch_kmeans.slurm

# Override profile in SLURM job
sbatch -C "profile=quick_test" slurm/slurmScripts/minibatch_kmeans.slurm

# Or modify any SLURM script to use specific profile:
python -u $LLM_PROJ_PATH/scripts/minibatch_kmeans.py \
  --config $LLM_PROJ_PATH/config/profile_quick_test.yaml
```

## Creating Custom Profiles

To create a custom profile for specific experiments:

1. Copy a base profile:
```bash
cp config/profile_production.yaml config/profile_my_experiment.yaml
```

2. Modify parameters as needed:
```yaml
# Profile for high-precision dimensionality reduction
dimensionality:
  n_components: 16            # Increased from 12
  n_neighbors: 100            # Increased from 50

training:
  epochs: 500                 # Longer training
  learning_rate: 5e-4         # Lower LR for stability
```

3. Use in scripts:
```bash
python scripts/train_and_save_isomap_12D.py --config config/profile_my_experiment.yaml
```

## Configuration Merging Priority

Parameters are loaded with the following priority (highest wins):

1. **CLI Arguments** (highest priority)
   ```bash
   --epochs 100 --learning_rate 1e-4
   ```

2. **Config File**
   ```bash
   --config config/profile_custom.yaml
   ```

3. **Default Configuration** (lowest priority)
   - Embedded in `config/default_config.yaml`

Example:
```bash
# Priority order for final values:
# learning_rate: 1e-4 (from CLI)
# epochs: 500 (from profile)
# n_clusters: 200 (from default)
python script.py \
  --config config/profile_production.yaml \
  --learning_rate 1e-4
```

## Common Workflow Patterns

### Development Workflow
```bash
# 1. Start with quick test to verify code works
python scripts/minibatch_kmeans.py --config config/profile_quick_test.yaml

# 2. Once working, test with more data
python scripts/minibatch_kmeans.py --config config/default_config.yaml

# 3. Run full experiment with production settings
python scripts/minibatch_kmeans.py --config config/profile_production.yaml
```

### Hyperparameter Tuning
```bash
# Test different learning rates
for lr in 1e-4 5e-4 1e-3; do
  python experiments/autoencoder_with_guidance.py \
    --config config/profile_production.yaml \
    --learning_rate $lr
done

# Test different centroid counts
for n in 100 150 200 250; do
  python scripts/minibatch_kmeans.py \
    --config config/profile_production.yaml \
    --n_clusters $n
done
```

### Reproducible Results
```bash
# Always include random_seed for reproducibility
python script.py \
  --config config/profile_production.yaml \
  --random_seed 42  # Or your specific seed
```

## Integration in Scripts

All refactored scripts follow this pattern:

```python
from config_manager import load_config_with_args

# At script start:
config = load_config_with_args(description="Your script description")

# Use config throughout:
model = Autoencoder(
    input_dim=2304,
    latent_dim=config.model.latent_dim,
    device='cuda'
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.training.learning_rate
)

clustering = MiniBatchKMeans(
    n_clusters=config.clustering.n_clusters,
    batch_size=config.data.batch_size,
    n_init=config.clustering.kmeans_n_init
)
```

## Configuration API

### Python API
```python
from config_manager import load_config, load_config_with_args

# Load default config
config = load_config()

# Load with CLI argument parsing
config = load_config_with_args(description="Experiment description")

# Access parameters
latent_dim = config.model.latent_dim
learning_rate = config.training.learning_rate
n_clusters = config.clustering.n_clusters

# All sections are accessible
print(config.training.epochs)          # 300
print(config.clustering.k_nearest_neighbors)  # 20
```

### Command Line
```bash
# View all available arguments
python script.py --help

# Override any config parameter
python script.py --param_name value

# Dot notation for nested parameters
python script.py --model.latent_dim 16 --training.epochs 500
```

## Parameter Dependencies

Some parameters are interdependent:

- **`n_total_dims`** should equal `n_data_dims + n_noise_dims` (for synthetic data)
- **`latent_dim`** must be less than input dimension (2304)
- **`val_fraction + train_fraction`** should be ≤ 1.0
- **`k_neighbors`** parameters should be ≤ dataset size
- **Visualization dims** (2D, 3D, 4D) are read-only constants

Always verify parameter consistency when creating custom profiles.

## Troubleshooting

### Configuration Not Loading
```bash
# Check config file syntax
python -c "import yaml; yaml.safe_load(open('config/profile_production.yaml'))"

# Verify config path
python script.py --config /path/to/config.yaml --verbose_level 10
```

### Parameter Not Recognized
```bash
# View available parameters
python script.py --help

# Note: use underscores for CLI (n_clusters), dots for Python (config.clustering.n_clusters)
python script.py --n_clusters 100  # CLI
config.clustering.n_clusters       # Python
```

### Results Not Reproducible
```bash
# Ensure random seed is set consistently
python script.py --random_seed 42 --config config/profile_production.yaml
```

## Future Enhancements

Potential improvements to the configuration system:

- [ ] Configuration validation schema (JSON Schema/Pydantic)
- [ ] Automatic configuration documentation generation
- [ ] Configuration diff tool for experiment comparison
- [ ] Web UI for parameter tuning
- [ ] Integration with wandb/mlflow for hyperparameter logging
- [ ] Bayesian optimization integration for hyperparameter search

## Questions & Support

For questions about the configuration system:
1. Check this README
2. View example profiles in `config/`
3. Examine existing scripts using config (e.g., `scripts/minibatch_kmeans.py`)
4. Check `src/config_manager.py` for implementation details
