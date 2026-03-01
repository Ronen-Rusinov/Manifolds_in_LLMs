# Sparse Autoencoder Module

## Overview

This module provides implementation of sparse autoencoders for learning interpretable and efficient latent representations. Sparse autoencoders apply regularization constraints to encourage zero activations in the hidden layers, leading to more interpretable and localized features.

The module includes:
- **SparseAutoencoder**: Standard sparse autoencoder architecture
- **TiedWeightSparseAutoencoder**: Sparse autoencoder with tied encoder-decoder weights
- Multiple sparsity regularization methods: L1, KL divergence, and Hoyer sparseness
- Training functions with early stopping and comprehensive logging
- Evaluation and visualization utilities

## Key Features

### 1. **Flexible Architecture**
- 2-layer encoder and decoder with intermediate hidden layers
- Configurable input and latent dimensions
- ReLU activations for local linearity (consistent with project standards)
- PyTorch-based for GPU support

### 2. **Multiple Sparsity Constraints**

#### L1 Sparsity (Mean Absolute Activation)
- Penalizes the average absolute value of latent activations
- Simple and computationally efficient
- Works well in practice for many applications
```python
loss = torch.mean(torch.abs(z))
```

#### KL Divergence Sparsity
- Compares empirical activation distribution to target sparse distribution
- More sophisticated approach that targets a specific sparsity level
- Better control over actual sparsity achieved
```python
KL(target_sparsity || empirical_sparsity)
```

#### Hoyer Sparseness
- Principled sparseness measure based on L1/L2 ratio
- Ranges from 0 (dense) to 1 (sparse)
- Scale-invariant and widely recognized metric
```python
hoyer = (sqrt(n) - ||z||_1/||z||_2) / (sqrt(n) - 1)
```

### 3. **Row-wise Weight Normalization**
- Decoder weights (basis vectors) are normalized row-wise after each training step
- Ensures each basis vector has unit norm (L2 normalization)
- Standard in sparse autoencoder literature for improved interpretability
- Automatically applied during training (no configuration needed)
- Methods:
  - `SparseAutoencoder.normalize_decoder_weights_row_wise()`
  - `TiedWeightSparseAutoencoder.normalize_weights_row_wise()`

### 4. **Training with Early Stopping**
- Validation-based early stopping to prevent overfitting
- Separate tracking of reconstruction and sparsity losses
- Detailed logging of training progress
- Support for gradient accumulation ready (via standard PyTorch)

### 4. **Tied Weight Variant**
- Encoder weights are transposes of decoder weights
- Reduces model parameters significantly
- Enforces symmetry between encoding and decoding
- Useful for interpretability and geometric constraints

## Installation

The module is part of the project and requires:
```bash
torch >= 1.9.0
numpy
matplotlib
```

## Quick Start

### Basic Usage

```python
import torch
from src.SparseAutoencoder import SparseAutoencoder
from src.train_sparse_autoencoder import train_sparse_autoencoder_with_early_stopping

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model
model = SparseAutoencoder(
    input_dim=200,
    latent_dim=50,
    device=device
)

# Prepare data
train_data = torch.randn(3200, 200, device='cpu')
val_data = torch.randn(800, 200, device='cpu')

# Train
history = train_sparse_autoencoder_with_early_stopping(
    model=model,
    train_data=train_data,
    val_data=val_data,
    num_epochs=500,
    learning_rate=1e-3,
    patience=50,
    sparsity_weight=0.1,
    target_sparsity=0.05,
    sparsity_type='kl',
    device=device
)
```

### Using Different Sparsity Types

```python
# L1 Sparsity
history_l1 = train_sparse_autoencoder_with_early_stopping(
    model, train_data, val_data,
    sparsity_type='l1',
    sparsity_weight=0.01,
    **other_args
)

# KL Divergence Sparsity (recommended)
history_kl = train_sparse_autoencoder_with_early_stopping(
    model, train_data, val_data,
    sparsity_type='kl',
    target_sparsity=0.05,
    sparsity_weight=0.1,
    **other_args
)

# Hoyer Sparseness
history_hoyer = train_sparse_autoencoder_with_early_stopping(
    model, train_data, val_data,
    sparsity_type='hoyer',
    sparsity_weight=1.0,
    **other_args
)
```

### Evaluation

```python
from src.train_sparse_autoencoder import evaluate_sparse_autoencoder

# Evaluate on test set
test_data = torch.randn(1000, 200)
results = evaluate_sparse_autoencoder(model, test_data, device=device)

print(f"Mean reconstruction error: {results['mean']:.6f}")
print(f"Sparsity level: {results['sparsity_level']*100:.2f}%")
print(f"Mean activation: {results['mean_activation_per_neuron']}")
```

### Visualization

```python
from src.train_sparse_autoencoder import (
    plot_sparse_training_history,
    plot_sparse_autoencoder_analysis
)

# Plot training history
plot_sparse_training_history(
    history,
    save_path='training_history.png'
)

# Plot analysis
plot_sparse_autoencoder_analysis(
    results,
    model_name='Sparse Autoencoder',
    save_path='analysis.png'
)
```

## API Reference

### SparseAutoencoder

```python
class SparseAutoencoder(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, device='cpu', dtype=torch.float32)
    def forward(x) -> x_reconstructed
    def encode(x) -> z
    def decode(z) -> x_reconstructed
    def normalize_decoder_weights_row_wise() -> None  # Normalize weights after training step
    @staticmethod
    def l1_sparsity_loss(z, target_sparsity=0.05) -> loss
    @staticmethod
    def kl_divergence_sparsity_loss(z, target_sparsity=0.05, epsilon=1e-10) -> loss
    @staticmethod
    def hoyer_sparsity_loss(z) -> loss
```

### TiedWeightSparseAutoencoder

Similar interface to `SparseAutoencoder` but with tied weights. Reduces parameters by ~2x.

```python
class TiedWeightSparseAutoencoder(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, device='cpu', dtype=torch.float32)
    def forward(x) -> x_reconstructed
    def encode(x) -> z
    def decode(z) -> x_reconstructed
    def normalize_weights_row_wise() -> None  # Normalize encoder weights row-wise
    # ... sparsity loss methods (same as SparseAutoencoder)
```

### train_sparse_autoencoder_with_early_stopping

```python
def train_sparse_autoencoder_with_early_stopping(
    model,
    train_data,
    val_data,
    num_epochs,
    learning_rate,
    patience,
    sparsity_weight=0.1,
    target_sparsity=0.05,
    sparsity_type='kl',
    device='cpu'
) -> history_dict
```

**Returns:**
```python
{
    'train_loss': list,              # Total training losses
    'train_recon_loss': list,        # Reconstruction losses
    'train_sparsity_loss': list,     # Sparsity losses
    'val_loss': list,                # Total validation losses
    'val_recon_loss': list,          # Validation reconstruction losses
    'val_sparsity_loss': list,       # Validation sparsity losses
    'epochs_trained': int            # Number of epochs completed
}
```

### evaluate_sparse_autoencoder

```python
def evaluate_sparse_autoencoder(model, test_data, device='cpu') -> results_dict
```

**Returns:**
```python
{
    'errors': np.ndarray,                    # Per-sample reconstruction errors
    'mean': float,                           # Mean reconstruction error
    'std': float,                            # Std of reconstruction errors
    'latent_codes': np.ndarray,              # Latent representations of test data
    'mean_activation_per_neuron': np.ndarray,# Mean activation per latent unit
    'sparsity_level': float                  # Fraction of near-zero activations
}
```

## Weight Normalization

### Row-wise Normalization Strategy

Decoder weights are automatically normalized row-wise (L2 normalization) **during each forward pass**. This ensures the normalization is part of the differentiable computation graph so gradients flow through it correctly. 

Benefits:
- **Ensures each basis vector has unit norm**: Each row of the decoder weight matrix represents a learned basis vector that is normalized
- **Improves interpretability**: Makes basis vectors directly comparable and easier to visualize
- **Enables proper gradient flow**: Normalization happens in forward() so autograd captures it
- **Standard practice**: Recommended in sparse autoencoder literature

### How It Works

For `SparseAutoencoder`:
```python
# In forward(), before computing reconstruction:
# Each row of the final decoder layer (latent_dim -> input_dim) is normalized
W = decoder_final_layer.weight
W_norm = ||W[i]||_2 for each row i
W_normalized = W / W_norm
x_recon = h @ W_normalized.T + bias
```

For `TiedWeightSparseAutoencoder`:
```python
# In decoder_forward():
# The encoder_mat_1.T is normalized row-wise (columns of encoder_mat_1)
decoder_weights = encoder_mat_1.T  # (input_dim, latent_dim) shape
W_norm = ||decoder_weights[i]||_2 for each row i
W_normalized = decoder_weights / W_norm
x_recon = h @ W_normalized.T + bias
```

### Manual Control

The normalization happens automatically in forward(), but you can manually normalize weights if needed (e.g., for inspection or special cases):
```python
model = SparseAutoencoder(input_dim=50, latent_dim=100)
# ... after training or at any point ...
model.normalize_decoder_weights_row_wise()  # SparseAutoencoder
# or
model.normalize_weights_row_wise()  # TiedWeightSparseAutoencoder
```

This can be useful if you want to check or adjust the current state of weights without forward pass.

## Hyperparameter Tuning

### Sparsity Weight
- Controls the trade-off between reconstruction quality and sparsity
- Typical range: 0.01 - 1.0
- Higher values enforce stronger sparsity but may hurt reconstruction
- Start with 0.1 and adjust based on results

### Target Sparsity (KL only)
- Target activation probability for each neuron
- Typical values: 0.01 - 0.1 (1% - 10%)
- 0.05 (5%) is a common starting point
- Lower targets create sparser representations

### Sparsity Type Selection
- **L1**: Use when you want a simple, interpretable penalty
- **KL**: Recommended when you need fine control over sparsity level
- **Hoyer**: Use for scale-invariant sparsity measurement

### Learning Rate
- Typical values: 1e-4 to 1e-2
- Start with 1e-3 and adjust if training is unstable
- With sparsity constraints, may need slightly lower rates

## Example Scripts

### Running the Main Example

```bash
cd examples/
python sparse_autoencoder_example.py
```

This script demonstrates:
1. Basic sparse autoencoder training
2. Comparison of sparsity types (L1, KL, Hoyer)
3. Tied-weight sparse autoencoder
4. Effect of sparsity weight

## Integration with Existing Code

The sparse autoencoder module is designed to integrate seamlessly with existing autoencoders in the project:

- Similar architecture to `StandardAutoencoder` and `TiedWeightAutoencoder`
- Compatible training loop patterns from `train_autoencoder.py`
- Consistent layer sizing and activation functions
- Supports same device and dtype specifications

## Performance Considerations

### Memory Usage
- Standard sparse autoencoder: ~(input_dim + latent_dim) × hidden_dim parameters
- Tied-weight variant: ~50% fewer parameters
- Latent codes storage: batch_size × latent_dim

### Computational Cost
- L1 sparsity: Minimal overhead, O(latent_dim) per batch
- KL divergence: Slightly higher, O(latent_dim) with logs
- Hoyer sparseness: Highest cost, O(latent_dim) with matrix ops
- Training time typically 1.5-2x standard autoencoder

### Optimization Tips
1. Use tied weights for parameter efficiency
2. Start with KL divergence for predictable sparsity
3. Consider L1 for faster training
4. Use early stopping to avoid overfitting
5. Batch normalization before sparsity (if needed)

## Troubleshooting

### High Reconstruction Error
- Reduce `sparsity_weight` to allow better reconstruction
- Increase `latent_dim`
- Train for more epochs

### Not Sparse Enough
- Increase `sparsity_weight`
- Lower `target_sparsity` (for KL)
- Try Hoyer sparseness instead

### Training Instability
- Reduce learning rate
- Use gradient clipping: `torch.nn.utils.clip_grad_norm_`
- Ensure proper weight initialization

### Dead Neurons (all near-zero)
- Reduce `sparsity_weight`
- Use L1 instead of KL divergence
- Increase learning rate slightly

## References

1. Sparse autoencoders in deep learning
2. Hoyer, P. O. (2004). Non-negative matrix factorization with sparseness constraints
3. Andrew Ng's sparse autoencoder tutorial
4. Bengio, Y., et al. (2013). Challenges in representation learning

## Future Enhancements

- [ ] Batch normalization support
- [ ] Layer-wise sparsity control
- [ ] Gradient checkpointing for memory efficiency
- [ ] Integration with geometric regularization
- [ ] Sparse activation functions (Soft Threshold, Hard Threshold)

## License

Part of the Manifolds in LLMs project.
