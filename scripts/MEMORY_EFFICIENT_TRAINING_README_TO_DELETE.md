# Memory-Efficient Autoencoder Training

This directory contains memory-efficient implementations for training autoencoders on large datasets that don't fit in GPU memory.

## Problem

The original `train_autoencoder.py` had two issues:
1. **Using float16 causes numerical overflow** during training
2. **Using float32 causes GPU out-of-memory errors** because the entire dataset is loaded to GPU at once

## Solution: Gradient Accumulation

Both new implementations use **gradient accumulation** to solve these problems:
- Use **float32** to avoid overflow
- Keep data on **CPU** and load only small partitions to GPU
- At each epoch:
  1. Shuffle the data
  2. Partition it into 6 sections
  3. Load one section at a time to GPU
  4. Calculate gradients and accumulate them
  5. After all sections, update weights with accumulated gradients

## Files Created

### 1. `train_autoencoder_gradient_accumulation.py`
**Manual implementation** with explicit gradient accumulation function.

**Usage:**
```bash
python scripts/train_autoencoder_gradient_accumulation.py
```

**Features:**
- Custom `train_with_gradient_accumulation()` function
- Clear step-by-step implementation
- Easy to understand and modify
- Validation data also processed in partitions

### 2. `train_autoencoder_native.py` + `src/standard_autoencoder_memory_efficient.py`
**PyTorch-native implementation** with gradient accumulation built into the model class.

**Usage:**
```bash
python scripts/train_autoencoder_native.py
```

**Features:**
- `MemoryEfficientAutoencoder` class with built-in `train_with_accumulation()` method
- Cleaner API - just call the training method
- Includes `predict_in_batches()` for memory-efficient inference
- Uses PyTorch's native gradient accumulation pattern

## How Gradient Accumulation Works in PyTorch

PyTorch natively supports gradient accumulation through this pattern:

```python
optimizer.zero_grad()  # Clear gradients at start of epoch

for partition in partitions:
    output = model(partition)
    loss = criterion(output, partition)
    loss = loss / num_partitions  # Scale loss
    loss.backward()  # Gradients accumulate in parameter.grad
    # DON'T call optimizer.zero_grad() here!

optimizer.step()  # Update weights using accumulated gradients
```

The key insight: **PyTorch automatically accumulates gradients** when you call `backward()` multiple times without calling `zero_grad()`.

## Comparison

| Feature | Manual Implementation | Native Implementation |
|---------|----------------------|----------------------|
| **File** | `train_autoencoder_gradient_accumulation.py` | `train_autoencoder_native.py` |
| **Approach** | Standalone function | Built into model class |
| **Code reusability** | Need to copy function | Import and use class |
| **Inference** | Manual batching needed | `predict_in_batches()` method |
| **Customization** | Very easy | Requires modifying class |

## Key Parameters

Both scripts use these parameters:
- `num_epochs=300` - Maximum training epochs
- `learning_rate=1e-3` - Adam optimizer learning rate
- `patience=20` - Early stopping patience
- `num_partitions=6` or `accumulation_steps=6` - Number of data partitions for gradient accumulation

## Memory Savings

With 6 partitions, **GPU memory usage is reduced by ~6x** compared to loading all data at once.

You can increase/decrease the number of partitions based on your GPU memory:
- More partitions = less GPU memory but slower training
- Fewer partitions = more GPU memory but faster training

## Results

Both implementations produce the same results and save:
- Trained model weights (`.pt` file)
- Reconstruction error histogram (`.png` file)
