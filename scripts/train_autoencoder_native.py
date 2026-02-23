import sys
import os
from pathlib import Path
import numpy as np
import joblib
import time
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.load_data import load_train_test_val_all_parquets, load_all_parquets
from standard_autoencoder_memory_efficient import MemoryEfficientAutoencoder

def main():
    print("Loading all activations...")
    start_time = time.time()
    train_df, val_df, test_df = load_train_test_val_all_parquets(timing=True)
    train_df = train_df.sample(frac=0.5, random_state=42)  # Use only 50% of the training data
    val_df = val_df.sample(frac=0.5, random_state=42)  # Use only 50% of the validation data 
    print(f"Total time to load: {time.time() - start_time:.2f}s")
    print(f"Train DataFrame shape: {train_df.shape}")
    print(f"Validation DataFrame shape: {val_df.shape}")
    print(f"Test DataFrame shape: {test_df.shape}")

    # Use float32 instead of float16 to avoid overflow
    train_activations = np.array(train_df['activation_layer_18'].tolist(), dtype=np.float32)
    val_activations = np.array(val_df['activation_layer_18'].tolist(), dtype=np.float32)
    test_activations = np.array(test_df['activation_layer_18'].tolist(), dtype=np.float32)
    print(f"Train activations shape: {train_activations.shape}")
    print(f"Validation activations shape: {val_activations.shape}")
    print(f"Test activations shape: {test_activations.shape}")

    print(f"total data: {train_activations.shape[0] + val_activations.shape[0] + test_activations.shape[0]}")

    # Print device name if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Initialize the memory-efficient autoencoder with float32
    input_dim = train_activations.shape[1]
    latent_dim = 12 
    print(f"Initializing MemoryEfficientAutoencoder with input_dim={input_dim} and latent_dim={latent_dim}")
    autoencoder = MemoryEfficientAutoencoder(
        input_dim=input_dim, 
        latent_dim=latent_dim, 
        device='cuda' if torch.cuda.is_available() else 'cpu', 
        dtype=torch.float32  # Use float32 to avoid overflow
    )

    # Train with PyTorch's native gradient accumulation
    print("Training autoencoder with PyTorch native gradient accumulation...")
    print("This uses PyTorch's built-in gradient accumulation by calling")
    print("loss.backward() multiple times before optimizer.step()")
    autoencoder.train_with_accumulation(
        data=train_activations,      # Keep on CPU, loaded to GPU in batches
        val_data=val_activations,    # Keep on CPU, loaded to GPU in batches
        num_epochs=300,
        learning_rate=1e-3,
        patience=20,
        accumulation_steps=6  # Split data into 6 partitions
    )

    print("Training complete!")
    
    # Evaluate on test data using the built-in batch prediction method
    print("Evaluating on test set...")
    reconstructions = autoencoder.predict_in_batches(
        test_activations, 
        accumulation_steps=6
    )
    
    # Calculate reconstruction errors
    reconstruction_errors = np.mean((reconstructions - test_activations) ** 2, axis=1)
    reconstruction_errors_mean = np.mean(reconstruction_errors)
    print(f"Mean reconstruction error on test set: {reconstruction_errors_mean:.6f}")
    reconstruction_errors_std = np.std(reconstruction_errors)
    print(f"Standard deviation of reconstruction error on test set: {reconstruction_errors_std:.6f}")

    # Save histogram
    import matplotlib.pyplot as plt
    plt.hist(reconstruction_errors, bins=50)
    plt.title(f"Reconstruction Error Distribution on Test Set\nMean: {reconstruction_errors_mean:.6f}, Std: {reconstruction_errors_std:.6f}")
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Frequency")
    plt.grid(True)
    hist_path = Path(__file__).parent.parent / "results" / "autoencoder" / "reconstruction_error_histogram_native.png"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(hist_path)
    print(f"Reconstruction error histogram saved to {hist_path}")

    # Save the model
    model_path = Path(__file__).parent.parent / "results" / "autoencoder" / "autoencoder_native.pt"
    torch.save(autoencoder.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
