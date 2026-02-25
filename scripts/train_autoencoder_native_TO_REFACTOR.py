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
from config_manager import load_config, add_config_argument
import argparse

def main(config):
    print("Loading all activations...")
    start_time = time.time()
    train_df, val_df, test_df = load_train_test_val_all_parquets(timing=True)
    print(f"Total time to load: {time.time() - start_time:.2f}s")
    print(f"Train DataFrame shape: {train_df.shape}")
    print(f"Validation DataFrame shape: {val_df.shape}")
    print(f"Test DataFrame shape: {test_df.shape}")

    # Extract activations from configured layer (using float32 for numerical stability)
    layer = config.model.layer_for_activation
    column_name = f'activation_layer_{layer}'
    if column_name not in train_df.columns:
        raise ValueError(f"Column '{column_name}' not found in data. Available columns: {list(train_df.columns)}")
    train_activations = np.array(train_df[column_name].tolist(), dtype=np.float32)
    val_activations = np.array(val_df[column_name].tolist(), dtype=np.float32)
    test_activations = np.array(test_df[column_name].tolist(), dtype=np.float32)
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
    latent_dim = config.model.latent_dim
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
        num_epochs=config.training.epochs_extended,
        learning_rate=config.training.learning_rate_alt,
        patience=config.training.patience_extended,
        accumulation_steps=config.training.accumulation_steps  # Split data into 6 partitions
    )

    print("Training complete!")
    
    # Evaluate on test data using the built-in batch prediction method
    print("Evaluating on test set...")
    reconstructions = autoencoder.predict_in_batches(
        test_activations, 
        accumulation_steps=config.training.accumulation_steps
    )
    
    # Calculate reconstruction errors
    reconstruction_errors = np.mean((reconstructions - test_activations) ** 2, axis=1)
    reconstruction_errors_mean = np.mean(reconstruction_errors)
    print(f"Mean reconstruction error on test set: {reconstruction_errors_mean:.6f}")
    reconstruction_errors_std = np.std(reconstruction_errors)
    print(f"Standard deviation of reconstruction error on test set: {reconstruction_errors_std:.6f}")

    # Save histogram
    import matplotlib.pyplot as plt
    plt.hist(reconstruction_errors, bins=config.visualization.histogram_bins_alt)
    plt.title(f"Reconstruction Error Distribution on Test Set\nMean: {reconstruction_errors_mean:.6f}, Std: {reconstruction_errors_std:.6f}")
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Frequency")
    plt.grid(True)
    hist_path = Path(__file__).parent.parent / "results" / "autoencoder" / "reconstruction_error_histogram_native_long_train.png"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(hist_path)
    print(f"Reconstruction error histogram saved to {hist_path}")

    # Save the model
    model_path = Path(__file__).parent.parent / "results" / "autoencoder" / "autoencoder_native_long_train.pt"
    torch.save(autoencoder.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train autoencoder with native PyTorch gradient accumulation")
    
    # Autoencoder parameters
    parser.add_argument("--latent_dim", type=int, help="Latent dimension for autoencoder")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--learning_rate", "--lr", type=float, help="Learning rate")
    parser.add_argument("--patience", type=int, help="Patience for early stopping")
    parser.add_argument("--accumulation_steps", type=int, help="Gradient accumulation steps")
    parser.add_argument("--random_seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--train_fraction", type=float, help="Fraction of data for training")
    parser.add_argument("--layer_for_activation", type=int, help="Layer index for activation extraction")
    
    add_config_argument(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    
    # Override config with CLI arguments
    if args.latent_dim is not None:
        config.model.latent_dim = args.latent_dim
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.patience is not None:
        config.training.patience = args.patience
    if args.accumulation_steps is not None:
        config.training.accumulation_steps = args.accumulation_steps
    if args.random_seed is not None:
        config.training.random_seed = args.random_seed
    if args.train_fraction is not None:
        config.data.train_fraction = args.train_fraction
    if args.layer_for_activation is not None:
        config.model.layer_for_activation = args.layer_for_activation
    
    main(config)
