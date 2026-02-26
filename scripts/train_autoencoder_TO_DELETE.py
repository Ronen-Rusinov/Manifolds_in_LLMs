
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
from standard_autoencoder import StandardAutoencoder
from config_manager import load_config, add_config_argument
import argparse

def main(config):
    print("Loading all activations...")
    start_time = time.time()
    train_df,val_df,test_df = load_train_test_val_all_parquets(timing=True)
    train_df = test_df.sample(frac=config.data.train_fraction, random_state=config.training.random_seed)  # Use only specified fraction of training data
    val_df = val_df.sample(frac=config.data.train_fraction, random_state=config.training.random_seed)  # Use only specified fraction of validation data 
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

    #Print devicde name is used in cuda is available:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Initialize and train the autoencoder
    input_dim = train_activations.shape[1]
    latent_dim = config.model.latent_dim
    print(f"Initializing StandardAutoencoder with input_dim={input_dim} and latent_dim={latent_dim}")
    autoencoder = StandardAutoencoder(input_dim=input_dim, latent_dim=latent_dim, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16)

    # Train with early stopping
    print("Training autoencoder with early stopping...")
    train_tensor = torch.from_numpy(train_activations).to(autoencoder.device)
    #print train tensor info
    print(f"Train tensor shape: {train_tensor.shape}, dtype: {train_tensor.dtype}")
    val_tensor = torch.from_numpy(val_activations).to(autoencoder.device)
    print(f"Validation tensor shape: {val_tensor.shape}, dtype: {val_tensor.dtype}")
    autoencoder.train(train_tensor, val_data=val_tensor, num_epochs=config.training.epochs, learning_rate=config.training.learning_rate, patience=config.training.patience)
    
    #remove the train and val data from gpu memory
    del train_tensor
    del val_tensor

    print("Training complete!")
    #save histogram of reconstruction errors on test data
    test_tensor = torch.from_numpy(test_activations).to(autoencoder.device)
    with torch.no_grad():
        reconstructions = autoencoder(test_tensor.to(autoencoder.device)).cpu().numpy()
    reconstruction_errors = np.mean((reconstructions - test_activations) ** 2, axis=1)
    reconstruction_errors_mean = np.mean(reconstruction_errors)
    print(f"Mean reconstruction error on test set: {reconstruction_errors_mean:.6f}")
    reconstruction_errors_std = np.std(reconstruction_errors)
    print(f"Standard deviation of reconstruction error on test set: {reconstruction_errors_std:.6f}")

    import matplotlib.pyplot as plt
    plt.hist(reconstruction_errors, bins=config.visualization.histogram_bins)
    plt.title(f"Reconstruction Error Distribution on Test Set - Mean: {reconstruction_errors_mean:.6f}, Std: {reconstruction_errors_std:.6f}")
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Frequency")
    plt.grid(True)
    hist_path = Path(__file__).parent.parent / "results" / "autoencoder" / "reconstruction_error_histogram.png"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(hist_path)
    print(f"Reconstruction error histogram saved to {hist_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train standard autoencoder with early stopping")
    
    # Autoencoder parameters
    parser.add_argument("--latent_dim", type=int, help="Latent dimension for autoencoder")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--learning_rate", "--lr", type=float, help="Learning rate")
    parser.add_argument("--patience", type=int, help="Patience for early stopping")
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
    if args.random_seed is not None:
        config.training.random_seed = args.random_seed
    if args.train_fraction is not None:
        config.data.train_fraction = args.train_fraction
    if args.layer_for_activation is not None:
        config.model.layer_for_activation = args.layer_for_activation
    
    main(config)