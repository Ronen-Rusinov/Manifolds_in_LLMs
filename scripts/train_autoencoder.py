
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

def main():
    print("Loading all activations...")
    start_time = time.time()
    train_df,val_df,test_df = load_train_test_val_all_parquets(timing=True)
    train_df = test_df.sample(frac=0.5, random_state=42)  # Use only 50% of the training data Cause otherwise GPU explodes
    val_df = val_df.sample(frac=0.5, random_state=42)  # Use only 50% of the validation data 
    print(f"Total time to load: {time.time() - start_time:.2f}s")
    print(f"Train DataFrame shape: {train_df.shape}")
    print(f"Validation DataFrame shape: {val_df.shape}")
    print(f"Test DataFrame shape: {test_df.shape}")

    train_activations = np.array(train_df['activation_layer_18'].tolist(), dtype=np.float16)
    val_activations = np.array(val_df['activation_layer_18'].tolist(), dtype=np.float16)
    test_activations = np.array(test_df['activation_layer_18'].tolist(), dtype=np.float16)
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
    latent_dim = 12 
    print(f"Initializing StandardAutoencoder with input_dim={input_dim} and latent_dim={latent_dim}")
    autoencoder = StandardAutoencoder(input_dim=input_dim, latent_dim=latent_dim, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16)

    # Train with early stopping
    print("Training autoencoder with early stopping...")
    train_tensor = torch.from_numpy(train_activations).to(autoencoder.device)
    #print train tensor info
    print(f"Train tensor shape: {train_tensor.shape}, dtype: {train_tensor.dtype}")
    val_tensor = torch.from_numpy(val_activations).to(autoencoder.device)
    print(f"Validation tensor shape: {val_tensor.shape}, dtype: {val_tensor.dtype}")
    autoencoder.train(train_tensor, val_data=val_tensor, num_epochs=300, learning_rate=1e-3, patience=20)
    
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
    plt.hist(reconstruction_errors, bins=50)
    plt.title(f"Reconstruction Error Distribution on Test Set - Mean: {reconstruction_errors_mean:.6f}, Std: {reconstruction_errors_std:.6f}")
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Frequency")
    plt.grid(True)
    hist_path = Path(__file__).parent.parent / "results" / "autoencoder" / "reconstruction_error_histogram.png"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(hist_path)
    print(f"Reconstruction error histogram saved to {hist_path}")

if __name__ == "__main__":
    main()