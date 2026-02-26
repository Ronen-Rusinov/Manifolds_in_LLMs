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

def train_with_gradient_accumulation(model, train_data, val_data, num_epochs=300, learning_rate=1e-3, 
                                     patience=20, num_partitions=6):
    """
    Train model with gradient accumulation to handle large datasets that don't fit in GPU memory.
    
    Args:
        model: The autoencoder model
        train_data: Training data as numpy array (on CPU)
        val_data: Validation data as numpy array (on CPU)
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        patience: Early stopping patience
        num_partitions: Number of partitions to split data into for gradient accumulation
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    
    # Early stopping variables
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_state = None
    
    model.to(model.device)
    
    for epoch in range(num_epochs):
        # Shuffle training data at the beginning of each epoch
        indices = np.random.permutation(len(train_data))
        shuffled_train_data = train_data[indices]
        
        # Split into partitions
        partition_size = len(shuffled_train_data) // num_partitions
        
        # Training with gradient accumulation
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        
        for partition_idx in range(num_partitions):
            # Get partition indices
            start_idx = partition_idx * partition_size
            if partition_idx == num_partitions - 1:
                # Last partition gets remaining samples
                end_idx = len(shuffled_train_data)
            else:
                end_idx = start_idx + partition_size
            
            # Load partition to GPU
            partition_data = torch.from_numpy(shuffled_train_data[start_idx:end_idx]).to(model.device)
            
            # Forward pass
            output = model(partition_data)
            loss = criterion(output, partition_data)
            
            # Normalize loss by number of partitions (for proper gradient scaling)
            loss = loss / num_partitions
            
            # Backward pass (accumulate gradients)
            loss.backward()
            
            epoch_loss += loss.item() * num_partitions
            
            # Free GPU memory
            del partition_data, output, loss
            torch.cuda.empty_cache()
        
        # Update weights after accumulating gradients from all partitions
        optimizer.step()
        
        # Validation with early stopping (load validation data only when needed)
        if val_data is not None:
            model.eval()
            with torch.no_grad():
                # Process validation data in partitions to avoid memory issues
                val_losses = []
                val_partition_size = len(val_data) // num_partitions
                
                for partition_idx in range(num_partitions):
                    start_idx = partition_idx * val_partition_size
                    if partition_idx == num_partitions - 1:
                        end_idx = len(val_data)
                    else:
                        end_idx = start_idx + val_partition_size
                    
                    val_partition = torch.from_numpy(val_data[start_idx:end_idx]).to(model.device)
                    val_output = model(val_partition)
                    val_loss = criterion(val_output, val_partition)
                    val_losses.append(val_loss.item())
                    
                    del val_partition, val_output
                    torch.cuda.empty_cache()
                
                val_loss = np.mean(val_losses)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                epochs_without_improvement += 1
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Best Val: {best_val_loss:.4f}', flush=True)
            
            # Check early stopping
            if epochs_without_improvement >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs. '
                      f'Best val loss: {best_val_loss:.4f}', flush=True)
                # Restore best model
                model.load_state_dict(best_state)
                break
        else:
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}', flush=True)
    
    return model


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
    
    # Initialize the autoencoder with float32
    input_dim = train_activations.shape[1]
    latent_dim = 12 
    print(f"Initializing StandardAutoencoder with input_dim={input_dim} and latent_dim={latent_dim}")
    autoencoder = StandardAutoencoder(
        input_dim=input_dim, 
        latent_dim=latent_dim, 
        device='cuda' if torch.cuda.is_available() else 'cpu', 
        dtype=torch.float32  # Use float32 to avoid overflow
    )

    # Train with gradient accumulation
    print("Training autoencoder with gradient accumulation and early stopping...")
    autoencoder = train_with_gradient_accumulation(
        model=autoencoder,
        train_data=train_activations,  # Keep on CPU
        val_data=val_activations,      # Keep on CPU
        num_epochs=300,
        learning_rate=1e-3,
        patience=20,
        num_partitions=6  # Split data into 6 partitions
    )

    print("Training complete!")
    
    # Evaluate on test data (also in partitions to avoid memory issues)
    print("Evaluating on test set...")
    num_partitions = 6
    partition_size = len(test_activations) // num_partitions
    all_reconstruction_errors = []
    
    autoencoder.eval()
    with torch.no_grad():
        for partition_idx in range(num_partitions):
            start_idx = partition_idx * partition_size
            if partition_idx == num_partitions - 1:
                end_idx = len(test_activations)
            else:
                end_idx = start_idx + partition_size
            
            test_partition = torch.from_numpy(test_activations[start_idx:end_idx]).to(autoencoder.device)
            reconstructions = autoencoder(test_partition).cpu().numpy()
            
            # Calculate reconstruction error for this partition
            partition_errors = np.mean((reconstructions - test_activations[start_idx:end_idx]) ** 2, axis=1)
            all_reconstruction_errors.extend(partition_errors)
            
            del test_partition, reconstructions
            torch.cuda.empty_cache()
    
    reconstruction_errors = np.array(all_reconstruction_errors)
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
    hist_path = Path(__file__).parent.parent / "results" / "autoencoder" / "reconstruction_error_histogram_gradient_accum.png"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(hist_path)
    print(f"Reconstruction error histogram saved to {hist_path}")

    # Save the model
    model_path = Path(__file__).parent.parent / "results" / "autoencoder" / "autoencoder_gradient_accum.pt"
    torch.save(autoencoder.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
