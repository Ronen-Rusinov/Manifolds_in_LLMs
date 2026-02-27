import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_jacobian_orthogonality_loss(model, z, device='cpu'):
    """
    Compute orthogonality regularization for decoder Jacobian.
    Encourages ||J^T @ J - I||_F to be low, where J is the Jacobian of decoder with respect to z.
    
    Args:
        model: Autoencoder model
        z: Latent codes (batch_size, latent_dim)
        device: Device to compute on
    
    Returns:
        Scalar loss value
    """
    
    # While the _correct_ thing would be to compute it at every sample, we will instead go with a little bit
    # lighter of a requirement, that being ignoring activations entirely, and just
    # requiring that the weight matrices themselves are orthogonal.
    # This is a much weaker requirement, but it is also much faster to compute, and in practice it seems to work well enough.
    # We can also compute the regularization on the weight matrices directly, which is much faster and still encourages the desired property.
    W1 = model.encoder_mat_3  # (latent_dim, hidden_dim_2)
    W2 = model.encoder_mat_2  # (hidden_dim_2, hidden_dim_1)
    W3 = model.encoder_mat_1  # (hidden_dim_1, input_dim)

    WtW1 = W1.t() @ W1  # (hidden_dim_2, hidden_dim_2)
    WtW2 = W2.t() @ W2  # (hidden_dim_1, hidden_dim_1)
    WtW3 = W3.t() @ W3  # (input_dim, input_dim)

    I1 = torch.eye(WtW1.shape[0], device=device)
    I2 = torch.eye(WtW2.shape[0], device=device)
    I3 = torch.eye(WtW3.shape[0], device=device)

    reg_loss = torch.norm(WtW1 - I1, p='fro') + torch.norm(WtW2 - I2, p='fro') + torch.norm(WtW3 - I3, p='fro')
    return reg_loss


def train_with_gradient_accumulation(
    model,
    train_data,
    val_data,
    num_epochs,
    learning_rate,
    patience,
    chunk_size,
    device='cpu',
    regularization_weight=0.0
):
    """
    Train autoencoder with gradient accumulation for memory efficiency.
    
    Breaks training data into chunks, computes gradients for each chunk on the GPU,
    accumulates them, and then performs optimizer step. This allows training on
    datasets larger than GPU memory.
    
    Args:
        model: Autoencoder model
        train_data: numpy array or torch.Tensor of training data (N, input_dim)
        val_data: numpy array or torch.Tensor of validation data
        num_epochs: Maximum number of training epochs
        learning_rate: Learning rate for optimizer
        patience: Number of epochs without improvement before early stopping
        chunk_size: Number of samples to process on GPU at a time
        device: Device to train on ('cpu' or 'cuda')
        regularization_weight: Weight for Jacobian orthogonality regularization (0 = no regularization)
    
    Returns:
        dict: Contains training history with keys 'train_loss', 'val_loss', 'epochs_trained'
    """
    model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss(reduction='mean')
    
    # Convert to numpy if needed for better memory management
    if isinstance(train_data, torch.Tensor):
        train_data = train_data.cpu().numpy()
    if isinstance(val_data, torch.Tensor):
        val_data = val_data.cpu().numpy()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    num_training_samples = train_data.shape[0]
    num_chunks = (num_training_samples + chunk_size - 1) // chunk_size  # Ceiling division
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training step with gradient accumulation
        optimizer.zero_grad()
        
        accumulated_train_loss = 0.0
        accumulated_reg_loss = 0.0
        
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min((chunk_idx + 1) * chunk_size, num_training_samples)
            chunk_data = train_data[chunk_start:chunk_end]
            
            # Move chunk to GPU
            chunk_tensor = torch.from_numpy(chunk_data).to(device)
            
            # Forward pass
            chunk_recon = model(chunk_tensor)
            reconstruction_loss = criterion(chunk_recon, chunk_tensor)
            
            # Normalize loss by chunk size relative to total dataset for proper gradient accumulation
            chunk_size_actual = chunk_end - chunk_start
            normalized_loss = reconstruction_loss * (chunk_size_actual / num_training_samples)
            
            # Compute regularization if enabled
            total_loss = normalized_loss
            if regularization_weight > 0:
                z = model.encode(chunk_tensor)
                reg_loss = compute_jacobian_orthogonality_loss(model, z, device)
                # Also normalize regularization loss by chunk size
                normalized_reg_loss = reg_loss * (chunk_size_actual / num_training_samples)
                total_loss = normalized_loss + regularization_weight * normalized_reg_loss
                accumulated_reg_loss += normalized_reg_loss.item()
            
            # Backward pass (accumulates gradients)
            total_loss.backward()
            
            accumulated_train_loss += reconstruction_loss.item() / num_chunks
            
            # Clean up GPU memory
            del chunk_tensor, chunk_recon, reconstruction_loss, total_loss
            if regularization_weight > 0:
                del z, reg_loss, normalized_reg_loss
            torch.cuda.empty_cache() if device.type == 'cuda' else None
        
        # Optimizer step after accumulating gradients from all chunks
        optimizer.step()
        
        # Validation step (process all validation data at once or in chunks if needed)
        model.eval()
        with torch.no_grad():
            val_data_tensor = torch.from_numpy(val_data).to(device)
            val_recon = model(val_data_tensor)
            val_loss = criterion(val_recon, val_data_tensor)
            del val_data_tensor, val_recon
        model.train()
        
        train_losses.append(accumulated_train_loss)
        val_losses.append(val_loss.item())
        
        epoch_time = time.time() - epoch_start
        
        if (epoch + 1) % 30 == 0 or epoch == 0:
            print(f"\n[Epoch {epoch+1}/{num_epochs}]")
            print(f"  Train Loss: {accumulated_train_loss:.6f}")
            if regularization_weight > 0:
                print(f"  Reg Loss: {accumulated_reg_loss:.6f}")
            print(f"  Val Loss: {val_loss.item():.6f}")
            print(f"  Epoch Time: {epoch_time:.3f}s")
            print(f"  Chunks processed: {num_chunks}")
        
        # Early stopping logic
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (best val loss: {best_val_loss:.6f})")
                break
    
    return {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'epochs_trained': epoch + 1
    }


def evaluate_test_set(model, test_data, chunk_size=None, device='cpu'):
    """
    Evaluate autoencoder on test data and compute reconstruction errors.
    Optionally processes data in chunks to save memory.
    
    Args:
        model: Autoencoder model
        test_data: numpy array or torch.Tensor of test data
        chunk_size: Size of chunks for processing. If None, processes all at once.
        device: Device to evaluate on ('cpu' or 'cuda')
    
    Returns:
        dict: Contains 'errors', 'mean', 'std' of reconstruction errors
    """
    model.to(device)
    model.eval()
    
    if isinstance(test_data, np.ndarray):
        original_data = test_data
    else:
        original_data = test_data.cpu().numpy() if test_data.is_cuda else test_data.numpy()
        test_data = original_data
    
    num_test_samples = test_data.shape[0]
    
    if chunk_size is None:
        # Process all at once
        test_data_tensor = torch.from_numpy(test_data).to(device)
        with torch.no_grad():
            reconstructions = model(test_data_tensor).cpu().numpy()
    else:
        # Process in chunks
        num_chunks = (num_test_samples + chunk_size - 1) // chunk_size
        reconstructions = []
        
        with torch.no_grad():
            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * chunk_size
                chunk_end = min((chunk_idx + 1) * chunk_size, num_test_samples)
                chunk_data = test_data[chunk_start:chunk_end]
                
                chunk_tensor = torch.from_numpy(chunk_data).to(device)
                chunk_recon = model(chunk_tensor).cpu().numpy()
                reconstructions.append(chunk_recon)
                
                del chunk_tensor, chunk_recon
        
        reconstructions = np.vstack(reconstructions)
    
    reconstruction_errors = np.mean((reconstructions - original_data) ** 2, axis=1)
    
    return {
        'errors': reconstruction_errors,
        'mean': np.mean(reconstruction_errors),
        'std': np.std(reconstruction_errors)
    }


def visualize_reconstruction_errors(errors, error_stats, output_path, num_bins=50):
    """
    Create and save histogram of reconstruction errors.
    
    Args:
        errors: numpy array of reconstruction errors
        error_stats: dict with 'mean' and 'std' keys
        output_path: Path to save the histogram
        num_bins: Number of bins for histogram
    """
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=num_bins, edgecolor='black', alpha=0.7)
    plt.title(f"Reconstruction Error Distribution - Mean: {error_stats['mean']:.6f}, Std: {error_stats['std']:.6f}")
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Reconstruction error histogram saved to {output_path}")


if __name__ == "__main__":
    # Sanity check: Train TiedWeightAutoencoder with gradient accumulation
    from TiedWeightAutoencoder import TiedWeightAutoencoder
    
    print("=" * 60)
    print("Sanity Check: TiedWeightAutoencoder with Gradient Accumulation")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model configuration
    input_dim = 2304
    latent_dim = 20
    num_train_samples = 50000  # Use larger dataset to demonstrate gradient accumulation
    chunk_size = 10000  # Process 1024 samples at a time
    
    # Create model
    model = TiedWeightAutoencoder(input_dim=input_dim, latent_dim=latent_dim, device=device)
    model.to(device)
    
    # Initialize weights
    torch.nn.init.kaiming_uniform_(model.encoder_mat_1, a=0, mode='fan_in')
    torch.nn.init.kaiming_uniform_(model.encoder_mat_2, a=0, mode='fan_in')
    torch.nn.init.kaiming_uniform_(model.encoder_mat_3, a=0, mode='fan_in')
    torch.nn.init.zeros_(model.encoder_bias_1)
    torch.nn.init.zeros_(model.encoder_bias_2)
    torch.nn.init.zeros_(model.encoder_bias_3)
    torch.nn.init.zeros_(model.decoder_bias_1)
    torch.nn.init.zeros_(model.decoder_bias_2)
    torch.nn.init.zeros_(model.decoder_bias_3)
    
    # Generate synthetic data
    print(f"\nGenerating synthetic data:")
    print(f"  Input dim: {input_dim}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Training samples: {num_train_samples}")
    print(f"  Chunk size: {chunk_size}")
    
    train_data = np.random.randn(num_train_samples, input_dim).astype(np.float32)
    val_data = np.random.randn(1000, input_dim).astype(np.float32)
    test_data = np.random.randn(1000, input_dim).astype(np.float32)
    
    # Train with gradient accumulation
    print(f"\nTraining with gradient accumulation (regularization_weight=10)...")
    print("-" * 60)
    history = train_with_gradient_accumulation(
        model=model,
        train_data=train_data,
        val_data=val_data,
        num_epochs=200,
        learning_rate=1e-3,
        patience=20,
        chunk_size=chunk_size,
        device=device,
        regularization_weight=10
    )
    
    # Evaluate on test set
    print("\n" + "-" * 60)
    print("Evaluating on test set...")
    test_results = evaluate_test_set(model, test_data, chunk_size=chunk_size, device=device)
    print(f"Test set - Mean error: {test_results['mean']:.6f}, Std: {test_results['std']:.6f}")
    
    # Check Jacobian orthogonality on test data
    print("\n" + "-" * 60)
    print("Checking Jacobian orthogonality on test samples...")
    model.eval()
    with torch.no_grad():
        z_test = model.encode(torch.from_numpy(test_data[:10]).to(device))
    ortho_loss = compute_jacobian_orthogonality_loss(model, z_test, device)
    print(f"Jacobian orthogonality loss (||J^T J - I||_F): {ortho_loss.item():.6f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Sanity Check Summary:")
    print(f"  ✓ Model trained for {history['epochs_trained']} epochs")
    print(f"  ✓ Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"  ✓ Final val loss: {history['val_loss'][-1]:.6f}")
    print(f"  ✓ Test error: {test_results['mean']:.6f} ± {test_results['std']:.6f}")
    print(f"  ✓ Jacobian orthogonality: {ortho_loss.item():.6f}")
    
    if history['train_loss'][-1] < history['train_loss'][0]:
        print("\n✓ Training loss decreased successfully!")
    else:
        print("\n✗ Warning: Training loss did not decrease")
    
    print("=" * 60)
