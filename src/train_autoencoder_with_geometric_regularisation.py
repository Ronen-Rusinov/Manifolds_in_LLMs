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
    latent_dim = z.shape[1]
    batch_size = z.shape[0]
    
    t_start = time.time()
    
    # Prepare z for jacobian computation (requires grad)
    z_for_jacobian = z.detach().requires_grad_(True)
    
    # Compute Jacobian for each sample in the batch
    t_loop_start = time.time()
    jacobians = []
    for i in range(batch_size):
        # Get single latent code
        z_i = z_for_jacobian[i:i+1]  # Keep batch dimension
        
        # Compute Jacobian for this sample: (output_dim, 1, latent_dim)
        jac_i = torch.autograd.functional.jacobian(
            lambda x: model.decode(x).squeeze(0),  # Remove batch dim from output
            z_i,
            create_graph=True
        )
        # Squeeze out the extra dimension: (output_dim, latent_dim)
        jac_i = jac_i.squeeze(1)
        jacobians.append(jac_i)
    
    # Stack to get (batch_size, output_dim, latent_dim)
    jacobians = torch.stack(jacobians, dim=0)
    t_loop_end = time.time()
    print(f"  [Timing] Jacobian loop: {t_loop_end - t_loop_start:.3f}s ({(t_loop_end - t_loop_start)/batch_size*1000:.1f}ms per sample)")
    
    # Compute J^T @ J for each sample: (batch_size, latent_dim, latent_dim)
    t_matmul_start = time.time()
    jtj = torch.bmm(jacobians.transpose(1, 2), jacobians)
    
    # Create batch of identity matrices
    identity = torch.eye(latent_dim, device=device).unsqueeze(0).expand(batch_size, -1, -1)
    
    # Compute Frobenius norm of (J^T @ J - I) for each sample and average
    ortho_loss = torch.norm(jtj - identity, p='fro', dim=(1, 2)).mean()
    t_end = time.time()
    print(f"  [Timing] Matrix ops: {t_end - t_matmul_start:.3f}s")
    print(f"  [Timing] Total regularization: {t_end - t_start:.3f}s")
    
    return ortho_loss


def train_with_early_stopping(model, train_data, val_data, num_epochs, learning_rate, patience, device='cpu', regularization_weight=0.0):
    """
    Train autoencoder with validation, early stopping, and Jacobian orthogonality regularization.
    
    Args:
        model: Autoencoder model
        train_data: torch.Tensor of training data
        val_data: torch.Tensor of validation data
        num_epochs: Maximum number of training epochs
        learning_rate: Learning rate for optimizer
        patience: Number of epochs without improvement before early stopping
        device: Device to train on ('cpu' or 'cuda')
        regularization_weight: Weight for Jacobian orthogonality regularization (0 = no regularization)
    
    Returns:
        dict: Contains training history with keys 'train_loss', 'val_loss', 'epochs_trained'
    """
    model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training step
        optimizer.zero_grad()
        
        # Forward pass
        t0 = time.time()
        train_recon = model(train_data)
        reconstruction_loss = criterion(train_recon, train_data)
        t1 = time.time()
        
        # Compute regularization if enabled
        total_loss = reconstruction_loss
        if regularization_weight > 0:
            t_encode = time.time()
            z = model.encode(train_data)
            print(f"[Timing] Epoch {epoch+1} - Encoding: {time.time() - t_encode:.3f}s")
            
            t_reg = time.time()
            reg_loss = compute_jacobian_orthogonality_loss(model, z, device)
            print(f"[Timing] Epoch {epoch+1} - Regularization total: {time.time() - t_reg:.3f}s")
            
            total_loss = reconstruction_loss + regularization_weight * reg_loss
        
        t2 = time.time()
        total_loss.backward()
        t3 = time.time()
        optimizer.step()
        t4 = time.time()
        
        # Validation step
        t5 = time.time()
        model.eval()
        with torch.no_grad():
            val_recon = model(val_data)
            val_loss = criterion(val_recon, val_data)
        model.train()
        t6 = time.time()
        
        train_losses.append(total_loss.item())
        val_losses.append(val_loss.item())
        
        epoch_time = time.time() - epoch_start
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"\n[Timing] Epoch {epoch+1} breakdown:")
            print(f"  Forward pass: {t1-t0:.3f}s")
            print(f"  Backward pass: {t3-t2:.3f}s")
            print(f"  Optimizer step: {t4-t3:.3f}s")
            print(f"  Validation: {t6-t5:.3f}s")
            print(f"  Total epoch time: {epoch_time:.3f}s")
            
            if regularization_weight > 0:
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {reconstruction_loss.item():.6f}, Reg Loss: {reg_loss.item():.6f}, Val Loss: {val_loss.item():.6f}\n")
            else:
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {reconstruction_loss.item():.6f}, Val Loss: {val_loss.item():.6f}\n")
        
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


def evaluate_test_set(model, test_data, device='cpu'):
    """
    Evaluate autoencoder on test data and compute reconstruction errors.
    
    Args:
        model: Autoencoder model
        test_data: numpy array or torch.Tensor of test data
        device: Device to evaluate on ('cpu' or 'cuda')
    
    Returns:
        dict: Contains 'errors', 'mean', 'std' of reconstruction errors
    """
    model.to(device)
    model.eval()
    
    if isinstance(test_data, np.ndarray):
        test_data_tensor = torch.from_numpy(test_data).to(device)
        original_data = test_data
    else:
        test_data_tensor = test_data.to(device)
        original_data = test_data.cpu().numpy() if test_data.is_cuda else test_data.numpy()
    
    with torch.no_grad():
        reconstructions = model(test_data_tensor).cpu().numpy()
    
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
    # Sanity check: Train TiedWeightAutoencoder with geometric regularization
    from TiedWeightAutoencoder import TiedWeightAutoencoder
    
    print("=" * 60)
    print("Sanity Check: TiedWeightAutoencoder with Geometric Regularization")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model configuration
    input_dim = 100
    latent_dim = 20
    batch_size = 256
    
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
    print(f"\nGenerating synthetic data (input_dim={input_dim}, latent_dim={latent_dim})...")
    train_data = torch.randn(batch_size, input_dim, device=device)
    val_data = torch.randn(batch_size // 4, input_dim, device=device)
    test_data = torch.randn(batch_size // 4, input_dim, device=device)
    
    # Train with regularization
    print(f"\nTraining with geometric regularization (weight=0.1)...")
    print("-" * 60)
    history = train_with_early_stopping(
        model=model,
        train_data=train_data,
        val_data=val_data,
        num_epochs=10,
        learning_rate=1e-3,
        patience=20,
        device=device,
        regularization_weight=0.1
    )
    
    # Evaluate on test set
    print("\n" + "-" * 60)
    print("Evaluating on test set...")
    test_results = evaluate_test_set(model, test_data, device=device)
    print(f"Test set - Mean error: {test_results['mean']:.6f}, Std: {test_results['std']:.6f}")
    
    # Check Jacobian orthogonality on test data
    print("\n" + "-" * 60)
    print("Checking Jacobian orthogonality on test samples...")
    model.eval()
    with torch.no_grad():
        z_test = model.encode(test_data[:10])  # Check first 10 samples
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
