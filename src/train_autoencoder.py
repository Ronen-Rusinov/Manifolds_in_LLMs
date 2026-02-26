import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train_with_early_stopping(model, train_data, val_data, num_epochs, learning_rate, patience, device='cpu'):
    """
    Train autoencoder with validation and early stopping.
    
    Args:
        model: StandardAutoencoder model
        train_data: torch.Tensor of training data
        val_data: torch.Tensor of validation data
        num_epochs: Maximum number of training epochs
        learning_rate: Learning rate for optimizer
        patience: Number of epochs without improvement before early stopping
        device: Device to train on ('cpu' or 'cuda')
    
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
        # Training step
        optimizer.zero_grad()
        train_recon = model(train_data)
        train_loss = criterion(train_recon, train_data)
        train_loss.backward()
        optimizer.step()
        
        # Validation step
        model.eval()
        with torch.no_grad():
            val_recon = model(val_data)
            val_loss = criterion(val_recon, val_data)
        model.train()
        
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss.item():.6f}, Val Loss: {val_loss.item():.6f}")
        
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
    Evaluate model on test data and compute reconstruction errors.
    
    Args:
        model: StandardAutoencoder model
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