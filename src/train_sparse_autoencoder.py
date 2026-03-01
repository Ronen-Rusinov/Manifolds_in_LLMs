import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train_sparse_autoencoder_with_early_stopping(
    model,
    train_data,
    val_data,
    num_epochs,
    learning_rate,
    patience,
    sparsity_weight=2,
    target_sparsity=0.05,
    sparsity_type='l1',
    device='cpu'
):
    """
    Train sparse autoencoder with validation and early stopping.
    
    The total loss is: L = MSE(reconstruction) + sparsity_weight * sparsity_loss
    
    Args:
        model: SparseAutoencoder model
        train_data: torch.Tensor of training data
        val_data: torch.Tensor of validation data
        num_epochs: Maximum number of training epochs
        learning_rate: Learning rate for optimizer
        patience: Number of epochs without improvement before early stopping
        sparsity_weight: Weight for sparsity loss in total loss (default: 2)
        target_sparsity: Target activation probability for KL divergence (default: 0.05)
        sparsity_type: Type of sparsity loss - 'l1', 'kl', or 'hoyer' (default: 'l1')
        device: Device to train on ('cpu' or 'cuda')
    
    Returns:
        dict: Contains training history with keys:
            - 'train_loss': List of total training losses
            - 'train_recon_loss': List of reconstruction losses
            - 'train_sparsity_loss': List of sparsity losses
            - 'val_loss': List of total validation losses
            - 'val_recon_loss': List of validation reconstruction losses
            - 'val_sparsity_loss': List of validation sparsity losses
            - 'epochs_trained': Number of epochs completed
    
    Raises:
        ValueError: If sparsity_type is not recognized
    """
    if sparsity_type not in ['l1', 'kl', 'hoyer']:
        raise ValueError(f"sparsity_type must be 'l1', 'kl', or 'hoyer', got {sparsity_type}")
    
    model.to(device)
    model.train()
    
    # Select sparsity loss function
    if sparsity_type == 'l1':
        sparsity_loss_fn = model.l1_sparsity_loss
    elif sparsity_type == 'kl':
        sparsity_loss_fn = lambda z: model.kl_divergence_sparsity_loss(z, target_sparsity)
    else:  # hoyer
        sparsity_loss_fn = model.hoyer_sparsity_loss
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    
    train_losses = []
    train_recon_losses = []
    train_sparsity_losses = []
    val_losses = []
    val_recon_losses = []
    val_sparsity_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training step
        optimizer.zero_grad()
        train_recon = model(train_data)
        recon_loss = criterion(train_recon, train_data)
        
        # Get latent codes for sparsity loss
        with torch.no_grad():
            z_train = model.encode(train_data)
        sparse_loss = sparsity_loss_fn(z_train)
        
        total_loss = recon_loss + sparsity_weight * sparse_loss
        total_loss.backward()
        optimizer.step()
        
        train_losses.append(total_loss.item())
        train_recon_losses.append(recon_loss.item())
        train_sparsity_losses.append(sparse_loss.item())
        
        # Validation step
        model.eval()
        with torch.no_grad():
            val_recon = model(val_data)
            val_recon_loss = criterion(val_recon, val_data)
            
            z_val = model.encode(val_data)
            val_sparse_loss = sparsity_loss_fn(z_val)
            
            val_total_loss = val_recon_loss + sparsity_weight * val_sparse_loss
        
        model.train()
        
        val_losses.append(val_total_loss.item())
        val_recon_losses.append(val_recon_loss.item())
        val_sparsity_losses.append(val_sparse_loss.item())
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {total_loss.item():.6f} "
                f"(Recon: {recon_loss.item():.6f}, Sparsity: {sparse_loss.item():.6f}) | "
                f"Val Loss: {val_total_loss.item():.6f} "
                f"(Recon: {val_recon_loss.item():.6f}, Sparsity: {val_sparse_loss.item():.6f})"
            )
        
        # Early stopping logic
        if val_total_loss.item() < best_val_loss:
            best_val_loss = val_total_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping at epoch {epoch+1} (best val loss: {best_val_loss:.6f})"
                )
                break
    
    return {
        'train_loss': train_losses,
        'train_recon_loss': train_recon_losses,
        'train_sparsity_loss': train_sparsity_losses,
        'val_loss': val_losses,
        'val_recon_loss': val_recon_losses,
        'val_sparsity_loss': val_sparsity_losses,
        'epochs_trained': epoch + 1
    }


def evaluate_sparse_autoencoder(model, test_data, device='cpu'):
    """
    Evaluate sparse autoencoder on test data and compute reconstruction errors.
    
    Args:
        model: SparseAutoencoder model
        test_data: numpy array or torch.Tensor of test data
        device: Device to evaluate on ('cpu' or 'cuda')
    
    Returns:
        dict: Contains:
            - 'errors': Reconstruction errors per sample
            - 'mean': Mean reconstruction error
            - 'std': Std of reconstruction errors
            - 'latent_codes': Latent representations of test data
            - 'mean_activation_per_neuron': Mean activation per latent neuron
            - 'sparsity_level': Percentage of near-zero activations
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
        reconstructed = model(test_data_tensor)
        latent_codes = model.encode(test_data_tensor)
        
        criterion = torch.nn.MSELoss(reduction='none')
        errors = criterion(reconstructed, test_data_tensor).mean(dim=1)
        
        # Compute sparsity metrics
        mean_activation = latent_codes.mean(dim=0).cpu().numpy()
        
        # Count near-zero activations (e.g., < 0.01)
        near_zero = (torch.abs(latent_codes) < 0.01).float().mean().item()
    
    errors_np = errors.cpu().numpy()
    
    return {
        'errors': errors_np,
        'mean': np.mean(errors_np),
        'std': np.std(errors_np),
        'latent_codes': latent_codes.cpu().numpy(),
        'mean_activation_per_neuron': mean_activation,
        'sparsity_level': near_zero,
    }


def plot_sparse_training_history(history, save_path=None, figsize=(15, 10)):
    """
    Plot training history for sparse autoencoder.
    
    Args:
        history: dict returned by train_sparse_autoencoder_with_early_stopping
        save_path: Path to save figure (optional)
        figsize: Figure size tuple
    
    Returns:
        matplotlib figure object
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Total loss
    axes[0, 0].plot(history['train_loss'], label='Train', alpha=0.7)
    axes[0, 0].plot(history['val_loss'], label='Val', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss (Reconstruction + Sparsity)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Reconstruction loss
    axes[0, 1].plot(history['train_recon_loss'], label='Train', alpha=0.7)
    axes[0, 1].plot(history['val_recon_loss'], label='Val', alpha=0.7)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE Loss')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Sparsity loss
    axes[0, 2].plot(history['train_sparsity_loss'], label='Train', alpha=0.7)
    axes[0, 2].plot(history['val_sparsity_loss'], label='Val', alpha=0.7)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Sparsity Loss')
    axes[0, 2].set_title('Sparsity Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Train total loss (log scale)
    axes[1, 0].semilogy(history['train_loss'], label='Train', alpha=0.7)
    axes[1, 0].semilogy(history['val_loss'], label='Val', alpha=0.7)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Total Loss (log scale)')
    axes[1, 0].set_title('Total Loss (Log Scale)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Train reconstruction loss (log scale)
    axes[1, 1].semilogy(history['train_recon_loss'], label='Train', alpha=0.7)
    axes[1, 1].semilogy(history['val_recon_loss'], label='Val', alpha=0.7)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MSE Loss (log scale)')
    axes[1, 1].set_title('Reconstruction Loss (Log Scale)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Train sparsity loss (log scale)
    axes[1, 2].semilogy(history['train_sparsity_loss'], label='Train', alpha=0.7)
    axes[1, 2].semilogy(history['val_sparsity_loss'], label='Val', alpha=0.7)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Sparsity Loss (log scale)')
    axes[1, 2].set_title('Sparsity Loss (Log Scale)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history plot to {save_path}")
    
    return fig


def plot_sparse_autoencoder_analysis(eval_results, model_name='Sparse Autoencoder', save_path=None):
    """
    Plot analysis of sparse autoencoder performance.
    
    Args:
        eval_results: dict returned by evaluate_sparse_autoencoder
        model_name: Name of model for plot title
        save_path: Path to save figure (optional)
    
    Returns:
        matplotlib figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Reconstruction error distribution
    axes[0, 0].hist(eval_results['errors'], bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(eval_results['mean'], color='r', linestyle='--', 
                       label=f"Mean: {eval_results['mean']:.4f}")
    axes[0, 0].set_xlabel('Reconstruction Error (MSE)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Reconstruction Error Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Mean activation per neuron
    mean_act = eval_results['mean_activation_per_neuron']
    axes[0, 1].bar(range(len(mean_act)), mean_act, alpha=0.7)
    axes[0, 1].set_xlabel('Latent Neuron')
    axes[0, 1].set_ylabel('Mean Activation')
    axes[0, 1].set_title('Mean Activation per Latent Neuron')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Latent code statistics
    latent_codes = eval_results['latent_codes']
    axes[1, 0].imshow(latent_codes[:100].T, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[1, 0].set_xlabel('Sample')
    axes[1, 0].set_ylabel('Latent Neuron')
    axes[1, 0].set_title('Latent Codes (first 100 samples)')
    plt.colorbar(axes[1, 0].images[0], ax=axes[1, 0], label='Activation')
    
    # Sparsity information
    sparsity_pct = eval_results['sparsity_level'] * 100
    axes[1, 1].text(
        0.5, 0.5,
        f"Sparsity Level: {sparsity_pct:.2f}%\n\n"
        f"Mean Reconstruction Error: {eval_results['mean']:.6f}\n"
        f"Std Reconstruction Error: {eval_results['std']:.6f}\n\n"
        f"Latent Dim: {latent_codes.shape[1]}\n"
        f"Num Samples: {latent_codes.shape[0]}",
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=12,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        transform=axes[1, 1].transAxes
    )
    axes[1, 1].axis('off')
    
    fig.suptitle(f'{model_name} Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved analysis plot to {save_path}")
    
    return fig
