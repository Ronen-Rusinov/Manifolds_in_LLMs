"""
Example script for training and evaluating sparse autoencoders.

This script demonstrates:
1. Creating SparseAutoencoder and TiedWeightSparseAutoencoder instances
2. Training with different sparsity constraints (L1, KL, Hoyer)
3. Evaluating model performance
4. Visualizing results
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.SparseAutoencoder import SparseAutoencoder, TiedWeightSparseAutoencoder
from src.train_sparse_autoencoder import (
    train_sparse_autoencoder_with_early_stopping,
    evaluate_sparse_autoencoder,
    plot_sparse_training_history,
    plot_sparse_autoencoder_analysis,
)


def example_basic_sparse_autoencoder():
    """
    Basic example: Train a sparse autoencoder with KL divergence sparsity.
    """
    print("=" * 80)
    print("EXAMPLE 1: Basic Sparse Autoencoder Training")
    print("=" * 80)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    input_dim = 200
    latent_dim = 50
    num_epochs = 500
    learning_rate = 1e-3
    patience = 50
    sparsity_weight = 0.1
    target_sparsity = 0.05
    sparsity_type = 'kl'  # Options: 'kl', 'l1', 'hoyer'
    
    print(f"\nModel Configuration:")
    print(f"  Input Dimension: {input_dim}")
    print(f"  Latent Dimension: {latent_dim}")
    print(f"  Sparsity Type: {sparsity_type}")
    print(f"  Sparsity Weight: {sparsity_weight}")
    print(f"  Target Sparsity (for KL): {target_sparsity}")
    
    # Generate synthetic data
    np.random.seed(42)
    torch.manual_seed(42)
    
    num_train = 3200
    num_val = 800
    num_test = 1000
    
    train_data = torch.randn(num_train, input_dim, device='cpu')
    val_data = torch.randn(num_val, input_dim, device='cpu')
    test_data = torch.randn(num_test, input_dim, device='cpu')
    
    print(f"\nData Configuration:")
    print(f"  Training samples: {num_train}")
    print(f"  Validation samples: {num_val}")
    print(f"  Test samples: {num_test}")
    
    # Create model
    model = SparseAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        device=device,
        dtype=torch.float32
    )
    print(f"\nModel created: {model.__class__.__name__}")
    
    # Train
    print("\nTraining...")
    history = train_sparse_autoencoder_with_early_stopping(
        model=model,
        train_data=train_data,
        val_data=val_data,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        patience=patience,
        sparsity_weight=sparsity_weight,
        target_sparsity=target_sparsity,
        sparsity_type=sparsity_type,
        device=device
    )
    
    print(f"\nTraining completed in {history['epochs_trained']} epochs")
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final val loss: {history['val_loss'][-1]:.6f}")
    
    # Evaluate
    print("\nEvaluating on test set...")
    eval_results = evaluate_sparse_autoencoder(model, test_data, device=device)
    
    print(f"\nTest Set Results:")
    print(f"  Mean Reconstruction Error: {eval_results['mean']:.6f}")
    print(f"  Std Reconstruction Error: {eval_results['std']:.6f}")
    print(f"  Sparsity Level (% near-zero activations): {eval_results['sparsity_level']*100:.2f}%")
    print(f"  Mean Activation per Neuron (min/max): {eval_results['mean_activation_per_neuron'].min():.4f} / {eval_results['mean_activation_per_neuron'].max():.4f}")
    
    # Visualization
    print("\nGenerating plots...")
    plot_sparse_training_history(history, save_path='sparse_training_history.png')
    plot_sparse_autoencoder_analysis(eval_results, model_name='Sparse Autoencoder', 
                                      save_path='sparse_autoencoder_analysis.png')
    
    return model, history, eval_results


def example_compare_sparsity_types():
    """
    Compare different sparsity regularization types (L1, KL, Hoyer).
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Compare Different Sparsity Types")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate data
    np.random.seed(42)
    torch.manual_seed(42)
    
    input_dim = 200
    latent_dim = 50
    num_train = 3200
    num_val = 800
    num_test = 1000
    
    train_data = torch.randn(num_train, input_dim, device='cpu')
    val_data = torch.randn(num_val, input_dim, device='cpu')
    test_data = torch.randn(num_test, input_dim, device='cpu')
    
    # Train models with different sparsity types
    sparsity_types = ['l1', 'kl', 'hoyer']
    results = {}
    
    for sparsity_type in sparsity_types:
        print(f"\nTraining with {sparsity_type.upper()} sparsity...")
        
        model = SparseAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            device=device
        )
        
        history = train_sparse_autoencoder_with_early_stopping(
            model=model,
            train_data=train_data,
            val_data=val_data,
            num_epochs=300,
            learning_rate=1e-3,
            patience=40,
            sparsity_weight=0.1,
            target_sparsity=0.05,
            sparsity_type=sparsity_type,
            device=device
        )
        
        eval_results = evaluate_sparse_autoencoder(model, test_data, device=device)
        
        results[sparsity_type] = {
            'history': history,
            'eval': eval_results,
            'model': model
        }
        
        print(f"  Final val loss: {history['val_loss'][-1]:.6f}")
        print(f"  Test MSE: {eval_results['mean']:.6f}")
        print(f"  Sparsity level: {eval_results['sparsity_level']*100:.2f}%")
    
    # Compare results
    print("\n" + "-" * 80)
    print("COMPARISON SUMMARY")
    print("-" * 80)
    print(f"{'Sparsity Type':<15} {'Val Loss':<15} {'Test MSE':<15} {'Sparsity %':<15}")
    print("-" * 80)
    for sparsity_type in sparsity_types:
        val_loss = results[sparsity_type]['history']['val_loss'][-1]
        test_mse = results[sparsity_type]['eval']['mean']
        sparsity = results[sparsity_type]['eval']['sparsity_level'] * 100
        print(f"{sparsity_type:<15} {val_loss:<15.6f} {test_mse:<15.6f} {sparsity:<15.2f}")
    
    return results


def example_tied_weight_sparse_autoencoder():
    """
    Example: Train a tied-weight sparse autoencoder.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Tied-Weight Sparse Autoencoder")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate data
    np.random.seed(42)
    torch.manual_seed(42)
    
    input_dim = 200
    latent_dim = 50
    num_train = 3200
    num_val = 800
    num_test = 1000
    
    train_data = torch.randn(num_train, input_dim, device='cpu')
    val_data = torch.randn(num_val, input_dim, device='cpu')
    test_data = torch.randn(num_test, input_dim, device='cpu')
    
    # Create tied-weight model
    model = TiedWeightSparseAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        device=device
    )
    
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
    
    print("Model created: TiedWeightSparseAutoencoder")
    print(f"  Encoder parameters: {sum(p.numel() for p in [model.encoder_mat_1, model.encoder_mat_2, model.encoder_mat_3, model.encoder_bias_1, model.encoder_bias_2, model.encoder_bias_3])}")
    print(f"  Decoder parameters: {sum(p.numel() for p in [model.decoder_bias_1, model.decoder_bias_2, model.decoder_bias_3])}")
    
    # Train
    print("\nTraining...")
    history = train_sparse_autoencoder_with_early_stopping(
        model=model,
        train_data=train_data,
        val_data=val_data,
        num_epochs=300,
        learning_rate=1e-3,
        patience=40,
        sparsity_weight=0.1,
        target_sparsity=0.05,
        sparsity_type='kl',
        device=device
    )
    
    print(f"Training completed in {history['epochs_trained']} epochs")
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final val loss: {history['val_loss'][-1]:.6f}")
    
    # Evaluate
    eval_results = evaluate_sparse_autoencoder(model, test_data, device=device)
    print(f"\nTest Results:")
    print(f"  Mean Reconstruction Error: {eval_results['mean']:.6f}")
    print(f"  Sparsity Level: {eval_results['sparsity_level']*100:.2f}%")
    
    return model, history, eval_results


def example_varying_sparsity_weight():
    """
    Example: Train models with different sparsity weights to see the effect.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Effect of Sparsity Weight")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate data
    np.random.seed(42)
    torch.manual_seed(42)
    
    input_dim = 200
    latent_dim = 50
    num_train = 3200
    num_val = 800
    num_test = 1000
    
    train_data = torch.randn(num_train, input_dim, device='cpu')
    val_data = torch.randn(num_val, input_dim, device='cpu')
    test_data = torch.randn(num_test, input_dim, device='cpu')
    
    # Test different sparsity weights
    sparsity_weights = [0.0, 0.05, 0.1, 0.5, 1.0]
    results = {}
    
    for weight in sparsity_weights:
        print(f"\nTraining with sparsity_weight={weight}...")
        
        model = SparseAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            device=device
        )
        
        history = train_sparse_autoencoder_with_early_stopping(
            model=model,
            train_data=train_data,
            val_data=val_data,
            num_epochs=200,
            learning_rate=1e-3,
            patience=30,
            sparsity_weight=weight,
            target_sparsity=0.05,
            sparsity_type='kl',
            device=device
        )
        
        eval_results = evaluate_sparse_autoencoder(model, test_data, device=device)
        results[weight] = {
            'history': history,
            'eval': eval_results
        }
        
        print(f"  Final val loss: {history['val_loss'][-1]:.6f}")
        print(f"  Test MSE: {eval_results['mean']:.6f}")
        print(f"  Sparsity level: {eval_results['sparsity_level']*100:.2f}%")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Reconstruction error vs sparsity weight
    recon_errors = [results[w]['eval']['mean'] for w in sparsity_weights]
    sparsity_levels = [results[w]['eval']['sparsity_level'] * 100 for w in sparsity_weights]
    
    axes[0].plot(sparsity_weights, recon_errors, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Sparsity Weight')
    axes[0].set_ylabel('Test MSE')
    axes[0].set_title('Reconstruction Error vs Sparsity Weight')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(sparsity_weights, sparsity_levels, 'o-', linewidth=2, markersize=8, color='orange')
    axes[1].set_xlabel('Sparsity Weight')
    axes[1].set_ylabel('Sparsity Level (%)')
    axes[1].set_title('Sparsity Level vs Sparsity Weight')
    axes[1].grid(True, alpha=0.3)
    
    # Validation loss curves
    for weight in sparsity_weights:
        axes[2].plot(results[weight]['history']['val_loss'], label=f'w={weight}', alpha=0.7)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Validation Loss')
    axes[2].set_title('Validation Loss Curves')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sparsity_weight_comparison.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to sparsity_weight_comparison.png")
    
    return results


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("SPARSE AUTOENCODER EXAMPLES")
    print("=" * 80)
    
    # Run examples
    try:
        # Example 1: Basic sparse autoencoder
        model1, history1, eval1 = example_basic_sparse_autoencoder()
        
        # Example 2: Compare sparsity types
        comparison_results = example_compare_sparsity_types()
        
        # Example 3: Tied-weight sparse autoencoder
        model3, history3, eval3 = example_tied_weight_sparse_autoencoder()
        
        # Example 4: Effect of sparsity weight
        weight_results = example_varying_sparsity_weight()
        
        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
