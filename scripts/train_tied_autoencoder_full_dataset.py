#!/usr/bin/env python
"""
Train a tied autoencoder on the full activation dataset with gradient accumulation.

This script:
1. Loads activations from the full dataset
2. Splits data into 70% train, 20% validation, 10% test
3. Trains a TiedWeightAutoencoder with geometric regularization
4. Uses gradient accumulation for memory efficiency
5. Saves the trained model and results
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config_manager import load_config, add_config_argument
from src import paths
from src.utils.common import load_activations, validate_data_consistency
from src.TiedWeightAutoencoder import TiedWeightAutoencoder
from src.train_tied_autoencoder_with_gradient_accumulation import (
    train_with_gradient_accumulation,
    evaluate_test_set,
    visualize_reconstruction_errors
)


def split_data(data, train_fraction=0.7, val_fraction=0.2, random_seed=42):
    """Split data into training, validation, and test sets.
    
    Args:
        data: numpy array of shape (n_samples, n_features)
        train_fraction: Fraction for training (default: 0.7)
        val_fraction: Fraction for validation (default: 0.2)
        random_seed: Random seed for reproducibility
    
    Returns:
        tuple: (train_data, val_data, test_data)
    """
    n_samples = data.shape[0]
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Generate random permutation
    indices = np.random.permutation(n_samples)
    
    # Calculate split points
    train_end = int(n_samples * train_fraction)
    val_end = train_end + int(n_samples * val_fraction)
    
    # Split data
    train_data = data[indices[:train_end]]
    val_data = data[indices[train_end:val_end]]
    test_data = data[indices[val_end:]]
    
    return train_data, val_data, test_data


def create_output_dir(base_dir="tied_autoencoder_full_dataset"):
    """Create output directory with timestamp.
    
    Args:
        base_dir: Base directory name
    
    Returns:
        Path to created output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = paths.get_results_dir() / base_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_training_results(
    model,
    history,
    test_results,
    config,
    output_dir,
    activations_shape=None
):
    """Save trained model, results, and metadata.
    
    Args:
        model: Trained TiedWeightAutoencoder
        history: Training history dict with 'train_loss' and 'val_loss'
        test_results: Test evaluation results dict
        config: Config object
        output_dir: Output directory path
        activations_shape: Shape of original activations data
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    model_path = output_dir / "model_weights.pt"
    torch.save(model.state_dict(), model_path)
    print(f"[{datetime.now()}] Model saved to {model_path}")
    
    # Save training history
    history_path = output_dir / "training_history.npz"
    np.savez(
        history_path,
        train_loss=np.array(history['train_loss']),
        val_loss=np.array(history['val_loss'])
    )
    print(f"[{datetime.now()}] Training history saved to {history_path}")
    
    # Save test results
    test_path = output_dir / "test_results.npz"
    np.savez(
        test_path,
        reconstruction_errors=test_results['errors'],
        mean_error=np.array([test_results['mean']]),
        std_error=np.array([test_results['std']])
    )
    print(f"[{datetime.now()}] Test results saved to {test_path}")
    
    # Save metadata and config
    metadata = {
        'input_dim': model.input_dim,
        'latent_dim': model.latent_dim,
        'epochs_trained': history['epochs_trained'],
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'test_mean_error': test_results['mean'],
        'test_std_error': test_results['std'],
        'activations_shape': str(activations_shape) if activations_shape else "unknown",
        'training_date': datetime.now().isoformat(),
        'layer_used': config.model.layer_for_activation,
        'regularization_weight': config.autoencoder.regularization_weight,
    }
    
    metadata_path = output_dir / "metadata.txt"
    with open(metadata_path, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    print(f"[{datetime.now()}] Metadata saved to {metadata_path}")
    
    return model_path


def plot_training_curves(history, output_dir):
    """Plot and save training/validation loss curves.
    
    Args:
        history: Training history dict with 'train_loss' and 'val_loss'
        output_dir: Directory to save plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = Path(output_dir) / "training_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[{datetime.now()}] Training curves saved to {plot_path}")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train TiedWeightAutoencoder on full activation dataset with gradient accumulation"
    )
    add_config_argument(parser)
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Override layer index for activation extraction"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Chunk size for gradient accumulation (default: 10000)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Override patience for early stopping"
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Force CPU training even if CUDA is available"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print("=" * 80)
    print("TIED AUTOENCODER TRAINING ON FULL DATASET")
    print("=" * 80)
    print(f"[{datetime.now()}] Configuration loaded")
    
    # Device setup
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    print(f"[{datetime.now()}] Using device: {device}")
    
    # Load activations
    print(f"\n[{datetime.now()}] Loading activations...")
    activations = load_activations(config=config, layer=args.layer)
    original_shape = activations.shape
    print(f"[{datetime.now()}] Activations shape: {activations.shape}")
    
    # Split data
    print(f"\n[{datetime.now()}] Splitting data (70% train, 20% val, 10% test)...")
    train_data, val_data, test_data = split_data(
        activations,
        train_fraction=0.7,
        val_fraction=0.2,
        random_seed=config.training.random_seed
    )
    print(f"[{datetime.now()}] Train: {train_data.shape}")
    print(f"[{datetime.now()}] Val:   {val_data.shape}")
    print(f"[{datetime.now()}] Test:  {test_data.shape}")
    
    # Create model
    input_dim = activations.shape[1]
    latent_dim = config.model.latent_dim
    
    print(f"\n[{datetime.now()}] Creating TiedWeightAutoencoder...")
    print(f"[{datetime.now()}] Input dim: {input_dim}, Latent dim: {latent_dim}")
    model = TiedWeightAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        device=device
    )
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
    print(f"[{datetime.now()}] Model weights initialized")
    
    # Training parameters
    num_epochs = args.epochs if args.epochs is not None else config.training.epochs
    learning_rate = args.learning_rate if args.learning_rate is not None else config.training.learning_rate
    patience = args.patience if args.patience is not None else config.training.patience
    chunk_size = args.chunk_size
    regularization_weight = config.autoencoder.regularization_weight
    
    print(f"\n[{datetime.now()}] Training parameters:")
    print(f"[{datetime.now()}] Epochs: {num_epochs}")
    print(f"[{datetime.now()}] Learning rate: {learning_rate}")
    print(f"[{datetime.now()}] Patience: {patience}")
    print(f"[{datetime.now()}] Chunk size: {chunk_size}")
    print(f"[{datetime.now()}] Regularization weight: {regularization_weight}")
    
    # Train model
    print(f"\n[{datetime.now()}] Starting training...")
    print("-" * 80)
    history = train_with_gradient_accumulation(
        model=model,
        train_data=train_data,
        val_data=val_data,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        patience=patience,
        chunk_size=chunk_size,
        device=device,
        regularization_weight=regularization_weight
    )
    print("-" * 80)
    
    # Evaluate on test set
    print(f"\n[{datetime.now()}] Evaluating on test set...")
    test_results = evaluate_test_set(
        model=model,
        test_data=test_data,
        chunk_size=chunk_size,
        device=device
    )
    print(f"[{datetime.now()}] Test set results:")
    print(f"[{datetime.now()}] Mean error: {test_results['mean']:.6f}")
    print(f"[{datetime.now()}] Std error:  {test_results['std']:.6f}")
    
    # Create output directory and save results
    print(f"\n[{datetime.now()}] Saving results...")
    output_dir = create_output_dir()
    print(f"[{datetime.now()}] Output directory: {output_dir}")
    
    model_path = save_training_results(
        model=model,
        history=history,
        test_results=test_results,
        config=config,
        output_dir=output_dir,
        activations_shape=original_shape
    )
    
    plot_training_curves(history, output_dir)
    visualize_reconstruction_errors(
        test_results['errors'],
        test_results,
        output_dir / "reconstruction_error_histogram.png",
        num_bins=50
    )
    
    # Summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"✓ Model trained for {history['epochs_trained']} epochs")
    print(f"✓ Final training loss: {history['train_loss'][-1]:.6f}")
    print(f"✓ Final validation loss: {history['val_loss'][-1]:.6f}")
    print(f"✓ Test reconstruction error: {test_results['mean']:.6f} ± {test_results['std']:.6f}")
    print(f"✓ Results saved to: {output_dir}")
    print(f"✓ Model weights saved to: {model_path}")
    print("=" * 80)
    
    return output_dir, model_path


if __name__ == "__main__":
    main()
