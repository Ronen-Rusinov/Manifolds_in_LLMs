#!/usr/bin/env python
"""
Train a tied autoencoder on the full activation dataset multiple times.

This script:
1. Loads activations from the full dataset
2. Splits data into 70% train, 20% validation, 10% test
3. Trains a TiedWeightAutoencoder multiple times with gradient accumulation
4. Tracks final loss values of all attempts in memory
5. Saves only the best model and summary statistics
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from copy import deepcopy

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


def create_output_dir(base_dir="tied_autoencoder_multiple_runs"):
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


def save_best_model(
    model,
    history,
    test_results,
    config,
    output_dir,
    activations_shape=None,
    run_number=None
):
    """Save the best model, results, and metadata.
    
    Args:
        model: Trained TiedWeightAutoencoder
        history: Training history dict with 'train_loss' and 'val_loss'
        test_results: Test evaluation results dict
        config: Config object
        output_dir: Output directory path
        activations_shape: Shape of original activations data
        run_number: The run number for this model
    """
    output_dir = Path(output_dir)
    
    # Create best_model directory
    best_dir = output_dir / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    model_path = best_dir / "model_weights.pt"
    torch.save(model.state_dict(), model_path)
    print(f"[{datetime.now()}] Best model saved to {model_path}")
    
    # Save training history
    history_path = best_dir / "training_history.npz"
    np.savez(
        history_path,
        train_loss=np.array(history['train_loss']),
        val_loss=np.array(history['val_loss'])
    )
    print(f"[{datetime.now()}] Training history saved to {history_path}")
    
    # Save test results
    test_path = best_dir / "test_results.npz"
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
        'best_from_run': run_number if run_number is not None else "unknown",
    }
    
    metadata_path = best_dir / "metadata.txt"
    with open(metadata_path, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    print(f"[{datetime.now()}] Metadata saved to {metadata_path}")
    
    return model_path, best_dir


def plot_training_curves(history, output_dir, run_number):
    """Plot and save training/validation loss curves.
    
    Args:
        history: Training history dict with 'train_loss' and 'val_loss'
        output_dir: Directory to save plot
        run_number: Run iteration number
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title(f'Training and Validation Loss Curves (Run {run_number})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = Path(output_dir) / f"training_curves_run_{run_number}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[{datetime.now()}] Training curves saved to {plot_path}")


def plot_attempts_histogram(
    all_final_losses,
    best_val_loss,
    output_dir
):
    """Plot histogram of final validation losses from all attempts.
    
    Args:
        all_final_losses: List of final validation losses for each run
        best_val_loss: Best validation loss achieved
        output_dir: Directory to save plot
    """
    plt.figure(figsize=(12, 6))
    
    n_runs = len(all_final_losses)
    runs = np.arange(1, n_runs + 1)
    colors = ['green' if loss == best_val_loss else 'steelblue' for loss in all_final_losses]
    
    plt.bar(runs, all_final_losses, color=colors, alpha=0.7, edgecolor='black')
    plt.axhline(y=best_val_loss, color='red', linestyle='--', linewidth=2, label=f'Best Loss: {best_val_loss:.6f}')
    plt.xlabel('Run Number')
    plt.ylabel('Final Validation Loss')
    plt.title(f'Final Validation Loss Across {n_runs} Runs')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (run, loss) in enumerate(zip(runs, all_final_losses)):
        plt.text(run, loss, f'{loss:.4f}', ha='center', va='bottom', fontsize=9)
    
    plot_path = Path(output_dir) / "validation_loss_histogram.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[{datetime.now()}] Validation loss histogram saved to {plot_path}")


def save_attempts_summary(
    all_attempts,
    output_dir
):
    """Save summary of all training attempts.
    
    Args:
        all_attempts: List of dicts with results from each run
        output_dir: Directory to save summary
    """
    summary_path = Path(output_dir) / "all_attempts_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("SUMMARY OF ALL TRAINING ATTEMPTS\n")
        f.write("=" * 80 + "\n\n")
        
        for i, attempt in enumerate(all_attempts, 1):
            f.write(f"Run {i}:\n")
            f.write(f"  Final Training Loss: {attempt['final_train_loss']:.6f}\n")
            f.write(f"  Final Validation Loss: {attempt['final_val_loss']:.6f}\n")
            f.write(f"  Epochs Trained: {attempt['epochs_trained']}\n")
            f.write(f"  Test Mean Error: {attempt['test_mean_error']:.6f}\n")
            f.write(f"  Test Std Error: {attempt['test_std_error']:.6f}\n")
            f.write("\n")
        
        # Add statistics
        all_final_losses = [attempt['final_val_loss'] for attempt in all_attempts]
        all_train_losses = [attempt['final_train_loss'] for attempt in all_attempts]
        all_test_errors = [attempt['test_mean_error'] for attempt in all_attempts]
        
        f.write("=" * 80 + "\n")
        f.write("STATISTICS ACROSS ALL RUNS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Final Validation Loss:\n")
        f.write(f"  Min: {np.min(all_final_losses):.6f}\n")
        f.write(f"  Max: {np.max(all_final_losses):.6f}\n")
        f.write(f"  Mean: {np.mean(all_final_losses):.6f}\n")
        f.write(f"  Std: {np.std(all_final_losses):.6f}\n\n")
        
        f.write("Final Training Loss:\n")
        f.write(f"  Min: {np.min(all_train_losses):.6f}\n")
        f.write(f"  Max: {np.max(all_train_losses):.6f}\n")
        f.write(f"  Mean: {np.mean(all_train_losses):.6f}\n")
        f.write(f"  Std: {np.std(all_train_losses):.6f}\n\n")
        
        f.write("Test Mean Error:\n")
        f.write(f"  Min: {np.min(all_test_errors):.6f}\n")
        f.write(f"  Max: {np.max(all_test_errors):.6f}\n")
        f.write(f"  Mean: {np.mean(all_test_errors):.6f}\n")
        f.write(f"  Std: {np.std(all_test_errors):.6f}\n")
    
    print(f"[{datetime.now()}] Attempts summary saved to {summary_path}")


def initialize_model(input_dim, latent_dim, device):
    """Initialize a TiedWeightAutoencoder with proper weight initialization.
    
    Args:
        input_dim: Input dimension
        latent_dim: Latent dimension
        device: Device to place model on
    
    Returns:
        Initialized TiedWeightAutoencoder
    """
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
    
    return model


def main():
    """Main training pipeline with multiple runs."""
    parser = argparse.ArgumentParser(
        description="Train TiedWeightAutoencoder multiple times and save the best model"
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
        "--num-runs",
        type=int,
        default=5,
        help="Number of training runs (default: 5)"
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
    print("TIED AUTOENCODER TRAINING - MULTIPLE RUNS")
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
    
    # Split data (same split for all runs)
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
    
    # Create output directory
    output_dir = create_output_dir()
    print(f"\n[{datetime.now()}] Output directory: {output_dir}")
    
    # Model configuration
    input_dim = activations.shape[1]
    latent_dim = config.model.latent_dim
    num_runs = args.num_runs
    
    # Training parameters
    num_epochs = args.epochs if args.epochs is not None else config.training.epochs
    learning_rate = args.learning_rate if args.learning_rate is not None else config.training.learning_rate
    patience = args.patience if args.patience is not None else config.training.patience
    chunk_size = args.chunk_size
    regularization_weight = config.autoencoder.regularization_weight
    
    print(f"\n[{datetime.now()}] Training configuration:")
    print(f"[{datetime.now()}] Input dim: {input_dim}, Latent dim: {latent_dim}")
    print(f"[{datetime.now()}] Number of runs: {num_runs}")
    print(f"[{datetime.now()}] Epochs: {num_epochs}")
    print(f"[{datetime.now()}] Learning rate: {learning_rate}")
    print(f"[{datetime.now()}] Patience: {patience}")
    print(f"[{datetime.now()}] Chunk size: {chunk_size}")
    print(f"[{datetime.now()}] Regularization weight: {regularization_weight}")
    
    # Track best model
    best_val_loss = float('inf')
    best_model_state = None
    best_history = None
    best_test_results = None
    best_run_number = None
    all_attempts = []
    
    # Run training multiple times
    print(f"\n[{datetime.now()}] Starting {num_runs} training runs...")
    print("=" * 80)
    
    for run_num in range(1, num_runs + 1):
        print(f"\n{'='*80}")
        print(f"RUN {run_num}/{num_runs}")
        print(f"{'='*80}")
        
        # Initialize model with fresh weights
        model = initialize_model(input_dim, latent_dim, device)
        print(f"[{datetime.now()}] Model initialized for run {run_num}")
        
        # Train model
        print(f"[{datetime.now()}] Starting training...")
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
        
        final_val_loss = history['val_loss'][-1]
        print(f"\n[{datetime.now()}] Run {run_num} - Final validation loss: {final_val_loss:.6f}")
        
        # Evaluate on test set
        print(f"[{datetime.now()}] Evaluating on test set...")
        test_results = evaluate_test_set(
            model=model,
            test_data=test_data,
            chunk_size=chunk_size,
            device=device
        )
        print(f"[{datetime.now()}] Test mean error: {test_results['mean']:.6f}")
        
        # Record attempt (in memory only)
        attempt = {
            'run_number': run_num,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': final_val_loss,
            'epochs_trained': history['epochs_trained'],
            'test_mean_error': test_results['mean'],
            'test_std_error': test_results['std']
        }
        all_attempts.append(attempt)
        
        # Check if this is the best run
        if final_val_loss < best_val_loss:
            best_val_loss = final_val_loss
            best_model_state = deepcopy(model.state_dict())
            best_history = deepcopy(history)
            best_test_results = deepcopy(test_results)
            best_run_number = run_num
            print(f"[{datetime.now()}] ✓ New best model! Validation loss: {best_val_loss:.6f}")
    
    print(f"\n[{datetime.now()}] All runs completed")
    print("=" * 80)
    
    # Save best model to final location
    print(f"\n[{datetime.now()}] Saving best model (from run {best_run_number})...")
    best_model = initialize_model(input_dim, latent_dim, device)
    best_model.load_state_dict(best_model_state)
    
    model_path, best_dir = save_best_model(
        model=best_model,
        history=best_history,
        test_results=best_test_results,
        config=config,
        output_dir=output_dir,
        activations_shape=original_shape,
        run_number=best_run_number
    )
    
    # Create visualizations for best model
    plot_training_curves(best_history, best_dir, "best")
    visualize_reconstruction_errors(
        best_test_results['errors'],
        best_test_results,
        best_dir / "reconstruction_error_histogram.png",
        num_bins=50
    )
    
    # Save summary of all attempts
    all_final_losses = [attempt['final_val_loss'] for attempt in all_attempts]
    plot_attempts_histogram(all_final_losses, best_val_loss, output_dir)
    save_attempts_summary(all_attempts, output_dir)
    
    # Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"✓ Completed {num_runs} training runs")
    print(f"✓ Best model from run {best_run_number}")
    print(f"✓ Best validation loss: {best_val_loss:.6f}")
    print(f"✓ Best model epochs trained: {best_history['epochs_trained']}")
    print(f"✓ Best model test error: {best_test_results['mean']:.6f} ± {best_test_results['std']:.6f}")
    print(f"✓ Validation loss range: {np.min(all_final_losses):.6f} - {np.max(all_final_losses):.6f}")
    print(f"✓ Results saved to: {output_dir}")
    print(f"✓ Best model saved to: {model_path}")
    print("=" * 80)
    
    return output_dir, model_path


if __name__ == "__main__":
    main()
