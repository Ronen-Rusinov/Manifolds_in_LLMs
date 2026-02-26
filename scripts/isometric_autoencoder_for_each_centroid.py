import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from config_manager import load_config, add_config_argument
import argparse
from utils import common

import numpy as np
import torch
import joblib
import plotly.graph_objects as go

# Import autoencoder and training functions
from TiedWeightAutoencoder import TiedWeightAutoencoder
from train_tied_autoencoder_with_geometric_regularisation import (
    train_with_early_stopping,
    evaluate_test_set,
    compute_jacobian_orthogonality_loss
)

# Load configuration with CLI argument overrides
parser = argparse.ArgumentParser(description="Train geometrically regularized autoencoder for each centroid")

# Autoencoder parameters
parser.add_argument("--offset", nargs="?", type=int, default=0, help="Starting centroid index (0-based)")
parser.add_argument("--count", nargs="?", type=int, default=None, help="Number of centroids to process")
parser.add_argument("--n-components", type=int, help="Number of components (latent dimension) for autoencoder")
parser.add_argument("--n-centroids", type=int, help="Number of centroids")
parser.add_argument("--random-seed", type=int, help="Random seed for reproducibility")
parser.add_argument("--no-3d", action="store_true", help="Disable 3D visualization embeddings")
parser.add_argument("--no-4d", action="store_true", help="Disable 4D visualization embeddings")
parser.add_argument("--visualise-every", type=int, help="Visualize every n centroids")
parser.add_argument("--epochs", type=int, help="Number of training epochs")
parser.add_argument("--learning-rate", type=float, help="Learning rate for training")
parser.add_argument("--patience", type=int, help="Early stopping patience")
parser.add_argument("--regularization-weight", type=float, help="Weight for geometric regularization")

add_config_argument(parser)
args = parser.parse_args()
config = load_config(args.config)

# Override config with CLI arguments
if args.n_components is not None:
    config.dimensionality.n_components = args.n_components
if args.n_centroids is not None:
    config.clustering.n_centroids = args.n_centroids
if args.random_seed is not None:
    config.training.random_seed = args.random_seed
if args.visualise_every is not None:
    config.visualization.visualise_every_n_centroids = args.visualise_every
if args.epochs is not None:
    config.autoencoder.epochs = args.epochs
if args.learning_rate is not None:
    config.autoencoder.learning_rate = args.learning_rate
if args.patience is not None:
    config.autoencoder.patience = args.patience
if args.regularization_weight is not None:
    config.autoencoder.regularization_weight = args.regularization_weight

# Set random seed for reproducibility
torch.manual_seed(config.training.random_seed)
np.random.seed(config.training.random_seed)


def train_autoencoder_on_neighborhood(activations, neighbor_indices, n_components, device='cpu'):
    """Train tied-weight autoencoder with geometric regularization on a neighborhood of activations.
    
    Args:
        activations: numpy array of activation vectors (n_samples, input_dim)
        neighbor_indices: indices of neighbors (for reference, not used in training)
        n_components: latent dimension for autoencoder
        device: device to train on ('cpu' or 'cuda')
    
    Returns:
        embeddings: numpy array of latent codes (n_samples, n_components)
        model: trained TiedWeightAutoencoder model
        history: training history dict
    """
    # Diagnostic checks
    print(f"[{datetime.now()}] Autoencoder input shape: {activations.shape}", flush=True)
    print(f"[{datetime.now()}] Latent dimension: {n_components}", flush=True)
    print(f"[{datetime.now()}] Activation stats - min: {activations.min():.6f}, max: {activations.max():.6f}, mean: {activations.mean():.6f}", flush=True)
    print(f"[{datetime.now()}] Contains NaN: {np.isnan(activations).any()}, Contains Inf: {np.isinf(activations).any()}", flush=True)
    
    # Convert to torch tensors
    activations_tensor = torch.from_numpy(activations).float().to(device)
    
    # Split into train/val/test
    n_samples = activations.shape[0]
    n_train = int(n_samples * config.autoencoder.train_fraction)
    n_val = int(n_samples * config.autoencoder.val_fraction)
    
    # Shuffle indices
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    train_data = activations_tensor[train_indices]
    val_data = activations_tensor[val_indices]
    test_data = activations_tensor[test_indices]
    
    print(f"[{datetime.now()}] Data split - Train: {train_data.shape[0]}, Val: {val_data.shape[0]}, Test: {test_data.shape[0]}", flush=True)
    
    # Create model
    input_dim = activations.shape[1]
    model = TiedWeightAutoencoder(input_dim=input_dim, latent_dim=n_components, device=device)
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
    
    # Train model
    print(f"[{datetime.now()}] Training autoencoder (epochs={config.autoencoder.epochs}, lr={config.autoencoder.learning_rate}, reg_weight={config.autoencoder.regularization_weight})...", flush=True)
    history = train_with_early_stopping(
        model=model,
        train_data=train_data,
        val_data=val_data,
        num_epochs=config.autoencoder.epochs,
        learning_rate=config.autoencoder.learning_rate,
        patience=config.autoencoder.patience,
        device=device,
        regularization_weight=config.autoencoder.regularization_weight
    )
    
    # Evaluate on test set
    test_results = evaluate_test_set(model, test_data, device=device)
    print(f"[{datetime.now()}] Test set reconstruction error - Mean: {test_results['mean']:.6f}, Std: {test_results['std']:.6f}", flush=True)
    
    # Compute Jacobian orthogonality
    model.eval()
    with torch.no_grad():
        z_test_sample = model.encode(test_data[:10])
    ortho_loss = compute_jacobian_orthogonality_loss(model, z_test_sample, device)
    print(f"[{datetime.now()}] Jacobian orthogonality loss (||J^T J - I||_F): {ortho_loss.item():.6f}", flush=True)
    
    # Generate embeddings for all data
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(activations_tensor).cpu().numpy()
    
    print(f"[{datetime.now()}] Autoencoder output shape: {embeddings.shape}", flush=True)
    print(f"[{datetime.now()}] Embeddings stats - min: {embeddings.min():.6f}, max: {embeddings.max():.6f}, mean: {embeddings.mean():.6f}", flush=True)
    print(f"[{datetime.now()}] Output contains NaN: {np.isnan(embeddings).any()}, Contains Inf: {np.isinf(embeddings).any()}", flush=True)
    
    return embeddings, model, history


def get_text_snippet(text, config, first_n=None, last_n=None):
    """Extract first and last n tokens from text."""
    if first_n is None:
        first_n = config.text.first_n_tokens_isomap
    if last_n is None:
        last_n = config.text.last_n_tokens_isomap
    if text is None:
        return "N/A"
    tokens = str(text).split()
    if len(tokens) <= first_n + last_n:
        return ' '.join(tokens)
    first_tokens = ' '.join(tokens[:first_n])
    last_tokens = ' '.join(tokens[-last_n:])
    return f"{first_tokens} [...] {last_tokens}"


def process_all_centroids(
    centroids,
    neighbor_indices,
    activations,
    prompts,
    n_components,
    enable_3d=True,
    enable_4d=True,
    offset=0,
    count=None,
):
    """Process centroids in a specified range and train autoencoder on their neighborhoods.
    
    Args:
        centroids: Array of centroid vectors
        neighbor_indices: Array of neighbor indices for each centroid
        activations: Array of activation vectors
        prompts: List of prompts corresponding to activations
        n_components: Target dimensionality for main autoencoder embeddings
        enable_3d: Whether to run 3D visualization embeddings
        enable_4d: Whether to run 4D visualization embeddings
        offset: Starting centroid index (default: 0)
        count: Number of centroids to process (default: None, process all remaining)
    """
    n_centroids = centroids.shape[0]
    
    # Validate offset
    if offset < 0 or offset >= n_centroids:
        raise ValueError(f"Offset {offset} is out of range [0, {n_centroids-1}]")
    
    # Set count to remaining centroids if not specified
    if count is None:
        count = n_centroids - offset
    else:
        if count <= 0:
            raise ValueError(f"Count must be positive, got {count}")
        if offset + count > n_centroids:
            print(f"[{datetime.now()}] Warning: count {count} exceeds available centroids. Processing {n_centroids - offset} centroids instead.", flush=True)
            count = n_centroids - offset
    
    output_dir = Path(__file__).parent.parent / "results" / f"autoencoder_atlas_{n_components}D"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[{datetime.now()}] Processing {count} centroids (offset: {offset}, range: [{offset}, {offset + count - 1}])...", flush=True)
    
    # Create subdirectories for different dimensions
    main_dim_dir = f"{n_components}D"
    for dim_dir in [main_dim_dir] + (["3D"] if enable_3d else []) + (["4D"] if enable_4d else []):
        (output_dir / dim_dir).mkdir(parents=True, exist_ok=True)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{datetime.now()}] Using device: {device}", flush=True)
    
    for i, centroid_idx in enumerate(range(offset, offset + count)):
        print(f"\n[{datetime.now()}] Processing centroid {centroid_idx + 1}/{n_centroids} (batch: {i + 1}/{count})...", flush=True)
        
        # Get the nearest neighbor indices for this centroid
        neighbor_idx = neighbor_indices[centroid_idx]
        
        # Retrieve activation vectors for neighbors
        neighborhood_activations = activations[neighbor_idx]
        print(f"[{datetime.now()}] Neighborhood activations shape: {neighborhood_activations.shape}", flush=True)
        
        # Train autoencoder to main dimensionality
        print(f"[{datetime.now()}] Training autoencoder to {n_components}D...", flush=True)
        embeddings_main, model_main, history_main = train_autoencoder_on_neighborhood(
            neighborhood_activations, 
            neighbor_idx, 
            n_components,
            device=device
        )
        
        # Save main results
        embeddings_main_path = output_dir / main_dim_dir / f"centroid_{centroid_idx:04d}_embeddings_{n_components}D.npy"
        model_main_path = output_dir / main_dim_dir / f"centroid_{centroid_idx:04d}_autoencoder_{n_components}D.pt"
        history_main_path = output_dir / main_dim_dir / f"centroid_{centroid_idx:04d}_history_{n_components}D.joblib"
        
        np.save(embeddings_main_path, embeddings_main)
        torch.save(model_main.state_dict(), model_main_path)
        joblib.dump(history_main, history_main_path)
        print(f"[{datetime.now()}] {n_components}D embeddings saved to {embeddings_main_path}", flush=True)
        print(f"[{datetime.now()}] {n_components}D model saved to {model_main_path}", flush=True)
        
        # Delete model to free memory before next training runs
        del model_main
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        if i % config.visualization.visualise_every_n_centroids == 0:
            print(f"[{datetime.now()}] Processed {i + 1} centroids so far...", flush=True)
            embeddings_3d = None
            embeddings_4d = None
            neighborhood_prompts_3d = None
            neighborhood_prompts_4d = None

            if enable_3d:
                # Train autoencoder to 3D for visualization (sample)
                print(f"[{datetime.now()}] Training autoencoder to 3D for visualization...", flush=True)
                # Use a subset of samples for 3D to reduce computation
                sample_size_3d = min(config.visualization.n_samples_visualization, len(neighborhood_activations))
                sample_indices_3d = np.random.choice(len(neighborhood_activations), sample_size_3d, replace=False)
                sampled_activations_3d = neighborhood_activations[sample_indices_3d]
            
                embeddings_3d, model_3d, history_3d = train_autoencoder_on_neighborhood(
                    sampled_activations_3d,
                    sample_indices_3d,
                    3,
                    device=device
                )
                    
                # Save 3D results
                embeddings_3d_path = output_dir / "3D" / f"centroid_{centroid_idx:04d}_embeddings_3D.npy"
                model_3d_path = output_dir / "3D" / f"centroid_{centroid_idx:04d}_autoencoder_3D.pt"
                history_3d_path = output_dir / "3D" / f"centroid_{centroid_idx:04d}_history_3D.joblib"
                
                np.save(embeddings_3d_path, embeddings_3d)
                torch.save(model_3d.state_dict(), model_3d_path)
                joblib.dump(history_3d, history_3d_path)
                print(f"[{datetime.now()}] 3D embeddings saved to {embeddings_3d_path}", flush=True)
                
                del model_3d
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Get prompts for all samples in this neighborhood
                neighborhood_prompts = [prompts[idx] for idx in neighbor_idx]
                neighborhood_prompts_3d = [neighborhood_prompts[i] for i in sample_indices_3d]

            if enable_4d:
                # Train autoencoder to 4D for visualization (sample)
                print(f"[{datetime.now()}] Training autoencoder to 4D for visualization...", flush=True)
                sample_size_4d = min(config.visualization.n_samples_visualization, len(neighborhood_activations))
                sample_indices_4d = np.random.choice(len(neighborhood_activations), sample_size_4d, replace=False)
                sampled_activations_4d = neighborhood_activations[sample_indices_4d]
                print(f"[{datetime.now()}] Sample 4D activations shape: {sampled_activations_4d.shape}", flush=True)
                
                embeddings_4d, model_4d, history_4d = train_autoencoder_on_neighborhood(
                    sampled_activations_4d,
                    sample_indices_4d,
                    4,
                    device=device
                )
                    
                # Save 4D results
                embeddings_4d_path = output_dir / "4D" / f"centroid_{centroid_idx:04d}_embeddings_4D.npy"
                model_4d_path = output_dir / "4D" / f"centroid_{centroid_idx:04d}_autoencoder_4D.pt"
                history_4d_path = output_dir / "4D" / f"centroid_{centroid_idx:04d}_history_4D.joblib"
                
                np.save(embeddings_4d_path, embeddings_4d)
                torch.save(model_4d.state_dict(), model_4d_path)
                joblib.dump(history_4d, history_4d_path)
                print(f"[{datetime.now()}] 4D embeddings saved to {embeddings_4d_path}", flush=True)
                
                del model_4d
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Get prompts for all samples in this neighborhood
                neighborhood_prompts = [prompts[idx] for idx in neighbor_idx]
                neighborhood_prompts_4d = [neighborhood_prompts[i] for i in sample_indices_4d]
                
            if enable_3d or enable_4d:
                # Create interactive HTML visualizations with prompt tooltips
                print(f"[{datetime.now()}] Creating interactive HTML visualizations...", flush=True)
                create_html_visualizations(
                    embeddings_3d,
                    embeddings_4d,
                    neighborhood_prompts_3d,
                    neighborhood_prompts_4d,
                    centroid_idx,
                    output_dir,
                )
        
    print(f"\n[{datetime.now()}] All centroids processed successfully!", flush=True)


def create_html_visualizations(embeddings_3d, embeddings_4d, prompts_3d, prompts_4d, centroid_idx, output_dir):
    """Create interactive HTML visualizations with prompt tooltips using Plotly.
    
    For 3D: projects to 2D (x, y) with z represented by color.
    For 4D: projects to 3D (x, y, z) with w represented by color.
    """
    def filter_valid_data(embeddings, prompts):
        """Filter out rows with NaN or Inf values."""
        valid_mask = np.isfinite(embeddings).all(axis=1)
        return embeddings[valid_mask], [prompts[i] for i in range(len(prompts)) if valid_mask[i]]
    
    if embeddings_3d is not None and prompts_3d is not None:
        # Filter out invalid data
        embeddings_3d, prompts_3d = filter_valid_data(embeddings_3d, prompts_3d)
        
        if len(embeddings_3d) == 0:
            print(f"[{datetime.now()}] Warning: No valid 3D embeddings for centroid {centroid_idx}", flush=True)
        else:
            # Create text snippets for hover
            hover_text_3d = [get_text_snippet(p, config) for p in prompts_3d]
            
            # 3D Interactive visualization (2D plot with color for 3rd dimension)
            fig_3d = go.Figure(data=go.Scatter(
                x=embeddings_3d[:, 0].tolist(),
                y=embeddings_3d[:, 1].tolist(),
                mode='markers',
                marker=dict(
                    size=6,
                    color=embeddings_3d[:, 2].tolist(),
                    colorscale='Sunsetdark',
                    showscale=True,
                    colorbar=dict(title="Component 3"),
                    opacity=0.7
                ),
                text=hover_text_3d,
                hovertemplate='<b>Prompt (first & last tokens):</b><br>%{text}<br><b>Component 3:</b> %{marker.color:.3f}<extra></extra>',
            ))
            fig_3d.update_layout(
                title=f"3D Autoencoder Embeddings (2D projection + color) - Centroid {centroid_idx}",
                xaxis_title="Component 1",
                yaxis_title="Component 2",
                hovermode='closest',
            )
            html_3d_path = output_dir / "3D" / f"centroid_{centroid_idx:04d}_visualization_3D.html"
            fig_3d.write_html(str(html_3d_path), include_plotlyjs=True)
            print(f"[{datetime.now()}] 3D interactive HTML saved to {html_3d_path}", flush=True)
    
    if embeddings_4d is not None and prompts_4d is not None:
        # Filter out invalid data
        embeddings_4d, prompts_4d = filter_valid_data(embeddings_4d, prompts_4d)
        
        if len(embeddings_4d) == 0:
            print(f"[{datetime.now()}] Warning: No valid 4D embeddings for centroid {centroid_idx}", flush=True)
        else:
            # Create text snippets for hover
            hover_text_4d = [get_text_snippet(p, config) for p in prompts_4d]
            
            # 4D Interactive visualization (3D plot with color for 4th dimension)
            fig_4d = go.Figure(data=go.Scatter3d(
                x=embeddings_4d[:, 0].tolist(),
                y=embeddings_4d[:, 1].tolist(),
                z=embeddings_4d[:, 2].tolist(),
                mode='markers',
                marker=dict(
                    size=4,
                    color=embeddings_4d[:, 3].tolist(),
                    colorscale='Sunsetdark',
                    showscale=True,
                    colorbar=dict(title="Component 4"),
                    opacity=0.7
                ),
                text=hover_text_4d,
                hovertemplate='<b>Prompt (first & last tokens):</b><br>%{text}<br><b>Component 4:</b> %{marker.color:.3f}<extra></extra>',
            ))
            fig_4d.update_layout(
                title=f"4D Autoencoder Embeddings (3D projection + color) - Centroid {centroid_idx}",
                scene=dict(
                    xaxis_title="Component 1",
                    yaxis_title="Component 2",
                    zaxis_title="Component 3"
                ),
                hovermode='closest',
            )
            html_4d_path = output_dir / "4D" / f"centroid_{centroid_idx:04d}_visualization_4D.html"
            fig_4d.write_html(str(html_4d_path), include_plotlyjs=True)
            print(f"[{datetime.now()}] 4D interactive HTML saved to {html_4d_path}", flush=True)


def main():
    print(f"[{datetime.now()}] Starting autoencoder processing for each centroid...", flush=True)
    
    # Parse command-line arguments for script-specific parameters
    # (config parameters are already loaded at module level)

    # Parse only script-specific args, ignoring config-related ones
    args, unknown = parser.parse_known_args()

    offset = args.offset
    count = args.count
    enable_3d = not args.no_3d
    enable_4d = not args.no_4d
    
    # Load required data using shared utilities
    # Centroid filename is of the format f"centroids_{config.clustering.n_centroids}.npy", 
    # and neighbor indices filename is of the format 
    # f'nearest_{k_nearest}_neighbors_indices_layer_{config.model.layer_for_activation}_n_centroids_{config.clustering.n_centroids}.npy
    centroids = common.load_centroids(f"minibatch_kmeans_{config.clustering.n_centroids}")
    neighbor_indices = common.load_neighbor_indices(
        f"nearest_{config.clustering.k_nearest_large}_neighbors_indices_layer_{config.model.layer_for_activation}_n_centroids_{config.clustering.n_centroids}.npy"
    )
    activations, prompts = common.load_activations_with_prompts(config=config)
    
    # Validate data consistency
    common.validate_data_consistency(centroids, neighbor_indices, activations, prompts)
    
    # Process centroids in specified range
    try:
        process_all_centroids(
            centroids,
            neighbor_indices,
            activations,
            prompts,
            config.dimensionality.n_components,
            enable_3d=enable_3d,
            enable_4d=enable_4d,
            offset=offset,
            count=count,
        )
    except ValueError as e:
        print(f"Error: {e}", flush=True)
        print(f"Total centroids available: {centroids.shape[0]}", flush=True)
        print(f"Usage: python {sys.argv[0]} [--offset N] [--count N]", flush=True)
        print(f"  --offset N: Starting centroid index (0-based, default: 0)", flush=True)
        print(f"  --count N: Number of centroids to process (default: all remaining)", flush=True)
        print(f"  --n-components N: Target dimensionality for main autoencoder embeddings", flush=True)
        print(f"  --no-3d: Disable 3D visualization embeddings", flush=True)
        print(f"  --no-4d: Disable 4D visualization embeddings", flush=True)
        print(f"  --epochs N: Number of training epochs", flush=True)
        print(f"  --learning-rate F: Learning rate for training", flush=True)
        print(f"  --patience N: Early stopping patience", flush=True)
        print(f"  --regularization-weight F: Weight for geometric regularization", flush=True)
        sys.exit(1)
    
    print(f"[{datetime.now()}] Pipeline completed successfully!", flush=True)


if __name__ == "__main__":
    main()
