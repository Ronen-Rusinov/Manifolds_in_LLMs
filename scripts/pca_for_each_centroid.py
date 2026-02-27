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
import joblib
from sklearn.decomposition import PCA
import plotly.graph_objects as go

# Load configuration with CLI argument overrides
parser = argparse.ArgumentParser(description="PCA for each centroid")

# PCA parameters
parser.add_argument("--offset", nargs="?", type=int, default=0, help="Starting centroid index (0-based)")
parser.add_argument("--count", nargs="?", type=int, default=None, help="Number of centroids to process")
parser.add_argument("--n-components", type=int ,help="Number of components for PCA")
parser.add_argument("--n-centroids", type=int, help="Number of centroids")
parser.add_argument("--random-seed", type=int, help="Random seed for reproducibility")
parser.add_argument("--no-3d", action="store_true", help="Disable 3D visualization embeddings")
parser.add_argument("--no-4d", action="store_true", help="Disable 4D visualization embeddings")
parser.add_argument("--visualise-every", type=int, help="Visualize every n centroids")

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


def apply_pca_to_neighborhood(activations, neighbor_indices, n_components):
    """Apply PCA to a neighborhood of activations."""
    # Diagnostic checks
    print(f"[{datetime.now()}] PCA input shape: {activations.shape}", flush=True)
    print(f"[{datetime.now()}] PCA n_components: {n_components}", flush=True)
    print(f"[{datetime.now()}] Activation stats - min: {activations.min():.6f}, max: {activations.max():.6f}, mean: {activations.mean():.6f}", flush=True)
    print(f"[{datetime.now()}] Contains NaN: {np.isnan(activations).any()}, Contains Inf: {np.isinf(activations).any()}", flush=True)
    
    # Handle NaN/Inf values - replace with zeros or mean
    if np.isnan(activations).any() or np.isinf(activations).any():
        print(f"[{datetime.now()}] WARNING: Detected NaN/Inf values, replacing with zeros", flush=True)
        activations = np.nan_to_num(activations, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check for zero/near-zero variance that can cause singular matrices
    variances = np.var(activations, axis=0)
    min_variance = np.min(variances)
    zero_var_count = np.sum(variances < 1e-10)
    print(f"[{datetime.now()}] Variance stats - min: {min_variance:.10f}, zero/near-zero count: {zero_var_count}/{len(variances)}", flush=True)
    
    # Add small regularization noise if variance is too low (prevents singular matrices)
    if min_variance < 1e-10 or zero_var_count > 0:
        print(f"[{datetime.now()}] WARNING: Low/zero variance detected, adding regularization noise (1e-8)", flush=True)
        noise = np.random.RandomState(config.training.random_seed).normal(0, 1e-8, activations.shape)
        activations = activations + noise
    
    # Ensure n_components doesn't exceed number of features or samples
    # Also leave room for numerical rank being less than theoretical rank
    n_components_safe = min(n_components, activations.shape[0] - 1, activations.shape[1] - 1)
    if n_components_safe != n_components:
        print(f"[{datetime.now()}] Warning: Reducing n_components from {n_components} to {n_components_safe} (input shape: {activations.shape})", flush=True)
    
    # Use svd_solver='full' for better numerical stability
    try:
        pca = PCA(n_components=n_components_safe, random_state=config.training.random_seed, svd_solver='full')
        embeddings = pca.fit_transform(activations)
    except Exception as e:
        print(f"[{datetime.now()}] ERROR: PCA failed with error: {e}", flush=True)
        print(f"[{datetime.now()}] Attempting with reduced components...", flush=True)
        # Fallback: try with fewer components
        n_components_fallback = min(n_components_safe // 2, 5)
        pca = PCA(n_components=n_components_fallback, random_state=config.training.random_seed, svd_solver='full')
        embeddings = pca.fit_transform(activations)
        print(f"[{datetime.now()}] Successfully reduced to {n_components_fallback} components", flush=True)
    
    print(f"[{datetime.now()}] PCA output shape: {embeddings.shape}", flush=True)
    print(f"[{datetime.now()}] Explained variance ratio: {pca.explained_variance_ratio_.sum():.6f}", flush=True)
    print(f"[{datetime.now()}] Embeddings stats - min: {embeddings.min():.6f}, max: {embeddings.max():.6f}, mean: {embeddings.mean():.6f}", flush=True)
    print(f"[{datetime.now()}] Output contains NaN: {np.isnan(embeddings).any()}, Contains Inf: {np.isinf(embeddings).any()}", flush=True)
    
    return embeddings, pca

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
    """Process centroids in a specified range and apply PCA to their neighborhoods.
    
    Args:
        centroids: Array of centroid vectors
        neighbor_indices: Array of neighbor indices for each centroid
        activations: Array of activation vectors
        prompts: List of prompts corresponding to activations
        n_components: Target dimensionality for main PCA embeddings
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
    
    output_dir = Path(__file__).parent.parent / "results" / f"pca_atlas_{n_components}D"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[{datetime.now()}] Processing {count} centroids (offset: {offset}, range: [{offset}, {offset + count - 1}])...", flush=True)
    
    # Create subdirectories for different dimensions
    main_dim_dir = f"{n_components}D"
    for dim_dir in [main_dim_dir] + (["3D"] if enable_3d else []) + (["4D"] if enable_4d else []):
        (output_dir / dim_dir).mkdir(parents=True, exist_ok=True)
    
    for i, centroid_idx in enumerate(range(offset, offset + count)):
        print(f"\n[{datetime.now()}] Processing centroid {centroid_idx + 1}/{n_centroids} (batch: {i + 1}/{count})...", flush=True)
        
        # Get the nearest neighbor indices for this centroid
        neighbor_idx = neighbor_indices[centroid_idx]
        
        # Retrieve activation vectors for neighbors
        neighborhood_activations = activations[neighbor_idx]
        print(f"[{datetime.now()}] Neighborhood activations shape: {neighborhood_activations.shape}", flush=True)
        
        # Apply PCA to main dimensionality
        print(f"[{datetime.now()}] Applying PCA to {n_components}D...", flush=True)
        embeddings_main, pca_main = apply_pca_to_neighborhood(
            neighborhood_activations, 
            neighbor_idx, 
            n_components
        )
        
        # Save main results
        embeddings_main_path = output_dir / main_dim_dir / f"centroid_{centroid_idx:04d}_embeddings_{n_components}D.npy"
        pca_main_path = output_dir / main_dim_dir / f"centroid_{centroid_idx:04d}_pca_{n_components}D.joblib"
        np.save(embeddings_main_path, embeddings_main)
        joblib.dump(pca_main, pca_main_path)
        print(f"[{datetime.now()}] {n_components}D embeddings saved to {embeddings_main_path}", flush=True)
        
        #delete pca_main to free memory before next PCA runs
        del pca_main
        
        if i % config.visualization.visualise_every_n_centroids == 0:
            print(f"[{datetime.now()}] Processed {i + 1} centroids so far...", flush=True)
            embeddings_3d = None
            embeddings_4d = None
            neighborhood_prompts_3d = None
            neighborhood_prompts_4d = None

            if enable_3d:
                # Apply PCA to 3D for visualization (sample)
                print(f"[{datetime.now()}] Applying PCA to 3D for visualization...", flush=True)
                # Use a subset of samples for 3D to reduce computation
                sample_size_3d = min(config.visualization.n_samples_visualization * 1000, len(neighborhood_activations))
                sample_indices_3d = np.random.choice(len(neighborhood_activations), sample_size_3d, replace=False)
                sampled_activations_3d = neighborhood_activations[sample_indices_3d]
            
                embeddings_3d, pca_3d = apply_pca_to_neighborhood(
                    sampled_activations_3d,
                    sample_indices_3d,
                    3
                )
                    
                # Save 3D results
                embeddings_3d_path = output_dir / "3D" / f"centroid_{centroid_idx:04d}_embeddings_3D.npy"
                pca_3d_path = output_dir / "3D" / f"centroid_{centroid_idx:04d}_pca_3D.joblib"
                np.save(embeddings_3d_path, embeddings_3d)
                joblib.dump(pca_3d, pca_3d_path)
                print(f"[{datetime.now()}] 3D embeddings saved to {embeddings_3d_path}", flush=True)
                del pca_3d
                # Get prompts for all samples in this neighborhood
                neighborhood_prompts = [prompts[idx] for idx in neighbor_idx]
                neighborhood_prompts_3d = [neighborhood_prompts[i] for i in sample_indices_3d]

            if enable_4d:
                # Apply PCA to 4D for visualization (sample)
                print(f"[{datetime.now()}] Applying PCA to 4D for visualization...", flush=True)
                sample_size_4d = min(config.visualization.n_samples_visualization * 1000, len(neighborhood_activations))
                sample_indices_4d = np.random.choice(len(neighborhood_activations), sample_size_4d, replace=False)
                sampled_activations_4d = neighborhood_activations[sample_indices_4d]
                print(f"[{datetime.now()}] Sample 4D activations shape: {sampled_activations_4d.shape}", flush=True)
                
                embeddings_4d, pca_4d = apply_pca_to_neighborhood(
                    sampled_activations_4d,
                    sample_indices_4d,
                    4
                )
                    
                # Save 4D results
                embeddings_4d_path = output_dir / "4D" / f"centroid_{centroid_idx:04d}_embeddings_4D.npy"
                pca_4d_path = output_dir / "4D" / f"centroid_{centroid_idx:04d}_pca_4D.joblib"
                np.save(embeddings_4d_path, embeddings_4d)
                joblib.dump(pca_4d, pca_4d_path)
                print(f"[{datetime.now()}] 4D embeddings saved to {embeddings_4d_path}", flush=True)
                del pca_4d
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
                hovertemplate='<b>Prompt (first & last 20 tokens):</b><br>%{text}<br><b>Component 3:</b> %{marker.color:.3f}<extra></extra>',
            ))
            fig_3d.update_layout(
                title=f"3D PCA Embeddings (2D projection + color) - Centroid {centroid_idx}",
                xaxis_title="Component 1",
                yaxis_title="Component 2",
                hovermode='closest',
                #width=config.visualization.fig_width_large,
                #height=config.visualization.fig_height_large
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
                hovertemplate='<b>Prompt (first & last 20 tokens):</b><br>%{text}<br><b>Component 4:</b> %{marker.color:.3f}<extra></extra>',
            ))
            fig_4d.update_layout(
                title=f"4D PCA Embeddings (3D projection + color) - Centroid {centroid_idx}",
                scene=dict(
                    xaxis_title="Component 1",
                    yaxis_title="Component 2",
                    zaxis_title="Component 3"
                ),
                hovermode='closest',
                #width=config.visualization.fig_width_large,
                #height=config.visualization.fig_height_large
            )
            html_4d_path = output_dir / "4D" / f"centroid_{centroid_idx:04d}_visualization_4D.html"
            fig_4d.write_html(str(html_4d_path), include_plotlyjs=True)
            print(f"[{datetime.now()}] 4D interactive HTML saved to {html_4d_path}", flush=True)

def main():
    print(f"[{datetime.now()}] Starting PCA processing for each centroid...", flush=True)
    
    # Parse command-line arguments for script-specific parameters
    # (config parameters are already loaded at module level)

    # Parse only script-specific args, ignoring config-related ones
    args, unknown = parser.parse_known_args()

    offset = args.offset
    count = args.count
    enable_3d = not args.no_3d
    enable_4d = not args.no_4d
    
    # Load required data using shared utilities
    #Centroid filename is of the format f"centroids_{config.clustering.n_centroids}.npy", and neighbor indices filename is of the format f'nearest_{k_nearest}_neighbors_indices_layer_{config.model.layer_for_activation}_n_centroids_{config.clustering.n_centroids}.npy
    centroids = common.load_centroids(f"minibatch_kmeans_{config.clustering.n_centroids}")
    neighbor_indices = common.load_neighbor_indices(f"nearest_{config.clustering.k_nearest_large}_neighbors_indices_layer_{config.model.layer_for_activation}_n_centroids_{config.clustering.n_centroids}.npy")
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
        print(f"Usage: python {sys.argv[0]} [offset] [count]", flush=True)
        print(f"  offset: Starting centroid index (0-based, default: 0)", flush=True)
        print(f"  count: Number of centroids to process (default: all remaining)", flush=True)
        print(f"  --n-components N: Target dimensionality for main PCA embeddings", flush=True)
        print(f"  --no-3d: Disable 3D visualization embeddings", flush=True)
        print(f"  --no-4d: Disable 4D visualization embeddings", flush=True)
        sys.exit(1)
    
    print(f"[{datetime.now()}] Pipeline completed successfully!", flush=True)

if __name__ == "__main__":
    main()
