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
from sklearn.manifold import Isomap
import plotly.graph_objects as go

# Load configuration with CLI argument overrides
parser = argparse.ArgumentParser(description="Isomap for each centroid")

# Isomap parameters
parser.add_argument("--n_components", type=int, help="Number of components for Isomap")
parser.add_argument("--n_neighbors", type=int, help="Number of neighbors for Isomap")
parser.add_argument("--k_neighbors_isomap_alt", type=int, help="Alternative number of neighbors for Isomap")
parser.add_argument("--n_centroids", type=int, help="Number of centroids")
parser.add_argument("--random_seed", type=int, help="Random seed for reproducibility")

add_config_argument(parser)
args = parser.parse_args()
config = load_config(args.config)

# Override config with CLI arguments
if args.n_components is not None:
    config.dimensionality.n_components = args.n_components
if args.n_neighbors is not None:
    config.dimensionality.n_neighbors = args.n_neighbors
if args.k_neighbors_isomap_alt is not None:
    config.clustering.k_neighbors_isomap_alt = args.k_neighbors_isomap_alt
if args.n_centroids is not None:
    config.clustering.n_centroids = args.n_centroids
if args.random_seed is not None:
    config.training.random_seed = args.random_seed

# Configuration from config object
N_NEIGHBORS = config.clustering.k_neighbors_isomap_alt
DEFAULT_N_COMPONENTS = config.dimensionality.n_components
N_COMPONENTS_3D = config.dimensionality.n_components_3d
N_COMPONENTS_4D = config.dimensionality.n_components_4d
N_VISUALIZATION_SAMPLES = config.data.n_samples_base  # Number of samples to visualize in 3D/4D

def apply_isomap_to_neighborhood(activations, neighbor_indices, n_components, n_neighbors):
    """Apply Isomap to a neighborhood of activations."""
    isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components, n_jobs=-1)
    embeddings = isomap.fit_transform(activations)
    return embeddings, isomap

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
    """Process centroids in a specified range and apply Isomap to their neighborhoods.
    
    Args:
        centroids: Array of centroid vectors
        neighbor_indices: Array of neighbor indices for each centroid
        activations: Array of activation vectors
        prompts: List of prompts corresponding to activations
        n_components: Target dimensionality for main Isomap embeddings
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
    
    output_dir = Path(__file__).parent.parent / "results" / "iso_atlas"
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
        
        # Apply Isomap to main dimensionality
        print(f"[{datetime.now()}] Applying Isomap to {n_components}D...", flush=True)
        embeddings_main, isomap_main = apply_isomap_to_neighborhood(
            neighborhood_activations, 
            neighbor_idx, 
            n_components, 
            N_NEIGHBORS
        )
        
        # Save main results
        embeddings_main_path = output_dir / main_dim_dir / f"centroid_{centroid_idx:04d}_embeddings_{n_components}D.npy"
        isomap_main_path = output_dir / main_dim_dir / f"centroid_{centroid_idx:04d}_isomap_{n_components}D.joblib"
        np.save(embeddings_main_path, embeddings_main)
        joblib.dump(isomap_main, isomap_main_path)
        print(f"[{datetime.now()}] {n_components}D embeddings saved to {embeddings_main_path}", flush=True)
        
        #delete isomap_main to free memory before next Isomap runs
        del isomap_main
        
        if i % 10 == 0:
            print(f"[{datetime.now()}] Processed {i + 1} centroids so far...", flush=True)
            embeddings_3d = None
            embeddings_4d = None
            neighborhood_prompts_3d = None
            neighborhood_prompts_4d = None

            if enable_3d:
                # Apply Isomap to 3D for visualization (sample)
                print(f"[{datetime.now()}] Applying Isomap to 3D for visualization...", flush=True)
                # Use a subset of samples for 3D to reduce computation
                sample_size_3d = min(N_VISUALIZATION_SAMPLES * 1000, len(neighborhood_activations))
                sample_indices_3d = np.random.choice(len(neighborhood_activations), sample_size_3d, replace=False)
                sampled_activations_3d = neighborhood_activations[sample_indices_3d]
            
                embeddings_3d, isomap_3d = apply_isomap_to_neighborhood(
                    sampled_activations_3d,
                    sample_indices_3d,
                    N_COMPONENTS_3D,
                    min(config.clustering.k_neighbors_3d, sample_size_3d - 1)
                )
                    
                # Save 3D results
                embeddings_3d_path = output_dir / "3D" / f"centroid_{centroid_idx:04d}_embeddings_3D.npy"
                isomap_3d_path = output_dir / "3D" / f"centroid_{centroid_idx:04d}_isomap_3D.joblib"
                np.save(embeddings_3d_path, embeddings_3d)
                joblib.dump(isomap_3d, isomap_3d_path)
                print(f"[{datetime.now()}] 3D embeddings saved to {embeddings_3d_path}", flush=True)
                del isomap_3d
                # Get prompts for all samples in this neighborhood
                neighborhood_prompts = [prompts[idx] for idx in neighbor_idx]
                neighborhood_prompts_3d = [neighborhood_prompts[i] for i in sample_indices_3d]

            if enable_4d:
                # Apply Isomap to 4D for visualization (sample)
                print(f"[{datetime.now()}] Applying Isomap to 4D for visualization...", flush=True)
                sample_size_4d = min(N_VISUALIZATION_SAMPLES * 1000, len(neighborhood_activations))
                sample_indices_4d = np.random.choice(len(neighborhood_activations), sample_size_4d, replace=False)
                sampled_activations_4d = neighborhood_activations[sample_indices_4d]
                
                embeddings_4d, isomap_4d = apply_isomap_to_neighborhood(
                    sampled_activations_4d,
                    sample_indices_4d,
                    N_COMPONENTS_4D,
                    min(config.clustering.k_neighbors_4d, sample_size_4d - 1)
                )
                    
                # Save 4D results
                embeddings_4d_path = output_dir / "4D" / f"centroid_{centroid_idx:04d}_embeddings_4D.npy"
                isomap_4d_path = output_dir / "4D" / f"centroid_{centroid_idx:04d}_isomap_4D.joblib"
                np.save(embeddings_4d_path, embeddings_4d)
                joblib.dump(isomap_4d, isomap_4d_path)
                print(f"[{datetime.now()}] 4D embeddings saved to {embeddings_4d_path}", flush=True)
                del isomap_4d
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
    if embeddings_3d is not None and prompts_3d is not None:
        # Create text snippets for hover
        hover_text_3d = [get_text_snippet(p, config) for p in prompts_3d]
        
        # 3D Interactive visualization (2D plot with color for 3rd dimension)
        fig_3d = go.Figure(data=go.Scatter(
            x=embeddings_3d[:, 0],
            y=embeddings_3d[:, 1],
            mode='markers',
            marker=dict(
                size=6,
                color=embeddings_3d[:, 2],
                colorscale='Sunsetdark',
                showscale=True,
                colorbar=dict(title="Component 3"),
                opacity=0.7
            ),
            text=hover_text_3d,
            hovertemplate='<b>Prompt (first & last 20 tokens):</b><br>%{text}<br><b>Component 3:</b> %{marker.color:.3f}<extra></extra>',
        ))
        fig_3d.update_layout(
            title=f"3D Isomap Embeddings (2D projection + color) - Centroid {centroid_idx}",
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            hovermode='closest',
            width=config.visualization.fig_width_large,
            height=config.visualization.fig_height_large
        )
        html_3d_path = output_dir / "3D" / f"centroid_{centroid_idx:04d}_visualization_3D.html"
        fig_3d.write_html(str(html_3d_path))
        print(f"[{datetime.now()}] 3D interactive HTML saved to {html_3d_path}", flush=True)
    
    if embeddings_4d is not None and prompts_4d is not None:
        # Create text snippets for hover
        hover_text_4d = [get_text_snippet(p, config) for p in prompts_4d]
        
        # 4D Interactive visualization (3D plot with color for 4th dimension)
        fig_4d = go.Figure(data=go.Scatter3d(
            x=embeddings_4d[:, 0],
            y=embeddings_4d[:, 1],
            z=embeddings_4d[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color=embeddings_4d[:, 3],
                colorscale='Sunsetdark',
                showscale=True,
                colorbar=dict(title="Component 4"),
                opacity=0.7
            ),
            text=hover_text_4d,
            hovertemplate='<b>Prompt (first & last 20 tokens):</b><br>%{text}<br><b>Component 4:</b> %{marker.color:.3f}<extra></extra>',
        ))
        fig_4d.update_layout(
            title=f"4D Isomap Embeddings (3D projection + color) - Centroid {centroid_idx}",
            scene=dict(
                xaxis_title="Component 1",
                yaxis_title="Component 2",
                zaxis_title="Component 3"
            ),
            hovermode='closest',
            width=config.visualization.fig_width_large,
            height=config.visualization.fig_height_large
        )
        html_4d_path = output_dir / "4D" / f"centroid_{centroid_idx:04d}_visualization_4D.html"
        fig_4d.write_html(str(html_4d_path))
        print(f"[{datetime.now()}] 4D interactive HTML saved to {html_4d_path}", flush=True)

def main():
    print(f"[{datetime.now()}] Starting isomap processing for each centroid...", flush=True)
    
    # Parse command-line arguments for script-specific parameters
    # (config parameters are already loaded at module level)
    parser = argparse.ArgumentParser(description="Run Isomap per centroid.")
    parser.add_argument("offset", nargs="?", type=int, default=0, help="Starting centroid index (0-based)")
    parser.add_argument("count", nargs="?", type=int, default=None, help="Number of centroids to process")
    parser.add_argument("--no-3d", action="store_true", help="Disable 3D visualization embeddings")
    parser.add_argument("--no-4d", action="store_true", help="Disable 4D visualization embeddings")
    parser.add_argument("--config", type=str, default=None, help="Path to config file (already handled)")
    
    # Parse only script-specific args, ignoring config-related ones
    args, unknown = parser.parse_known_args()

    offset = args.offset
    count = args.count
    enable_3d = not args.no_3d
    enable_4d = not args.no_4d
    
    # Load required data using shared utilities
    centroids = common.load_centroids()
    neighbor_indices = common.load_neighbor_indices()
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
        print(f"  --n-components N: Target dimensionality for main Isomap embeddings", flush=True)
        print(f"  --no-3d: Disable 3D visualization embeddings", flush=True)
        print(f"  --no-4d: Disable 4D visualization embeddings", flush=True)
        sys.exit(1)
    
    print(f"[{datetime.now()}] Pipeline completed successfully!", flush=True)

if __name__ == "__main__":
    main()

