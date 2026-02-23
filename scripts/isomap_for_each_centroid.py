#This script relies on the outputs of obtain_10000_nearest_to_centroids.py
#And subsequently relies on the output of minibatch_kmeans.py as well as
#on the output of produce_balltree.py

#Their outputs are stored in 
#/outputs/Balltree/nearest_neighbors_indices_1.npy
#/outputs/minibatch_kmeans/centroids.npy
#and /outputs/Balltree/balltree_layer_18_all_parquets.pkl respectively.

#For each centroid, we use the precomputed nearest neighbor indices
#to retrieve a sorounding area of activations,
#and then we apply isomap to reduce the dimensionality of this area to 12D
#for visualisation purposes, several samples will also be reduced to 2D and 3D using isomap.

import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils.load_data import load_all_parquets

import numpy as np
import joblib
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Configuration
N_NEIGHBORS = 10 #number of neighbors for geodesic distance estimation in Isomap
N_COMPONENTS_12D = 12
N_COMPONENTS_3D = 3
N_COMPONENTS_4D = 4
N_VISUALIZATION_SAMPLES = 5  # Number of samples to visualize in 3D/4D

def load_centroids():
    """Load centroids from minibatch_kmeans."""
    centroids_path = Path(__file__).parent.parent / "results" / "minibatch_kmeans" / "centroids.npy"
    print(f"[{datetime.now()}] Loading centroids from {centroids_path}...", flush=True)
    centroids = np.load(centroids_path)
    print(f"[{datetime.now()}] Centroids loaded. Shape: {centroids.shape}", flush=True)
    return centroids

def load_neighbor_indices(indices_file="nearest_neighbors_indices_1.npy"):
    """Load precomputed nearest neighbor indices."""
    indices_path = Path(__file__).parent.parent / "results" / "Balltree" / indices_file
    print(f"[{datetime.now()}] Loading nearest neighbor indices from {indices_path}...", flush=True)
    neighbor_indices = np.load(indices_path)
    print(f"[{datetime.now()}] Neighbor indices loaded. Shape: {neighbor_indices.shape}", flush=True)
    return neighbor_indices

def load_activations():
    """Load all activation vectors and corresponding prompts."""
    print(f"[{datetime.now()}] Loading all activations and prompts...", flush=True)
    df = load_all_parquets(timing=True)
    activations = np.array(df['activation_layer_18'].tolist(), dtype=np.float32)
    prompts = df['text_prefix'].tolist() if 'text_prefix' in df.columns else [None] * len(df)
    print(f"[{datetime.now()}] Activations loaded. Shape: {activations.shape}", flush=True)
    print(f"[{datetime.now()}] Prompts loaded. Count: {len(prompts)}", flush=True)
    return activations, prompts

def apply_isomap_to_neighborhood(activations, neighbor_indices, n_components, n_neighbors):
    """Apply Isomap to a neighborhood of activations."""
    isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components, n_jobs=-1)
    embeddings = isomap.fit_transform(activations)
    return embeddings, isomap

def get_text_snippet(text, first_n=5, last_n=10):
    """Extract first and last n tokens from text."""
    if text is None:
        return "N/A"
    tokens = str(text).split()
    if len(tokens) <= first_n + last_n:
        return ' '.join(tokens)
    first_tokens = ' '.join(tokens[:first_n])
    last_tokens = ' '.join(tokens[-last_n:])
    return f"{first_tokens} [...] {last_tokens}"

def process_all_centroids(centroids, neighbor_indices, activations, prompts, offset=0, count=None):
    """Process centroids in a specified range and apply Isomap to their neighborhoods.
    
    Args:
        centroids: Array of centroid vectors
        neighbor_indices: Array of neighbor indices for each centroid
        activations: Array of activation vectors
        prompts: List of prompts corresponding to activations
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
    for dim_dir in ["12D", "3D", "4D"]:
        (output_dir / dim_dir).mkdir(parents=True, exist_ok=True)
    
    for i, centroid_idx in enumerate(range(offset, offset + count)):
        print(f"\n[{datetime.now()}] Processing centroid {centroid_idx + 1}/{n_centroids} (batch: {i + 1}/{count})...", flush=True)
        
        # Get the nearest neighbor indices for this centroid
        neighbor_idx = neighbor_indices[centroid_idx]
        
        # Retrieve activation vectors for neighbors
        neighborhood_activations = activations[neighbor_idx]
        print(f"[{datetime.now()}] Neighborhood activations shape: {neighborhood_activations.shape}", flush=True)
        
        # Apply Isomap to 12D
        print(f"[{datetime.now()}] Applying Isomap to 12D...", flush=True)
        embeddings_12d, isomap_12d = apply_isomap_to_neighborhood(
            neighborhood_activations, 
            neighbor_idx, 
            N_COMPONENTS_12D, 
            N_NEIGHBORS
        )
        
        # Save 12D results
        embeddings_12d_path = output_dir / "12D" / f"centroid_{centroid_idx:04d}_embeddings_12D.npy"
        isomap_12d_path = output_dir / "12D" / f"centroid_{centroid_idx:04d}_isomap_12D.joblib"
        np.save(embeddings_12d_path, embeddings_12d)
        joblib.dump(isomap_12d, isomap_12d_path)
        print(f"[{datetime.now()}] 12D embeddings saved to {embeddings_12d_path}", flush=True)
        
        #delete isomap_12d to free memory before next Isomap runs
        del isomap_12d
        
        if i % 10 == 0:
            print(f"[{datetime.now()}] Processed {i + 1} centroids so far...", flush=True)
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
                min(50, sample_size_3d - 1)
            )
                
            # Save 3D results
            embeddings_3d_path = output_dir / "3D" / f"centroid_{centroid_idx:04d}_embeddings_3D.npy"
            isomap_3d_path = output_dir / "3D" / f"centroid_{centroid_idx:04d}_isomap_3D.joblib"
            np.save(embeddings_3d_path, embeddings_3d)
            joblib.dump(isomap_3d, isomap_3d_path)
            print(f"[{datetime.now()}] 3D embeddings saved to {embeddings_3d_path}", flush=True)
            del isomap_3d
            
            # Apply Isomap to 4D for visualization (sample)
            print(f"[{datetime.now()}] Applying Isomap to 4D for visualization...", flush=True)
            sample_size_4d = min(N_VISUALIZATION_SAMPLES * 1000, len(neighborhood_activations))
            sample_indices_4d = np.random.choice(len(neighborhood_activations), sample_size_4d, replace=False)
            sampled_activations_4d = neighborhood_activations[sample_indices_4d]
            
            embeddings_4d, isomap_4d = apply_isomap_to_neighborhood(
                sampled_activations_4d,
                sample_indices_4d,
                N_COMPONENTS_4D,
                min(30, sample_size_4d - 1)
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
            neighborhood_prompts_3d = [neighborhood_prompts[i] for i in sample_indices_3d]
            neighborhood_prompts_4d = [neighborhood_prompts[i] for i in sample_indices_4d]
                
            # Create interactive HTML visualizations with prompt tooltips
            print(f"[{datetime.now()}] Creating interactive HTML visualizations...", flush=True)
            create_html_visualizations(embeddings_3d, embeddings_4d, neighborhood_prompts_3d, neighborhood_prompts_4d, centroid_idx, output_dir)
        
    print(f"\n[{datetime.now()}] All centroids processed successfully!", flush=True)

def create_html_visualizations(embeddings_3d, embeddings_4d, prompts_3d, prompts_4d, centroid_idx, output_dir):
    """Create interactive HTML visualizations with prompt tooltips using Plotly.
    
    For 3D: projects to 2D (x, y) with z represented by color.
    For 4D: projects to 3D (x, y, z) with w represented by color.
    """
    # Create text snippets for hover
    hover_text_3d = [get_text_snippet(p) for p in prompts_3d]
    hover_text_4d = [get_text_snippet(p) for p in prompts_4d]
    
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
        width=1000,
        height=800
    )
    html_3d_path = output_dir / "3D" / f"centroid_{centroid_idx:04d}_visualization_3D.html"
    fig_3d.write_html(str(html_3d_path))
    print(f"[{datetime.now()}] 3D interactive HTML saved to {html_3d_path}", flush=True)
    
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
        width=1000,
        height=800
    )
    html_4d_path = output_dir / "4D" / f"centroid_{centroid_idx:04d}_visualization_4D.html"
    fig_4d.write_html(str(html_4d_path))
    print(f"[{datetime.now()}] 4D interactive HTML saved to {html_4d_path}", flush=True)

def main():
    print(f"[{datetime.now()}] Starting isomap processing for each centroid...", flush=True)
    
    # Parse command-line arguments
    offset = 0
    count = None
    
    if len(sys.argv) >= 2:
        try:
            offset = int(sys.argv[1])
        except ValueError:
            print(f"Error: First argument (offset) must be an integer, got '{sys.argv[1]}'", flush=True)
            sys.exit(1)
    
    if len(sys.argv) >= 3:
        try:
            count = int(sys.argv[2])
        except ValueError:
            print(f"Error: Second argument (count) must be an integer, got '{sys.argv[2]}'", flush=True)
            sys.exit(1)
    
    if len(sys.argv) > 3:
        print(f"Warning: Ignoring extra command-line arguments beyond offset and count", flush=True)
    
    # Load required data
    centroids = load_centroids()
    neighbor_indices = load_neighbor_indices()
    activations, prompts = load_activations()
    
    # Verify shapes match
    print(f"[{datetime.now()}] Verifying data shapes...", flush=True)
    print(f"  Centroids shape: {centroids.shape}", flush=True)
    print(f"  Neighbor indices shape: {neighbor_indices.shape}", flush=True)
    print(f"  Activations shape: {activations.shape}", flush=True)
    print(f"  Prompts count: {len(prompts)}", flush=True)
    
    if centroids.shape[0] != neighbor_indices.shape[0]:
        raise ValueError("Number of centroids does not match number of neighbor indices!")
    if activations.shape[0] != len(prompts):
        raise ValueError("Number of activations does not match number of prompts!")
    
    # Process centroids in specified range
    try:
        process_all_centroids(centroids, neighbor_indices, activations, prompts, offset=offset, count=count)
    except ValueError as e:
        print(f"Error: {e}", flush=True)
        print(f"Total centroids available: {centroids.shape[0]}", flush=True)
        print(f"Usage: python {sys.argv[0]} [offset] [count]", flush=True)
        print(f"  offset: Starting centroid index (0-based, default: 0)", flush=True)
        print(f"  count: Number of centroids to process (default: all remaining)", flush=True)
        sys.exit(1)
    
    print(f"[{datetime.now()}] Pipeline completed successfully!", flush=True)

if __name__ == "__main__":
    main()

