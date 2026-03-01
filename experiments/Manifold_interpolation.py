"""
This experiment is about naive interpolation vs manifold-projectd interpolation.

That is, between interpolating the hidden states of two prompts, and then decoding the interpolated hidden states, vs projecting the interpolated hidden states back onto the manifold of valid hidden states
using the infrastructure in this repo, specifically atlas_loader.py, and then decoding the projected hidden states.

It should be noted that projection is done via reconstruction through an encoder-decoder pair, whatever equivalent exists in the method
I.e, for the autoencoder atlas it really is just a forward pass
For for isomap it uses the built in methods it supplies
And for PCA it uses projection onto the principle components, and then reconstruction back to the original space

Ideally we'd run gradient descent on the projected path to achieve a geodesic, but this is waaay outside the scope of this paper

Afterwards, each point in both interpolation methods will be patched into the generation process of a prompt designed to
decode continous embeddings into natural language, in a form reminiscent of:

"A -> A ; B -> B ; C -> C ; ? -> "
Where ? is the placeholder for the interpolated hidden state, and A,B,C,... are 
randomly sampled tokens and phrases to encourage the model to decode the hidden state into natural language, and not just output a string of unintalligable tokens.
"""

import argparse
import os
import sys
import random
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.neighbors import BallTree

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from config_manager import Config, load_config, add_config_argument
from atlas_loader import AtlasLoader
from patchscopes import PatchscopesWrapper, prepare_soft_prompt_source_inputs
import paths


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_balltree(layer: int, config: Config) -> Tuple[BallTree, np.ndarray]:
    """Load pre-built BallTree for k-nearest neighbor queries.
    
    Args:
        layer: Layer index for activation extraction
        config: Config object
    
    Returns:
        Tuple of (BallTree, data_points) where data_points are the activation vectors
    
    Raises:
        FileNotFoundError: If BallTree file does not exist
    """
    balltree_path = paths.get_results_dir() / 'Balltree' / f'balltree_layer_{layer}_all_parquets.pkl'
    
    if not balltree_path.exists():
        raise FileNotFoundError(
            f"BallTree not found at {balltree_path}. "
            f"Please run scripts/produce_balltree.py first."
        )
    
    print(f"[{datetime.now()}] Loading BallTree from {balltree_path}...", flush=True)
    with open(balltree_path, 'rb') as f:
        tree = pickle.load(f)
    
    # Load the actual data points for weighted projection
    # Note: BallTree stores the data internally
    # Convert to proper numpy array to avoid Cython memoryview indexing issues
    data_points = np.asarray(tree.data, dtype=np.float32)
    
    print(f"[{datetime.now()}] BallTree loaded with {len(data_points)} points.", flush=True)
    return tree, data_points


def linear_interpolate(start: torch.Tensor, end: torch.Tensor, num_steps: int = 10) -> List[torch.Tensor]:
    """Create linear interpolation between two tensors.
    
    Args:
        start: Starting tensor
        end: Ending tensor
        num_steps: Number of interpolation steps (including start and end)
    
    Returns:
        List of interpolated tensors
    """
    alphas = torch.linspace(0, 1, num_steps)
    interpolations = []
    
    for alpha in alphas:
        interpolated = (1 - alpha) * start + alpha * end
        interpolations.append(interpolated)
    
    return interpolations


def project_local_pca(
    hidden_state: torch.Tensor,
    balltree: BallTree,
    data_points: np.ndarray,
    latent_dim: int = 50,
    k_nearest: Optional[int] = None
) -> torch.Tensor:
    """Project a hidden state back onto the manifold using local PCA.
    
    This function:
    1. Finds the k nearest neighbors (k = 2*latent_dim by default)
    2. Applies PCA to the local neighborhood defined by these neighbors
    3. Projects the original point onto the lower-dimensional manifold
    4. Reconstructs back to ambient space
    
    Args:
        hidden_state: Hidden state tensor to project (shape: [hidden_dim])
        balltree: Pre-built BallTree for efficient nearest neighbor search
        data_points: The actual data points corresponding to the BallTree
        latent_dim: Latent dimensionality, used to determine k if not provided
        k_nearest: Number of nearest neighbors (default: 2*latent_dim)
    
    Returns:
        Projected hidden state tensor
    """
    from sklearn.decomposition import PCA
    
    if k_nearest is None:
        k_nearest = 2 * latent_dim
    
    # Convert to numpy
    hidden_np = hidden_state.cpu().numpy()
    original_shape = hidden_np.shape
    if hidden_np.ndim == 1:
        hidden_np = hidden_np[np.newaxis, :]  # Add batch dimension
    
    # Find k nearest neighbors in the actual data
    distances, indices = balltree.query(hidden_np, k=k_nearest + 1)  # +1 to include the point itself if it's in the tree
    
    # Flatten if necessary
    distances = distances[0] if distances.ndim > 1 else distances
    indices = indices[0] if indices.ndim > 1 else indices
    
    # Convert indices to numpy array to fix Cython memoryview indexing issue
    indices = np.asarray(indices, dtype=np.int64)
    
    # Get the nearest neighbor points (excluding the query point itself if it's the closest)
    # If the closest point has distance ~0, it's likely the point itself, so skip it
    if distances[0] < 1e-6:
        neighbor_points = data_points[indices[1:k_nearest + 1]]
    else:
        test = data_points[0]
        test = data_points[indices[0]]
        neighbor_points = data_points[indices[:k_nearest]]
    
    # Compute the center of the local neighborhood
    center = np.mean(neighbor_points, axis=0)
    
    # Center the neighbors and the query point relative to the neighborhood center
    centered_neighbors = neighbor_points - center
    centered_query = hidden_np[0] - center
    
    # Apply PCA to the centered neighbors
    pca = PCA(n_components=min(latent_dim, centered_neighbors.shape[0], centered_neighbors.shape[1]))
    pca.fit(centered_neighbors)
    
    # Project the centered query point onto the local PCA space
    query_embedded = pca.transform(centered_query[np.newaxis, :])[0]
    
    # Reconstruct back to the centered ambient space
    projected_centered = pca.inverse_transform(query_embedded[np.newaxis, :])[0]
    
    # Add back the center offset to get the final projected point
    projected = projected_centered + center
    
    # Convert back to torch tensor, preserving original shape
    projected_tensor = torch.from_numpy(projected).to(hidden_state.device).to(hidden_state.dtype)
    
    return projected_tensor


def project_atlas_method(
    hidden_state: torch.Tensor,
    atlas_loader: AtlasLoader,
    method: str = "pca",
    n_components: int = 50,
    device: str = 'cpu'
) -> torch.Tensor:
    """Project a hidden state onto the manifold using atlas-based methods.
    
    This function uses pre-trained dimensionality reduction models from the atlas
    (PCA or Autoencoder) to project points onto the learned manifold.
    
    Args:
        hidden_state: Hidden state tensor to project (shape: [hidden_dim])
        atlas_loader: AtlasLoader instance
        method: Projection method ("pca" or "autoencoder")
        n_components: Latent dimensionality
        device: Device for computation
    
    Returns:
        Projected hidden state tensor
    """
    # Convert to numpy
    hidden_np = hidden_state.cpu().numpy()
    if hidden_np.ndim == 1:
        hidden_np = hidden_np[np.newaxis, :]  # Add batch dimension
    
    # Find nearest centroid
    centroids = atlas_loader.load_centroids()
    distances, centroid_indices = atlas_loader.get_nearest_centroids(hidden_np[0], k=1)
    nearest_centroid_idx = int(centroid_indices[0])
    
    print(f"[{datetime.now()}] Projecting onto {method} atlas (centroid {nearest_centroid_idx})...", flush=True)
    
    # Embed into latent space
    embedding = atlas_loader.embed(hidden_np, nearest_centroid_idx, method=method, n_components=n_components)
    
    # Reconstruct back to ambient space (this projects onto the manifold)
    reconstructed = atlas_loader.reconstruct(
        embedding,
        nearest_centroid_idx,
        method=method,
        n_components=n_components
    )
    
    # Convert back to torch tensor
    projected_tensor = torch.from_numpy(reconstructed[0]).to(hidden_state.device).to(hidden_state.dtype)
    
    return projected_tensor


def project_onto_manifold(
    hidden_state: torch.Tensor,
    method: str = "local_pca",
    balltree: Optional[BallTree] = None,
    data_points: Optional[np.ndarray] = None,
    atlas_loader: Optional[AtlasLoader] = None,
    latent_dim: int = 50,
    device: str = 'cpu'
) -> torch.Tensor:
    """Project a hidden state onto the manifold using the specified method.
    
    Args:
        hidden_state: Hidden state tensor to project (shape: [hidden_dim])
        method: Projection method ("local_pca", "pca", or "autoencoder")
        balltree: Pre-built BallTree (required for local_pca)
        data_points: Actual data points (required for local_pca)
        atlas_loader: AtlasLoader instance (required for atlas methods)
        latent_dim: Latent dimensionality
        device: Device for computation
    
    Returns:
        Projected hidden state tensor
    """
    if method == "local_pca":
        if balltree is None or data_points is None:
            raise ValueError("BallTree and data_points are required for local_pca method")
        return project_local_pca(hidden_state, balltree, data_points, latent_dim)
    elif method in ["pca", "autoencoder"]:
        if atlas_loader is None:
            raise ValueError(f"atlas_loader is required for {method} method")
        return project_atlas_method(hidden_state, atlas_loader, method, latent_dim, device)
    else:
        raise ValueError(f"Unknown projection method: {method}. Must be 'local_pca', 'pca', or 'autoencoder'")


def generate_decoding_prompt(num_examples: int = 3) -> str:
    """Generate a prompt with random examples to help decode continuous embeddings.
    
    Args:
        num_examples: Number of random token examples to include
    
    Returns:
        Decoding prompt template with placeholder
    """
    # Sample random tokens/phrases
    example_tokens = [
        "cat" , "seven", "hello"
    ]
    
    random.shuffle(example_tokens)
    examples = example_tokens[:num_examples]
    
    # Build prompt
    prompt_parts = [f"{token}->{token}" for token in examples]
    prompt_parts.append("?")  # Placeholder for the interpolated hidden state
    
    return "; ".join(prompt_parts)


def decode_hidden_state_with_patchscopes(
    hidden_state: torch.Tensor,
    patchscopes_wrapper: PatchscopesWrapper,
    source_layer: int,
    target_layer: int,
    device: torch.device
) -> str:
    """Decode a hidden state using patchscopes.
    
    Args:
        hidden_state: Hidden state to decode (shape: [hidden_dim])
        patchscopes_wrapper: PatchscopesWrapper instance
        source_layer: Layer the hidden state came from
        target_layer: Layer to patch into
        device: Device to run on
    
    Returns:
        Decoded text string
    """
    # Ensure hidden state has correct shape (1, hidden_dim) for patchscopes
    if hidden_state.ndim == 1:
        hidden_state = hidden_state.unsqueeze(0)
    
    # Generate decoding prompt
    target_prompt = generate_decoding_prompt(num_examples=3)
    target_placeholder = "?"
    
    # Get forbidden token IDs from the prompt to avoid repetition
    tokenizer = patchscopes_wrapper.tokenizer
    prompt_token_ids = tokenizer.encode(target_prompt, add_special_tokens=False)
    bad_words_ids = [[token_id] for token_id in set(prompt_token_ids)]
    
    # Use patchscopes to decode with forbidden tokens from the prompt
    decoded_text = patchscopes_wrapper.patchscopes(
        source_reprs=hidden_state,
        target_prompt=target_prompt,
        target_placeholder=target_placeholder,
        source_layer=source_layer,
        target_layer=target_layer,
        end_phrase="\n",
        max_new_tokens=10,
        do_sample=False,
        bad_words_ids=bad_words_ids
    )
    
    return decoded_text.strip()


def run_interpolation_experiment(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: Config,
    prompt_a: str,
    prompt_b: str,
    source_layer: int = 18,
    target_layer: int = 1,
    num_interpolation_steps: int = 11,
    method: str = "local_pca",
    latent_dim: int = 50,
    balltree: Optional[BallTree] = None,
    data_points: Optional[np.ndarray] = None,
    atlas_loader: Optional[AtlasLoader] = None,
    device: torch.device = None
):
    """Run the full interpolation experiment.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        config: Config object
        prompt_a: First prompt
        prompt_b: Second prompt
        source_layer: Layer to extract hidden states from
        target_layer: Layer to patch into for decoding
        num_interpolation_steps: Number of interpolation points
        method: Projection method ("local_pca", "pca", or "autoencoder")
        latent_dim: Latent dimensionality
        balltree: Pre-built BallTree (required for local_pca)
        data_points: Actual data points (required for local_pca)
        atlas_loader: AtlasLoader instance (required for atlas methods)
        device: Device to run on
    """
    if device is None:
        device = next(model.parameters()).device
    
    print(f"\n{'='*80}")
    print(f"MANIFOLD INTERPOLATION EXPERIMENT")
    print(f"{'='*80}")
    print(f"Prompt A: {prompt_a}")
    print(f"Prompt B: {prompt_b}")
    print(f"Source layer: {source_layer}, Target layer: {target_layer}")
    print(f"Interpolation steps: {num_interpolation_steps}")
    print(f"Projection method: {method} (latent_dim={latent_dim})")
    print(f"{'='*80}\n")
    
    # Initialize patchscopes
    patchscopes_wrapper = PatchscopesWrapper(model, tokenizer, device=device)
    
    # Extract hidden states from both prompts
    print(f"[{datetime.now()}] Extracting hidden states from prompts...", flush=True)
    hidden_states_a = patchscopes_wrapper.prepare_source_inputs(prompt_a, prompt_a)
    hidden_states_b = patchscopes_wrapper.prepare_source_inputs(prompt_b, prompt_b)
    
    # Get the hidden states at the source layer (take mean if multiple tokens)
    hs_a = hidden_states_a[source_layer].mean(dim=0)  # Shape: [hidden_dim]
    hs_b = hidden_states_b[source_layer].mean(dim=0)  # Shape: [hidden_dim]
    
    print(f"Hidden state A shape: {hs_a.shape}")
    print(f"Hidden state B shape: {hs_b.shape}")
    
    # Create naive linear interpolations
    print(f"\n[{datetime.now()}] Creating naive linear interpolations...", flush=True)
    naive_interpolations = linear_interpolate(hs_a, hs_b, num_steps=num_interpolation_steps)
    
    # Create manifold-projected interpolations
    print(f"\n[{datetime.now()}] Creating manifold-projected interpolations...", flush=True)
    projected_interpolations = []
    for i, hidden_state in enumerate(naive_interpolations):
        print(f"  Projecting interpolation point {i+1}/{num_interpolation_steps}...", flush=True)
        projected = project_onto_manifold(
            hidden_state,
            method=method,
            balltree=balltree,
            data_points=data_points,
            atlas_loader=atlas_loader,
            latent_dim=latent_dim,
            device=str(device)
        )
        projected_interpolations.append(projected)
    
    # Decode all interpolations using patchscopes
    print(f"\n[{datetime.now()}] Decoding naive interpolations...", flush=True)
    naive_decodings = []
    for i, hidden_state in enumerate(naive_interpolations):
        print(f"  Decoding naive point {i+1}/{num_interpolation_steps}...", flush=True)
        decoded = decode_hidden_state_with_patchscopes(
            hidden_state,
            patchscopes_wrapper,
            source_layer=source_layer,
            target_layer=target_layer,
            device=device
        )
        naive_decodings.append(decoded)
        print(f"    Decoded: {decoded}")
    
    print(f"\n[{datetime.now()}] Decoding projected interpolations...", flush=True)
    projected_decodings = []
    for i, hidden_state in enumerate(projected_interpolations):
        print(f"  Decoding projected point {i+1}/{num_interpolation_steps}...", flush=True)
        decoded = decode_hidden_state_with_patchscopes(
            hidden_state,
            patchscopes_wrapper,
            source_layer=source_layer,
            target_layer=target_layer,
            device=device
        )
        projected_decodings.append(decoded)
        print(f"    Decoded: {decoded}")
    
    # Print comparison
    print(f"\n{'='*80}")
    print(f"RESULTS COMPARISON")
    print(f"{'='*80}\n")
    print(f"{'Step':<6} {'Alpha':<8} {'Naive Decoding':<30} {'Projected Decoding':<30}")
    print(f"{'-'*80}")
    
    alphas = np.linspace(0, 1, num_interpolation_steps)
    for i, (alpha, naive, projected) in enumerate(zip(alphas, naive_decodings, projected_decodings)):
        print(f"{i:<6} {alpha:<8.3f} {naive:<30} {projected:<30}")
    
    print(f"\n{'='*80}\n")
    
    return {
        "naive_interpolations": naive_interpolations,
        "projected_interpolations": projected_interpolations,
        "naive_decodings": naive_decodings,
        "projected_decodings": projected_decodings,
        "alphas": alphas
    }


def save_results(
    results: Dict,
    output_dir: Path,
    prompt_a: str,
    prompt_b: str,
    method: str
):
    """Save experiment results to disk.
    
    Args:
        results: Results dictionary from run_interpolation_experiment
        output_dir: Directory to save results
        prompt_a: First prompt
        prompt_b: Second prompt
        method: Dimensionality reduction method
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save text results
    results_file = output_dir / f"interpolation_results_{method}_{timestamp}.txt"
    with open(results_file, "w") as f:
        f.write("MANIFOLD INTERPOLATION EXPERIMENT RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Prompt A: {prompt_a}\n")
        f.write(f"Prompt B: {prompt_b}\n")
        f.write(f"Method: {method}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Step':<6} {'Alpha':<8} {'Naive Decoding':<30} {'Projected Decoding':<30}\n")
        f.write("-" * 80 + "\n")
        
        for i, (alpha, naive, projected) in enumerate(zip(
            results["alphas"], 
            results["naive_decodings"], 
            results["projected_decodings"]
        )):
            f.write(f"{i:<6} {alpha:<8.3f} {naive:<30} {projected:<30}\n")
    
    print(f"\n[{datetime.now()}] Results saved to {results_file}")
    
    # Save numerical data
    data_file = output_dir / f"interpolation_data_{method}_{timestamp}.npz"
    
    # Convert tensors to numpy for saving
    naive_interp_np = np.stack([t.cpu().numpy() for t in results["naive_interpolations"]])
    proj_interp_np = np.stack([t.cpu().numpy() for t in results["projected_interpolations"]])
    
    np.savez(
        data_file,
        alphas=results["alphas"],
        naive_interpolations=naive_interp_np,
        projected_interpolations=proj_interp_np,
        naive_decodings=np.array(results["naive_decodings"]),
        projected_decodings=np.array(results["projected_decodings"])
    )
    
    print(f"[{datetime.now()}] Data saved to {data_file}")


def visualize_results(
    results: Dict,
    prompt_a: str,
    prompt_b: str,
    output_dir: Optional[Path] = None
):
    """Create visualization of interpolation results.
    
    Args:
        results: Results dictionary from run_interpolation_experiment
        prompt_a: First prompt
        prompt_b: Second prompt
        output_dir: Optional directory to save plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib not available, skipping visualization")
        return
    
    alphas = results["alphas"]
    naive_decodings = results["naive_decodings"]
    projected_decodings = results["projected_decodings"]
    
    # Compute distances between interpolations
    naive_interp = torch.stack(results["naive_interpolations"])
    proj_interp = torch.stack(results["projected_interpolations"])
    
    # Distance from start point
    start_point = naive_interp[0]
    naive_distances = torch.norm(naive_interp - start_point, dim=1).cpu().numpy()
    proj_distances = torch.norm(proj_interp - start_point, dim=1).cpu().numpy()
    
    # Distance between naive and projected at each step
    diff_distances = torch.norm(naive_interp - proj_interp, dim=1).cpu().numpy()
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Distance from start
    axes[0].plot(alphas, naive_distances, 'o-', label='Naive interpolation', linewidth=2, markersize=6)
    axes[0].plot(alphas, proj_distances, 's-', label='Projected interpolation', linewidth=2, markersize=6)
    axes[0].set_xlabel('Interpolation parameter (α)', fontsize=12)
    axes[0].set_ylabel('Distance from start point', fontsize=12)
    axes[0].set_title('Distance from Start Point vs Interpolation Parameter', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Distance between naive and projected
    axes[1].plot(alphas, diff_distances, 'o-', color='purple', linewidth=2, markersize=6)
    axes[1].set_xlabel('Interpolation parameter (α)', fontsize=12)
    axes[1].set_ylabel('Distance between naive and projected', fontsize=12)
    axes[1].set_title('Deviation: Naive vs Manifold-Projected Interpolation', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = output_dir / f"interpolation_plot_{timestamp}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"\n[{datetime.now()}] Visualization saved to {plot_file}")
    
    plt.show()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Manifold interpolation experiment: compare naive vs manifold-projected interpolation"
    )
    add_config_argument(parser)
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/gemma-2-2b",
        help="Model name or path (e.g., gemma-2-2b)"
    )
    parser.add_argument(
        "--prompt-a",
        type=str,
        default="The capital of France is Paris",
        help="First prompt for interpolation"
    )
    parser.add_argument(
        "--prompt-b",
        type=str,
        default="The capital of Italy is Rome",
        help="Second prompt for interpolation"
    )
    parser.add_argument(
        "--source-layer",
        type=int,
        default=18,
        help="Layer to extract hidden states from (should use config.model.layer_for_activation, otherwise risk throwing an error)"
    )
    parser.add_argument(
        "--target-layer",
        type=int,
        default=1,
        help="Layer to patch into for decoding"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=20,
        help="Number of interpolation steps"
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=50,
        help="Latent dimensionality for projection"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["local_pca", "pca", "autoencoder"],
        default="local_pca",
        help="Projection method (local_pca uses BallTree, pca/autoencoder use atlases)"
    )
    parser.add_argument(
        "--layer-for-balltree",
        type=int,
        default=None,
        help="Layer to use for BallTree (default: uses config.model.layer_for_activation)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on ('cpu' or 'cuda')"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save results to disk"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../results/manifold_interpolation",
        help="Directory to save results"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create and display visualization"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"\n[{datetime.now()}] Loading model and tokenizer: {args.model_name}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        device_map=None
    )
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded successfully. Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load resources based on projection method
    balltree = None
    data_points = None
    atlas_loader = None
    
    if args.method == "local_pca":
        # Load BallTree for local PCA projection
        balltree_layer = args.layer_for_balltree if args.layer_for_balltree is not None else config.model.layer_for_activation
        print(f"\n[{datetime.now()}] Loading BallTree for layer {balltree_layer}...", flush=True)
        balltree, data_points = load_balltree(balltree_layer, config)
    else:
        # Load AtlasLoader for atlas-based methods
        print(f"\n[{datetime.now()}] Initializing AtlasLoader for {args.method} atlas...", flush=True)
        atlas_loader = AtlasLoader(config, device=str(device))
        atlas_loader.load_centroids()
    
    # Run experiment
    results = run_interpolation_experiment(
        model=model,
        tokenizer=tokenizer,
        config=config,
        prompt_a=args.prompt_a,
        prompt_b=args.prompt_b,
        source_layer=args.source_layer,
        target_layer=args.target_layer,
        num_interpolation_steps=args.num_steps,
        method=args.method,
        latent_dim=args.n_components,
        balltree=balltree,
        data_points=data_points,
        atlas_loader=atlas_loader,
        device=device
    )
    
    # Save results if requested
    if args.save_results:
        output_dir = Path(args.output_dir)
        save_results(
            results,
            output_dir,
            prompt_a=args.prompt_a,
            prompt_b=args.prompt_b,
            method=args.method
        )
    
    # Visualize results if requested
    if args.visualize:
        output_dir = Path(args.output_dir) if args.save_results else None
        visualize_results(
            results,
            prompt_a=args.prompt_a,
            prompt_b=args.prompt_b,
            output_dir=output_dir
        )
    
    print(f"\n[{datetime.now()}] Experiment completed successfully!")


if __name__ == "__main__":
    main()
