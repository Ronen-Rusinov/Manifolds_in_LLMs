"""
We'll go by the method proposed by bills et al (2023)

For a feature chosen, we evaluate how strongly it activates in all tokens in a given wikipedia article.
What activation means in our Atlas method is to be defined later.

We then send the article, with the activations, to GPT-5-nano to ask it to identify the pattern of tokens that activate the feature, and explain it.

Next, we give a different article along with the proposed explanation to GPT-5-nano and ask it to guess which tokens activate the feature, and to what extent.

We then compare the activations predicted by GPT-5-nano to the actual activations, and evaluate the interpretability of the feature based on how well GPT-5-nano can predict the activations.



As for the question of what a feature is, it i given by an equivalence class of directions in embedding spaces of different centroids.
We choose one prototype centroid, and we identify the direction of maximal variance in the embedding space of the neighbors of that centroid, which is given by the first principal component of the PCA performed on those embeddings.
Then, we transfer this direction to other centroids by identifying the direction in the embedding space of the neighbors of each other centroid by procrustes analysis as is seen in mapping_alignment.py

Naturally, only centroids with significant overlap of neighborhoods will be considered, as the accuracy of the procrustes analysis will be poor otherwise, producing inaccurate transfers.
To find the strength of a given feature on a given activation, we will find the nearest centroid to it, and if it is either the prototype centroid or one sufficiently overlapping with it, we will project the embedding of the activation on the direction associated with the feature in that centroid, and take the value of the projection as the strength of the feature on that activation.
Otherwise, if the closest centroid is not sufficiently overlapping with the prototype centroid, we will say that the feature is not active on that activation, and assign it a strength of 0.

Some functions, such as identifying the closest centroid to a given activation, and the transfer of directions between centroids are implemented in src/atlas-loader.py and src/transfer_latent_embeddings.py

"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import numpy as np
from sklearn.decomposition import PCA
import json

from src.atlas_loader import AtlasLoader
from src.transfer_latent_embeddings import transfer_embedding_between_centroids
from config_manager import load_config
from src.utils import common


# ============================================================================
# FEATURE DIRECTION EXTRACTION
# ============================================================================

def compute_feature_direction_at_centroid(
    atlas_loader: AtlasLoader,
    centroid_idx: int,
    n_components: int,
    embedding_type: str = "pca",
    n_pca_components: int = 5,
    pc_index: int = 0
) -> np.ndarray:
    """Compute a specific feature direction (principal component) at a prototype centroid.
    
    Given a centroid and its neighborhood of embedded activations, compute the direction
    of maximal variance using PCA. This direction serves as the feature direction for 
    this centroid's neighborhood.
    
    Args:
        atlas_loader: AtlasLoader instance for accessing models
        centroid_idx: Index of the prototype centroid
        n_components: Dimensionality of the embeddings
        embedding_type: Type of embeddings ("pca", "isomap", "autoencoder")
        n_pca_components: Number of principal components to compute (default 5)
        pc_index: Which PC to return (0=PC1, 1=PC2, ..., 4=PC5)
    
    Returns:
        Feature direction vector of shape (n_components,) representing the direction
        of maximal variance in the embedding space of the centroid's neighborhood
    
    Raises:
        FileNotFoundError: If embeddings cannot be loaded for the centroid
        ValueError: If embeddings have insufficient variance
    """
    print(
        f"[{datetime.now()}] Computing feature direction at centroid {centroid_idx} "
        f"({embedding_type}, {n_components}D)...",
        flush=True
    )
    
    # Load the embeddings for the neighborhood of the prototype centroid
    if embedding_type.lower() == "pca":
        embeddings = common.load_pca_embeddings(centroid_idx, n_components)
    elif embedding_type.lower() == "isomap":
        embeddings = common.load_isomap_embeddings(centroid_idx, n_components)
    elif embedding_type.lower() == "autoencoder":
        embeddings = common.load_autoencoder_embeddings(centroid_idx, n_components)
    else:
        raise ValueError(f"Unknown embedding_type: {embedding_type}")
    
    if embeddings is None or len(embeddings) == 0:
        raise FileNotFoundError(f"No embeddings found for centroid {centroid_idx}")
    
    print(
        f"[{datetime.now()}] Loaded {len(embeddings)} embeddings for centroid {centroid_idx}",
        flush=True
    )
    
    # Compute PCA on the embeddings to find the direction of maximal variance
    pca = PCA(n_components=n_pca_components)
    pca.fit(embeddings)
    
    # Extract the requested principal component as the feature direction
    if pc_index >= len(pca.components_):
        raise ValueError(f"Requested PC{pc_index+1} but only {len(pca.components_)} components available")
    
    feature_direction = pca.components_[pc_index]  # Shape: (n_components,)
    
    variance_explained = pca.explained_variance_ratio_[pc_index]
    print(
        f"[{datetime.now()}] Feature direction PC{pc_index+1} computed. Variance explained: {variance_explained:.4f}",
        flush=True
    )
    
    return feature_direction


def transfer_feature_direction(
    atlas_loader: AtlasLoader,
    prototype_centroid_idx: int,
    target_centroid_idx: int,
    feature_direction: np.ndarray,
    neighbor_indices: np.ndarray,
    n_components: int,
    embedding_type: str = "pca"
) -> Tuple[np.ndarray, bool]:
    """Transfer a feature direction from one centroid to another using Procrustes analysis.
    
    Given a feature direction (e.g., first principal component) at a prototype centroid,
    transfer it to another centroid's embedding space using Procrustes analysis on the
    common neighbor points between the two centroids.
    
    Args:
        atlas_loader: AtlasLoader instance
        prototype_centroid_idx: Index of the centroid where the direction originated
        target_centroid_idx: Index of the target centroid for transfer
        feature_direction: Feature direction vector to transfer (shape: n_components,)
        neighbor_indices: Array of neighbor indices for each centroid (shape: n_centroids, k)
        n_components: Dimensionality of embeddings
        embedding_type: Type of embeddings to use
    
    Returns:
        Tuple of (transferred_direction, success_flag) where:
        - transferred_direction: The direction in the target centroid's space
        - success_flag: Boolean indicating if transfer was successful (sufficient overlap)
    """
    print(
        f"[{datetime.now()}] Transferring feature direction from centroid {prototype_centroid_idx} "
        f"to centroid {target_centroid_idx}...",
        flush=True
    )
    
    try:
        # Load embeddings for both centroids
        if embedding_type.lower() == "pca":
            embeddings_source = common.load_pca_embeddings(prototype_centroid_idx, n_components)
            embeddings_target = common.load_pca_embeddings(target_centroid_idx, n_components)
        elif embedding_type.lower() == "isomap":
            embeddings_source = common.load_isomap_embeddings(prototype_centroid_idx, n_components)
            embeddings_target = common.load_isomap_embeddings(target_centroid_idx, n_components)
        elif embedding_type.lower() == "autoencoder":
            embeddings_source = common.load_autoencoder_embeddings(prototype_centroid_idx, n_components)
            embeddings_target = common.load_autoencoder_embeddings(target_centroid_idx, n_components)
        else:
            raise ValueError(f"Unknown embedding_type: {embedding_type}")
        
        # Transfer the direction using procrustes analysis
        _, transferred_direction = transfer_embedding_between_centroids(
            centroid_i=prototype_centroid_idx,
            centroid_j=target_centroid_idx,
            embedding_point=feature_direction,  # The direction itself
            neighbor_indices=neighbor_indices,
            embeddings_i=embeddings_source,
            embeddings_j=embeddings_target,
            direction_vector=feature_direction,
            embedding_type=embedding_type
        )
        
        print(
            f"[{datetime.now()}] Successfully transferred feature direction "
            f"from centroid {prototype_centroid_idx} to {target_centroid_idx}",
            flush=True
        )
        
        return transferred_direction, True
        
    except ValueError as e:
        # Insufficient overlap or other transfer error
        print(
            f"[{datetime.now()}] Failed to transfer direction from centroid {prototype_centroid_idx} "
            f"to {target_centroid_idx}: {str(e)}",
            flush=True
        )
        return feature_direction, False


# ============================================================================
# FEATURE ACTIVATION STRENGTH COMPUTATION
# ============================================================================

def compute_activation_strength(
    activation_embedding: np.ndarray,
    centroid_idx: int,
    feature_direction: np.ndarray,
    prototype_centroid_idx: int,
    transferred_directions: Dict[int, Tuple[np.ndarray, bool]],
    overlap_threshold: float = 0.3
) -> float:
    """Compute the strength of a feature's activation on a given token.
    
    Given a token's embedding and its nearest centroid, compute the projection of the
    embedding onto the feature direction. If the nearest centroid is the prototype or
    has sufficient overlap with it, return the projection strength. Otherwise, return 0.
    
    Args:
        activation_embedding: The embedding vector of the token (shape: n_components,)
        centroid_idx: Index of the nearest centroid to this activation
        feature_direction: The feature direction at this centroid (shape: n_components,)
        prototype_centroid_idx: Index of the prototype centroid
        transferred_directions: Dict mapping centroid_idx -> (transferred_direction, success_flag)
        overlap_threshold: Minimum overlap fraction required for reliable feature measurement
    
    Returns:
        Scalar float value representing the strength of feature activation
    """
    # If the nearest centroid is the prototype itself, use the original direction
    if centroid_idx == prototype_centroid_idx:
        # Project the embedding onto the feature direction
        strength = np.dot(activation_embedding, feature_direction)
        return strength
    
    # Check if we have a successfully transferred direction for this centroid
    if centroid_idx in transferred_directions:
        transferred_dir, success = transferred_directions[centroid_idx]
        
        if success:
            # Use the transferred direction for projection
            strength = np.dot(activation_embedding, transferred_dir)
            return strength
    
    # If no successful transfer or insufficient overlap, return 0 (feature not active)
    return 0.0


def compute_all_activation_strengths(
    article_tokens: List[str],
    article_activations: np.ndarray,
    article_global_indices: np.ndarray,
    centroid_indices: np.ndarray,
    neighbor_indices: np.ndarray,
    prototype_centroid_idx: int,
    feature_direction: np.ndarray,
    transferred_directions: Dict[int, Tuple[np.ndarray, bool]],
    atlas_loader: AtlasLoader,
    n_components: int,
    embedding_type: str = "pca"
) -> np.ndarray:
    """Compute feature activation strengths for all tokens in an article.
    
    Note: Per-centroid embeddings are pre-computed. For each activation, we look up
    its pre-computed embedding in the local embedding space of its nearest centroid.
    The key insight: neighbor_indices[centroid][i] gives the global index a_i, and
    embeddings[centroid][i] = E(A_{a_i}) where E is the centroid's embedding function.
    
    Args:
        article_tokens: List of token strings in the article
        article_activations: Array of raw activations in ambient space (shape: n_tokens, activation_dim)
        article_global_indices: Global indices of activations (shape: n_tokens,)
        centroid_indices: Array of nearest centroid indices for each token
        neighbor_indices: Array of neighbor indices for each centroid (shape: n_centroids, k)
        prototype_centroid_idx: Index of the prototype centroid
        feature_direction: Feature direction at the prototype centroid
        transferred_directions: Dict of transferred directions for other centroids
        atlas_loader: AtlasLoader instance for loading pre-computed embeddings
        n_components: Dimensionality of the embedding space
        embedding_type: Type of embedding to use ("pca", "isomap", "autoencoder")
    
    Returns:
        Array of activation strengths (shape: n_tokens,) where each value is the
        strength of feature activation for the corresponding token
    """
    n_tokens = len(article_tokens)
    activation_strengths = np.zeros(n_tokens)
    
    print(f"[{datetime.now()}] Computing activation strengths for {n_tokens} tokens...", flush=True)
    
    # Cache for loaded embeddings to avoid repeated disk access
    embedding_cache = {}
    
    for i in range(n_tokens):
        nearest_centroid = int(centroid_indices[i])
        global_activation_idx = int(article_global_indices[i])
        
        # Get pre-computed embeddings for this centroid if not already cached
        if nearest_centroid not in embedding_cache:
            try:
                if embedding_type.lower() == "pca":
                    embeddings_file = f"centroid_{nearest_centroid:04d}_embeddings_{n_components}D.npy"
                    embeddings = common.load_pca_embeddings(nearest_centroid, n_components)
                elif embedding_type.lower() == "isomap":
                    embeddings = common.load_isomap_embeddings(nearest_centroid, n_components)
                elif embedding_type.lower() == "autoencoder":
                    embeddings = common.load_autoencoder_embeddings(nearest_centroid, n_components)
                else:
                    raise ValueError(f"Unknown embedding_type: {embedding_type}")
                
                embedding_cache[nearest_centroid] = embeddings
            except Exception as e:
                print(f"[{datetime.now()}] Warning: Could not load embeddings for centroid {nearest_centroid}: {str(e)}", flush=True)
                embedding_cache[nearest_centroid] = None
        
        embeddings = embedding_cache[nearest_centroid]
        
        # Find the position of this activation in the centroid's neighbor list
        if embeddings is not None and neighbor_indices is not None:
            neighbor_list = neighbor_indices[nearest_centroid]
            
            # Try to find this global index in the neighbor list
            try:
                # Find where global_activation_idx appears in neighbor_list
                position = np.where(neighbor_list == global_activation_idx)[0]
                
                if len(position) > 0:
                    # Found it! Use the pre-computed embedding at this position
                    local_embedding = embeddings[position[0]]
                else:
                    # Activation not in this centroid's neighborhood, use fallback
                    print(f"[{datetime.now()}] Warning: Activation {global_activation_idx} not in centroid {nearest_centroid}'s neighborhood", flush=True)
                    # Fallback: transform the ambient activation to the local embedding space using the atlas_loader
                    try:
                        raw_activation = article_activations[i].reshape(1, -1)
                        embedded = atlas_loader.embed(
                            raw_activation,
                            nearest_centroid,
                            method=embedding_type,
                            n_components=n_components
                        )
                        local_embedding = embedded[0]
                    except Exception as e:
                        print(f"[{datetime.now()}] Error in fallback embedding: {str(e)}", flush=True)
                        local_embedding = np.zeros(n_components)

            except Exception as e:
                print(f"[{datetime.now()}] Error looking up embedding: {str(e)}", flush=True)
                local_embedding = np.zeros(n_components)
        else:
            # Could not load embeddings, use fallback
            local_embedding = np.zeros(n_components)
        
        # Compute activation strength using the local embedding
        strength = compute_activation_strength(
            local_embedding,
            nearest_centroid,
            feature_direction,
            prototype_centroid_idx,
            transferred_directions
        )
        
        activation_strengths[i] = strength
    
    print(f"[{datetime.now()}] Activation strengths computed", flush=True)
    
    return activation_strengths


# ============================================================================
# LLM-BASED INTERPRETATION
# ============================================================================

def call_llm_for_explanation(
    article_text: str,
    article_tokens: List[str],
    activation_strengths: np.ndarray,
    feature_id: str,
    llm_client: Any,
    model: str = "gpt-5-nano"
) -> str:
    """Call LLM to generate explanation of feature pattern based on activations.
    
    Sends an article with token-level activation strengths to an LLM (e.g., GPT-4)
    and asks it to identify the semantic pattern or feature explanation for why
    certain tokens activate strongly.
    
    Args:
        article_text: The full text of the article
        article_tokens: List of individual tokens
        activation_strengths: Array of feature activation strengths for each token
        feature_id: Identifier for the feature (e.g., "feature_42")
        llm_client: OpenAI client or similar LLM interface
        model: Name of the LLM model to use (default: "gpt-5-nano")
    
    Returns:
        String containing the LLM's explanation of the feature pattern
    """
    print(
        f"[{datetime.now()}] Calling LLM to explain feature {feature_id}...",
        flush=True
    )
    
    # Build the prompt with token activations
    # Annotate the article text with activation strengths
    annotated_text = _annotate_text_with_activations(
        article_tokens, activation_strengths
    )
    
    prompt = f"""We are studying feature representations in an LLM. Each feature looks for one particular thing in the input text.
    Look at the parts of the text the feature activates most strongly on, and summarise IN A SINGLE SENTENCE what that feature is looking for. DON'T list examples of words.
    Format is token[activation_strength]. A feature finding what it's looking for is represented by a non-zero activation value. The higher the activation value, the stronger the match.

    Article with activation strengths. (in brackets):
    {annotated_text}

    """
    response = llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert at analyzing neural network features and identifying their semantic patterns."},
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=5000
    )
    
    explanation = response.choices[0].message.content
    print(f"[{datetime.now()}] LLM explanation received", flush=True)
    
    return explanation


def call_llm_for_activation_prediction(
    article_text: str,
    article_tokens: List[str],
    feature_explanation: str,
    feature_id: str,
    llm_client: Any,
    model: str = "gpt-5-nano"
) -> Tuple[np.ndarray, str]:
    """Call LLM to predict feature activations based on explanation.
    
    Given a different article and a feature explanation, ask the LLM to predict
    which tokens should activate the feature and to what degree.
    
    Args:
        article_text: Text of a different article (not used for explanation generation)
        article_tokens: List of tokens in this article
        feature_explanation: The explanation from the initial LLM analysis
        feature_id: Identifier for the feature
        llm_client: OpenAI client or similar interface
        model: Name of the LLM model to use
    
    Returns:
        Tuple of (predicted_strengths, reasoning) where:
        - predicted_strengths: Array of shape (n_tokens,) with LLM's predicted activation values
        - reasoning: String explaining the LLM's predictions
    """
    print(
        f"[{datetime.now()}] Calling LLM to predict activations for feature {feature_id}...",
        flush=True
    )
    
    # Format tokens with indices for the prompt
    tokens_list = '\n'.join([f"{i}\t{token}" for i, token in enumerate(article_tokens)])
    
    prompt = f"""We are studying feature representations in an LLM. Each feature looks for one particular thing in the input text.
Look at an explanation of what the feature does, and try to predict its activations on tokens in a new article.

Feature Explanation:
{feature_explanation}

Now predict activations for these tokens. For each token, provide a predicted activation strength between 100 and -100, with mean 0 and standard deviation of around 20. 

Tokens (with index):
{tokens_list}

Format your response EXACTLY as follows, one line per token with index, tab, token, tab, and activation value. FOr example:
0\ttoken0\t45
1\ttoken1\t-1
2\ttoken2\t81
...

Provide ALL {len(article_tokens)} predictions."""
    
    response = llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert at interpreting neural network features. Provide predictions in the exact tab-separated format requested."},
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=5000
    )
    
    response_text = response.choices[0].message.content
    # Check if response is empty
    if not response_text or not response_text.strip():
        print(f"[{datetime.now()}] WARNING: LLM returned empty response!", flush=True)
        predicted_strengths = np.zeros(len(article_tokens))
        reasoning = "LLM returned empty response"
        return predicted_strengths, reasoning
    
    
    # Parse the tab-separated response
    predicted_strengths = np.zeros(len(article_tokens))
    reasoning = ""
    
    try:
        lines = response_text.strip().split('\n')
        parsed_count = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) >= 3:
                try:
                    idx = int(parts[0])
                    # token = parts[1]  # We can validate this if needed
                    activation = float(parts[2])
                    
                    if 0 <= idx < len(article_tokens):
                        predicted_strengths[idx] = activation
                        parsed_count += 1
                except (ValueError, IndexError) as e:
                    continue
        
        print(f"[{datetime.now()}] Successfully parsed {parsed_count}/{len(article_tokens)} predictions", flush=True)
        
        if parsed_count == 0:
            print(f"[{datetime.now()}] WARNING: Could not parse any predictions from LLM response!", flush=True)
            print(f"[{datetime.now()}] Response preview: {response_text[:1000]}", flush=True)
            reasoning = f"No predictions parsed. Raw response: {response_text[:500]}"
        elif parsed_count < len(article_tokens) * 0.5:
            print(f"[{datetime.now()}] Warning: Only parsed {parsed_count} predictions, less than 50% of tokens", flush=True)
            reasoning = f"Incomplete response. Parsed {parsed_count}/{len(article_tokens)} predictions. Raw: {response_text[:500]}"
        
    except Exception as e:
        print(
            f"[{datetime.now()}] Warning: Could not parse LLM response: {str(e)}",
            flush=True
        )
        print(f"[{datetime.now()}] Response preview: {response_text[:500]}", flush=True)
        reasoning = f"Parse error: {str(e)}. Raw: {response_text[:500]}"
    
    print(f"[{datetime.now()}] LLM predictions received", flush=True)
    
    return predicted_strengths, reasoning


def _annotate_text_with_activations(
    tokens: List[str],
    strengths: np.ndarray
) -> str:
    """Create an annotated text representation with activation strengths.
    
    Args:
        tokens: List of token strings
        strengths: Array of activation strengths for each token
    
    Returns:
        Formatted string with tokens and their activation strengths
    """
    annotated_parts = []
    for token, strength in zip(tokens, strengths):
        # Format the strength with 2 decimal places
        strength_str = f"{strength:+.2f}"  # e.g., "+0.45", "-0.12"
        annotated_parts.append(f"{token}[{strength_str}]")
    
    return " ".join(annotated_parts)


# ============================================================================
# INTERPRETABILITY EVALUATION
# ============================================================================

def evaluate_interpretability(
    actual_strengths: np.ndarray,
    predicted_strengths: np.ndarray
) -> Dict[str, float]:
    """Evaluate feature interpretability by comparing actual vs predicted activations.
    
    Compute multiple metrics to assess how well the LLM explanation generalizes:
    - Correlation: Pearson correlation between actual and predicted
    - MSE: Mean squared error of predictions
    - MAE: Mean absolute error
    - RMSE: Root mean squared error
    - Spearman correlation: Rank correlation (more robust to outliers)
    
    Args:
        actual_strengths: Array of actual feature activation strengths
        predicted_strengths: Array of LLM-predicted activation strengths
    
    Returns:
        Dictionary with evaluation metrics:
        {
            "pearson_correlation": float,
            "spearman_correlation": float,
            "mean_squared_error": float,
            "mean_absolute_error": float,
            "root_mean_squared_error": float,
            "perfect_direction_accuracy": float  # % of correctly predicted high/low activations
        }
    """
    from scipy.stats import pearsonr, spearmanr
    
    print(f"[{datetime.now()}] Evaluating interpretability...", flush=True)
    
    # Ensure arrays are the same length (truncate if necessary)
    min_len = min(len(actual_strengths), len(predicted_strengths))
    actual = actual_strengths[:min_len]
    predicted = predicted_strengths[:min_len]
    
    # Compute Pearson correlation (linear relationship)
    pearson_r, pearson_p = pearsonr(actual, predicted)
    
    # Compute Spearman correlation (rank-based, more robust)
    spearman_r, spearman_p = spearmanr(actual, predicted)
    
    # Compute error metrics
    mse = np.mean((actual - predicted) ** 2)
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(mse)
    
    # Compute directional accuracy (did the LLM correctly predict high vs low?)
    actual_binary = (actual > np.median(actual)).astype(int)
    predicted_binary = (predicted > np.median(predicted)).astype(int)
    direction_accuracy = np.mean(actual_binary == predicted_binary)
    
    metrics = {
        "pearson_correlation": float(pearson_r),
        "pearson_pvalue": float(pearson_p),
        "spearman_correlation": float(spearman_r),
        "spearman_pvalue": float(spearman_p),
        "mean_squared_error": float(mse),
        "mean_absolute_error": float(mae),
        "root_mean_squared_error": float(rmse),
        "direction_accuracy": float(direction_accuracy)
    }
    
    print(
        f"[{datetime.now()}] Interpretability evaluation complete:\n"
        f"  Pearson r: {pearson_r:.4f}\n"
        f"  Spearman r: {spearman_r:.4f}\n"
        f"  RMSE: {rmse:.4f}\n"
        f"  Direction accuracy: {direction_accuracy:.2%}",
        flush=True
    )
    
    return metrics


def save_predicted_vs_ground_truth_plot(
    actual_strengths: np.ndarray,
    predicted_strengths: np.ndarray,
    output_path: str,
    title: str = "Predicted vs Ground Truth Activations"
) -> None:
    """Save a scatter plot comparing predicted and ground-truth activations."""
    import matplotlib.pyplot as plt

    min_len = min(len(actual_strengths), len(predicted_strengths))
    actual = actual_strengths[:min_len]
    predicted = predicted_strengths[:min_len]

    plt.figure(figsize=(8, 8))
    plt.scatter(actual, predicted, alpha=0.4, s=10)

    # y=x reference line
    min_val = min(float(np.min(actual)), float(np.min(predicted)))
    max_val = max(float(np.max(actual)), float(np.max(predicted)))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1)

    plt.xlabel("Ground Truth Activation")
    plt.ylabel("Predicted Activation")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to evaluate interpretability on first two articles of the dataset."""
    import argparse
    import pandas as pd
    from pathlib import Path
    
    parser = argparse.ArgumentParser(
        description="Evaluate feature interpretability using Bills et al. (2023) method"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=18,
        help="Layer to use for activations (6 or 18)"
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=50,
        help="Number of components for dimensionality reduction"
    )
    parser.add_argument(
        "--embedding-type",
        type=str,
        choices=["pca", "isomap", "autoencoder"],
        default="pca",
        help="Type of embedding to use"
    )
    parser.add_argument(
        "--prototype-centroid",
        type=int,
        default=0,
        help="Index of prototype centroid for feature. If 0 (default), auto-selects centroid with most activations"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4.1-nano",
        help="LLM model to use (e.g., gpt-4, gpt-3.5-turbo)"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Whether to call actual LLM (requires OPENAI_API_KEY)"
    )
    args = parser.parse_args()
    
    print(f"[{datetime.now()}] Starting feature interpretability evaluation", flush=True)
    print(f"[{datetime.now()}] Layer: {args.layer}, Components: {args.n_components}, " 
          f"Type: {args.embedding_type}", flush=True)
    
    # Load the activation data
    print(f"[{datetime.now()}] Loading activation data from parquet...", flush=True)
    data_dir = Path(__file__).parent.parent / "data" / "activations_data"
    parquet_file = data_dir / "activations_part_01_of_10.parquet"
    
    if not parquet_file.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_file}")
    
    df = pd.read_parquet(parquet_file)
    print(f"[{datetime.now()}] Loaded {len(df)} total tokens from parquet", flush=True)
    
    # Get unique articles
    unique_articles = df["article_id"].unique()
    print(f"[{datetime.now()}] Found {len(unique_articles)} unique articles", flush=True)
    
    if len(unique_articles) < 1:
        raise ValueError("Need at least 1 article in the dataset")
    
    # Initialize AtlasLoader once (needed for all articles)
    print(f"[{datetime.now()}] Initializing AtlasLoader...", flush=True)
    config = load_config()
    atlas_loader = AtlasLoader(config)
    centroids = atlas_loader.load_centroids()
    
    # Keep original row indices for neighbor lookup, then split by position
    activation_col = f"activation_layer_{args.layer}"
    
    # Iterate over all articles
    all_articles_results = {}
    
    for article_idx, article_id in enumerate(unique_articles):
        print(f"\n{'#'*80}", flush=True)
        print(f"[{datetime.now()}] PROCESSING ARTICLE {article_idx+1}/{len(unique_articles)}: {article_id}", flush=True)
        print(f"{'#'*80}\n", flush=True)
        
        # Load article data
        article_data = df[df["article_id"] == article_id]
        article_global_indices = article_data.index.values
        article_data = article_data.reset_index(drop=True)
        article_title = article_data["article_title"].iloc[0]
        n_tokens = len(article_data)
        
        print(f"[{datetime.now()}] Article: {article_id}", flush=True)
        print(f"[{datetime.now()}] Title: '{article_title}'", flush=True)
        print(f"[{datetime.now()}] Total tokens: {n_tokens}", flush=True)
        
        # Ensure article has at least 2 tokens
        if n_tokens < 2:
            print(f"[{datetime.now()}] Skipping article {article_id} (too few tokens)", flush=True)
            continue
        
        # Split into two halves for reference and prediction
        split_idx = n_tokens // 2
        ref_data = article_data.iloc[:split_idx].reset_index(drop=True)
        pred_data = article_data.iloc[split_idx:].reset_index(drop=True)
        
        ref_global_indices = article_global_indices[:split_idx]
        pred_global_indices = article_global_indices[split_idx:]
        
        # Extract tokens and activations
        ref_tokens = ref_data["token_string"].tolist()
        pred_tokens = pred_data["token_string"].tolist()
        
        ref_activations = np.array([np.array(x) for x in ref_data[activation_col].values])
        pred_activations = np.array([np.array(x) for x in pred_data[activation_col].values])
        
        print(f"[{datetime.now()}] Reference half (1st): {len(ref_tokens)} tokens", flush=True)
        print(f"[{datetime.now()}] Prediction half (2nd): {len(pred_tokens)} tokens", flush=True)
        
        # Find nearest centroids
        print(f"[{datetime.now()}] Finding nearest centroids...", flush=True)
        ref_centroid_indices = np.zeros(len(ref_tokens), dtype=int)
        for i, activation in enumerate(ref_activations):
            distances, indices = atlas_loader.get_nearest_centroids(activation, k=1)
            ref_centroid_indices[i] = indices[0]
        
        pred_centroid_indices = np.zeros(len(pred_tokens), dtype=int)
        for i, activation in enumerate(pred_activations):
            distances, indices = atlas_loader.get_nearest_centroids(activation, k=1)
            pred_centroid_indices[i] = indices[0]
        
        # Determine prototype centroid for this article based on reference half
        unique_ref_centroids, counts = np.unique(ref_centroid_indices, return_counts=True)
        top_centroid_idx = unique_ref_centroids[np.argmax(counts)]
        max_count = np.max(counts)
        
        if args.prototype_centroid == 0 and top_centroid_idx != 0:
            prototype_centroid_idx = int(top_centroid_idx)
        else:
            prototype_centroid_idx = args.prototype_centroid
        
        print(f"[{datetime.now()}] Using prototype centroid: {prototype_centroid_idx} ({max_count} activations in reference half)", flush=True)
        
        # Create results directory for this centroid and embedding type
        centroid_results_dir = Path(__file__).parent.parent / "results" / "interpretability" / f"{args.embedding_type}_centroid_{prototype_centroid_idx}"
        centroid_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create article-specific results directory
        article_results_dir = centroid_results_dir / f"article_{article_id}"
        article_results_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{datetime.now()}] Saving results to: {article_results_dir}", flush=True)
        
        # Set article identifiers for reference and prediction halves
        ref_article_id = article_id
        pred_article_id = article_id
        ref_title = f"{article_title} [first_half]"
        pred_title = f"{article_title} [second_half]"
        
        # Store results for this article
        article_all_results = []
        
        for pc_idx in range(5):
            print(f"\n{'='*80}", flush=True)
            print(f"[{datetime.now()}] ===== PROCESSING PRINCIPAL COMPONENT {pc_idx+1}/5 =====", flush=True)
            print(f"{'='*80}\n", flush=True)
        
            # Compute feature direction at prototype centroid
            try:
                feature_direction = compute_feature_direction_at_centroid(
                    atlas_loader,
                    prototype_centroid_idx,
                    args.n_components,
                    embedding_type=args.embedding_type,
                    n_pca_components=5,
                    pc_index=pc_idx
                )
            except Exception as e:
                print(f"[{datetime.now()}] Warning: Could not compute feature direction for PC{pc_idx+1}: {str(e)}", flush=True)
                # Use a random direction as fallback
                feature_direction = np.random.randn(args.n_components)
                feature_direction /= np.linalg.norm(feature_direction)
                continue  # Skip this PC if it fails
        
            # Get unique centroids in both articles and transfer directions
            unique_centroids = np.unique(
                np.concatenate([ref_centroid_indices, pred_centroid_indices])
            )
    
            # Load neighbor indices (needed for procrustes transfer)
            # Try to load from existing data
            try:
                # Construct the neighbor indices filename based on config
                neighbor_indices_file = f"nearest_10000_neighbors_indices_layer_{args.layer}_n_centroids_{config.clustering.n_centroids}.npy"
                neighbor_indices = common.load_neighbor_indices(neighbor_indices_file)
                print(f"[{datetime.now()}] Loaded neighbor indices", flush=True)
            except Exception as e:
                print(f"[{datetime.now()}] Could not load neighbor indices: {str(e)}", flush=True)
                neighbor_indices = None
    
            # Transfer feature direction to other centroids
            transferred_directions = {prototype_centroid_idx: (feature_direction, True)}
        
            if neighbor_indices is not None:
                for centroid_idx in unique_centroids:
                    if centroid_idx != prototype_centroid_idx and centroid_idx not in transferred_directions:
                        transferred_dir, success = transfer_feature_direction(
                            atlas_loader,
                            prototype_centroid_idx,
                            int(centroid_idx),
                            feature_direction,
                            neighbor_indices,
                            args.n_components,
                            embedding_type=args.embedding_type
                        )
                        transferred_directions[int(centroid_idx)] = (transferred_dir, success)
    
            # Compute activation strengths for reference article
            print(f"[{datetime.now()}] Computing activation strengths for reference article (PC{pc_idx+1})...", flush=True)
            ref_strengths = compute_all_activation_strengths(
                ref_tokens,
                ref_activations,
                ref_global_indices,
                ref_centroid_indices,
                neighbor_indices,
                prototype_centroid_idx,
                feature_direction,
                transferred_directions,
                atlas_loader,
                args.n_components,
                args.embedding_type
            )
    
            # Compute activation strengths for prediction article
            print(f"[{datetime.now()}] Computing activation strengths for prediction article (PC{pc_idx+1})...", flush=True)
            pred_strengths = compute_all_activation_strengths(
                pred_tokens,
                pred_activations,
                pred_global_indices,
                pred_centroid_indices,
                neighbor_indices,
                prototype_centroid_idx,
                feature_direction,
                transferred_directions,
                atlas_loader,
                args.n_components,
                args.embedding_type
            )
    
            # Generate LLM explanation if requested
            feature_explanation = None
            if args.use_llm:
                #If args.use_llm is true, we terminate with error if the request fails
                from openai import OpenAI
                llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
                # Generate explanation from reference article
                ref_text = " ".join(ref_tokens)
                feature_explanation = call_llm_for_explanation(
                    ref_text,
                    ref_tokens,
                    ref_strengths,
                    f"feature_{prototype_centroid_idx}_PC{pc_idx+1}",
                    llm_client,
                    model=args.llm_model
                )
            
                print(f"[{datetime.now()}] Feature explanation (PC{pc_idx+1}):", flush=True)
                print(feature_explanation, flush=True)
        


            # Generate predictions using explanation if available
            pred_strengths_llm = None
            if feature_explanation is not None and args.use_llm:
                try:
                    pred_text = " ".join(pred_tokens)
                    pred_strengths_llm, reasoning = call_llm_for_activation_prediction(
                        pred_text,
                        pred_tokens,
                        feature_explanation,
                        f"feature_{prototype_centroid_idx}_PC{pc_idx+1}",
                        llm_client,
                        model=args.llm_model
                    )
                
                    print(f"[{datetime.now()}] LLM Reasoning (PC{pc_idx+1}):", flush=True)
                    print(reasoning, flush=True)
                
                except Exception as e:
                    print(f"[{datetime.now()}] Error generating predictions for PC{pc_idx+1}: {str(e)}", flush=True)
    
            # Evaluate interpretability if we have LLM predictions
            metrics = None
            prediction_plot_path = None
            if pred_strengths_llm is not None:
                # Use actual activations from prediction article as ground truth
                metrics = evaluate_interpretability(
                    pred_strengths,
                    pred_strengths_llm
                )

                # Save plot: predicted vs ground truth activations
                prediction_plot_path = article_results_dir / f"predicted_vs_ground_truth_activations_PC{pc_idx+1}.png"
                save_predicted_vs_ground_truth_plot(
                    pred_strengths,
                    pred_strengths_llm,
                    str(prediction_plot_path),
                    title=f"Predicted vs Ground Truth Activations (PC{pc_idx+1})"
                )
                print(f"[{datetime.now()}] Activation comparison plot saved to {prediction_plot_path}", flush=True)
            
                print(f"[{datetime.now()}] === INTERPRETABILITY METRICS (PC{pc_idx+1}) ===", flush=True)
                for key, value in metrics.items():
                    print(f"  {key}: {value}", flush=True)
            else:
                # Use reference and prediction strengths as comparison
                print(f"[{datetime.now()}] No LLM predictions available for PC{pc_idx+1}, comparing article strengths", flush=True)
                metrics = evaluate_interpretability(
                    ref_strengths,
                    pred_strengths
                )
            
                print(f"[{datetime.now()}] === ACTIVATION STRENGTH COMPARISON (PC{pc_idx+1}) ===", flush=True)
                for key, value in metrics.items():
                    print(f"  {key}: {value}", flush=True)
    
            # Save results for this PC
            results = {
                "pc_index": pc_idx + 1,
                "reference_article": {
                    "id": str(ref_article_id),
                    "title": ref_title,
                    "n_tokens": len(ref_tokens),
                    "activation_strengths": ref_strengths.tolist()
                },
                "prediction_article": {
                    "id": str(pred_article_id),
                    "title": pred_title,
                    "n_tokens": len(pred_tokens),
                    "activation_strengths": pred_strengths.tolist()
                },
                "feature_info": {
                    "prototype_centroid": prototype_centroid_idx,
                    "embedding_type": args.embedding_type,
                    "n_components": args.n_components,
                    "layer": args.layer
                },
                "metrics": metrics,
                "feature_explanation": feature_explanation,
                "prediction_vs_ground_truth_plot": str(prediction_plot_path) if prediction_plot_path is not None else None
            }
        
            # Save individual PC results to JSON
            output_path = article_results_dir / f"PC_{pc_idx+1}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
        
            print(f"[{datetime.now()}] Results for PC{pc_idx+1} saved to {output_path}", flush=True)
        
            # Append to article results
            article_all_results.append(results)
    
        # Save summary of all PCs for this article
        summary_path = article_results_dir / "all_PCs_summary.json"
        with open(summary_path, "w") as f:
            json.dump(article_all_results, f, indent=2)
        
        print(f"[{datetime.now()}] Summary of all PCs for article {article_id} saved to {summary_path}", flush=True)
        print(f"[{datetime.now()}] Completed processing {len(article_all_results)} PCs for article {article_id}", flush=True)
    
    print(f"\n[{datetime.now()}] Feature interpretability evaluation complete for all {len(unique_articles)} articles", flush=True)
    
    return all_articles_results


if __name__ == "__main__":
    main()
