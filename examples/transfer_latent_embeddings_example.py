import os
import sys
from datetime import datetime

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config_manager import load_config
from utils import common
from transfer_latent_embeddings import (
    VALID_EMBEDDING_TYPES,
    batch_load_embeddings_by_type,
    transfer_embedding_between_centroids,
)


def run_single_transfer_for_type(
    embedding_type: str,
    source_centroid: int,
    target_centroid: int,
    neighbor_indices: np.ndarray,
    config,
) -> None:
    """Run one point + one direction transfer for a given embedding type."""
    print(
        f"[{datetime.now()}] Running transfer for embedding type '{embedding_type}' "
        f"from centroid {source_centroid} to {target_centroid}...",
        flush=True,
    )

    all_embeddings = batch_load_embeddings_by_type(
        config.clustering.n_centroids,
        config.dimensionality.n_components,
        embedding_type=embedding_type,
    )

    embeddings_source = all_embeddings[source_centroid]
    embeddings_target = all_embeddings[target_centroid]

    point_to_transfer = embeddings_source[0]
    direction_to_transfer = np.random.randn(config.dimensionality.n_components).astype(
        embeddings_source.dtype
    )

    transferred_point, transferred_direction = transfer_embedding_between_centroids(
        source_centroid,
        target_centroid,
        point_to_transfer,
        neighbor_indices,
        embeddings_source,
        embeddings_target,
        direction_vector=direction_to_transfer,
        embedding_type=embedding_type,
        config=config,
    )

    point_norm_before = float(np.linalg.norm(point_to_transfer))
    point_norm_after = float(np.linalg.norm(transferred_point))
    direction_norm_before = float(np.linalg.norm(direction_to_transfer))
    direction_norm_after = float(np.linalg.norm(transferred_direction))

    print(
        f"[{datetime.now()}] ✅ {embedding_type}: transfer succeeded | "
        f"point_norm {point_norm_before:.4f} -> {point_norm_after:.4f} | "
        f"direction_norm {direction_norm_before:.4f} -> {direction_norm_after:.4f}",
        flush=True,
    )


def main() -> None:
    """Try transfers for isomap, pca, and autoencoder from centroid 3 to 4."""
    source_centroid = 3
    target_centroid = 4

    config = load_config()

    print(f"[{datetime.now()}] Loading neighbor indices...", flush=True)
    neighbor_indices = common.load_neighbor_indices(
        f"nearest_{config.clustering.k_nearest_large}_neighbors_indices_layer_"
        f"{config.model.layer_for_activation}_n_centroids_{config.clustering.n_centroids}.npy"
    )

    print(
        f"[{datetime.now()}] Starting transfer test for all embedding types: "
        f"{sorted(VALID_EMBEDDING_TYPES)}",
        flush=True,
    )

    for embedding_type in sorted(VALID_EMBEDDING_TYPES):
        try:
            run_single_transfer_for_type(
                embedding_type=embedding_type,
                source_centroid=source_centroid,
                target_centroid=target_centroid,
                neighbor_indices=neighbor_indices,
                config=config,
            )
        except Exception as exc:
            print(
                f"[{datetime.now()}] ❌ {embedding_type}: transfer failed with error: {exc}",
                flush=True,
            )

    print(f"[{datetime.now()}] Finished transfer example run.", flush=True)


if __name__ == "__main__":
    main()
