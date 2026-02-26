import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from sklearn.manifold import Isomap, trustworthiness
import numpy as np
import joblib
import datetime
from pathlib import Path
from config_manager import load_config, add_config_argument
import argparse

def train_and_save_isomap(dataframes, save_path, config):
    """
    Train and save Isomap model with parameters from config.
    
    Args:
        dataframes: Tuple of (train_data, val_data, test_data)
        save_path: Path to save isomap model and embeddings
        config: Config object with isomap parameters
    """
    n_neighbors = config.dimensionality.n_neighbors
    n_components = config.dimensionality.n_components
    
    train_data, val_data, test_data = dataframes
    #Use only specified fraction of train data. More than that is infeasable
    train_data = train_data.sample(frac=config.data.train_fraction, random_state=config.training.random_seed)

    layer = config.model.layer_for_activation
    column_name = f'activation_layer_{layer}'
    activations = np.array(train_data[column_name].tolist(), dtype=np.float32)
    with joblib.parallel_backend(backend='loky', n_jobs=-1):
        isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components, n_jobs=-1)
        embeddings = isomap.fit_transform(activations)
        joblib.dump(isomap, save_path / f"isomap_n_neighbors_{n_neighbors}_n_components_{n_components}.joblib")
        #dump embeddings and original training data
        np.save(save_path / f"embeddings_n_neighbors_{n_neighbors}_n_components_{n_components}.npy", embeddings)
        train_data.to_parquet(save_path / f"train_data_n_neighbors_{n_neighbors}_n_components_{n_components}.parquet")
    
        #Evaluate on val data
        #use specified fraction of val data for evaluation
        val_data = val_data.sample(frac=config.data.val_fraction, random_state=config.training.random_seed)
        activations_val = np.array(val_data[column_name].tolist(), dtype=np.float32)
        val_embeddings = isomap.transform(activations_val)
        #check trustworthiness score for different values of k
        scores = {}
        for k in range(5, 51, 5):
            score = trustworthiness(activations_val, val_embeddings, n_neighbors=k)
            scores[k] = score
            print(f"Trustworthiness score for k={k}: {score}")
    
    #save graph of scores
    import matplotlib.pyplot as plt
    plt.figure(figsize=(config.visualization.fig_width_compact, config.visualization.fig_height_compact))
    plt.plot(list(scores.keys()), list(scores.values()), marker='o')
    plt.title("Trustworthiness Score for Different Values of k")
    plt.xlabel("k (number of neighbors)")
    plt.ylabel("Trustworthiness Score")
    plt.xticks(list(scores.keys()))
    plt.grid()
    plt.savefig(save_path / f"trustworthiness_scores_n_neighbors_{n_neighbors}_n_components_{n_components}.png")
    plt.close()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train and save Isomap model")
    
    # Isomap parameters
    parser.add_argument("--n_components", type=int, help="Number of components for Isomap")
    parser.add_argument("--n_neighbors", type=int, help="Number of neighbors for Isomap")
    parser.add_argument("--layer_for_activation", type=int, help="Layer index for activation extraction")
    parser.add_argument("--train_fraction", type=float, help="Fraction of data for training")
    parser.add_argument("--val_fraction", type=float, help="Fraction of data for validation")
    parser.add_argument("--random_seed", type=int, help="Random seed for reproducibility")
    
    add_config_argument(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    
    # Override config with CLI arguments
    if args.n_components is not None:
        config.dimensionality.n_components = args.n_components
    if args.n_neighbors is not None:
        config.dimensionality.n_neighbors = args.n_neighbors
    if args.layer_for_activation is not None:
        config.model.layer_for_activation = args.layer_for_activation
    if args.train_fraction is not None:
        config.data.train_fraction = args.train_fraction
    if args.val_fraction is not None:
        config.data.val_fraction = args.val_fraction
    if args.random_seed is not None:
        config.training.random_seed = args.random_seed
    
    #print beggining timestamp
    
    print(f"Starting Isomap training at {datetime.datetime.now()}")
    
    from utils import load_data
    from pathlib import Path
    dataframes = load_data.load_train_test_val_first_parquet(
        train_size=config.data.test_train_split, 
        val_size=config.data.val_fraction, 
        timing=True
    )
    save_path = Path(__file__).parent.parent / "results" / f"isomap_{config.dimensionality.n_components}D"
    save_path.mkdir(exist_ok=True)
    train_and_save_isomap(dataframes, save_path, config)

    print(f"Finished Isomap training at {datetime.datetime.now()}")

