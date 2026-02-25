import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from guided_autoencoder import GuidedAutoencoder
from utils import load_data
from config_manager import load_config_with_args


def to_tensor(array, device):
	return torch.tensor(array, dtype=torch.float32, device=device)


if __name__ == "__main__":
    # Load configuration with CLI argument overrides
    config = load_config_with_args(
        description="Train guided autoencoder with Isomap pretraining"
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_dim = config.model.latent_dim
    pretrain_epochs = config.training.pretrain_epochs
    finetune_epochs = config.training.epochs
    learning_rate = config.training.learning_rate

    dataframes = load_data.load_train_test_val_first_parquet(
        train_size=config.data.test_train_split,
        val_size=config.data.val_fraction,
        timing=True,
    )

    train_data, val_data, _ = dataframes
    pretrain_data = train_data.sample(frac=config.data.train_fraction, random_state=config.training.random_seed)
    pretrain_val = val_data.sample(frac=config.data.val_fraction, random_state=config.training.random_seed)

    train_data = train_data[~train_data.index.isin(pretrain_data.index)]
    val_data = val_data[~val_data.index.isin(pretrain_val.index)]

    train_acts = np.array(train_data["activation_layer_18"].tolist(), dtype=np.float32)
    val_acts = np.array(val_data["activation_layer_18"].tolist(), dtype=np.float32)

    isomap_path = Path(__file__).parent.parent / "results" / "isomap_12D" / "isomap_n_neighbors_50_n_components_12.joblib"
    isomap_fitted = joblib.load(isomap_path)
    pretrain_embeddings = isomap_fitted.transform(np.array(pretrain_data["activation_layer_18"].tolist(), dtype=np.float32))
    pretrain_acts = np.array(pretrain_data["activation_layer_18"].tolist(), dtype=np.float32)

    model = GuidedAutoencoder(input_dim=train_acts.shape[1], latent_dim=latent_dim, device=device)
    model.train_with_isomap_pretraining(
        to_tensor(pretrain_acts, device),
        to_tensor(train_acts, device),
        to_tensor(pretrain_embeddings, device),
        pretrain_epochs=pretrain_epochs,
        finetune_epochs=finetune_epochs,
        learning_rate=learning_rate,
    )

    with torch.no_grad():
        val_inputs = to_tensor(val_acts, device)
        recon = model(val_inputs)
        errors = torch.mean((recon - val_inputs) ** 2, dim=1).cpu().numpy()

    results_dir = Path(__file__).parent.parent / "results" / "guided_autoencoder"
    results_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), results_dir / "guided_autoencoder_state.pt")

    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, color='blue', alpha=0.7)
    plt.title("Reconstruction Error Distribution on Validation Set")
    plt.xlabel("Squared Error")    
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(results_dir / "reconstruction_error_histogram.png")

    #print timestamp
    print(f"Finished training at {pd.Timestamp.now()}")

