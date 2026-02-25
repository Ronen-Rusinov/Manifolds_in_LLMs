import os
import sys
import torch
from sklearn.datasets import make_s_curve

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from standard_autoencoder import StandardAutoencoder
from config_manager import load_config, add_config_argument
import argparse

if __name__ == "__main__":
    # Load configuration with CLI argument overrides
    parser = argparse.ArgumentParser(description="Train autoencoder with geometric regularization")
    
    # Autoencoder parameters
    parser.add_argument("--latent_dim", type=int, help="Latent dimension for autoencoder")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--learning_rate", "--lr", type=float, help="Learning rate")
    parser.add_argument("--random_seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--regularization_weight", type=float, help="Regularization weight")
    parser.add_argument("--noise_level", type=float, help="Noise level for swiss roll data")
    
    add_config_argument(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    
    # Override config with CLI arguments
    if args.latent_dim is not None:
        config.model.latent_dim = args.latent_dim
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.random_seed is not None:
        config.training.random_seed = args.random_seed
    if args.regularization_weight is not None:
        config.training.regularization_weight = args.regularization_weight
    if args.noise_level is not None:
        config.synthetic_data.noise_level = args.noise_level

    #print the details of the current cuda device if available
    if torch.cuda.is_available():
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("CUDA not available, using CPU.", flush=True)

    # Generate synthetic swiss roll data
    n_samples = config.data.n_samples_synthetic
    noise = config.synthetic_data.noise_level
    data, color = make_s_curve(n_samples=n_samples, noise=noise)
    #Expand the data to 120 dimensions by adding random noise dimensions
    data = torch.tensor(data, dtype=torch.float32)
    noise_dims = config.synthetic_data.n_noise_dims
    random_noise = torch.randn(n_samples, noise_dims) * noise
    data = torch.cat((data, random_noise), dim=1).numpy()

    # Define autoencoder parameters
    input_dim = data.shape[1]
    latent_dim = config.model.latent_dim_2d  # We want to reduce to 2 dimensions for visualization

    # Initialize and train the autoencoder with geometric regularization
    autoencoder = StandardAutoencoder(input_dim=input_dim, latent_dim=latent_dim, device='cuda' if torch.cuda.is_available() else 'cpu')

    print(autoencoder.hidden_dim_1, autoencoder.hidden_dim_2, flush=True) 

    autoencoder.train_with_geometric_regularisation(data=torch.tensor(data, dtype=torch.float32), num_epochs=config.training.epochs, learning_rate=config.training.learning_rate, reg_weight=config.training.regularization_weight)

    #initialise and train the autoencoder without geometric regularisation for comparison
    autoencoder_no_reg = StandardAutoencoder(input_dim=input_dim, latent_dim=latent_dim, device='cuda' if torch.cuda.is_available() else 'cpu')
    autoencoder_no_reg.train(data=torch.tensor(data, dtype=torch.float32), num_epochs=config.training.epochs, learning_rate=config.training.learning_rate)

    test_data = torch.tensor(data, dtype=torch.float32).to(autoencoder.device)
    with torch.no_grad():
        z_reg = autoencoder.encoder(test_data).cpu().numpy()
        z_no_reg = autoencoder_no_reg.encoder(test_data).cpu().numpy()
    # Visualize the latent space representations
    import matplotlib.pyplot as plt
    plt.figure(figsize=(config.visualization.fig_width_standard, config.visualization.fig_height_standard))
    plt.subplot(1, 2, 1)
    plt.scatter(z_reg[:, 0], z_reg[:, 1], c=color, s=5)
    plt.title('Latent Space with Geometric Regularization')
    plt.subplot(1, 2, 2)
    plt.scatter(z_no_reg[:, 0], z_no_reg[:, 1], c=color, s=5)
    plt.title('Latent Space without Geometric Regularization')
    plt.savefig('latent_space_comparison.png')


