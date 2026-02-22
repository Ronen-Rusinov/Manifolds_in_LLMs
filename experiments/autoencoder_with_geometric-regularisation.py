import os
import sys
import torch
from sklearn.datasets import make_s_curve

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from standard_autoencoder import StandardAutoencoder

if __name__ == "__main__":

    #print the details of the current cuda device if available
    if torch.cuda.is_available():
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("CUDA not available, using CPU.", flush=True)

    # Generate synthetic swiss roll data
    n_samples = 1000
    noise = 0.05
    data, color = make_s_curve(n_samples=n_samples, noise=noise)
    #Expand the data to 120 dimensions by adding random noise dimensions
    data = torch.tensor(data, dtype=torch.float32)
    noise_dims = 118
    random_noise = torch.randn(n_samples, noise_dims) * noise
    data = torch.cat((data, random_noise), dim=1).numpy()

    # Define autoencoder parameters
    input_dim = data.shape[1]
    latent_dim = 2  # We want to reduce to 2 dimensions for visualization

    # Initialize and train the autoencoder with geometric regularization
    autoencoder = StandardAutoencoder(input_dim=input_dim, latent_dim=latent_dim, device='cuda' if torch.cuda.is_available() else 'cpu')

    print(autoencoder.hidden_dim_1, autoencoder.hidden_dim_2, flush=True) 

    autoencoder.train_with_geometric_regularisation(data=torch.tensor(data, dtype=torch.float32), num_epochs=300, learning_rate=1e-3, reg_weight=0.1)

    #initialise and train the autoencoder without geometric regularisation for comparison
    autoencoder_no_reg = StandardAutoencoder(input_dim=input_dim, latent_dim=latent_dim, device='cuda' if torch.cuda.is_available() else 'cpu')
    autoencoder_no_reg.train(data=torch.tensor(data, dtype=torch.float32), num_epochs=300, learning_rate=1e-3)

    test_data = torch.tensor(data, dtype=torch.float32).to(autoencoder.device)
    with torch.no_grad():
        z_reg = autoencoder.encoder(test_data).cpu().numpy()
        z_no_reg = autoencoder_no_reg.encoder(test_data).cpu().numpy()
    # Visualize the latent space representations
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(z_reg[:, 0], z_reg[:, 1], c=color, s=5)
    plt.title('Latent Space with Geometric Regularization')
    plt.subplot(1, 2, 2)
    plt.scatter(z_no_reg[:, 0], z_no_reg[:, 1], c=color, s=5)
    plt.title('Latent Space without Geometric Regularization')
    plt.savefig('latent_space_comparison.png')


