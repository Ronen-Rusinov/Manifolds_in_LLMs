import torch

class StandardAutoencoder(torch.nn.Module):
    """
    Autoencoder with 2 hidden layers in each direction.
    hidden_dim_1 = 2/3 * input_dim + 1/3 * latent_dim
    hidden_dim_2 = 1/3 * input_dim + 2/3 * latent_dim
    ReLU activation is used specifically for local linearity
    """

    def __init__(self, input_dim, latent_dim, device='cpu', dtype=torch.float32):
        super(StandardAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = int(1/2 * input_dim + 1/2 * latent_dim)
        self.device = device
        self.dtype = dtype

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_dim, dtype=self.dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim, dtype=self.dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.latent_dim, dtype=self.dtype)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, self.hidden_dim, dtype=self.dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim, dtype=self.dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.input_dim, dtype=self.dtype)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    
    