import torch

class TiedWeightAutoencoder(torch.nn.Module):
    """
    Autoencoder where encoder weight matrices are transposes of decoder weight matrices.
    Biases are decoupled (independent for encoder and decoder).
    Uses ReLU activation for local linearity.
    """
    def encoder_forward(self,x):
        h1 = torch.nn.functional.relu(x @ self.encoder_mat_1 + self.encoder_bias_1)
        h2 = torch.nn.functional.relu(h1 @ self.encoder_mat_2 + self.encoder_bias_2)
        z = h2 @ self.encoder_mat_3 + self.encoder_bias_3
        return z

    def decoder_forward(self,z):
        h2 = torch.nn.functional.relu(z @ self.encoder_mat_3.t() + self.decoder_bias_1)
        h1 = torch.nn.functional.relu(h2 @ self.encoder_mat_2.t() + self.decoder_bias_2)
        x_recon = h1 @ self.encoder_mat_1.t() + self.decoder_bias_3
        return x_recon

    def __init__(self, input_dim, latent_dim, device='cpu', dtype=torch.float32):
        super(TiedWeightAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = int(1/2 * input_dim + 1/2 * latent_dim)
        self.device = device
        self.dtype = dtype

        self.encoder_mat_1 = torch.nn.Parameter(torch.empty(self.input_dim, self.hidden_dim, dtype=self.dtype))
        self.encoder_bias_1 = torch.nn.Parameter(torch.empty(self.hidden_dim, dtype=self.dtype))
        self.encoder_mat_2 = torch.nn.Parameter(torch.empty(self.hidden_dim, self.hidden_dim, dtype=self.dtype))
        self.encoder_bias_2 = torch.nn.Parameter(torch.empty(self.hidden_dim, dtype=self.dtype))
        self.encoder_mat_3 = torch.nn.Parameter(torch.empty(self.hidden_dim, self.latent_dim, dtype=self.dtype))
        self.encoder_bias_3 = torch.nn.Parameter(torch.empty(self.latent_dim, dtype=self.dtype))

        self.decoder_bias_1 = torch.nn.Parameter(torch.empty(self.hidden_dim, dtype=self.dtype))
        self.decoder_bias_2 = torch.nn.Parameter(torch.empty(self.hidden_dim, dtype=self.dtype))
        self.decoder_bias_3 = torch.nn.Parameter(torch.empty(self.input_dim, dtype=self.dtype))

    def forward(self, x):
        z = self.encoder_forward(x)
        x_recon = self.decoder_forward(z)
        return x_recon

    def encode(self, x):
        return self.encoder_forward(x)

    def decode(self, z):
        return self.decoder_forward(z)


if __name__ == "__main__":
    # Sanity check: verify training works
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model and data
    model = TiedWeightAutoencoder(input_dim=100, latent_dim=20, device=device)
    model.to(device)
    
    # Initialize weights properly
    torch.nn.init.kaiming_uniform_(model.encoder_mat_1, a=0, mode='fan_in')
    torch.nn.init.kaiming_uniform_(model.encoder_mat_2, a=0, mode='fan_in')
    torch.nn.init.kaiming_uniform_(model.encoder_mat_3, a=0, mode='fan_in')
    torch.nn.init.zeros_(model.encoder_bias_1)
    torch.nn.init.zeros_(model.encoder_bias_2)
    torch.nn.init.zeros_(model.encoder_bias_3)
    torch.nn.init.zeros_(model.decoder_bias_1)
    torch.nn.init.zeros_(model.decoder_bias_2)
    torch.nn.init.zeros_(model.decoder_bias_3)
    
    # Dummy data
    batch_size = 3200
    x = torch.randn(batch_size, 100, device=device)
    val_data = torch.randn(batch_size//8, 100, device=device)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    
    # Training loop
    losses = []
    val_losses = []
    for epoch in range(300):
        optimizer.zero_grad()
        x_recon = model(x)
        loss = criterion(x_recon, x)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")
        
        # Compute validation loss
        with torch.no_grad():
            val_recon = model(val_data)
            val_loss = criterion(val_recon, val_data)
            val_losses.append(val_loss.item())
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss.item():.6f}")
    print(f"\nFinal Training Loss: {losses[-1]:.6f}")
    print(f"Final Validation Loss: {val_losses[-1]:.6f}")

    # Check that loss decreased
    if val_losses[-1] < val_losses[0]:
        print(f"\n✓ Sanity check passed! Validation loss decreased from {val_losses[0]:.6f} to {val_losses[-1]:.6f}")
    else:
        print(f"\n✗ Warning: Validation loss did not decrease consistently")