import torch

class StandardAutoencoder(torch.nn.Module):
    """
    Autoencoder with 2 hidden layers in each direction.
    hidden_dim_1 = 2/3 * input_dim + 1/3 * latent_dim
    hidden_dim_2 = 1/3 * input_dim + 2/3 * latent_dim
    ReLU activation is used specifically for local linearity
    """

    def __init__(self, input_dim, latent_dim, device='cpu'):
        super(StandardAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = int(1/2 * input_dim + 1/2 * latent_dim)
        self.device = device

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.latent_dim)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

    def train(self, data, num_epochs=300, learning_rate=1e-3):
        
        #make sure that data is of the same dimension as the input dimension of the autoencoder
        assert data.shape[1] == self.input_dim, f"Data dimension {data.shape[1]} does not match input dimension {self.input_dim} of the autoencoder."

        data = data.to(self.device)
        self.to(self.device)

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()

        for epoch in range(num_epochs):
            self.forward(data)
            loss = criterion(self.forward(data), data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}', flush=True)
            
    def train_with_geometric_regularisation(self, data, num_epochs=300, learning_rate=1e-3, reg_weight=1e-1):
        #the regularisation is that we want each of the matricies in the decoder, labled A_i
        #Where the forward operation is x' = A_n * A_(n-1) * ... * A_1 * z

        #satisfy that ||A_i^T * A_i - I||_F is small, where I is the identity matrix and ||.||_F is the Frobenius norm
        #This encourages the decoder to be isometric, as the differential of a linear operation is
        #the operation itself.
        assert data.shape[1] == self.input_dim, f"Data dimension {data.shape[1]} does not match input dimension {self.input_dim} of the autoencoder."
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()

        data = data.to(self.device)
        self.to(self.device)

        for epoch in range(num_epochs):
            self.forward(data)
            loss = criterion(self.forward(data), data) + reg_weight * self.geometric_regularisation()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}',flush=True)
    
    def geometric_regularisation(self):
        reg = 0
        for module in self.decoder:
            if isinstance(module, torch.nn.Linear):
                # Compute the regularisation term for each linear layer in the decoder
                weight = module.weight
                # Compute the difference between weight^T * weight and identity matrix
                reg += torch.norm(weight.t() @ weight - torch.eye(weight.shape[1], device=weight.device), p='fro')
        return reg

