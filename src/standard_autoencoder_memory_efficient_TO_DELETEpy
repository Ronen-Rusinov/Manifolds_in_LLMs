import torch
import numpy as np

class MemoryEfficientAutoencoder(torch.nn.Module):
    """
    Memory-efficient autoencoder with gradient accumulation support.
    Uses the same architecture as StandardAutoencoder but with built-in
    support for training on datasets that don't fit in GPU memory.
    
    Architecture:
    - 2 hidden layers in each direction
    - hidden_dim = 1/2 * input_dim + 1/2 * latent_dim
    - ReLU activation for local linearity
    """

    def __init__(self, input_dim, latent_dim, device='cpu', dtype=torch.float32):
        super(MemoryEfficientAutoencoder, self).__init__()
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

    def train_with_accumulation(self, data, val_data=None, num_epochs=300, learning_rate=1e-3, 
                                patience=20, accumulation_steps=6):
        """
        Train with gradient accumulation for memory efficiency.
        
        PyTorch natively supports gradient accumulation through the following pattern:
        1. Call loss.backward() multiple times without calling optimizer.zero_grad()
        2. Gradients accumulate automatically in parameter.grad
        3. Call optimizer.step() after accumulating from all partitions
        
        Args:
            data: Training data as numpy array (kept on CPU)
            val_data: Validation data as numpy array (kept on CPU)
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            patience: Early stopping patience
            accumulation_steps: Number of partitions for gradient accumulation
        """
        
        # Validate input dimensions
        assert data.shape[1] == self.input_dim, \
            f"Data dimension {data.shape[1]} does not match input dimension {self.input_dim}"

        self.to(self.device)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()
        
        # Early stopping variables
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_state = None

        for epoch in range(num_epochs):
            # Shuffle data at the beginning of each epoch
            indices = np.random.permutation(len(data))
            shuffled_data = data[indices]
            
            # Calculate partition size
            partition_size = len(shuffled_data) // accumulation_steps
            
            # Training with gradient accumulation
            self.train()
            optimizer.zero_grad()  # Zero gradients at the start of epoch
            epoch_loss = 0.0
            
            for step in range(accumulation_steps):
                # Get partition indices
                start_idx = step * partition_size
                if step == accumulation_steps - 1:
                    # Last partition gets remaining samples
                    end_idx = len(shuffled_data)
                else:
                    end_idx = start_idx + partition_size
                
                # Load partition to GPU
                partition = torch.from_numpy(shuffled_data[start_idx:end_idx]).to(self.device)
                
                # Forward pass
                output = self.forward(partition)
                loss = criterion(output, partition)
                
                # Scale loss by number of accumulation steps for proper gradient averaging
                loss = loss / accumulation_steps
                
                # Backward pass - gradients accumulate in parameter.grad
                loss.backward()
                
                epoch_loss += loss.item() * accumulation_steps
                
                # Free GPU memory immediately
                del partition, output, loss
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
            
            # Update weights after accumulating gradients from all partitions
            optimizer.step()
            
            # Validation with early stopping
            if val_data is not None:
                self.eval()
                with torch.no_grad():
                    # Process validation data in partitions to avoid memory issues
                    val_losses = []
                    val_partition_size = len(val_data) // accumulation_steps
                    
                    for step in range(accumulation_steps):
                        start_idx = step * val_partition_size
                        if step == accumulation_steps - 1:
                            end_idx = len(val_data)
                        else:
                            end_idx = start_idx + val_partition_size
                        
                        # Load validation partition to GPU only when needed
                        val_partition = torch.from_numpy(val_data[start_idx:end_idx]).to(self.device)
                        val_output = self.forward(val_partition)
                        val_loss = criterion(val_output, val_partition)
                        val_losses.append(val_loss.item())
                        
                        # Free GPU memory
                        del val_partition, val_output, val_loss
                        if self.device == 'cuda':
                            torch.cuda.empty_cache()
                    
                    val_loss = np.mean(val_losses)
                
                # Early stopping logic
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    # Store best state on CPU to free GPU memory
                    best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                else:
                    epochs_without_improvement += 1
                
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, '
                          f'Val Loss: {val_loss:.4f}, Best Val: {best_val_loss:.4f}', flush=True)
                
                # Check early stopping
                if epochs_without_improvement >= patience:
                    print(f'Early stopping triggered after {epoch + 1} epochs. '
                          f'Best val loss: {best_val_loss:.4f}', flush=True)
                    # Restore best model
                    self.load_state_dict(best_state)
                    break
            else:
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}', flush=True)

    def predict_in_batches(self, data, batch_size=None, accumulation_steps=6):
        """
        Make predictions in batches to avoid memory issues.
        
        Args:
            data: Input data as numpy array (on CPU)
            batch_size: Size of each batch (if None, uses len(data) // accumulation_steps)
            accumulation_steps: Number of batches to split data into
        
        Returns:
            Predictions as numpy array
        """
        self.eval()
        predictions = []
        
        if batch_size is None:
            batch_size = len(data) // accumulation_steps
        
        num_batches = (len(data) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(data))
                
                batch = torch.from_numpy(data[start_idx:end_idx]).to(self.device)
                batch_pred = self.forward(batch).cpu().numpy()
                predictions.append(batch_pred)
                
                del batch
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
        
        return np.vstack(predictions)
