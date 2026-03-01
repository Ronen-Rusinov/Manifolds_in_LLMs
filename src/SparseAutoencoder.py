import torch
import numpy as np


class SparseAutoencoder(torch.nn.Module):
    """
    Autoencoder with sparsity constraints on the latent representation.
    
    Uses ReLU activation for local linearity (consistent with other autoencoders in the project).
    Supports both L1 sparsity and KL-divergence based sparsity penalties.
    
    Architecture:
    - Encoder: input_dim -> hidden_dim -> hidden_dim -> latent_dim
    - Decoder: latent_dim -> hidden_dim -> hidden_dim -> input_dim
    """
    
    def __init__(self, input_dim, latent_dim, device='cpu', dtype=torch.float32):
        """
        Initialize the sparse autoencoder.
        
        Args:
            input_dim: Dimension of input data
            latent_dim: Dimension of latent representation
            device: Device to use ('cpu' or 'cuda')
            dtype: Data type (torch.float32 or torch.float64)
        """
        super(SparseAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = int(1/2 * input_dim + 1/2 * latent_dim)
        self.device = device
        self.dtype = dtype
        
        # Encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_dim, dtype=self.dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim, dtype=self.dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.latent_dim, dtype=self.dtype)
        )
        
        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, self.hidden_dim, dtype=self.dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim, dtype=self.dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.input_dim, dtype=self.dtype)
        )
    
    def forward(self, x):
        """
        Forward pass through the autoencoder with weight normalization.
        
        Normalizes the decoder's final layer weights row-wise during the forward pass
        so that gradients flow through the normalization operation.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Reconstructed tensor of shape (batch_size, input_dim)
        """
        z = self.encoder(x)
        
        # Manual decoder forward with weight normalization
        # This applies normalization within the computation graph so autograd captures it
        h = self.decoder[0](z)  # Linear: latent_dim -> hidden_dim
        h = self.decoder[1](h)  # ReLU
        h = self.decoder[2](h)  # Linear: hidden_dim -> hidden_dim
        h = self.decoder[3](h)  # ReLU
        
        # Final layer with normalized weights
        final_layer = self.decoder[4]
        w = final_layer.weight
        # Normalize rows (each row is a basis vector)
        w_norm = torch.norm(w, dim=1, keepdim=True)
        w_norm = torch.clamp(w_norm, min=1e-8)
        w_normalized = w / w_norm
        
        x_recon = torch.nn.functional.linear(h, w_normalized, final_layer.bias)
        return x_recon
    
    def encode(self, x):
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor
        
        Returns:
            Latent representation
        """
        return self.encoder(x)
    
    def decode(self, z):
        """
        Decode latent representation back to input space.
        
        Args:
            z: Latent representation
        
        Returns:
            Reconstructed tensor
        """
        return self.decoder(z)
    
    def normalize_decoder_weights_row_wise(self):
        """
        Manually normalize decoder weights row-wise (L2 normalization).
        
        Note: Weight normalization is applied automatically during forward pass,
        so this method is provided mainly for manual inspection/adjustment.
        
        This ensures each basis vector (row of the final decoder weight matrix) 
        has unit norm, which improves interpretability and is standard in sparse 
        autoencoder literature.
        
        Operates on the final linear layer of the decoder (latent_dim -> input_dim).
        """
        # Get the final linear layer (maps from latent to input)
        final_layer = self.decoder[-1]
        
        # Normalize rows of the weight matrix (each row is a basis vector)
        with torch.no_grad():
            # Weight shape: (output_dim, input_dim), so we normalize across dim 1
            norms = torch.norm(final_layer.weight.data, dim=1, keepdim=True)
            # Avoid division by zero
            norms = torch.clamp(norms, min=1e-8)
            final_layer.weight.data = final_layer.weight.data / norms
    
    @staticmethod
    def l1_sparsity_loss(z, target_sparsity=0.05):
        """
        Compute L1 sparsity loss (mean absolute activation).
        
        Args:
            z: Latent activations of shape (batch_size, latent_dim)
            target_sparsity: Target sparsity level (not directly used in L1, but kept for API consistency)
        
        Returns:
            Scalar loss value
        """
        return torch.mean(torch.abs(z))
    
    @staticmethod
    def kl_divergence_sparsity_loss(z, target_sparsity=0.05, epsilon=1e-10):
        """
        Compute KL divergence sparsity loss (compared to target sparsity).
        
        Measures KL divergence between empirical activation distribution and target sparse distribution.
        
        Args:
            z: Latent activations of shape (batch_size, latent_dim)
            target_sparsity: Target average activation probability (e.g., 0.05 for 5%)
            epsilon: Small value to prevent log(0)
        
        Returns:
            Scalar loss value
        """
        # Compute average activation per neuron (across batch)
        p = torch.mean(torch.relu(z), dim=0)  # (latent_dim,)
        
        # Clamp to avoid numerical issues
        p = torch.clamp(p, epsilon, 1 - epsilon)
        
        # KL(target || p) = target * log(target/p) + (1-target) * log((1-target)/(1-p))
        target_sparsity = torch.tensor(target_sparsity, device=z.device, dtype=z.dtype)
        
        kl_loss = (
            target_sparsity * torch.log(target_sparsity / p) +
            (1 - target_sparsity) * torch.log((1 - target_sparsity) / (1 - p))
        )
        
        return torch.mean(kl_loss)
    
    @staticmethod
    def hoyer_sparsity_loss(z):
        """
        Compute Hoyer sparsity loss (sparseness measure).
        
        Hoyer sparseness = (sqrt(n) - L1/L2) / (sqrt(n) - 1)
        where n is the dimension. Ranges from 0 (dense) to 1 (sparse).
        
        Args:
            z: Latent activations of shape (batch_size, latent_dim)
        
        Returns:
            Scalar loss value (negative of sparseness, as we want to minimize it)
        """
        batch_size, latent_dim = z.shape
        n = latent_dim
        
        # Compute L1 and L2 norms per sample
        abs_z = torch.abs(z)
        l1_norm = torch.sum(abs_z, dim=1)  # (batch_size,)
        l2_norm = torch.sqrt(torch.sum(z ** 2, dim=1))  # (batch_size,)
        
        # Compute Hoyer sparseness
        hoyer = (torch.sqrt(torch.tensor(n, dtype=z.dtype)) - l1_norm / (l2_norm + 1e-10)) / (
            torch.sqrt(torch.tensor(n, dtype=z.dtype)) - 1
        )
        
        # We want to maximize sparseness, so minimize its negative
        return -torch.mean(hoyer)
    
    def extract_active_features(self, x, activation_threshold=0.0):
        """
        Extract features activated for a given input.
        
        Returns the basis vectors (decoder weights) that are multiplied by positive 
        activations in the latent space to reconstruct the input.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim) or (input_dim,)
            activation_threshold: Minimum activation value to consider a feature "active" (default: 0.0)
        
        Returns:
            dict: Contains:
                - 'latent_codes': Latent activations (batch_size, latent_dim)
                - 'active_features': List of dicts per sample with:
                    - 'indices': Indices of active latent units
                    - 'activations': Activation values for active units
                    - 'basis_vectors': Corresponding decoder basis vectors (active_count, input_dim)
                - 'reconstructed': Reconstructed input (batch_size, input_dim)
                - 'basis_vectors_all': All decoder basis vectors (latent_dim, input_dim)
        """
        # Handle single sample input
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        x_device = x.to(self.encoder[0].weight.device) if isinstance(x, torch.Tensor) else \
                   torch.from_numpy(x).to(self.encoder[0].weight.device)
        
        with torch.no_grad():
            z = self.encode(x_device)
            x_recon = self.decode(z)
            
            # Extract decoder basis vectors (weights and biases last layer of decoder)
            # Decoder last layer: Linear(hidden_dim, input_dim)
            decoder_last_layer = self.decoder[-1]
            basis_vectors = decoder_last_layer.weight.data.detach().clone()  # (input_dim, latent_dim) -> transpose for (latent_dim, input_dim)
            basis_vectors = basis_vectors.t()  # Now (latent_dim, input_dim)
            
            # Extract active features per sample
            active_features_list = []
            for i in range(z.shape[0]):
                z_i = z[i]  # (latent_dim,)
                active_mask = z_i > activation_threshold
                active_indices = torch.where(active_mask)[0]
                
                if len(active_indices) > 0:
                    active_activations = z_i[active_indices]
                    active_basis = basis_vectors[active_indices]  # (num_active, input_dim)
                else:
                    active_activations = torch.tensor([], device=z.device)
                    active_basis = torch.empty((0, self.input_dim), device=z.device)
                
                active_features_list.append({
                    'indices': active_indices.cpu().numpy(),
                    'activations': active_activations.cpu().numpy(),
                    'basis_vectors': active_basis.cpu().numpy()
                })
        
        return {
            'latent_codes': z.cpu().numpy(),
            'active_features': active_features_list,
            'reconstructed': x_recon.cpu().numpy(),
            'basis_vectors_all': basis_vectors.cpu().numpy()
        }
    
    def get_feature_contribution(self, x, activation_threshold=0.0):
        """
        Compute how much each active feature contributes to the reconstruction.
        
        For each sample, returns the weighted basis vectors (activation * basis_vector)
        that sum to approximate the reconstruction.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim) or (input_dim,)
            activation_threshold: Minimum activation value to consider a feature "active"
        
        Returns:
            dict: Contains:
                - 'latent_codes': Latent activations
                - 'feature_contributions': List of dicts per sample with:
                    - 'indices': Indices of active latent units
                    - 'contributions': activation * basis_vector for each active feature (num_active, input_dim)
                    - 'contribution_sum': Sum of all contributions (should approximate reconstruction)
                - 'reconstructed': Reconstructed input
                - 'original': Original input
                - 'reconstruction_error': Per-dimension reconstruction error
        """
        # Handle single sample input
        if isinstance(x, torch.Tensor):
            x_orig = x.clone()
            if x.dim() == 1:
                x = x.unsqueeze(0)
        else:
            x_orig = torch.from_numpy(x)
            if x_orig.dim() == 1:
                x_orig = x_orig.unsqueeze(0)
            x = torch.from_numpy(x).unsqueeze(0) if isinstance(x, np.ndarray) and x.ndim == 1 else torch.from_numpy(x)
        
        device = self.encoder[0].weight.device
        x = x.to(device) if isinstance(x, torch.Tensor) else x.to(device)
        
        with torch.no_grad():
            z = self.encode(x)
            x_recon = self.decode(z)
            
            # Extract decoder basis vectors
            decoder_last_layer = self.decoder[-1]
            basis_vectors = decoder_last_layer.weight.data.t().detach().clone()  # (latent_dim, input_dim)
            
            # Compute feature contributions
            contributions_list = []
            for i in range(z.shape[0]):
                z_i = z[i]  # (latent_dim,)
                active_mask = z_i > activation_threshold
                active_indices = torch.where(active_mask)[0]
                
                if len(active_indices) > 0:
                    active_activations = z_i[active_indices].unsqueeze(1)  # (num_active, 1)
                    active_basis = basis_vectors[active_indices]  # (num_active, input_dim)
                    contributions = active_activations * active_basis  # (num_active, input_dim)
                    contribution_sum = contributions.sum(dim=0)  # (input_dim,)
                else:
                    active_indices = torch.tensor([], dtype=torch.long, device=device)
                    contributions = torch.empty((0, self.input_dim), device=device)
                    contribution_sum = torch.zeros(self.input_dim, device=device)
                
                contributions_list.append({
                    'indices': active_indices.cpu().numpy(),
                    'contributions': contributions.cpu().numpy(),
                    'contribution_sum': contribution_sum.cpu().numpy()
                })
        
        x_orig_device = x_orig.to(device) if not x_orig.is_cuda else x_orig
        reconstruction_error = (x_recon - x_orig_device).abs()
        
        return {
            'latent_codes': z.cpu().numpy(),
            'feature_contributions': contributions_list,
            'reconstructed': x_recon.cpu().numpy(),
            'original': x_orig.cpu().numpy(),
            'reconstruction_error': reconstruction_error.cpu().numpy()
        }


class TiedWeightSparseAutoencoder(torch.nn.Module):
    """
    Sparse autoencoder with tied weights.
    
    Encoder weight matrices are transposes of decoder weight matrices.
    Applies sparsity constraints on the latent representation.
    Uses ReLU activation for local linearity.
    """
    
    def __init__(self, input_dim, latent_dim, device='cpu', dtype=torch.float32):
        """
        Initialize the tied weight sparse autoencoder.
        
        Args:
            input_dim: Dimension of input data
            latent_dim: Dimension of latent representation
            device: Device to use ('cpu' or 'cuda')
            dtype: Data type (torch.float32 or torch.float64)
        """
        super(TiedWeightSparseAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = int(1/2 * input_dim + 1/2 * latent_dim)
        self.device = device
        self.dtype = dtype
        
        # Tied weight parameters
        self.encoder_mat_1 = torch.nn.Parameter(
            torch.empty(self.input_dim, self.hidden_dim, dtype=self.dtype)
        )
        self.encoder_bias_1 = torch.nn.Parameter(torch.empty(self.hidden_dim, dtype=self.dtype))
        
        self.encoder_mat_2 = torch.nn.Parameter(
            torch.empty(self.hidden_dim, self.hidden_dim, dtype=self.dtype)
        )
        self.encoder_bias_2 = torch.nn.Parameter(torch.empty(self.hidden_dim, dtype=self.dtype))
        
        self.encoder_mat_3 = torch.nn.Parameter(
            torch.empty(self.hidden_dim, self.latent_dim, dtype=self.dtype)
        )
        self.encoder_bias_3 = torch.nn.Parameter(torch.empty(self.latent_dim, dtype=self.dtype))
        
        # Decoder biases (weights are tied)
        self.decoder_bias_1 = torch.nn.Parameter(torch.empty(self.hidden_dim, dtype=self.dtype))
        self.decoder_bias_2 = torch.nn.Parameter(torch.empty(self.hidden_dim, dtype=self.dtype))
        self.decoder_bias_3 = torch.nn.Parameter(torch.empty(self.input_dim, dtype=self.dtype))
    
    def encoder_forward(self, x):
        """Encode input through encoder layers."""
        h1 = torch.nn.functional.relu(x @ self.encoder_mat_1 + self.encoder_bias_1)
        h2 = torch.nn.functional.relu(h1 @ self.encoder_mat_2 + self.encoder_bias_2)
        z = h2 @ self.encoder_mat_3 + self.encoder_bias_3
        return z
    
    def decoder_forward(self, z):
        """
        Decode latent representation through decoder layers with weight normalization.
        
        Normalizes the final decoder weights (encoder_mat_1.t()) row-wise during 
        the forward pass so that gradients flow through the normalization.
        """
        h2 = torch.nn.functional.relu(z @ self.encoder_mat_3.t() + self.decoder_bias_1)
        h1 = torch.nn.functional.relu(h2 @ self.encoder_mat_2.t() + self.decoder_bias_2)
        
        # Normalize decoder weights (encoder_mat_1.t()) row-wise
        decoder_weights = self.encoder_mat_1.t()  # (input_dim, latent_dim) when transposed from (latent_dim, input_dim)
        w_norm = torch.norm(decoder_weights, dim=1, keepdim=True)
        w_norm = torch.clamp(w_norm, min=1e-8)
        decoder_weights_normalized = decoder_weights / w_norm
        
        x_recon = h1 @ decoder_weights_normalized.t() + self.decoder_bias_3
        return x_recon
    
    def forward(self, x):
        """
        Forward pass through the autoencoder with weight normalization.
        
        Normalizes decoder weights row-wise during the forward pass so that 
        gradients flow through the normalization operation.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Reconstructed tensor of shape (batch_size, input_dim)
        """
        z = self.encoder_forward(x)
        x_recon = self.decoder_forward(z)  # Normalization is handled in decoder_forward
        return x_recon
    
    def encode(self, x):
        """Encode input to latent representation."""
        return self.encoder_forward(x)
    
    def decode(self, z):
        """Decode latent representation back to input space."""
        return self.decoder_forward(z)
    
    def normalize_weights_row_wise(self):
        """
        Manually normalize decoder weights row-wise (L2 normalization).
        
        Note: Weight normalization is applied automatically during forward pass,
        so this method is provided mainly for manual inspection/adjustment.
        
        For tied-weight autoencoders, normalizes encoder_mat_1.t() (the decoder weights).
        Each row (basis vector) is normalized to have unit norm.
        This improves interpretability and is standard in sparse autoencoder literature.
        """
        with torch.no_grad():
            # encoder_mat_1 has shape (input_dim, hidden_dim)
            # After transpose, it's (hidden_dim, input_dim), which is the decoder's first layer
            # We normalize rows of the final projection (encoder_mat_1.t())
            norms = torch.norm(self.encoder_mat_1.data, dim=0, keepdim=True)
            # Avoid division by zero
            norms = torch.clamp(norms, min=1e-8)
            self.encoder_mat_1.data = self.encoder_mat_1.data / norms
    
    @staticmethod
    def l1_sparsity_loss(z, target_sparsity=0.05):
        """Compute L1 sparsity loss."""
        return torch.mean(torch.abs(z))
    
    @staticmethod
    def kl_divergence_sparsity_loss(z, target_sparsity=0.05, epsilon=1e-10):
        """Compute KL divergence sparsity loss."""
        p = torch.mean(torch.relu(z), dim=0)
        p = torch.clamp(p, epsilon, 1 - epsilon)
        target_sparsity = torch.tensor(target_sparsity, device=z.device, dtype=z.dtype)
        
        kl_loss = (
            target_sparsity * torch.log(target_sparsity / p) +
            (1 - target_sparsity) * torch.log((1 - target_sparsity) / (1 - p))
        )
        
        return torch.mean(kl_loss)
    
    @staticmethod
    def hoyer_sparsity_loss(z):
        """Compute Hoyer sparsity loss."""
        batch_size, latent_dim = z.shape
        n = latent_dim
        
        abs_z = torch.abs(z)
        l1_norm = torch.sum(abs_z, dim=1)
        l2_norm = torch.sqrt(torch.sum(z ** 2, dim=1))
        
        hoyer = (torch.sqrt(torch.tensor(n, dtype=z.dtype)) - l1_norm / (l2_norm + 1e-10)) / (
            torch.sqrt(torch.tensor(n, dtype=z.dtype)) - 1
        )
        
        return -torch.mean(hoyer)
    
    def extract_active_features(self, x, activation_threshold=0.0):
        """
        Extract features activated for a given input (tied weight version).
        
        Returns the basis vectors (decoder weights) that are multiplied by positive 
        activations in the latent space to reconstruct the input.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim) or (input_dim,)
            activation_threshold: Minimum activation value to consider a feature "active"
        
        Returns:
            dict: Contains:
                - 'latent_codes': Latent activations (batch_size, latent_dim)
                - 'active_features': List of dicts per sample with:
                    - 'indices': Indices of active latent units
                    - 'activations': Activation values for active units
                    - 'basis_vectors': Corresponding decoder basis vectors (active_count, input_dim)
                - 'reconstructed': Reconstructed input (batch_size, input_dim)
                - 'basis_vectors_all': All decoder basis vectors (latent_dim, input_dim)
        """
        # Handle single sample input
        if isinstance(x, torch.Tensor):
            if x.dim() == 1:
                x = x.unsqueeze(0)
        else:
            x = torch.from_numpy(x)
            if x.dim() == 1:
                x = x.unsqueeze(0)
        
        device = self.encoder_mat_1.device
        x_device = x.to(device) if isinstance(x, torch.Tensor) else torch.from_numpy(x).to(device)
        
        with torch.no_grad():
            z = self.encode(x_device)
            x_recon = self.decode(z)
            
            # For tied weights, decoder uses transposed encoder weights
            # Decoder path: z @ encoder_mat_3.t() (hidden_dim)
            # Then output from that layer is multiplied with encoder_mat_2.t()
            # But we want the final basis vectors into input space
            # The decoder weights going to input_dim are encoder_mat_1.t() (hidden_dim -> input_dim)
            
            # To get the full path from latent to input:
            # z @ encoder_mat_3.t() @ encoder_mat_2.t() @ encoder_mat_1.t() + biases
            # So basis vectors are encoder_mat_3 @ encoder_mat_2 @ encoder_mat_1 transposed
            # Actually, the effective basis for latent -> input is encoder_mat_3.t() composed with others
            # For simplicity, we compute it as the derivative/influence of each latent unit
            
            # Create basis vectors by looking at the final transformation
            basis_vectors = self.encoder_mat_1.t() @ self.encoder_mat_2.t() @ self.encoder_mat_3.t()
            basis_vectors = basis_vectors.t()  # (latent_dim, input_dim)
            
            # Extract active features per sample
            active_features_list = []
            for i in range(z.shape[0]):
                z_i = z[i]  # (latent_dim,)
                active_mask = z_i > activation_threshold
                active_indices = torch.where(active_mask)[0]
                
                if len(active_indices) > 0:
                    active_activations = z_i[active_indices]
                    active_basis = basis_vectors[active_indices]  # (num_active, input_dim)
                else:
                    active_activations = torch.tensor([], device=z.device)
                    active_basis = torch.empty((0, self.input_dim), device=z.device)
                
                active_features_list.append({
                    'indices': active_indices.cpu().numpy(),
                    'activations': active_activations.cpu().numpy(),
                    'basis_vectors': active_basis.cpu().numpy()
                })
        
        return {
            'latent_codes': z.cpu().numpy(),
            'active_features': active_features_list,
            'reconstructed': x_recon.cpu().numpy(),
            'basis_vectors_all': basis_vectors.cpu().numpy()
        }
    
    def get_feature_contribution(self, x, activation_threshold=0.0):
        """
        Compute how much each active feature contributes to the reconstruction (tied weight version).
        
        Args:
            x: Input tensor of shape (batch_size, input_dim) or (input_dim,)
            activation_threshold: Minimum activation value to consider a feature "active"
        
        Returns:
            dict: Contains:
                - 'latent_codes': Latent activations
                - 'feature_contributions': List of dicts per sample with:
                    - 'indices': Indices of active latent units
                    - 'contributions': activation * basis_vector for each active feature
                    - 'contribution_sum': Sum of all contributions
                - 'reconstructed': Reconstructed input
                - 'original': Original input
                - 'reconstruction_error': Per-dimension reconstruction error
        """
        # Handle single sample input
        if isinstance(x, torch.Tensor):
            x_orig = x.clone()
            if x.dim() == 1:
                x = x.unsqueeze(0)
        else:
            x_orig = torch.from_numpy(x)
            if x_orig.dim() == 1:
                x_orig = x_orig.unsqueeze(0)
            x = torch.from_numpy(x)
            if x.dim() == 1:
                x = x.unsqueeze(0)
        
        device = self.encoder_mat_1.device
        x = x.to(device)
        
        with torch.no_grad():
            z = self.encode(x)
            x_recon = self.decode(z)
            
            # Compute basis vectors
            basis_vectors = self.encoder_mat_1.t() @ self.encoder_mat_2.t() @ self.encoder_mat_3.t()
            basis_vectors = basis_vectors.t()  # (latent_dim, input_dim)
            
            # Compute contributions
            contributions_list = []
            for i in range(z.shape[0]):
                z_i = z[i]  # (latent_dim,)
                active_mask = z_i > activation_threshold
                active_indices = torch.where(active_mask)[0]
                
                if len(active_indices) > 0:
                    active_activations = z_i[active_indices].unsqueeze(1)  # (num_active, 1)
                    active_basis = basis_vectors[active_indices]  # (num_active, input_dim)
                    contributions = active_activations * active_basis  # (num_active, input_dim)
                    contribution_sum = contributions.sum(dim=0)  # (input_dim,)
                else:
                    active_indices = torch.tensor([], dtype=torch.long, device=device)
                    contributions = torch.empty((0, self.input_dim), device=device)
                    contribution_sum = torch.zeros(self.input_dim, device=device)
                
                contributions_list.append({
                    'indices': active_indices.cpu().numpy(),
                    'contributions': contributions.cpu().numpy(),
                    'contribution_sum': contribution_sum.cpu().numpy()
                })
        
        x_orig_device = x_orig.to(device)
        reconstruction_error = (x_recon - x_orig_device).abs()
        
        return {
            'latent_codes': z.cpu().numpy(),
            'feature_contributions': contributions_list,
            'reconstructed': x_recon.cpu().numpy(),
            'original': x_orig.cpu().numpy(),
            'reconstruction_error': reconstruction_error.cpu().numpy()
        }

