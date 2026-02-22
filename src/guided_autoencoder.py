#This is an extension of the standard autoencoder, as detailed in src/standard_autoencoder.py.
#With the added pretraining step of fitting the encoder and the decoder
#To the Isomap embeddings.
#I.e Fit encoder to reduce MSE between encoder output and Isomap embedding, and fit decoder to reduce MSE between decoder output and original activations.
#Then finetune as normal.

import torch

from standard_autoencoder import StandardAutoencoder


class GuidedAutoencoder(StandardAutoencoder):
	"""Standard autoencoder with Isomap-guided pretraining."""

	def pretrain_encoder_to_isomap(self, activations, isomap_embeddings, num_epochs=200, learning_rate=1e-3):
		assert activations.shape[1] == self.input_dim, (
			f"Activations dim {activations.shape[1]} does not match input dim {self.input_dim}."
		)
		assert isomap_embeddings.shape[1] == self.latent_dim, (
			f"Isomap dim {isomap_embeddings.shape[1]} does not match latent dim {self.latent_dim}."
		)

		activations = activations.to(self.device)
		isomap_embeddings = isomap_embeddings.to(self.device)
		self.to(self.device)

		optimizer = torch.optim.Adam(self.encoder.parameters(), lr=learning_rate)
		criterion = torch.nn.MSELoss()

		for epoch in range(num_epochs):
			preds = self.encoder(activations)
			loss = criterion(preds, isomap_embeddings)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if (epoch + 1) % 10 == 0:
				print(f"Encoder pretrain [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}", flush=True)

	def pretrain_decoder_from_isomap(self, isomap_embeddings, activations, num_epochs=200, learning_rate=1e-3):
		assert activations.shape[1] == self.input_dim, (
			f"Activations dim {activations.shape[1]} does not match input dim {self.input_dim}."
		)
		assert isomap_embeddings.shape[1] == self.latent_dim, (
			f"Isomap dim {isomap_embeddings.shape[1]} does not match latent dim {self.latent_dim}."
		)

		activations = activations.to(self.device)
		isomap_embeddings = isomap_embeddings.to(self.device)
		self.to(self.device)

		optimizer = torch.optim.Adam(self.decoder.parameters(), lr=learning_rate)
		criterion = torch.nn.MSELoss()

		for epoch in range(num_epochs):
			preds = self.decoder(isomap_embeddings)
			loss = criterion(preds, activations)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if (epoch + 1) % 10 == 0:
				print(f"Decoder pretrain [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}", flush=True)

	def train_with_isomap_pretraining(
		self,
		pretrain_activations,
        finetune_activations,
		isomap_embeddings,
		pretrain_epochs=200,
		finetune_epochs=300,
		learning_rate=1e-3,
	):
		"""Run encoder/decoder pretraining, then finetune on reconstruction."""
		self.pretrain_encoder_to_isomap(
			pretrain_activations,
			isomap_embeddings,
			num_epochs=pretrain_epochs,
			learning_rate=learning_rate,
		)
		self.pretrain_decoder_from_isomap(
			isomap_embeddings,
			pretrain_activations,
			num_epochs=pretrain_epochs,
			learning_rate=learning_rate,
		)
		self.train(finetune_activations, num_epochs=finetune_epochs, learning_rate=learning_rate)

