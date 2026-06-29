"""Standard autoencoder model."""

import torch
import torch.nn as nn

from . import Decoder, Encoder


class AutoEncoder(nn.Module):
    """Standard autoencoder for a single omics source.

    Args:
        encoder: Deterministic encoder network.
        decoder: Deterministic decoder network.
        device: Compute device.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.to(device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode `x` and reconstruct it.

        Args:
            x: Input tensor, shape (batch, input_dim).

        Returns:
            A tuple `(x_hat, z)`:
                - x_hat: Reconstruction, shape (batch, input_dim).
                - z: Latent representation, shape (batch, latent_dim).
        """
        z = self.encoder(x)
        return self.decoder(z), z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent vector to data space.

        Args:
            z: Latent tensor, shape (batch, latent_dim).

        Returns:
            Reconstructed tensor, shape (batch, input_dim).
        """
        return self.decoder(z)

    def loss(self, x: torch.Tensor, beta: float) -> torch.Tensor:
        """Compute reconstruction loss (MSE).

        Args:
            x: Input tensor.
            beta: Unused here; kept for a consistent interface with `VAE`.

        Returns:
            Scalar MSE reconstruction loss.
        """
        x_hat, _ = self.forward(x)
        return nn.MSELoss()(x, x_hat)
