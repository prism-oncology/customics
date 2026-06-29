"""Variational autoencoder model with MMD regularisation."""

import torch
import torch.nn as nn

from ..loss import compute_mmd
from . import ProbabilisticDecoder, ProbabilisticEncoder


class VAE(nn.Module):
    """Variational autoencoder using Maximum Mean Discrepancy (MMD) regularisation.

    The VAE loss combines MSE reconstruction with an MMD penalty between the
    learned posterior and a standard Gaussian prior.

    Args:
        encoder: Inference network that outputs `(mean, log_var)`.
        decoder: Generative network.
        device: Compute device.
    """

    def __init__(
        self,
        encoder: ProbabilisticEncoder,
        decoder: ProbabilisticDecoder,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.to(device)

    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Apply the reparameterisation trick.

        Args:
            mean: Posterior mean.
            log_var: Posterior log-variance.

        Returns:
            Sampled latent vector.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std).to(self.device)
        return mean + std * eps

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode `x`, sample `z`, and reconstruct.

        Args:
            x: Input tensor, shape (batch, input_dim).

        Returns:
            A tuple `(x_hat, z)`:
                - x_hat: Reconstruction, shape (batch, input_dim).
                - z: Sampled latent vector, shape (batch, latent_dim).
        """
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        return self.decoder(z), z

    def loss(self, x: torch.Tensor, beta: float) -> torch.Tensor:
        """Compute the VAE loss: reconstruction + `beta` * MMD.

        Args:
            x: Input tensor.
            beta: Weight for the MMD regularisation term.

        Returns:
            Scalar loss value.
        """
        x_hat, z = self.forward(x)
        recon = nn.MSELoss()(x, x_hat)
        prior_samples = torch.randn_like(z).to(self.device)
        mmd = compute_mmd(prior_samples, z)
        return recon + beta * mmd
