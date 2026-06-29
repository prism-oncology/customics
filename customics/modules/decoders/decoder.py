"""Standard (deterministic) decoder network."""

from collections import OrderedDict

import torch
import torch.nn as nn

from .. import FullyConnectedLayer


class Decoder(nn.Module):
    """Deterministic decoder that maps a latent vector back to data space.

    Args:
        latent_dim: Dimension of the latent representation.
        hidden_dim: Sizes of intermediate hidden layers (in encoder order; reversed
            internally).
        output_dim: Dimension of the reconstructed output.
        norm_layer: Normalization layer class or `True` for `nn.BatchNorm1d`.
        leaky_slope: Negative slope for LeakyReLU activations.
        dropout: Dropout rate (0 = disabled).
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: list[int],
        output_dim: int,
        norm_layer: type | bool = nn.BatchNorm1d,
        leaky_slope: float = 0.2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        rev = list(reversed(hidden_dim))
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        layers["InputLayer"] = FullyConnectedLayer(
            latent_dim,
            rev[0],
            norm_layer=norm_layer,
            leaky_slope=leaky_slope,
            dropout=dropout,
            activation=True,
        )
        for i in range(1, len(rev)):
            layers[f"Layer{i}"] = FullyConnectedLayer(
                rev[i - 1],
                rev[i],
                norm_layer=norm_layer,
                leaky_slope=leaky_slope,
                dropout=dropout if i % 2 == 0 else 0.0,
                activation=True,
            )
        layers["OutputLayer"] = FullyConnectedLayer(
            rev[-1],
            output_dim,
            norm_layer=norm_layer,
            leaky_slope=leaky_slope,
            dropout=0.0,
            activation=False,
            normalization=False,
        )
        self.net = nn.Sequential(layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector `z` to data space.

        Args:
            z: Latent tensor, shape (batch, latent_dim).

        Returns:
            Reconstructed tensor, shape (batch, output_dim).
        """
        return self.net(z)
