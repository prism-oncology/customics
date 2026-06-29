"""Multi-layer fully-connected classifier."""

from collections import OrderedDict

import torch
import torch.nn as nn

from ..modules import FullyConnectedLayer


class MultiClassifier(nn.Module):
    """Multi-layer fully-connected classifier.

    Args:
        n_class: Number of output classes.
        latent_dim: Dimension of the input latent representation.
        norm_layer: Normalization layer class (default: `nn.BatchNorm1d`).
        leaky_slope: Negative slope for LeakyReLU.
        dropout: Dropout probability.
        class_dim: Hidden layer sizes between `latent_dim` and the output.
    """

    def __init__(
        self,
        n_class: int = 2,
        latent_dim: int = 256,
        norm_layer: type = nn.BatchNorm1d,
        leaky_slope: float = 0.2,
        dropout: float = 0.0,
        class_dim: list[int] | None = None,
    ) -> None:
        super().__init__()
        if class_dim is None:
            class_dim = [128, 64]

        layers: OrderedDict[str, nn.Module] = OrderedDict()
        layers["InputLayer"] = FullyConnectedLayer(
            latent_dim,
            class_dim[0],
            norm_layer=norm_layer,
            leaky_slope=leaky_slope,
            dropout=dropout,
            activation=True,
        )
        for i in range(1, len(class_dim)):
            layers[f"Layer{i}"] = FullyConnectedLayer(
                class_dim[i - 1],
                class_dim[i],
                norm_layer=norm_layer,
                leaky_slope=leaky_slope,
                dropout=dropout if i % 2 == 1 else 0.0,
                activation=True,
            )
        layers["OutputLayer"] = FullyConnectedLayer(
            class_dim[-1],
            n_class,
            norm_layer=norm_layer,
            leaky_slope=leaky_slope,
            dropout=0.0,
            activation=False,
            normalization=False,
        )
        self.net = nn.Sequential(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits.

        Args:
            x: Input tensor, shape (batch, latent_dim).

        Returns:
            Logits, shape (batch, n_class).
        """
        return self.net(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return the predicted class index.

        Args:
            x: Input tensor, shape (batch, latent_dim).

        Returns:
            Integer class predictions, shape (batch,).
        """
        return torch.argmax(self.forward(x), dim=1)
