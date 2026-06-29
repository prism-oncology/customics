"""Cross-modal consensus reconstruction loss."""

import torch
import torch.nn as nn


def consensus_loss(x: list[torch.Tensor], autoencoders: nn.ModuleList) -> torch.Tensor:
    """Compute the cross-modal consensus loss.

    For every pair of sources (i, j), measures how well source i's encoder
    followed by source j's decoder can reconstruct source j's input.

    Args:
        x: Per-source input tensors.
        autoencoders: Per-source autoencoders with accessible `encoder` and `decoder`
            sub-modules.

    Returns:
        Scalar consensus loss.
    """
    mse = nn.MSELoss()
    loss = torch.tensor(0.0, device=x[0].device)
    for i, ae_i in enumerate(autoencoders):
        for j, ae_j in enumerate(autoencoders):
            rep_i = ae_i.encoder(x[i])
            recon_j = ae_j.decoder(rep_i)
            loss = loss + mse(recon_j, x[j])
    return loss
