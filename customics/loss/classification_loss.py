"""Classification loss functions."""

import torch
import torch.nn as nn


def classification_loss(
    loss_name: str,
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute a classification loss.

    Args:
        loss_name: Loss function identifier: `'CE'` (cross-entropy) or `'BCE'`
            (binary cross-entropy with logits).
        y_pred: Model output logits.
        y_true: Ground-truth class labels (integer indices for CE, floats for BCE).
        reduction: Reduction to apply: `'mean'` or `'sum'`.

    Returns:
        Scalar loss value.

    Raises:
        ValueError: If `loss_name` is not `'CE'` or `'BCE'`.
    """
    if loss_name == "CE":
        return nn.CrossEntropyLoss(reduction=reduction)(y_pred, y_true)
    if loss_name == "BCE":
        return nn.BCEWithLogitsLoss(reduction=reduction)(y_pred, y_true)
    raise ValueError(f"Loss '{loss_name}' is not supported. Choose 'CE' or 'BCE'.")
