"""SHAP-based explainability utilities for customics models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .. import CustOMICS


class ModelWrapper(nn.Module):
    """Thin wrapper that exposes a single-source forward pass for SHAP.

    The wrapper takes a raw omics tensor for one source and returns class
    logits, making it compatible with `shap.DeepExplainer`.

    Args:
        model: A fitted customics model.
        source: The omics source key to explain.
    """

    def __init__(self, model: CustOMICS, source: str) -> None:
        super().__init__()
        self.model = model
        self.source = source

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.source_predict(x, self.source)


def process_phenotype_data_for_samples(
    clinical_df: pd.DataFrame,
    sample_id: list[str],
    label_encoder,
) -> pd.DataFrame:
    """Subset clinical DataFrame to the given samples.

    Args:
        clinical_df: Full clinical metadata.
        sample_id: Sample IDs to retain.
        label_encoder: Fitted label encoder (unused here; kept for API compatibility).

    Returns:
        Subsetted clinical DataFrame.
    """
    return clinical_df.loc[sample_id, :]


def random_training_sample(expr: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    """Draw a random subset of rows from an expression DataFrame.

    Args:
        expr: Expression matrix, rows are samples.
        sample_size: Number of rows to sample.

    Returns:
        Randomly sampled rows.
    """
    return expr.sample(n=sample_size, axis=0)


def split_expr_and_sample(
    condition: pd.Series,
    sample_size: int,
    expr: pd.DataFrame,
) -> pd.DataFrame:
    """Filter rows by a boolean condition and draw a random subset.

    Args:
        condition: Boolean mask aligned with `expr` rows.
        sample_size: Number of rows to sample from the filtered set.
        expr: Expression matrix.

    Returns:
        Filtered and sampled expression rows.
    """
    return expr[condition].sample(n=sample_size, axis=0)


def add_to_tensor(expr_selection: pd.DataFrame, device: str) -> torch.Tensor:
    """Convert an expression DataFrame to a float32 tensor on `device`.

    Args:
        expr_selection: Expression matrix to convert.
        device: Target device string (e.g. `'cpu'` or `'cuda'`).

    Returns:
        Float32 tensor on the specified device.
    """
    return torch.tensor(expr_selection.values.astype("float32"), dtype=torch.float32).to(device)
