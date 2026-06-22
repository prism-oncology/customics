"""PyTorch Dataset classes for multi-omics data."""

import numpy as np
import pandas as pd
import torch
from mudata import MuData
from torch.utils.data import Dataset

from .._constants import Keys


class MultiOmicsDataset(Dataset):
    """A PyTorch Dataset that pairs multi-omics matrices with clinical outcomes.

    Each sample returns a list of per-source tensors together with its class
    label, survival time, and event indicator.  Survival metadata is read
    directly from ``mdata.obs``; class labels are taken from the pre-encoded
    `labels` Series so the dataset stays agnostic to label encoding.

    Parameters
    ----------
    mdata : MuData
        Multi-omics object whose ``obs`` holds the clinical annotations.
    lt_samples : list of str
        Ordered list of sample IDs to include (must be present in every
        omics modality and in ``mdata.obs``).
    labels : pd.Series or None
        Encoded class labels indexed by sample ID.  Pass `None` to disable
        label loading (returns 0).

    Examples
    --------
    >>> dataset = MultiOmicsDataset(mdata, samples, labels, "OS", "OS.time")
    >>> omics_tensors, label, time, event = dataset[0]
    """

    def __init__(
        self,
        mdata: MuData,
        lt_samples: list[str],
        encoded_labels: pd.Series | None,
    ) -> None:
        self.mdata = mdata
        self.clinical = mdata.obs
        self.lt_samples = lt_samples
        self.encoded_labels = encoded_labels

    def __len__(self) -> int:
        return len(self.lt_samples)

    def __getitem__(self, index: int) -> tuple[list[torch.Tensor], int, int, int]:
        sample = self.lt_samples[index]
        omics_data = [torch.tensor(adata[sample].X.astype(np.float32).ravel()) for adata in self.mdata.mod.values()]
        lbl = self.encoded_labels.loc[sample] if self.encoded_labels is not None else 0
        os_time = int(self.clinical.loc[sample, self.mdata.uns[Keys.SURV_TIME]])
        os_event = int(self.clinical.loc[sample, self.mdata.uns[Keys.SURV_TIME]])
        return omics_data, lbl, os_time, os_event

    def get_samples(self) -> list[str]:
        """Return the list of sample IDs in dataset order."""
        return self.lt_samples
