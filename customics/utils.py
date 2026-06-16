"""Data utilities: sample alignment, splitting, and visualisation helpers."""

import os

import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from mudata import MuData
from sklearn.model_selection import KFold, train_test_split

from customics.exceptions import DataValidationError

sns.set_style("darkgrid")
sns.set_palette("muted")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

# ---------------------------------------------------------------------------
# Toy dataset loader
# ---------------------------------------------------------------------------


def toy_dataset() -> MuData:
    """Load toy multi-omics dataset from GitHub.

    Returns
    -------
    MuData object:
        - mdata.mod: Multi-omics dictionary (modality name → AnnData).
        - mdata.obs: Clinical metadata with sample IDs as index.
    """
    PREFIX = "https://raw.githubusercontent.com/prism-oncology/customics/refs/heads/main/data"

    omics_df = {
        "protein": pd.read_csv(f"{PREFIX}/toy_data/protein.txt", sep="\t", index_col=0).T,
        "gene_exp": pd.read_csv(f"{PREFIX}/toy_data/gene_exp.txt", sep="\t", index_col=0).T,
        "methyl": pd.read_csv(f"{PREFIX}/toy_data/methyl.txt", sep="\t", index_col=0).T,
    }

    clinical_df = pd.read_csv(f"{PREFIX}/toy_data/labels.txt", sep="\t", index_col=1, header=0)

    rng = np.random.default_rng(42)
    clinical_df["OS"] = rng.integers(0, 2, size=len(clinical_df))  # 0 = censored, 1 = event
    clinical_df["OS.time"] = rng.integers(200, 3000, size=len(clinical_df))  # days

    mdata = MuData({
        "rna": AnnData(omics_df["gene_exp"]),
        "protein": AnnData(omics_df["protein"]),
        "methyl": AnnData(omics_df["methyl"]),
    })

    mdata.obs = clinical_df

    return mdata


def prepare_input(mdata: MuData, label: str, event: str, surv_time: str) -> None:
    """Validate clinical columns and register them in ``mdata.uns``.

    Parameters
    ----------
    mdata : MuData
        Multi-omics object whose ``obs`` holds the clinical annotations.
    label : str
        Name of the ``mdata.obs`` column used as the classification target.
    event : str
        Name of the ``mdata.obs`` column holding the survival event indicator
        (1 = event, 0 = censored).
    surv_time : str
        Name of the ``mdata.obs`` column holding the survival time.

    Raises
    ------
    DataValidationError
        If any of the given columns is missing from ``mdata.obs``.
    """
    for key, column in {"label": label, "event": event, "surv_time": surv_time}.items():
        if column not in mdata.obs:
            raise DataValidationError(f"Column '{column}' not found in mdata.obs")
        mdata.uns[f"customics_{key}"] = column


# ---------------------------------------------------------------------------
# Sample alignment
# ---------------------------------------------------------------------------


def get_common_samples(adatas: list[AnnData]) -> list[str]:
    """Return sample IDs present in every AnnData.

    Parameters
    ----------
    adatas : list of AnnData
        AnnData objects whose ``obs_names`` are sample IDs.

    Returns
    -------
    list of str
        Sorted list of common sample IDs.
    """
    common = set(adatas[0].obs_names)
    for adata in adatas[1:]:
        common &= set(adata.obs_names)
    return sorted(common)


def get_sub_omics_df(omics_df: dict[str, pd.DataFrame], lt_samples: list[str]) -> dict[str, pd.DataFrame]:
    """Subset every omics DataFrame to the given samples.

    Parameters
    ----------
    omics_df : dict
        Multi-omics dictionary (source name → DataFrame).
    lt_samples : list of str
        Sample IDs to keep.

    Returns
    -------
    dict
        Same structure as `omics_df` but rows restricted to `lt_samples`.
    """
    return {key: df.loc[lt_samples, :] for key, df in omics_df.items()}


# ---------------------------------------------------------------------------
# Cross-validation split management
# ---------------------------------------------------------------------------


def save_splits(lt_samples: list[str], cohort: str, split_dir: str = "splits") -> None:
    """Compute 5-fold cross-validation splits and persist them to disk.

    Parameters
    ----------
    lt_samples : list of str
        All sample IDs.
    cohort : str
        Cohort name used to create a subdirectory under `split_dir`.
    split_dir : str
        Root directory for split files.
    """
    kf = KFold(n_splits=5)
    out_dir = os.path.join(split_dir, cohort)
    os.makedirs(out_dir, exist_ok=True)
    for i, (train_idx, test_idx) in enumerate(kf.split(lt_samples), start=1):
        train_idx, val_idx = train_test_split(train_idx, test_size=0.15)
        for name, indices in [
            ("train", train_idx),
            ("val", val_idx),
            ("test", test_idx),
        ]:
            with open(os.path.join(out_dir, f"split_{name}_{i}.txt"), "w") as f:
                for idx in indices:
                    f.write(lt_samples[idx] + "\n")


def get_splits(cohort: str, split: int, split_dir: str = "splits") -> tuple[list[str], list[str], list[str]]:
    """Load pre-computed train/val/test sample IDs for a given fold.

    Parameters
    ----------
    cohort : str
        Cohort name.
    split : int
        Fold index (1-based).
    split_dir : str
        Root directory containing split files.

    Returns
    -------
    tuple of (list, list, list)
        `(samples_train, samples_val, samples_test)`.
    """
    out_dir = os.path.join(split_dir, cohort)
    result = []
    for name in ("train", "val", "test"):
        path = os.path.join(out_dir, f"split_{name}_{split}.txt")
        with open(path) as f:
            result.append([line.rstrip() for line in f])
    return tuple(result)  # type: ignore[return-value]
