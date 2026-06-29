"""Visualisation helpers for trained customics models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from mudata import MuData

if TYPE_CHECKING:
    from customics import CustOMICS


def plot_loss(history: list, switch_epoch: int, figsize: tuple[float, float], show: bool = True) -> None:
    """Plot training (and optional validation) loss curves.

    Args:
        history: Each element is either a scalar train loss or a tuple
            `(train_loss, val_loss)`.
        switch_epoch: Epoch at which the model switches to phase 2 (drawn as a dashed line).
        figsize: Figure size `(width, height)` in inches.
        show: If True, call `plt.show()` after rendering.
    """
    n_epochs = len(history)
    plt.figure(figsize=figsize)
    plt.title("Loss vs. epochs")
    plt.vlines(
        x=switch_epoch,
        ymin=0,
        ymax=max(h[0] if isinstance(h, tuple) else h for h in history) * 1.1,
        colors="purple",
        ls="--",
        lw=2,
        label="phase 2 switch",
    )
    train_losses = [h[0] if isinstance(h, tuple) else h for h in history]
    plt.plot(range(n_epochs), train_losses, label="train loss")
    if history and isinstance(history[0], tuple) and len(history[0]) > 1:
        val_losses = [h[1] for h in history]
        plt.plot(range(n_epochs), val_losses, label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    sns.despine(offset=10, trim=True)
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0, frameon=False)

    if show:
        plt.show()


def plot_representation(model: CustOMICS, mdata: MuData, color: str, show: bool = True) -> None:
    """Compute the latent representation and save a t-SNE scatter plot.

    Args:
        model: A fitted model.
        mdata: Multi-omics object.
        color: Column in `mdata.obs` to use for colouring.
        show: If True, display the figure interactively.
    """
    from customics.utils import get_shared_samples

    shared_samples = get_shared_samples(mdata)

    adata = AnnData(X=model.get_latent_representation(mdata))
    adata.obs[color] = mdata.obs.loc[shared_samples, color].values

    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

    sc.pl.umap(adata, color=color, show=show)


def plot_survival_stratification(
    model: CustOMICS,
    mdata: MuData,
    event: str,
    surv_time: str,
    show: bool = True,
) -> None:
    """Stratify patients by median hazard and plot Kaplan-Meier curves.

    Args:
        model: A fitted model.
        mdata: Multi-omics object.
        event: Event indicator column.
        surv_time: Survival time column.
        show: If True, display the figure interactively.
    """
    import torch
    from lifelines import KaplanMeierFitter

    from customics.metrics.survival import cox_log_rank
    from customics.utils import get_shared_samples

    shared_samples = get_shared_samples(mdata)
    z = model.get_latent_representation(mdata)
    hazard_pred = model.survival_predictor(torch.tensor(z, dtype=torch.float32).to(model.device)).cpu().detach().numpy()
    median_hazard = np.mean(hazard_pred)
    high = [s for s, h in zip(shared_samples, hazard_pred) if h > median_hazard]
    low = [s for s, h in zip(shared_samples, hazard_pred) if h <= median_hazard]

    kmf_low = KaplanMeierFitter(label="low risk")
    kmf_high = KaplanMeierFitter(label="high risk")
    kmf_low.fit(mdata.obs.loc[low, surv_time], mdata.obs.loc[low, event])
    kmf_high.fit(mdata.obs.loc[high, surv_time], mdata.obs.loc[high, event])

    p_value = cox_log_rank(
        hazard_pred.reshape(-1),
        np.array(mdata.obs.loc[shared_samples, event].values, dtype=float),
        np.array(mdata.obs.loc[shared_samples, surv_time].values, dtype=float),
    )
    kmf_low.plot()
    kmf_high.plot()
    plt.title(f"Survival stratification (p-value = {p_value:.3g})")
    sns.despine(offset=10, trim=True)
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0, frameon=False)

    if show:
        plt.show()
