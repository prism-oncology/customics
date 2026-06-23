"""Visualisation helpers for trained customics models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mudata import MuData
from sklearn.manifold import TSNE

if TYPE_CHECKING:
    import pandas as pd

    from customics import CustOMICS


def plot_loss(history: list, switch_epoch: int, figsize: tuple = (10, 5), show: bool = True) -> None:
    """Plot training (and optional validation) loss curves.

    Parameters
    ----------
    history : list
        Each element is either a scalar train loss or a tuple
        `(train_loss, val_loss)`.
    switch_epoch : int
        Epoch at which the model switches to phase 2 (drawn as a dashed line).
    figsize : tuple
        Figure size `(width, height)` in inches.
    show : bool
        If True, call `plt.show()` after rendering.
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
    if show:
        plt.show()


def plot_representation(
    model: CustOMICS,
    mdata: MuData,
    label: str,
    filename: str,
    title: str,
    show: bool = True,
) -> None:
    """Compute the latent representation and save a t-SNE scatter plot.

    Parameters
    ----------
    model : CustOMICS
        A fitted model.
    mdata : MuData
            Multi-omics object.
    label : str
        Column in `clinical_df` to use for colouring.
    filename : str
        Output path (without extension).
    title : str
        Plot title.
    show : bool
        If True, display the figure interactively.
    """
    from customics.utils import get_common_samples

    lt_samples = get_common_samples([*list(mdata.mod.values()), mdata.obs])
    z = model.get_latent_representation(mdata)
    labels_arr = mdata.obs.loc[lt_samples, label].values
    save_plot_score(filename, z, labels_arr, title, show=show)


def plot_survival_stratification(
    model: CustOMICS,
    mdata: MuData,
    event: str,
    surv_time: str,
    plot_title: str = "",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Stratify patients by median hazard and plot Kaplan-Meier curves.

    Parameters
    ----------
    model : CustOMICS
        A fitted model.
    mdata : MuData
            Multi-omics object.
    event : str
        Event indicator column.
    surv_time : str
        Survival time column.
    plot_title : str
        Title prefix for the figure.
    save_path : str, optional
        If provided, save the figure to this path.
    show : bool
        If True, display the figure interactively.
    """
    import torch
    from lifelines import KaplanMeierFitter

    from customics.metrics.survival import cox_log_rank
    from customics.utils import get_common_samples

    lt_samples = get_common_samples([*list(mdata.mod.values()), mdata.obs])
    z = model.get_latent_representation(mdata)
    hazard_pred = model.survival_predictor(torch.tensor(z, dtype=torch.float32).to(model.device)).cpu().detach().numpy()
    median_hazard = np.mean(hazard_pred)
    high = [s for s, h in zip(lt_samples, hazard_pred) if h > median_hazard]
    low = [s for s, h in zip(lt_samples, hazard_pred) if h <= median_hazard]

    kmf_low = KaplanMeierFitter(label="low risk")
    kmf_high = KaplanMeierFitter(label="high risk")
    kmf_low.fit(mdata.obs.loc[low, surv_time], mdata.obs.loc[low, event])
    kmf_high.fit(mdata.obs.loc[high, surv_time], mdata.obs.loc[high, event])

    p_value = cox_log_rank(
        hazard_pred.reshape(-1),
        np.array(mdata.obs.loc[lt_samples, event].values, dtype=float),
        np.array(mdata.obs.loc[lt_samples, surv_time].values, dtype=float),
    )
    kmf_low.plot()
    kmf_high.plot()
    plt.title(f"{plot_title} (p-value = {p_value:.3g})")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()


def save_plot_score(filename: str, z: np.ndarray, y: np.ndarray, title: str, show: bool = False) -> None:
    """Compute a t-SNE embedding and save a colour-coded scatter plot.

    Parameters
    ----------
    filename : str
        Output file path (without extension; a `.png` suffix is appended).
    z : np.ndarray
        High-dimensional feature matrix, shape (n_samples, n_features).
    y : np.ndarray
        Class labels for colouring, shape (n_samples,).
    title : str
        Plot title.
    show : bool
        If True, display the plot interactively after saving.
    """
    tsne = TSNE(n_components=2, verbose=0, perplexity=40)
    embedding = tsne.fit_transform(z)
    df = pd.DataFrame({"targets": y, "x-axis": embedding[:, 0], "y-axis": embedding[:, 1]})
    sns.scatterplot(
        x="x-axis",
        y="y-axis",
        hue=df["targets"].tolist(),
        palette=sns.color_palette("hls", len(np.unique(y))),
        data=df,
    )
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.5, 1.1), loc=2, borderaxespad=0.0)
    plt.savefig(str(filename) + ".png", bbox_inches="tight")
    if show:
        plt.show()
    plt.clf()
