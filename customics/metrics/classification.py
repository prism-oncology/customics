"""Classification evaluation metrics and plots."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.preprocessing import OneHotEncoder


def roc_auc_score_multiclass(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ohe: OneHotEncoder,
    average: str = "macro",
) -> float:
    """Compute multi-class ROC-AUC using one-vs-one strategy.

    Args:
        y_true: Integer class labels.
        y_pred: Predicted class probabilities, shape (n_samples, n_classes).
        ohe: Fitted encoder used to binarise `y_true`.
        average: Averaging strategy passed to `sklearn.metrics.roc_auc_score`.

    Returns:
        ROC-AUC score.
    """
    y_true_bin = ohe.transform(np.array(y_true).reshape(-1, 1))
    return roc_auc_score(y_true_bin, y_pred, average=average, multi_class="ovo")


def multi_classification_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    average: str = "weighted",
    save_confusion: bool = False,
    filename: str | None = None,
    ohe: OneHotEncoder | None = None,
) -> dict[str, float]:
    """Compute a standard classification metrics dictionary.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        y_pred_proba: Predicted class probabilities, shape (n_samples, n_classes).
        average: Averaging strategy for precision, recall, and F1.
        save_confusion: If True, save a confusion matrix heatmap to `filename`.
        filename: Path prefix for the confusion matrix image (without extension).
        ohe: Required when computing AUC.

    Returns:
        Keys: `'Accuracy'`, `'F1-score'`, `'Precision'`, `'Recall'`,
        `'AUC'`.
    """
    scores: dict[str, float] = {
        "Accuracy": metrics.accuracy_score(y_true, y_pred),
        "F1-score": metrics.f1_score(y_true, y_pred, average=average),
        "Precision": metrics.precision_score(y_true, y_pred, average=average),
        "Recall": metrics.recall_score(y_true, y_pred, average=average),
        "AUC": roc_auc_score_multiclass(y_true, y_pred_proba, ohe, average=average)
        if ohe is not None
        else float("nan"),
    }
    if save_confusion and filename is not None:
        plt.figure(figsize=(10, 4))
        sns.heatmap(
            metrics.confusion_matrix(y_true, y_pred),
            annot=True,
            xticklabels=np.unique(y_true),
            yticklabels=np.unique(y_true),
            cmap="summer",
        )
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.savefig(str(filename) + ".png")
        plt.clf()
    return scores


def plot_roc_multiclass(
    y_test: np.ndarray,
    y_pred_proba: np.ndarray,
    filename: str = "",
    n_classes: int = 2,
    var_names: list[str] | None = None,
    figsize: tuple[float, float] = (4, 3),
) -> None:
    """Plot per-class ROC curves for a multi-class model.

    Args:
        y_test: Ground-truth integer labels.
        y_pred_proba: Predicted class probabilities, shape (n_samples, n_classes).
        filename: If non-empty, save the figure to `roc_multi_{filename}.png`.
        n_classes: Number of classes.
        var_names: Class names used in the legend.
        figsize: Figure size for the ROC curve.
    """
    if var_names is None:
        var_names = [str(i) for i in range(n_classes)]

    colors = ["red", "green", "blue", "magenta", "orange", "cyan"]
    plt.figure(figsize=figsize)
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        color = colors[i % len(colors)]
        plt.plot(fpr, tpr, color=color, lw=1, label=f"{var_names[i]} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=2, label="random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve — {filename}")
    plt.legend(loc="lower right")
    sns.despine(offset=10, trim=True)
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0, frameon=False)

    if filename:
        plt.savefig(f"roc_multi_{filename}.png")
