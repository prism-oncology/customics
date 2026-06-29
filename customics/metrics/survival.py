"""Survival analysis evaluation metrics."""

import numpy as np
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index


def CIndex_lifeline(
    hazards: np.ndarray,
    labels: np.ndarray,
    survtime_all: np.ndarray,
) -> float:
    """Compute the concordance index (C-index).

    Args:
        hazards: Predicted hazard scores, shape (n_samples, 1) or (n_samples,).
        labels: Event indicators (1 = event, 0 = censored).
        survtime_all: Observed survival times.

    Returns:
        Concordance index in [0, 1].
    """
    return concordance_index(survtime_all, -hazards, labels)


def cox_log_rank(
    hazardsdata: np.ndarray,
    labels: np.ndarray,
    survtime_all: np.ndarray,
) -> float:
    """Compute the log-rank test p-value after dichotomising hazard scores at the median.

    Args:
        hazardsdata: Predicted hazard scores, shape (n_samples,).
        labels: Event indicators.
        survtime_all: Observed survival times.

    Returns:
        Log-rank p-value.
    """
    median = np.median(hazardsdata)
    high_risk = hazardsdata > median
    results = logrank_test(
        survtime_all[~high_risk],
        survtime_all[high_risk],
        event_observed_A=labels[~high_risk],
        event_observed_B=labels[high_risk],
    )
    return results.p_value


def accuracy_cox(hazardsdata: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of correctly stratified patients (median-split accuracy).

    Args:
        hazardsdata: Predicted hazard scores.
        labels: Ground-truth event indicators.

    Returns:
        Accuracy in [0, 1].
    """
    predicted = (hazardsdata > np.median(hazardsdata)).astype(int)
    return np.mean(predicted == labels)
