"""
Evaluation metrics for Nullstrap procedures.

This module provides utilities for computing FDR/FDP metrics and other
evaluation measures for variable selection performance.
"""

from typing import Optional, Tuple

import numpy as np


def compute_fdp(
    coef_real: np.ndarray, coef_knockoff: np.ndarray, threshold: float, fdr: float
) -> float:
    """
    Compute the false discovery proportion at a given threshold.

    Parameters
    ----------
    coef_real : array-like of shape (n_features,)
        Test statistics from the real data.
    coef_knockoff : array-like of shape (n_features,)
        Test statistics from knockoff/null data.
    threshold : float
        Selection threshold.
    fdr : float
        Target FDR (used for conservativeness adjustment).

    Returns
    -------
    fdp : float
        False discovery proportion at the threshold.
    """
    coef_real = np.asarray(coef_real)
    coef_knockoff = np.asarray(coef_knockoff)

    num_knockoff = int(np.sum(coef_knockoff >= threshold))
    num_real = int(np.sum(coef_real >= threshold))

    return (num_knockoff + fdr / 2) / max(num_real, 1)


def compute_fdr(
    selected: np.ndarray, true_features: np.ndarray, total_features: int
) -> float:
    """
    Compute False Discovery Rate.

    Parameters
    ----------
    selected : array-like
        Indices of selected features.
    true_features : array-like
        Indices of true non-null features.
    total_features : int
        Total number of features.

    Returns
    -------
    fdr : float
        False Discovery Rate.
    """
    selected = np.asarray(selected)
    true_features = np.asarray(true_features)

    if len(selected) == 0:
        return 0.0

    false_discoveries = len(set(selected) - set(true_features))
    return false_discoveries / len(selected)


def compute_power(selected: np.ndarray, true_features: np.ndarray) -> float:
    """
    Compute statistical power (True Positive Rate).

    Parameters
    ----------
    selected : array-like
        Indices of selected features.
    true_features : array-like
        Indices of true non-null features.

    Returns
    -------
    power : float
        Statistical power (fraction of true features discovered).
    """
    selected = np.asarray(selected)
    true_features = np.asarray(true_features)

    if len(true_features) == 0:
        return 1.0  # No true features to find

    true_discoveries = len(set(selected) & set(true_features))
    return true_discoveries / len(true_features)


def compute_precision_recall(
    selected: np.ndarray, true_features: np.ndarray, total_features: int
) -> Tuple[float, float]:
    """
    Compute precision and recall.

    Parameters
    ----------
    selected : array-like
        Indices of selected features.
    true_features : array-like
        Indices of true non-null features.
    total_features : int
        Total number of features.

    Returns
    -------
    precision : float
        Precision (1 - FDR).
    recall : float
        Recall (same as power).
    """
    precision = 1.0 - compute_fdr(selected, true_features, total_features)
    recall = compute_power(selected, true_features)
    return precision, recall


def compute_f1_score(
    selected: np.ndarray, true_features: np.ndarray, total_features: int
) -> float:
    """
    Compute F1 score.

    Parameters
    ----------
    selected : array-like
        Indices of selected features.
    true_features : array-like
        Indices of true non-null features.
    total_features : int
        Total number of features.

    Returns
    -------
    f1 : float
        F1 score.
    """
    precision, recall = compute_precision_recall(
        selected, true_features, total_features
    )

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def compute_selection_metrics(
    selected: np.ndarray, true_features: np.ndarray, total_features: int
) -> dict:
    """
    Compute comprehensive selection metrics.

    Parameters
    ----------
    selected : array-like
        Indices of selected features.
    true_features : array-like
        Indices of true non-null features.
    total_features : int
        Total number of features.

    Returns
    -------
    metrics : dict
        Dictionary containing various metrics:
        - fdr: False Discovery Rate
        - power: Statistical Power
        - precision: Precision (1 - FDR)
        - recall: Recall (same as power)
        - f1: F1 score
        - n_selected: Number of selected features
        - n_true: Number of true features
        - n_true_selected: Number of correctly selected features
        - n_false_selected: Number of falsely selected features
    """
    selected = np.asarray(selected)
    true_features = np.asarray(true_features)

    n_selected = len(selected)
    n_true = len(true_features)
    n_true_selected = len(set(selected) & set(true_features))
    n_false_selected = n_selected - n_true_selected

    fdr = compute_fdr(selected, true_features, total_features)
    power = compute_power(selected, true_features)
    precision = 1.0 - fdr
    recall = power
    f1 = compute_f1_score(selected, true_features, total_features)

    return {
        "fdr": fdr,
        "power": power,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_selected": n_selected,
        "n_true": n_true,
        "n_true_selected": n_true_selected,
        "n_false_selected": n_false_selected,
    }


def empirical_fdr_curve(
    statistics: np.ndarray,
    true_features: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    n_thresholds: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute empirical FDR curve across thresholds.

    Parameters
    ----------
    statistics : array-like
        Test statistics for all features.
    true_features : array-like
        Indices of true non-null features.
    thresholds : array-like, optional
        Thresholds to evaluate. If None, uses quantiles of statistics.
    n_thresholds : int, default=100
        Number of thresholds to use if thresholds not provided.

    Returns
    -------
    thresholds : ndarray
        Threshold values.
    fdrs : ndarray
        FDR values at each threshold.
    """
    statistics = np.asarray(statistics)
    true_features = np.asarray(true_features)

    if thresholds is None:
        thresholds = np.quantile(statistics, np.linspace(0, 1, n_thresholds))
    else:
        thresholds = np.asarray(thresholds)

    fdrs = []
    for threshold in thresholds:
        selected = np.where(statistics >= threshold)[0]
        fdr = compute_fdr(selected, true_features, len(statistics))
        fdrs.append(fdr)

    return thresholds, np.array(fdrs)


def power_curve(
    statistics: np.ndarray,
    true_features: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    n_thresholds: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute power curve across thresholds.

    Parameters
    ----------
    statistics : array-like
        Test statistics for all features.
    true_features : array-like
        Indices of true non-null features.
    thresholds : array-like, optional
        Thresholds to evaluate. If None, uses quantiles of statistics.
    n_thresholds : int, default=100
        Number of thresholds to use if thresholds not provided.

    Returns
    -------
    thresholds : ndarray
        Threshold values.
    powers : ndarray
        Power values at each threshold.
    """
    statistics = np.asarray(statistics)
    true_features = np.asarray(true_features)

    if thresholds is None:
        thresholds = np.quantile(statistics, np.linspace(0, 1, n_thresholds))
    else:
        thresholds = np.asarray(thresholds)

    powers = []
    for threshold in thresholds:
        selected = np.where(statistics >= threshold)[0]
        power = compute_power(selected, true_features)
        powers.append(power)

    return thresholds, np.array(powers)
