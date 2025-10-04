"""
Core Nullstrap algorithmic utilities.

This module contains the fundamental algorithms used across all Nullstrap
estimators, including thresholding, correction factor estimation, and
data preprocessing.
"""
import warnings
from typing import Optional, Tuple, Any

import numpy as np
from sklearn.preprocessing import StandardScaler


def inflate_signal(
    base_coefficients: np.ndarray, alpha_reg: float, inflation_type: str = "additive"
) -> np.ndarray:
    """
    Inflate coefficients to create artificial signal for correction factor estimation.

    This function artificially inflates coefficient magnitudes to create
    synthetic signal patterns used in the correction factor estimation process.

    Parameters
    ----------
    base_coefficients : array-like
        Original fitted coefficients.
    alpha_reg : float
        Regularization parameter used for inflation.
    inflation_type : str, default="additive"
        Type of inflation: "additive" or "multiplicative".

    Returns
    -------
    inflated_coefficients : ndarray
        Coefficients with artificial signal inflation.
    """
    coefficients = np.copy(base_coefficients)

    if inflation_type == "additive":
        # Add lambda to positive coefficients, subtract from negative
        coefficients[coefficients > 0] += alpha_reg
        coefficients[coefficients < 0] -= alpha_reg
    elif inflation_type == "multiplicative":
        # Multiply by (1 + lambda)
        coefficients *= 1 + alpha_reg
    else:
        raise ValueError("inflation_type must be 'additive' or 'multiplicative'")

    return coefficients


def binary_search_threshold(
    coef_real: np.ndarray,
    coef_knockoff: np.ndarray,
    fdr: float,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> float:
    """
    Find selection threshold using binary search to control FDR.

    This implements the core Nullstrap thresholding procedure, finding the
    largest threshold such that the false discovery proportion is at most
    the target FDR level.

    Parameters
    ----------
    coef_real : array-like of shape (n_features,)
        Test statistics from the real data (typically absolute coefficients).
    coef_knockoff : array-like of shape (n_features,)
        Test statistics from knockoff/null data, potentially with correction
        factor applied.
    fdr : float
        Target false discovery rate (between 0 and 1).
    max_iter : int, default=1000
        Maximum number of binary search iterations.
    tol : float, default=1e-8
        Convergence tolerance for binary search.

    Returns
    -------
    threshold : float
        The computed selection threshold.

    Notes
    -----
    The algorithm finds the largest threshold τ such that:
    FDP(τ) = (#{knockoffs ≥ τ} + fdr/2) / max(#{real ≥ τ}, 1) ≤ fdr

    The +fdr/2 term provides additional conservativeness in the procedure.
    """
    coef_real = np.asarray(coef_real)
    coef_knockoff = np.asarray(coef_knockoff)

    if len(coef_real) != len(coef_knockoff):
        raise ValueError("coef_real and coef_knockoff must have same length")

    if not 0 < fdr < 1:
        raise ValueError("fdr must be between 0 and 1")

    left = 0.0
    right = float(np.max(coef_real))

    # Handle edge case where all coefficients are zero
    if right == 0:
        return 0.0

    for _ in range(max_iter):
        if abs(right - left) <= tol:
            break

        mid = (left + right) / 2

        # Count coefficients above threshold
        num_knockoff = int(np.sum(coef_knockoff >= mid))
        num_real = int(np.sum(coef_real >= mid))

        # Compute false discovery proportion
        # Add fdr/2 for conservativeness, ensure denominator is at least 1
        fdp = (num_knockoff + fdr / 2) / max(num_real, 1)

        if fdp > fdr:
            left = mid
        else:
            right = mid

    return right


def binary_search_correction_factor(
    coef_correction_abs: np.ndarray,
    coef_snp_abs: np.ndarray,
    signal_indices: np.ndarray,
    fdr: float,
    initial_correction_factor: float,
    binary_search_tol: float,
    scale_factor: float = 1.0,
    max_iterations: Optional[int] = None,
) -> float:
    """
    Binary search for correction factor that achieves target FDR.

    This is the shared correction factor estimation procedure used across
    all Nullstrap models. It uses binary search to find the correction factor
    that maintains the target FDR on a positive control.

    Parameters
    ----------
    coef_correction_abs : array-like of shape (n_features,)
        Absolute coefficients from positive control data.
    coef_snp_abs : array-like of shape (n_features,)
        Absolute coefficients from knockoff/null data.
    signal_indices : array-like
        Indices of true signal features in positive control.
    fdr : float
        Target false discovery rate.
    initial_correction_factor : float
        Initial left bound for binary search.
    binary_search_tol : float
        Convergence tolerance for binary search.
    scale_factor : float, default=1.0
        Scaling factor for correction application (model-specific).
    max_iterations : int, optional
        Maximum iterations for binary search. If None, no limit.

    Returns
    -------
    correction_factor : float
        Estimated correction factor.

    Notes
    -----
    This function implements the core binary search logic used across all
    Nullstrap models, with model-specific scaling factors applied during
    correction application.
    """
    import warnings

    coef_correction_abs = np.asarray(coef_correction_abs)
    coef_snp_abs = np.asarray(coef_snp_abs)
    signal_indices = np.asarray(signal_indices)

    # Calculate right bound (model-specific scaling)
    right_correction = (
        np.max(coef_correction_abs) / scale_factor
        if scale_factor > 0
        else np.max(coef_correction_abs)
    )
    left_correction = initial_correction_factor

    iteration = 0

    while abs(right_correction - left_correction) > binary_search_tol:
        if max_iterations is not None and iteration >= max_iterations:
            warnings.warn(
                f"Binary search did not converge after {max_iterations} iterations",
                UserWarning,
            )
            break

        mid = (right_correction + left_correction) / 2

        # Apply correction to knockoff coefficients with model-specific scaling
        corrected = coef_snp_abs + mid * scale_factor

        # Find threshold
        tau_mid = binary_search_threshold(coef_correction_abs, corrected, fdr)

        # Compute FDR
        rejected_mid = np.where(coef_correction_abs >= tau_mid)[0]

        if len(rejected_mid) == 0:
            FDR_mid = 0.0
        else:
            # Count false discoveries
            false_discoveries = len(set(rejected_mid) - set(signal_indices))
            FDR_mid = false_discoveries / len(rejected_mid)

        # Adjust binary search bounds
        if FDR_mid > fdr:
            left_correction = mid
        else:
            right_correction = mid

        iteration += 1

    return right_correction


def standardize_data(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    scale_by_sample_size: bool = False,
    n_samples: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Standardize data for Nullstrap models.

    Applies z-score standardization (mean=0, std=1) to the predictor matrix
    and optionally centers the response vector. Can additionally apply sample size
    normalization (1/sqrt(n)) for certain model types.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Predictor matrix.
    y : array-like of shape (n_samples,), optional
        Response vector.
    scale_by_sample_size : bool, default=False
        Whether to apply sample size normalization (1/sqrt(n)) after z-score standardization.

        **When to use True (GLM/Cox models):**
        - GLM: Uses log-likelihood optimization where gradients scale with sample size
        - Cox: Uses partial likelihood optimization where risk scores scale with sample size
        - Without scaling, regularization becomes sample-size dependent
        - Ensures consistent regularization strength across different sample sizes

        **When to use False (LM/GGM models):**
        - LM: Uses OLS where residual sum of squares naturally scales with sample size
        - GGM: Uses covariance estimation where sample covariance already has proper scaling
        - Additional scaling would double-count the sample size effect

    n_samples : int, optional
        Number of samples for sample size normalization. If None, inferred from X.

    Returns
    -------
    X_scaled : ndarray
        Standardized predictor matrix (mean=0, std=1).
        If scale_by_sample_size=True, additionally scaled by 1/sqrt(n).
    y_centered : ndarray, optional
        Centered response vector (if y provided).

    Examples
    --------
    >>> # Linear models (standard z-score only)
    >>> X_scaled, y_centered = standardize_data(X, y, scale_by_sample_size=False)

    >>> # GLM/Cox models (z-score + sample size normalization)
    >>> X_scaled, y_centered = standardize_data(X, y, scale_by_sample_size=True, n_samples=n)
    """
    X = np.asarray(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply sample size normalization if requested
    if scale_by_sample_size:
        if n_samples is None:
            n_samples = X.shape[0]
        X_scaled = X_scaled / np.sqrt(n_samples)

    # Center response vector if provided
    y_centered = None
    if y is not None:
        y_centered = y - np.mean(y)

    return X_scaled, y_centered
