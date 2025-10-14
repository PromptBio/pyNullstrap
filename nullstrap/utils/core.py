"""
Core Nullstrap algorithmic utilities.

This module contains the fundamental algorithms used across all Nullstrap estimators, 
including thresholding, correction factor estimation, and data preprocessing.
"""
import warnings
from typing import Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler


def inflate_signal(
    coef_base: np.ndarray,
    inflation_factor: float,
    inflation_type: str = "additive",
) -> np.ndarray:
    """
    Inflate coefficient magnitudes to create synthetic positive control signal.

    Parameters
    ----------
    coef_base : ndarray of shape (n_features,)
        Original fitted coefficients.
    inflation_factor : float
        Magnitude of coefficient inflation.
    inflation_type : str, default="additive"
        Type of inflation:
        - "additive": coef -> coef +/- inflation_factor (sign-preserving)
        - "multiplicative": coef -> coef * (1 + inflation_factor)

    Returns
    -------
    inflated_coefficients : ndarray of shape (n_features,)
        Coefficients with amplified signal strength.
    """
    coefficients = np.copy(coef_base)

    if inflation_type == "additive":
        coefficients[coefficients > 0] += inflation_factor
        coefficients[coefficients < 0] -= inflation_factor
    elif inflation_type == "multiplicative":
        coefficients *= 1 + inflation_factor
    else:
        raise ValueError("inflation_type must be 'additive' or 'multiplicative'")

    return coefficients

def binary_search_threshold(
    coef_real: np.ndarray,
    coef_knockoff: np.ndarray,
    fdr: float = 0.1,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> float:
    """
    Find optimal selection threshold via binary search to control FDR.

    Parameters
    ----------
    coef_real : ndarray of shape (n_features,)
        Test statistics from real data (typically absolute coefficients).
    coef_knockoff : ndarray of shape (n_features,)
        Test statistics from knockoff/null data, with optional correction factor.
    fdr : float, default=0.1
        Target false discovery rate (0 < fdr < 1).
    max_iter : int, default=1000
        Maximum number of binary search iterations.
    tol : float, default=1e-8
        Convergence tolerance for binary search.

    Returns
    -------
    threshold : float
        Largest threshold τ satisfying FDR control: FDP(τ) ≤ fdr.

    Notes
    -----
    The algorithm finds the largest threshold τ such that:
    FDP(τ) = (#{knockoffs ≥ τ} + fdr/2) / max(#{real ≥ τ}, 1) ≤ fdr

    The +fdr/2 term provides additional conservativeness in the procedure.
    """
    if len(coef_real) != len(coef_knockoff):
        raise ValueError("coef_real and coef_knockoff must have same length")
    if not 0 < fdr < 1:
        raise ValueError("fdr must be between 0 and 1")

    left, right = 0.0, float(np.max(coef_real))
    
    for _ in range(max_iter):
        if abs(right - left) <= tol:
            break

        mid = (left + right) / 2

        # Count coefficients above threshold
        num_real = np.count_nonzero(coef_real >= mid)
        num_knockoff = np.count_nonzero(coef_knockoff >= mid)

        # Compute false discovery proportion (add fdr/2 to adjust conservativeness)
        fdp = (num_knockoff + fdr / 2) / max(num_real, 1)

        if fdp > fdr:
            left = mid
        else:
            right = mid

    return right

def binary_search_correction_factor(
    coef_augmented_abs: np.ndarray,
    coef_knockoff_abs: np.ndarray,
    signal_indices: np.ndarray,
    fdr: float = 0.1,
    scale_factor: float = 1.0,
    correction_min: float = 0.05,
    correction_max: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> float:
    """
    Find correction factor via binary search to achieve target FDR on positive control.

    Parameters
    ----------
    coef_augmented_abs : ndarray of shape (n_features,)
        Absolute coefficients from augmented (positive control) data.
    coef_knockoff_abs : ndarray of shape (n_features,)
        Absolute coefficients from knockoff (null) data.
    signal_indices : ndarray
        Indices of true signal features in positive control data.
    fdr : float, default=0.1
        Target false discovery rate.
    scale_factor : float, default=1.0
        Model-specific scaling factor for correction.
    correction_min : float, default=0.05
        Lower bound for correction factor binary search.
    correction_max : float, default=1.0
        Upper bound for correction factor binary search.
    max_iter : int, default=1000
        Maximum number of binary search iterations.
    tol : float, default=1e-8
        Convergence tolerance for binary search.

    Returns
    -------
    correction_factor : float
        Estimated correction factor for FDR control.
    """
    # Calculate search bounds (model-specific scaling)
    left, right = correction_min, correction_max
    
    for _ in range(max_iter):
        if abs(right - left) <= tol:
            break

        mid = (right + left) / 2

        # Apply correction to knockoff coefficients with model-specific scaling
        corrected = coef_knockoff_abs + mid * scale_factor

        # Find threshold
        tau_mid = binary_search_threshold(coef_augmented_abs, corrected, fdr, max_iter=max_iter, tol=tol)

        # Compute FDR
        rejected_mid = np.where(coef_augmented_abs >= tau_mid)[0]
        
        # Count false discoveries
        false_discoveries = set(rejected_mid) - set(signal_indices)
        FDR_mid = len(false_discoveries) / max(len(rejected_mid), 1)

        # Adjust binary search bounds
        if FDR_mid > fdr:
            left = mid
        else:
            right = mid
    else:
        # Loop completed without breaking (didn't converge)
        warnings.warn(
            f"Binary search did not converge after {max_iter} iterations",
            UserWarning,
        )

    return right

def standardize_data(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    scale_by_sample_size: bool = False,
    n_samples: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Standardize data for Nullstrap models.

    Applies z-score standardization (mean=0, std=1) to predictors and centers
    the response. Optionally applies 1/sqrt(n) scaling for likelihood-based models.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Predictor matrix.
    y : ndarray of shape (n_samples,), optional
        Response vector.
    scale_by_sample_size : bool, default=False
        Whether to apply 1/sqrt(n) scaling after standardization.
        True for GLM/Cox (likelihood-based: gradients scale with n, needs correction).
        False for LM/GGM (least squares/covariance: loss naturally scales with n).
    n_samples : int, optional
        Number of samples for scaling. If None, inferred from X.

    Returns
    -------
    X_scaled : ndarray of shape (n_samples, n_features)
        Standardized predictor matrix. If scale_by_sample_size=True, scaled by 1/sqrt(n).
    y_centered : ndarray of shape (n_samples,) or None
        Centered response vector (mean=0), or None if y not provided.

    Examples
    --------
    >>> X_scaled, y_centered = standardize_data(X, y)  # LM/GGM
    >>> X_scaled, y_centered = standardize_data(X, y, scale_by_sample_size=True)  # GLM/Cox
    """
    # Standardize predictor matrix
    X_scaled = StandardScaler().fit_transform(X)
    
    # Apply sample size normalization if requested
    if scale_by_sample_size:
        n_samples = X.shape[0] if n_samples is None else n_samples
        X_scaled /= np.sqrt(n_samples)  # In-place division (more efficient)
    
    # Center response vector if provided
    y_centered = (y - np.mean(y)) if y is not None else None
    
    return X_scaled, y_centered
