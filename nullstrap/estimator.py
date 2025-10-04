"""
Base Nullstrap estimator class following scikit-learn interface.
"""

import warnings
from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin


class BaseNullstrap(BaseEstimator, SelectorMixin):
    """
    Base class for Nullstrap feature selectors with FDR control.

    Parameters
    ----------
    fdr : float, default=0.1
        Target false discovery rate (0 < fdr < 1).
    alpha_ : float, optional
        Regularization parameter. Auto-selected if None.
    B_reps : int, optional
        Bootstrap repetitions for correction factor estimation.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    threshold_ : float
        Selection threshold after fitting.
    selected_ : ndarray
        Indices of selected features.
    statistic_ : ndarray
        Test statistics for all features.
    alpha_used_ : float
        Regularization parameter used in final fit.
    correction_factor_ : float
        Estimated correction factor for FDR control.
    """

    def __init__(
        self,
        fdr: float = 0.1,
        alpha_: Optional[float] = None,
        B_reps: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        self.fdr = fdr
        self.alpha_ = alpha_
        self.B_reps = B_reps
        self.random_state = random_state

        # Attributes set during fitting
        self.threshold_: Optional[float] = None
        self.selected_: Optional[np.ndarray] = None
        self.statistic_: Optional[np.ndarray] = None
        self.alpha_used_: Optional[float] = None
        self.correction_factor_: Optional[float] = None

    def _validate_parameters(self):
        """Validate input parameters."""
        if not 0 < self.fdr < 1:
            raise ValueError(f"fdr must be between 0 and 1, got {self.fdr}")

        if self.alpha_ is not None and self.alpha_ <= 0:
            raise ValueError(f"alpha_ must be positive, got {self.alpha_}")

        if self.B_reps is not None and (
            self.B_reps <= 0 or not isinstance(self.B_reps, int)
        ):
            raise ValueError(
                f"B_reps must be a positive integer, got {self.B_reps} ({type(self.B_reps)})"
            )

        if self.random_state is not None and not isinstance(self.random_state, int):
            raise ValueError(
                f"random_state must be an integer, got {type(self.random_state)}"
            )

    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        """Validate X only - no dependency on y."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2D array, got {X.ndim}D array with shape {X.shape}"
            )
        if X.shape[0] == 0:
            raise ValueError("X cannot be empty (0 samples)")
        if X.shape[1] == 0:
            raise ValueError("X cannot be empty (0 features)")
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("X contains NaN or infinite values")
        return X

    def _validate_y(self, y: np.ndarray) -> np.ndarray:
        """Validate y only - no dependency on X."""
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError(
                f"y must be 1D array, got {y.ndim}D array with shape {y.shape}"
            )
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            raise ValueError("y contains NaN or infinite values")
        return y

    def _validate_data(self, X: np.ndarray, y: Optional[np.ndarray] = None, require_y: bool = False):
        """Validate X and y jointly - handles relationships between X and y."""
        X = self._validate_X(X)
        
        if require_y and y is None:
            raise ValueError("This model requires y parameter")
        
        if y is not None:
            y = self._validate_y(y)
            # Joint validation: X and y must have same number of samples
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    f"X and y must have same number of samples. X has {X.shape[0]}, y has {y.shape[0]}"
                )
        
        return X, y

    def _set_random_state(self):
        """Set random state for reproducibility."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

    def _get_support_mask(self) -> np.ndarray:
        """
        Get boolean mask of selected features.

        Required by SelectorMixin for transform() and pipeline compatibility.

        Returns
        -------
        mask : ndarray of shape (n_features,)
            Boolean mask indicating selected features.
        """
        if self.selected_ is None:
            raise ValueError("Model has not been fitted yet.")

        mask = np.zeros(self.n_features_, dtype=bool)
        mask[self.selected_] = True
        return mask

    def get_threshold(self) -> float:
        """Get the computed selection threshold."""
        if self.threshold_ is None:
            raise ValueError("Model has not been fitted yet.")
        return self.threshold_

    def get_statistic(self) -> np.ndarray:
        """Get the computed test statistics."""
        if self.statistic_ is None:
            raise ValueError("Model has not been fitted yet.")
        return self.statistic_

    def get_correction_factor(self) -> float:
        """Get the estimated correction factor."""
        if self.correction_factor_ is None:
            raise ValueError("Model has not been fitted yet.")
        return self.correction_factor_

    def get_selected_features(
        self, statistics: np.ndarray, threshold: float
    ) -> np.ndarray:
        """Get indices of features selected at a given threshold."""
        return np.where(statistics >= threshold)[0]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "BaseNullstrap":
        """
        Fit the Nullstrap estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), optional
            Target values (not used for graphical models).

        Returns
        -------
        self : BaseNullstrap
            Fitted estimator.
        """
        raise NotImplementedError("Subclasses must implement fit method")

    def _fit_base_model(self, X: np.ndarray, y: Optional[np.ndarray]) -> Any:
        """Fit the base statistical model (e.g., LASSO)."""
        raise NotImplementedError("Subclasses must implement _fit_base_model")

    def _generate_knockoff_data(self, X: np.ndarray, y: Optional[np.ndarray], **kwargs) -> tuple:
        """Generate knockoff/null data for the specific model type."""
        raise NotImplementedError("Subclasses must implement _generate_knockoff_data")

    def _estimate_correction_factor(
        self, X: np.ndarray, y: Optional[np.ndarray], base_coefficients: np.ndarray
    ) -> float:
        """Estimate the correction factor for FDR control."""
        raise NotImplementedError(
            "Subclasses must implement _estimate_correction_factor"
        )
