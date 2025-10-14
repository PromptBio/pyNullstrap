"""
Base Nullstrap estimator class following scikit-learn interface.
"""

from typing import Optional

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
        Regularization parameter. Auto-selected if None. Set to 0 for no regularization.
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

    # =============================================================================
    # ABSTRACT METHODS - Must be implemented by subclasses
    # =============================================================================
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "BaseNullstrap":
        """Fit the Nullstrap estimator."""
        raise NotImplementedError("Subclasses must implement fit method")

    def _fit_lasso_model(self, *args, **kwargs):
        """Fit LASSO model for the specific model type."""
        raise NotImplementedError("Subclasses must implement _fit_lasso_model")

    def _generate_knockoff_data(self, *args, **kwargs):
        """Generate knockoff data for the specific model type."""
        raise NotImplementedError("Subclasses must implement _generate_knockoff_data")

    def _estimate_correction_factor(self, *args, **kwargs):
        """Estimate correction factor for FDR control."""
        raise NotImplementedError("Subclasses must implement _estimate_correction_factor")

    def _compute_scale_factor(self, *args, **kwargs):
        """Compute scale factor for correction."""
        raise NotImplementedError("Subclasses must implement _compute_scale_factor")

    # =============================================================================
    # VALIDATION METHODS
    # =============================================================================

    def _validate_parameters(self) -> None:
        """
        Validate input parameters.

        Raises
        ------
        ValueError
            If fdr is not between 0 and 1, alpha_ is negative, B_reps is not a positive integer,
            or random_state is not an integer.
        """
        if not 0 < self.fdr < 1:
            raise ValueError(f"fdr must be between 0 and 1, got {self.fdr}")

        if self.alpha_ is not None and self.alpha_ < 0:
            raise ValueError(f"alpha_ must be non-negative, got {self.alpha_}")

        if self.B_reps is not None and (self.B_reps <= 0 or not isinstance(self.B_reps, int)):
            raise ValueError(f"B_reps must be a positive integer, got {self.B_reps} ({type(self.B_reps)})")

        if self.random_state is not None and not isinstance(self.random_state, int):
            raise ValueError(f"random_state must be an integer, got {type(self.random_state)}")

    def _validate_X(self, X: np.ndarray) -> None:
        """
        Validate the input matrix X for appropriate type, shape, and absence of NaN or infinite values.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data matrix.

        Raises
        ------
        ValueError
            If X is not 2D, is empty, or contains NaN/infinite values.
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D array with shape {X.shape}")
        if X.shape[0] == 0:
            raise ValueError("X cannot be empty (0 samples)")
        if X.shape[1] == 0:
            raise ValueError("X cannot be empty (0 features)")
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("X contains NaN or infinite values")

    def _validate_y(self, y: np.ndarray) -> None:
        """
        Validate the target vector y for appropriate type, shape, and absence of NaN or infinite values.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target values.

        Raises
        ------
        ValueError
            If y is not a 1D array or contains NaN/infinite values.
        """
        if y.ndim != 1:
            raise ValueError(f"y must be 1D array, got {y.ndim}D array with shape {y.shape}")
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            raise ValueError("y contains NaN or infinite values")

    def _validate_sample_sizes(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Validate that X and y have the same number of samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data matrix.
        y : ndarray of shape (n_samples,)
            Target values.

        Raises
        ------
        ValueError
            If X and y have different numbers of samples.
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same number of samples. X has {X.shape[0]}, y has {y.shape[0]}")

    def _validate_data(self, X: np.ndarray, y: Optional[np.ndarray] = None, require_y: bool = False) -> None:
        """
        Wrapper method that validates X and y jointly by calling individual validation methods.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data matrix.
        y : ndarray of shape (n_samples,), optional
            Target values.
        require_y : bool, default=False
            Whether y is required for this model.

        Raises
        ------
        ValueError
            If require_y is True but y is None, or if X and y have different sample sizes.
        """
        self._validate_X(X)
        
        if y is not None:
            self._validate_y(y)
            self._validate_sample_sizes(X, y)
        elif require_y:
            raise ValueError("This model requires y parameter")

    # =============================================================================
    # UTILITY METHODS
    # =============================================================================

    def _set_random_state(self):
        """
        Set random state for reproducibility.

        Sets the global numpy random seed if random_state is provided.
        This ensures reproducible results across runs.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

    def get_selected_features(self, statistics: np.ndarray, threshold: float) -> np.ndarray:
        """
        Get indices of features selected at a given threshold.

        Parameters
        ----------
        statistics : array-like of shape (n_features,)
            Test statistics for all features.
        threshold : float
            Selection threshold.

        Returns
        -------
        selected : ndarray
            Indices of features with statistics >= threshold.
        """
        return np.where(statistics >= threshold)[0]

    # =============================================================================
    # TRANSFORM METHOD
    # =============================================================================

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Return selected features from X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_selected : ndarray of shape (n_samples, n_selected_features)
            Selected features from X.
        """
        if self.selected_ is None or not hasattr(self, "n_features_"):
            raise ValueError("Model has not been fitted yet.")

        # Validate input data
        self._validate_X(X)
        if X.shape[1] != self.n_features_:
            raise ValueError(f"X must have {self.n_features_} features, got {X.shape[1]} features")

        return X[:, self.selected_]

    # =============================================================================
    # SCKIT-LEARN COMPATIBILITY METHODS
    # =============================================================================

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

    # =============================================================================
    # GETTER METHODS
    # =============================================================================

    def get_threshold(self) -> float:
        """
        Get the computed selection threshold.

        Returns
        -------
        threshold : float
            The computed selection threshold.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if self.threshold_ is None:
            raise ValueError("Model has not been fitted yet.")
        return self.threshold_

    def get_statistic(self) -> np.ndarray:
        """
        Get the computed test statistics.

        Returns
        -------
        statistic : ndarray
            The computed test statistics for all features.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if self.statistic_ is None:
            raise ValueError("Model has not been fitted yet.")
        return self.statistic_
