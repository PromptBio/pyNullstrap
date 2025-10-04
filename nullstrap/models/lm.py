"""Nullstrap estimator for Linear Models"""
import warnings
from typing import Optional, Tuple, Any

import numpy as np
from sklearn.linear_model import Lasso, LassoCV

from ..estimator import BaseNullstrap
from ..utils.core import (binary_search_correction_factor,
                          binary_search_threshold, inflate_signal,
                          standardize_data)


class NullstrapLM(BaseNullstrap):
    """
    Nullstrap feature selector for linear regression with FDR control, using LASSO regularization.

    Parameters
    ----------
    fdr : float, default=0.1
        Target false discovery rate.
    alpha_ : float, optional
        LASSO regularization parameter (maps to 'lambda' in glmnet). If None, selected by cross-validation.
    B_reps : int, default=5
        Number of repetitions for correction factor estimation.
    error_dist : str, default="normal"
        Error distribution for knockoff data generation: "normal" or "resample".
    cv_folds : int, default=10
        Cross-validation folds for alpha selection.
    max_iter : int, default=10000
        Maximum number of iterations for LASSO optimization.
    alpha_scale_factor : float, default=0.5
        Scaling factor for alpha selection (0.5 * alpha_min for conservative selection).
    binary_search_tol : float, default=1e-8
        Tolerance for binary search convergence in correction factor estimation.
    initial_correction_factor : float, default=0.05
        Initial left bound for binary search of correction factor.
    alpha_candidates_count : int, default=50
        Number of alpha candidates for cross-validation selection. Only used if alpha_range is not None.
    alpha_range : tuple, default=None
        Range of alpha values for cross-validation selection (min_alpha, max_alpha).
        If None, uses LassoCV default range (data-dependent alpha_max to alpha_max*eps).
    alpha_log_scale_range : tuple, default=(-4, 1)
        Range for alpha log scale in logspace (min_log10, max_log10). Used for np.logspace(min_log10, max_log10, count).
        Only used if alpha_range is not None.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    threshold_ : float
        Selection threshold after fitting.
    selected_ : ndarray
        Indices of selected features.
    n_features_selected_ : int
        Number of selected features.
    statistic_ : ndarray
        Test statistics (absolute LASSO coefficients).
    alpha_used_ : float
        Regularization parameter used in final fit.
    correction_factor_ : float
        Estimated correction factor for FDR control.
    sigma_hat_ : float
        Estimated noise standard deviation.
    """

    def __init__(
        self,
        fdr: float = 0.1,
        alpha_: Optional[float] = None,
        B_reps: int = 5,
        error_dist: str = "normal",
        cv_folds: int = 10,
        max_iter: int = 10000,
        alpha_scale_factor: float = 0.5,
        binary_search_tol: float = 1e-8,
        initial_correction_factor: float = 0.05,
        alpha_candidates_count: int = 50,
        alpha_range: Optional[Tuple[float, float]] = None,
        alpha_log_scale_range: Tuple[float, float] = (-4, 1),
        random_state: Optional[int] = None,
    ):
        super().__init__(
            fdr=fdr, alpha_=alpha_, B_reps=B_reps, random_state=random_state
        )
        self.error_dist = error_dist
        self.cv_folds = cv_folds
        self.max_iter = max_iter
        self.alpha_scale_factor = alpha_scale_factor
        self.binary_search_tol = binary_search_tol
        self.initial_correction_factor = initial_correction_factor
        self.alpha_candidates_count = alpha_candidates_count
        self.alpha_range = alpha_range
        self.alpha_log_scale_range = alpha_log_scale_range

        # Initialize dedicated random number generator for sampling operations
        self.sample_rng = np.random.RandomState(random_state)

        # Additional attributes for linear models
        self.sigma_hat_: Optional[float] = None
        self.residuals_: Optional[np.ndarray] = None
        self.base_model_: Optional[Any] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NullstrapLM":
        """
        Fit the Nullstrap linear regression estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : NullstrapLM
            Fitted estimator.
        """
        self._validate_parameters()
        self._set_random_state()

        # Validate and convert data
        X, y = self._validate_data(X, y, require_y=True)

        self.n_samples_, self.n_features_ = X.shape

        # Standardize data for linear models (standard z-score only)
        X_scaled, y_centered = standardize_data(X, y, scale_by_sample_size=False)

        # Fit base LASSO model
        base_model, alpha_used = self._fit_lasso_model(
            X_scaled, y_centered, alpha=self.alpha_
        )
        self.base_model_ = base_model
        self.alpha_used_ = alpha_used

        # Get base coefficients and compute residuals
        base_coefficients = base_model.coef_
        y_pred = X_scaled @ base_coefficients
        residuals = y_centered - y_pred

        # Estimate noise standard deviation
        n_nonzero = np.sum(base_coefficients != 0)
        self.sigma_hat_ = np.sqrt(
            np.sum(residuals**2) / max(1, self.n_samples_ - n_nonzero)
        )

        # Scale residuals for resampling
        scaling_factor = min(
            self.n_samples_ ** (1 / 4),
            np.sqrt(self.n_samples_ / (self.n_samples_ - n_nonzero)),
        )
        self.residuals_ = scaling_factor * residuals

        # Generate knockoff data
        X_knockoff, y_knockoff = self._generate_knockoff_data(
            X_scaled, sigma_hat=self.sigma_hat_, residuals=self.residuals_
        )

        # Fit model to knockoff data
        knockoff_model, _ = self._fit_lasso_model(
            X_knockoff, y_knockoff, alpha=alpha_used
        )
        knockoff_coefficients_abs = np.abs(knockoff_model.coef_)

        # Estimate correction factor
        correction_factor = self._estimate_correction_factor(
            X_scaled, y_centered, base_coefficients
        )
        self.correction_factor_ = correction_factor

        # Apply correction to knockoff coefficients
        scale_factor = self._compute_scale_factor(
            self.sigma_hat_, alpha_used, self.n_samples_, self.n_features_
        )
        corrected_knockoff = (
            knockoff_coefficients_abs + correction_factor * scale_factor
        )

        # Compute test statistics and find threshold
        self.statistic_ = np.abs(base_coefficients)
        self.threshold_ = binary_search_threshold(
            self.statistic_, corrected_knockoff, self.fdr
        )

        # Get selected features
        self.selected_ = self.get_selected_features(self.statistic_, self.threshold_)
        self.n_features_selected_ = len(self.selected_)

        return self

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
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X must have {self.n_features_} features, got {X.shape[1]} features"
            )

        return X[:, self.selected_]

    def _fit_lasso_model(
        self, X: np.ndarray, y: np.ndarray, alpha: Optional[float] = None
    ) -> Tuple[Lasso, float]:
        """Fit LASSO model with given alpha or cross-validation for alpha selection."""
        if alpha is None:
            # Use cross-validation to select alpha
            # Set up LassoCV with alpha range
            lasso_cv_kwargs = {
                "cv": self.cv_folds,
                "fit_intercept": False,
                "max_iter": self.max_iter,
                "random_state": self.random_state,
            }

            if self.alpha_range is not None:
                # Use explicit alpha range, otherwise omit alphas - let LassoCV use its default
                min_log10, max_log10 = self.alpha_log_scale_range
                alphas = np.logspace(min_log10, max_log10, self.alpha_candidates_count)
                lasso_cv_kwargs["alphas"] = alphas

            lasso_cv = LassoCV(**lasso_cv_kwargs)

            lasso_cv.fit(X, y)
            # Use alpha_scale_factor * alpha_min as default (more conservative)
            alpha_selected = self.alpha_scale_factor * lasso_cv.alpha_
        else:
            alpha_selected = alpha

        # Fit final model with selected alpha
        model = Lasso(alpha=alpha_selected, fit_intercept=False, max_iter=self.max_iter)
        model.fit(X, y)

        return model, alpha_selected

    def _generate_knockoff_data(
        self,
        X: np.ndarray,
        sigma_hat: Optional[float] = None,
        residuals: Optional[np.ndarray] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate knockoff data for linear models.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Original predictor matrix.
        sigma_hat : float, optional
            Estimated noise standard deviation. Required if error_dist="normal".
        residuals : array-like, optional
            Residuals for resampling. Required if error_dist="resample".
        rng : RandomState, optional
            Random number generator. If None, uses self.sample_rng.

        Returns
        -------
        X_knockoff : ndarray of shape (n_samples, n_features)
            Knockoff predictor matrix (same as original for linear case).
        y_knockoff : ndarray of shape (n_samples,)
            Knockoff response vector.
        """
        # For linear models, we use the same X but generate null response
        X_knockoff = X.copy()

        # Use provided RNG or fall back to self.sample_rng
        random_gen = rng if rng is not None else self.sample_rng

        if self.error_dist == "normal":
            if sigma_hat is None:
                raise ValueError("sigma_hat must be provided for normal distribution")
            y_knockoff = random_gen.normal(0, sigma_hat, self.n_samples_)
        elif self.error_dist == "resample":
            if residuals is None:
                raise ValueError("residuals must be provided for resampling")
            y_knockoff = random_gen.choice(
                residuals, size=self.n_samples_, replace=True
            )
        else:
            raise ValueError("error_dist must be 'normal' or 'resample'")

        # Center the response
        y_knockoff = y_knockoff - np.mean(y_knockoff)

        return X_knockoff, y_knockoff

    def _estimate_correction_factor(
        self, X: np.ndarray, y: np.ndarray, base_coefficients: np.ndarray
    ) -> float:
        """Estimate correction factor using multiple repetitions"""
        correction_factors = []

        for b in range(self.B_reps):
            # Create positive control coefficients
            beta_real = inflate_signal(base_coefficients, self.alpha_used_, "additive")

            # Generate correction data
            y_correction = self._generate_correction_data(X, beta_real, b)

            # Find signal indices
            signal_indices = np.where(beta_real != 0)[0]

            # Fit models to correction and knockoff data
            coef_correction_abs, coef_snp_abs = self._fit_correction_models(
                X, y_correction, b
            )

            # Calculate model-specific scale factor for linear models
            scale_factor = self.sigma_hat_ * (
                self.alpha_used_ + np.sqrt(np.log(self.n_features_) / self.n_samples_)
            )

            # Binary search for correction factor
            correction = binary_search_correction_factor(
                coef_correction_abs=coef_correction_abs,
                coef_snp_abs=coef_snp_abs,
                signal_indices=signal_indices,
                fdr=self.fdr,
                initial_correction_factor=self.initial_correction_factor,
                binary_search_tol=self.binary_search_tol,
                scale_factor=scale_factor,
            )

            correction_factors.append(correction)

        # Return maximum correction factor across repetitions
        return max(correction_factors)

    def _generate_correction_data(
        self, X: np.ndarray, beta_real: np.ndarray, iteration: int
    ) -> np.ndarray:
        """Generate correction data for linear models."""
        iteration_rng = (
            np.random
            if self.random_state is None
            else np.random.RandomState(self.random_state + iteration)
        )

        if self.error_dist == "normal":
            y_correction = X @ beta_real + iteration_rng.normal(
                0, self.sigma_hat_, self.n_samples_
            )
        else:
            y_correction = X @ beta_real + iteration_rng.choice(
                self.residuals_, size=self.n_samples_, replace=True
            )

        # Standardize and center
        y_correction = y_correction - np.mean(y_correction)
        return y_correction

    def _fit_correction_models(
        self, X: np.ndarray, y_correction: np.ndarray, iteration: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit models to correction and knockoff data."""
        iteration_rng = (
            np.random
            if self.random_state is None
            else np.random.RandomState(self.random_state + iteration)
        )

        # Fit model to correction data
        X_correction = X.copy()  # Already standardized
        model_correction, _ = self._fit_lasso_model(
            X_correction, y_correction, alpha=self.alpha_used_
        )
        coef_correction_abs = np.abs(model_correction.coef_)

        # Generate knockoff data
        X_snp, y_snp = self._generate_knockoff_data(
            X, sigma_hat=self.sigma_hat_, residuals=self.residuals_, rng=iteration_rng
        )

        # Fit model to knockoff data
        model_snp, _ = self._fit_lasso_model(X_snp, y_snp, alpha=self.alpha_used_)
        coef_snp_abs = np.abs(model_snp.coef_)

        return coef_correction_abs, coef_snp_abs

    def _compute_scale_factor(
        self, sigma_hat: float, alpha_reg: float, n_samples: int, n_features: int
    ) -> float:
        """
        Compute scale factor for linear models.

        Parameters
        ----------
        sigma_hat : float
            Estimated noise standard deviation.
        alpha_reg : float
            Regularization parameter (alpha).
        n_samples : int
            Number of samples.
        n_features : int
            Number of features.

        Returns
        -------
        scale_factor : float
            Scale factor for correction.
        """
        return sigma_hat * (alpha_reg + np.sqrt(np.log(n_features) / n_samples))
