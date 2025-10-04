"""Nullstrap estimator for Generalized Linear Models"""
import warnings
from typing import Optional, Tuple, Any

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from ..estimator import BaseNullstrap
from ..utils.core import (binary_search_correction_factor,
                          binary_search_threshold, inflate_signal,
                          standardize_data)


class NullstrapGLM(BaseNullstrap):
    """
    Nullstrap feature selector for generalized linear models with FDR control, using LASSO regularization.

    Parameters
    ----------
    fdr : float, default=0.1
        Target false discovery rate.
    alpha_ : float, optional
        LASSO regularization parameter (maps to 'lambda' in glmnet). If None, selected by cross-validation.
    B_reps : int, default=5
        Number of repetitions for correction factor estimation.
    family : str, default="binomial"
        GLM family. Currently only "binomial" is supported.
    cv_folds : int, default=10
        Cross-validation folds for alpha selection.
    max_iter : int, default=10000
        Maximum number of iterations for LogisticRegression optimization.
    alpha_scale_factor : float, default=0.5
        Scaling factor for alpha selection (0.5 * alpha_min for conservative selection).
    binary_search_tol : float, default=1e-10
        Tolerance for binary search convergence in correction factor estimation.
    initial_correction_factor : float, default=1e-14
        Initial left bound for binary search of correction factor.
    convergence_tol : float, default=1e-10
        Convergence tolerance for LogisticRegression (matches R thresh=1e-10).
    prob_clip_bounds : float, default=1e-15
        Probability clipping bounds to avoid extreme values in binomial sampling.
    alpha_candidates_count : int, default=50
        Number of alpha candidates for cross-validation selection. Only used if alpha_range is not None.
    alpha_range : tuple, default=None
        Range of alpha values for cross-validation selection (min_alpha, max_alpha).
        If None, uses LogisticRegressionCV default range.
    alpha_log_scale_range : tuple, default=(-4, 2)
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
        Test statistics (absolute GLM coefficients).
    alpha_used_ : float
        Regularization parameter used in final fit.
    correction_factor_ : float
        Estimated correction factor for FDR control.
    """

    def __init__(
        self,
        fdr: float = 0.1,
        alpha_: Optional[float] = None,
        B_reps: int = 5,
        family: str = "binomial",
        cv_folds: int = 10,
        max_iter: int = 10000,
        alpha_scale_factor: float = 0.5,
        binary_search_tol: float = 1e-10,
        initial_correction_factor: float = 1e-14,
        convergence_tol: float = 1e-10,
        prob_clip_bounds: float = 1e-15,
        alpha_candidates_count: int = 50,
        alpha_range: Optional[Tuple[float, float]] = None,
        alpha_log_scale_range: Tuple[float, float] = (-4, 2),
        random_state: Optional[int] = None,
    ):
        super().__init__(
            fdr=fdr, alpha_=alpha_, B_reps=B_reps, random_state=random_state
        )
        self.family = family
        self.cv_folds = cv_folds
        self.max_iter = max_iter
        self.alpha_scale_factor = alpha_scale_factor
        self.binary_search_tol = binary_search_tol
        self.initial_correction_factor = initial_correction_factor
        self.convergence_tol = convergence_tol
        self.prob_clip_bounds = prob_clip_bounds
        self.alpha_candidates_count = alpha_candidates_count
        self.alpha_range = alpha_range
        self.alpha_log_scale_range = alpha_log_scale_range

        if family != "binomial":
            raise ValueError("Currently only 'binomial' family is supported")

        # Initialize dedicated random number generator for sampling operations
        self.sample_rng = np.random.RandomState(random_state)

        # Additional attributes
        self.base_model_: Optional[Any] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NullstrapGLM":
        """
        Fit the Nullstrap GLM estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Binary target values (0 or 1).

        Returns
        -------
        self : NullstrapGLM
            Fitted estimator.
        """
        self._validate_parameters()
        self._set_random_state()

        # Validate and convert data
        X, y = self._validate_data(X, y)

        self.n_samples_, self.n_features_ = X.shape

        # Standardize data for GLM models (z-score + sample size normalization)
        X_scaled, _ = standardize_data(
            X, n_samples=self.n_samples_, scale_by_sample_size=True
        )

        # Fit base logistic regression model
        base_model, alpha_used = self._fit_lasso_model(X_scaled, y)
        self.base_model_ = base_model
        self.alpha_used_ = alpha_used

        # Get base coefficients
        base_coefficients = base_model.coef_.flatten()

        # Generate knockoff data
        X_knockoff, y_knockoff = self._generate_knockoff_data(X_scaled, self.family)

        # Fit model to knockoff data
        # Match R implementation: glmnet with thresh=1e-10
        C_knockoff = 1.0 / alpha_used if alpha_used > 0 else 1.0
        knockoff_model = LogisticRegression(
            C=C_knockoff,
            penalty="l1",
            solver="liblinear",
            fit_intercept=False,
            max_iter=self.max_iter,
            tol=self.convergence_tol,
            random_state=self.random_state,
        )
        knockoff_model.fit(X_knockoff, y_knockoff)
        knockoff_coefficients_abs = np.abs(knockoff_model.coef_.flatten())

        # Estimate correction factor
        correction_factor = self._estimate_correction_factor(
            X_scaled, y, base_coefficients
        )
        self.correction_factor_ = correction_factor

        # Apply correction to knockoff coefficients
        scale_factor = self._compute_scale_factor(
            1.0, alpha_used, self.n_samples_, self.n_features_
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

    def _validate_data(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Validate input data for GLM models.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Binary target values (0 and 1). Required for GLM models.

        Returns
        -------
        X : ndarray
            Validated and converted X.
        y : ndarray
            Validated and converted y.
        """
        X = super()._validate_X(X)
        
        if y is None:
            raise ValueError("GLM models require y parameter")
        
        y = super()._validate_y(y)
        
        # Joint validation: same samples
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same number of samples. X has {X.shape[0]}, y has {y.shape[0]}")
        
        # GLM-specific validation
        unique_y = np.unique(y)
        if not np.all(np.isin(unique_y, [0, 1])):
            raise ValueError("y must contain binary values (0 and 1)")
        
        # Warn about single class (all 0s or all 1s)
        if len(unique_y) == 1:
            warnings.warn(
                f"Single class detected in y (only class {unique_y[0]}). "
                "This may lead to poor model performance and meaningless predictions.",
                UserWarning,
                stacklevel=2,
            )
        
        return X, y

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
    ) -> Tuple[LogisticRegression, float]:
        """Fit LASSO model with given alpha or cross-validation for alpha selection."""
        if alpha is None:
            # Use cross-validation to select regularization strength
            # Match R implementation: cv.glmnet with measure="mse"
            # Note: sklearn uses C = 1/alpha, so we need to invert

            # Set up LogisticRegressionCV with C range
            logit_cv_kwargs = {
                "cv": self.cv_folds,
                "penalty": "l1",
                "solver": "liblinear",
                "fit_intercept": False,
                "max_iter": self.max_iter,
                "scoring": "neg_mean_squared_error",
                "random_state": self.random_state,
            }

            if self.alpha_range is not None:
                # Use explicit alpha range, otherwise omit Cs - let LogisticRegressionCV use its default
                min_log10, max_log10 = self.alpha_log_scale_range
                Cs = np.logspace(min_log10, max_log10, self.alpha_candidates_count)
                logit_cv_kwargs["Cs"] = Cs

            logit_cv = LogisticRegressionCV(**logit_cv_kwargs)
            logit_cv.fit(X, y)

            # Convert back to alpha and use more conservative choice
            # R implementation uses 0.5 * alpha.1se, we approximate this
            C_selected = logit_cv.C_[0]  # Best C from CV
            alpha_selected = 1.0 / (
                self.alpha_scale_factor * C_selected
            )  # More conservative
        else:
            alpha_selected = alpha
            C_selected = 1.0 / alpha_selected

        # Fit final model with selected regularization
        # Match R implementation: glmnet with thresh=1e-10
        base_model = LogisticRegression(
            C=C_selected,
            penalty="l1",
            solver="liblinear",
            fit_intercept=False,
            max_iter=self.max_iter,
            tol=self.convergence_tol,
            random_state=self.random_state,
        )
        base_model.fit(X, y)

        return base_model, alpha_selected

    def _generate_knockoff_data(
        self,
        X: np.ndarray,
        family: str = "binomial",
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate knockoff data for generalized linear models.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Original predictor matrix.
        family : str, default="binomial"
            GLM family type.
        rng : RandomState, optional
            Random number generator. If None, uses self.sample_rng.

        Returns
        -------
        X_knockoff : ndarray of shape (n_samples, n_features)
            Knockoff predictor matrix (same as original for linear case).
        y_knockoff : ndarray of shape (n_samples,)
            Knockoff response vector.
        """
        n_samples, n_features = X.shape

        X_knockoff = X.copy()

        # Use provided RNG or fall back to self.sample_rng
        random_gen = rng if rng is not None else self.sample_rng

        if family == "binomial":
            beta_0 = np.zeros(n_features)

            linear_predictor = X @ beta_0

            prob = 1.0 / (1.0 + np.exp(-linear_predictor))

            y_knockoff = random_gen.binomial(1, prob)
        else:
            raise ValueError(f"Unsupported GLM family: {family}")

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

            # GLM uses simple scale factor of 1.0 (no additional scaling)
            scale_factor = 1.0

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
        """Generate correction data for generalized linear models."""
        iteration_rng = (
            np.random
            if self.random_state is None
            else np.random.RandomState(self.random_state + iteration)
        )

        # Generate positive control data
        linear_predictor = X @ beta_real
        prob = 1.0 / (1.0 + np.exp(-linear_predictor))
        prob = np.clip(
            prob, self.prob_clip_bounds, 1 - self.prob_clip_bounds
        )  # Avoid extreme probabilities
        y_correction = iteration_rng.binomial(1, prob)

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
        model_correction, _ = self._fit_lasso_model(X, y_correction)
        coef_correction_abs = np.abs(model_correction.coef_.flatten())

        # Generate knockoff data
        X_snp, y_snp = self._generate_knockoff_data(
            X, family=self.family, rng=iteration_rng
        )

        # Fit model to knockoff data
        model_snp, _ = self._fit_lasso_model(X_snp, y_snp)
        coef_snp_abs = np.abs(model_snp.coef_.flatten())

        return coef_correction_abs, coef_snp_abs

    def _compute_scale_factor(
        self, sigma_hat: float, alpha_reg: float, n_samples: int, n_features: int
    ) -> float:
        """
        Compute scale factor for generalized linear models.

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
        return alpha_reg + np.sqrt(np.log(n_features) / n_samples)
