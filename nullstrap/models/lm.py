"""Nullstrap estimator for Linear Models"""
from typing import Optional, Tuple, Union, Sequence

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
        Regularization parameter (corresponds to lambda in glmnet). If None, selected via cross-validation.
    B_reps : int, default=5
        Number of repetitions for correction factor estimation.
    cv_folds : int, default=10
        Number of cross-validation folds for alpha selection.
    max_iter : int, default=10000
        Maximum number of iterations for LASSO optimization.
    lasso_tol : float, default=1e-7
        Convergence tolerance for LASSO optimization.
    alpha_scale_factor : float, default=0.5
        Scaling factor for selected alpha (more conservative).
    binary_search_tol : float, default=1e-8
        Convergence tolerance for binary search.
    correction_min : float, default=0.05
        Minimum bound for correction factor search.
    error_dist : str, default="normal"
        Error distribution for knockoff data generation: "normal" or "resample".
    alphas : int or array-like, default=100
       Alphas for CV: int (number of alphas, with eps determining range) or array (explicit values).
        Matches sklearn's LassoCV parameter (1.7+) and glmnet's nlambda.
    eps : float, default=1e-3
        Length of the path. eps=1e-3 means alpha_min/alpha_max = 1e-3. 
        Only used when alphas is an integer. Matches glmnet's lambda.min.ratio.
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
    residuals_ : ndarray
        Scaled residuals from the base model fit, used for resampling.
    """

    def __init__(
        self,
        fdr: float = 0.1,
        alpha_: Optional[float] = None,
        B_reps: int = 5,
        cv_folds: int = 10,
        max_iter: int = 10000,
        lasso_tol: float = 1e-7,
        alpha_scale_factor: float = 0.5,
        binary_search_tol: float = 1e-8,
        correction_min: float = 0.05,
        error_dist: str = "normal",
        alphas: Union[int, Sequence[float]] = 100,
        eps: float = 1e-3,
        random_state: Optional[int] = None,
    ):
        super().__init__(fdr=fdr, alpha_=alpha_, B_reps=B_reps, random_state=random_state)
        self.cv_folds = cv_folds
        self.max_iter = max_iter
        self.lasso_tol = lasso_tol
        self.alpha_scale_factor = alpha_scale_factor
        self.binary_search_tol = binary_search_tol
        self.correction_min = correction_min
        self.error_dist = error_dist
        self.alphas = alphas
        self.eps = eps

        # Initialize dedicated random number generator for sampling operations
        self.sample_rng = np.random.RandomState(random_state)

        # Additional attributes for linear models
        self.sigma_hat_: Optional[float] = None
        self.residuals_: Optional[np.ndarray] = None

    def _validate_parameters(self) -> None:
        """
        Validate LM-specific parameters. Checks that error_dist is valid.
        """
        # Call base class validation first
        super()._validate_parameters()
        
        # Validate error_dist parameter
        if self.error_dist not in ["normal", "resample"]:
            raise ValueError("error_dist must be 'normal' or 'resample'")

    def _validate_data(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Validate input data for linear models.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values. Required for linear models.
        """
        super()._validate_X(X)
        
        if y is None:
            raise ValueError("Linear models require y parameter")
        
        super()._validate_y(y)
        super()._validate_sample_sizes(X, y)

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
        # Validate parameters and input data
        self._validate_parameters()
        self._validate_data(X, y)
        self._set_random_state()

        self.n_samples_, self.n_features_ = X.shape

        # Standardize data for linear models (standard z-score only)
        X_scaled, y_centered = standardize_data(X, y, scale_by_sample_size=False)

        # Fit base LASSO model and compute residuals
        model_base, alpha_used = self._fit_lasso_model(X_scaled, y_centered, alpha=self.alpha_)
        self.alpha_used_ = alpha_used
        
        # Get base coefficients and compute residuals
        coef_base = model_base.coef_
        residuals = y_centered - model_base.predict(X_scaled)

        # Estimate noise standard deviation and scale residuals for resampling
        n_nonzero = np.sum(coef_base != 0)
        self.sigma_hat_ = np.sqrt(np.sum(residuals**2) / max(1, self.n_samples_ - n_nonzero))
        
        # Scale residuals for resampling
        scaling_factor = min(
            self.n_samples_ ** (1 / 4),
            np.sqrt(self.n_samples_ / (self.n_samples_ - n_nonzero))
        )
        self.residuals_ = scaling_factor * residuals

        # Generate knockoff data, fit model to knockoff data
        y_knockoff = self._generate_synthetic_data(X_scaled, beta=None)
        model_knockoff, _ = self._fit_lasso_model(X_scaled, y_knockoff, alpha=self.alpha_used_)
        coef_knockoff = model_knockoff.coef_

        # Estimate correction factor
        correction_factor = self._estimate_correction_factor(X_scaled, y_centered, coef_base)
        self.correction_factor_ = correction_factor
        self.statistic_ = np.abs(coef_base)

        # Apply correction to knockoff coefficients
        scale_factor = self._compute_scale_factor()
        corrected_knockoff = np.abs(coef_knockoff) + correction_factor * scale_factor

        # Compute test statistics and find threshold
        self.threshold_ = binary_search_threshold(
            self.statistic_,
            corrected_knockoff,
            self.fdr,
            max_iter=self.max_iter,
            tol=self.binary_search_tol
        )

        # Get selected features
        self.selected_ = self.get_selected_features(self.statistic_, self.threshold_)
        self.n_features_selected_ = len(self.selected_)

        return self

    def _fit_lasso_model(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        alpha: Optional[float] = None
    ) -> Tuple[Lasso, float]:
        """
        Fit LASSO with specified or cross-validated regularization parameter.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Predictor matrix (standardized).
        y : ndarray of shape (n_samples,)
            Response vector (standardized).
        alpha : float, optional
            Regularization parameter (corresponds to lambda in glmnet).
            If None, selected via CV.

        Returns
        -------
        model : Lasso
            Fitted LASSO model.
        alpha_used : float
            Regularization parameter used, scaled by alpha_scale_factor (default 0.5)
            for conservative selection.

        Notes
        -----
        Uses LassoCV when alpha is None. If self.alphas is an array, uses those values
        for CV; if integer, generates that many alphas automatically using eps ratio.
        Model fitted without intercept as data should be pre-standardized.
        """
        if alpha is None:
            # Set up LassoCV for alpha selection via cross-validation
            lasso_cv_kwargs = {
                "cv": self.cv_folds,
                "fit_intercept": False,
                "max_iter": self.max_iter,
                "tol": self.lasso_tol,
                "random_state": self.random_state,
            }

            # Handle alphas parameter (flexible: int or array)
            lasso_cv_kwargs["alphas"] = self.alphas
            # Only add eps if alphas is an integer (for automatic generation)
            if isinstance(self.alphas, int):
                lasso_cv_kwargs["eps"] = self.eps

            lasso_cv = LassoCV(**lasso_cv_kwargs)

            lasso_cv.fit(X, y)
            # Scale selected alpha by alpha_scale_factor (more conservative)
            alpha_used = self.alpha_scale_factor * lasso_cv.alpha_
        else:
            alpha_used = alpha

        # Fit final model with selected alpha
        model = Lasso(alpha=alpha_used, fit_intercept=False, max_iter=self.max_iter)
        model.fit(X, y)

        return model, alpha_used

    def _generate_synthetic_data(
        self,
        X: np.ndarray,
        beta: Optional[np.ndarray] = None,
        center: bool = True,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        """
        Generate synthetic response: y = X @ beta + noise (centered if requested).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Predictor matrix.
        beta : ndarray of shape (n_features,), optional
            Signal coefficients. If None, generates null/knockoff data (noise only).
        center : bool, default=True
            Whether to center the response.
        rng : np.random.RandomState, optional
            Random generator. If None, uses self.sample_rng.

        Returns
        -------
        ndarray of shape (n_samples,)
            Synthetic response vector.
        """
        random_gen = self.sample_rng if rng is None else rng

        # Generate noise component
        if self.error_dist == "normal":
            noise = random_gen.normal(0, self.sigma_hat_, self.n_samples_)
        elif self.error_dist == "resample":
            noise = random_gen.choice(self.residuals_, size=self.n_samples_, replace=True)
        else:
            raise ValueError("error_dist must be 'normal' or 'resample'")

        # Add signal if beta is provided
        signal = 0 if beta is None else X @ beta
        y_synthetic = signal + noise
        
        # Center the response as needed
        y_synthetic = y_synthetic - np.mean(y_synthetic) if center else y_synthetic

        return y_synthetic

    def _estimate_correction_factor(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        coef_base: np.ndarray
    ) -> float:
        """
        Estimate correction factor for FDR control using bootstrap repetitions.

        Creates positive control (augmented) and knockoff data, fits models to both,
        and uses binary search to find correction factor. Repeats B_reps times and
        returns maximum for conservative FDR control.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Predictor matrix (standardized).
        y : ndarray of shape (n_samples,)
            Response vector (standardized).
        coef_base : ndarray of shape (n_features,)
            Coefficients from original model fit.

        Returns
        -------
        correction_factor : float
            Maximum correction factor across repetitions, used to adjust knockoff
            statistics for FDR control.
        """
        correction_factors = []

        for b in range(self.B_reps):
            # Create positive control coefficients (synthetic truth), get signal indices
            beta_augmented = inflate_signal(coef_base, self.alpha_used_, "additive")
            signal_indices = np.where(beta_augmented != 0)[0]

            # Create iteration-specific RNG
            iteration_rng = None if self.random_state is None else np.random.RandomState(self.random_state + b)

            # Generate augmented and knockoff data
            y_augmented = self._generate_synthetic_data(X, beta=beta_augmented, rng=iteration_rng)
            y_knockoff  = self._generate_synthetic_data(X, beta=None, rng=iteration_rng)

            # Fit models to augmented and knockoff data (estimate coefficients)
            model_augmented, _ = self._fit_lasso_model(X, y_augmented, alpha=self.alpha_used_)
            model_knockoff, _  = self._fit_lasso_model(X, y_knockoff, alpha=self.alpha_used_)
            coef_augmented = model_augmented.coef_
            coef_knockoff = model_knockoff.coef_
            
            # Calculate model-specific scale factor for linear models
            scale_factor = self._compute_scale_factor()
            correction_min = self.correction_min
            correction_max = np.max(np.abs(coef_augmented)) / scale_factor

            # Binary search for correction factor
            correction = binary_search_correction_factor(
                coef_augmented_abs=np.abs(coef_augmented),
                coef_knockoff_abs=np.abs(coef_knockoff),
                signal_indices=signal_indices,
                fdr=self.fdr,
                scale_factor=scale_factor,
                correction_min=correction_min,
                correction_max=correction_max,
                max_iter=self.max_iter,
                tol=self.binary_search_tol
            )

            correction_factors.append(correction)

        # Return maximum correction factor across repetitions
        return max(correction_factors)

    def _compute_scale_factor(self) -> float:
        """
        Compute scale factor for correction: sigma_hat * (alpha + sqrt(log(p) / n)).

        Returns
        -------
        float
            Scale factor for knockoff correction.
        
        Notes
        -----
        Uses instance attributes: sigma_hat_, alpha_used_, n_samples_, n_features_.
        """
        return self.sigma_hat_ * (self.alpha_used_ + np.sqrt(np.log(self.n_features_) / self.n_samples_))
