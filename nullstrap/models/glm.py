"""Nullstrap estimator for Generalized Linear Models"""
from typing import Optional, Tuple, Union, Sequence
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from ..estimator import BaseNullstrap
from ..utils.core import (binary_search_correction_factor,
                          binary_search_threshold, inflate_signal,
                          standardize_data)


class NullstrapGLM(BaseNullstrap):
    """
    Nullstrap feature selector for binary classification with FDR control using regularized logistic regression.

    Parameters
    ----------
    fdr : float, default=0.1
        Target false discovery rate.
    alpha_ : float, optional
        Regularization parameter (corresponds to lambda in glmnet). If None, selected via cross-validation.
    B_reps : int, default=2
        Number of repetitions for correction factor estimation.
    cv_folds : int, default=10
        Number of cross-validation folds for alpha selection.
    max_iter : int, default=10000
        Maximum number of iterations for LASSO optimization.
    lasso_tol : float, default=1e-10
        Convergence tolerance for LASSO optimization.
    alpha_scale_factor : float, default=0.5
        Scaling factor for selected alpha (more conservative).
    binary_search_tol : float, default=1e-10
        Convergence tolerance for binary search.
    correction_min : float, default=1e-14
        Minimum bound for correction factor search.
    family : str, default="binomial"
        GLM family. Currently only "binomial" supported.
    alphas : int or array-like, default=10
        Alphas for CV: int (number of alphas, sklearn auto-generates range) or array (explicit values).
        Internally converted to C = 1/alpha for sklearn.
    penalty : str, default="l1"
        Regularization penalty: "l1" (LASSO), "l2" (Ridge), or "elasticnet".
        Must be compatible with solver.
    solver : str, default="saga"
        Optimization algorithm: "saga", "liblinear", "lbfgs", etc.
        For L1 penalty, use "saga" or "liblinear".
    scoring : str, default="neg_log_loss"
        Scoring metric for CV: "neg_log_loss", "accuracy", "roc_auc", etc.
    l1_ratio : float, optional, default=None
        Mixing parameter for elasticnet (0 <= l1_ratio <= 1).
        Only used when penalty='elasticnet'. l1_ratio=1 is L1, l1_ratio=0 is L2.
    n_jobs : int, optional
        Number of parallel jobs for CV. None=1, -1=all cores.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    threshold_ : float
        Selection threshold.
    selected_ : ndarray
        Indices of selected features.
    n_features_selected_ : int
        Number of selected features.
    statistic_ : ndarray
        Test statistics (absolute coefficients).
    alpha_used_ : float
        Regularization parameter used.
    correction_factor_ : float
        Correction factor for FDR control.
    """

    def __init__(
        self,
        fdr: float = 0.1,
        alpha_: Optional[float] = None,
        B_reps: int = 2,
        cv_folds: int = 10,
        max_iter: int = 10000,
        lasso_tol: float = 1e-10,
        alpha_scale_factor: float = 0.5,
        binary_search_tol: float = 1e-8,
        correction_min: float = 1e-14,
        family: str = "binomial",
        alphas: Union[int, Sequence[float]] = 10,
        penalty: str = "l1",
        solver: str = "saga",
        scoring: str = "neg_log_loss",
        l1_ratio: Optional[float] = None,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        super().__init__(fdr=fdr, alpha_=alpha_, B_reps=B_reps, random_state=random_state)
        self.cv_folds = cv_folds
        self.max_iter = max_iter
        self.lasso_tol = lasso_tol
        self.alpha_scale_factor = alpha_scale_factor
        self.binary_search_tol = binary_search_tol
        self.correction_min = correction_min
        self.family = family
        self.alphas = alphas
        self.penalty = penalty
        self.solver = solver
        self.scoring = scoring
        self.l1_ratio = l1_ratio
        self.n_jobs = n_jobs

        # Initialize dedicated random number generator for sampling operations
        self.sample_rng = np.random.RandomState(random_state)

    def _validate_parameters(self) -> None:
        """
        Validate GLM-specific parameters. Checks family, penalty, and solver compatibility.
        """
        # Call base class validation first
        super()._validate_parameters()
        
        # Validate family
        if self.family != "binomial":
            raise ValueError("Currently only 'binomial' family is supported")
        
        # Validate penalty
        valid_penalties = ["l1", "l2", "elasticnet"]
        if self.penalty not in valid_penalties:
            raise ValueError(f"penalty must be one of {valid_penalties}, got '{self.penalty}'")
        
        # Validate solver and penalty compatibility
        if self.penalty == "l1":
            valid_l1_solvers = ["saga", "liblinear"]
            if self.solver not in valid_l1_solvers:
                raise ValueError(
                    f"For L1 penalty, solver must be one of {valid_l1_solvers}, got '{self.solver}'. "
                    f"Consider using solver='saga' (recommended) or solver='liblinear'."
                )
        elif self.penalty == "elasticnet":
            if self.solver != "saga":
                raise ValueError(f"For elasticnet penalty, solver must be 'saga', got '{self.solver}'")
            if self.l1_ratio is None:
                raise ValueError("For elasticnet penalty, l1_ratio must be specified")
            if not 0 <= self.l1_ratio <= 1:
                raise ValueError(f"l1_ratio must be between 0 and 1, got {self.l1_ratio}")
        
        # Validate l1_ratio is only used with elasticnet
        if self.l1_ratio is not None and self.penalty != "elasticnet":
            raise ValueError(
                f"l1_ratio parameter is only used with penalty='elasticnet', got penalty='{self.penalty}'. "
                f"Either set penalty='elasticnet' or remove l1_ratio parameter."
            )

    def _validate_data(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Validate input data for GLM models. Checks that y contains only binary values (0 and 1).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Binary target values. Required for GLM models.
        """
        super()._validate_X(X)
        
        if y is None:
            raise ValueError("GLM models require y parameter")
        
        super()._validate_y(y)
        super()._validate_sample_sizes(X, y)
        
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NullstrapGLM":
        """
        Fit the Nullstrap GLM estimator for binary classification.

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
        # Validate parameters and input data
        self._validate_parameters()
        self._validate_data(X, y)
        self._set_random_state()

        self.n_samples_, self.n_features_ = X.shape

        # Standardize data for GLM models (z-score + sample size normalization)
        X_scaled, _ = standardize_data(X, y=None, scale_by_sample_size=True, n_samples=self.n_samples_)

        # Fit base logistic regression model
        model_base, alpha_used = self._fit_lasso_model(X_scaled, y, alpha=self.alpha_)
        self.alpha_used_ = alpha_used

        # Get base coefficients
        coef_base = model_base.coef_.flatten() # returns a 2D array (n_classes, n_features), thus flatten it

        # Generate knockoff data, fit model to knockoff data
        y_knockoff = self._generate_synthetic_data(X_scaled, beta=None, family=self.family)
        model_knockoff, _ = self._fit_lasso_model(X_scaled, y_knockoff, alpha=self.alpha_used_)
        coef_knockoff = model_knockoff.coef_.flatten()

        # Estimate correction factor
        correction_factor = self._estimate_correction_factor(X_scaled, y, coef_base)
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
    ) -> Tuple[LogisticRegression, float]:
        """
        Fit LASSO logistic regression with specified or cross-validated regularization parameter.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Predictor matrix (should be standardized).
        y : ndarray of shape (n_samples,)
            Binary response vector (0 or 1).
        alpha : float, optional
            Regularization parameter (corresponds to lambda in glmnet).
            If None, selected via cross-validation.

        Returns
        -------
        model : LogisticRegression
            Fitted LASSO logistic regression model.
        alpha_used : float
            Regularization parameter used in final fit, scaled by alpha_scale_factor
            (default 0.5) for conservative selection.

        Notes
        -----
        Uses LogisticRegressionCV when alpha is None. If self.alphas is an array, uses those
        values for CV; if integer, generates that many alphas automatically.
        
        sklearn uses C = 1/alpha for regularization strength (inverse relationship), which is
        handled internally. The model is fitted without intercept as data should be pre-standardized.

        See Also
        --------
        sklearn.linear_model.LogisticRegressionCV : Cross-validated logistic regression.
        sklearn.linear_model.LogisticRegression : Logistic regression implementation.
        """
        if alpha is None:
            # Set up LogisticRegressionCV for alpha selection via cross-validation
            logit_cv_kwargs = {
                "cv": self.cv_folds,
                "fit_intercept": False,
                "max_iter": self.max_iter,
                "tol": self.lasso_tol,
                "random_state": self.random_state,
                "penalty": self.penalty,
                "solver": self.solver,
                "scoring": self.scoring,
                "n_jobs": self.n_jobs,
            }

            # Add l1_ratio for elasticnet
            if self.penalty == "elasticnet":
                logit_cv_kwargs["l1_ratios"] = [self.l1_ratio]

            # Convert alphas to Cs (sklearn uses C = 1/alpha)
            if isinstance(self.alphas, int):
                logit_cv_kwargs["Cs"] = self.alphas
            else:
                logit_cv_kwargs["Cs"] = 1.0 / np.array(self.alphas)

            logit_cv = LogisticRegressionCV(**logit_cv_kwargs)

            logit_cv.fit(X, y)
            # R implementation uses 0.5 * alpha.1se, we approximate this
            C_selected = logit_cv.C_[0]  # Best C from CV
            alpha_used = 1.0 / (self.alpha_scale_factor * C_selected)  # More conservative
        else:
            alpha_used = alpha
            C_selected = 1.0 / alpha_used

        # Fit final model with selected regularization
        logit_kwargs = {
            "C": C_selected,
            "fit_intercept": False,
            "max_iter": self.max_iter,
            "tol": self.lasso_tol,
            "random_state": self.random_state,
            "penalty": self.penalty,
            "solver": self.solver,
        }
        
        # Add l1_ratio for elasticnet
        if self.penalty == "elasticnet":
            logit_kwargs["l1_ratio"] = self.l1_ratio
        
        model = LogisticRegression(**logit_kwargs)
        model.fit(X, y)

        return model, alpha_used

    def _generate_synthetic_data(
        self,
        X: np.ndarray,
        beta: Optional[np.ndarray] = None,
        family: str = "binomial",
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        """
        Generate synthetic response data for generalized linear models (binomial family).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Predictor matrix (should be standardized).
        beta : ndarray of shape (n_features,), optional
            Coefficients for signal. If None, generates null data (knockoff) with p=0.5 for all samples.
        family : str, default="binomial"
            GLM family type. Currently only "binomial" is supported.
        rng : np.random.RandomState, optional
            Random number generator for reproducibility. If None, uses self.sample_rng.

        Returns
        -------
        y_synthetic : ndarray of shape (n_samples,)
            Binary synthetic response vector (0s and 1s).
        """
        # Use provided RNG or fall back to self.sample_rng
        random_gen = self.sample_rng if rng is None else rng

        if family == "binomial":
            # Compute linear predictor (0 for null data, X @ beta for signal data)
            linear_predictor = np.zeros(X.shape[0]) if beta is None else X @ beta
            
            # Compute probability via sigmoid (clip to prevent overflow in exp)
            prob = 1.0 / (1.0 + np.exp(-np.clip(linear_predictor, -500, 500)))
            
            # Generate binomial response
            y_synthetic = random_gen.binomial(1, prob)
        else:
            raise ValueError(f"Unsupported GLM family: {family}")

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
            Response vector.
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
            y_augmented = self._generate_synthetic_data(X, beta=beta_augmented, family=self.family, rng=iteration_rng)
            y_knockoff  = self._generate_synthetic_data(X, beta=None, family=self.family, rng=iteration_rng)

            # Fit models to augmented and knockoff data (estimate coefficients)
            model_augmented, _ = self._fit_lasso_model(X, y_augmented, alpha=self.alpha_used_)
            model_knockoff, _  = self._fit_lasso_model(X, y_knockoff, alpha=self.alpha_used_)
            coef_augmented = model_augmented.coef_.flatten()
            coef_knockoff = model_knockoff.coef_.flatten()
            
            # Calculate model-specific scale factor for GLM models
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
        Compute scale factor for correction (GLM-specific).

        Returns
        -------
        float
            Scale factor for knockoff correction. For GLM, returns 1.0 as scale
            is inherently normalized in the logistic model.
        
        Notes
        -----
        Unlike linear models which use sigma_hat * (alpha + sqrt(log(p) / n)),
        GLM uses unit scale as the binomial family naturally normalizes the coefficients.
        """
        return 1.0
