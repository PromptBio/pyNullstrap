"""Nullstrap estimator for Cox proportional hazards models"""
import warnings
from typing import Optional, Tuple, Any

import numpy as np
from sklearn.linear_model import Lasso, LassoCV

try:
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored

    SKLEARN_SURVIVAL_AVAILABLE = True
except ImportError:
    SKLEARN_SURVIVAL_AVAILABLE = False
    warnings.warn(
        "scikit-survival is not available. Cox models will not work. "
        "Install with: pip install scikit-survival"
    )

from ..estimator import BaseNullstrap
from ..utils.core import (binary_search_correction_factor,
                          binary_search_threshold, inflate_signal,
                          standardize_data)


class NullstrapCox(BaseNullstrap):
    """
    Nullstrap feature selector for Cox proportional hazards models with FDR control, using LASSO regularization.

    This estimator expects survival data in scikit-survival format: a structured NumPy array
    with 'event' (boolean) and 'time' (float) fields. See the fit() method for detailed examples.

    Parameters
    ----------
    fdr : float, default=0.1
        Target false discovery rate.
    alpha_ : float, optional
        LASSO regularization parameter (maps to 'lambda' in glmnet). If None, selected by cross-validation.
    B_reps : int, default=5
        Number of repetitions for correction factor estimation.
    cv_folds : int, default=10
        Cross-validation folds for alpha selection.
    max_iter : int, default=10000
        Maximum number of iterations for Lasso optimization.
    alpha_scale_factor : float, default=0.5
        Scaling factor for alpha selection (0.5 * alpha_min for conservative selection).
    binary_search_tol : float, default=1e-8
        Tolerance for binary search convergence in correction factor estimation.
    initial_correction_factor : float, default=0.05
        Initial left bound for binary search of correction factor.
    feature_selection_threshold : float, default=1e-6
        Threshold for feature selection in Lasso coefficients.
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
        Test statistics (absolute Cox coefficients).
    alpha_used_ : float
        Regularization parameter used in final fit.
    correction_factor_ : float
        Estimated correction factor for FDR control.
    baseline_hazard_ : callable
        Estimated baseline hazard function.
    event_times_ : ndarray
        Survival/censoring times from training data.
    event_indicators_ : ndarray
        Boolean event indicators from training data (True=event, False=censored).
    """

    def __init__(
        self,
        fdr: float = 0.1,
        alpha_: Optional[float] = None,
        B_reps: int = 5,
        cv_folds: int = 10,
        max_iter: int = 10000,
        alpha_scale_factor: float = 0.5,
        binary_search_tol: float = 1e-8,
        initial_correction_factor: float = 0.05,
        feature_selection_threshold: float = 1e-6,
        alpha_candidates_count: int = 50,
        alpha_range: Optional[Tuple[float, float]] = None,
        alpha_log_scale_range: Tuple[float, float] = (-4, 1),
        random_state: Optional[int] = None,
    ):
        super().__init__(
            fdr=fdr, alpha_=alpha_, B_reps=B_reps, random_state=random_state
        )
        self.cv_folds = cv_folds
        self.max_iter = max_iter
        self.alpha_scale_factor = alpha_scale_factor
        self.binary_search_tol = binary_search_tol
        self.initial_correction_factor = initial_correction_factor
        self.feature_selection_threshold = feature_selection_threshold
        self.alpha_candidates_count = alpha_candidates_count
        self.alpha_range = alpha_range
        self.alpha_log_scale_range = alpha_log_scale_range

        if not SKLEARN_SURVIVAL_AVAILABLE:
            raise ImportError(
                "scikit-survival is required for Cox models. Install with: pip install scikit-survival"
            )

        # Initialize dedicated random number generator for sampling operations
        self.sample_rng = np.random.RandomState(random_state)

        # Additional attributes
        self.base_model_: Optional[Any] = None
        self.baseline_hazard_: Optional[np.ndarray] = None
        self.time_range_: Optional[Tuple[float, float]] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NullstrapCox":
        """
        Fit the Nullstrap Cox estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : structured array of shape (n_samples,)
            Survival data in scikit-survival format. Must be a structured NumPy array with two fields:
            - 'event': boolean, True if event occurred, False if censored
            - 'time': float, time to event or censoring

            Example:
            ```python
            import numpy as np

            # Create survival data
            events = np.array([True, True, False, True])  # boolean
            times = np.array([5.2, 1.1, 8.0, 2.3])      # float

            # Create structured array
            y = np.array([(event, time) for event, time in zip(events, times)],
                        dtype=[('event', bool), ('time', float)])
            ```

        Returns
        -------
        self : NullstrapCox
            Fitted estimator.
        """
        self._validate_parameters()
        self._set_random_state()

        # Validate and convert data
        X, y = self._validate_data(X, y)

        # y should already be in scikit-survival format (structured array)
        # No conversion needed - scikit-survival expects structured array with boolean events

        self.n_samples_, self.n_features_ = X.shape

        # Standardize data for Cox models (z-score + sample size normalization)
        X_scaled, _ = standardize_data(
            X, n_samples=self.n_samples_, scale_by_sample_size=True
        )

        # Store time range for simulation and survival data
        times = y["time"]
        events = y["event"]
        self.time_range_ = (np.min(times), np.max(times))

        # Store event times and indicators for easy access
        self.event_times_ = times
        self.event_indicators_ = events

        # Fit base Cox model
        base_model, alpha_used = self._fit_lasso_model(X_scaled, y)
        self.base_model_ = base_model
        self.alpha_used_ = alpha_used

        # Get base coefficients
        base_coefficients = self._get_coefficients(base_model)

        # Estimate baseline hazard function
        self.baseline_hazard_ = self._estimate_baseline_hazard(
            X_scaled, y, base_coefficients
        )

        # Generate knockoff data
        X_knockoff, y_knockoff = self._generate_knockoff_data(
            X_scaled, self.baseline_hazard_, self.time_range_
        )

        # Fit model to knockoff data
        knockoff_model = self._fit_cox_model(X_knockoff, y_knockoff, alpha_used)
        knockoff_coefficients_abs = np.abs(self._get_coefficients(knockoff_model))

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
        Validate input data for Cox survival models.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : structured array of shape (n_samples,)
            Survival data in scikit-survival format with 'event' and 'time' fields.
            Required for Cox models.

        Returns
        -------
        X : ndarray
            Validated and converted X.
        y : structured array
            Validated survival data.
        """
        X = super()._validate_X(X)
        
        if y is None:
            raise ValueError("Cox models require y parameter")
        
        y = self._validate_y(y)
        
        # Joint validation: same samples
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same number of samples. X has {X.shape[0]}, y has {y.shape[0]}")
        
        return X, y

    def _validate_y(self, y: np.ndarray) -> np.ndarray:
        """Validate survival data y for Cox models."""
        y = np.asarray(y)
        
        # Format detection and conversion
        if (y.dtype.names is not None and 
            'event' in y.dtype.names and 'time' in y.dtype.names and
            np.issubdtype(y.dtype['event'], np.bool_) and
            np.issubdtype(y.dtype['time'], np.number)):
            # Already in correct format - no conversion needed
            pass
        elif y.ndim == 2 and y.shape[1] == 2:
            # Convert 2D array to structured array
            # Issue warning about auto-conversion
            warnings.warn(
                f"Cox model received 2D array. Auto-converting from [event, time] format to structured array.",
                UserWarning
            )
            events, times = y[:, 0], y[:, 1]
            
            print(f"Cox model inference: [event, time], "
                  f"time range: [{np.min(times):.3f}, {np.max(times):.3f}], "
                  f"events: {np.sum(events)}/{len(events)} ({np.mean(events):.3f})")
            
            # Convert events to boolean
            unique_events = np.unique(events)
            if np.all(np.isin(unique_events, [0, 1])):
                events_bool = events.astype(bool)
            elif np.all(np.isin(unique_events, [True, False])):
                events_bool = events.astype(bool)
            else:
                raise ValueError(
                    "y first column (events) must contain binary values (0/1 or True/False). "
                    "Expected format: [event, time] where event is binary and time is numeric."
                )
            
            # Create structured array
            y = np.array([(events_bool[i], times[i]) for i in range(len(y))],
                         dtype=[('event', bool), ('time', float)])
        else:
            raise ValueError(
                "y must be either:\n"
                "1. A structured array with 'event' and 'time' fields, or\n"
                "2. A 2D array with shape (n_samples, 2) where first column is event, second is time"
            )
        
        # Check required fields
        required_fields = ["event", "time"]
        if not all(field in y.dtype.names for field in required_fields):
            raise ValueError(f"y must contain fields: {required_fields}")
        
        # Validate event field (should be boolean)
        events = y["event"]
        if not np.issubdtype(events.dtype, np.bool_):
            raise ValueError("y['event'] must be boolean (True/False)")
        if np.sum(events) == 0:
            raise ValueError("No events observed - cannot fit Cox model")
        if np.sum(events) == len(events):
            warnings.warn("All observations are events - no censoring detected", UserWarning)
        
        # Validate time field (should be numeric and non-negative)
        times = y["time"]
        if not np.issubdtype(times.dtype, np.number):
            raise ValueError("y['time'] must be numeric")
        if np.any(times < 0):
            raise ValueError("y['time'] must be non-negative")
        if np.any(np.isnan(times)) or np.any(np.isinf(times)):
            raise ValueError("y['time'] contains NaN or infinite values")
        
        return y

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
    ) -> Tuple[object, float]:
        """Fit Cox model with given alpha or cross-validation for alpha selection."""
        if alpha is None:
            # Use LassoCV for Cox model (matches R implementation using glmnet with family="cox")
            # Convert survival data to format expected by LassoCV
            survival_times = y["time"]

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

            # Fit LassoCV on survival times (matches R: cv.glmnet(X, y, family = "cox"))
            lasso_cv.fit(X, survival_times)
            # Use alpha_scale_factor * alpha_min as default (more conservative, match lm.py)
            alpha_selected = self.alpha_scale_factor * lasso_cv.alpha_
        else:
            alpha_selected = alpha

        # Fit final Cox model with selected alpha
        base_model = self._fit_cox_model(X, y, alpha_selected)

        return base_model, alpha_selected

    def _generate_knockoff_data(
        self,
        X: np.ndarray,
        baseline_hazard_func: callable,
        time_range: Tuple[float, float],
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate knockoff data for Cox survival models.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Original predictor matrix.
        baseline_hazard_func : callable
            Function that computes baseline cumulative hazard (unused in current implementation).
        time_range : tuple of (min_time, max_time)
            Time range for survival simulation.
        rng : RandomState, optional
            Random number generator. If None, uses self.sample_rng.

        Returns
        -------
        X_knockoff : ndarray of shape (n_samples, n_features)
            Knockoff predictor matrix.
        y_knockoff : structured array
            Knockoff survival data with 'time' and 'event' fields.
        """
        n_samples, n_features = X.shape
        min_time, max_time = time_range

        # Use same X structure
        X_knockoff = X.copy()

        # Use provided RNG or fall back to self.sample_rng
        random_gen = rng if rng is not None else self.sample_rng

        # Generate null survival times (no covariate effect) - matches R implementation
        # Use baseline hazard function for realistic survival simulation
        if baseline_hazard_func is not None:
            # Generate times using baseline hazard (matches R: simsurv with betas = rep(0, p))
            survival_times = []
            for i in range(n_samples):
                # Sample from exponential with rate based on baseline hazard
                hazard_rate = baseline_hazard_func(min_time + (max_time - min_time) / 2)
                survival_times.append(
                    random_gen.exponential(1.0 / max(hazard_rate, 1e-6))
                )
        else:
            # Fallback to exponential distribution
            time_scale = (max_time - min_time) / 2
            survival_times = random_gen.exponential(time_scale, n_samples)

        # Ensure times are within the observed range
        survival_times = np.clip(survival_times, min_time, max_time)

        # All events observed for knockoff data (matches R: status_snp = rep(1, n))
        events = np.ones(n_samples, dtype=bool)

        # Create structured array for scikit-survival
        # Use same format as main fit method: (event, time) order
        y_knockoff = np.array(
            [(bool(event), time) for event, time in zip(events, survival_times)],
            dtype=[("event", bool), ("time", float)],
        )

        return X_knockoff, y_knockoff

    def _fit_cox_model(self, X: np.ndarray, y: np.ndarray, alpha_reg: float) -> object:
        """Fit Cox model with L1 regularization using scikit-survival."""
        # Use two-step approach: Lasso for feature selection, then Cox (matches R implementation)
        if alpha_reg > 0:
            # First, use Lasso for feature selection (matches R: glmnet with family="cox")
            lasso = Lasso(
                alpha=alpha_reg, max_iter=self.max_iter, random_state=self.random_state
            )
            lasso.fit(X, y["time"])

            # Get selected features
            selected_features = np.abs(lasso.coef_) > self.feature_selection_threshold
            if np.sum(selected_features) == 0:
                # If no features selected, use the feature with highest coefficient
                selected_features[np.argmax(np.abs(lasso.coef_))] = True

            X_selected = X[:, selected_features]
        else:
            X_selected = X
            selected_features = np.ones(X.shape[1], dtype=bool)

        # Fit Cox model on selected features
        cox_model = CoxPHSurvivalAnalysis(alpha=alpha_reg)
        cox_model.fit(X_selected, y)

        # Store feature selection mask for coefficient extraction
        cox_model._feature_mask = selected_features
        cox_model._n_features_original = X.shape[1]

        return cox_model

    def _get_coefficients(self, model: object) -> np.ndarray:
        """Extract coefficients from fitted Cox model."""
        # Get coefficients from the fitted model
        coefs = model.coef_

        # If we used feature selection, expand to original feature space
        if hasattr(model, "_feature_mask"):
            full_coefs = np.zeros(model._n_features_original)
            full_coefs[model._feature_mask] = coefs
            return full_coefs
        else:
            return coefs

    def _score_cox_model(self, model: object, X: np.ndarray, y: np.ndarray) -> float:
        """Score Cox model using concordance index."""
        try:
            # Get the selected features if feature selection was used
            if hasattr(model, "_feature_mask"):
                X_selected = X[:, model._feature_mask]
            else:
                X_selected = X

            # Predict risk scores
            risk_scores = model.predict(X_selected)

            # Calculate concordance index
            c_index = concordance_index_censored(y["event"], y["time"], risk_scores)[0]
            return c_index
        except Exception:
            return 0.0

    def _estimate_baseline_hazard(
        self, X: np.ndarray, y: np.ndarray, coefficients: np.ndarray
    ) -> callable:
        """Estimate baseline hazard function."""
        # Simplified baseline hazard estimation using scikit-survival
        # Fit model with selected features only
        nonzero_idx = np.where(coefficients != 0)[0]
        if len(nonzero_idx) > 0:
            X_selected = X[:, nonzero_idx]
            model = self._fit_cox_model(X_selected, y, self.alpha_used_)

            # Get baseline hazard from scikit-survival
            try:
                # scikit-survival provides baseline hazard estimation
                baseline_hazard = model.baseline_hazard_

                # Create interpolation function
                def hazard_func(t):
                    if hasattr(baseline_hazard, "index"):
                        times = baseline_hazard.index.values
                        hazards = baseline_hazard.values.flatten()
                        return np.interp(t, times, hazards, left=0, right=hazards[-1])
                    else:
                        return 0.0

            except Exception:
                # Fallback to constant hazard
                def hazard_func(t):
                    return 0.1

        else:
            # No selected features, return constant hazard
            def hazard_func(t):
                return 0.1  # Small constant hazard

        return hazard_func

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

            # Calculate model-specific scale factor for Cox models
            scale_factor = self.alpha_used_ + np.sqrt(
                np.log(self.n_features_) / self.n_samples_
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
        """Generate correction data for Cox survival models (matches R implementation)."""
        iteration_rng = (
            np.random
            if self.random_state is None
            else np.random.RandomState(self.random_state + iteration)
        )

        # Generate survival data using baseline hazard function (matches R: simsurv with cumhazard)
        min_time, max_time = self.time_range_

        # Use baseline hazard function for realistic survival simulation
        # This is simplified compared to R's simsurv, but captures the key idea
        if hasattr(self, "baseline_hazard_") and self.baseline_hazard_ is not None:
            # Generate times using baseline hazard
            survival_times = []
            for i in range(self.n_samples_):
                # Sample from exponential with rate based on baseline hazard
                # This is a simplified version of R's simsurv
                hazard_rate = self.baseline_hazard_(
                    min_time + (max_time - min_time) / 2
                )
                survival_times.append(
                    iteration_rng.exponential(1.0 / max(hazard_rate, 1e-6))
                )
        else:
            # Fallback to exponential distribution
            time_scale = (max_time - min_time) / 2
            survival_times = iteration_rng.exponential(time_scale, self.n_samples_)

        # Ensure times are within the observed range
        survival_times = np.clip(survival_times, min_time, max_time)

        # All events observed for correction data (matches R: status_correction = rep(1, n))
        events = np.ones(self.n_samples_, dtype=bool)

        # Create structured array for scikit-survival
        # Use same format as main fit method: (event, time) order
        y_correction = np.array(
            [(bool(event), time) for event, time in zip(events, survival_times)],
            dtype=[("event", bool), ("time", float)],
        )

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
        model_correction, _ = self._fit_lasso_model(
            X, y_correction, alpha=self.alpha_used_
        )
        coef_correction_abs = np.abs(self._get_coefficients(model_correction))

        # Generate knockoff data
        X_snp, y_snp = self._generate_knockoff_data(
            X, self.baseline_hazard_, self.time_range_, rng=iteration_rng
        )

        # Fit model to knockoff data
        model_snp, _ = self._fit_lasso_model(X_snp, y_snp, alpha=self.alpha_used_)
        coef_snp_abs = np.abs(self._get_coefficients(model_snp))

        return coef_correction_abs, coef_snp_abs

    def _compute_scale_factor(
        self, sigma_hat: float, alpha_reg: float, n_samples: int, n_features: int
    ) -> float:
        """
        Compute scale factor for Cox models.

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
