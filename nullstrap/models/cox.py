"""Nullstrap estimator for Cox proportional hazards models"""
from typing import Optional, Tuple, Union, Sequence
import warnings

import numpy as np
from scipy import stats
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
        Regularization parameter (corresponds to lambda in glmnet). If None, selected via cross-validation.
    B_reps : int, default=2
        Number of repetitions for correction factor estimation (matches R default for Cox models).
    cv_folds : int, default=10
        Number of cross-validation folds for alpha selection.
    max_iter : int, default=10000
        Maximum number of iterations for LASSO optimization.
    lasso_tol : float, default=1e-7
        Convergence tolerance for LASSO optimization.
    alpha_scale_factor : float, default=1.0
        Scaling factor for selected alpha. Set to 1.0 to match R (uses lambda.min without scaling).
    binary_search_tol : float, default=1e-8
        Convergence tolerance for binary search.
    correction_min : float, default=0.05
        Minimum bound for correction factor search.
    feature_selection_threshold : float, default=1e-6
        Threshold for feature selection in LASSO coefficients (Cox-specific).
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
        B_reps: int = 2,
        cv_folds: int = 10,
        max_iter: int = 10000,
        lasso_tol: float = 1e-7,
        alpha_scale_factor: float = 1.0,
        binary_search_tol: float = 1e-8,
        correction_min: float = 0.05,
        feature_selection_threshold: float = 1e-6,
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
        self.feature_selection_threshold = feature_selection_threshold
        self.alphas = alphas
        self.eps = eps

        if not SKLEARN_SURVIVAL_AVAILABLE:
            raise ImportError(
                "scikit-survival is required for Cox models. Install with: pip install scikit-survival"
            )

        # Initialize dedicated random number generator for sampling operations
        self.sample_rng = np.random.RandomState(random_state)

        # Additional attributes for Cox models
        self.baseline_hazard_: Optional[callable] = None
        self.event_times_: Optional[np.ndarray] = None
        self.event_indicators_: Optional[np.ndarray] = None

    def _validate_parameters(self) -> None:
        """
        Validate Cox-specific parameters.
        """
        # Call base class validation first
        super()._validate_parameters()
        
        # Cox-specific validation (currently no additional validation needed)
        # Future: Could add validation for survival-specific parameters if needed

    def _validate_data(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Validate input data for Cox models.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : structured array with shape (n_samples,) and fields ['event', 'time']
            Survival data. Required for Cox models.
        """
        super()._validate_X(X)
        
        if y is None:
            raise ValueError("Cox models require y parameter")
        
        # Use Cox-specific validation for survival data
        y = self._validate_y(y)
        super()._validate_sample_sizes(X, y)

    def _validate_y(self, y: np.ndarray) -> np.ndarray:
        """
        Validate survival data y for Cox models.

        Parameters
        ----------
        y : array-like
            Survival data in scikit-survival format or 2D array [event, time].

        Returns
        -------
        y : structured array
            Validated survival data with 'event' and 'time' fields.

        Raises
        ------
        ValueError
            If y format is invalid, events are not binary, times are negative,
            or no events are observed.
        """
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
            y = np.array([(events_bool[i], times[i]) for i in range(len(y))], dtype=[('event', bool), ('time', float)])
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NullstrapCox":
        """
        Fit the Nullstrap Cox estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : structured array with shape (n_samples,) and fields ['event', 'time']
            Survival data in scikit-survival format. Must be a structured NumPy array with fields:
            - 'event': bool, True if event occurred, False if censored
            - 'time': float, time to event or censoring

            Example:
            ```python
            import numpy as np

            # Create survival data
            events = np.array([True, True, False, True])
            times = np.array([5.2, 1.1, 8.0, 2.3])

            # Create structured array - shape is (4,) not (4, 2)
            y = np.array([(e, t) for e, t in zip(events, times)], 
                         dtype=[('event', bool), ('time', float)])
            print(y.shape)      # (4,)
            print(y['event'])   # [True True False True]
            print(y['time'])    # [5.2 1.1 8.0 2.3]
            ```

        Returns
        -------
        self : NullstrapCox
            Fitted estimator.
        """
        # ===== Validation and Setup =====
        self._validate_parameters()
        self._validate_data(X, y)
        self._set_random_state()

        self.n_samples_, self.n_features_ = X.shape

        # ===== Data Preprocessing =====
        # Standardize: z-score normalization + 1/sqrt(n) scaling for Cox models
        X_scaled, _ = standardize_data(X, n_samples=self.n_samples_, scale_by_sample_size=True)

        # Store survival data for later use in synthetic data generation
        self.event_times_ = y["time"]
        self.event_indicators_ = y["event"]
        time_range = (np.min(self.event_times_), np.max(self.event_times_))

        # ===== Fit Base Model on Real Data =====
        # Fit LASSO-penalized Cox model with cross-validated or user-specified alpha
        model_base, alpha_used = self._fit_lasso_model(X_scaled, y, alpha=self.alpha_)
        self.alpha_used_ = alpha_used
        coef_base = self._get_coefficients(model_base)

        # Estimate cumulative baseline hazard function from real data
        self.baseline_hazard_ = self._estimate_baseline_hazard(X_scaled, y, coef_base)

        # ===== Generate Knockoff (Null) Data =====
        # Generate synthetic survival times with no covariate effects (beta=0)
        y_knockoff = self._generate_synthetic_data(
            X_scaled, 
            baseline_hazard=self.baseline_hazard_, 
            time_range=time_range
        )
        model_knockoff = self._fit_cox_model(X_scaled, y_knockoff, alpha_used)
        coef_knockoff = np.abs(self._get_coefficients(model_knockoff))

        # ===== Estimate Correction Factor =====
        # Use positive/negative controls to estimate data-driven correction factor
        correction_factor = self._estimate_correction_factor(X_scaled, y, coef_base)
        self.correction_factor_ = correction_factor
        self.statistic_ = np.abs(coef_base)

        # ===== Apply Correction and Compute Threshold =====
        # Inflate knockoff statistics by correction factor * scale
        scale_factor = self._compute_scale_factor()
        corrected_knockoff = coef_knockoff + correction_factor * scale_factor

        # Binary search for threshold that controls FDR
        self.threshold_ = binary_search_threshold(
            self.statistic_,
            corrected_knockoff,
            self.fdr,
            max_iter=self.max_iter,
            tol=self.binary_search_tol
        )

        # ===== Feature Selection =====
        # Select features exceeding threshold
        self.selected_ = self.get_selected_features(self.statistic_, self.threshold_)
        self.n_features_selected_ = len(self.selected_)

        return self

    def _fit_lasso_model(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        alpha: Optional[float] = None
    ) -> Tuple[object, float]:
        """
        Fit Cox model with specified or cross-validated regularization parameter.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Predictor matrix (standardized).
        y : structured array with shape (n_samples,) and fields ['event', 'time']
            Survival data.
        alpha : float, optional
            Regularization parameter (corresponds to lambda in glmnet).
            If None, selected via CV.

        Returns
        -------
        model : object
            Fitted Cox model.
        alpha_used : float
            Regularization parameter used, scaled by alpha_scale_factor (default 1.0 for Cox models).

        Notes
        -----
        Uses LassoCV when alpha is None. If self.alphas is an array, uses those values
        for CV; if integer, generates that many alphas automatically using eps ratio.
        Fits on survival times for alpha selection, then uses Cox model with selected alpha.
        """
        if alpha is None:
            # Set up LassoCV for alpha selection via cross-validation
            # Use survival times for CV (approximates Cox partial likelihood optimization)
            survival_times = y["time"]

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

            lasso_cv.fit(X, survival_times)
            # Scale selected alpha by alpha_scale_factor (more conservative)
            alpha_used = self.alpha_scale_factor * lasso_cv.alpha_
        else:
            alpha_used = alpha

        # Fit final Cox model with selected alpha
        model = self._fit_cox_model(X, y, alpha_used)

        return model, alpha_used

    def _fit_cox_model(self, X: np.ndarray, y: np.ndarray, alpha_reg: float) -> object:
        """
        Fit Cox model with L1 regularization using scikit-survival.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : structured array with shape (n_samples,) and fields ['event', 'time']
            Survival data.
        alpha_reg : float
            Regularization parameter.

        Returns
        -------
        model : object
            Fitted Cox model with feature selection mask.
        """
        # Use two-step approach: Lasso for feature selection, then Cox (matches R implementation)
        if alpha_reg > 0:
            # First, use Lasso for feature selection (matches R: glmnet with family="cox")
            lasso = Lasso(alpha=alpha_reg, max_iter=self.max_iter, random_state=self.random_state)
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
        """
        Extract coefficients from fitted Cox model.

        Parameters
        ----------
        model : object
            Fitted Cox model.

        Returns
        -------
        coefficients : ndarray of shape (n_features,)
            Model coefficients expanded to full feature space.
        """
        # Get coefficients from the fitted model
        coefs = model.coef_

        # If we used feature selection, expand to original feature space
        if hasattr(model, "_feature_mask"):
            full_coefs = np.zeros(model._n_features_original)
            full_coefs[model._feature_mask] = coefs
            return full_coefs
        else:
            return coefs

    def _estimate_baseline_hazard(self, X: np.ndarray, y: np.ndarray, coefficients: np.ndarray) -> callable:
        """
        Estimate cumulative baseline hazard function from fitted Cox model.
        
        Matches R implementation using basehaz() which returns cumulative hazard.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Predictor matrix (standardized).
        y : structured array with shape (n_samples,) and fields ['event', 'time']
            Survival data.
        coefficients : ndarray of shape (n_features,)
            Coefficients from fitted model.

        Returns
        -------
        callable
            Cumulative baseline hazard function H_0(t) that takes time t and returns 
            cumulative hazard rate. Matches R's basehaz() output.
            
        Notes
        -----
        R implementation:
            fit_baseline <- coxph(y_baseline ~ X[,which(coef_corr!=0)])
            cum_hazard = basehaz(fit_baseline)
            cum_hazard_fun = approxfun(cum_hazard$time, cum_hazard$hazard)
            # With edge case handling for t <= min_time and t >= max_time
        """
        # Fit model with selected features only (nonzero coefficients)
        nonzero_idx = np.where(coefficients != 0)[0]
        if len(nonzero_idx) > 0:
            X_selected = X[:, nonzero_idx]
            model = self._fit_cox_model(X_selected, y, self.alpha_used_)

            # Get cumulative baseline hazard from scikit-survival
            try:
                # scikit-survival provides baseline_survival_, convert to cumulative hazard
                # H_0(t) = -log(S_0(t))
                baseline_survival = model.baseline_survival_
                
                if hasattr(baseline_survival, "x"):
                    # breslow estimator format
                    times = baseline_survival.x
                    surv_probs = baseline_survival.y
                elif hasattr(baseline_survival, "index"):
                    # pandas Series format
                    times = baseline_survival.index.values
                    surv_probs = baseline_survival.values
                else:
                    raise AttributeError("Unknown baseline_survival format")
                
                # Convert survival probabilities to cumulative hazard
                # Avoid log(0) by clipping survival probabilities
                surv_probs_clipped = np.clip(surv_probs, 1e-10, 1.0)
                cum_hazards = -np.log(surv_probs_clipped)
                
                # Store time range for edge case handling (matches R implementation)
                min_time = np.min(times)
                max_time = np.max(times)
                max_hazard = cum_hazards[-1]
                
                # Create interpolation function with edge case handling
                # This matches R's cum_hazard_fun_extension
                def hazard_func(t):
                    """
                    Cumulative baseline hazard with edge case handling.
                    - t <= min_time: return 0
                    - t >= max_time: return max_hazard (slightly below to avoid boundary)
                    - otherwise: interpolate
                    """
                    if np.isscalar(t):
                        if t <= min_time:
                            return 0.0
                        elif t >= max_time:
                            return max_hazard * 0.9999  # R uses max_time - 1e-6
                        else:
                            return np.interp(t, times, cum_hazards)
                    else:
                        # Vectorized version
                        t_array = np.asarray(t)
                        result = np.interp(t_array, times, cum_hazards)
                        result[t_array <= min_time] = 0.0
                        result[t_array >= max_time] = max_hazard * 0.9999
                        return result

            except Exception as e:
                # Fallback to constant cumulative hazard (linear growth)
                warnings.warn(f"Baseline hazard estimation failed: {e}. Using constant hazard fallback.", UserWarning)
                def hazard_func(t):
                    # Constant hazard rate of 0.1 -> cumulative hazard H(t) = 0.1 * t
                    return 0.1 * np.maximum(t, 0)

        else:
            # No selected features, return constant cumulative hazard
            def hazard_func(t):
                return 0.1 * np.maximum(t, 0)

        return hazard_func

    def _fit_weibull_to_times(
        self, 
        times: np.ndarray
    ) -> Tuple[float, float]:
        """
        Fit Weibull distribution to observed survival times.
        
        Matches R implementation: fit_wei <- fitdistr(time_simu$eventtime, "weibull")

        Parameters
        ----------
        times : ndarray of shape (n_samples,)
            Observed survival/event times.

        Returns
        -------
        shape : float
            Weibull shape parameter (also called k or gamma).
        scale : float
            Weibull scale parameter (also called lambda).
            
        Notes
        -----
        Uses Maximum Likelihood Estimation (MLE) via scipy.stats.weibull_min.fit().
        Falls back to method of moments if MLE fails.
        """
        try:
            # Use scipy to fit Weibull (returns shape, loc, scale)
            # loc is typically 0 for survival data (no shift)
            shape, loc, scale = stats.weibull_min.fit(times, floc=0)
        except Exception:
            # Fallback to method of moments if MLE fails
            mean_time = np.mean(times)
            std_time = np.std(times)
            # Approximate shape parameter using coefficient of variation
            shape = (std_time / mean_time) ** (-1.086)  # Empirical approximation
            scale = mean_time / stats.gamma(1 + 1/shape)
        
        return shape, scale
    
    def _generate_weibull_survival_times(
        self,
        X: np.ndarray,
        beta: Optional[np.ndarray],
        shape: float,
        scale: float,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """
        Generate survival times from Weibull distribution with optional covariate effects.
        
        Implements Cox proportional hazards model structure:
        h(t|X) = h_0(t) * exp(X @ beta)
        
        For Weibull baseline: T ~ Weibull(shape, scale) / exp(X @ beta)

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Predictor matrix.
        beta : ndarray of shape (n_features,) or None
            Coefficient vector. If None or all zeros, generates from pure Weibull.
        shape : float
            Weibull shape parameter.
        scale : float
            Weibull scale parameter.
        rng : np.random.RandomState
            Random number generator.

        Returns
        -------
        survival_times : ndarray of shape (n_samples,)
            Generated survival times.
            
        Notes
        -----
        Matches R implementation (lines 430-433, 444-454 in nullstrap_filter.R):
        survival_time_correction <- simsurv(lambdas = fit_wei$estimate[2], 
                                            gammas = fit_wei$estimate[1], 
                                            x = as.data.frame(X), betas = beta_real)
        """
        n_samples = X.shape[0]
        
        if beta is not None and np.any(beta != 0):
            # Generate survival times with covariate effects
            # Using inverse CDF method: T = scale * (-log(U) / exp(X @ beta))^(1/shape)
            linear_predictor = X @ beta
            u = rng.uniform(0, 1, n_samples)
            survival_times = scale * np.power(
                -np.log(u) / np.exp(linear_predictor), 
                1.0 / shape
            )
        else:
            # No covariate effect: pure Weibull distribution
            # Matches R null case: simsurv(..., betas = rep(0, p))
            survival_times = stats.weibull_min.rvs(
                shape, 
                loc=0, 
                scale=scale, 
                size=n_samples, 
                random_state=rng
            )
        
        return survival_times

    def _generate_synthetic_data(
        self,
        X: np.ndarray,
        beta: Optional[np.ndarray] = None,
        baseline_hazard: Optional[callable] = None,
        time_range: Optional[Tuple[float, float]] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        """
        Generate synthetic survival data using Weibull distribution with covariate effects.
        
        Matches R implementation: first generates null times, fits Weibull, then generates
        times with covariate effects using the Cox proportional hazards model structure.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Predictor matrix.
        beta : ndarray of shape (n_features,), optional
            Signal coefficients. If None, generates null/knockoff data (no covariate effect).
        baseline_hazard : callable, optional
            Function that computes baseline hazard. Not used in current implementation.
        time_range : tuple of (min_time, max_time), optional
            Time range for survival simulation. Required for Weibull fitting.
        rng : np.random.RandomState, optional
            Random generator. If None, uses self.sample_rng.

        Returns
        -------
        y_synthetic : ndarray (structured array)
            Synthetic survival data with 'event' and 'time' fields.
            
        Notes
        -----
        Implementation follows R nullstrap_filter approach (lines 430-454):
        1. Generate null survival times (beta=0) from exponential distribution
        2. Fit Weibull distribution to null times
        3. Generate survival times from Weibull with covariate effects
        
        R equivalent:
            time_simu = simsurv(cumhazard = baseline_fun, x = X, betas = rep(0, p))
            fit_wei <- fitdistr(time_simu$eventtime, "weibull")
            survival_time <- simsurv(lambdas = fit_wei$estimate[2], 
                                     gammas = fit_wei$estimate[1], 
                                     x = X, betas = beta_real)
        """
        random_gen = self.sample_rng if rng is None else rng
        n_samples = X.shape[0]
        
        if time_range is None:
            raise ValueError("time_range must be provided for Weibull fitting")
        
        min_time, max_time = time_range
        
        # Step 1: Generate null survival times to fit Weibull distribution
        # Matches R: time_simu = simsurv(cumhazard = baseline_fun, x = X, betas = rep(0, p))
        time_scale = (max_time - min_time) / 2
        null_times = random_gen.exponential(time_scale, n_samples)
        null_times = np.clip(null_times, min_time * 0.1, max_time * 2.0)
        
        # Step 2: Fit Weibull distribution to null times
        # Matches R: fit_wei <- fitdistr(time_simu$eventtime, "weibull")
        shape, scale = self._fit_weibull_to_times(null_times)
        
        # Step 3: Generate survival times with covariate effects
        # Matches R: simsurv(lambdas, gammas, x = X, betas = beta_real)
        survival_times = self._generate_weibull_survival_times(
            X, beta, shape, scale, random_gen
        )
        
        # Ensure times are positive and reasonable
        survival_times = np.clip(survival_times, 1e-6, max_time * 10)

        # All events observed for synthetic data (no censoring)
        # Matches R: status_correction = rep(1, n)
        events = np.ones(n_samples, dtype=bool)

        # Create structured array for scikit-survival
        y_synthetic = np.array(
            [(bool(event), time) for event, time in zip(events, survival_times)],
            dtype=[("event", bool), ("time", float)]
        )

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
        y : structured array with shape (n_samples,) and fields ['event', 'time']
            Survival data.
        coef_base : ndarray of shape (n_features,)
            Coefficients from original model fit.

        Returns
        -------
        correction_factor : float
            Maximum correction factor across repetitions, used to adjust knockoff
            statistics for FDR control.
        """
        correction_factors = []
        time_range = (np.min(y["time"]), np.max(y["time"]))

        for b in range(self.B_reps):
            # ===== 1. Create Positive Control (Augmented Data) =====
            # Inflate coefficients to create synthetic truth with known signal
            beta_augmented = inflate_signal(coef_base, inflation_factor, "additive")
            signal_indices = np.where(beta_augmented != 0)[0]

            # Create iteration-specific RNG for reproducibility
            iteration_rng = None if self.random_state is None else np.random.RandomState(self.random_state + b)

            # ===== 2. Generate Synthetic Survival Data =====
            # Augmented: survival times with inflated covariate effects (positive control)
            y_augmented = self._generate_synthetic_data(
                X, beta=beta_augmented, 
                baseline_hazard=self.baseline_hazard_, 
                time_range=time_range, 
                rng=iteration_rng
            )
            # Knockoff: survival times with no covariate effects (null/negative control)
            y_knockoff = self._generate_synthetic_data(
                X, beta=None, 
                baseline_hazard=self.baseline_hazard_, 
                time_range=time_range, 
                rng=iteration_rng
            )

            # ===== 3. Fit Models to Synthetic Data =====
            model_augmented, _ = self._fit_lasso_model(X, y_augmented, alpha=self.alpha_used_)
            model_knockoff, _ = self._fit_lasso_model(X, y_knockoff, alpha=self.alpha_used_)
            coef_augmented = np.abs(self._get_coefficients(model_augmented))
            coef_knockoff = np.abs(self._get_coefficients(model_knockoff))
            
            # ===== 4. Binary Search for Correction Factor =====
            correction_min = self.correction_min
            correction_max = np.max(coef_augmented) / scale_factor

            correction = binary_search_correction_factor(
                coef_augmented_abs=coef_augmented,
                coef_knockoff_abs=coef_knockoff,
                signal_indices=signal_indices,
                fdr=self.fdr,
                scale_factor=scale_factor,
                correction_min=correction_min,
                correction_max=correction_max,
                max_iter=self.max_iter,
                tol=self.binary_search_tol
            )

            correction_factors.append(correction)

        # Return maximum correction factor for conservative FDR control
        return max(correction_factors)

    def _compute_scale_factor(self) -> float:
        """
        Compute scale factor for correction: alpha + sqrt(log(p) / n).

        Returns
        -------
        float
            Scale factor for knockoff correction.
        
        Notes
        -----
        Uses instance attributes: alpha_used_, n_samples_, n_features_.
        For Cox models, does not use sigma_hat (unlike linear models).
        """
        return self.alpha_used_ + np.sqrt(np.log(self.n_features_) / self.n_samples_)

    def _score_cox_model(self, model: object, X: np.ndarray, y: np.ndarray) -> float:
        """
        Score Cox model using concordance index (optional utility method).

        Parameters
        ----------
        model : object
            Fitted Cox model.
        X : array-like of shape (n_samples, n_features)
            Test data.
        y : structured array with shape (n_samples,) and fields ['event', 'time']
            Survival data.

        Returns
        -------
        c_index : float
            Concordance index (between 0 and 1).
        """
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
