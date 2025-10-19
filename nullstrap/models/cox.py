"""Nullstrap estimator for Cox proportional hazards models"""
from typing import Optional, Tuple, Union, Sequence
import warnings

import numpy as np
from sklearn.model_selection import KFold
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

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
    alphas : int or array-like, default=100
        Alphas for CV: int (number of alphas) or array (explicit values).
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
        Estimated cumulative baseline hazard function H_0(t).
    time_range_ : tuple of (min_time, max_time)
        Time range for synthetic data generation, derived from observed data.
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
        correction_min: float = 1e-12,
        alphas: Union[int, Sequence[float]] = 100,
        random_state: Optional[int] = None,
    ):
        super().__init__(fdr=fdr, alpha_=alpha_, B_reps=B_reps, random_state=random_state)
        self.cv_folds = cv_folds
        self.max_iter = max_iter
        self.lasso_tol = lasso_tol
        self.alpha_scale_factor = alpha_scale_factor
        self.binary_search_tol = binary_search_tol
        self.correction_min = correction_min
        self.alphas = alphas

        # Initialize dedicated random number generator for sampling operations
        self.sample_rng = np.random.RandomState(random_state)

        # Additional attributes for Cox models
        self.baseline_hazard_: Optional[callable] = None
        self.time_range_: Optional[Tuple[float, float]] = None

    def _validate_parameters(self) -> None:
        """
        Validate Cox-specific parameters.
        """
        # Call base class validation first
        super()._validate_parameters()
        
        # Validate Cox-specific parameters
        if isinstance(self.alphas, int):
            if self.alphas <= 0:
                raise ValueError("alphas (int) must be > 0")
        elif isinstance(self.alphas, (list, tuple, np.ndarray)):
            if len(self.alphas) == 0:
                raise ValueError("alphas (sequence) cannot be empty")
            if not all(a > 0 for a in self.alphas):
                raise ValueError("all alphas must be > 0")
        else:
            raise ValueError("alphas must be int or sequence of floats")
        

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
        return y

    def _validate_y(self, y: np.ndarray) -> np.ndarray:
        """
        Validate survival data y for Cox models.

        Parameters
        ----------
        y : ndarray
            Survival data in scikit-survival format (structured array with 'event' and 'time' fields)
            or 2D array with shape (n_samples, 2) where first column is event, second is time.

        Returns
        -------
        y : structured array
            Validated survival data with 'event' (bool) and 'time' (float) fields.

        Raises
        ------
        ValueError
            If y format is invalid, events are not binary, times are negative, or no events are observed.
        """
        # Check if already valid structured array
        if not (y.dtype.names and 'event' in y.dtype.names and 'time' in y.dtype.names):
            # Try auto-conversion from 2D array
            if y.ndim == 2 and y.shape[1] == 2:
                warnings.warn("Auto-converting 2D array to structured array format", UserWarning)
                events, times = y[:, 0], y[:, 1]
                
                # Validate and convert events to boolean
                if not np.all(np.isin(np.unique(events), [0, 1, True, False])):
                    raise ValueError("Events must be binary (0/1 or True/False)")
                
                # Convert to structured array
                y = np.array(list(zip(events.astype(bool), times.astype(float))), 
                             dtype=[('event', bool), ('time', float)])
            else:
                raise ValueError("y must be structured array with 'event' and 'time' fields")
        
        # Validate events
        events = y["event"]
        if not np.issubdtype(events.dtype, np.bool_):
            raise ValueError("y['event'] must be boolean")
        
        n_events = np.sum(events)
        if n_events == 0:
            raise ValueError("No events observed - cannot fit Cox model")
        if n_events == len(events):
            warnings.warn("All observations are events - no censoring detected", UserWarning)
        
        # Validate times
        times = y["time"]
        if not np.issubdtype(times.dtype, np.number):
            raise ValueError("y['time'] must be numeric")
        if np.any(times < 0):
            raise ValueError("y['time'] must be non-negative")
        if np.any(~np.isfinite(times)):
            raise ValueError("y['time'] contains NaN or infinite values")
        
        return y

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NullstrapCox":
        """
        Fit the Nullstrap Cox estimator.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
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
        # Validate parameters and input data
        self._validate_parameters()
        y = self._validate_data(X, y)
        self._set_random_state()

        self.n_samples_, self.n_features_ = X.shape

        # Standardize data for Cox models (z-score + sample size normalization)
        X_scaled, _ = standardize_data(X, y=None, scale_by_sample_size=True, n_samples=self.n_samples_)
        
        # Store time range for synthetic data generation
        self.time_range_ = (np.min(y["time"]), np.max(y["time"]))

        # Fit base LASSO Cox model
        model_base, alpha_used = self._fit_lasso_model(X_scaled, y, alpha=self.alpha_)
        self.alpha_used_ = alpha_used
        coef_base = model_base.coef_.ravel()
        
        # Extract baseline hazard from CoxPHSurvivalAnalysis model fitted on selected features
        X_selected = X_scaled[:, coef_base != 0]
        if X_selected.shape[1] > 0:
            cox_ph_model = CoxPHSurvivalAnalysis()
            cox_ph_model.fit(X_selected, y)
            self.baseline_hazard_ = self._get_baseline_hazard(cox_ph_model, X_selected)
        else:
            # No features selected, use constant hazard
            self.baseline_hazard_ = lambda t: 0.1 * np.maximum(t, 0)

        # Generate knockoff data, fit model to knockoff data
        y_knockoff = self._generate_synthetic_data(X_scaled, beta=None)
        model_knockoff, _ = self._fit_lasso_model(X_scaled, y_knockoff, alpha_used)
        coef_knockoff = model_knockoff.coef_.ravel()

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
    ) -> Tuple[object, float]:
        """
        Fit Cox model with specified or cross-validated regularization parameter using CoxnetSurvivalAnalysis.

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
        When alpha is None, performs k-fold cross-validation (k=cv_folds) to select the optimal alpha
        based on concordance index (C-index).
        """
        if alpha is None:
            # Use cross-validation to select optimal alpha
            # First, generate the alpha path using CoxnetSurvivalAnalysis
            if isinstance(self.alphas, int):
                # Generate alphas automatically
                coxnet_path = CoxnetSurvivalAnalysis(n_alphas=self.alphas, l1_ratio=1.0, fit_baseline_model=False, tol=self.lasso_tol)
                coxnet_path.fit(X, y)
                alpha_candidates = coxnet_path.alphas_
            else:
                # Use provided alpha values
                alpha_candidates = np.array(self.alphas)
            
            # Perform cross-validation to select best alpha
            kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            cv_scores = np.zeros((len(alpha_candidates), self.cv_folds))
            
            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                for alpha_idx, alpha_candidate in enumerate(alpha_candidates):
                    try:
                        # Fit model on training fold
                        model = CoxnetSurvivalAnalysis(alphas=[alpha_candidate], l1_ratio=1.0, fit_baseline_model=False, tol=self.lasso_tol)
                        model.fit(X_train, y_train)
                        
                        # Evaluate on validation fold using concordance index
                        c_index = self._score_cox_model(model, X_val, y_val)
                        cv_scores[alpha_idx, fold_idx] = c_index
                    except Exception:
                        # If fitting fails, assign poor score
                        cv_scores[alpha_idx, fold_idx] = 0.5
            
            # Select alpha with best mean CV score
            mean_cv_scores = cv_scores.mean(axis=1)
            best_alpha_idx = np.argmax(mean_cv_scores)
            best_alpha = alpha_candidates[best_alpha_idx]
            
            # Scale selected alpha by alpha_scale_factor (more conservative)
            alpha_used = self.alpha_scale_factor * best_alpha
        else:
            alpha_used = alpha

        # Fit final Cox model with selected alpha
        model = CoxnetSurvivalAnalysis(alphas=[alpha_used], l1_ratio=1.0, fit_baseline_model=True, tol=self.lasso_tol)
        model.fit(X, y)

        return model, alpha_used

    def _get_baseline_hazard(self, model: object, X: np.ndarray) -> callable:
        """
        Extract cumulative baseline hazard function from fitted Cox model.

        Parameters
        ----------
        model : CoxPHSurvivalAnalysis
            Fitted Cox model on selected features.
        X : ndarray of shape (n_samples, n_features)
            Predictor matrix used for model fitting.

        Returns
        -------
        callable
            Cumulative baseline hazard function H_0(t).
        """
        try:
            cum_hazard_func = model.predict_cumulative_hazard_function(X[:1])[0]
            times, cum_hazards = cum_hazard_func.x, cum_hazard_func.y
            
            def hazard_func(t):
                result = np.interp(t, times, cum_hazards)
                result = np.where(t <= times[0], 0.0, result)
                result = np.where(t >= times[-1], cum_hazards[-1] * 0.9999, result)
                return result
            
            return hazard_func
            
        except Exception as e:
            warnings.warn(f"Baseline hazard extraction failed: {e}. Using constant hazard fallback.", UserWarning)
            return lambda t: 0.1 * np.maximum(t, 0)

    def _generate_synthetic_data(
        self,
        X: np.ndarray,
        beta: Optional[np.ndarray] = None,
        time_range: Optional[Tuple[float, float]] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        """
        Generate synthetic survival data using baseline hazard with inverse transform sampling.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Predictor matrix.
        beta : ndarray of shape (n_features,), optional
            Signal coefficients. If None, generates null/knockoff data.
        time_range : tuple of (min_time, max_time), optional
            Time range for simulation. If None, uses self.time_range_.
        rng : np.random.RandomState, optional
            Random generator. If None, uses self.sample_rng.

        Returns
        -------
        ndarray (structured array)
            Synthetic survival data with 'event' and 'time' fields.
        """
        random_gen = self.sample_rng if rng is None else rng
        n_samples = X.shape[0]
        min_time, max_time = time_range if time_range is not None else self.time_range_
        
        # Generate uniform random numbers
        U = random_gen.uniform(0, 1, n_samples)
        
        # Generate target hazard
        if beta is not None and np.any(beta != 0):
            # With covariate effects: H(t) = H₀(t) * exp(X @ beta)
            hazard_multiplier = np.exp(X @ beta)
            target_hazard = -np.log(U) / hazard_multiplier
        else:
            # Baseline only: H(t) = H₀(t)
            target_hazard = -np.log(U)
        
        # Use interpolation to find survival times
        times_grid = np.linspace(min_time, max_time, 1000)
        hazards_grid = self.baseline_hazard_(times_grid)
        survival_times = np.interp(target_hazard, hazards_grid, times_grid)
        survival_times = np.clip(survival_times, min_time, max_time)
        
        # Create structured array
        events = np.ones(n_samples, dtype=bool)
        return np.array([(event, time) for event, time in zip(events, survival_times)], 
                       dtype=[('event', bool), ('time', float)])

    def _estimate_correction_factor(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        coef_base: np.ndarray
    ) -> float:
        """
        Estimate correction factor for FDR control.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Predictor matrix (standardized).
        y : structured array with fields ['event', 'time']
            Survival data.
        coef_base : ndarray of shape (n_features,)
            Coefficients from original model fit.

        Returns
        -------
        float
            Maximum correction factor across repetitions for conservative FDR control.
        """
        correction_factors = []

        for b in range(self.B_reps):
            # Augment signal for positive control
            inflated_factor = self.alpha_used_ + np.sqrt(np.log(self.n_features_) / self.n_samples_)
            beta_augmented = inflate_signal(coef_base, inflated_factor, "additive")
            signal_indices = np.where(beta_augmented != 0)[0]

            # Generate random number generator for reproducibility
            iteration_rng = None if self.random_state is None else np.random.RandomState(self.random_state + b)

            # Generate synthetic survival data
            y_augmented = self._generate_synthetic_data(X, beta=beta_augmented, rng=iteration_rng)
            y_knockoff  = self._generate_synthetic_data(X, beta=None, rng=iteration_rng)

            # Fit models to synthetic data
            model_augmented, _ = self._fit_lasso_model(X, y_augmented, alpha=self.alpha_used_)
            model_knockoff, _  = self._fit_lasso_model(X, y_knockoff, alpha=self.alpha_used_)
            coef_augmented_abs = np.abs(model_augmented.coef_.ravel())
            coef_knockoff_abs  = np.abs(model_knockoff.coef_.ravel())
            
            # Binary search for correction factor
            scale_factor = self._compute_scale_factor()
            correction_min = self.correction_min
            correction_max = np.max(coef_augmented_abs) / scale_factor

            correction = binary_search_correction_factor(
                coef_augmented_abs=coef_augmented_abs,
                coef_knockoff_abs=coef_knockoff_abs,
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
        Compute scale factor for knockoff correction.

        Returns
        -------
        float
            Scale factor = alpha + sqrt(log(p) / n).
        """
        return self.alpha_used_ + np.sqrt(np.log(self.n_features_) / self.n_samples_)

    def _score_cox_model(self, model: object, X: np.ndarray, y: np.ndarray) -> float:
        """
        Score Cox model using concordance index.

        Parameters
        ----------
        model : object
            Fitted Cox model.
        X : ndarray of shape (n_samples, n_features)
            Test data.
        y : structured array with fields ['event', 'time']
            Survival data.

        Returns
        -------
        float
            Concordance index (0-1, higher is better).
        """
        c_index = 0.0   
        try:
            risk_scores = model.predict(X)
            c_index = concordance_index_censored(y["event"], y["time"], risk_scores)[0]
        except Exception as e:
            warnings.warn(f"Error scoring Cox model: {e}. Returning 0.0.", UserWarning)
        
        return c_index
