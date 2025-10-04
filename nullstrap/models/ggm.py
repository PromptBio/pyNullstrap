"""Nullstrap estimator for Gaussian Graphical Models"""
import warnings
from typing import Optional, Tuple, Any

import numpy as np
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV

from ..estimator import BaseNullstrap
from ..utils.core import (binary_search_correction_factor,
                          binary_search_threshold, inflate_signal,
                          standardize_data)


class NullstrapGGM(BaseNullstrap):
    """
    Nullstrap feature selector for Gaussian Graphical Models with FDR control, using graphical LASSO regularization.

    This class implements the Nullstrap procedure for Gaussian Graphical Models,
    providing edge selection with false discovery rate control using the
    graphical LASSO (GLASSO). Inherits from SelectorMixin for pipeline compatibility.

    Note: For graphical models, "features" refer to edges in the graph.
    Use get_adjacency_matrix() to get the selected graph structure.

    Parameters
    ----------
    fdr : float, default=0.1
        Target false discovery rate.
    alpha_ : float, optional
        Regularization parameter for graphical LASSO (maps to 'lambda' in glmnet). If None, selected by the method specified in selection_method.
    selection_method : str, default='cv'
        Method for alpha selection: 'cv' for cross-validation (log-likelihood) or 'aic' for AIC-based selection.
    B_reps : int, default=5
        Number of repetitions for correction factor estimation.
    cv_folds : int, default=10
        Cross-validation folds for alpha selection.
    max_iter : int, default=1000
        Maximum number of iterations for graphical LASSO.
    alpha_scale_factor : float, default=0.5
        Scaling factor for alpha selection (0.5 * alpha_min for conservative selection).
    feature_selection_threshold : float, default=1e-6
        Threshold for feature selection in precision matrix elements.
    alpha_candidates_count : int, default=50
        Number of alpha candidates for alpha selection (used by both CV and AIC methods).
    alpha_log_scale_range : tuple, default=(-3, 1)
        Range for alpha log scale in logspace (min_log10, max_log10). Used for np.logspace(min_log10, max_log10, count).
    target_sparsity : float, default=0.1
        Target sparsity level for positive control matrix.
    sparsity_tolerance : float, default=0.05
        Tolerance for sparsity matching in positive control.
    binary_search_tol : float, default=1e-8
        Tolerance for binary search convergence in correction factor estimation.
    initial_correction_factor : float, default=0.05
        Initial left bound for binary search of correction factor.
    default_correction_factor : float, default=1.0
        Default correction factor when estimation fails.
    min_eigenvalue_adjustment : float, default=0.01
        Minimum adjustment for eigenvalue regularization.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    threshold_ : float
        Computed selection threshold after fitting.
    selected_ : ndarray
        Indices of selected edges (upper triangular).
    n_features_selected_ : int
        Number of selected features.
    statistic_ : ndarray
        Test statistics (absolute precision matrix elements).
    alpha_used_ : float
        Regularization parameter used in final fit.
    correction_factor_ : float
        Estimated correction factor for FDR control.
    precision_ : ndarray
        Estimated precision matrix.
    covariance_ : ndarray
        Estimated covariance matrix.

    Notes
    -----
    This implementation uses graphical LASSO for sparse precision matrix estimation
    and follows the nullstrap methodology for FDR-controlled feature selection.

    Alpha selection methods:
    - 'cv': Uses cross-validation with log-likelihood scoring (default, more robust)
    - 'aic': Uses AIC-based selection (matches original R implementation)
    """

    def __init__(
        self,
        fdr: float = 0.1,
        alpha_: Optional[float] = None,
        selection_method: str = "cv",
        B_reps: int = 5,
        cv_folds: int = 10,
        max_iter: int = 1000,
        alpha_scale_factor: float = 0.5,
        feature_selection_threshold: float = 1e-6,
        alpha_candidates_count: int = 50,
        alpha_log_scale_range: Tuple[float, float] = (-3, 1),
        target_sparsity: float = 0.1,
        sparsity_tolerance: float = 0.05,
        binary_search_tol: float = 1e-8,
        initial_correction_factor: float = 0.05,
        default_correction_factor: float = 1.0,
        min_eigenvalue_adjustment: float = 0.01,
        random_state: Optional[int] = None,
    ):
        # Note: GraphicalLasso doesn't use y, so we don't pass B_reps to super().__init__
        super().__init__(
            fdr=fdr, alpha_=alpha_, B_reps=B_reps, random_state=random_state
        )
        self.selection_method = selection_method
        self.cv_folds = cv_folds
        self.max_iter = max_iter
        self.alpha_scale_factor = alpha_scale_factor
        self.feature_selection_threshold = feature_selection_threshold
        self.alpha_candidates_count = alpha_candidates_count
        self.alpha_log_scale_range = alpha_log_scale_range
        self.target_sparsity = target_sparsity
        self.sparsity_tolerance = sparsity_tolerance
        self.binary_search_tol = binary_search_tol
        self.initial_correction_factor = initial_correction_factor
        self.default_correction_factor = default_correction_factor
        self.min_eigenvalue_adjustment = min_eigenvalue_adjustment

        # Initialize dedicated random number generator for sampling operations
        self.sample_rng = np.random.RandomState(random_state)

        # Additional attributes
        self.precision_: Optional[np.ndarray] = None
        self.covariance_: Optional[np.ndarray] = None
        self.base_model_: Optional[Any] = None
        self._upper_tri_mask: Optional[np.ndarray] = None  # Cache for performance

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "NullstrapGGM":
        """
        Fit the Nullstrap Graphical Model estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored
            Not used for graphical models.

        Returns
        -------
        self : NullstrapGGM
            Fitted estimator.
        """
        self._validate_parameters()
        self._set_random_state()

        # Validate and convert data
        X = self._validate_data(X, y)

        self.n_samples_, self.n_features_ = X.shape

        # Standardize data for graphical models (standard z-score only)
        X_scaled, _ = standardize_data(X, scale_by_sample_size=False)

        # Fit base graphical LASSO model
        base_model, alpha_used = self._fit_lasso_model(X_scaled)
        self.base_model_ = base_model
        self.alpha_used_ = alpha_used

        # Get precision matrix and extract upper triangular elements
        self.precision_ = base_model.precision_
        self.covariance_ = base_model.covariance_

        # Extract upper triangular elements (excluding diagonal)
        self._upper_tri_mask = np.triu(np.ones_like(self.precision_), k=1).astype(bool)
        precision_upper_tri = np.abs(self.precision_[self._upper_tri_mask])

        # Generate knockoff data and fit model
        X_knockoff = self._generate_knockoff_data(
            self.n_samples_,
            self.n_features_,
            covariance_structure="identity",
            rng=self.sample_rng,
        )
        knockoff_model = GraphicalLasso(
            alpha=alpha_used, max_iter=self.max_iter, tol=1e-4
        )
        knockoff_model.fit(X_knockoff)
        knockoff_precision = knockoff_model.precision_
        knockoff_upper_tri = np.abs(knockoff_precision[self._upper_tri_mask])

        # Estimate correction factor
        correction_factor = self._estimate_correction_factor(
            X_scaled, None, precision_upper_tri
        )
        self.correction_factor_ = correction_factor

        # Apply correction to knockoff statistics
        scale_factor = self._compute_scale_factor(
            self.covariance_, alpha_used, self.n_samples_, self.n_features_
        )
        corrected_knockoff = knockoff_upper_tri + correction_factor * scale_factor

        # Compute test statistics and find threshold
        self.statistic_ = precision_upper_tri
        self.threshold_ = binary_search_threshold(
            self.statistic_, corrected_knockoff, self.fdr
        )

        # Get selected edges
        self.selected_ = self.get_selected_features(self.statistic_, self.threshold_)
        self.n_features_selected_ = len(self.selected_)

        return self

    def _validate_data(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Validate input data for GGM models.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like, optional
            Target values (ignored for GGM models).

        Returns
        -------
        X : ndarray
            Validated and converted X.
        """
        X = super()._validate_X(X)
        
        # GGM-specific validation
        if X.shape[0] < X.shape[1]:
            warnings.warn(
                f"More features ({X.shape[1]}) than samples ({X.shape[0]}). "
                "This may lead to numerical issues in graphical model estimation.",
                UserWarning,
            )
        
        return X

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
        X = self._validate_data(X)
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X must have {self.n_features_} features, got {X.shape[1]} features"
            )

        return X[:, self.selected_]

    def _fit_lasso_model(self, X: np.ndarray) -> Tuple[GraphicalLasso, float]:
        """Fit graphical LASSO model with alpha selection by CV or AIC."""
        if self.alpha_ is None:
            min_log10, max_log10 = self.alpha_log_scale_range
            alpha_candidates = np.logspace(
                min_log10, max_log10, self.alpha_candidates_count
            )

            if self.selection_method == "cv":
                # Use cross-validation to select alpha (GraphicalLassoCV uses log-likelihood)
                try:
                    gl_cv = GraphicalLassoCV(
                        alphas=alpha_candidates,
                        cv=self.cv_folds,
                        max_iter=self.max_iter,
                        tol=1e-4,  # Add tolerance for numerical stability
                    )
                    gl_cv.fit(X)
                    alpha_selected = self.alpha_scale_factor * gl_cv.alpha_
                except Exception as e:
                    warnings.warn(
                        f"Cross-validation failed: {e}. Using fallback alpha selection."
                    )
                    # Fallback to a conservative alpha
                    alpha_selected = alpha_candidates[len(alpha_candidates) // 2]

            elif self.selection_method == "aic":
                # Use AIC for alpha selection (matches R implementation)
                best_aic = np.inf
                best_alpha = alpha_candidates[0]
                successful_fits = 0

                for alpha in alpha_candidates:
                    try:
                        model = GraphicalLasso(
                            alpha=alpha, max_iter=self.max_iter, tol=1e-4
                        )
                        model.fit(X)

                        # Compute AIC: AIC = -2 * log_likelihood + 2 * df
                        # For graphical models: log_likelihood ≈ n/2 * log(det(Θ)) - n/2 * tr(SΘ)
                        # where Θ is precision matrix, S is sample covariance
                        precision = model.precision_
                        sample_cov = np.cov(X.T)

                        # Check for numerical stability
                        log_det_sign, log_det = np.linalg.slogdet(precision)
                        if log_det_sign <= 0:
                            continue  # Skip non-positive definite matrices

                        trace_term = np.trace(sample_cov @ precision)
                        log_likelihood = self.n_samples_ / 2 * (log_det - trace_term)

                        # Degrees of freedom = number of non-zero off-diagonal elements
                        upper_tri_mask = np.triu(np.ones_like(precision), k=1).astype(
                            bool
                        )
                        df = np.sum(
                            np.abs(precision[upper_tri_mask])
                            > self.feature_selection_threshold
                        )

                        aic = -2 * log_likelihood + 2 * df

                        if aic < best_aic:
                            best_aic = aic
                            best_alpha = alpha

                        successful_fits += 1

                    except (np.linalg.LinAlgError, ValueError):
                        continue

                # Check if we had any successful fits
                if successful_fits == 0:
                    warnings.warn(
                        "No successful AIC fits found, using default alpha", UserWarning
                    )
                    alpha_selected = self.alpha_scale_factor * alpha_candidates[0]
                else:
                    # Use alpha_scale_factor * best_alpha for conservative selection
                    alpha_selected = self.alpha_scale_factor * best_alpha

            else:
                raise ValueError(
                    f"selection_method must be 'cv' or 'aic', got '{self.selection_method}'"
                )
        else:
            alpha_selected = self.alpha_

        # Fit final model with selected alpha
        base_model = GraphicalLasso(
            alpha=alpha_selected, max_iter=self.max_iter, tol=1e-4
        )
        try:
            base_model.fit(X)
        except Exception as e:
            warnings.warn(
                f"GraphicalLasso fitting failed: {e}. Trying with increased regularization."
            )
            # Try with increased regularization
            base_model = GraphicalLasso(
                alpha=alpha_selected * 2.0,  # Increase regularization
                max_iter=self.max_iter,
                tol=1e-3,  # Relax tolerance
            )
            base_model.fit(X)

        return base_model, alpha_selected

    def _generate_knockoff_data(
        self,
        n_samples: int,
        n_features: int,
        covariance_structure: str = "identity",
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        """
        Generate knockoff data for Gaussian graphical models.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        n_features : int
            Number of features.
        covariance_structure : str, default="identity"
            Type of covariance structure for null data.
        rng : RandomState, optional
            Random number generator for reproducibility.

        Returns
        -------
        X_knockoff : ndarray of shape (n_samples, n_features)
            Knockoff data matrix.
        """
        random_gen = rng if rng is not None else self.sample_rng

        if covariance_structure == "identity":
            # Generate independent Gaussian variables
            X_knockoff = random_gen.multivariate_normal(
                np.zeros(n_features), np.eye(n_features), n_samples
            )
        else:
            raise ValueError(
                f"Unsupported covariance structure: {covariance_structure}"
            )

        return X_knockoff

    def _estimate_correction_factor(
        self, X: np.ndarray, y: Optional[np.ndarray], base_coefficients: np.ndarray
    ) -> float:
        """Estimate correction factor using multiple repetitions."""
        correction_factors = []

        for b in range(self.B_reps):
            # Create positive control coefficients (inflate signal for graphical models)
            # For graphical models, we inflate the precision matrix elements
            beta_real = inflate_signal(base_coefficients, self.alpha_used_, "additive")

            # Generate correction data
            X_correction = self._generate_correction_data(X, beta_real, b)

            # Find signal indices (non-zero elements in positive control)
            signal_indices = np.where(beta_real != 0)[0]

            # Fit models to correction and knockoff data
            coef_correction_abs, coef_snp_abs = self._fit_correction_models(
                X, X_correction, b
            )

            # Calculate model-specific scale factor for GGM models
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
                max_iterations=100,  # GGM has iteration limit
            )

            correction_factors.append(correction)

        # Return maximum correction factor across repetitions
        return max(correction_factors)

    def _generate_correction_data(
        self, X: np.ndarray, beta_real: np.ndarray, iteration: int
    ) -> np.ndarray:
        """Generate correction data for Gaussian graphical models."""
        iteration_rng = (
            np.random
            if self.random_state is None
            else np.random.RandomState(self.random_state + iteration)
        )

        # For graphical models, beta_real represents inflated precision matrix elements
        # We need to reconstruct a precision matrix from the upper triangular elements
        precision_real = np.zeros((self.n_features_, self.n_features_))
        upper_tri_mask = np.triu(np.ones_like(precision_real), k=1).astype(bool)
        precision_real[upper_tri_mask] = beta_real
        precision_real = precision_real + precision_real.T  # Make symmetric

        # Ensure positive definite with better numerical stability
        eigenvals = np.linalg.eigvals(precision_real)
        min_eigenval = np.min(eigenvals)
        if min_eigenval <= 0:
            # Add regularization to ensure positive definiteness
            regularization = max(
                self.min_eigenvalue_adjustment, abs(min_eigenval) + 1e-6
            )
            precision_real += np.eye(self.n_features_) * regularization

        # Generate data from the inflated precision matrix
        try:
            cov_real = np.linalg.inv(precision_real)
            X_correction = iteration_rng.multivariate_normal(
                np.zeros(self.n_features_), cov_real, self.n_samples_
            )
            X_correction, _ = standardize_data(X_correction, scale_by_sample_size=False)
        except np.linalg.LinAlgError:
            # Fallback to identity if inversion fails
            X_correction = iteration_rng.multivariate_normal(
                np.zeros(self.n_features_), np.eye(self.n_features_), self.n_samples_
            )

        return X_correction

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
        model_correction, _ = self._fit_lasso_model(y_correction)
        upper_tri_mask = np.triu(np.ones_like(model_correction.precision_), k=1).astype(
            bool
        )
        coef_correction_abs = np.abs(model_correction.precision_[upper_tri_mask])

        # Generate knockoff data
        X_snp = self._generate_knockoff_data(
            self.n_samples_,
            self.n_features_,
            covariance_structure="identity",
            rng=iteration_rng,
        )

        # Fit model to knockoff data
        model_snp, _ = self._fit_lasso_model(X_snp)
        coef_snp_abs = np.abs(model_snp.precision_[upper_tri_mask])

        return coef_correction_abs, coef_snp_abs

    def get_adjacency_matrix(self, threshold: Optional[float] = None) -> np.ndarray:
        """
        Get adjacency matrix of selected edges.

        Parameters
        ----------
        threshold : float, optional
            Selection threshold. If None, uses the fitted threshold.

        Returns
        -------
        adjacency : ndarray of shape (n_features, n_features)
            Binary adjacency matrix with 1s for selected edges.
        """
        if self.precision_ is None:
            raise ValueError("Model has not been fitted yet.")

        if threshold is None:
            threshold = self.threshold_

        # Create adjacency matrix
        adjacency = np.zeros((self.n_features_, self.n_features_))
        upper_tri_mask = np.triu(np.ones_like(self.precision_), k=1).astype(bool)

        # Mark selected edges
        selected_edges = np.abs(self.precision_[upper_tri_mask]) >= threshold

        # Fill upper triangular part
        adjacency[upper_tri_mask] = selected_edges.astype(int)

        # Make symmetric
        adjacency = adjacency + adjacency.T

        return adjacency

    def get_precision_matrix(self) -> np.ndarray:
        """
        Get the estimated precision matrix.

        Returns
        -------
        precision : ndarray of shape (n_features, n_features)
            Estimated precision matrix.
        """
        if self.precision_ is None:
            raise ValueError("Model has not been fitted yet.")
        return self.precision_.copy()

    def get_covariance_matrix(self) -> np.ndarray:
        """
        Get the estimated covariance matrix.

        Returns
        -------
        covariance : ndarray of shape (n_features, n_features)
            Estimated covariance matrix.
        """
        if self.covariance_ is None:
            raise ValueError("Model has not been fitted yet.")
        return self.covariance_.copy()

    def _compute_scale_factor(
        self,
        covariance_matrix: np.ndarray,
        alpha_reg: float,
        n_samples: int,
        n_features: int,
    ) -> float:
        """
        Compute scale factor for graphical models.

        Parameters
        ----------
        covariance_matrix : array-like of shape (n_features, n_features)
            Sample covariance matrix.
        alpha_reg : float
            Regularization parameter.
        n_samples : int
            Number of samples.
        n_features : int
            Number of features.

        Returns
        -------
        scale_factor : float
            Scale factor for correction.
        """
        # Compute infinity norm of covariance matrix
        inf_norm = np.max(np.sum(np.abs(covariance_matrix), axis=1)) / n_features

        return alpha_reg + inf_norm * np.sqrt(np.log(n_features) / n_samples)
