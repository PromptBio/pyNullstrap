"""Nullstrap estimator for Gaussian Graphical Models"""
from typing import Optional, Tuple, Union, Sequence
import warnings

import numpy as np
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV

from ..estimator import BaseNullstrap
from ..utils.core import (binary_search_correction_factor,
                          binary_search_threshold, inflate_signal,
                          standardize_data)


class NullstrapGGM(BaseNullstrap):
    """
    Nullstrap selector for Gaussian Graphical Models with FDR control via graphical LASSO.
    For graphical models, "features" refer to edges in the graph.
    
    Parameters
    ----------
    fdr : float, default=0.1
        Target false discovery rate.
    alpha_ : float, optional
        Regularization parameter for graphical LASSO (maps to 'lambda' in glmnet). If None, selected by selection_method.
    B_reps : int, default=5
        Number of repetitions for correction factor estimation.
    cv_folds : int, default=10
        Number of cross-validation folds for alpha selection.
    max_iter : int, default=1000
        Maximum number of iterations for graphical LASSO optimization.
    lasso_tol : float, default=1e-4
        Convergence tolerance for graphical LASSO optimization.
    alpha_scale_factor : float, default=0.5
        Scaling factor for selected alpha (more conservative).
    binary_search_tol : float, default=1e-8
        Convergence tolerance for binary search.
    correction_min : float, default=0.05
        Minimum bound for correction factor search.
    alphas : int or array-like, default=50
        Alphas for CV: int (number of alphas) or array (explicit values).
        Matches sklearn's GraphicalLassoCV parameter.
    selection_method : str, default='cv'
        Alpha selection methods:
        - 'cv': Uses cross-validation with log-likelihood scoring (default, more robust)
        - 'aic':  Uses AIC-based selection (matches original R implementation)
    edge_threshold : float, default=1e-6
        Threshold for considering an edge as non-zero in the precision matrix.
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
        Test statistics (absolute precision matrix elements).
    alpha_used_ : float
        Regularization parameter used in final fit.
    correction_factor_ : float
        Estimated correction factor for FDR control.
    precision_ : ndarray
        Estimated precision matrix.
    covariance_ : ndarray
        Estimated covariance matrix.
    """

    def __init__(
        self,
        fdr: float = 0.1,
        alpha_: Optional[float] = None,
        B_reps: int = 5,
        cv_folds: int = 10,
        max_iter: int = 1000,
        lasso_tol: float = 1e-4,
        alpha_scale_factor: float = 0.5,
        binary_search_tol: float = 1e-8,
        correction_min: float = 0.05,
        alphas: Union[int, Sequence[float]] = 50,
        selection_method: str = "cv",
        edge_threshold: float = 1e-6,
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
        self.selection_method = selection_method
        self.edge_threshold = edge_threshold

        # Initialize dedicated random number generator for sampling operations
        self.sample_rng = np.random.RandomState(random_state)

        # Additional attributes for graphical models
        self.precision_: Optional[np.ndarray] = None
        self.covariance_: Optional[np.ndarray] = None

    def _validate_parameters(self) -> None:
        """
        Validate GGM-specific parameters. Checks that selection_method and alphas are valid.
        """
        # Call base class validation first
        super()._validate_parameters()
        
        # GGM-specific validation
        if self.selection_method not in ["cv", "aic"]:
            raise ValueError(f"selection_method must be 'cv' or 'aic', got '{self.selection_method}'")
        
        # Validate alphas parameter
        if isinstance(self.alphas, int):
            if self.alphas <= 0:
                raise ValueError(f"alphas (int) must be positive, got {self.alphas}")
        elif hasattr(self.alphas, '__iter__') and not isinstance(self.alphas, str):
            # Check if it's array-like (but not a string)
            try:
                alphas_array = np.array(self.alphas)
                if alphas_array.ndim == 0:
                    raise ValueError("alphas must be an int or array-like, got scalar")
                if len(alphas_array) == 0:
                    raise ValueError("alphas array cannot be empty")
                if not np.all(alphas_array > 0):
                    raise ValueError("all alphas values must be positive")
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid alphas array: {e}")
        else:
            raise ValueError(f"alphas must be an int or array-like, got {type(self.alphas).__name__}")

    def _validate_data(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Validate input data for graphical models.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray, optional
            Target values. Not used for graphical models (kept for API consistency).
        """
        super()._validate_X(X)
        
        # GGM-specific validation
        if X.shape[0] < X.shape[1]:
            warnings.warn(
                f"More features ({X.shape[1]}) than samples ({X.shape[0]}). "
                "This may lead to numerical issues in graphical model estimation.",
                UserWarning,
            )

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "NullstrapGGM":
        """
        Fit the Nullstrap Gaussian Graphical Model estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), optional
            Target values. Not used for graphical models.

        Returns
        -------
        self : NullstrapGGM
            Fitted estimator.
        """
        # Validate parameters and input data
        self._validate_parameters()
        self._validate_data(X, y=None)
        self._set_random_state()

        self.n_samples_, self.n_features_ = X.shape

        # Standardize data for graphical models (standard z-score only)
        X_scaled, _ = standardize_data(X, scale_by_sample_size=False)

        # Fit base graphical LASSO model
        model_base, alpha_used = self._fit_lasso_model(X_scaled, alpha=self.alpha_)
        self.alpha_used_ = alpha_used
        
        # Get precision matrix and extract upper triangular elements
        self.precision_ = model_base.precision_
        self.covariance_ = model_base.covariance_

        # Extract upper triangular elements (excluding diagonal)
        upper_tri_mask = np.triu(np.ones_like(self.precision_), k=1).astype(bool)
        coef_base = self.precision_[upper_tri_mask]

        # Generate knockoff data, fit model to knockoff data
        X_knockoff = self._generate_synthetic_data(X_scaled, covariance_structure="identity")
        model_knockoff, _ = self._fit_lasso_model(X_knockoff, alpha=self.alpha_used_)
        coef_knockoff = model_knockoff.precision_[upper_tri_mask]

        # Estimate correction factor
        correction_factor = self._estimate_correction_factor(X_scaled, coef_base)
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
        alpha: Optional[float] = None
    ) -> Tuple[GraphicalLasso, float]:
        """
        Fit graphical LASSO with specified or cross-validated regularization parameter.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data matrix (standardized).
        alpha : float, optional
            Regularization parameter (corresponds to lambda in glmnet).
            If None, selected via CV.

        Returns
        -------
        model : GraphicalLasso
            Fitted graphical LASSO model.
        alpha_used : float
            Regularization parameter used, scaled by alpha_scale_factor (default 0.5)
            for conservative selection.

        Notes
        -----
        Uses GraphicalLassoCV when alpha is None (selection_method='cv') or AIC-based
        selection (selection_method='aic'). If self.alphas is an array, uses those values
        for selection; if integer, generates that many alphas automatically using logspace.
        Model fitted on pre-standardized data.
        """
        if alpha is None:
            # Generate or use alpha candidates (validated in _validate_parameters)
            if isinstance(self.alphas, int):
                # Generate alpha candidates using logspace (default range: 10^-3 to 10^1)
                alpha_candidates = np.logspace(-3, 1, self.alphas)
            else:
                # Use provided alpha values
                alpha_candidates = np.array(self.alphas)

            if self.selection_method == "cv":
                # Use cross-validation to select alpha (GraphicalLassoCV uses log-likelihood)
                try:
                    glasso_cv = GraphicalLassoCV(alphas=alpha_candidates, cv=self.cv_folds, max_iter=self.max_iter, tol=self.lasso_tol)
                    glasso_cv.fit(X)
                    # Scale selected alpha by alpha_scale_factor (more conservative)
                    alpha_used = self.alpha_scale_factor * glasso_cv.alpha_
                except Exception as e:
                    warnings.warn(
                        f"Cross-validation failed: {e}. Using fallback alpha selection."
                    )
                    # Fallback to a conservative alpha (middle of valid range)
                    alpha_used = self.alpha_scale_factor * alpha_candidates[len(alpha_candidates) // 2]

            elif self.selection_method == "aic":
                # Use AIC for alpha selection (matches R implementation)
                best_aic = np.inf
                best_alpha = alpha_candidates[0]
                successful_fits = 0

                for alpha in alpha_candidates:
                    try:
                        model = GraphicalLasso(alpha=alpha, max_iter=self.max_iter, tol=self.lasso_tol)
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
                        upper_tri_mask = np.triu(np.ones_like(precision), k=1).astype(bool)
                        df = np.count_nonzero(np.abs(precision[upper_tri_mask]) > self.edge_threshold)

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
                    alpha_used = self.alpha_scale_factor * alpha_candidates[0]
                else:
                    # Use alpha_scale_factor * best_alpha for conservative selection
                    alpha_used = self.alpha_scale_factor * best_alpha

            else:
                raise ValueError(
                    f"selection_method must be 'cv' or 'aic', got '{self.selection_method}'"
                )
        else:
            alpha_used = alpha

        # Fit final model with selected alpha
        model = GraphicalLasso(alpha=alpha_used, max_iter=self.max_iter, tol=self.lasso_tol)
        try:
            model.fit(X)
        except Exception as e:
            warnings.warn(
                f"GraphicalLasso fitting failed: {e}. Trying with increased regularization."
            )
            # Try with increased regularization (use relaxed tolerance for stability)
            model = GraphicalLasso(alpha=alpha_used * 2.0, max_iter=self.max_iter, tol=self.lasso_tol * 10)
            model.fit(X)

        return model, alpha_used

    def _generate_synthetic_data(
        self,
        X: np.ndarray,
        precision: Optional[np.ndarray] = None,
        covariance_structure: str = "identity",
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        """
        Generate synthetic data: X ~ N(0, Sigma) where Sigma = precision^(-1) (or identity for null data).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data matrix.
        precision : ndarray of shape (n_features, n_features), optional
            Precision matrix for signal generation. If None, generates null/knockoff data (identity covariance).
        covariance_structure : str, default="identity"
            Type of covariance structure for null data (when precision is None).
        rng : np.random.RandomState, optional
            Random generator. If None, uses self.sample_rng.

        Returns
        -------
        ndarray of shape (n_samples, n_features)
            Synthetic data matrix.
        """
        random_gen = self.sample_rng if rng is None else rng

        if precision is None:
            # Generate null/knockoff data with specified covariance structure
            if covariance_structure == "identity":
                X_synthetic = random_gen.multivariate_normal(
                    np.zeros(self.n_features_), 
                    np.eye(self.n_features_), 
                    self.n_samples_
                )
            else:
                raise ValueError(f"Unsupported covariance structure: {covariance_structure}")
        else:
            # Generate data from precision matrix (signal + structure)
            try:
                covariance = np.linalg.inv(precision)
                X_synthetic = random_gen.multivariate_normal(
                    np.zeros(self.n_features_), 
                    covariance, 
                    self.n_samples_
                )
            except np.linalg.LinAlgError:
                # Fallback to identity if inversion fails
                warnings.warn("Precision matrix inversion failed, using identity covariance.")
                X_synthetic = random_gen.multivariate_normal(
                    np.zeros(self.n_features_), 
                    np.eye(self.n_features_), 
                    self.n_samples_
                )

        return X_synthetic

    def _estimate_correction_factor(
        self, 
        X: np.ndarray, 
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
            Data matrix (standardized).
        coef_base : ndarray of shape (n_edges,)
            Precision matrix elements from original model fit.

        Returns
        -------
        correction_factor : float
            Maximum correction factor across repetitions, used to adjust knockoff
            statistics for FDR control.
        """
        correction_factors = []

        for b in range(self.B_reps):
            # Create positive control coefficients (synthetic truth), get signal indices
            precision_augmented = inflate_signal(coef_base, self.alpha_used_, "additive")
            signal_indices = np.where(precision_augmented != 0)[0]

            # Reconstruct symmetric precision matrix from upper triangular elements
            precision_matrix = np.zeros((self.n_features_, self.n_features_))
            i_upper, j_upper = np.triu_indices(self.n_features_, k=1)
            precision_matrix[i_upper, j_upper] = precision_augmented  # Upper triangle
            precision_matrix[j_upper, i_upper] = precision_augmented  # Mirror to lower triangle
            # Use diagonal from original precision matrix (preserve conditional variances)
            np.fill_diagonal(precision_matrix, np.diag(self.precision_))

            # Check if positive definite (should rarely fail with preserved diagonal)
            eigenvals = np.linalg.eigvals(precision_matrix)
            min_eigenval = np.min(eigenvals)
            if min_eigenval <= 0:
                regularization = abs(min_eigenval) + 0.01
                precision_matrix += np.eye(self.n_features_) * regularization

            # Create iteration-specific RNG
            iteration_rng = None if self.random_state is None else np.random.RandomState(self.random_state + b)

            # Generate augmented and knockoff data
            X_augmented = self._generate_synthetic_data(X, precision=precision_matrix, rng=iteration_rng)
            X_knockoff  = self._generate_synthetic_data(X, precision=None, rng=iteration_rng)

            # Standardize augmented data
            X_augmented, _ = standardize_data(X_augmented, scale_by_sample_size=False)
            X_knockoff, _  = standardize_data(X_knockoff, scale_by_sample_size=False)

            # Fit models to augmented and knockoff data (estimate precision matrices)
            model_augmented, _ = self._fit_lasso_model(X_augmented, alpha=self.alpha_used_)
            model_knockoff, _  = self._fit_lasso_model(X_knockoff, alpha=self.alpha_used_)
            coef_augmented = model_augmented.precision_[i_upper, j_upper]
            coef_knockoff = model_knockoff.precision_[i_upper, j_upper]
            
            # Calculate model-specific scale factor for graphical models
            scale_factor = self._compute_scale_factor()
            correction_min = self.correction_min
            correction_max = np.max(np.abs(coef_augmented)) / scale_factor if scale_factor > 0 else np.max(np.abs(coef_augmented))

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
        Compute scale factor for correction: alpha + (p / ||Sigma||_1) * sqrt(log(p) / n).

        Returns
        -------
        float
            Scale factor for knockoff correction.
        
        Notes
        -----
        Uses instance attributes: covariance_, alpha_used_, n_samples_, n_features_.
        For graphical models, uses the matrix 1-norm of the covariance matrix:
        - Matrix 1-norm: ||A||_1 = max_j sum_i |a_ij| (maximum absolute column sum)
        - Matrix infinity-norm: ||A||_inf = max_i sum_j |a_ij| (maximum absolute row sum)
        For symmetric matrices like covariance, ||A||_1 = ||A||_inf
        
        Matches R implementation: inf_norm = 1/((norm(K_gamma1, type = "1")/p)) = p / ||Sigma||_1
        """
        # Compute matrix 1-norm (or equivalently infinity-norm for symmetric matrix)
        sigma_norm = np.max(np.sum(np.abs(self.covariance_), axis=1))
        return (self.alpha_used_ + np.sqrt(np.log(self.n_features_) / self.n_samples_)) * (self.n_features_ / sigma_norm)
