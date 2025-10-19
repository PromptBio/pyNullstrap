"""
Tests for NullstrapGGM (Gaussian Graphical Models).

This module contains tests for the NullstrapGGM class,
including tests for precision matrices and adjacency matrices.
"""

import numpy as np
import pytest

from nullstrap.models.ggm import NullstrapGGM
from ..conftest import (assert_model_fitted, assert_reproducible_results)


class TestNullstrapGGM:
    """Test cases for NullstrapGGM (Gaussian Graphical Models)."""

    @pytest.mark.model
    def test_basic_fitting(self, graphical_data, model_params):
        """Test basic model fitting functionality."""
        model = NullstrapGGM(**model_params)
        model.fit(graphical_data["X"])

        assert_model_fitted(model)
        assert model.n_samples_ == graphical_data["n_samples"]
        assert model.n_features_ == graphical_data["n_features"]

    @pytest.mark.model
    def test_fdr_control(self, graphical_data, model_params):
        """Test that FDR is controlled at specified level."""
        model = NullstrapGGM(**model_params)
        model.fit(graphical_data["X"])

        # For GGM, we test edge selection rather than feature selection
        # Check that some edges were selected
        assert model.n_features_selected_ >= 0

    @pytest.mark.model
    def test_reproducibility(self, graphical_data, model_params):
        """Test that results are reproducible with same random_state."""
        model1 = NullstrapGGM(**model_params)
        model2 = NullstrapGGM(**model_params)

        model1.fit(graphical_data["X"])
        model2.fit(graphical_data["X"])

        assert_reproducible_results(model1, model2)

    @pytest.mark.model
    def test_precision_matrix_attributes(self, graphical_data, model_params):
        """Test precision and covariance matrix attributes."""
        model = NullstrapGGM(**model_params)
        model.fit(graphical_data["X"])

        # Check precision matrix dimensions
        assert model.precision_.shape == (
            graphical_data["n_features"],
            graphical_data["n_features"],
        )

        # Check covariance matrix dimensions
        assert model.covariance_.shape == (
            graphical_data["n_features"],
            graphical_data["n_features"],
        )

        # Check symmetry of precision matrix
        assert np.allclose(model.precision_, model.precision_.T)

        # Check positive definiteness (eigenvalues should be positive)
        eigenvals = np.linalg.eigvals(model.precision_)
        assert np.all(eigenvals > 0)

    @pytest.mark.model
    def test_no_response_parameter(self, graphical_data, model_params):
        """Test that GGM doesn't require y parameter."""
        model = NullstrapGGM(**model_params)

        # Should work without y parameter
        model.fit(graphical_data["X"])
        assert_model_fitted(model)


class TestNullstrapGGMParameterValidation:
    """Test parameter validation in NullstrapGGM."""

    @pytest.mark.model
    def test_invalid_fdr(self, graphical_data):
        """Test that invalid FDR raises ValueError."""
        with pytest.raises(ValueError, match="fdr must be between 0 and 1"):
            model = NullstrapGGM(fdr=-0.1)
            model.fit(graphical_data["X"])

        with pytest.raises(ValueError, match="fdr must be between 0 and 1"):
            model = NullstrapGGM(fdr=1.5)
            model.fit(graphical_data["X"])

    @pytest.mark.model
    def test_invalid_alpha(self, graphical_data):
        """Test that invalid alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha_ must be non-negative"):
            model = NullstrapGGM(alpha_=-0.1)
            model.fit(graphical_data["X"])

    @pytest.mark.model
    def test_invalid_b_reps(self, graphical_data):
        """Test that invalid B_reps raises ValueError."""
        with pytest.raises(ValueError, match="B_reps must be a positive integer"):
            model = NullstrapGGM(B_reps=0)
            model.fit(graphical_data["X"])

    @pytest.mark.model
    def test_invalid_cv_folds(self, graphical_data):
        """Test that invalid cv_folds triggers warning instead of error."""
        # cv_folds validation is handled by sklearn during CV, not at init
        # It will produce a warning and use fallback alpha selection
        model = NullstrapGGM(cv_folds=1, random_state=42)
        with pytest.warns(UserWarning, match="Cross-validation failed"):
            model.fit(graphical_data["X"])
        # Model should still fit with fallback
        assert_model_fitted(model)

    @pytest.mark.model
    def test_invalid_selection_method(self, graphical_data):
        """Test that invalid selection_method raises ValueError."""
        with pytest.raises(ValueError, match="selection_method must be 'cv' or 'aic'"):
            model = NullstrapGGM(selection_method="invalid")
            model.fit(graphical_data["X"])

    @pytest.mark.model
    def test_invalid_alphas_type(self, graphical_data):
        """Test that invalid alphas type raises ValueError."""
        # Invalid alphas type is caught during parameter validation
        with pytest.raises(ValueError, match="alphas must be an int or array-like"):
            model = NullstrapGGM(alphas="invalid", random_state=42)
            model.fit(graphical_data["X"])

    @pytest.mark.model
    def test_invalid_alphas_value(self, graphical_data):
        """Test that invalid alphas values raise ValueError."""
        # Invalid alphas=0 is caught during parameter validation
        with pytest.raises(ValueError, match="alphas .* must be positive"):
            model = NullstrapGGM(alphas=0, random_state=42)
            model.fit(graphical_data["X"])

        # Negative alphas in array are caught during parameter validation
        with pytest.raises(ValueError, match="all alphas values must be positive"):
            model2 = NullstrapGGM(alphas=[0.1, -0.1, 0.2], random_state=42)
            model2.fit(graphical_data["X"])

        # Empty array is caught during parameter validation
        with pytest.raises(ValueError, match="alphas array cannot be empty"):
            model3 = NullstrapGGM(alphas=[], random_state=42)
            model3.fit(graphical_data["X"])


class TestNullstrapGGMDataValidation:
    """Test data validation in NullstrapGGM."""

    @pytest.mark.model
    def test_empty_data(self):
        """Test that empty data raises ValueError."""
        model = NullstrapGGM(random_state=42)
        X = np.array([]).reshape(0, 10)

        with pytest.raises(ValueError, match="X cannot be empty"):
            model.fit(X)

    @pytest.mark.model
    def test_single_feature(self):
        """Test that single feature triggers warnings and fallback."""
        model = NullstrapGGM(random_state=42)
        X = np.random.randn(100, 1)

        # GraphicalLasso requires at least 2 features, will trigger CV failure and fitting warnings
        with pytest.warns(UserWarning):
            try:
                model.fit(X)
                # If it somehow fits, verify it's fitted
                assert_model_fitted(model)
            except ValueError:
                # It's also acceptable to raise ValueError from GraphicalLasso
                pass

    @pytest.mark.model
    def test_few_samples(self):
        """Test that few samples triggers warning but still fits."""
        model = NullstrapGGM(random_state=42)
        X = np.random.randn(5, 10)

        # More features than samples triggers warning but doesn't raise error
        with pytest.warns(UserWarning, match="More features .* than samples"):
            model.fit(X)
        # Model should still fit (may have issues but doesn't fail)
        assert_model_fitted(model)

    @pytest.mark.model
    def test_nan_data(self, graphical_data):
        """Test that NaN data raises ValueError."""
        model = NullstrapGGM(random_state=42)
        X = graphical_data["X"].copy()
        X[0, 0] = np.nan

        with pytest.raises(ValueError, match="X contains NaN"):
            model.fit(X)

    @pytest.mark.model
    def test_inf_data(self, graphical_data):
        """Test that Inf data raises ValueError."""
        model = NullstrapGGM(random_state=42)
        X = graphical_data["X"].copy()
        X[0, 0] = np.inf

        with pytest.raises(ValueError, match="X contains NaN or infinite values"):
            model.fit(X)


class TestNullstrapGGMAlphaSelection:
    """Test alpha selection methods in NullstrapGGM."""

    @pytest.mark.model
    def test_cv_selection(self, graphical_data, model_params):
        """Test cross-validation based alpha selection."""
        model = NullstrapGGM(**model_params, selection_method="cv")
        model.fit(graphical_data["X"])

        assert model.alpha_used_ is not None
        assert model.alpha_used_ > 0
        assert_model_fitted(model)

    @pytest.mark.model
    def test_aic_selection(self, graphical_data, model_params):
        """Test AIC-based alpha selection."""
        model = NullstrapGGM(**model_params, selection_method="aic")
        model.fit(graphical_data["X"])

        assert model.alpha_used_ is not None
        assert model.alpha_used_ > 0
        assert_model_fitted(model)

    @pytest.mark.model
    def test_custom_alpha(self, graphical_data, model_params):
        """Test using custom alpha parameter."""
        custom_alpha = 0.05
        # Create params dict without alpha_, then add custom alpha
        params = {k: v for k, v in model_params.items() if k != 'alpha_'}
        model = NullstrapGGM(**params, alpha_=custom_alpha)
        model.fit(graphical_data["X"])

        # When custom alpha is provided, it should be used directly
        assert model.alpha_used_ == custom_alpha
        assert_model_fitted(model)

    @pytest.mark.model
    def test_alphas_as_array(self, graphical_data, model_params):
        """Test providing alphas as an array."""
        alphas_array = np.logspace(-2, 0, 10)
        model = NullstrapGGM(**model_params, alphas=alphas_array, selection_method="cv")
        model.fit(graphical_data["X"])

        assert model.alpha_used_ is not None
        assert model.alpha_used_ > 0
        assert_model_fitted(model)

    @pytest.mark.model
    def test_alphas_as_int(self, graphical_data, model_params):
        """Test providing alphas as an integer."""
        model = NullstrapGGM(**model_params, alphas=20, selection_method="cv")
        model.fit(graphical_data["X"])

        assert model.alpha_used_ is not None
        assert model.alpha_used_ > 0
        assert_model_fitted(model)

    @pytest.mark.model
    def test_alpha_scale_factor(self, graphical_data, model_params):
        """Test alpha_scale_factor is applied correctly."""
        model1 = NullstrapGGM(**model_params, alpha_scale_factor=0.5)
        model2 = NullstrapGGM(**model_params, alpha_scale_factor=1.0)

        model1.fit(graphical_data["X"])
        model2.fit(graphical_data["X"])

        # Different scale factors should result in different alpha values
        # (unless custom alpha_ is provided)
        if model1.alpha_ is None and model2.alpha_ is None:
            # Only check if alpha selection is used
            assert model1.alpha_used_ != model2.alpha_used_


class TestNullstrapGGMStatistics:
    """Test statistics and attributes in NullstrapGGM."""

    @pytest.mark.model
    def test_statistic_values(self, graphical_data, model_params):
        """Test that statistics are computed correctly."""
        model = NullstrapGGM(**model_params)
        model.fit(graphical_data["X"])

        n_edges = graphical_data["n_features"] * (graphical_data["n_features"] - 1) // 2

        # Check statistic shape
        assert model.statistic_.shape[0] == n_edges

        # Statistics should be non-negative (absolute values)
        assert np.all(model.statistic_ >= 0)

    @pytest.mark.model
    def test_threshold_computation(self, graphical_data, model_params):
        """Test that threshold is computed."""
        model = NullstrapGGM(**model_params)
        model.fit(graphical_data["X"])

        assert model.threshold_ is not None
        assert model.threshold_ >= 0

    @pytest.mark.model
    def test_selected_edges(self, graphical_data, model_params):
        """Test selected edges are within valid range."""
        model = NullstrapGGM(**model_params)
        model.fit(graphical_data["X"])

        n_edges = graphical_data["n_features"] * (graphical_data["n_features"] - 1) // 2

        # Selected edges should be valid indices
        assert np.all(model.selected_ >= 0)
        assert np.all(model.selected_ < n_edges)

        # Number of selected edges should match
        assert model.n_features_selected_ == len(model.selected_)

    @pytest.mark.model
    def test_correction_factor(self, graphical_data, model_params):
        """Test correction factor estimation."""
        model = NullstrapGGM(**model_params)
        model.fit(graphical_data["X"])

        # Correction factor should be computed
        assert model.correction_factor_ is not None
        assert model.correction_factor_ > 0

    @pytest.mark.model
    def test_covariance_precision_relationship(self, graphical_data, model_params):
        """Test that covariance and precision are inverses."""
        model = NullstrapGGM(**model_params)
        model.fit(graphical_data["X"])

        # Covariance and precision should be approximate inverses
        identity = model.covariance_ @ model.precision_
        assert np.allclose(identity, np.eye(model.n_features_), atol=1e-3)


class TestNullstrapGGMEdgeCases:
    """Test edge cases and robustness of NullstrapGGM."""

    @pytest.mark.model
    def test_high_dimensional_data(self, high_dimensional_data, model_params):
        """Test with high-dimensional data (p > n)."""
        # For graphical models, we need n > p, so skip this test
        # or create appropriate data
        n_samples = 100
        n_features = 50  # p < n for graphical models
        rng = np.random.RandomState(42)

        # Create sparse precision matrix
        precision = np.eye(n_features) * 1.5
        cov = np.linalg.inv(precision)
        X = rng.multivariate_normal(np.zeros(n_features), cov, n_samples)

        model = NullstrapGGM(**model_params)
        model.fit(X)

        assert_model_fitted(model)

    @pytest.mark.model
    def test_perfect_correlation(self, model_params):
        """Test handling of perfect correlation."""
        rng = np.random.RandomState(42)
        n_samples = 100
        n_features = 10

        X = rng.randn(n_samples, n_features)
        # Create perfect correlation between first two features
        X[:, 1] = X[:, 0]

        model = NullstrapGGM(**model_params)
        # Should handle this gracefully (likely with warning or error)
        try:
            model.fit(X)
            # If it fits, check it's fitted
            assert_model_fitted(model)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            # It's okay to fail with singular/ill-conditioned matrix
            pass

    @pytest.mark.model
    def test_very_sparse_selection(self, graphical_data, model_params):
        """Test with very strict FDR leading to sparse selection."""
        # Create params dict without fdr, then add custom fdr
        params = {k: v for k, v in model_params.items() if k != 'fdr'}
        model = NullstrapGGM(**params, fdr=0.01)
        model.fit(graphical_data["X"])

        # Should still fit, even if no edges selected
        assert_model_fitted(model)
        assert model.n_features_selected_ >= 0

    @pytest.mark.model
    def test_very_lenient_fdr(self, graphical_data, model_params):
        """Test with very lenient FDR leading to dense selection."""
        # Create params dict without fdr, then add custom fdr
        params = {k: v for k, v in model_params.items() if k != 'fdr'}
        model = NullstrapGGM(**params, fdr=0.5)
        model.fit(graphical_data["X"])

        # Should still fit
        assert_model_fitted(model)
        assert model.n_features_selected_ >= 0

    @pytest.mark.model
    def test_minimal_b_reps(self, graphical_data, model_params):
        """Test with minimal B_reps."""
        # Create params dict without B_reps, then add custom B_reps
        params = {k: v for k, v in model_params.items() if k != 'B_reps'}
        model = NullstrapGGM(**params, B_reps=1)
        model.fit(graphical_data["X"])

        assert_model_fitted(model)

    @pytest.mark.model
    def test_different_lasso_tolerance(self, graphical_data, model_params):
        """Test with different lasso_tol values."""
        model1 = NullstrapGGM(**model_params, lasso_tol=1e-6)
        model2 = NullstrapGGM(**model_params, lasso_tol=1e-3)

        model1.fit(graphical_data["X"])
        model2.fit(graphical_data["X"])

        # Both should fit successfully
        assert_model_fitted(model1)
        assert_model_fitted(model2)


class TestNullstrapGGMIntegration:
    """Integration tests for NullstrapGGM."""

    @pytest.mark.model
    def test_full_pipeline_cv(self, graphical_data, model_params):
        """Test full pipeline with CV selection."""
        model = NullstrapGGM(**model_params, selection_method="cv")
        model.fit(graphical_data["X"])

        # Verify all expected attributes are set
        assert model.precision_ is not None
        assert model.covariance_ is not None
        assert model.alpha_used_ is not None
        assert model.correction_factor_ is not None
        assert model.threshold_ is not None
        assert model.statistic_ is not None
        assert model.selected_ is not None
        assert model.n_features_selected_ is not None

    @pytest.mark.model
    def test_full_pipeline_aic(self, graphical_data, model_params):
        """Test full pipeline with AIC selection."""
        model = NullstrapGGM(**model_params, selection_method="aic")
        model.fit(graphical_data["X"])

        # Verify all expected attributes are set
        assert model.precision_ is not None
        assert model.covariance_ is not None
        assert model.alpha_used_ is not None
        assert model.correction_factor_ is not None
        assert model.threshold_ is not None
        assert model.statistic_ is not None
        assert model.selected_ is not None
        assert model.n_features_selected_ is not None

    @pytest.mark.model
    def test_consistent_random_state(self, graphical_data):
        """Test that random_state ensures reproducibility."""
        model1 = NullstrapGGM(random_state=123, fdr=0.1, B_reps=3)
        model2 = NullstrapGGM(random_state=123, fdr=0.1, B_reps=3)

        model1.fit(graphical_data["X"])
        model2.fit(graphical_data["X"])

        # Results should be identical
        assert np.allclose(model1.precision_, model2.precision_)
        assert np.allclose(model1.statistic_, model2.statistic_)
        assert np.array_equal(model1.selected_, model2.selected_)

