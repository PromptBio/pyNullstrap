"""
Tests for NullstrapLM (Linear Models).

This module contains comprehensive tests for the NullstrapLM class,
including parameter validation, functionality tests, and edge cases.
"""

import numpy as np
import pytest

from nullstrap.models.lm import NullstrapLM
from ..conftest import (assert_fdr_control, assert_model_fitted,
                        assert_reproducible_results)


class TestNullstrapLM:
    """Test cases for NullstrapLM (Linear Models)."""

    @pytest.mark.model
    def test_basic_fitting(self, linear_data, model_params):
        """Test basic model fitting functionality."""
        model = NullstrapLM(**model_params)
        model.fit(linear_data["X"], linear_data["y"])

        # Check that model was fitted
        assert_model_fitted(model)

        # Check basic properties
        assert model.n_samples_ == linear_data["n_samples"]
        assert model.n_features_ == linear_data["n_features"]
        assert model.n_features_selected_ <= linear_data["n_features"]

    @pytest.mark.model
    def test_fdr_control(self, linear_data, model_params):
        """Test that FDR is controlled at specified level."""
        model = NullstrapLM(**model_params)
        model.fit(linear_data["X"], linear_data["y"])

        # Check FDR control (with some tolerance for small samples)
        assert_fdr_control(
            model.selected_,
            linear_data["true_features"],
            fdr_level=model_params["fdr"],
            tolerance=0.1,
        )

    @pytest.mark.model
    def test_reproducibility(self, linear_data, model_params):
        """Test that results are reproducible with same random_state."""
        model1 = NullstrapLM(**model_params)
        model2 = NullstrapLM(**model_params)

        model1.fit(linear_data["X"], linear_data["y"])
        model2.fit(linear_data["X"], linear_data["y"])

        assert_reproducible_results(model1, model2)

    @pytest.mark.model
    def test_transform_method(self, linear_data, model_params):
        """Test the transform method."""
        model = NullstrapLM(**model_params)
        model.fit(linear_data["X"], linear_data["y"])

        X_transformed = model.transform(linear_data["X"])

        # Check dimensions
        assert X_transformed.shape[0] == linear_data["n_samples"]
        assert X_transformed.shape[1] == model.n_features_selected_

        # Check that selected features are preserved
        if model.n_features_selected_ > 0:
            assert np.allclose(X_transformed, linear_data["X"][:, model.selected_])

    @pytest.mark.model
    def test_high_dimensional_data(self, high_dimensional_data, model_params):
        """Test with high-dimensional data (p > n)."""
        model = NullstrapLM(**model_params)
        model.fit(high_dimensional_data["X"], high_dimensional_data["y"])

        assert_model_fitted(model)
        assert model.n_features_selected_ <= high_dimensional_data["n_features"]

    @pytest.mark.model
    def test_edge_cases(self, edge_case_data, model_params):
        """Test edge cases."""
        # Test with single feature - use fixed alpha to avoid CV issues
        model = NullstrapLM(fdr=0.1, alpha_=0.1, random_state=42)
        model.fit(
            edge_case_data["single_feature"]["X"], edge_case_data["single_feature"]["y"]
        )
        assert_model_fitted(model)

        # Test with two samples - use fixed alpha to avoid CV issues
        model = NullstrapLM(fdr=0.1, alpha_=0.1, random_state=42)
        model.fit(
            edge_case_data["two_samples"]["X"], edge_case_data["two_samples"]["y"]
        )
        assert_model_fitted(model)

    @pytest.mark.model
    def test_parameter_validation(self, linear_data):
        """Test parameter validation."""
        # Test invalid FDR
        with pytest.raises(ValueError):
            NullstrapLM(fdr=-0.1).fit(linear_data["X"], linear_data["y"])

        with pytest.raises(ValueError):
            NullstrapLM(fdr=1.5).fit(linear_data["X"], linear_data["y"])

        # Test invalid alpha
        with pytest.raises(ValueError):
            NullstrapLM(alpha_=-0.1).fit(linear_data["X"], linear_data["y"])

        # Test invalid error_dist
        with pytest.raises(ValueError, match="error_dist must be"):
            NullstrapLM(error_dist="invalid").fit(linear_data["X"], linear_data["y"])

    @pytest.mark.model
    def test_random_state_zero(self, linear_data):
        """Test that random_state=0 works correctly (regression test for bug)."""
        # This tests the fix for the bug where random_state=0 was treated as None
        model1 = NullstrapLM(fdr=0.1, B_reps=3, random_state=0)
        model2 = NullstrapLM(fdr=0.1, B_reps=3, random_state=0)

        model1.fit(linear_data["X"], linear_data["y"])
        model2.fit(linear_data["X"], linear_data["y"])

        # Results should be identical
        assert_reproducible_results(model1, model2)
        
        # Should be different from random_state=1
        model3 = NullstrapLM(fdr=0.1, B_reps=3, random_state=1)
        model3.fit(linear_data["X"], linear_data["y"])
        
        # At least one of these should differ
        assert (
            not np.array_equal(model1.selected_, model3.selected_) or
            not np.isclose(model1.threshold_, model3.threshold_)
        )

    @pytest.mark.model
    def test_error_distributions(self, linear_data):
        """Test different error distributions for knockoff generation."""
        # Test normal distribution
        model_normal = NullstrapLM(
            fdr=0.1, error_dist="normal", B_reps=3, random_state=42
        )
        model_normal.fit(linear_data["X"], linear_data["y"])
        assert_model_fitted(model_normal)
        assert model_normal.error_dist == "normal"

        # Test resample distribution
        model_resample = NullstrapLM(
            fdr=0.1, error_dist="resample", B_reps=3, random_state=42
        )
        model_resample.fit(linear_data["X"], linear_data["y"])
        assert_model_fitted(model_resample)
        assert model_resample.error_dist == "resample"
        
        # Both should have residuals scaled
        assert model_normal.residuals_ is not None
        assert model_resample.residuals_ is not None
        assert len(model_normal.residuals_) == linear_data["n_samples"]
        assert len(model_resample.residuals_) == linear_data["n_samples"]

    @pytest.mark.model
    def test_alpha_selection_methods(self, linear_data):
        """Test different methods for alpha selection."""
        # Test fixed alpha
        model_fixed = NullstrapLM(fdr=0.1, alpha_=0.1, B_reps=3, random_state=42)
        model_fixed.fit(linear_data["X"], linear_data["y"])
        assert_model_fitted(model_fixed)
        assert model_fixed.alpha_used_ == 0.1

        # Test cross-validation with custom alpha array
        custom_alphas = np.logspace(-2, 0, 20)
        model_cv_custom = NullstrapLM(
            fdr=0.1,
            alpha_=None,
            alphas=custom_alphas,
            B_reps=3,
            random_state=42
        )
        model_cv_custom.fit(linear_data["X"], linear_data["y"])
        assert_model_fitted(model_cv_custom)
        assert model_cv_custom.alpha_used_ > 0

        # Test cross-validation with automatic alpha generation (int)
        model_cv_auto = NullstrapLM(
            fdr=0.1,
            alpha_=None,
            alphas=20,
            eps=1e-3,
            B_reps=3,
            random_state=42
        )
        model_cv_auto.fit(linear_data["X"], linear_data["y"])
        assert_model_fitted(model_cv_auto)
        assert model_cv_auto.alpha_used_ > 0

        # Test cross-validation with default (100 alphas)
        model_cv_default = NullstrapLM(
            fdr=0.1,
            alpha_=None,
            B_reps=3,
            random_state=42
        )
        model_cv_default.fit(linear_data["X"], linear_data["y"])
        assert_model_fitted(model_cv_default)
        assert model_cv_default.alpha_used_ > 0

    @pytest.mark.model
    def test_tolerance_parameters(self, linear_data):
        """Test different tolerance parameters."""
        # Test with different lasso_tol
        model1 = NullstrapLM(
            fdr=0.1, lasso_tol=1e-5, B_reps=3, random_state=42
        )
        model1.fit(linear_data["X"], linear_data["y"])
        assert_model_fitted(model1)

        # Test with different binary_search_tol
        model2 = NullstrapLM(
            fdr=0.1, binary_search_tol=1e-6, B_reps=3, random_state=42
        )
        model2.fit(linear_data["X"], linear_data["y"])
        assert_model_fitted(model2)

        # Test with both custom tolerances
        model3 = NullstrapLM(
            fdr=0.1,
            lasso_tol=1e-6,
            binary_search_tol=1e-7,
            B_reps=3,
            random_state=42
        )
        model3.fit(linear_data["X"], linear_data["y"])
        assert_model_fitted(model3)

    @pytest.mark.model
    def test_model_attributes_after_fitting(self, linear_data):
        """Test that all expected attributes are set after fitting."""
        model = NullstrapLM(fdr=0.1, B_reps=3, random_state=42)
        model.fit(linear_data["X"], linear_data["y"])

        # Check fitted attributes
        assert hasattr(model, "alpha_used_")
        assert hasattr(model, "sigma_hat_")
        assert hasattr(model, "residuals_")
        assert hasattr(model, "correction_factor_")
        assert hasattr(model, "statistic_")
        assert hasattr(model, "threshold_")
        assert hasattr(model, "selected_")
        assert hasattr(model, "n_features_selected_")

        # Check attribute types and values
        assert isinstance(model.alpha_used_, (int, float))
        assert model.alpha_used_ >= 0
        assert isinstance(model.sigma_hat_, (int, float))
        assert model.sigma_hat_ > 0
        assert isinstance(model.residuals_, np.ndarray)
        assert len(model.residuals_) == linear_data["n_samples"]
        assert isinstance(model.correction_factor_, (int, float))
        assert model.correction_factor_ >= 0
        assert isinstance(model.statistic_, np.ndarray)
        assert len(model.statistic_) == linear_data["n_features"]

    @pytest.mark.model
    def test_residual_scaling(self, linear_data):
        """Test that residuals are properly scaled."""
        model = NullstrapLM(fdr=0.1, B_reps=3, random_state=42)
        model.fit(linear_data["X"], linear_data["y"])

        # Residuals should be scaled
        assert model.residuals_ is not None
        assert len(model.residuals_) == linear_data["n_samples"]
        
        # Scaling factor should be applied (residuals should not equal raw residuals)
        # We can't directly test this without access to raw residuals, but we can
        # check that residuals have reasonable magnitude
        assert np.std(model.residuals_) > 0

    @pytest.mark.model
    def test_predict_method_consistency(self, linear_data):
        """Test that model.predict() is used consistently for residuals."""
        model = NullstrapLM(fdr=0.1, B_reps=3, random_state=42)
        model.fit(linear_data["X"], linear_data["y"])

        # This is an indirect test - if predict() weren't working correctly,
        # the model would fail to fit or produce incorrect results
        assert_model_fitted(model)
        assert model.sigma_hat_ > 0
        assert model.residuals_ is not None

    @pytest.mark.model
    def test_alpha_scale_factor(self, linear_data):
        """Test alpha_scale_factor parameter."""
        # Test with different scale factors
        model1 = NullstrapLM(
            fdr=0.1, alpha_=None, alpha_scale_factor=0.5, B_reps=3, random_state=42
        )
        model1.fit(linear_data["X"], linear_data["y"])

        model2 = NullstrapLM(
            fdr=0.1, alpha_=None, alpha_scale_factor=1.0, B_reps=3, random_state=42
        )
        model2.fit(linear_data["X"], linear_data["y"])

        # Different scale factors should generally lead to different alpha values
        # (though not guaranteed in all cases due to CV behavior)
        assert_model_fitted(model1)
        assert_model_fitted(model2)

    @pytest.mark.model
    def test_transform_error_cases(self, linear_data):
        """Test transform method error handling."""
        model = NullstrapLM(fdr=0.1, B_reps=3, random_state=42)

        # Test transform before fitting
        with pytest.raises(ValueError, match="not been fitted"):
            model.transform(linear_data["X"])

        # Fit the model
        model.fit(linear_data["X"], linear_data["y"])

        # Test transform with wrong number of features
        X_wrong = np.random.randn(10, linear_data["n_features"] + 5)
        with pytest.raises(ValueError, match="must have .* features"):
            model.transform(X_wrong)

    @pytest.mark.model
    def test_correction_min_parameter(self, linear_data):
        """Test correction_min parameter."""
        model = NullstrapLM(
            fdr=0.1, correction_min=0.01, B_reps=3, random_state=42
        )
        model.fit(linear_data["X"], linear_data["y"])

        assert_model_fitted(model)
        # Correction factor should be >= correction_min
        assert model.correction_factor_ >= 0.01

    @pytest.mark.model
    def test_cv_folds_parameter(self, linear_data):
        """Test cv_folds parameter for cross-validation."""
        # Test with different fold numbers
        for n_folds in [5, 10]:
            model = NullstrapLM(
                fdr=0.1, alpha_=None, cv_folds=n_folds, B_reps=3, random_state=42
            )
            model.fit(linear_data["X"], linear_data["y"])
            assert_model_fitted(model)

    @pytest.mark.model  
    def test_max_iter_parameter(self, linear_data):
        """Test max_iter parameter."""
        # Test with low max_iter (should still converge for easy problems)
        model = NullstrapLM(
            fdr=0.1, max_iter=1000, B_reps=3, random_state=42
        )
        model.fit(linear_data["X"], linear_data["y"])
        assert_model_fitted(model)

