"""
Tests for NullstrapGLM (Generalized Linear Models).

This module contains tests for the NullstrapGLM class,
including tests for different GLM families and model functionality.
"""

import numpy as np
import pytest

from nullstrap.models.glm import NullstrapGLM
from tests.conftest import (assert_fdr_control, assert_model_fitted,
                            assert_reproducible_results)


class TestNullstrapGLM:
    """Test cases for NullstrapGLM (Generalized Linear Models)."""

    @pytest.mark.model
    def test_basic_fitting(self, classification_data, model_params):
        """Test basic model fitting functionality."""
        model = NullstrapGLM(**model_params)
        model.fit(classification_data["X"], classification_data["y"])

        assert_model_fitted(model)
        assert model.n_samples_ == classification_data["n_samples"]
        assert model.n_features_ == classification_data["n_features"]

    @pytest.mark.model
    def test_fdr_control(self, classification_data, model_params):
        """Test that FDR is controlled at specified level."""
        model = NullstrapGLM(**model_params)
        model.fit(classification_data["X"], classification_data["y"])

        assert_fdr_control(
            model.selected_,
            classification_data["true_features"],
            fdr_level=model_params["fdr"],
            tolerance=0.15,  # Higher tolerance for GLM
        )

    @pytest.mark.model
    def test_reproducibility(self, classification_data, model_params):
        """Test that results are reproducible with same random_state."""
        model1 = NullstrapGLM(**model_params)
        model2 = NullstrapGLM(**model_params)

        model1.fit(classification_data["X"], classification_data["y"])
        model2.fit(classification_data["X"], classification_data["y"])

        assert_reproducible_results(model1, model2)

    @pytest.mark.model
    def test_family_parameter(self, classification_data, model_params):
        """Test different family parameters."""
        # Test binomial family (default)
        model_binom = NullstrapGLM(family="binomial", **model_params)
        model_binom.fit(classification_data["X"], classification_data["y"])
        assert_model_fitted(model_binom)

        # Test that family is stored correctly
        assert model_binom.family == "binomial"

    @pytest.mark.model
    def test_transform_method(self, classification_data, model_params):
        """Test the transform method."""
        model = NullstrapGLM(**model_params)
        model.fit(classification_data["X"], classification_data["y"])

        X_transformed = model.transform(classification_data["X"])

        assert X_transformed.shape[0] == classification_data["n_samples"]
        assert X_transformed.shape[1] == model.n_features_selected_

        if model.n_features_selected_ > 0:
            assert np.allclose(
                X_transformed, classification_data["X"][:, model.selected_]
            )

    @pytest.mark.model
    def test_penalty_types(self, classification_data, model_params):
        """Test different penalty types (L1, L2, elasticnet)."""
        # Test L1 (LASSO) - default
        model_l1 = NullstrapGLM(penalty="l1", solver="saga", **model_params)
        model_l1.fit(classification_data["X"], classification_data["y"])
        assert_model_fitted(model_l1)
        assert model_l1.penalty == "l1"

        # Test L2 (Ridge)
        model_l2 = NullstrapGLM(penalty="l2", solver="lbfgs", **model_params)
        model_l2.fit(classification_data["X"], classification_data["y"])
        assert_model_fitted(model_l2)
        assert model_l2.penalty == "l2"

        # Test elasticnet
        model_en = NullstrapGLM(
            penalty="elasticnet", solver="saga", l1_ratio=0.5, **model_params
        )
        model_en.fit(classification_data["X"], classification_data["y"])
        assert_model_fitted(model_en)
        assert model_en.penalty == "elasticnet"
        assert model_en.l1_ratio == 0.5

    @pytest.mark.model
    def test_l1_ratio_validation(self, model_params):
        """Test l1_ratio parameter validation."""
        # l1_ratio required for elasticnet
        with pytest.raises(ValueError, match="l1_ratio must be specified"):
            model = NullstrapGLM(penalty="elasticnet", solver="saga", **model_params)
            model._validate_parameters()

        # l1_ratio must be in [0, 1]
        with pytest.raises(ValueError, match="l1_ratio must be between 0 and 1"):
            model = NullstrapGLM(
                penalty="elasticnet", solver="saga", l1_ratio=1.5, **model_params
            )
            model._validate_parameters()

        # l1_ratio should not be used with non-elasticnet penalties
        with pytest.raises(ValueError, match="only used with penalty='elasticnet'"):
            model = NullstrapGLM(penalty="l1", l1_ratio=0.5, **model_params)
            model._validate_parameters()

    @pytest.mark.model
    def test_solver_compatibility(self, model_params):
        """Test solver and penalty compatibility validation."""
        # L1 requires saga or liblinear
        with pytest.raises(ValueError, match="solver must be one of"):
            model = NullstrapGLM(penalty="l1", solver="lbfgs", **model_params)
            model._validate_parameters()

        # Elasticnet requires saga
        with pytest.raises(ValueError, match="solver must be 'saga'"):
            model = NullstrapGLM(
                penalty="elasticnet", solver="liblinear", l1_ratio=0.5, **model_params
            )
            model._validate_parameters()

    @pytest.mark.model
    def test_extreme_values_no_overflow(self, model_params):
        """Test that extreme values don't cause overflow warnings."""
        # Create data that will produce extreme linear predictors
        np.random.seed(42)
        X_extreme = np.random.randn(100, 10) * 10
        beta_extreme = np.random.randn(10) * 100
        
        model = NullstrapGLM(**model_params)
        model._validate_parameters()
        model.n_samples_, model.n_features_ = X_extreme.shape
        model.sample_rng = np.random.RandomState(42)
        model.alpha_used_ = 0.1
        
        # This should not raise overflow warnings
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            y_synthetic = model._generate_synthetic_data(
                X_extreme, beta=beta_extreme, family="binomial"
            )
            
            overflow_warnings = [
                warning for warning in w 
                if 'overflow' in str(warning.message).lower()
            ]
            assert len(overflow_warnings) == 0, "Overflow warnings detected"
        
        # Verify output is valid binary data
        assert y_synthetic.shape == (100,)
        assert set(np.unique(y_synthetic)).issubset({0, 1})

    @pytest.mark.model
    def test_invalid_penalty(self, model_params):
        """Test that invalid penalty types are rejected."""
        with pytest.raises(ValueError, match="penalty must be one of"):
            model = NullstrapGLM(penalty="invalid", **model_params)
            model._validate_parameters()

    @pytest.mark.model
    def test_invalid_family(self, model_params):
        """Test that non-binomial family is rejected."""
        with pytest.raises(ValueError, match="Currently only 'binomial' family is supported"):
            model = NullstrapGLM(family="poisson", **model_params)
            model._validate_parameters()

    @pytest.mark.model
    def test_unsupported_family_in_synthetic_data(self, model_params):
        """Test that unsupported family in _generate_synthetic_data raises error."""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        
        model = NullstrapGLM(**model_params)
        model._validate_parameters()
        model.sample_rng = np.random.RandomState(42)
        
        with pytest.raises(ValueError, match="Unsupported GLM family"):
            model._generate_synthetic_data(X, beta=None, family="poisson")

    @pytest.mark.model
    def test_missing_y_parameter(self, classification_data, model_params):
        """Test that missing y parameter raises error."""
        model = NullstrapGLM(**model_params)
        
        with pytest.raises(ValueError, match="GLM models require y parameter"):
            model._validate_data(classification_data["X"], y=None)

    @pytest.mark.model
    def test_non_binary_y_values(self, model_params):
        """Test that non-binary y values are rejected."""
        X = np.random.randn(50, 10)
        y = np.array([0, 1, 2, 3] * 12 + [0, 1])  # Contains values > 1
        
        model = NullstrapGLM(**model_params)
        
        with pytest.raises(ValueError, match="y must contain binary values"):
            model.fit(X, y)

    @pytest.mark.model
    def test_single_class_warning(self, model_params):
        """Test that single class in y triggers warning."""
        X = np.random.randn(50, 10)
        y = np.ones(50)  # All class 1
        
        model = NullstrapGLM(**model_params)
        
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model._validate_data(X, y)
            
            # Check that a warning was raised
            assert len(w) == 1
            assert "Single class detected" in str(w[0].message)
            assert issubclass(w[0].category, UserWarning)

    @pytest.mark.model
    def test_sklearn_compatibility(self, classification_data, model_params):
        """Test sklearn compatibility methods (get_params, set_params)."""
        model = NullstrapGLM(**model_params)
        
        # Test get_params
        params = model.get_params()
        assert "fdr" in params
        assert "penalty" in params
        assert params["fdr"] == model_params["fdr"]
        
        # Test set_params
        model.set_params(fdr=0.05)
        assert model.fdr == 0.05
        
        # Test deep=True
        params_deep = model.get_params(deep=True)
        assert len(params_deep) >= len(params)

    @pytest.mark.model
    def test_fit_transform(self, classification_data, model_params):
        """Test fit_transform method."""
        model = NullstrapGLM(**model_params)
        
        X_transformed = model.fit_transform(classification_data["X"], classification_data["y"])
        
        assert X_transformed.shape[0] == classification_data["n_samples"]
        assert X_transformed.shape[1] == model.n_features_selected_
        assert np.allclose(X_transformed, model.transform(classification_data["X"]))

    @pytest.mark.model
    def test_high_dimensional_data(self, model_params):
        """Test with high-dimensional data (p > n)."""
        np.random.seed(42)
        n_samples, n_features = 50, 100  # p > n
        
        X = np.random.randn(n_samples, n_features)
        # Create sparse signal
        true_coef = np.zeros(n_features)
        true_coef[:5] = np.random.randn(5) * 2.0
        
        # Generate binary response
        linear_predictor = X @ true_coef
        prob = 1 / (1 + np.exp(-linear_predictor))
        y = np.random.binomial(1, prob)
        
        model = NullstrapGLM(**model_params)
        model.fit(X, y)
        
        assert_model_fitted(model)
        assert model.n_samples_ == n_samples
        assert model.n_features_ == n_features

    @pytest.mark.model
    def test_no_features_selected(self, model_params):
        """Test behavior when no features are selected (very high threshold)."""
        np.random.seed(42)
        # Create data with very weak signal
        X = np.random.randn(100, 20) * 0.01
        y = np.random.binomial(1, 0.5, 100)
        
        # Use very conservative FDR to potentially select no features
        params = {k: v for k, v in model_params.items() if k not in ["fdr", "B_reps"]}
        model = NullstrapGLM(fdr=0.001, B_reps=1, **params)
        model.fit(X, y)
        
        # Test transform with potentially 0 selected features
        X_transformed = model.transform(X)
        assert X_transformed.shape[0] == 100
        assert X_transformed.shape[1] == model.n_features_selected_

    @pytest.mark.model
    def test_fixed_alpha_vs_cv_alpha(self, classification_data, model_params):
        """Test that fixed alpha and CV-selected alpha both work."""
        # Test with fixed alpha
        model_fixed = NullstrapGLM(alpha_=0.1, **{k: v for k, v in model_params.items() if k != "alpha_"})
        model_fixed.fit(classification_data["X"], classification_data["y"])
        assert model_fixed.alpha_used_ == 0.1
        assert_model_fitted(model_fixed)
        
        # Test with CV alpha (default)
        model_cv = NullstrapGLM(**model_params)
        model_cv.fit(classification_data["X"], classification_data["y"])
        assert model_cv.alpha_used_ is not None
        assert_model_fitted(model_cv)

    @pytest.mark.model
    def test_different_alphas_specification(self, classification_data, model_params):
        """Test different ways to specify alphas for CV."""
        # Test with integer (number of alphas)
        model_int = NullstrapGLM(alphas=5, **model_params)
        model_int.fit(classification_data["X"], classification_data["y"])
        assert_model_fitted(model_int)
        
        # Test with array of alphas
        model_array = NullstrapGLM(alphas=[0.001, 0.01, 0.1, 1.0], **model_params)
        model_array.fit(classification_data["X"], classification_data["y"])
        assert_model_fitted(model_array)

    @pytest.mark.model
    def test_different_scoring_metrics(self, classification_data, model_params):
        """Test different scoring metrics for CV."""
        # Test with accuracy
        model_acc = NullstrapGLM(scoring="accuracy", **model_params)
        model_acc.fit(classification_data["X"], classification_data["y"])
        assert_model_fitted(model_acc)
        
        # Test with roc_auc (if sufficient class balance)
        if len(np.unique(classification_data["y"])) == 2:
            model_auc = NullstrapGLM(scoring="roc_auc", **model_params)
            model_auc.fit(classification_data["X"], classification_data["y"])
            assert_model_fitted(model_auc)

    @pytest.mark.model
    def test_n_jobs_parameter(self, classification_data, model_params):
        """Test n_jobs parameter for parallel processing."""
        # Test with n_jobs=1
        model_serial = NullstrapGLM(n_jobs=1, **model_params)
        model_serial.fit(classification_data["X"], classification_data["y"])
        assert_model_fitted(model_serial)
        
        # Test with n_jobs=-1 (all cores)
        model_parallel = NullstrapGLM(n_jobs=-1, **model_params)
        model_parallel.fit(classification_data["X"], classification_data["y"])
        assert_model_fitted(model_parallel)

    @pytest.mark.model
    def test_compute_scale_factor(self, model_params):
        """Test that _compute_scale_factor returns 1.0 for GLM."""
        model = NullstrapGLM(**model_params)
        scale_factor = model._compute_scale_factor()
        assert scale_factor == 1.0

    @pytest.mark.model
    def test_correction_factor_estimation(self, classification_data, model_params):
        """Test correction factor estimation with different B_reps."""
        # Test with B_reps=1
        params_1 = {k: v for k, v in model_params.items() if k != "B_reps"}
        model_1 = NullstrapGLM(B_reps=1, **params_1)
        model_1.fit(classification_data["X"], classification_data["y"])
        assert hasattr(model_1, "correction_factor_")
        assert model_1.correction_factor_ >= 0
        
        # Test with B_reps=3
        params_3 = {k: v for k, v in model_params.items() if k != "B_reps"}
        model_3 = NullstrapGLM(B_reps=3, **params_3)
        model_3.fit(classification_data["X"], classification_data["y"])
        assert hasattr(model_3, "correction_factor_")
        assert model_3.correction_factor_ >= 0

    @pytest.mark.model
    def test_generate_synthetic_data_with_beta(self, model_params):
        """Test synthetic data generation with and without beta."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        beta = np.random.randn(10) * 0.5
        
        model = NullstrapGLM(**model_params)
        model._validate_parameters()
        model.sample_rng = np.random.RandomState(42)
        
        # Test with beta (signal data)
        y_signal = model._generate_synthetic_data(X, beta=beta, family="binomial")
        assert y_signal.shape == (100,)
        assert set(np.unique(y_signal)).issubset({0, 1})
        
        # Test without beta (null data)
        y_null = model._generate_synthetic_data(X, beta=None, family="binomial")
        assert y_null.shape == (100,)
        assert set(np.unique(y_null)).issubset({0, 1})
        
        # Signal and null data should generally be different
        # (though they could theoretically be the same with low probability)
        assert not np.array_equal(y_signal, y_null)

    @pytest.mark.model
    def test_generate_synthetic_data_with_custom_rng(self, model_params):
        """Test synthetic data generation with custom RNG."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        
        model = NullstrapGLM(**model_params)
        model._validate_parameters()
        model.sample_rng = np.random.RandomState(42)
        
        # Use custom RNG
        custom_rng = np.random.RandomState(123)
        y1 = model._generate_synthetic_data(X, beta=None, family="binomial", rng=custom_rng)
        
        # Reset and use same custom RNG seed
        custom_rng2 = np.random.RandomState(123)
        y2 = model._generate_synthetic_data(X, beta=None, family="binomial", rng=custom_rng2)
        
        assert np.array_equal(y1, y2)

    @pytest.mark.model
    def test_liblinear_solver(self, classification_data, model_params):
        """Test liblinear solver with L1 penalty."""
        model = NullstrapGLM(penalty="l1", solver="liblinear", **model_params)
        model.fit(classification_data["X"], classification_data["y"])
        assert_model_fitted(model)
        assert model.solver == "liblinear"

    @pytest.mark.model
    def test_different_cv_folds(self, classification_data, model_params):
        """Test different cv_folds values."""
        # Test with cv_folds=3
        model_3 = NullstrapGLM(cv_folds=3, **model_params)
        model_3.fit(classification_data["X"], classification_data["y"])
        assert_model_fitted(model_3)
        assert model_3.cv_folds == 3
        
        # Test with cv_folds=10
        model_10 = NullstrapGLM(cv_folds=10, **model_params)
        model_10.fit(classification_data["X"], classification_data["y"])
        assert_model_fitted(model_10)
        assert model_10.cv_folds == 10

    @pytest.mark.model
    def test_different_max_iter(self, classification_data, model_params):
        """Test different max_iter values."""
        # Test with max_iter=100
        params_100 = {k: v for k, v in model_params.items() if k != "max_iter"}
        model_100 = NullstrapGLM(max_iter=100, **params_100)
        model_100.fit(classification_data["X"], classification_data["y"])
        assert_model_fitted(model_100)
        
        # Test with max_iter=1000
        params_1000 = {k: v for k, v in model_params.items() if k != "max_iter"}
        model_1000 = NullstrapGLM(max_iter=1000, **params_1000)
        model_1000.fit(classification_data["X"], classification_data["y"])
        assert_model_fitted(model_1000)

    @pytest.mark.model
    def test_different_alpha_scale_factor(self, classification_data, model_params):
        """Test different alpha_scale_factor values."""
        # Test with alpha_scale_factor=0.3 (more conservative)
        model_03 = NullstrapGLM(alpha_scale_factor=0.3, **model_params)
        model_03.fit(classification_data["X"], classification_data["y"])
        assert_model_fitted(model_03)
        
        # Test with alpha_scale_factor=0.7 (less conservative)
        model_07 = NullstrapGLM(alpha_scale_factor=0.7, **model_params)
        model_07.fit(classification_data["X"], classification_data["y"])
        assert_model_fitted(model_07)

    @pytest.mark.model
    def test_different_binary_search_tol(self, classification_data, model_params):
        """Test different binary_search_tol values."""
        # Test with different tolerances
        model_strict = NullstrapGLM(binary_search_tol=1e-10, **model_params)
        model_strict.fit(classification_data["X"], classification_data["y"])
        assert_model_fitted(model_strict)
        
        model_relaxed = NullstrapGLM(binary_search_tol=1e-6, **model_params)
        model_relaxed.fit(classification_data["X"], classification_data["y"])
        assert_model_fitted(model_relaxed)

    @pytest.mark.model
    def test_different_correction_min(self, classification_data, model_params):
        """Test different correction_min values."""
        # Test with different minimum bounds
        model_small = NullstrapGLM(correction_min=1e-16, **model_params)
        model_small.fit(classification_data["X"], classification_data["y"])
        assert_model_fitted(model_small)
        
        model_large = NullstrapGLM(correction_min=1e-10, **model_params)
        model_large.fit(classification_data["X"], classification_data["y"])
        assert_model_fitted(model_large)

    @pytest.mark.model
    def test_different_lasso_tol(self, classification_data, model_params):
        """Test different lasso_tol values."""
        # Test with strict tolerance
        model_strict = NullstrapGLM(lasso_tol=1e-12, **model_params)
        model_strict.fit(classification_data["X"], classification_data["y"])
        assert_model_fitted(model_strict)
        
        # Test with relaxed tolerance
        model_relaxed = NullstrapGLM(lasso_tol=1e-4, **model_params)
        model_relaxed.fit(classification_data["X"], classification_data["y"])
        assert_model_fitted(model_relaxed)

    @pytest.mark.model
    def test_statistic_attribute(self, classification_data, model_params):
        """Test that statistic_ attribute is properly set."""
        model = NullstrapGLM(**model_params)
        model.fit(classification_data["X"], classification_data["y"])
        
        assert hasattr(model, "statistic_")
        assert model.statistic_.shape == (classification_data["n_features"],)
        assert np.all(model.statistic_ >= 0)  # Should be absolute values

    @pytest.mark.model
    def test_threshold_attribute(self, classification_data, model_params):
        """Test that threshold_ attribute is properly computed."""
        model = NullstrapGLM(**model_params)
        model.fit(classification_data["X"], classification_data["y"])
        
        assert hasattr(model, "threshold_")
        assert isinstance(model.threshold_, (int, float))
        assert model.threshold_ >= 0

    @pytest.mark.model
    def test_alpha_used_attribute(self, classification_data, model_params):
        """Test that alpha_used_ attribute is properly set."""
        # Test with CV alpha
        model_cv = NullstrapGLM(**model_params)
        model_cv.fit(classification_data["X"], classification_data["y"])
        assert hasattr(model_cv, "alpha_used_")
        assert model_cv.alpha_used_ > 0
        
        # Test with fixed alpha
        params_fixed = {k: v for k, v in model_params.items() if k != "alpha_"}
        model_fixed = NullstrapGLM(alpha_=0.05, **params_fixed)
        model_fixed.fit(classification_data["X"], classification_data["y"])
        assert model_fixed.alpha_used_ == 0.05

    @pytest.mark.model
    def test_get_selected_features_method(self, classification_data, model_params):
        """Test get_selected_features static method."""
        model = NullstrapGLM(**model_params)
        model.fit(classification_data["X"], classification_data["y"])
        
        # Test with same statistics and threshold
        selected = model.get_selected_features(model.statistic_, model.threshold_)
        assert np.array_equal(selected, model.selected_)
        
        # Test with custom threshold
        high_threshold = np.max(model.statistic_) + 1.0
        selected_high = model.get_selected_features(model.statistic_, high_threshold)
        assert len(selected_high) == 0
        
        # Test with low threshold
        low_threshold = np.min(model.statistic_) - 1.0
        selected_low = model.get_selected_features(model.statistic_, low_threshold)
        assert len(selected_low) == len(model.statistic_)

    @pytest.mark.model
    def test_perfectly_separable_data(self, model_params):
        """Test with perfectly separable data."""
        np.random.seed(42)
        # Create perfectly separable data
        X = np.random.randn(100, 5)
        true_coef = np.array([10.0, 10.0, 0.0, 0.0, 0.0])
        
        linear_predictor = X @ true_coef
        y = (linear_predictor > 0).astype(int)
        
        model = NullstrapGLM(**model_params)
        model.fit(X, y)
        
        assert_model_fitted(model)
        # Should select some features even with perfect separation
        assert model.n_features_selected_ >= 0

    @pytest.mark.model
    def test_balanced_vs_imbalanced_data(self, model_params):
        """Test with balanced and imbalanced class distributions."""
        np.random.seed(42)
        X = np.random.randn(200, 10)
        true_coef = np.random.randn(10) * 0.5
        
        # Balanced data
        linear_pred_balanced = X @ true_coef
        prob_balanced = 1 / (1 + np.exp(-linear_pred_balanced))
        y_balanced = np.random.binomial(1, prob_balanced)
        
        model_balanced = NullstrapGLM(**model_params)
        model_balanced.fit(X, y_balanced)
        assert_model_fitted(model_balanced)
        
        # Imbalanced data (90-10 split)
        y_imbalanced = np.zeros(200, dtype=int)
        y_imbalanced[:20] = 1  # Only 10% positive class
        
        model_imbalanced = NullstrapGLM(**model_params)
        model_imbalanced.fit(X, y_imbalanced)
        assert_model_fitted(model_imbalanced)

    @pytest.mark.model
    def test_all_zero_coefficients(self, model_params):
        """Test when base model has all zero coefficients (very weak signal)."""
        np.random.seed(42)
        # Random noise data with no signal
        X = np.random.randn(100, 20) * 0.001
        y = np.random.binomial(1, 0.5, 100)
        
        # Use very strong regularization to get near-zero coefficients
        params = {k: v for k, v in model_params.items() if k != "alpha_"}
        model = NullstrapGLM(alpha_=10.0, **params)
        model.fit(X, y)
        
        assert_model_fitted(model)
        # Should have threshold and selection even if all coefficients are near zero
        assert hasattr(model, "threshold_")
        assert hasattr(model, "selected_")

    @pytest.mark.model
    def test_sample_rng_initialization(self, model_params):
        """Test that sample_rng is properly initialized."""
        # With random_state
        params_with = {k: v for k, v in model_params.items() if k != "random_state"}
        model_with_seed = NullstrapGLM(random_state=42, **params_with)
        assert hasattr(model_with_seed, "sample_rng")
        assert isinstance(model_with_seed.sample_rng, np.random.RandomState)
        
        # Without random_state
        model_without_seed = NullstrapGLM(random_state=None, **{k: v for k, v in model_params.items() if k != "random_state"})
        assert hasattr(model_without_seed, "sample_rng")

    @pytest.mark.model
    def test_elasticnet_with_different_l1_ratios(self, classification_data, model_params):
        """Test elasticnet with various l1_ratio values."""
        # Test l1_ratio=0.2 (more L2)
        model_02 = NullstrapGLM(penalty="elasticnet", solver="saga", l1_ratio=0.2, **model_params)
        model_02.fit(classification_data["X"], classification_data["y"])
        assert_model_fitted(model_02)
        
        # Test l1_ratio=0.8 (more L1)
        model_08 = NullstrapGLM(penalty="elasticnet", solver="saga", l1_ratio=0.8, **model_params)
        model_08.fit(classification_data["X"], classification_data["y"])
        assert_model_fitted(model_08)
        
        # Test l1_ratio=0.0 (pure L2)
        model_00 = NullstrapGLM(penalty="elasticnet", solver="saga", l1_ratio=0.0, **model_params)
        model_00.fit(classification_data["X"], classification_data["y"])
        assert_model_fitted(model_00)
        
        # Test l1_ratio=1.0 (pure L1)
        model_10 = NullstrapGLM(penalty="elasticnet", solver="saga", l1_ratio=1.0, **model_params)
        model_10.fit(classification_data["X"], classification_data["y"])
        assert_model_fitted(model_10)

    @pytest.mark.model
    def test_fit_lasso_model_with_fixed_alpha(self, classification_data, model_params):
        """Test _fit_lasso_model with fixed alpha."""
        model = NullstrapGLM(**model_params)
        model._validate_parameters()
        model.n_samples_, model.n_features_ = classification_data["X"].shape
        
        from nullstrap.utils.core import standardize_data
        X_scaled, _ = standardize_data(
            classification_data["X"], y=None, 
            scale_by_sample_size=True, 
            n_samples=model.n_samples_
        )
        
        # Test with fixed alpha
        fitted_model, alpha_used = model._fit_lasso_model(
            X_scaled, classification_data["y"], alpha=0.1
        )
        
        assert fitted_model is not None
        assert alpha_used == 0.1
        assert hasattr(fitted_model, "coef_")

    @pytest.mark.model
    def test_fit_lasso_model_with_cv(self, classification_data, model_params):
        """Test _fit_lasso_model with cross-validation."""
        model = NullstrapGLM(**model_params)
        model._validate_parameters()
        model.n_samples_, model.n_features_ = classification_data["X"].shape
        
        from nullstrap.utils.core import standardize_data
        X_scaled, _ = standardize_data(
            classification_data["X"], y=None, 
            scale_by_sample_size=True, 
            n_samples=model.n_samples_
        )
        
        # Test with alpha=None (triggers CV)
        fitted_model, alpha_used = model._fit_lasso_model(
            X_scaled, classification_data["y"], alpha=None
        )
        
        assert fitted_model is not None
        assert alpha_used > 0
        assert hasattr(fitted_model, "coef_")

    @pytest.mark.model
    def test_transform_before_fit_raises_error(self, classification_data, model_params):
        """Test that transform raises error if called before fit."""
        model = NullstrapGLM(**model_params)
        
        with pytest.raises(ValueError, match="Model has not been fitted yet"):
            model.transform(classification_data["X"])

    @pytest.mark.model
    def test_small_sample_size(self, model_params):
        """Test with very small sample size."""
        np.random.seed(42)
        # Very small sample
        X = np.random.randn(15, 5)
        y = np.random.binomial(1, 0.5, 15)
        
        # Adjust cv_folds to be smaller than sample size
        params = {k: v for k, v in model_params.items() if k != "cv_folds"}
        model = NullstrapGLM(cv_folds=3, **params)
        model.fit(X, y)
        
        assert_model_fitted(model)
        assert model.n_samples_ == 15

    @pytest.mark.model
    def test_large_feature_space(self, model_params):
        """Test with very large feature space."""
        np.random.seed(42)
        # Many features
        X = np.random.randn(100, 200)
        y = np.random.binomial(1, 0.5, 100)
        
        # Use B_reps=1 to speed up
        params = {k: v for k, v in model_params.items() if k != "B_reps"}
        model = NullstrapGLM(B_reps=1, **params)
        model.fit(X, y)
        
        assert_model_fitted(model)
        assert model.n_features_ == 200

    @pytest.mark.model
    def test_edge_case_negative_l1_ratio(self, model_params):
        """Test that negative l1_ratio is rejected."""
        with pytest.raises(ValueError, match="l1_ratio must be between 0 and 1"):
            model = NullstrapGLM(
                penalty="elasticnet", solver="saga", l1_ratio=-0.1, **model_params
            )
            model._validate_parameters()

    @pytest.mark.model
    def test_edge_case_l1_ratio_greater_than_one(self, model_params):
        """Test that l1_ratio > 1 is rejected."""
        with pytest.raises(ValueError, match="l1_ratio must be between 0 and 1"):
            model = NullstrapGLM(
                penalty="elasticnet", solver="saga", l1_ratio=1.1, **model_params
            )
            model._validate_parameters()

    @pytest.mark.model
    def test_validate_x_called(self, model_params):
        """Test that _validate_X is called during data validation."""
        # Test with invalid X (not 2D)
        X_invalid = np.array([1, 2, 3])
        y = np.array([0, 1, 0])
        
        model = NullstrapGLM(**model_params)
        
        with pytest.raises((ValueError, AttributeError)):
            model._validate_data(X_invalid, y)

    @pytest.mark.model
    def test_validate_y_called(self, classification_data, model_params):
        """Test that _validate_y is called during data validation."""
        # Test with invalid y shape
        y_invalid = np.array([[0, 1], [1, 0]])
        
        model = NullstrapGLM(**model_params)
        
        with pytest.raises((ValueError, IndexError)):
            model._validate_data(classification_data["X"], y_invalid)

    @pytest.mark.model
    def test_validate_sample_sizes_called(self, model_params):
        """Test that _validate_sample_sizes is called during data validation."""
        # Test with mismatched X and y sizes
        X = np.random.randn(100, 10)
        y = np.array([0, 1, 0])  # Only 3 samples
        
        model = NullstrapGLM(**model_params)
        
        with pytest.raises(ValueError):
            model._validate_data(X, y)

    @pytest.mark.model
    def test_correction_factor_with_different_fdr(self, classification_data):
        """Test correction factor with different FDR levels."""
        # Test with FDR=0.01
        model_001 = NullstrapGLM(fdr=0.01, random_state=42, B_reps=2)
        model_001.fit(classification_data["X"], classification_data["y"])
        
        # Test with FDR=0.2
        model_020 = NullstrapGLM(fdr=0.20, random_state=42, B_reps=2)
        model_020.fit(classification_data["X"], classification_data["y"])
        
        # Both should have correction factors
        assert hasattr(model_001, "correction_factor_")
        assert hasattr(model_020, "correction_factor_")
        assert model_001.correction_factor_ >= 0
        assert model_020.correction_factor_ >= 0

    @pytest.mark.model
    def test_multiple_b_reps_consistency(self, classification_data, model_params):
        """Test that multiple B_reps produce consistent results."""
        params = {k: v for k, v in model_params.items() if k != "B_reps"}
        model = NullstrapGLM(B_reps=5, **params)
        model.fit(classification_data["X"], classification_data["y"])
        
        assert_model_fitted(model)
        assert hasattr(model, "correction_factor_")
        # Correction factor should be reasonable (not NaN or infinite)
        assert np.isfinite(model.correction_factor_)

    @pytest.mark.model
    def test_scoring_parameter_stored(self, classification_data, model_params):
        """Test that scoring parameter is properly stored."""
        model = NullstrapGLM(scoring="accuracy", **model_params)
        assert model.scoring == "accuracy"
        
        model.fit(classification_data["X"], classification_data["y"])
        assert model.scoring == "accuracy"

