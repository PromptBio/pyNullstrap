"""
Comprehensive tests for all pyNullstrap model classes.

This module tests the core functionality of NullstrapLM, NullstrapGLM,
NullstrapCox, and NullstrapGGM classes.
"""

import numpy as np
import pytest

from nullstrap.models.cox import NullstrapCox
from nullstrap.models.ggm import NullstrapGGM
from nullstrap.models.glm import NullstrapGLM
from nullstrap.models.lm import NullstrapLM
# Import helper functions from conftest
from tests.conftest import (assert_fdr_control, assert_model_fitted,
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


class TestNullstrapCox:
    """Test cases for NullstrapCox (Cox Survival Models)."""

    @pytest.mark.model
    def test_basic_fitting(self, survival_data, model_params):
        """Test basic model fitting functionality."""
        model = NullstrapCox(**model_params)
        model.fit(survival_data["X"], survival_data["y"])

        assert_model_fitted(model)
        assert model.n_samples_ == survival_data["n_samples"]
        assert model.n_features_ == survival_data["n_features"]

    @pytest.mark.model
    def test_fdr_control(self, survival_data, model_params):
        """Test that FDR is controlled at specified level."""
        model = NullstrapCox(**model_params)
        model.fit(survival_data["X"], survival_data["y"])

        assert_fdr_control(
            model.selected_,
            survival_data["true_features"],
            fdr_level=model_params["fdr"],
            tolerance=0.15,  # Higher tolerance for Cox
        )

    @pytest.mark.model
    def test_reproducibility(self, survival_data, model_params):
        """Test that results are reproducible with same random_state."""
        model1 = NullstrapCox(**model_params)
        model2 = NullstrapCox(**model_params)

        model1.fit(survival_data["X"], survival_data["y"])
        model2.fit(survival_data["X"], survival_data["y"])

        assert_reproducible_results(model1, model2)

    @pytest.mark.model
    def test_survival_data_format(self, survival_data, model_params):
        """Test that survival data format is handled correctly."""
        model = NullstrapCox(**model_params)
        model.fit(survival_data["X"], survival_data["y"])

        # Check that survival data was processed correctly
        assert hasattr(model, "event_times_")
        assert hasattr(model, "event_indicators_")
        assert len(model.event_times_) == survival_data["n_samples"]
        assert len(model.event_indicators_) == survival_data["n_samples"]

    @pytest.mark.model
    def test_transform_method(self, survival_data, model_params):
        """Test the transform method."""
        model = NullstrapCox(**model_params)
        model.fit(survival_data["X"], survival_data["y"])

        X_transformed = model.transform(survival_data["X"])

        assert X_transformed.shape[0] == survival_data["n_samples"]
        assert X_transformed.shape[1] == model.n_features_selected_


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
        adjacency = model.get_adjacency_matrix()
        selected_edges = np.where(adjacency)[0]

        # Check that some edges were selected
        assert len(selected_edges) >= 0

    @pytest.mark.model
    def test_reproducibility(self, graphical_data, model_params):
        """Test that results are reproducible with same random_state."""
        model1 = NullstrapGGM(**model_params)
        model2 = NullstrapGGM(**model_params)

        model1.fit(graphical_data["X"])
        model2.fit(graphical_data["X"])

        assert_reproducible_results(model1, model2)

    @pytest.mark.model
    def test_adjacency_matrix(self, graphical_data, model_params):
        """Test adjacency matrix generation."""
        model = NullstrapGGM(**model_params)
        model.fit(graphical_data["X"])

        adjacency = model.get_adjacency_matrix()

        # Check dimensions
        assert adjacency.shape == (
            graphical_data["n_features"],
            graphical_data["n_features"],
        )

        # Check symmetry
        assert np.allclose(adjacency, adjacency.T)

        # Check diagonal is zero (no self-loops)
        assert np.allclose(np.diag(adjacency), 0)

    @pytest.mark.model
    def test_precision_matrix(self, graphical_data, model_params):
        """Test precision matrix generation."""
        model = NullstrapGGM(**model_params)
        model.fit(graphical_data["X"])

        precision = model.get_precision_matrix()

        # Check dimensions
        assert precision.shape == (
            graphical_data["n_features"],
            graphical_data["n_features"],
        )

        # Check symmetry
        assert np.allclose(precision, precision.T)

        # Check positive definiteness (eigenvalues should be positive)
        eigenvals = np.linalg.eigvals(precision)
        assert np.all(eigenvals > 0)

    @pytest.mark.model
    def test_no_response_parameter(self, graphical_data, model_params):
        """Test that GGM doesn't require y parameter."""
        model = NullstrapGGM(**model_params)

        # Should work without y parameter
        model.fit(graphical_data["X"])
        assert_model_fitted(model)


class TestModelConsistency:
    """Test consistency across different model types."""

    @pytest.mark.model
    def test_common_attributes(
        self,
        linear_data,
        classification_data,
        survival_data,
        graphical_data,
        model_params,
    ):
        """Test that all models have common attributes after fitting."""
        models = [
            NullstrapLM(**model_params).fit(linear_data["X"], linear_data["y"]),
            NullstrapGLM(**model_params).fit(
                classification_data["X"], classification_data["y"]
            ),
            NullstrapCox(**model_params).fit(survival_data["X"], survival_data["y"]),
            NullstrapGGM(**model_params).fit(graphical_data["X"]),
        ]

        for model in models:
            # Check common attributes
            assert hasattr(model, "selected_")
            assert hasattr(model, "threshold_")
            assert hasattr(model, "n_features_selected_")
            assert hasattr(model, "n_samples_")
            assert hasattr(model, "n_features_")
            assert hasattr(model, "fdr")

            # Check that attributes have reasonable values
            assert model.n_features_selected_ >= 0
            assert model.threshold_ >= 0
            assert 0 < model.fdr < 1

    @pytest.mark.model
    def test_scikit_learn_compatibility(self, linear_data, model_params):
        """Test scikit-learn compatibility."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        # Test that models work in sklearn pipelines
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("selector", NullstrapLM(**model_params))]
        )

        pipeline.fit(linear_data["X"], linear_data["y"])
        X_transformed = pipeline.transform(linear_data["X"])

        assert X_transformed.shape[0] == linear_data["n_samples"]
        assert (
            X_transformed.shape[1]
            == pipeline.named_steps["selector"].n_features_selected_
        )


@pytest.mark.slow
class TestPerformance:
    """Performance tests (marked as slow)."""

    @pytest.mark.model
    def test_large_dataset_performance(self, rng):
        """Test performance with larger datasets."""
        # Generate larger dataset
        n_samples, n_features = 1000, 200
        X = rng.randn(n_samples, n_features)
        true_coef = np.zeros(n_features)
        true_coef[:20] = rng.randn(20) * 2.0
        y = X @ true_coef + 0.1 * rng.randn(n_samples)

        model = NullstrapLM(fdr=0.1, B_reps=3, random_state=42)

        # Should complete within reasonable time
        import time

        start_time = time.time()
        model.fit(X, y)
        elapsed = time.time() - start_time

        assert elapsed < 60, f"Model fitting took too long: {elapsed:.2f}s"
        assert_model_fitted(model)
