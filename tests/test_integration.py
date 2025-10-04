"""
Integration tests for pyNullstrap.

This module tests the integration between different components and
end-to-end workflows.
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nullstrap.models.cox import NullstrapCox
from nullstrap.models.ggm import NullstrapGGM
from nullstrap.models.glm import NullstrapGLM
from nullstrap.models.lm import NullstrapLM


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""

    @pytest.mark.integration
    def test_linear_model_workflow(self, linear_data, model_params):
        """Test complete linear model workflow."""
        model = NullstrapLM(**model_params)

        # Fit model
        model.fit(linear_data["X"], linear_data["y"])

        # Check all expected attributes
        assert hasattr(model, "selected_")
        assert hasattr(model, "threshold_")
        assert hasattr(model, "n_features_selected_")
        assert hasattr(model, "correction_factor_")
        assert hasattr(model, "statistic_")

        # Transform data
        X_transformed = model.transform(linear_data["X"])
        assert X_transformed.shape[0] == linear_data["n_samples"]
        assert X_transformed.shape[1] == model.n_features_selected_

        # Check that transform is consistent
        if model.n_features_selected_ > 0:
            expected = linear_data["X"][:, model.selected_]
            assert np.allclose(X_transformed, expected)

    @pytest.mark.integration
    def test_glm_workflow(self, classification_data, model_params):
        """Test complete GLM workflow."""
        model = NullstrapGLM(**model_params)

        # Fit model
        model.fit(classification_data["X"], classification_data["y"])

        # Check attributes
        assert hasattr(model, "selected_")
        assert hasattr(model, "threshold_")
        assert hasattr(model, "n_features_selected_")
        assert hasattr(model, "correction_factor_")

        # Transform data
        X_transformed = model.transform(classification_data["X"])
        assert X_transformed.shape[0] == classification_data["n_samples"]
        assert X_transformed.shape[1] == model.n_features_selected_


class TestSklearnIntegration:
    """Test integration with scikit-learn components."""

    @pytest.mark.integration
    def test_pipeline_integration(self, linear_data, model_params):
        """Test integration with sklearn Pipeline."""
        # Create pipeline
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("selector", NullstrapLM(**model_params))]
        )

        # Fit pipeline
        pipeline.fit(linear_data["X"], linear_data["y"])

        # Transform data
        X_transformed = pipeline.transform(linear_data["X"])

        # Check dimensions
        assert X_transformed.shape[0] == linear_data["n_samples"]
        assert (
            X_transformed.shape[1]
            == pipeline.named_steps["selector"].n_features_selected_
        )

        # Check that pipeline works with predict
        selector = pipeline.named_steps["selector"]
        assert hasattr(selector, "selected_")
        assert hasattr(selector, "threshold_")

    @pytest.mark.integration
    def test_cross_validation_integration(self, classification_data, model_params):
        """Test integration with cross-validation."""
        # Create pipeline with classifier
        pipeline = Pipeline(
            [
                ("selector", NullstrapGLM(**model_params)),
                ("classifier", LogisticRegression(random_state=42)),
            ]
        )

        # Cross-validation should work
        scores = cross_val_score(
            pipeline,
            classification_data["X"],
            classification_data["y"],
            cv=3,
            scoring="accuracy",
        )

        # Check that scores are reasonable
        assert len(scores) == 3
        assert all(0 <= score <= 1 for score in scores)

    @pytest.mark.integration
    def test_grid_search_integration(self, linear_data):
        """Test integration with parameter grid search."""
        from sklearn.model_selection import GridSearchCV

        # Create model with parameter grid
        model = NullstrapLM(random_state=42)

        param_grid = {"fdr": [0.05, 0.1, 0.2], "alpha_": [None, 0.1, 0.5]}

        # Grid search should work
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring="neg_mean_squared_error"
        )

        grid_search.fit(linear_data["X"], linear_data["y"])

        # Check that best parameters are found
        assert "fdr" in grid_search.best_params_
        assert "alpha_" in grid_search.best_params_
        assert grid_search.best_score_ is not None


class TestModelComparison:
    """Test comparison between different model types."""

    @pytest.mark.integration
    def test_model_consistency(
        self,
        linear_data,
        classification_data,
        survival_data,
        graphical_data,
        model_params,
    ):
        """Test that all models follow consistent patterns."""
        models = [
            ("LM", NullstrapLM(**model_params), linear_data["X"], linear_data["y"]),
            (
                "GLM",
                NullstrapGLM(**model_params),
                classification_data["X"],
                classification_data["y"],
            ),
            (
                "Cox",
                NullstrapCox(**model_params),
                survival_data["X"],
                survival_data["y"],
            ),
            ("GGM", NullstrapGGM(**model_params), graphical_data["X"], None),
        ]

        for name, model, X, y in models:
            # Fit model
            if y is not None:
                model.fit(X, y)
            else:
                model.fit(X)

            # Check common attributes
            assert hasattr(model, "selected_"), f"{name} missing selected_"
            assert hasattr(model, "threshold_"), f"{name} missing threshold_"
            assert hasattr(
                model, "n_features_selected_"
            ), f"{name} missing n_features_selected_"
            assert hasattr(model, "n_samples_"), f"{name} missing n_samples_"
            assert hasattr(model, "n_features_"), f"{name} missing n_features_"

            # Check attribute types
            assert isinstance(
                model.selected_, np.ndarray
            ), f"{name} selected_ should be array"
            assert isinstance(
                model.threshold_, (int, float)
            ), f"{name} threshold_ should be numeric"
            assert isinstance(
                model.n_features_selected_, int
            ), f"{name} n_features_selected_ should be int"

            # Check reasonable values
            assert (
                model.n_features_selected_ >= 0
            ), f"{name} n_features_selected_ should be non-negative"
            assert model.threshold_ >= 0, f"{name} threshold_ should be non-negative"
            assert (
                len(model.selected_) == model.n_features_selected_
            ), f"{name} selected_ length mismatch"

    @pytest.mark.integration
    def test_parameter_consistency(self, linear_data, model_params):
        """Test that parameters are handled consistently across models."""
        models = [
            NullstrapLM(**model_params),
            NullstrapGLM(**model_params),
            NullstrapCox(**model_params),
            NullstrapGGM(**model_params),
        ]

        for model in models:
            # Check that parameters are stored correctly
            assert model.fdr == model_params["fdr"]
            assert model.random_state == model_params["random_state"]

            # Check that B_reps is accessible
            assert hasattr(model, "B_reps")


class TestDataHandling:
    """Test data handling and edge cases."""

    @pytest.mark.integration
    def test_different_data_types(self, rng):
        """Test handling of different data types."""
        n_samples, n_features = 100, 20

        # Test with different data types
        data_types = [
            ("float32", np.float32),
            ("float64", np.float64),
            ("int32", np.int32),
            ("int64", np.int64),
        ]

        for dtype_name, dtype in data_types:
            X = rng.randn(n_samples, n_features).astype(dtype)
            y = rng.randn(n_samples).astype(dtype)

            model = NullstrapLM(fdr=0.1, random_state=42)
            model.fit(X, y)

            assert_model_fitted(model)

    @pytest.mark.integration
    def test_missing_data_handling(self, linear_data, model_params):
        """Test handling of missing data scenarios."""
        # Test with NaN values (should raise appropriate error)
        X_with_nan = linear_data["X"].copy()
        X_with_nan[0, 0] = np.nan

        model = NullstrapLM(**model_params)

        with pytest.raises((ValueError, np.linalg.LinAlgError)):
            model.fit(X_with_nan, linear_data["y"])

    @pytest.mark.integration
    def test_constant_features_handling(self, rng, model_params):
        """Test handling of constant features."""
        n_samples, n_features = 100, 20

        # Create data with constant features
        X = np.ones((n_samples, n_features))
        X[:, :10] = rng.randn(n_samples, 10)  # First 10 features vary
        y = rng.randn(n_samples)

        model = NullstrapLM(**model_params)
        model.fit(X, y)

        assert_model_fitted(model)
        # Should handle constant features gracefully


class TestPerformanceIntegration:
    """Test performance-related integration."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_large_dataset_integration(self, rng):
        """Test integration with larger datasets."""
        n_samples, n_features = 500, 100

        # Generate larger dataset
        X = rng.randn(n_samples, n_features)
        true_coef = np.zeros(n_features)
        true_coef[:20] = rng.randn(20) * 2.0
        y = X @ true_coef + 0.1 * rng.randn(n_samples)

        # Test with reduced parameters for speed
        model = NullstrapLM(fdr=0.1, B_reps=3, random_state=42)

        import time

        start_time = time.time()
        model.fit(X, y)
        elapsed = time.time() - start_time

        assert_model_fitted(model)
        assert elapsed < 120, f"Large dataset fitting took too long: {elapsed:.2f}s"

    @pytest.mark.integration
    def test_memory_efficiency(self, rng):
        """Test memory efficiency with moderate datasets."""
        n_samples, n_features = 200, 50

        X = rng.randn(n_samples, n_features)
        y = rng.randn(n_samples)

        model = NullstrapLM(fdr=0.1, B_reps=3, random_state=42)

        # Should not consume excessive memory
        model.fit(X, y)

        # Check that model doesn't store unnecessary data
        assert hasattr(model, "selected_")
        assert hasattr(model, "threshold_")
        # Should not store full data matrices
        assert not hasattr(model, "X_")
        assert not hasattr(model, "y_")


class TestReproducibilityIntegration:
    """Test reproducibility across different scenarios."""

    @pytest.mark.integration
    def test_cross_platform_reproducibility(self, linear_data):
        """Test that results are reproducible across different runs."""
        model_params = {"fdr": 0.1, "random_state": 42, "B_reps": 3}

        # Run multiple times
        results = []
        for _ in range(3):
            model = NullstrapLM(**model_params)
            model.fit(linear_data["X"], linear_data["y"])
            results.append(
                {
                    "selected": model.selected_.copy(),
                    "threshold": model.threshold_,
                    "n_selected": model.n_features_selected_,
                }
            )

        # All runs should be identical
        for i in range(1, len(results)):
            assert np.array_equal(results[0]["selected"], results[i]["selected"])
            assert results[0]["threshold"] == results[i]["threshold"]
            assert results[0]["n_selected"] == results[i]["n_selected"]

    @pytest.mark.integration
    def test_different_random_states(self, linear_data):
        """Test that different random states give different results."""
        model1 = NullstrapLM(fdr=0.1, random_state=42, B_reps=3)
        model2 = NullstrapLM(fdr=0.1, random_state=123, B_reps=3)

        model1.fit(linear_data["X"], linear_data["y"])
        model2.fit(linear_data["X"], linear_data["y"])

        # Results should be different (with high probability)
        # Note: This test might occasionally fail due to randomness
        # but should pass most of the time
        assert (
            not np.array_equal(model1.selected_, model2.selected_)
            or model1.threshold_ != model2.threshold_
        )


# Helper function for integration tests
def assert_model_fitted(model):
    """Assert that a model has been properly fitted."""
    assert hasattr(model, "selected_"), "Model should have selected_ attribute"
    assert hasattr(model, "threshold_"), "Model should have threshold_ attribute"
    assert hasattr(
        model, "n_features_selected_"
    ), "Model should have n_features_selected_ attribute"
    assert (
        model.n_features_selected_ >= 0
    ), "Number of selected features should be non-negative"
