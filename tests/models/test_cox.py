"""
Tests for NullstrapCox (Cox Proportional Hazards Models).

This module contains tests for the NullstrapCox class,
including tests for survival data handling and model functionality.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from nullstrap.models.cox import NullstrapCox
from ..conftest import (assert_fdr_control, assert_model_fitted,
                        assert_reproducible_results)


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
        assert hasattr(model, "alpha_used_")
        assert model.alpha_used_ > 0

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
    def test_transform_method(self, survival_data, model_params):
        """Test the transform method."""
        model = NullstrapCox(**model_params)
        model.fit(survival_data["X"], survival_data["y"])

        X_transformed = model.transform(survival_data["X"])

        assert X_transformed.shape[0] == survival_data["n_samples"]
        assert X_transformed.shape[1] == model.n_features_selected_

    @pytest.mark.model
    def test_baseline_hazard_extraction(self, survival_data, model_params):
        """Test baseline hazard function extraction and properties."""
        model = NullstrapCox(**model_params)
        model.fit(survival_data["X"], survival_data["y"])

        # Check baseline hazard function exists and is callable
        assert hasattr(model, "baseline_hazard_")
        assert callable(model.baseline_hazard_)
        
        # Test baseline hazard function with scalar input
        hazard_scalar = model.baseline_hazard_(1.0)
        assert hazard_scalar >= 0, "Baseline hazard should be non-negative"
        
        # Test baseline hazard function with array input
        test_times = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        hazard_values = model.baseline_hazard_(test_times)
        
        # Check that hazard values are non-negative
        assert np.all(hazard_values >= 0), "Baseline hazard should be non-negative"
        
        # Check that hazard is monotonic (cumulative hazard should be non-decreasing)
        assert np.all(np.diff(hazard_values) >= -1e-10), "Cumulative hazard should be monotonic"

    @pytest.mark.model
    def test_synthetic_data_generation(self, survival_data, model_params):
        """Test synthetic survival data generation."""
        model = NullstrapCox(**model_params)
        model.fit(survival_data["X"], survival_data["y"])

        # Test generating synthetic data with null signal
        X_test = survival_data["X"][:10]
        synthetic_data = model._generate_synthetic_data(X_test, beta=None)
        
        # Check output format
        assert synthetic_data.dtype.names == ('event', 'time'), "Should have event and time fields"
        assert synthetic_data.shape[0] == X_test.shape[0], "Should generate same number of samples"
        
        # Check that all events are True (no censoring in synthetic data)
        assert np.all(synthetic_data['event']), "All synthetic events should be True"
        
        # Check that times are within time range
        assert np.all(synthetic_data['time'] >= model.time_range_[0])
        assert np.all(synthetic_data['time'] <= model.time_range_[1])

    @pytest.mark.model
    def test_synthetic_data_with_signal(self, survival_data, model_params):
        """Test synthetic data generation with covariate effects."""
        model = NullstrapCox(**model_params)
        model.fit(survival_data["X"], survival_data["y"])

        X_test = survival_data["X"][:10]
        beta_test = np.zeros(survival_data["n_features"])
        beta_test[:3] = [0.5, -0.3, 0.8]
        
        synthetic_data = model._generate_synthetic_data(X_test, beta=beta_test)
        
        # Check output format
        assert synthetic_data.dtype.names == ('event', 'time')
        assert synthetic_data.shape[0] == X_test.shape[0]
        assert np.all(synthetic_data['event'])
        assert np.all(synthetic_data['time'] >= model.time_range_[0])
        assert np.all(synthetic_data['time'] <= model.time_range_[1])

    @pytest.mark.model
    def test_time_range_attribute(self, survival_data, model_params):
        """Test that time_range_ attribute is set correctly."""
        model = NullstrapCox(**model_params)
        model.fit(survival_data["X"], survival_data["y"])

        assert hasattr(model, "time_range_")
        assert isinstance(model.time_range_, tuple)
        assert len(model.time_range_) == 2
        assert model.time_range_[0] <= model.time_range_[1]
        
        # Check that time range matches observed data
        observed_times = survival_data["y"]["time"]
        assert model.time_range_[0] == np.min(observed_times)
        assert model.time_range_[1] == np.max(observed_times)

    @pytest.mark.model
    def test_data_validation_2d_array(self, survival_data):
        """Test that 2D array format is auto-converted correctly."""
        model = NullstrapCox(random_state=42, B_reps=2)
        
        # Create a simple 2D array manually to avoid any indexing issues
        n_samples = 20
        events_2d = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        times_2d = np.array([1.2, 2.3, 0.8, 1.5, 3.1, 0.9, 2.1, 1.3, 0.7, 2.8, 
                           1.1, 2.5, 0.6, 1.8, 3.2, 0.5, 2.7, 1.4, 0.9, 2.2])
        y_2d = np.column_stack([events_2d, times_2d])
        
        # Create simple X data
        X_simple = np.random.randn(n_samples, 10)
        
        # Should work with auto-conversion warning
        model.fit(X_simple, y_2d)
        assert_model_fitted(model)

    @pytest.mark.model
    def test_data_validation_invalid_events(self, survival_data):
        """Test data validation for invalid event values."""
        model = NullstrapCox(random_state=42)
        
        # Create 2D array with invalid event values
        events = np.array([0, 1, 2, 0, 1])  # 2 is invalid
        times = survival_data["y"]["time"][:5]
        y_invalid = np.column_stack([events, times])
        
        X_subset = survival_data["X"][:5]
        
        with pytest.raises(ValueError, match="Events must be binary"):
            model.fit(X_subset, y_invalid)

    @pytest.mark.model
    def test_data_validation_negative_times(self, survival_data):
        """Test data validation for negative times."""
        model = NullstrapCox(random_state=42)
        
        y_invalid = survival_data["y"].copy()
        y_invalid["time"][0] = -1.0
        
        with pytest.raises(ValueError, match="must be non-negative"):
            model.fit(survival_data["X"], y_invalid)

    @pytest.mark.model
    def test_correction_factor_estimation(self, survival_data, model_params):
        """Test correction factor estimation."""
        model = NullstrapCox(**model_params)
        model.fit(survival_data["X"], survival_data["y"])

        assert hasattr(model, "correction_factor_")
        assert isinstance(model.correction_factor_, (int, float))
        assert model.correction_factor_ >= 0

    @pytest.mark.model
    def test_alpha_selection_cv(self, survival_data):
        """Test alpha selection via cross-validation."""
        model = NullstrapCox(
            fdr=0.1,
            alpha_=None,  # Let model choose via CV
            B_reps=2,
            cv_folds=3,
            random_state=42,
        )
        
        model.fit(survival_data["X"], survival_data["y"])

        assert hasattr(model, "alpha_used_")
        assert model.alpha_used_ > 0
        assert_model_fitted(model)

    @pytest.mark.model
    def test_fixed_alpha(self, survival_data):
        """Test with fixed alpha parameter."""
        model = NullstrapCox(
            fdr=0.1,
            alpha_=0.01,  # Lower alpha to ensure some features are selected
            B_reps=2,
            random_state=42,
        )
        
        model.fit(survival_data["X"], survival_data["y"])

        assert model.alpha_used_ == 0.01
        assert_model_fitted(model)

    @pytest.mark.model
    def test_no_features_selected(self, survival_data):
        """Test handling when no features are selected due to high regularization."""
        model = NullstrapCox(
            fdr=0.1,
            alpha_=1000.0,  # Very high alpha to ensure no features selected
            B_reps=2,
            random_state=42,
        )
        
        model.fit(survival_data["X"], survival_data["y"])

        # Model should still fit successfully
        assert_model_fitted(model)
        
        # Should have baseline hazard function (fallback)
        assert hasattr(model, "baseline_hazard_")
        assert callable(model.baseline_hazard_)
        
        # Test that baseline hazard works
        test_times = np.array([0.1, 1.0, 2.0])
        hazard_values = model.baseline_hazard_(test_times)
        assert np.all(hazard_values >= 0)

    @pytest.mark.model
    def test_all_attributes_set(self, survival_data, model_params):
        """Test that all expected attributes are set after fitting."""
        model = NullstrapCox(**model_params)
        model.fit(survival_data["X"], survival_data["y"])

        # Check all expected attributes
        expected_attrs = [
            "n_samples_", "n_features_",
            "threshold_", "selected_", "n_features_selected_",
            "statistic_", "alpha_used_", "correction_factor_",
            "baseline_hazard_", "time_range_"
        ]
        
        for attr in expected_attrs:
            assert hasattr(model, attr), f"Model should have {attr} attribute"

    @pytest.mark.model
    def test_invalid_alphas_negative(self):
        """Test validation for negative alphas (int)."""
        with pytest.raises(ValueError, match="alphas .* must be > 0"):
            model = NullstrapCox(alphas=-1, random_state=42)
            model._validate_parameters()

    @pytest.mark.model
    def test_invalid_alphas_empty_sequence(self):
        """Test validation for empty alphas sequence."""
        with pytest.raises(ValueError, match="alphas .* cannot be empty"):
            model = NullstrapCox(alphas=[], random_state=42)
            model._validate_parameters()

    @pytest.mark.model
    def test_invalid_alphas_negative_in_sequence(self):
        """Test validation for negative values in alphas sequence."""
        with pytest.raises(ValueError, match="all alphas must be > 0"):
            model = NullstrapCox(alphas=[0.1, -0.5, 1.0], random_state=42)
            model._validate_parameters()

    @pytest.mark.model
    def test_invalid_alphas_type(self):
        """Test validation for invalid alphas type."""
        with pytest.raises(ValueError, match="alphas must be int or sequence"):
            model = NullstrapCox(alphas="invalid", random_state=42)
            model._validate_parameters()

    @pytest.mark.model
    def test_explicit_alphas_array(self, survival_data):
        """Test with explicit alphas array."""
        model = NullstrapCox(
            fdr=0.1,
            alpha_=None,
            alphas=[0.001, 0.01, 0.1, 1.0],  # Explicit alpha values
            B_reps=2,
            cv_folds=3,
            random_state=42,
        )
        
        model.fit(survival_data["X"], survival_data["y"])
        assert_model_fitted(model)
        assert model.alpha_used_ > 0

    @pytest.mark.model
    def test_fit_without_y_parameter(self, survival_data):
        """Test that fitting without y raises ValueError."""
        model = NullstrapCox(random_state=42)
        
        with pytest.raises(ValueError, match="Cox models require y parameter"):
            model.fit(survival_data["X"], None)

    @pytest.mark.model
    def test_all_censored_data(self, survival_data):
        """Test with all censored data (no events)."""
        model = NullstrapCox(random_state=42)
        
        # Create data where all events are censored
        y_all_censored = survival_data["y"].copy()
        y_all_censored["event"][:] = False
        
        with pytest.raises(ValueError, match="No events observed"):
            model.fit(survival_data["X"], y_all_censored)

    @pytest.mark.model
    def test_all_events_no_censoring(self, survival_data):
        """Test with all events (no censoring) - should warn."""
        model = NullstrapCox(B_reps=2, random_state=42)
        
        # Create data where all observations are events
        y_all_events = survival_data["y"].copy()
        y_all_events["event"][:] = True
        
        # Should fit but warn about no censoring
        with pytest.warns(UserWarning, match="All observations are events"):
            model.fit(survival_data["X"], y_all_events)
        
        assert_model_fitted(model)

    @pytest.mark.model
    def test_nan_in_times(self, survival_data):
        """Test validation for NaN values in times."""
        model = NullstrapCox(random_state=42)
        
        y_with_nan = survival_data["y"].copy()
        y_with_nan["time"][0] = np.nan
        
        with pytest.raises(ValueError, match="contains NaN or infinite"):
            model.fit(survival_data["X"], y_with_nan)

    @pytest.mark.model
    def test_inf_in_times(self, survival_data):
        """Test validation for infinite values in times."""
        model = NullstrapCox(random_state=42)
        
        y_with_inf = survival_data["y"].copy()
        y_with_inf["time"][0] = np.inf
        
        with pytest.raises(ValueError, match="contains NaN or infinite"):
            model.fit(survival_data["X"], y_with_inf)

    @pytest.mark.model
    def test_malformed_structured_array(self, survival_data):
        """Test validation for structured array with wrong field names."""
        model = NullstrapCox(random_state=42)
        
        # Create structured array with wrong field names
        y_malformed = np.zeros(
            survival_data["n_samples"],
            dtype=[("wrong_event", bool), ("wrong_time", float)]
        )
        
        with pytest.raises(ValueError, match="must be structured array with 'event' and 'time'"):
            model.fit(survival_data["X"], y_malformed)

    @pytest.mark.model
    def test_non_boolean_event_dtype(self, survival_data):
        """Test validation for non-boolean event dtype."""
        model = NullstrapCox(random_state=42)
        
        # Create structured array with integer events instead of boolean
        y_int_events = np.zeros(
            survival_data["n_samples"],
            dtype=[("event", int), ("time", float)]
        )
        y_int_events["event"] = survival_data["y"]["event"].astype(int)
        y_int_events["time"] = survival_data["y"]["time"]
        
        with pytest.raises(ValueError, match="event.*must be boolean"):
            model.fit(survival_data["X"], y_int_events)

    @pytest.mark.model
    def test_non_numeric_time_dtype(self, survival_data):
        """Test validation for non-numeric time dtype."""
        model = NullstrapCox(random_state=42)
        
        # Create structured array with string times
        y_string_times = np.zeros(
            survival_data["n_samples"],
            dtype=[("event", bool), ("time", "U10")]
        )
        y_string_times["event"] = survival_data["y"]["event"]
        y_string_times["time"] = ["1.0"] * survival_data["n_samples"]
        
        with pytest.raises(ValueError, match="time.*must be numeric"):
            model.fit(survival_data["X"], y_string_times)

    @pytest.mark.model
    def test_baseline_hazard_fallback(self, survival_data):
        """Test baseline hazard fallback when extraction fails."""
        model = NullstrapCox(fdr=0.1, alpha_=0.01, B_reps=2, random_state=42)
        
        # Create a mock model that will fail during baseline hazard extraction
        mock_model = MagicMock()
        mock_model.predict_cumulative_hazard_function.side_effect = RuntimeError("Simulated extraction failure")
        
        # Directly test _get_baseline_hazard with the broken model
        with pytest.warns(UserWarning, match="Baseline hazard extraction failed.*Using constant hazard fallback"):
            fallback_hazard = model._get_baseline_hazard(mock_model, survival_data["X"])
        
        # Verify the fallback function works correctly
        assert callable(fallback_hazard)
        
        # Test the fallback baseline hazard function (should be lambda t: 0.1 * np.maximum(t, 0))
        test_times = np.array([0.0, 1.0, 2.0, 5.0])
        expected_hazards = 0.1 * test_times
        hazard_values = fallback_hazard(test_times)
        np.testing.assert_array_almost_equal(hazard_values, expected_hazards)
        
        # Test with negative time (should be clamped to 0)
        negative_time_hazard = fallback_hazard(-1.0)
        assert negative_time_hazard == 0.0
