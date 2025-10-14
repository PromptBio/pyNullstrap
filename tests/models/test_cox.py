"""
Tests for NullstrapCox (Cox Proportional Hazards Models).

This module contains tests for the NullstrapCox class,
including tests for survival data handling and model functionality.
"""

import numpy as np
import pytest

from nullstrap.models.cox import NullstrapCox
from tests.conftest import (assert_fdr_control, assert_model_fitted,
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

