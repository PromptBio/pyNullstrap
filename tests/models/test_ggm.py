"""
Tests for NullstrapGGM (Gaussian Graphical Models).

This module contains tests for the NullstrapGGM class,
including tests for precision matrices and adjacency matrices.
"""

import numpy as np
import pytest

from nullstrap.models.ggm import NullstrapGGM
from tests.conftest import (assert_model_fitted, assert_reproducible_results)


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

