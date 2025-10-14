"""
Configuration file for pytest containing fixtures and test utilities for pyNullstrap.

This module provides shared fixtures and utilities used across all test modules.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression


@pytest.fixture(scope="session")
def random_seed():
    """Set a consistent random seed for reproducible tests."""
    return 42


@pytest.fixture
def rng(random_seed):
    """Provide a numpy random number generator for tests."""
    return np.random.RandomState(random_seed)


@pytest.fixture
def linear_data(rng):
    """Generate synthetic linear regression data with known signal."""
    n_samples, n_features = 200, 50

    # Create sparse true coefficients
    true_coef = np.zeros(n_features)
    true_coef[:10] = rng.randn(10) * 2.0  # First 10 features have signal

    # Generate design matrix
    X = rng.randn(n_samples, n_features)

    # Generate response with noise
    y = X @ true_coef + 0.1 * rng.randn(n_samples)

    return {
        "X": X,
        "y": y,
        "true_coef": true_coef,
        "true_features": np.arange(10),
        "n_samples": n_samples,
        "n_features": n_features,
    }


@pytest.fixture
def classification_data(rng):
    """Generate synthetic binary classification data with known signal."""
    n_samples, n_features = 150, 40

    # Create sparse true coefficients
    true_coef = np.zeros(n_features)
    true_coef[:8] = rng.randn(8) * 1.5  # First 8 features have signal

    # Generate design matrix
    X = rng.randn(n_samples, n_features)

    # Generate binary response
    linear_predictor = X @ true_coef
    prob = 1 / (1 + np.exp(-linear_predictor))
    y = rng.binomial(1, prob)

    return {
        "X": X,
        "y": y,
        "true_coef": true_coef,
        "true_features": np.arange(8),
        "n_samples": n_samples,
        "n_features": n_features,
    }


@pytest.fixture
def survival_data(rng):
    """Generate synthetic survival data with known signal."""
    n_samples, n_features = 300, 60

    # Create sparse true coefficients
    true_coef = np.zeros(n_features)
    true_coef[:12] = rng.randn(12) * 0.8  # First 12 features have signal

    # Generate design matrix
    X = rng.randn(n_samples, n_features)

    # Generate survival times and events
    hazard = np.exp(X @ true_coef)
    times = rng.exponential(1 / hazard)
    events = rng.binomial(1, 0.7, n_samples).astype(
        bool
    )  # 70% events, convert to boolean

    # Create structured array in scikit-survival format
    # scikit-survival expects structured array with boolean events and float times
    y_survival = np.array(
        [(event, time) for event, time in zip(events, times)],
        dtype=[("event", bool), ("time", float)],
    )

    return {
        "X": X,
        "y": y_survival,
        "true_coef": true_coef,
        "true_features": np.arange(12),
        "n_samples": n_samples,
        "n_features": n_features,
    }


@pytest.fixture
def graphical_data(rng):
    """Generate synthetic data from a sparse precision matrix."""
    n_samples, n_features = 200, 30

    # Create sparse precision matrix with better conditioning
    precision = np.eye(n_features) * 1.5  # Strong diagonal for better conditioning
    # Add some off-diagonal structure with smaller values
    for i in range(0, n_features - 1, 3):
        precision[i, i + 1] = precision[i + 1, i] = 0.15  # Small off-diagonal values

    # Ensure positive definiteness and good conditioning
    precision = precision + np.eye(n_features) * 0.2

    # Verify positive definiteness
    eigenvals = np.linalg.eigvals(precision)
    min_eigenval = np.min(eigenvals)
    if min_eigenval <= 0:
        precision = precision + np.eye(n_features) * (abs(min_eigenval) + 0.1)

    # Generate data
    cov = np.linalg.inv(precision)
    X = rng.multivariate_normal(np.zeros(n_features), cov, n_samples)

    return {
        "X": X,
        "true_precision": precision,
        "true_edges": [(i, i + 1) for i in range(0, n_features - 1, 3)],
        "n_samples": n_samples,
        "n_features": n_features,
    }


@pytest.fixture
def high_dimensional_data(rng):
    """Generate high-dimensional data (p > n) for testing."""
    n_samples, n_features = 100, 200

    # Create very sparse signal
    true_coef = np.zeros(n_features)
    true_coef[:5] = rng.randn(5) * 3.0  # Only first 5 features have signal

    X = rng.randn(n_samples, n_features)
    y = X @ true_coef + 0.1 * rng.randn(n_samples)

    return {
        "X": X,
        "y": y,
        "true_coef": true_coef,
        "true_features": np.arange(5),
        "n_samples": n_samples,
        "n_features": n_features,
    }


@pytest.fixture
def edge_case_data():
    """Generate edge case data for testing robustness."""
    return {
        "single_feature": {
            "X": np.array([[1], [2], [3], [4], [5]]),
            "y": np.array([1, 2, 3, 4, 5]),
        },
        "two_samples": {"X": np.array([[1, 2], [3, 4]]), "y": np.array([1, 2])},
        "constant_features": {"X": np.ones((10, 5)), "y": np.random.randn(10)},
        "duplicate_features": {
            "X": np.column_stack([np.random.randn(20, 3), np.random.randn(20, 3)]),
            "y": np.random.randn(20),
        },
    }


@pytest.fixture
def model_params():
    """Default parameters for testing models."""
    return {
        "fdr": 0.1,
        "alpha_": None,  # Let model choose automatically
        "B_reps": 3,  # Reduced for faster testing
        "random_state": 42,
        "max_iter": 1000,
    }


@pytest.fixture
def strict_model_params():
    """Strict parameters for testing model behavior."""
    return {
        "fdr": 0.05,  # More conservative
        "alpha_": 0.1,  # Fixed regularization
        "B_reps": 5,  # More bootstrap reps
        "random_state": 42,
        "max_iter": 2000,
    }


# Test markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "model: marks tests for model classes")
    config.addinivalue_line("markers", "utils: marks tests for utility functions")
    config.addinivalue_line(
        "markers", "cox: marks tests that require scikit-survival (skip on Windows)"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their names."""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid or "_int_" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Mark unit tests
        if "unit" in item.nodeid or "_unit_" in item.nodeid:
            item.add_marker(pytest.mark.unit)

        # Mark model tests
        if any(model in item.nodeid for model in ["lm", "glm", "cox", "ggm"]):
            item.add_marker(pytest.mark.model)

        # Mark Cox tests (require scikit-survival)
        if "cox" in item.nodeid.lower() or "TestNullstrapCox" in item.nodeid:
            item.add_marker(pytest.mark.cox)

        # Mark utility tests
        if "utils" in item.nodeid or "core" in item.nodeid:
            item.add_marker(pytest.mark.utils)


# Utility functions for tests
def assert_array_almost_equal(a, b, decimal=6):
    """Assert that two arrays are almost equal."""
    np.testing.assert_array_almost_equal(a, b, decimal=decimal)


def assert_fdr_control(selected_features, true_features, fdr_level=0.1, tolerance=0.05):
    """Assert that FDR is controlled within tolerance."""
    if len(selected_features) == 0:
        return  # No discoveries, FDR is undefined

    false_positives = len(set(selected_features) - set(true_features))
    false_discovery_rate = false_positives / len(selected_features)

    assert (
        false_discovery_rate <= fdr_level + tolerance
    ), f"FDR {false_discovery_rate:.3f} exceeds target {fdr_level:.3f} + tolerance {tolerance:.3f}"


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


def assert_reproducible_results(model1, model2, tolerance=1e-10):
    """Assert that two models with same random_state produce identical results."""
    assert np.allclose(
        model1.selected_, model2.selected_
    ), "Selected features should be identical"
    assert (
        abs(model1.threshold_ - model2.threshold_) < tolerance
    ), "Thresholds should be identical"
    assert (
        model1.n_features_selected_ == model2.n_features_selected_
    ), "Number of selected features should be identical"
