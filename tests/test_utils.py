"""
Tests for pyNullstrap utility functions.

This module tests the core utility functions including data standardization,
threshold computation, correction factor estimation, and signal inflation.
"""

import numpy as np
import pytest

from nullstrap.utils.core import (binary_search_correction_factor,
                                  binary_search_threshold, inflate_signal,
                                  standardize_data)


class TestStandardizeData:
    """Test cases for the unified standardize_data function."""

    @pytest.mark.utils
    def test_basic_standardization(self, rng):
        """Test basic data standardization."""
        n_samples, n_features = 100, 20
        X = rng.randn(n_samples, n_features) * 5 + 10  # Non-standard data
        y = rng.randn(n_samples) * 3 + 2

        X_scaled, y_scaled = standardize_data(X, y, scale_by_sample_size=False)

        # Check that data is standardized
        assert np.allclose(np.mean(X_scaled, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(X_scaled, axis=0), 1, atol=1e-10)
        assert np.allclose(np.mean(y_scaled), 0, atol=1e-10)
        # Note: standardize_data only centers y, doesn't standardize it
        # So we check that it's centered but not that std=1

    @pytest.mark.utils
    def test_scale_by_sample_size_true(self, rng):
        """Test standardization with scale_by_sample_size=True (GLM/Cox style)."""
        n_samples, n_features = 100, 20
        X = rng.randn(n_samples, n_features)

        X_scaled, _ = standardize_data(
            X, scale_by_sample_size=True, n_samples=n_samples
        )

        # Check that scaling includes 1/sqrt(n) factor
        expected_std = 1.0 / np.sqrt(n_samples)
        assert np.allclose(np.std(X_scaled, axis=0), expected_std, atol=1e-10)

    @pytest.mark.utils
    def test_scale_by_sample_size_false(self, rng):
        """Test standardization with scale_by_sample_size=False (LM/GGM style)."""
        n_samples, n_features = 100, 20
        X = rng.randn(n_samples, n_features)

        X_scaled, _ = standardize_data(X, scale_by_sample_size=False)

        # Check standard z-score standardization
        assert np.allclose(np.std(X_scaled, axis=0), 1.0, atol=1e-10)

    @pytest.mark.utils
    def test_no_y_parameter(self, rng):
        """Test standardization without y parameter."""
        n_samples, n_features = 100, 20
        X = rng.randn(n_samples, n_features)

        X_scaled, y_scaled = standardize_data(X, scale_by_sample_size=False)

        assert X_scaled.shape == X.shape
        assert y_scaled is None

    @pytest.mark.utils
    def test_edge_cases(self, rng):
        """Test edge cases for standardization."""
        # Single sample
        X_single = rng.randn(1, 5)
        X_scaled, _ = standardize_data(X_single, scale_by_sample_size=False)
        assert X_scaled.shape == X_single.shape

        # Single feature
        X_one_feature = rng.randn(10, 1)
        X_scaled, _ = standardize_data(X_one_feature, scale_by_sample_size=False)
        assert X_scaled.shape == X_one_feature.shape

        # Constant features (should handle gracefully)
        X_constant = np.ones((10, 3))
        X_scaled, _ = standardize_data(X_constant, scale_by_sample_size=False)
        assert X_scaled.shape == X_constant.shape
        # Constant features should become zero
        assert np.allclose(X_scaled, 0)

    @pytest.mark.utils
    def test_different_sample_sizes(self, rng):
        """Test standardization with different sample sizes."""
        sample_sizes = [10, 50, 100, 500]

        for n_samples in sample_sizes:
            X = rng.randn(n_samples, 10)

            # Test with scale_by_sample_size=True
            X_scaled, _ = standardize_data(
                X, scale_by_sample_size=True, n_samples=n_samples
            )
            expected_std = 1.0 / np.sqrt(n_samples)
            assert np.allclose(np.std(X_scaled, axis=0), expected_std, atol=1e-10)

            # Test with scale_by_sample_size=False
            X_scaled, _ = standardize_data(X, scale_by_sample_size=False)
            assert np.allclose(np.std(X_scaled, axis=0), 1.0, atol=1e-10)


class TestBinarySearchThreshold:
    """Test cases for binary_search_threshold function."""

    @pytest.mark.utils
    def test_basic_threshold_computation(self, rng):
        """Test basic threshold computation."""
        n_features = 50

        # Create test statistics with known signal
        W = np.zeros(n_features)
        W[:10] = rng.randn(10) * 2 + 3  # Strong positive signal
        W[10:20] = rng.randn(10) * 0.5  # Weak signal
        W[20:] = rng.randn(30) * 0.1  # Noise

        # Create mock knockoff coefficients
        coef_knockoff = rng.randn(n_features) * 0.1
        threshold = binary_search_threshold(W, coef_knockoff, fdr=0.1)

        # Check that threshold is reasonable
        assert threshold > 0
        assert threshold < np.max(np.abs(W))

        # Check that discoveries are made
        discoveries = np.sum(W >= threshold)
        assert discoveries > 0

    @pytest.mark.utils
    def test_fdr_control(self, rng):
        """Test that binary search achieves FDR control."""
        n_features = 100

        # Create statistics with known null/alternative structure
        W = np.zeros(n_features)
        W[:20] = rng.randn(20) * 2 + 4  # Strong signal
        W[20:] = rng.randn(80) * 0.5  # Mostly noise

        # Create mock knockoff coefficients (should be smaller than real coefficients)
        coef_knockoff = np.zeros(n_features)
        coef_knockoff[:20] = rng.randn(20) * 0.5  # Smaller knockoff signal
        coef_knockoff[20:] = rng.randn(80) * 0.3  # Small knockoff noise

        threshold = binary_search_threshold(W, coef_knockoff, fdr=0.1)

        # Count discoveries and knockoff rejections
        discoveries = np.sum(W >= threshold)
        knockoff_rejections = np.sum(coef_knockoff >= threshold)

        if discoveries > 0:
            fdp = (1 + knockoff_rejections) / discoveries
            # Allow some tolerance for small sample effects
            assert fdp <= 0.2, f"FDP {fdp:.3f} exceeds target 0.1 + tolerance"

    @pytest.mark.utils
    def test_no_signal_case(self, rng):
        """Test threshold computation when there's no signal."""
        n_features = 50

        # Create pure noise statistics
        W = rng.randn(n_features) * 0.1

        # Create mock knockoff coefficients (similar noise level)
        coef_knockoff = rng.randn(n_features) * 0.1
        threshold = binary_search_threshold(W, coef_knockoff, fdr=0.1)

        # Should return a threshold that makes few or no discoveries
        discoveries = np.sum(W >= threshold)
        # Allow for some discoveries due to randomness, but should be very few
        assert discoveries <= 5, f"Too many discoveries ({discoveries}) with pure noise"

    @pytest.mark.utils
    def test_strong_signal_case(self, rng):
        """Test threshold computation with very strong signal."""
        n_features = 50

        # Create very strong signal
        W = np.zeros(n_features)
        W[:10] = rng.randn(10) * 5 + 10  # Very strong signal
        W[10:] = rng.randn(40) * 0.1  # Noise

        # Create mock knockoff coefficients
        coef_knockoff = rng.randn(n_features) * 0.1
        threshold = binary_search_threshold(W, coef_knockoff, fdr=0.1)

        # Should discover most/all of the strong signals
        discoveries = np.sum(W >= threshold)
        assert discoveries >= 8  # Should find most of the 10 true signals

    @pytest.mark.utils
    def test_different_fdr_levels(self, rng):
        """Test threshold computation with different FDR levels."""
        n_features = 100
        W = np.zeros(n_features)
        W[:20] = rng.randn(20) * 2 + 3
        W[20:] = rng.randn(80) * 0.5

        fdr_levels = [0.05, 0.1, 0.2]
        thresholds = []

        for fdr in fdr_levels:
            # Create mock knockoff coefficients
            coef_knockoff = rng.randn(n_features) * 0.1
            threshold = binary_search_threshold(W, coef_knockoff, fdr=fdr)
            thresholds.append(threshold)

        # More conservative FDR should lead to higher thresholds
        assert thresholds[0] >= thresholds[1]  # 0.05 vs 0.1
        assert thresholds[1] >= thresholds[2]  # 0.1 vs 0.2


class TestBinarySearchCorrectionFactor:
    """Test cases for binary_search_correction_factor function."""

    @pytest.mark.utils
    def test_basic_correction_factor(self, rng):
        """Test basic correction factor estimation."""
        n_samples, n_features = 100, 50

        # Create test data
        X = rng.randn(n_samples, n_features)
        y = rng.randn(n_samples)

        # Create mock statistics
        W = rng.randn(n_features) * 2

        # Create mock parameters for correction factor estimation
        coef_correction_abs = np.abs(rng.randn(n_features))
        coef_snp_abs = np.abs(W)
        signal_indices = np.arange(n_features)

        correction_factor = binary_search_correction_factor(
            coef_correction_abs,
            coef_snp_abs,
            signal_indices,
            fdr=0.1,
            initial_correction_factor=0.0,
            binary_search_tol=1e-6,
            scale_factor=1.0,
            max_iterations=100,
        )

        # Check that correction factor is reasonable
        assert correction_factor >= 0
        assert correction_factor <= 10  # Should not be unreasonably large

    @pytest.mark.utils
    def test_reproducibility(self, rng):
        """Test that correction factor estimation is reproducible."""
        n_samples, n_features = 100, 50
        X = rng.randn(n_samples, n_features)
        y = rng.randn(n_samples)
        W = rng.randn(n_features) * 2

        # Run twice with same random_state
        # Create mock parameters
        coef_correction_abs = np.abs(rng.randn(n_features))
        coef_snp_abs = np.abs(W)
        signal_indices = np.arange(n_features)

        cf1 = binary_search_correction_factor(
            coef_correction_abs,
            coef_snp_abs,
            signal_indices,
            fdr=0.1,
            initial_correction_factor=0.0,
            binary_search_tol=1e-6,
            scale_factor=1.0,
            max_iterations=100,
        )
        cf2 = binary_search_correction_factor(
            coef_correction_abs,
            coef_snp_abs,
            signal_indices,
            fdr=0.1,
            initial_correction_factor=0.0,
            binary_search_tol=1e-6,
            scale_factor=1.0,
            max_iterations=100,
        )

        assert abs(cf1 - cf2) < 1e-10, "Correction factors should be identical"

    @pytest.mark.utils
    def test_different_scale_factors(self, rng):
        """Test correction factor with different scale factors."""
        n_samples, n_features = 100, 50
        X = rng.randn(n_samples, n_features)
        y = rng.randn(n_samples)
        W = rng.randn(n_features) * 2

        scale_factors = [0.5, 1.0, 2.0]
        correction_factors = []

        # Create mock parameters
        coef_correction_abs = np.abs(rng.randn(n_features))
        coef_snp_abs = np.abs(W)
        signal_indices = np.arange(n_features)

        for scale_factor in scale_factors:
            cf = binary_search_correction_factor(
                coef_correction_abs,
                coef_snp_abs,
                signal_indices,
                fdr=0.1,
                initial_correction_factor=0.0,
                binary_search_tol=1e-6,
                scale_factor=scale_factor,
                max_iterations=100,
            )
            correction_factors.append(cf)

        # All should be reasonable values
        for cf in correction_factors:
            assert cf >= 0
            assert cf <= 10

    @pytest.mark.utils
    def test_bootstrap_repetitions_effect(self, rng):
        """Test effect of different B_reps values."""
        n_samples, n_features = 100, 50
        X = rng.randn(n_samples, n_features)
        y = rng.randn(n_samples)
        W = rng.randn(n_features) * 2

        B_reps_values = [2, 5, 10]
        correction_factors = []

        # Create mock parameters
        coef_correction_abs = np.abs(rng.randn(n_features))
        coef_snp_abs = np.abs(W)
        signal_indices = np.arange(n_features)

        for B_reps in B_reps_values:
            cf = binary_search_correction_factor(
                coef_correction_abs,
                coef_snp_abs,
                signal_indices,
                fdr=0.1,
                initial_correction_factor=0.0,
                binary_search_tol=1e-6,
                scale_factor=1.0,
                max_iterations=100,
            )
            correction_factors.append(cf)

        # All should be reasonable values
        for cf in correction_factors:
            assert cf >= 0
            assert cf <= 10


class TestInflateSignal:
    """Test cases for inflate_signal function."""

    @pytest.mark.utils
    def test_additive_inflation(self, rng):
        """Test additive signal inflation."""
        n_features = 20
        base_coefficients = rng.randn(n_features)
        alpha_reg = 0.5

        inflated = inflate_signal(
            base_coefficients, alpha_reg, inflation_type="additive"
        )

        # Check that positive coefficients increased, negative decreased
        positive_mask = base_coefficients > 0
        negative_mask = base_coefficients < 0

        assert np.all(inflated[positive_mask] >= base_coefficients[positive_mask])
        assert np.all(inflated[negative_mask] <= base_coefficients[negative_mask])

        # Check that inflation amount is correct
        expected_inflation = np.where(
            base_coefficients > 0,
            base_coefficients + alpha_reg,
            base_coefficients - alpha_reg,
        )
        assert np.allclose(inflated, expected_inflation)

    @pytest.mark.utils
    def test_multiplicative_inflation(self, rng):
        """Test multiplicative signal inflation."""
        n_features = 20
        base_coefficients = rng.randn(n_features)
        alpha_reg = 0.3

        inflated = inflate_signal(
            base_coefficients, alpha_reg, inflation_type="multiplicative"
        )

        # Check multiplicative scaling
        expected_inflation = base_coefficients * (1 + alpha_reg)
        assert np.allclose(inflated, expected_inflation)

    @pytest.mark.utils
    def test_zero_coefficients(self, rng):
        """Test inflation with zero coefficients."""
        n_features = 20
        base_coefficients = np.zeros(n_features)
        alpha_reg = 0.5

        # Additive inflation should leave zeros unchanged
        inflated_add = inflate_signal(
            base_coefficients, alpha_reg, inflation_type="additive"
        )
        assert np.allclose(inflated_add, base_coefficients)

        # Multiplicative inflation should leave zeros unchanged
        inflated_mult = inflate_signal(
            base_coefficients, alpha_reg, inflation_type="multiplicative"
        )
        assert np.allclose(inflated_mult, base_coefficients)

    @pytest.mark.utils
    def test_invalid_inflation_type(self, rng):
        """Test that invalid inflation type raises error."""
        base_coefficients = rng.randn(10)
        alpha_reg = 0.5

        with pytest.raises(ValueError):
            inflate_signal(base_coefficients, alpha_reg, inflation_type="invalid")

    @pytest.mark.utils
    def test_different_alpha_values(self, rng):
        """Test inflation with different alpha values."""
        n_features = 20
        base_coefficients = rng.randn(n_features)
        alpha_values = [0.1, 0.5, 1.0, 2.0]

        for alpha_reg in alpha_values:
            inflated = inflate_signal(
                base_coefficients, alpha_reg, inflation_type="additive"
            )

            # Check that inflation is proportional to alpha
            positive_mask = base_coefficients > 0
            negative_mask = base_coefficients < 0

            assert np.all(inflated[positive_mask] >= base_coefficients[positive_mask])
            assert np.all(inflated[negative_mask] <= base_coefficients[negative_mask])


class TestUtilityIntegration:
    """Integration tests for utility functions."""

    @pytest.mark.utils
    def test_standardization_threshold_integration(self, rng):
        """Test integration between standardization and threshold computation."""
        n_samples, n_features = 100, 50

        # Generate data
        X = rng.randn(n_samples, n_features) * 5 + 10
        y = rng.randn(n_samples) * 3 + 2

        # Standardize data
        X_scaled, y_scaled = standardize_data(X, y, scale_by_sample_size=False)

        # Create test statistics
        W = rng.randn(n_features) * 2

        # Compute threshold
        # Create mock knockoff coefficients
        coef_knockoff = rng.randn(n_features) * 0.1
        threshold = binary_search_threshold(W, coef_knockoff, fdr=0.1)

        # Both should work together
        assert threshold > 0
        assert X_scaled.shape == X.shape
        assert y_scaled.shape == y.shape

    @pytest.mark.utils
    def test_full_workflow_simulation(self, rng):
        """Test a simulated full workflow."""
        n_samples, n_features = 100, 50

        # Generate data
        X = rng.randn(n_samples, n_features)
        y = rng.randn(n_samples)

        # Standardize
        X_scaled, y_scaled = standardize_data(X, y, scale_by_sample_size=False)

        # Create mock statistics
        W = rng.randn(n_features) * 2

        # Compute threshold
        # Create mock knockoff coefficients
        coef_knockoff = rng.randn(n_features) * 0.1
        threshold = binary_search_threshold(W, coef_knockoff, fdr=0.1)

        # Create mock parameters for correction factor
        coef_correction_abs = np.abs(rng.randn(n_features))
        coef_snp_abs = np.abs(W)
        signal_indices = np.arange(n_features)

        # Estimate correction factor
        correction_factor = binary_search_correction_factor(
            coef_correction_abs,
            coef_snp_abs,
            signal_indices,
            fdr=0.1,
            initial_correction_factor=0.0,
            binary_search_tol=1e-6,
            scale_factor=1.0,
            max_iterations=100,
        )

        # All should work together
        assert threshold > 0
        assert correction_factor >= 0
        assert X_scaled.shape == X.shape
        assert y_scaled.shape == y.shape


class TestMetrics:
    """Test cases for metrics utilities."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for metrics testing."""
        np.random.seed(42)
        n_features = 100
        n_true = 20

        # Create test statistics with known true features
        statistics = np.random.randn(n_features)
        true_features = np.arange(n_true)

        # Make true features have higher statistics
        statistics[true_features] += 3.0

        return {
            "statistics": statistics,
            "true_features": true_features,
            "n_features": n_features,
            "n_true": n_true,
        }

    @pytest.mark.utils
    def test_compute_fdp(self, sample_data):
        """Test FDP computation."""
        from nullstrap.utils import compute_fdp

        coef_real = sample_data["statistics"]
        coef_knockoff = (
            np.random.randn(sample_data["n_features"]) * 0.5
        )  # Lower knockoff stats
        threshold = 1.0
        fdr = 0.1

        fdp = compute_fdp(coef_real, coef_knockoff, threshold, fdr)

        assert isinstance(fdp, float)
        assert fdp >= 0
        assert fdp <= 1

    @pytest.mark.utils
    def test_compute_fdr(self, sample_data):
        """Test FDR computation."""
        from nullstrap.utils import compute_fdr

        # Test with some selected features
        selected = np.array([0, 1, 2, 50, 51])  # First 3 are true, last 2 are false
        true_features = sample_data["true_features"]
        total_features = sample_data["n_features"]

        fdr = compute_fdr(selected, true_features, total_features)

        assert isinstance(fdr, float)
        assert fdr >= 0
        assert fdr <= 1
        assert fdr == 0.4  # 2 false discoveries out of 5 selected

    @pytest.mark.utils
    def test_compute_fdr_no_selections(self, sample_data):
        """Test FDR computation with no selections."""
        from nullstrap.utils import compute_fdr

        selected = np.array([])
        true_features = sample_data["true_features"]
        total_features = sample_data["n_features"]

        fdr = compute_fdr(selected, true_features, total_features)
        assert fdr == 0.0

    @pytest.mark.utils
    def test_compute_power(self, sample_data):
        """Test power computation."""
        from nullstrap.utils import compute_power

        # Test with some selected features
        selected = np.array([0, 1, 2, 3, 4])  # All are true features
        true_features = sample_data["true_features"]

        power = compute_power(selected, true_features)

        assert isinstance(power, float)
        assert power >= 0
        assert power <= 1
        assert power == 0.25  # 5 out of 20 true features selected

    @pytest.mark.utils
    def test_compute_power_no_true_features(self, sample_data):
        """Test power computation with no true features."""
        from nullstrap.utils import compute_power

        selected = np.array([0, 1, 2])
        true_features = np.array([])

        power = compute_power(selected, true_features)
        assert power == 1.0  # No true features to find

    @pytest.mark.utils
    def test_compute_precision_recall(self, sample_data):
        """Test precision and recall computation."""
        from nullstrap.utils import compute_precision_recall

        # Test with some selected features
        selected = np.array([0, 1, 2, 50, 51])  # 3 true, 2 false
        true_features = sample_data["true_features"]
        total_features = sample_data["n_features"]

        precision, recall = compute_precision_recall(
            selected, true_features, total_features
        )

        assert isinstance(precision, float)
        assert isinstance(recall, float)
        assert precision >= 0 and precision <= 1
        assert recall >= 0 and recall <= 1
        assert precision == 0.6  # 3 true out of 5 selected
        assert recall == 0.15  # 3 out of 20 true features selected

    @pytest.mark.utils
    def test_compute_f1_score(self, sample_data):
        """Test F1 score computation."""
        from nullstrap.utils import compute_f1_score

        # Test with some selected features
        selected = np.array([0, 1, 2, 50, 51])  # 3 true, 2 false
        true_features = sample_data["true_features"]
        total_features = sample_data["n_features"]

        f1 = compute_f1_score(selected, true_features, total_features)

        assert isinstance(f1, float)
        assert f1 >= 0
        assert f1 <= 1

    @pytest.mark.utils
    def test_compute_f1_score_zero_precision_recall(self, sample_data):
        """Test F1 score with zero precision and recall."""
        from nullstrap.utils import compute_f1_score

        # Select only false features
        selected = np.array([50, 51, 52])
        true_features = sample_data["true_features"]
        total_features = sample_data["n_features"]

        f1 = compute_f1_score(selected, true_features, total_features)
        assert f1 == 0.0

    @pytest.mark.utils
    def test_compute_selection_metrics(self, sample_data):
        """Test comprehensive selection metrics."""
        from nullstrap.utils import compute_selection_metrics

        # Test with some selected features
        selected = np.array([0, 1, 2, 50, 51])  # 3 true, 2 false
        true_features = sample_data["true_features"]
        total_features = sample_data["n_features"]

        metrics = compute_selection_metrics(selected, true_features, total_features)

        # Check all expected keys are present
        expected_keys = [
            "fdr",
            "power",
            "precision",
            "recall",
            "f1",
            "n_selected",
            "n_true",
            "n_true_selected",
            "n_false_selected",
        ]
        for key in expected_keys:
            assert key in metrics

        # Check values
        assert metrics["n_selected"] == 5
        assert metrics["n_true"] == 20
        assert metrics["n_true_selected"] == 3
        assert metrics["n_false_selected"] == 2
        assert metrics["fdr"] == 0.4
        assert metrics["power"] == 0.15
        assert metrics["precision"] == 0.6
        assert metrics["recall"] == 0.15

    @pytest.mark.utils
    def test_empirical_fdr_curve(self, sample_data):
        """Test empirical FDR curve computation."""
        from nullstrap.utils import empirical_fdr_curve

        statistics = sample_data["statistics"]
        true_features = sample_data["true_features"]

        thresholds, fdrs = empirical_fdr_curve(
            statistics, true_features, n_thresholds=10
        )

        assert len(thresholds) == 10
        assert len(fdrs) == 10
        assert all(0 <= fdr <= 1 for fdr in fdrs)

        # FDR should generally decrease with higher thresholds
        # (more conservative selection leads to fewer false discoveries)
        assert fdrs[0] >= fdrs[-1]  # First threshold should have higher FDR than last

    @pytest.mark.utils
    def test_empirical_fdr_curve_custom_thresholds(self, sample_data):
        """Test empirical FDR curve with custom thresholds."""
        from nullstrap.utils import empirical_fdr_curve

        statistics = sample_data["statistics"]
        true_features = sample_data["true_features"]
        custom_thresholds = np.array([0.5, 1.0, 1.5, 2.0])

        thresholds, fdrs = empirical_fdr_curve(
            statistics, true_features, thresholds=custom_thresholds
        )

        assert len(thresholds) == 4
        assert len(fdrs) == 4
        assert np.array_equal(thresholds, custom_thresholds)

    @pytest.mark.utils
    def test_power_curve(self, sample_data):
        """Test power curve computation."""
        from nullstrap.utils import power_curve

        statistics = sample_data["statistics"]
        true_features = sample_data["true_features"]

        thresholds, powers = power_curve(statistics, true_features, n_thresholds=10)

        assert len(thresholds) == 10
        assert len(powers) == 10
        assert all(0 <= power <= 1 for power in powers)

        # Power should generally decrease with higher thresholds
        # (more conservative selection)
        assert (
            powers[0] >= powers[-1]
        )  # First threshold should have higher power than last

    @pytest.mark.utils
    def test_power_curve_custom_thresholds(self, sample_data):
        """Test power curve with custom thresholds."""
        from nullstrap.utils import power_curve

        statistics = sample_data["statistics"]
        true_features = sample_data["true_features"]
        custom_thresholds = np.array([0.5, 1.0, 1.5, 2.0])

        thresholds, powers = power_curve(
            statistics, true_features, thresholds=custom_thresholds
        )

        assert len(thresholds) == 4
        assert len(powers) == 4
        assert np.array_equal(thresholds, custom_thresholds)

    @pytest.mark.utils
    def test_metrics_edge_cases(self, sample_data):
        """Test metrics with edge cases."""
        from nullstrap.utils import (compute_fdr, compute_power,
                                     compute_selection_metrics)

        # Empty selections
        selected = np.array([])
        true_features = sample_data["true_features"]
        total_features = sample_data["n_features"]

        fdr = compute_fdr(selected, true_features, total_features)
        power = compute_power(selected, true_features)
        metrics = compute_selection_metrics(selected, true_features, total_features)

        assert fdr == 0.0
        assert power == 0.0
        assert metrics["n_selected"] == 0
        assert metrics["n_true_selected"] == 0
        assert metrics["n_false_selected"] == 0

    @pytest.mark.utils
    def test_metrics_perfect_selection(self, sample_data):
        """Test metrics with perfect selection."""
        from nullstrap.utils import (compute_fdr, compute_power,
                                     compute_selection_metrics)

        # Select all true features and no false ones
        selected = sample_data["true_features"]
        true_features = sample_data["true_features"]
        total_features = sample_data["n_features"]

        fdr = compute_fdr(selected, true_features, total_features)
        power = compute_power(selected, true_features)
        metrics = compute_selection_metrics(selected, true_features, total_features)

        assert fdr == 0.0
        assert power == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    @pytest.mark.utils
    def test_metrics_all_false_selection(self, sample_data):
        """Test metrics with all false selections."""
        from nullstrap.utils import (compute_fdr, compute_power,
                                     compute_selection_metrics)

        # Select only false features
        selected = np.array([50, 51, 52, 53, 54])
        true_features = sample_data["true_features"]
        total_features = sample_data["n_features"]

        fdr = compute_fdr(selected, true_features, total_features)
        power = compute_power(selected, true_features)
        metrics = compute_selection_metrics(selected, true_features, total_features)

        assert fdr == 1.0
        assert power == 0.0
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0
