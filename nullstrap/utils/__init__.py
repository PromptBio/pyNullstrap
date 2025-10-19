"""
Utility functions for Nullstrap procedures.

This module contains shared utilities used across different Nullstrap estimators,
including threshold computation, correction factor estimation, data simulation and evaluation metrics.
"""

from .core import (binary_search_correction_factor, binary_search_threshold,
                   inflate_signal, standardize_data)
from .metrics import (compute_f1_score, compute_fdp, compute_fdr,
                      compute_power, compute_precision_recall,
                      compute_selection_metrics, empirical_fdr_curve,
                      power_curve)

__all__ = [
    # Core utilities
    "binary_search_threshold",
    "binary_search_correction_factor",
    "standardize_data",
    # Simulation utilities
    "inflate_signal",
    # Metrics utilities
    "compute_fdp",
    "compute_fdr",
    "compute_power",
    "compute_precision_recall",
    "compute_f1_score",
    "compute_selection_metrics",
    "empirical_fdr_curve",
    "power_curve",
]
