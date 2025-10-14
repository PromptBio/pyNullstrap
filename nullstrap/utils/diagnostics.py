"""
Diagnostic and logging utilities for Nullstrap procedures.

This module provides comprehensive logging, timing, and diagnostic collection
tools for monitoring and debugging Nullstrap estimators throughout the fitting
process. It includes structured logging with timing information, diagnostic
data collection, and progress tracking capabilities.

Classes
-------
NullstrapLogger
    Custom logger with structured formatting and key-value pair support.
DiagnosticCollector
    Collects and manages diagnostic information during fitting.

Functions
---------
timer
    Decorator for timing function execution.
log_fitting_progress
    Log the start of fitting process with key parameters.
log_fitting_results
    Log the results of fitting process.
log_convergence_warning
    Log convergence warnings for iterative procedures.
log_parameter_selection
    Log automatic parameter selection events.
get_global_diagnostics
    Get the global diagnostic collector instance.
reset_global_diagnostics
    Reset the global diagnostic collector.
"""

import logging
import time
from functools import wraps
from typing import Any, Dict, Optional

import numpy as np


class NullstrapLogger:
    """
    Custom logger for Nullstrap procedures.

    Provides structured logging with timing information and procedure-specific
    formatting. Supports key-value pair logging for enhanced readability.

    Parameters
    ----------
    name : str, default="nullstrap"
        Logger name identifier.
    level : int, default=logging.INFO
        Logging level threshold.

    Examples
    --------
    >>> logger = NullstrapLogger()
    >>> logger.info("Starting fitting", n_samples=1000, fdr=0.05)
    >>> logger.warning("Convergence issue", iterations=1000)
    """

    def __init__(self, name: str = "nullstrap", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Create formatter with timestamp and structured output
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Create console handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def info(self, message: str, **kwargs):
        """
        Log info message with optional key-value pairs.

        Parameters
        ----------
        message : str
            Main log message.
        **kwargs
            Additional key-value pairs to include in log.
        """
        if kwargs:
            message += " | " + " | ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.info(message)

    def debug(self, message: str, **kwargs):
        """
        Log debug message with optional key-value pairs.

        Parameters
        ----------
        message : str
            Main log message.
        **kwargs
            Additional key-value pairs to include in log.
        """
        if kwargs:
            message += " | " + " | ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.debug(message)

    def warning(self, message: str, **kwargs):
        """
        Log warning message with optional key-value pairs.

        Parameters
        ----------
        message : str
            Main log message.
        **kwargs
            Additional key-value pairs to include in log.
        """
        if kwargs:
            message += " | " + " | ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.warning(message)

    def error(self, message: str, **kwargs):
        """
        Log error message with optional key-value pairs.

        Parameters
        ----------
        message : str
            Main log message.
        **kwargs
            Additional key-value pairs to include in log.
        """
        if kwargs:
            message += " | " + " | ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.error(message)


def timer(func):
    """
    Decorator to time function execution.

    Automatically logs the execution time of decorated functions using
    the NullstrapLogger. Useful for performance monitoring and debugging.

    Parameters
    ----------
    func : callable
        Function to be timed.

    Returns
    -------
    wrapper : callable
        Wrapped function with timing capability.

    Examples
    --------
    >>> @timer
    ... def slow_function():
    ...     time.sleep(1)
    ...     return "done"
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        logger = NullstrapLogger()
        logger.debug(
            f"Function {func.__name__} completed",
            duration=f"{end_time - start_time:.3f}s",
        )

        return result

    return wrapper


class DiagnosticCollector:
    """
    Collect diagnostic information during Nullstrap fitting.

    This class helps track various metrics and intermediate results during
    the fitting process for debugging and analysis. Supports both value
    recording and timing operations.

    Attributes
    ----------
    diagnostics : dict
        Collected diagnostic values.
    timing : dict
        Timing information for operations.

    Examples
    --------
    >>> collector = DiagnosticCollector()
    >>> collector.record("n_iterations", 150)
    >>> collector.start_timer("correction_search")
    >>> # ... do work ...
    >>> collector.end_timer("correction_search")
    >>> summary = collector.get_summary()
    """

    def __init__(self):
        self.diagnostics = {}
        self.timing = {}

    def record(self, key: str, value: Any):
        """
        Record a diagnostic value.

        Parameters
        ----------
        key : str
            Diagnostic key identifier.
        value : Any
            Value to record (supports any type).
        """
        self.diagnostics[key] = value

    def start_timer(self, key: str):
        """
        Start timing an operation.

        Parameters
        ----------
        key : str
            Timer identifier.
        """
        self.timing[key] = {"start": time.time()}

    def end_timer(self, key: str):
        """
        End timing an operation and compute duration.

        Parameters
        ----------
        key : str
            Timer identifier (must have been started).
        """
        if key in self.timing and "start" in self.timing[key]:
            self.timing[key]["end"] = time.time()
            self.timing[key]["duration"] = (
                self.timing[key]["end"] - self.timing[key]["start"]
            )

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all collected diagnostics.

        Returns
        -------
        summary : dict
            Dictionary containing diagnostics and timing information.
        """
        summary = {"diagnostics": self.diagnostics.copy(), "timing": {}}

        for key, timing_info in self.timing.items():
            if "duration" in timing_info:
                summary["timing"][key] = timing_info["duration"]

        return summary

    def log_summary(self, logger: Optional[NullstrapLogger] = None):
        """
        Log diagnostic summary with structured formatting.

        Parameters
        ----------
        logger : NullstrapLogger, optional
            Logger instance to use. Creates new instance if None.
        """
        if logger is None:
            logger = NullstrapLogger()

        summary = self.get_summary()

        logger.info("=== Nullstrap Diagnostic Summary ===")

        # Log timing information
        if summary["timing"]:
            logger.info("Timing Information:")
            for operation, duration in summary["timing"].items():
                logger.info(f"  {operation}: {duration:.3f}s")

        # Log diagnostic values
        if summary["diagnostics"]:
            logger.info("Diagnostic Values:")
            for key, value in summary["diagnostics"].items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {key}: {value}")
                elif isinstance(value, np.ndarray):
                    logger.info(f"  {key}: array shape={value.shape}")
                else:
                    logger.info(f"  {key}: {type(value).__name__}")


def log_fitting_progress(
    estimator_name: str,
    n_samples: int,
    n_features: int,
    fdr: float,
    alpha_: Optional[float] = None,
    B_reps: Optional[int] = None,
):
    """
    Log the start of fitting process with key parameters.

    Provides structured logging of the initial fitting parameters for
    tracking and debugging purposes.

    Parameters
    ----------
    estimator_name : str
        Name of the estimator being fitted.
    n_samples : int
        Number of samples in the dataset.
    n_features : int
        Number of features in the dataset.
    fdr : float
        Target false discovery rate.
    alpha_ : float, optional
        Regularization parameter value.
    B_reps : int, optional
        Number of bootstrap replications.
    """
    logger = NullstrapLogger()

    logger.info(
        f"Starting {estimator_name} fitting",
        n_samples=n_samples,
        n_features=n_features,
        fdr=fdr,
        alpha_=alpha_ if alpha_ is not None else "auto",
        B_reps=B_reps if B_reps is not None else "default",
    )


def log_fitting_results(
    threshold: float, n_selected: int, correction_factor: float, alpha_used: float
):
    """
    Log the results of fitting process.

    Provides structured logging of the final fitting results for
    monitoring and validation purposes.

    Parameters
    ----------
    threshold : float
        Final selection threshold.
    n_selected : int
        Number of selected features.
    correction_factor : float
        Estimated correction factor.
    alpha_used : float
        Regularization parameter used.
    """
    logger = NullstrapLogger()

    logger.info(
        "Fitting completed",
        threshold=f"{threshold:.6f}",
        n_selected=n_selected,
        correction_factor=f"{correction_factor:.6f}",
        alpha_used=f"{alpha_used:.6f}",
    )


def log_convergence_warning(operation: str, max_iter: int):
    """
    Log convergence warning for iterative procedures.

    Parameters
    ----------
    operation : str
        Name of the operation that may not have converged.
    max_iter : int
        Maximum iterations reached.
    """
    logger = NullstrapLogger()
    logger.warning(f"{operation} may not have converged", max_iterations=max_iter)


def log_parameter_selection(parameter_name: str, selected_value: float, method: str):
    """
    Log automatic parameter selection events.

    Parameters
    ----------
    parameter_name : str
        Name of the parameter being selected.
    selected_value : float
        Selected parameter value.
    method : str
        Method used for parameter selection.
    """
    logger = NullstrapLogger()
    logger.info(
        f"Parameter {parameter_name} selected",
        value=f"{selected_value:.6f}",
        method=method,
    )


# Global diagnostic collector instance
_global_diagnostics = DiagnosticCollector()


def get_global_diagnostics() -> DiagnosticCollector:
    """
    Get the global diagnostic collector instance.

    Returns
    -------
    DiagnosticCollector
        Global diagnostic collector instance.
    """
    return _global_diagnostics


def reset_global_diagnostics():
    """
    Reset the global diagnostic collector.

    Clears all collected diagnostics and timing information from the
    global diagnostic collector instance.
    """
    global _global_diagnostics
    _global_diagnostics = DiagnosticCollector()
