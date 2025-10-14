"""
Comprehensive tests for all pyNullstrap model classes.

This module imports all model-specific tests from the tests/models/ subdirectory.
The tests have been organized into separate files for better maintainability:

- tests/models/test_lm.py: Tests for NullstrapLM (Linear Models)
- tests/models/test_glm.py: Tests for NullstrapGLM (Generalized Linear Models)
- tests/models/test_cox.py: Tests for NullstrapCox (Cox Survival Models)
- tests/models/test_ggm.py: Tests for NullstrapGGM (Gaussian Graphical Models)

Cross-model tests are in separate files:
- tests/test_integration.py: Cross-model consistency and integration tests

For backward compatibility, we import all test classes here so that running
`pytest tests/test_models.py` will still run all model tests.
"""

# Import all test classes for backward compatibility
from tests.models.test_lm import TestNullstrapLM
from tests.models.test_glm import TestNullstrapGLM
from tests.models.test_cox import TestNullstrapCox
from tests.models.test_ggm import TestNullstrapGGM

# Make test classes available
__all__ = [
    "TestNullstrapLM",
    "TestNullstrapGLM",
    "TestNullstrapCox",
    "TestNullstrapGGM",
]
