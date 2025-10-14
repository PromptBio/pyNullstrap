"""
pyNullstrap: A Python implementation of the Nullstrap method for variable selection with FDR control.

This package provides a flexible, model-agnostic framework for variable selection
with false discovery rate (FDR) control across multiple statistical models.
"""

from .estimator import BaseNullstrap
from .models.cox import NullstrapCox
from .models.ggm import NullstrapGGM
from .models.glm import NullstrapGLM
from .models.lm import NullstrapLM

__version__ = "0.1.0"
__author__ = "Wenbin Guo"
__email__ = "wbguo@ucla.edu"

__all__ = [
    "BaseNullstrap",
    "NullstrapLM",
    "NullstrapGLM",
    "NullstrapCox",
    "NullstrapGGM",
]
