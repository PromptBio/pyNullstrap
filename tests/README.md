# pyNullstrap Testing Guide

Comprehensive testing framework for the pyNullstrap package with unit tests, integration tests, and interactive Jupyter notebooks.

## 📁 Test Structure

```
tests/
├── README.md              # This file
├── __init__.py            # Test package initialization
├── conftest.py            # Pytest fixtures and configuration
├── notebooks/             # Interactive test notebooks
│   ├── test_cox.ipynb     # Cox model testing
│   ├── test_glm.ipynb     # GLM model testing
│   └── test_lm.ipynb      # Linear model testing
├── test_integration.py    # End-to-end workflow tests
├── test_models.py         # Model tests (LM, GLM, Cox, GGM)
└── test_utils.py          # Utility function tests
```

## 🧪 Test Types

| Type | Files | Purpose | Markers |
|------|-------|---------|---------|
| **Unit** | `test_models.py`, `test_utils.py` | Individual components | `@pytest.mark.unit` |
| **Integration** | `test_integration.py` | End-to-end workflows | `@pytest.mark.integration` |
| **Performance** | Various | Large datasets, memory efficiency | `@pytest.mark.slow` |
| **Notebooks** | `notebooks/*.ipynb` | Interactive testing & demos | Manual execution |

## 🚀 Running Tests

### Quick Start

```bash
# Install and run tests
pip install -e ".[test]"
pytest

# Recommended: Use make commands
make test              # All tests (excluding slow)
make test-unit         # Unit tests only
make test-integration  # Integration tests only
make test-coverage     # With coverage report

# Alternative: Test runner script
python tests/run_tests.py quick --install-deps  # Quick tests with auto-install
python tests/run_tests.py coverage              # Tests with coverage
python tests/run_tests.py all                  # All tests including slow
```

> **💡 Quick Reference**: For a standalone testing cheat sheet, see [TEST.md](../TEST.md) in the project root.

### Test Categories

```bash
# Run specific test types
pytest -m "unit"                    # Unit tests only
pytest -m "integration"             # Integration tests only
pytest -m "not slow"                # Exclude slow tests
pytest -m "slow"                    # Performance tests only

# Run specific test files
pytest tests/test_models.py         # Model tests only
pytest tests/test_utils.py          # Utility tests only
pytest tests/test_integration.py   # Integration tests only
```

### Coverage & Performance

```bash
# Coverage reports
pytest --cov=nullstrap --cov-report=html    # HTML report
pytest --cov=nullstrap --cov-report=term    # Terminal report

# Parallel execution
pytest -n auto                    # Auto-detect CPU cores
pytest -n 4                      # Use 4 workers
pytest -n 0                      # Sequential execution
```

## 🔧 Test Fixtures & Utilities

### Data Fixtures (`conftest.py`)
- `linear_data`: Linear regression data with known signal
- `classification_data`: Binary classification data
- `survival_data`: Survival data with time-to-event outcomes
- `graphical_data`: Data from sparse precision matrix
- `high_dimensional_data`: High-dimensional data (p > n)
- `edge_case_data`: Edge cases (single feature, constant features)

### Helper Functions
- `assert_model_fitted()`: Verify model has required attributes
- `assert_fdr_control()`: Verify FDR is controlled within tolerance
- `assert_reproducible_results()`: Verify deterministic results

### Test Markers
- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.slow`: Performance tests

## 🔄 Continuous Integration

GitHub Actions workflow (`.github/workflows/test.yml`):
- **Platforms**: Ubuntu, Windows, macOS
- **Python versions**: 3.9, 3.10, 3.11, 3.12
- **Checks**: Linting (flake8, black, isort, mypy)
- **Coverage**: Upload to Codecov
- **Dependencies**: Tests optional dependencies

## ✍️ Writing New Tests

### Naming Convention
- **Files**: `test_*.py`
- **Classes**: `Test*`
- **Functions**: `test_*`

### Example Structure
```python
import pytest
from nullstrap.models.lm import NullstrapLM

class TestNewFeature:
    @pytest.mark.unit
    def test_basic_functionality(self, linear_data, model_params):
        """Test basic functionality."""
        model = NullstrapLM(**model_params)
        model.fit(linear_data['X'], linear_data['y'])
        assert_model_fitted(model)
```

### Best Practices
- **Use fixtures** for consistent test data
- **Test edge cases** and boundary conditions
- **Test error handling** and exceptions
- **Ensure reproducibility** with fixed random seeds
- **Use appropriate markers** (`@pytest.mark.unit`, etc.)
- **Write descriptive docstrings**

## 🐛 Debugging & Troubleshooting

### Running Specific Tests
```bash
pytest tests/test_models.py                    # Specific file
pytest tests/test_models.py::TestNullstrapLM   # Specific class
pytest -k "test_basic"                         # Pattern matching
pytest -v                                      # Verbose output
pytest --pdb                                   # Debugger on failure
```

### Common Issues
- **Import errors**: Ensure `pip install -e .` (development mode)
- **Missing dependencies**: Install optional dependencies for specific tests
- **Random failures**: Some tests may occasionally fail due to randomness
- **Slow tests**: Use `-m "not slow"` to skip during development

### Performance Notes
- **Parallel execution**: Default with `pytest-xdist` (`-n auto`)
- **Coverage reports**: Use `-n 0` for accurate measurement
- **Memory usage**: Tests use moderate dataset sizes

## 🤝 Contributing

When adding new features:
1. **Add unit tests** for new functionality
2. **Add integration tests** for multi-component features
3. **Add performance tests** for computational features
4. **Add notebook tests** for interactive testing
5. **Update fixtures** if new test data is needed

### Getting Help
- Check the main project [README.md](../README.md) for installation
- Review [pytest documentation](https://docs.pytest.org/) for advanced usage
- Open an issue for persistent test failures
