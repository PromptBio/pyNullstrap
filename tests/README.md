# pyNullstrap Testing Guide

Comprehensive testing framework for the pyNullstrap package with unit tests, integration tests, and interactive Jupyter notebooks.

## 📁 Test Structure

```
tests/
├── README.md              # This file
├── __init__.py            # Test package initialization
├── conftest.py            # Pytest fixtures and configuration
├── models/                # Model-specific tests
│   ├── test_lm.py         # Linear model tests
│   ├── test_glm.py        # GLM tests
│   ├── test_cox.py        # Cox model tests
│   └── test_ggm.py        # Gaussian Graphical Model tests
├── notebooks/             # Interactive test notebooks
│   ├── test_lm.ipynb      # Linear model testing
│   ├── test_glm.ipynb     # GLM model testing
│   ├── test_cox.ipynb     # Cox model testing
│   └── test_ggm.ipynb     # GGM model testing
├── run_tests.py           # Test runner script
├── test_integration.py    # End-to-end workflow tests
├── test_models.py         # Model test aggregator
└── test_utils.py          # Utility function tests
```

## 🧪 Test Types

| Type            | Files                               | Purpose                           | Markers                    |
| --------------- | ----------------------------------- | --------------------------------- | -------------------------- |
| **Unit**        | `models/test_*.py`, `test_utils.py` | Individual components             | `@pytest.mark.unit`        |
| **Integration** | `test_integration.py`               | End-to-end workflows              | `@pytest.mark.integration` |
| **Performance** | Various                             | Large datasets, memory efficiency | `@pytest.mark.slow`        |
| **Notebooks**   | `notebooks/*.ipynb`                 | Interactive testing & demos       | Manual execution           |

## 🚀 Running Tests

### Quick Start

```bash
# Install and run tests
pip install -e ".[test]"
pytest

# Recommended: Use make commands
make test              # Quick tests (recommended for development)
make test-all          # All tests including slow tests
make test-unit         # Unit tests only
make test-integration  # Integration tests only
make test-models       # All model tests (LM, GLM, Cox, GGM)
make test-utils        # Utility function tests only
make test-coverage     # Tests with coverage report
make help              # Show all available commands
```

### Using the Test Runner Script

The `run_tests.py` script provides a convenient interface for running tests with various options.

**Available Test Types:**
| Command       | Description            | Use Case                    |
| ------------- | ---------------------- | --------------------------- |
| `quick`       | Excludes slow tests    | Recommended for development |
| `all`         | Runs all tests         | Comprehensive testing       |
| `unit`        | Unit tests only        | Test individual components  |
| `integration` | Integration tests only | Test component interactions |
| `models`      | All model tests        | Test LM, GLM, Cox, GGM      |
| `utils`       | Utility tests only     | Test helper functions       |
| `slow`        | Performance tests      | Test with large datasets    |
| `coverage`    | Tests with coverage    | Generate coverage reports   |

**Quick Reference:**
```bash
# Quick tests (recommended for development)
python tests/run_tests.py quick

# Run all tests including slow ones
python tests/run_tests.py all

# Run specific test categories
python tests/run_tests.py unit              # Unit tests only
python tests/run_tests.py integration       # Integration tests only
python tests/run_tests.py models            # All model tests (LM, GLM, Cox, GGM)
python tests/run_tests.py utils             # Utility function tests only
python tests/run_tests.py slow              # Performance tests only

# Run with coverage report
python tests/run_tests.py coverage

# Additional options
python tests/run_tests.py quick --verbose          # Verbose output
python tests/run_tests.py all -n 4                 # Use 4 parallel workers
python tests/run_tests.py unit -n 0                # Sequential execution
python tests/run_tests.py quick --install-deps     # Auto-install dependencies
python tests/run_tests.py unit -x                  # Stop on first failure
python tests/run_tests.py quick --lf               # Run only last failed tests

# Get help
python tests/run_tests.py --help
```

**Command-Line Options:**
| Option           | Short  | Description                                                  |
| ---------------- | ------ | ------------------------------------------------------------ |
| `--verbose`      | `-v`   | Show verbose test output                                     |
| `--workers N`    | `-n N` | Number of parallel workers (0 for sequential, default: auto) |
| `--install-deps` |        | Install test dependencies before running                     |
| `--failfast`     | `-x`   | Stop execution on first test failure                         |
| `--last-failed`  | `--lf` | Run only tests that failed in the last run                   |
| `--help`         | `-h`   | Show help message and exit                                   |

### Test Categories

```bash
# Run specific test types
pytest -m "unit"                    # Unit tests only
pytest -m "integration"             # Integration tests only
pytest -m "not slow"                # Exclude slow tests
pytest -m "slow"                    # Performance tests only

# Run specific test files
pytest tests/test_models.py         # All model tests (aggregator)
pytest tests/models/test_lm.py      # Linear model tests only
pytest tests/models/test_glm.py     # GLM tests only
pytest tests/models/test_cox.py     # Cox model tests only
pytest tests/models/test_ggm.py     # GGM tests only
pytest tests/test_utils.py          # Utility tests only
pytest tests/test_integration.py    # Integration tests only
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
pytest tests/models/test_lm.py                 # Specific file
pytest tests/models/test_lm.py::TestNullstrapLM # Specific class
pytest tests/models/test_lm.py::TestNullstrapLM::test_basic_fitting # Specific test
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
