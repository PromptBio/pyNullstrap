# Makefile for pyNullstrap development
# Python 3.10+ required

.PHONY: help install install-dev install-all install-survival test test-quick test-unit test-integration test-models test-utils test-slow test-all test-coverage test-fast test-failfast test-lf lint format format-check type-check clean clean-all check ci docs

help:  ## Show this help message
	@echo "========================================="
	@echo "pyNullstrap Development Commands"
	@echo "========================================="
	@echo ""
	@echo "Installation:"
	@grep -E '^install.*:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Testing:"
	@grep -E '^test.*:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Code Quality:"
	@grep -E '^(lint|format|type-check|check|ci):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Maintenance:"
	@grep -E '^(clean|docs):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

install:  ## Install package in development mode
	pip install -e .

install-dev:  ## Install package with all development dependencies
	pip install -e ".[test,dev,docs]"

install-all:  ## Install package with all optional dependencies
	pip install -e ".[all]"

install-survival:  ## Install survival analysis dependencies (scikit-survival)
	pip install -e ".[survival]"

# ============================================================
# Testing Targets
# ============================================================

test:  ## Run quick tests (recommended for development)
	python tests/run_tests.py quick

test-quick:  ## Alias for 'test' - run quick tests excluding slow tests
	python tests/run_tests.py quick

test-all:  ## Run all tests including slow tests
	python tests/run_tests.py all

test-unit:  ## Run unit tests only
	python tests/run_tests.py unit

test-integration:  ## Run integration tests only
	python tests/run_tests.py integration

test-models:  ## Run all model tests (LM, GLM, Cox, GGM)
	python tests/run_tests.py models

test-utils:  ## Run utility function tests only
	python tests/run_tests.py utils

test-slow:  ## Run slow/performance tests only
	python tests/run_tests.py slow

test-coverage:  ## Run tests with coverage report (HTML + terminal)
	python tests/run_tests.py coverage

test-fast:  ## Run only fast tests (unit, excluding integration and slow)
	pytest tests/ -m "not slow and not integration" -v

test-failfast:  ## Run tests and stop on first failure
	python tests/run_tests.py quick -x

test-lf:  ## Run only tests that failed in the last run
	python tests/run_tests.py quick --lf

test-verbose:  ## Run tests with verbose output
	python tests/run_tests.py quick --verbose

# ============================================================
# Code Quality Targets
# ============================================================

lint:  ## Run linting checks (flake8)
	@echo "Running flake8 linting..."
	@flake8 nullstrap/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics || true
	@flake8 nullstrap/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:  ## Format code with black and isort
	@echo "Formatting code with black..."
	@black nullstrap/ tests/
	@echo "Sorting imports with isort..."
	@isort nullstrap/ tests/

format-check:  ## Check code formatting without making changes
	@echo "Checking code format..."
	@black --check nullstrap/ tests/
	@isort --check-only nullstrap/ tests/

type-check:  ## Run type checking with mypy
	@echo "Running mypy type checking..."
	@mypy nullstrap/ --ignore-missing-imports || true

check: format-check lint type-check test  ## Run all quality checks (format, lint, type, test)

ci:  ## Run CI-style checks locally (for pre-commit validation)
	@echo "Running CI checks..."
	@python tests/run_tests.py quick --verbose
	@python tests/run_tests.py coverage

# ============================================================
# Maintenance Targets
# ============================================================

clean:  ## Clean up build artifacts and cache files
	@echo "Cleaning build artifacts..."
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info/
	@rm -rf .pytest_cache/
	@rm -rf .coverage
	@rm -rf htmlcov/
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Clean complete!"

clean-all: clean  ## Clean everything including docs
	@echo "Cleaning documentation..."
	@rm -rf docs/_build/ 2>/dev/null || true
	@echo "All clean!"

docs:  ## Build documentation (if docs/ exists)
	@if [ -d "docs" ]; then \
		echo "Building documentation..."; \
		cd docs && make html; \
	else \
		echo "Documentation directory not found (docs/ in .gitignore)"; \
	fi

# ============================================================
# Utility Targets
# ============================================================

coverage-report:  ## Open coverage report in browser
	@python -m webbrowser htmlcov/index.html 2>/dev/null || open htmlcov/index.html || xdg-open htmlcov/index.html

watch-tests:  ## Watch for changes and run tests automatically (requires pytest-watch)
	@command -v ptw >/dev/null 2>&1 && ptw tests/ -- -m "not slow" || \
		echo "pytest-watch not installed. Install with: pip install pytest-watch"

list-tests:  ## List all available tests
	@pytest --collect-only -q tests/

# Default target
.DEFAULT_GOAL := help
