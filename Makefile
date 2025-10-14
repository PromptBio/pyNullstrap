# Makefile for pyNullstrap development

.PHONY: help install install-dev test test-unit test-integration test-slow test-all test-runner test-runner-all test-runner-coverage test-runner-install lint format type-check clean docs docs-clean docs-clean-all

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package in development mode
	pip install -e .

install-dev:  ## Install package with all development dependencies
	pip install -e ".[test,dev,docs]"

test:  ## Run all tests (excluding slow tests) with parallel execution
	pytest tests/ -m "not slow" -v

test-unit:  ## Run unit tests only with parallel execution
	pytest tests/ -m "unit" -v

test-integration:  ## Run integration tests only with parallel execution
	pytest tests/ -m "integration" -v

test-slow:  ## Run slow tests only (sequential for stability)
	pytest tests/ -m "slow" -v -n 0

test-all:  ## Run all tests including slow tests with parallel execution
	pytest tests/ -v

test-coverage:  ## Run tests with coverage report (sequential for coverage accuracy)
	pytest tests/ -m "not slow" --cov=nullstrap --cov-report=html --cov-report=term-missing -v -n 0

test-fast:  ## Run only fast tests with maximum parallelization
	pytest tests/ -m "not slow and not integration" -v -n auto

test-parallel:  ## Run tests with specific number of workers
	pytest tests/ -m "not slow" -v -n 4

test-runner:  ## Run tests using the test runner script (quick tests)
	python tests/run_tests.py quick

test-runner-all:  ## Run all tests using the test runner script
	python tests/run_tests.py all

test-runner-coverage:  ## Run tests with coverage using the test runner script
	python tests/run_tests.py coverage

test-runner-install:  ## Install dependencies and run tests using the test runner script
	python tests/run_tests.py quick --install-deps

lint:  ## Run linting checks
	flake8 nullstrap/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 nullstrap/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:  ## Format code with black and isort
	black nullstrap/ tests/
	isort nullstrap/ tests/

format-check:  ## Check code formatting
	black --check nullstrap/ tests/
	isort --check-only nullstrap/ tests/

type-check:  ## Run type checking with mypy
	mypy nullstrap/ --ignore-missing-imports

docs:  ## Build documentation
	cd docs && make html

docs-clean:  ## Clean documentation build
	cd docs && make clean

docs-clean-all:  ## Clean all documentation artifacts (docs + root)
	cd docs && make clean
	rm -rf docs/_build/
	rm -rf docs/_static/
	rm -rf docs/_templates/

clean:  ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

check: format-check lint type-check test  ## Run all checks (format, lint, type, test)

ci:  ## Run CI checks locally
	pytest tests/ -m "not slow and not integration" --cov=nullstrap --cov-report=xml --cov-report=term-missing -v
	pytest tests/ -m "integration" --cov=nullstrap --cov-report=xml --cov-report=term-missing -v

install-survival:  ## Install survival analysis dependencies
	pip install scikit-survival

install-graphical:  ## Install graphical model dependencies
	pip install scikit-learn

install-all:  ## Install all optional dependencies
	pip install -e ".[all]"
