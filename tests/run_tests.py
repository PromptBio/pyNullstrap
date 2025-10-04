#!/usr/bin/env python3
"""
Test runner script for pyNullstrap.

This script provides convenient commands for running different types of tests.
"""

import argparse
import subprocess
import sys


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        print("Make sure pytest is installed: pip install pytest")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run pyNullstrap tests")
    parser.add_argument(
        "test_type",
        choices=["all", "unit", "integration", "slow", "coverage", "quick"],
        help="Type of tests to run",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--install-deps", action="store_true", help="Install test dependencies first"
    )
    parser.add_argument(
        "--workers",
        "-n",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto)",
    )

    args = parser.parse_args()

    # Install dependencies if requested
    if args.install_deps:
        print("Installing test dependencies...")
        install_cmd = [sys.executable, "-m", "pip", "install", "-e", ".[test]"]
        if not run_command(install_cmd, "Installing test dependencies"):
            return 1

    # Base pytest command
    base_cmd = [sys.executable, "-m", "pytest"]

    if args.verbose:
        base_cmd.append("-v")

    # Add parallel execution
    if args.workers is not None:
        if args.workers == 0:
            base_cmd.append("-n 0")  # Sequential execution
        else:
            base_cmd.extend(["-n", str(args.workers)])
    else:
        # Use auto for most test types, sequential for slow/coverage
        if args.test_type not in ["slow", "coverage"]:
            base_cmd.append("-n auto")

    # Test type specific commands
    if args.test_type == "all":
        cmd = base_cmd + ["tests/"]
        description = "All tests"
    elif args.test_type == "unit":
        cmd = base_cmd + ["tests/", "-m", "unit"]
        description = "Unit tests"
    elif args.test_type == "integration":
        cmd = base_cmd + ["tests/", "-m", "integration"]
        description = "Integration tests"
    elif args.test_type == "slow":
        cmd = base_cmd + ["tests/", "-m", "slow"]
        description = "Slow tests"
    elif args.test_type == "coverage":
        cmd = base_cmd + [
            "tests/",
            "--cov=nullstrap",
            "--cov-report=html",
            "--cov-report=term-missing",
        ]
        description = "Tests with coverage"
    elif args.test_type == "quick":
        cmd = base_cmd + ["tests/", "-m", "not slow"]
        description = "Quick tests (excluding slow tests)"

    # Run the tests
    success = run_command(cmd, description)

    if success:
        print(f"\n🎉 {description} completed successfully!")
        if args.test_type == "coverage":
            print("📊 Coverage report generated in htmlcov/index.html")
        return 0
    else:
        print(f"\n💥 {description} failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
