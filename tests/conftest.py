"""Pytest configuration and shared fixtures."""

import os
import sys
import pytest

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    os.environ["GATEWAY_API_KEYS"] = ""
    os.environ["CACHE_ENABLED"] = "false"
    os.environ["RATE_LIMIT_ENABLED"] = "false"
    os.environ["REQUEST_LOGGING_ENABLED"] = "false"
    os.environ["METRICS_ENABLED"] = "false"
    os.environ["LOG_LEVEL"] = "WARNING"
    yield

