"""Shared pytest fixtures for ask-the-database."""

import pytest


@pytest.fixture
def sample_test_fixture() -> bool:
    """A baseline fixture for testing setup."""
    return True
