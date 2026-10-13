"""Test package initialization."""

import ask_the_database


def test_package_version() -> None:
    """Verify package version is defined."""
    assert ask_the_database.__version__ == "0.1.0"
