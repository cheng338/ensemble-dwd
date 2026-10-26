"""Pytest setup shared by pytest functions and existing unittest TestCases."""
from path_support import extended_temporary_environment


def pytest_configure(config):
    # Run before test fixtures or module collection can select a temporary root.
    environment = extended_temporary_environment()
    environment.__enter__()
    config.add_cleanup(lambda: environment.__exit__(None, None, None))
