"""Fit profiling for checks that need counts or stage arithmetic."""

from contextlib import contextmanager
from itertools import count
from unittest.mock import patch

from ensemble_dwd import profiling


@contextmanager
def synthetic_fit_profile():
    """Use synthetic ticks in both the parent and its managed worker profiles."""
    with patch.object(profiling, "perf_counter", count().__next__), profiling.fit_profile() as profile:
        yield profile
