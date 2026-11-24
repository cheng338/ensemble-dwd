"""Observe complete training-set fits in the adaptive workflow tests."""
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np


@contextmanager
def record_full_pipeline_fits(training_X, training_y):
    import adaptive_search

    models = []
    final_fit_threads = adaptive_search.final_fit_threads

    @contextmanager
    def observe_final_fit(threads, model, receipt, **kwargs):
        original_fit = model.fit

        def observed_fit(X, y, **params):
            np.testing.assert_array_equal(X, training_X)
            np.testing.assert_array_equal(y, training_y)
            models.append(model)
            return original_fit(X, y, **params)

        # Keep Pipeline.fit intact throughout CV so reuse qualification sees
        # the real implementation. Observe calls only on the selected model.
        with final_fit_threads(threads, model, receipt, **kwargs), patch.object(model, 'fit', observed_fit):
            yield

    with patch.object(adaptive_search, 'final_fit_threads', observe_final_fit):
        yield models


def faer_thread_limit(solver):
    """Read the native FAER limit and check the actual count when exposed."""
    settings = solver.get_settings()
    assert settings.direct_solve_method == 'faer'
    if hasattr(solver, 'get_info'):
        observed = solver.get_info().linsolver.threads
        assert observed == settings.max_threads
        return observed
    return settings.max_threads
