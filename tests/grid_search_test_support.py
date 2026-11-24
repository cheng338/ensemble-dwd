"""Resource scope for GridSearchCV fixtures that exercise sklearn's own refit."""

from joblib import parallel_config
from threadpoolctl import threadpool_limits


def fit_with_serial_refit(search, X, y):
    # These fixtures specify the CV job count on the search and use one worker
    # per estimator. The parent also fits the winner and records refit_time_.
    # Keep the caller's CV backend and its worker-native limits in effect.
    with parallel_config(n_jobs=1), threadpool_limits(limits=1):
        return search.fit(X, y)
