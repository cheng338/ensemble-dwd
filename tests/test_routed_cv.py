"""Check scikit-learn's routed CV defaults against manually fitted folds."""
import numpy as np
from sklearn import config_context
from sklearn.base import clone
from sklearn.model_selection import (
    StratifiedKFold, cross_val_predict, cross_val_score, cross_validate,
)

from ensemble_dwd import EnsembleKernelDWD


def test_routed_cv_without_params_matches_manual_folds():
    X = np.array([[-3.], [3.], [-2.5], [2.5], [-2.], [2.],
                  [-1.5], [1.5], [-1.], [1.], [-0.5], [0.5]])
    y = np.where(X[:, 0] < 0, "low", "high")
    model = EnsembleKernelDWD(
        M=1, theta=1.0, sv_selection_ratio=1.0, kernel="linear",
        lambd=0.1, stopping="fixed", max_iter=8, random_state=0,
    )
    folds = list(StratifiedKFold(3, shuffle=True, random_state=27).split(X, y))
    predictions = np.empty_like(y)
    accuracies = []
    for train, test in folds:
        fitted = clone(model).fit(X[train], y[train])
        predictions[test] = fitted.predict(X[test])
        accuracies.append(np.mean(predictions[test] == y[test]))
    np.testing.assert_array_equal(predictions, y)

    with config_context(enable_metadata_routing=True):
        options = dict(cv=folds, n_jobs=1, scoring="accuracy", error_score="raise")
        np.testing.assert_array_equal(cross_val_score(model, X, y, **options), accuracies)
        results = cross_validate(model, X, y, **options)
        np.testing.assert_array_equal(results["test_score"], accuracies)
        out_of_fold = cross_val_predict(model, X, y, cv=folds, n_jobs=1)
        np.testing.assert_array_equal(out_of_fold, predictions)
