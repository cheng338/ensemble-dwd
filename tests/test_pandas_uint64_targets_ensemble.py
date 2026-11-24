"""Binary and multiclass wrappers retain sparse uint64 class IDs."""
import numpy as np
import pandas as pd
import pytest
from sklearn.svm import SVC

from ensemble_dwd import EnsembleKernelDWD, EnsembleKernelSVM, OvOClassifier


@pytest.mark.parametrize("cls", [EnsembleKernelDWD, EnsembleKernelSVM])
def test_binary_sparse_target_keeps_uint64_extrema(cls):
    ids = [2**64 - 2, 2**64 - 1]
    labels = np.tile(np.array(ids, dtype=np.uint64), 24)
    X = np.tile([[-1.], [1.]], (24, 1))
    sparse = pd.Series(pd.arrays.SparseArray(labels, fill_value=ids[0]))
    model = cls(M=3, random_state=2, sv_selection_ratio=1.).fit(X, sparse)
    assert model.classes_.dtype == np.uint64
    assert model.classes_.tolist() == ids
    assert model.predict(X).tolist() == labels.tolist()
    assert model.score(X, sparse) == 1.


def test_multiclass_sparse_target_pair_votes_and_score_are_exact():
    ids = [0, 2**53, 2**53 + 1]
    labels = np.tile(np.array(ids, dtype=np.uint64), 8)
    X = np.tile(np.eye(3), (8, 1))
    sparse = pd.Series(pd.arrays.SparseArray(labels, fill_value=0))
    model = OvOClassifier(SVC(kernel="linear")).fit(X, sparse)
    assert model.classes_.dtype == np.uint64
    assert model.classes_.tolist() == ids
    assert model.predict(X).tolist() == labels.tolist()
    truth = pd.Series(pd.arrays.SparseArray(np.array(ids, dtype=np.uint64), fill_value=0))
    assert model.score(X[[0, 1, 1]], truth) == pytest.approx(2/3)


@pytest.mark.parametrize("cls", [EnsembleKernelDWD, EnsembleKernelSVM, OvOClassifier])
@pytest.mark.parametrize("storage", ["nullable", "sparse_categories"])
def test_exact_integer_labels_in_binary_and_multiclass_wrappers(cls, storage):
    upper = 2**53 + 1
    ids = [0, upper] if cls is not OvOClassifier else [0, 7, upper]
    codes = np.tile(np.arange(len(ids)), 24)
    labels = np.array(ids, dtype=np.uint64)
    X = np.eye(len(ids))[codes]

    def target(classes, positions):
        if storage == "nullable":
            return pd.Series(classes[positions], dtype="UInt64")
        categories = pd.Index(pd.arrays.SparseArray(classes, fill_value=0))
        return pd.Series(pd.Categorical.from_codes(positions, categories=categories))

    estimator = (OvOClassifier(SVC(kernel="linear")) if cls is OvOClassifier
                 else cls(M=3, random_state=2, sv_selection_ratio=1.))
    model = estimator.fit(X, target(labels, codes))
    assert model.classes_.tolist() == ids
    assert model.predict(X).tolist() == labels[codes].tolist()
    # The query predicts 0 and upper. The adjacent label is deliberately wrong.
    truth = target(np.array([0, upper - 1], dtype=np.uint64), [0, 1])
    assert model.score(X[[0, len(ids) - 1]], truth, sample_weight=[1, 3]) == .25
