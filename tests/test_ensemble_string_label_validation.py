"""Ensemble classifiers retain exact string labels for training and prediction."""
from decimal import Decimal
import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import DataConversionWarning, NotFittedError
from sklearn.utils.validation import check_is_fitted

from dwd.kern_md import KernMD
from ensemble_dwd import EnsembleKernelDWD, EnsembleKernelSVM, OvOClassifier


X = np.array([[-3.], [3.], [-2.], [2.], [-1.], [1.], [-0.5], [0.5]])
LABELS = ["q", "q\0"] * 4
requires_string_dtype = pytest.mark.skipif(
    not hasattr(np.dtypes, "StringDType"), reason="NumPy has no StringDType"
)


@pytest.fixture(params=["ovo", "ensemble-dwd", "ensemble-svm"])
def classifier(request):
    if request.param == "ovo":
        return OvOClassifier(KernMD(), n_jobs=1)
    options = dict(M=1, theta=1.0, sv_selection_ratio=1.0, kernel="linear", random_state=4)
    if request.param == "ensemble-dwd":
        return EnsembleKernelDWD(max_iter=8, stopping="fixed", **options)
    return EnsembleKernelSVM(**options)


@requires_string_dtype
def test_string_classes_survive_fit_and_pickle(classifier):
    y = np.array(LABELS, dtype=np.dtypes.StringDType())
    model = classifier.fit(X, y)
    assert model.classes_.dtype == y.dtype
    assert model.classes_.tolist() == ["q", "q\0"]
    assert model.predict(X).tolist() == LABELS
    assert model.score(X, y) == 1.0

    restored = pickle.loads(pickle.dumps(model))
    assert restored.classes_.dtype == y.dtype
    assert restored.predict(X[::-1]).tolist() == LABELS[::-1]


@requires_string_dtype
def test_ovo_column_target_warns_and_retains_feature_names():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(X, columns=["measurement"])
    y = np.array(LABELS, dtype=np.dtypes.StringDType()).reshape(-1, 1)
    with pytest.warns(DataConversionWarning):
        model = OvOClassifier(KernMD(), n_jobs=1).fit(frame, y)
    assert model.feature_names_in_.tolist() == ["measurement"]
    assert model.predict(frame).tolist() == LABELS
    assert model.classes_.dtype == y.dtype


@requires_string_dtype
def test_svm_single_row_prediction_batches_preserve_trailing_nuls():
    model = EnsembleKernelSVM(M=1, theta=1.0, sv_selection_ratio=1.0, kernel="linear",
                              random_state=4, prediction_batch_size=1)
    model.fit(X, np.array(LABELS, dtype=np.dtypes.StringDType()))
    assert model.predict(X).tolist() == LABELS


@pytest.mark.parametrize("dtype, rows", [
    (object, ["left", "right"] * 4), (str, ["left", "right"] * 4), (np.int64, [1, 2] * 4),
], ids=["object", "unicode", "int64"])
def test_ordinary_label_dtype_is_preserved(classifier, dtype, rows):
    y = np.array(rows, dtype=dtype)
    model = classifier.fit(X, y)
    assert model.classes_.dtype == y.dtype
    assert model.predict(X).tolist() == rows


@requires_string_dtype
@pytest.mark.parametrize("missing", [float("nan"), np.float32("nan"), Decimal("NaN"),
                                     object(), None, "pandas.NA"],
                         ids=["nan", "numpy-nan", "decimal-nan", "object", "none", "pandas-na"])
def test_only_an_observed_string_sentinel_invalidates_fit(classifier, missing):
    if isinstance(missing, str):
        missing = pytest.importorskip("pandas").NA
    dtype = np.dtypes.StringDType(na_object=missing)
    model = classifier.fit(X, np.array(LABELS, dtype=dtype))
    assert model.classes_.dtype == dtype
    assert model.predict(X).tolist() == LABELS
    with pytest.raises(ValueError, match="missing"):
        model.fit(X, np.array([missing, "right"] * 4, dtype=dtype))
    with pytest.raises(NotFittedError):
        check_is_fitted(model)


@requires_string_dtype
def test_string_valued_sentinel_is_a_class_name(classifier):
    dtype = np.dtypes.StringDType(na_object="NA")
    rows = ["NA", "right"] * 4
    model = classifier.fit(X, np.array(rows, dtype=dtype))
    assert model.classes_.dtype == dtype
    assert model.classes_.tolist() == ["NA", "right"]
    assert model.predict(X).tolist() == rows


@requires_string_dtype
def test_embedded_nul_fit_matches_object_control(classifier):
    rows = ["id\0B", "id\0A"] * 4
    y = np.array(rows, dtype=np.dtypes.StringDType(coerce=False, na_object="id\0A"))
    control = clone(classifier).fit(X, y.astype(object))
    model = classifier.fit(X, y)
    assert model.classes_.dtype == y.dtype
    assert model.classes_.tolist() == ["id\0A", "id\0B"]
    predictions = model.predict(X)
    assert predictions.dtype == y.dtype
    assert predictions.tolist() == control.predict(X).tolist() == rows
    assert model.score(X, y[::-1]) == 0.
    if isinstance(model, (EnsembleKernelDWD, EnsembleKernelSVM)):
        np.testing.assert_array_equal(model.selected_indices_, control.selected_indices_)
        np.testing.assert_allclose(model.consensus_scores_, control.consensus_scores_)
        if isinstance(model, EnsembleKernelSVM):
            np.testing.assert_allclose(model.selection_distances_, control.selection_distances_)


@requires_string_dtype
def test_multiclass_pairs_split_on_the_full_label():
    x = np.array([[3.], [-3.], [.1], [4.], [-4.], [-.1]])
    rows = ["id\0C", "id\0A", "id\0B"] * 2
    y = np.array(rows, dtype="T")
    model = OvOClassifier(KernMD()).fit(x, y)
    assert model.classes_.tolist() == ["id\0A", "id\0B", "id\0C"]
    assert model.pairs_ == (("id\0A", "id\0B"), ("id\0A", "id\0C"), ("id\0B", "id\0C"))
    assert [indices.tolist() for indices in model.pair_indices_] == [
        [1, 2, 4, 5], [0, 1, 3, 4], [0, 2, 3, 5]]
    assert model.predict(x).tolist() == rows


def test_unicode_sequence_fit_and_score_preserve_trailing_nuls(classifier):
    model = classifier.fit(X, list(LABELS))
    assert model.classes_.tolist() == ["q", "q\0"]
    assert model.predict(X).tolist() == LABELS
    assert model.score(X, tuple(LABELS)) == 1.
    assert model.score(X, [name + "\0" for name in LABELS]) == 0.


def test_selection_report_retains_numpy_string_scalars():
    y = [np.str_(value) for value in LABELS]
    model = EnsembleKernelDWD(M=1, theta=1., sv_selection_ratio=1., kernel="linear",
                              random_state=4, stopping="fixed", max_iter=8).fit(X, y)
    assert [row["label"] for row in model.selection_composition_["classes"]] == ["q", "q\0"]


@requires_string_dtype
def test_native_array_rows_match_object_labels(classifier):
    rows = ["id\0A", "id\0B"] * 4
    y = np.array(rows, dtype=np.dtypes.StringDType(coerce=False, na_object="id\0A"))
    control = clone(classifier).fit(X, y.astype(object))
    with pytest.warns(DataConversionWarning):
        classifier.fit(X, list(y[:, None]))
    assert classifier.classes_.dtype == y.dtype
    assert classifier.classes_.tolist() == control.classes_.tolist()
    assert classifier.predict(X).tolist() == control.predict(X).tolist() == rows
    assert classifier.score(X, tuple(row.reshape(()) for row in y[:, None])) == 1.
