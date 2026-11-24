"""OvO uses the observed class representatives in row selection and encoding."""

from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
from joblib import parallel_config
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from dwd.kern_md import KernMD
from ensemble_dwd import EnsembleKernelDWD, IneligibleSelectionError, OvOClassifier
from ensemble_dwd._failure_context import failure_context


def training(classes):
    # Interleaved rows reveal the order used by each binary estimator.
    codes = np.tile(np.arange(len(classes)), 3)
    return codes[:, None].astype(float), [classes[code] for code in codes]


def check_pairs(model, X, rows, classes):
    assert model.classes_.tolist() == classes
    assert model.pairs_ == tuple(combinations(classes, 2))
    assert model.class_pair_indices_.tolist() == list(map(list, combinations(range(len(classes)), 2)))
    for (first, second), selected, child in zip(
            model.pairs_, model.pair_indices_, model.estimators_, strict=True):
        # Scalar row-to-class comparison is independent of NumPy promotion.
        expected = [i for i, row in enumerate(rows) if row == first or row == second]
        assert selected.tolist() == expected
        np.testing.assert_array_equal(child._Xfit, X[expected])
        signs = [1 if rows[i] == first else -1 for i in expected]
        np.testing.assert_array_equal(np.sign(child.dual_coef_[0]), signs)
    assert model.predict(X).tolist() == rows


@pytest.mark.parametrize("classes,backend,jobs", [
    (["q", "q\0"], "threading", 1),
    (["q", "q\0", "z"], "threading", 2),
    (["q", "q\0", "z"], "loky", 2),
], ids=["two", "three-threading", "three-loky"])
def test_object_string_pairs(classes, backend, jobs):
    X, rows = training(classes)
    with parallel_config(backend=backend, n_jobs=jobs,
                         inner_max_num_threads=1 if backend == "loky" else None):
        model = OvOClassifier(KernMD(), n_jobs=None).fit(X, np.fromiter(rows, dtype=object))
    assert model.classes_.dtype == object
    check_pairs(model, X, rows, classes)


@pytest.mark.parametrize("classes", [["q", "q\0"], ["q", "q\0", "z"]], ids=["two", "three"])
def test_sparse_iterator_string_pairs(classes):
    pd = pytest.importorskip("pandas")
    X, rows = training(classes)
    calls = []

    class SelectedRows(pd.arrays.SparseArray):
        @property
        def __array__(self):
            raise AttributeError

        def __iter__(self):
            calls.append("iter")
            return iter(rows)

    target = SelectedRows(np.arange(len(X)), fill_value=0)
    model = OvOClassifier(KernMD()).fit(X, target)
    assert calls == ["iter"]
    check_pairs(model, X, rows, classes)
    # Give the wrong member of the padded pair for one heavily weighted row.
    rows[1] = rows[0]
    weights = np.ones(len(rows), dtype=int)
    weights[1] = 5
    assert model.score(X, target, sample_weight=weights) == (weights.sum() - 5) / weights.sum()
    assert calls == ["iter", "iter"]


def test_variable_width_string_dtype_pairs():
    if not hasattr(np.dtypes, "StringDType"):
        pytest.skip("NumPy has no variable-width string dtype")
    classes = ["q", "q\0", "z"]
    X, rows = training(classes)
    y = np.array(rows, dtype=np.dtypes.StringDType())
    model = OvOClassifier(KernMD()).fit(X, y)
    assert model.classes_.dtype == y.dtype
    check_pairs(model, X, rows, classes)


def test_supplied_fixed_width_projection():
    pd = pytest.importorskip("pandas")
    X, rows = training(["q", "q\0", "z"])
    projected = np.asarray(rows, dtype="U2")
    calls = []

    class Supplied(pd.arrays.SparseArray):
        def __array__(self, dtype=None, copy=None):
            calls.append("array")
            return np.asarray(projected, dtype=dtype)

        def __iter__(self):
            raise AssertionError("unselected iterator")

    model = OvOClassifier(KernMD()).fit(X, Supplied(np.arange(len(X)), fill_value=0))
    assert calls == ["array"]
    assert model.classes_.dtype == projected.dtype
    check_pairs(model, X, projected.tolist(), ["q", "z"])


@pytest.mark.parametrize("n_classes", [2, 3])
def test_bytes_follow_classification_target_validation(n_classes):
    # scikit-learn versions differ in their support for byte-string labels.
    classes = [b"q", b"q\0", b"z"][:n_classes]
    X, rows = training(classes)
    y = np.fromiter(rows, dtype=object)
    model = OvOClassifier(KernMD()).fit(X, X[:, 0])
    try:
        check_classification_targets(y)
    except (ValueError, TypeError) as reference:
        with pytest.raises(type(reference)) as actual:
            model.fit(X, y)
        assert str(actual.value) == str(reference)
        with pytest.raises(NotFittedError):
            check_is_fitted(model)
    else:
        check_pairs(model.fit(X, y), X, rows, classes)


def test_comparisons_use_original_object_representatives():
    class Folded(str):
        def __eq__(self, other):
            return type(other) is Folded and self.casefold() == other.casefold()

        def __ne__(self, other):
            return not self == other

        def __lt__(self, other):
            return self.casefold() < other.casefold()

        __hash__ = None

    classes = [Folded("q"), Folded("q\0"), Folded("z")]
    X, rows = training(classes)
    rows[3:6] = [Folded(label.upper()) for label in classes]
    model = OvOClassifier(KernMD()).fit(X, np.fromiter(rows, dtype=object))
    check_pairs(model, X, rows, classes)
    assert all(any(label is row for row in rows) for label in model.classes_)


def test_object_binary_failure_counts():
    y = np.fromiter(["q", "q\0", "q", "q\0", "q"], dtype=object)
    model = EnsembleKernelDWD(sv_selection_ratio=.2)
    with pytest.raises(IneligibleSelectionError, match="fewer than two") as error:
        model.fit(np.arange(len(y))[:, None], y)
    record = failure_context(error.value)
    assert record["classes"] == ["q", "q\0"]
    assert record["training_class_counts"] == [3, 2]


def test_classification_report_counts(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from evaluation_reporting import classification_report

    actual = np.fromiter(["q", "q", "q\0", "q\0", "q\0", "z"], dtype=object)
    # Fixed-width predictions lack the padded class that is present in truth.
    predicted = np.array(["q", "z", "q", "q", "z", "z"])
    report = classification_report(actual, predicted)
    assert [(r["label"], r["support"], r["predicted_count"], r["correct"])
            for r in report["per_class"]] == [("q", 2, 3, 1), ("q\0", 3, 0, 0), ("z", 1, 3, 1)]
    assert report["accuracy"] == 2 / 6
    assert report["balanced_accuracy"] == .5


def test_pair_selection_reports(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from cv_checkpoint import diagnostics
    from evaluation_reporting import selection_composition

    classes = ["q", "q\0", "z"]
    rows = ["q", "q\0", "z", "q", "q\0", "z", "q\0"]
    X = np.asarray([classes.index(label) for label in rows])[:, None]
    y = np.fromiter(rows, dtype=object)
    binary = EnsembleKernelDWD(M=1, theta=1., sv_selection_ratio=1.,
                               gamma=1., max_iter=3, random_state=42)
    model = OvOClassifier(binary).fit(X, y)
    facts = diagnostics(model, y)
    report = selection_composition(facts, y)
    for pair, detail, composition in zip(model.pairs_, facts["pairwise_estimators"],
                                         report["pairwise"], strict=True):
        counts = [rows.count(pair[1]), rows.count(pair[0])]
        assert detail["pair"] == list(pair)
        selected = detail["diagnostics"]["selected_rows"]["class_counts"]
        assert selected == [{"label": label, "count": n} for label, n in zip([-1, 1], counts)]
        assert composition["available"]
        assert [r["training_count"] for r in composition["classes"]] == counts
        assert [r["selected_count"] for r in composition["classes"]] == counts


def report_target(rows, carrier, calls):
    pd = pytest.importorskip("pandas")
    exact = np.fromiter(rows, dtype=object)
    if carrier in ("sparse", "sparse-series"):
        array = pd.arrays.SparseArray(exact, fill_value=np.nan)
        return array if carrier == "sparse" else pd.Series(array)
    backing = pd.arrays.SparseArray(np.arange(len(rows)), fill_value=0)
    if carrier == "projection":
        class Supplied(pd.Series):
            def __array__(self, dtype=None, copy=None):
                calls.append("array")
                return np.asarray(np.asarray(rows, dtype="U2"), dtype=dtype)

            def __iter__(self):
                raise AssertionError("unselected iterator")
        return Supplied(backing)

    class Selected(pd.arrays.SparseArray):
        @property
        def __array__(self):
            raise AttributeError

        def __iter__(self):
            calls.append("iter")
            return iter(rows)
    return Selected(backing)


@pytest.mark.parametrize("carrier", ["iterator", "sparse", "sparse-series", "projection"])
def test_original_report_inputs(carrier, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from cv_checkpoint import diagnostics, _row_diagnostics
    from evaluation_reporting import classification_report, selection_composition

    rows = ["q", "q\0", "z", "q", "q\0", "z", "q\0"]
    expected = np.asarray(rows, dtype="U2").tolist() if carrier == "projection" else rows
    classes = sorted(set(expected))
    X = np.asarray([classes.index(label) for label in expected])[:, None]
    calls = []
    target = report_target(rows, carrier, calls)
    per_call = {"iterator": ["iter"], "projection": ["array"]}.get(carrier, [])

    def check_read():
        assert calls == per_call
        calls.clear()

    binary = EnsembleKernelDWD(M=1, theta=1., sv_selection_ratio=1.,
                               gamma=1., max_iter=5, random_state=42)
    model = OvOClassifier(binary).fit(X, target)
    check_read()
    predictions = model.predict(X)
    assert predictions.tolist() == expected
    facts = diagnostics(model, target)
    check_read()
    composition = selection_composition(facts, target)
    check_read()
    for (first, second), pair, summary in zip(model.pairs_, facts["pairwise_estimators"],
                                              composition["pairwise"], strict=True):
        counts = [{"label": -1, "count": expected.count(second)},
                  {"label": 1, "count": expected.count(first)}]
        detail = pair["diagnostics"]
        assert detail["selected_rows"]["class_counts"] == counts
        assert detail["subset_rows"][0]["class_counts"] == counts
        assert [r["training_count"] for r in summary["classes"]] == [c["count"] for c in counts]
        assert [r["selected_count"] for r in summary["classes"]] == [c["count"] for c in counts]

    # The row helper and the public binary diagnostics entry also accept the
    # original carrier. A binary fit reports both subset and selected rows.
    binary_rows = [value for value in rows if value != "q"]
    binary_expected = [value.rstrip("\0") for value in binary_rows] if carrier == "projection" else binary_rows
    binary_target = report_target(binary_rows, carrier, calls)
    binary.fit(np.asarray([value == "z" for value in binary_rows])[:, None], binary_target)
    check_read()
    counts = [{"label": cls, "count": binary_expected.count(cls)} for cls in sorted(set(binary_expected))]
    direct = _row_diagnostics(binary.selected_indices_, binary_target)
    check_read()
    binary_facts = diagnostics(binary, binary_target)
    check_read()
    assert direct["class_counts"] == counts
    assert binary_facts["selected_rows"]["class_counts"] == counts
    assert binary_facts["subset_rows"][0]["class_counts"] == counts

    # One genuinely wrong padded-class observation distinguishes accuracy from
    # a comparison of the two fixed-width projections.
    held_rows = rows.copy()
    held_rows[1] = "q"
    held = report_target(held_rows, carrier, calls)
    held_expected = expected.copy()
    held_expected[1] = "q"
    correct = [a == b for a, b in zip(held_expected, expected)]
    weights = np.arange(1, len(rows) + 1)
    assert model.score(X, held, sample_weight=weights) == np.average(correct, weights=weights)
    check_read()
    for actual, predicted, truth, guesses in [
            (held, predictions, held_expected, expected), (predictions, held, expected, held_expected)]:
        report = classification_report(actual, predicted)
        check_read()
        assert report["accuracy"] == np.mean(correct)
        assert [(r["label"], r["support"], r["predicted_count"], r["correct"])
                for r in report["per_class"]] == [
            (cls, truth.count(cls), guesses.count(cls), sum(a == b == cls for a, b in zip(truth, guesses)))
            for cls in classes]


def test_report_projection_error(monkeypatch):
    pd = pytest.importorskip("pandas")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from cv_checkpoint import diagnostics, _row_diagnostics
    from evaluation_reporting import classification_report, selection_composition

    calls = []

    class Refused(pd.arrays.SparseArray):
        def __array__(self, dtype=None, copy=None):
            calls.append("array")
            raise ValueError("selected projection failed")

        def __iter__(self):
            raise AssertionError("unselected iterator")

    y = np.tile(np.arange(3), 2)
    model = OvOClassifier(KernMD()).fit(y[:, None], y)
    refused = Refused(y, fill_value=0)
    for action in (lambda: diagnostics(model, refused), lambda: _row_diagnostics([0, 1], refused),
                   lambda: selection_composition(diagnostics(model), refused),
                   lambda: classification_report(refused, y), lambda: classification_report(y, refused)):
        with pytest.raises(ValueError, match="selected projection failed"):
            action()
        assert calls == ["array"]
        calls.clear()


@pytest.mark.parametrize("projected", [False, True], ids=["dense", "sparse-projection"])
@pytest.mark.parametrize("values,predictions", [
    (np.fromiter([2**63, np.uint64(2**63)], dtype=object),
     np.array([2**63 - 1, 2**63 - 1], dtype=np.int64)),
    (np.fromiter([2**63 + 1, np.uint32(3)], dtype=object),
     np.array([float(2**63), 3.])),
    (np.array([2**63 - 1, 2**63, 5], dtype=np.uint64),
     np.array([2**63 - 1, -1, 5], dtype=np.int64)),
    (np.fromiter([np.uint64(2**63 + 1), np.uint64(3)], dtype=object),
     np.array([float(2**63), 3.])),
], ids=["signed-boundary", "integral-float", "native-integers", "boxed-numpy-integers"])
def test_mixed_numeric_report_counts(monkeypatch, projected, values, predictions):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from evaluation_reporting import classification_report

    calls = []
    target = values
    if projected:
        pd = pytest.importorskip("pandas")

        class Projected(pd.arrays.SparseArray):
            def __array__(self, dtype=None, copy=None):
                calls.append("array")
                return np.asarray(values, dtype=dtype)

            def __iter__(self):
                raise AssertionError("unselected iterator")

        target = Projected(np.arange(len(values)), fill_value=0)

    # Python int/float equality distinguishes a large ID from the adjacent
    # represented float even where NumPy would promote both to float64.
    exact = [v.item() if isinstance(v, np.generic) else v for v in values]
    guesses = predictions.tolist()
    classes = sorted(set(exact + guesses))
    for actual, predicted, truth, observed in (
            (target, predictions, exact, guesses), (predictions, target, guesses, exact)):
        report = classification_report(actual, predicted)
        counts = [(cls, truth.count(cls), observed.count(cls),
                   sum(a == b == cls for a, b in zip(truth, observed))) for cls in classes]
        assert [(r["label"], r["support"], r["predicted_count"], r["correct"])
                for r in report["per_class"]] == counts
        recalls = [correct / support if support else None for _, support, _, correct in counts]
        assert [r["recall"] for r in report["per_class"]] == recalls
        assert report["accuracy"] == sum(a == b for a, b in zip(truth, observed)) / len(truth)
        assert report["balanced_accuracy"] == np.mean([r for r in recalls if r is not None])
        assert calls == (["array"] if projected else [])
        calls.clear()


class _UnhashableTextType(type):
    __hash__ = None


class _UnhashableWord(str, metaclass=_UnhashableTextType):
    __hash__ = None


class _Word(str):
    pass


class _Byte(bytes, metaclass=_UnhashableTextType):
    pass


class _PassiveByte(bytes):
    pass


def check_byte_conversion_refusal(rows, operations):
    # Probe a separate iterator so the public operations must each read their own input.
    native_calls = []
    native_target = report_target(rows, "iterator", native_calls)
    try:
        np.asarray(native_target)
    except TypeError as reference:
        assert native_calls == ["iter"]
        assert reference.args == (f"unhashable type: '{_UnhashableTextType.__name__}'",)
        for operation in operations:
            calls = []
            target = report_target(rows, "iterator", calls)
            with pytest.raises(type(reference)) as actual:
                operation(target)
            assert type(actual.value) is type(reference)
            assert actual.value.args == reference.args
            assert type(actual.value.__cause__) is type(reference.__cause__)
            if reference.__cause__ is not None:
                assert actual.value.__cause__.args == reference.__cause__.args
            assert calls == ["iter"]
        return True
    assert native_calls == ["iter"]
    return False


def check_mixed_counts(report, truth, guesses):
    classes = []
    for label in [*truth, *guesses]:
        if not any(label == previous for previous in classes):
            classes.append(label)
    assert len(report["per_class"]) == len(classes)
    correct = [bool(a == b) for a, b in zip(truth, guesses)]
    assert report["accuracy"] == np.mean(correct)
    recalls = []
    for row, cls in zip(report["per_class"], classes):
        label = cls.item() if isinstance(cls, np.generic) else cls
        assert type(row["label"]) is type(label) and row["label"] == label
        included = [bool(a == cls) for a in truth]
        predicted_count = sum(bool(b == cls) for b in guesses)
        matched = sum(hit and member for hit, member in zip(correct, included))
        support = sum(included)
        assert (row["support"], row["predicted_count"], row["correct"]) == (support, predicted_count, matched)
        assert row["recall"] == (matched / support if support else None)
        if support:
            recalls.append(matched / support)
    assert report["balanced_accuracy"] == np.mean(recalls)


@pytest.mark.parametrize("token", ["1", b"1", _Word("1"), _Byte(b"1"), _PassiveByte(b"1")],
                         ids=["text", "bytes", "inherited-text", "inherited-bytes", "passive-bytes"])
@pytest.mark.parametrize("trained", ["numeric", "text"])
def test_mixed_iterator_report_membership(monkeypatch, token, trained):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from evaluation_reporting import classification_report

    X, numbers = training([np.int16(1), np.int16(2)])
    rows = numbers.copy()
    rows[2] = token
    train = np.asarray(numbers) if trained == "numeric" else np.array(["1", "2"] * 3)
    model = KernMD().fit(X, train)
    predictions = model.predict(X)
    if type(token) is _Byte and check_byte_conversion_refusal(rows, (
            lambda y: model.score(X, y),
            lambda y: classification_report(y, predictions),
            lambda y: classification_report(predictions, y),
            lambda y: classification_report(y, y))):
        return
    calls = []
    target = report_target(rows, "iterator", calls)
    score = model.score(X, target)
    assert calls == ["iter"]
    calls.clear()
    report = classification_report(target, predictions)
    assert calls == ["iter"]
    assert report["accuracy"] == score
    check_mixed_counts(report, rows, predictions)
    calls.clear()
    check_mixed_counts(classification_report(predictions, target), predictions, rows)
    assert calls == ["iter"]
    calls.clear()
    check_mixed_counts(classification_report(target, target), rows, rows)
    assert calls == ["iter"]


@pytest.mark.parametrize("truth,guesses", [
    (np.array([2, 1]), np.array(["1", "2"])),
    (np.array([b"\xff", b"1"]), np.array(["1", "2"])),
    (np.array([b"2", b"1"]), np.array([1, 2])),
], ids=["number-text", "bytes-text", "bytes-number"])
def test_disjoint_native_report_domains(monkeypatch, truth, guesses):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from evaluation_reporting import classification_report

    check_mixed_counts(classification_report(truth, guesses), truth, guesses)
    check_mixed_counts(classification_report(guesses, truth), guesses, truth)


@pytest.mark.parametrize("token", [_Byte(b"1"), _PassiveByte(b"1")],
                         ids=["inherited-bytes", "passive-bytes"])
def test_mixed_report_rejects_overlapping_classes(monkeypatch, token):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from evaluation_reporting import classification_report

    X, numbers = training([np.int16(1), np.int16(2)])
    rows = numbers.copy()
    rows[2] = token
    assert rows[2] == np.int16(1) and np.int16(1) == 1 and rows[2] != 1
    model = KernMD().fit(X, np.fromiter(map(int, numbers), dtype=object))
    predictions = model.predict(X)
    if type(token) is _Byte and check_byte_conversion_refusal(rows, (
            lambda y: model.score(X, y),
            lambda y: classification_report(y, predictions),
            lambda y: classification_report(predictions, y))):
        return
    calls = []
    target = report_target(rows, "iterator", calls)
    assert model.score(X, target) == 5 / 6
    assert calls == ["iter"]
    for actual, predicted in ((target, predictions), (predictions, target)):
        calls.clear()
        with pytest.raises(ValueError) as error:
            classification_report(actual, predicted)
        assert error.value.args == ("Mixed labels compare inconsistently and do not define disjoint report classes.",)
        assert calls == ["iter"]


def test_unhashable_text_report_projection(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from evaluation_reporting import classification_report

    X, rows = training([np.int16(1), np.int16(2)])
    rows[2] = _UnhashableWord("1")
    train = np.array(["1", "2"] * 3)
    model = KernMD().fit(X, train)
    predictions = model.predict(X)
    exact = np.fromiter(rows, dtype=object)
    try:
        rows[0] == rows[2]
    except TypeError as reference:
        with pytest.raises(type(reference)) as actual:
            classification_report(exact, predictions)
        assert actual.value.args == reference.args
    else:
        check_mixed_counts(classification_report(exact, predictions), rows, predictions)

    calls = []
    target = report_target(rows, "iterator", calls)
    # NumPy versions differ in whether the selected conversion accepts this
    # metaclass. Preserve the selected array or the original exception.
    try:
        projected = np.asarray(target)
    except (TypeError, ValueError, SystemError) as reference:
        for operation in (lambda: model.score(X, target),
                          lambda: classification_report(target, predictions)):
            calls.clear()
            with pytest.raises(type(reference)) as actual:
                operation()
            assert (actual.value.__cause__ or actual.value).args == (reference.__cause__ or reference).args
            assert calls == ["iter"]
    else:
        calls.clear()
        assert classification_report(target, predictions) == classification_report(projected, predictions)
        assert calls == ["iter"]


def test_boxed_numpy_text_report_values(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from cv_checkpoint import diagnostics, _row_diagnostics
    from evaluation_reporting import classification_report
    from ensemble_dwd.multiclass import _pair_context

    for scalar, plain in ((np.str_, ["q", "q\0"]), (np.bytes_, [b"q", b"q\0"])):
        y = np.fromiter((scalar(value) for value in plain * 3), dtype=object)
        report = classification_report(y, y)
        assert [row["label"] for row in report["per_class"]] == plain
        assert [row["correct"] for row in report["per_class"]] == [3, 3]
        assert _pair_context(y[:2]) == f"OVO pair {tuple(plain)!r}"
    plain = ["q", "q\0"]
    y = np.fromiter((np.str_(v) for v in plain * 2 + ["q"]), dtype=object)
    X = np.asarray([0, 1, 0, 1, 0])[:, None]
    model = OvOClassifier(KernMD()).fit(X, y)
    assert diagnostics(model, y)["pairwise_estimators"][0]["pair"] == plain
    assert _row_diagnostics(np.arange(len(y)), y)["class_counts"] == [
        {"label": "q", "count": 3}, {"label": "q\0", "count": 2}]
    with pytest.raises(IneligibleSelectionError) as error:
        EnsembleKernelDWD(sv_selection_ratio=.2).fit(X, y)
    assert failure_context(error.value)["classes"] == plain
    assert failure_context(error.value)["training_class_counts"] == [3, 2]
