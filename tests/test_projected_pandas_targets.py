"""One-vs-one fitting and scoring use the supplied pandas carrier's representation."""

from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.multiclass import check_classification_targets

from dwd.kern_md import KernMD
from ensemble_dwd import EnsembleKernelDWD, EnsembleKernelSVM, OvOClassifier


BASE = 2**53
FEATURES = np.array([[-3.], [1.], [2.]] * 2)
IDS = [0, BASE, BASE + 1] * 2


@pytest.mark.parametrize("cls", [EnsembleKernelDWD, EnsembleKernelSVM, OvOClassifier])
@pytest.mark.parametrize("dtype", [np.longdouble,
                                  pytest.param(np.dtype(np.float64).newbyteorder("S"), id="swapped-float64")])
@pytest.mark.parametrize("projected", [False, True], ids=["array", "series"])
def test_float_training_labels_require_exact_discrete_values(cls, dtype, projected):
    X = np.tile([[-1.], [1.]], (8, 1))
    reads = []

    class Labels(pd.Series):
        def __array__(self, dtype=None, copy=None):
            reads.append(1)
            return values

    series = Labels(np.zeros(len(X)))
    model = (OvOClassifier(KernMD()) if cls is OvOClassifier else
             cls(M=1, theta=1., sv_selection_ratio=1., random_state=0))
    if np.finfo(dtype).nmant >= 63:
        ids = [2**63, 2**63 + 1]
    elif not np.dtype(dtype).isnative:
        ids = [2**63, 2**63 + 2048]
    elif np.iinfo(int).bits == 32:
        ids = [2**30, 2**30 + 1]
    else:
        ids = [BASE, BASE + 2]
    values = np.asarray(ids * 8, dtype=np.uint64).astype(dtype)
    model.fit(X, series if projected else values)
    assert list(map(int, model.classes_)) == ids
    assert list(map(int, model.predict(X))) == ids * 8
    assert reads == ([1] if projected else [])

    weights = np.ones(len(X), dtype=int)
    weights[0] = 5
    for expected in (1., .75):
        if expected == .75:
            values[0] = values[1]
        for truth in (values, series):
            reads.clear()
            assert model.score(X, truth, sample_weight=weights) == expected
            assert reads == ([1] if truth is series else [])

    for other in (.5, np.nan, np.inf, -np.inf):
        values = np.asarray([2**63, other] * 8, dtype=dtype)
        reads.clear()
        with pytest.raises(ValueError, match="continuous" if np.isfinite(other) else "NaN|infinity"):
            model.fit(X, series if projected else values)
        assert reads == ([1] if projected else [])


@pytest.mark.parametrize("cls", [EnsembleKernelDWD, EnsembleKernelSVM, OvOClassifier])
@pytest.mark.parametrize("projected", [False, True], ids=["array", "series"])
def test_float_training_labels_outside_native_integer_range(cls, projected):
    X = np.tile([[-1.], [1.]], (8, 1))
    model = (OvOClassifier(KernMD()) if cls is OvOClassifier else
             cls(M=1, theta=1., sv_selection_ratio=1., random_state=0))
    dtype = np.dtype(np.longdouble).newbyteorder("S")
    high = np.ldexp(np.longdouble(1), 65)
    above_signed = np.nextafter(np.ldexp(np.longdouble(1), 63), np.longdouble(np.inf))

    class Labels(pd.Series):
        def __array__(self, dtype=None, copy=None):
            return values

    for pair in ((high, np.nextafter(high, np.longdouble(np.inf))), (-1, above_signed)):
        values = np.tile(np.asarray(pair, dtype=dtype), 8).astype(dtype)
        ids = list(map(int, pair))
        # Explicit projections already expose exact Python integers here.
        # Raw arrays keep the available sklearn support for their float dtype.
        reference = np.array(ids * 8, dtype=object) if projected else values
        target = Labels(np.zeros(len(X))) if projected else values
        with np.errstate(invalid="ignore"):
            try:
                check_classification_targets(reference)
            except ValueError as error:
                with pytest.raises(ValueError) as actual:
                    model.fit(X, target)
                assert str(actual.value) == str(error)
            else:
                model.fit(X, target)
                assert list(map(int, model.classes_)) == ids
                assert list(map(int, model.predict(X))) == ids * 8
                assert model.score(X, target) == 1.


def test_ovo_scores_adjacent_bare_sparse_ids_without_the_training_fill():
    ids = np.array(IDS, dtype=np.int64)
    y = pd.arrays.SparseArray(ids, fill_value=0.)
    model = OvOClassifier(KernMD(), n_jobs=1).fit(FEATURES, y)
    assert list(map(int, model.classes_)) == IDS[:3]
    assert list(map(int, model.predict(FEATURES))) == IDS
    assert y.sp_index.ngaps == 2 and model.score(FEATURES, y) == 1.

    X = FEATURES[[1, 2, 4, 5]]
    rows = [2, 1, 4, 5]
    held = y.take(rows)
    assert held.sp_index.ngaps == 0
    for weights, expected in ((None, .5), ([1, 2, 3, 4], .7)):
        assert model.score(X, ids[rows], sample_weight=weights) == expected
        assert model.score(X, held, sample_weight=weights) == expected


@pytest.mark.parametrize("carrier", ["bare", "frame", "category_frame"])
def test_ovo_scores_a_finer_sparse_datetime_fill_against_exact_ticks(carrier):
    ids = np.array([0, 1, 2] * 2, dtype="datetime64[ns]")
    y = pd.arrays.SparseArray(ids, fill_value=ids[0])
    model = OvOClassifier(KernMD(), n_jobs=1).fit(FEATURES, y)
    y.fill_value = np.datetime64(1, "ps")
    if carrier == "category_frame":
        levels = pd.arrays.SparseArray(ids[:3], fill_value=ids[0])
        levels.fill_value = np.datetime64(1, "ps")
        y = pd.Categorical.from_codes([0, 1, 2] * 2, categories=pd.Index(levels))
    if carrier != "bare":
        y = pd.Series(y).to_frame()
    assert model.score(FEATURES, y) == 2 / 3
    assert model.score(FEATURES, y, sample_weight=[1, 2, 3, 4, 5, 6]) == 16 / 21
    fitted = OvOClassifier(KernMD(), n_jobs=1).fit(FEATURES, y)
    assert fitted.classes_.dtype == np.dtype("datetime64[ps]")
    assert fitted.classes_.view("i8").tolist() == [1, 1000, 2000]
    assert fitted.score(FEATURES, y) == 1.


@pytest.mark.parametrize(("dtype", "fill", "levels"), [
    (np.int64, 0, [0, BASE, BASE + 1]),
    (np.uint64, 2**64 - 1, [0, 2**64 - 2, 2**64 - 1]),
])
def test_ovo_native_sparse_frame_preserves_pairs_predictions_and_weighted_score(dtype, fill, levels):
    def target(codes):
        labels = np.asarray(levels, dtype=dtype)[codes]
        return pd.DataFrame({"y": pd.arrays.SparseArray(labels, fill_value=dtype(fill))})

    model = OvOClassifier(KernMD(), n_jobs=1).fit(FEATURES, target([0, 1, 2] * 2))
    assert list(map(int, model.classes_)) == levels
    assert [tuple(map(int, pair)) for pair in model.pairs_] == list(combinations(levels, 2))
    assert list(map(int, model.predict(FEATURES))) == levels * 2
    assert model.score(FEATURES, target([0, 2, 2] * 2), sample_weight=[1, 2, 4] * 2) == 5 / 7


@pytest.mark.parametrize("carrier", ["optional_outer", "nested_levels", "optional_frame", "sequence"])
def test_ovo_optional_and_unused_pandas_protocols_preserve_exact_pairs_and_held_score(carrier):
    class OptionalSeries(pd.Series):
        @property
        def __array_interface__(self):
            return self.values.__array_interface__

    class Levels(pd.arrays.SparseArray):
        def to_numpy(self, *args, **kwargs):
            raise AssertionError("categorical level selection does not invoke to_numpy")

    class OptionalFrame(pd.DataFrame):
        @property
        def __array_interface__(self):
            raise AttributeError("unavailable")

        @property
        def _mgr(self):
            return self.__dict__["_mgr"]

        @_mgr.setter
        def _mgr(self, manager):
            self.__dict__["_mgr"] = manager

    class OptionalSparse(pd.arrays.SparseArray):
        @property
        def __array_interface__(self):
            raise AttributeError("unavailable")

    class Sequence(pd.Series):
        @property
        def __array__(self):
            raise AttributeError("use iteration")

    def target(codes):
        ids = np.array(IDS[:3], dtype=np.int64)
        if carrier == "optional_outer":
            return OptionalSeries(pd.arrays.SparseArray(ids[codes], fill_value=0.))
        if carrier == "optional_frame":
            return OptionalFrame({"y": OptionalSparse(ids[codes], fill_value=0.)})
        if carrier == "sequence":
            values = [0., np.uint64(BASE), np.uint64(BASE + 1)]
            return Sequence([values[i] for i in codes], dtype=object)
        levels = Levels(ids, fill_value=np.int64(0))
        labels = pd.Categorical.from_codes(codes, categories=pd.Index(levels))
        levels.fill_value = 0.
        class Series(pd.Series):
            @property
            def _values(self):
                return labels
        return Series(np.zeros(len(codes)))

    model = OvOClassifier(KernMD(), n_jobs=1).fit(FEATURES, target([0, 1, 2] * 2))
    assert list(map(int, model.classes_)) == IDS[:3]
    assert [tuple(map(int, pair)) for pair in model.pairs_] == list(combinations(IDS[:3], 2))
    assert list(map(int, model.predict(FEATURES))) == IDS
    assert model.score(FEATURES, target([0, 2, 2] * 2), sample_weight=[1, 2, 4] * 2) == 5 / 7


@pytest.mark.parametrize("kind", ["sparse", "levels", "interval_frame"])
def test_ovo_preserves_rows_actually_yielded_by_an_extension(kind):
    def target(codes):
        armed = False
        if kind == "sparse":
            class Sparse(pd.arrays.SparseArray):
                @property
                def __array__(self):
                    raise AttributeError("use the sparse iterator")

            ids = np.array(IDS[:3], dtype=np.uint64)[codes]
            return pd.Series(Sparse(ids, fill_value=0.), copy=False)

        base = pd.arrays.IntegerArray if kind == "levels" else pd.arrays.IntervalArray

        class Labels(base):
            @property
            def __array__(self):
                if armed:
                    raise AttributeError("use the iterator")
                return super().__array__

            def __iter__(self):
                if not armed:
                    return super().__iter__()
                indices = super().__iter__() if kind == "levels" else self.left
                values = [0., np.uint64(BASE), np.uint64(BASE + 1)]
                return (values[int(i)] for i in indices)

        if kind == "levels":
            levels = Labels(np.arange(3), np.zeros(3, dtype=bool))
            y = pd.Categorical.from_codes(codes, categories=pd.Index(levels))
        else:
            left = np.array(codes)
            y = pd.DataFrame({"y": Labels.from_arrays(left, left + 1)})
        armed = True
        return y

    model = OvOClassifier(KernMD(), n_jobs=1).fit(FEATURES, target([0, 1, 2] * 2))
    assert list(map(int, model.classes_)) == IDS[:3]
    assert [tuple(map(int, pair)) for pair in model.pairs_] == list(combinations(IDS[:3], 2))
    assert list(map(int, model.predict(FEATURES))) == IDS
    assert model.score(FEATURES, target([0, 2, 2] * 2), sample_weight=[1, 2, 4] * 2) == 5 / 7


@pytest.fixture(params=[pd.Categorical, pd.arrays.IntegerArray])
def projected_array(request):
    base = request.param

    def make(ids):
        class Declared(base):
            def __array__(self, dtype=None, copy=None):
                result = np.array(ids, dtype=np.uint64)
                return result if dtype is None else result.astype(dtype, copy=False)

        stored = np.array([0, BASE, BASE] * 2, dtype=np.int64)
        if base is pd.Categorical:
            return Declared(stored)
        return Declared(stored, np.zeros(len(stored), dtype=bool))

    return make


@pytest.fixture(params=["direct", "series", "index", "frame"])
def target(request, projected_array):
    def make(ids):
        array = projected_array(ids)
        if request.param == "direct":
            outer = array
        elif request.param == "index":
            outer = pd.Index(array)
        elif request.param == "series":
            outer = pd.Series(array)
        else:
            outer = pd.DataFrame({"y": array})
        visible = list(map(int, np.asarray(outer).ravel()))
        if request.param != "frame" or isinstance(array, pd.Categorical):
            assert visible == ids
        return outer, visible

    return make


def test_projected_extension_ovo_pairs_and_predictions(target):
    y, visible = target(IDS)
    levels = sorted(set(visible))
    expected_pairs = list(combinations(levels, 2))
    model = OvOClassifier(KernMD(), n_jobs=1).fit(FEATURES, y)
    assert list(map(int, model.classes_)) == levels
    assert [tuple(map(int, pair)) for pair in model.pairs_] == expected_pairs
    assert list(map(int, model.predict(FEATURES))) == visible


def test_projected_extension_ovo_scores_against_python_integer_truth(target):
    native = np.array(IDS, dtype=np.uint64)
    model = OvOClassifier(KernMD(), n_jobs=1).fit(FEATURES, native)
    predicted = list(map(int, model.predict(FEATURES)))
    assert predicted == IDS
    y, visible = target([BASE, BASE + 1, BASE + 1] * 2)
    correct = [a == b for a, b in zip(predicted, visible)]
    weights = [1, 2, 4] * 2
    assert model.score(FEATURES, y) == sum(correct) / len(correct)
    assert model.score(FEATURES, y, sample_weight=weights) == sum(
        w for w, hit in zip(weights, correct) if hit) / sum(weights)


@pytest.mark.parametrize("carrier", ["categorical", "index", "series"])
def test_ovo_uses_the_first_selected_storage_for_pairs_and_weighted_scoring(carrier):
    first = pd.Index([0, BASE, BASE + 1], dtype="uint64")
    later = pd.Index([0, BASE, BASE + 4], dtype="uint64")
    reads = []

    class Category(pd.Categorical):
        @property
        def categories(self):
            reads.append("categories")
            return first if len(reads) == 1 else later

    class Index(pd.Index):
        @property
        def _data(self):
            reads.append("_data")
            return self.first if len(reads) == 1 else self.later

        @_data.setter
        def _data(self, values):
            self.first = values

    class Series(pd.Series):
        @property
        def _values(self):
            reads.append("_values")
            return super()._values if len(reads) == 1 else self.later

    def target(codes):
        if carrier == "index":
            y = Index._simple_new(np.asarray(first)[codes])
            y.later = np.asarray(later)[codes]
        elif carrier == "series":
            y = Series(pd.arrays.SparseArray(np.asarray(first)[codes].astype(np.int64), fill_value=0.))
            object.__setattr__(y, "later", np.asarray(later)[codes])
        else:
            y = Category.from_codes(codes, categories=[0, 10, 20])
        reads.clear()
        return y

    model = OvOClassifier(KernMD(), n_jobs=1).fit(FEATURES, target([0, 1, 2] * 2))
    assert len(reads) == 1
    assert list(map(int, model.classes_)) == IDS[:3]
    assert [tuple(map(int, pair)) for pair in model.pairs_] == list(combinations(IDS[:3], 2))
    assert list(map(int, model.predict(FEATURES))) == IDS
    assert model.score(FEATURES, target([0, 2, 2] * 2), sample_weight=[1, 2, 4] * 2) == 5 / 7
    assert len(reads) == 1
