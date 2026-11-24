"""Temporal classification reports preserve the represented class and its counts."""

from datetime import date, datetime, timedelta, timezone
from itertools import permutations
import json
from pathlib import Path

import numpy as np
import pytest

from dwd.kern_md import KernMD


@pytest.fixture
def report(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from evaluation_reporting import classification_report
    return classification_report


def sparse(values):
    pd = pytest.importorskip("pandas")
    # The guard prevents pandas from inferring a date or duration resolution.
    stored = pd.arrays.SparseArray(np.array(["guard", *values], dtype=object),
                                   dtype=pd.SparseDtype(object, "unused"))
    return stored.take(np.arange(1, len(values) + 1))


def counts(result):
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    return [(r["support"], r["predicted_count"], r["correct"], r["recall"])
            for r in result["per_class"]]


def test_sparse_month_and_week_against_independent_matches(report):
    X = np.array([[-1.], [1.]] * 2)
    weeks = np.array(["2023-12-28", "2024-01-04"] * 2, dtype="datetime64[W]")
    model = KernMD().fit(X, weeks)
    actual = sparse(np.array(["2024-01"] * 4, dtype="datetime64[M]"))
    summary = report(actual, model.predict(X))
    assert summary["accuracy"] == summary["balanced_accuracy"] == model.score(X, actual) == 0.
    assert counts(summary) == [(0, 2, 0, None), (4, 0, 0, 0.), (0, 2, 0, None)]


@pytest.mark.parametrize("kind", ["M", "m"])
@pytest.mark.parametrize("carrier", [np.asarray, sparse])
def test_exact_ticks_with_and_without_shared_unit(report, kind, carrier):
    large = 2**61 + 1
    predictions = np.array([1, large, 1, large], dtype=kind + "8[ns]")
    for tick, expected in [(1000, .5), (1, 0.)]:
        actual = carrier(np.array([tick] * 4, dtype=kind + "8[ps]"))
        summary = report(actual, predictions)
        assert summary["accuracy"] == summary["balanced_accuracy"] == expected
        if tick == 1000:
            assert counts(summary) == [(4, 2, 2, .5), (0, 2, 0, None)]
        else:
            # Both 1 ps and 1 ns have raw tick count 1; they need distinct labels.
            assert counts(summary) == [(4, 0, 0, 0.), (0, 2, 0, None), (0, 2, 0, None)]
            labels = [row["label"] for row in summary["per_class"]]
            assert [v["ticks"] for v in labels] == [1, 1, large]
            assert len(set(v["dtype"] for v in labels[:2])) == 2


def test_nat_counts_as_missing_even_when_paired_with_itself(report):
    truth = np.array(["NaT", "2024-01-01", "NaT"], dtype="M8[D]")
    guesses = np.array(["NaT", "2024-01", "2024-01"], dtype="M8[M]")
    for predicted in (guesses, guesses.astype("M8[D]")):
        result = report(truth, predicted)
        assert result["accuracy"] == 1 / 3 and result["balanced_accuracy"] == .5
        assert counts(result) == [(1, 2, 1, 1.), (2, 1, 0, 0.)]
        assert result["per_class"][1]["label"] is None


@pytest.mark.parametrize("kind", ["datetime64", "timedelta64"])
@pytest.mark.parametrize("unit", ["D", "h"])
@pytest.mark.parametrize("carrier", [lambda v: np.array(v, dtype=object), sparse])
def test_nat_and_none_are_distinct_when_both_publish_as_null(report, kind, unit, carrier):
    day = np.array(1, dtype=f"{kind}[D]")[()]
    same_day = np.array(1 if unit == "D" else 24, dtype=f"{kind}[{unit}]")[()]
    nat = np.array("NaT", dtype=f"{kind}[D]")[()]
    other_nat = np.array("NaT", dtype=f"{kind}[{unit}]")[()]
    a = carrier([day, None, nat, nat, None])
    b = np.array([same_day, None, other_nat, same_day, other_nat], dtype=object)
    for truth, guesses, expected in [
            (a, a, [(1, 1, 1, 1.), (2, 2, 2, 1.), (2, 2, 0, 0.)]),
            (a, b, [(1, 2, 1, 1.), (2, 1, 1, .5), (2, 2, 0, 0.)]),
            (b, a, [(2, 1, 1, .5), (1, 2, 1, 1.), (2, 2, 0, 0.)])]:
        result = report(truth, guesses)
        assert counts(result) == expected
        assert result["accuracy"] == sum(row[2] for row in expected) / 5
        assert result["balanced_accuracy"] == sum(row[3] for row in expected) / 3
        labels = [row["label"] for row in result["per_class"]]
        assert labels[1] == {"type": "builtins.NoneType", "value": None}
        assert labels[2] == {"type": f"numpy.{kind}", "value": None}


def test_temporal_report_credits_none_when_nat_is_absent(report):
    truth = np.array([np.datetime64("2024-01-01"), None], dtype=object)
    for target in (truth, sparse(truth)):
        result = report(target, truth)
        assert counts(result) == [(1, 1, 1, 1.), (1, 1, 1, 1.)]
        assert result["per_class"][1]["label"] is None
        assert result["accuracy"] == result["balanced_accuracy"] == 1.


def test_temporal_report_preserves_reflexive_custom_class_with_nat_payload(report):
    pd = pytest.importorskip("pandas")

    class Token(type(pd.NaT)):
        def __eq__(self, other):
            return type(other) is Token

        def __ne__(self, other):
            return not self == other

    token = Token()
    day = np.datetime64("2024-01-01", "D")
    for rows, hits in [([day, token], [1, 1]), ([token, day], [1, 1]),
                       ([day, token, pd.NaT], [1, 1, 0])]:
        truth = np.array(rows, dtype=object)
        result = report(truth, truth)
        assert [r["correct"] for r in result["per_class"]] == hits
        assert result["accuracy"] == result["balanced_accuracy"] == sum(hits) / len(rows)
        assert all(r["label"] is not None for r in result["per_class"] if r["correct"])
    rows = np.array([token, pd.NaT], dtype=object)
    result = report(rows, rows)
    assert result["per_class"][0]["label"] is token
    assert result["per_class"][1]["label"] is None
    assert result["accuracy"] == result["balanced_accuracy"] == .5


@pytest.mark.parametrize("carrier", [lambda v: np.array(v, dtype=object), sparse])
def test_nonreflexive_custom_nat_shares_one_missing_class(report, carrier):
    pd = pytest.importorskip("pandas")

    class Missing(type(pd.NaT)):
        # Scalar membership can be nonreflexive and unhashable.
        __hash__ = None

    missing = Missing()
    day = np.datetime64("2024-01-01", "D")
    truth = carrier([day, missing, None, pd.NaT])
    guesses = carrier([day, pd.NaT, None, np.datetime64("NaT", "ns")])
    for a, b in ((truth, guesses), (guesses, truth)):
        result = report(a, b)
        assert result["accuracy"] == .5 and result["balanced_accuracy"] == 2 / 3
        assert sorted(counts(result)) == [(1, 1, 1, 1.), (1, 1, 1, 1.), (2, 2, 0, 0.)]


def test_missing_temporal_objects_and_none_without_a_numpy_datetime(report):
    pd = pytest.importorskip("pandas")
    from search_support import json_safe

    class Missing(type(pd.NaT)):
        pass

    a = np.array([datetime(2024, 1, 1), Missing(), None, pd.NaT], dtype=object)
    for rows in (a, a[::-1]):
        result = report(rows, rows)
        assert result["accuracy"] == .5 and result["balanced_accuracy"] == 2 / 3
        saved = json_safe(result)
        assert sorted(counts(saved)) == [(1, 1, 1, 1.), (1, 1, 1, 1.), (2, 2, 0, 0.)]
        assert len({json.dumps(row["label"], sort_keys=True) for row in saved["per_class"]}) == 3


@pytest.mark.parametrize("carrier", [lambda v: np.array(v, dtype=object), sparse])
def test_missing_subclass_does_not_change_order_or_display_of_python_time(report, carrier):
    pd = pytest.importorskip("pandas")

    class Missing(type(pd.NaT)):
        __hash__ = None

    for finite in ([date(2024, 1, 1), datetime(2024, 1, 1)],
                   [datetime(2024, 1, 1), date(2024, 1, 1)]):
        ordinary = carrier([*finite, pd.NaT])
        custom = carrier([*finite, Missing()])
        expected = report(ordinary, ordinary)
        for actual, predicted in ((ordinary, custom), (custom, ordinary), (custom, custom)):
            result = report(actual, predicted)
            assert result == expected == json.loads(json.dumps(result, allow_nan=False))
            assert result["accuracy"] == result["balanced_accuracy"] == 2 / 3
            assert [r["label"] for r in result["per_class"]] == [
                finite[0].isoformat(), finite[1].isoformat(), None]
            assert counts(result) == [(1, 1, 1, 1.), (1, 1, 1, 1.), (1, 1, 0, 0.)]


def test_native_missing_marker_does_not_change_custom_class_order(report):
    pd = pytest.importorskip("pandas")

    class LatestFirst(date):
        def __lt__(self, other):
            return date.__gt__(self, other)

        def __gt__(self, other):
            return date.__lt__(self, other)

    later, earlier = LatestFirst(2024, 2, 1), LatestFirst(2024, 1, 1)
    ordinary = np.array([earlier, later, pd.NaT], dtype=object)
    rows = np.array([earlier, later, np.datetime64("NaT", "D")], dtype=object)
    for actual, predicted in ((ordinary, rows), (rows, ordinary)):
        result = report(actual, predicted)
        assert result == report(ordinary, ordinary)
        assert [r["label"]["value"] for r in result["per_class"][:2]] == [
            later.isoformat(), earlier.isoformat()]
        assert result["per_class"][2]["label"] is None


def test_sparse_object_durations_keep_their_published_labels_with_either_nat(report):
    pd = pytest.importorskip("pandas")

    class Missing(type(pd.NaT)):
        pass

    durations = [timedelta(days=1), pd.Timedelta(days=2)]
    ordinary = sparse([*durations, pd.NaT])
    custom = sparse([*durations, Missing()])
    for actual, predicted in ((ordinary, ordinary), (ordinary, custom), (custom, ordinary)):
        result = report(actual, predicted)
        assert [r["label"] for r in result["per_class"]] == [str(v) for v in durations] + [None]
        assert counts(result) == [(1, 1, 1, 1.), (1, 1, 1, 1.), (1, 1, 0, 0.)]


@pytest.mark.parametrize("categorical", [False, True])
@pytest.mark.parametrize("occupied_fill", [False, True])
def test_sparse_boxed_datetimes_remain_distinct_from_date_predictions(report, categorical, occupied_fill):
    pd = pytest.importorskip("pandas")
    far = datetime(1600, 1, 1)
    fine = pd.Timestamp("2024-01-02T00:00:00.000000001")
    actual = np.array([far, fine] * 2, dtype=object)
    dates = np.array([v.date() for v in actual], dtype=object)
    X = np.array([[-1.], [1.], [-2.], [2.]])
    guesses = KernMD().fit(X, dates).predict(X)
    assert all(a != b and a.date() == b for a, b in zip(actual, guesses))
    dtype = pd.SparseDtype(object, far if occupied_fill else np.nan)
    if categorical:
        levels = pd.arrays.SparseArray([far, fine], dtype=dtype)
        source = pd.Categorical.from_codes([0, 1, 0, 1], categories=pd.Index(levels))
    else:
        source = pd.arrays.SparseArray(actual, dtype=dtype)
    for carrier in (source, pd.Series(source), pd.Index(source)):
        for left, right in ((carrier, guesses), (guesses, carrier)):
            summary = report(left, right)
            assert summary["accuracy"] == summary["balanced_accuracy"] == 0.
            observed = {v.isoformat() for v in (actual if left is carrier else guesses)}
            predicted = {v.isoformat() for v in (guesses if left is carrier else actual)}
            expected = {v: (2, 0, 0, 0.) for v in observed}
            expected.update({v: (0, 2, 0, None) for v in predicted})
            assert dict(zip((r["label"] for r in summary["per_class"]), counts(summary))) == expected


@pytest.mark.parametrize("year", [1600, 2000])
def test_sparse_temporal_category_missing_code_keeps_zero_credit(report, year):
    pd = pytest.importorskip("pandas")
    midnight = datetime(year, 1, 1)
    fine = pd.Timestamp("2024-01-02T00:00:00.000000001")
    levels = pd.arrays.SparseArray([midnight, fine, "unused"], dtype=pd.SparseDtype(object, midnight))
    source = pd.Categorical.from_codes([0, 1, -1, 0], categories=pd.Index(levels))
    guesses = np.array([midnight.date(), fine.date(), pd.NaT, midnight.date()], dtype=object)
    summary = report(source, guesses)
    assert summary["accuracy"] == summary["balanced_accuracy"] == 0.
    assert dict(zip((r["label"] for r in summary["per_class"]), counts(summary))) == {
        midnight.isoformat(): (2, 0, 0, 0.), fine.isoformat(): (1, 0, 0, 0.),
        midnight.date().isoformat(): (0, 2, 0, None), fine.date().isoformat(): (0, 1, 0, None),
        None: (1, 1, 0, 0.)}


@pytest.mark.parametrize("year", [1600, 2000])
def test_sparse_temporal_category_preserves_native_units_with_missing_code(report, year):
    pd = pytest.importorskip("pandas")
    day = np.datetime64(f"{year}-01-01", "D")
    fine = np.datetime64("2024-01-02T00:00:00.000000001", "ns")
    levels = pd.arrays.SparseArray(["unused", day, fine], dtype=pd.SparseDtype(object, "fill"))
    source = pd.Categorical.from_codes([1, 2, -1, 1], categories=pd.Index(levels))
    summary = report(source, source)
    assert summary["accuracy"] == .75 and summary["balanced_accuracy"] == 2 / 3
    assert counts(summary) == [(2, 2, 2, 1.), (1, 1, 1, 1.), (1, 1, 0, 0.)]
    assert [r["label"] for r in summary["per_class"]] == [
        {"dtype": str(v.dtype), "ticks": int(v.view("i8"))} for v in (day, fine)] + [None]


@pytest.mark.parametrize("year", [1600, 2000])
@pytest.mark.parametrize("protocol", ["array", "iteration"])
def test_sparse_boxed_temporal_projection_reads_selected_labels_once(report, year, protocol):
    pd = pytest.importorskip("pandas")
    midnight = datetime(year, 1, 1)
    fine = pd.Timestamp("2024-01-02T00:00:00.000000001")
    selected = np.array([midnight, fine], dtype=object)
    calls = []

    class Projected(pd.Series):
        if protocol == "array":
            def __array__(self, dtype=None, copy=None):
                calls.append("array")
                return selected
        else:
            @property
            def __array__(self):
                raise AttributeError

            def __iter__(self):
                calls.append("iteration")
                return iter(selected)

    source = Projected(pd.arrays.SparseArray([7, 8]))
    guesses = np.array([midnight.date(), fine.date()], dtype=object)
    result = report(source, guesses)
    assert calls == [protocol]
    assert result == report(selected, guesses)
    assert result["accuracy"] == result["balanced_accuracy"] == 0.
    assert len(result["per_class"]) == 4


def test_mixed_native_custom_order_uses_chosen_representatives(report):
    class Civil(date):
        pass

    first, second = np.array(["2024-01-01", "2024-01-02"], dtype="M8[D]")
    custom = Civil(2024, 1, 2)
    for observations, supports in [([second, first, custom], [1, 2]),
                                  ([custom, first, second], [2, 1])]:
        rows = np.array(observations, dtype=object)
        result = report(rows, rows)
        assert [r["support"] for r in result["per_class"]] == supports
        assert result["accuracy"] == result["balanced_accuracy"] == 1.


@pytest.mark.parametrize("carrier", [lambda v: np.array(v, dtype=object), sparse])
def test_absorbed_native_date_alias_still_publishes_exact_calendar_classes(report, carrier):
    pd = pytest.importorskip("pandas")

    class Civil(date):
        pass

    class Missing(type(pd.NaT)):
        __hash__ = None

    civil = Civil(2024, 1, 1)
    native = np.datetime64("2024-01-01", "D")
    tomorrow = datetime(2024, 1, 2)
    native_label = {"dtype": "datetime64[D]", "ticks": int(native.view("i8"))}
    for order in permutations((civil, tomorrow, native)):
        # Let the native alias occur only in predictions, only in observations,
        # or on both sides. The calendar day at each position stays the same.
        boxed = tuple(civil if v is native else v for v in order)
        for actual, predicted in ((order, order), (boxed, order), (order, boxed)):
            first_day = next(v for v in actual if v is not tomorrow)
            groups = [(civil.isoformat(), 2), (tomorrow.isoformat(), 1)]
            if first_day is native:
                groups[0] = (native_label, 2)
            elif actual[0] is tomorrow:
                groups.reverse()
            for extra in ((), (pd.NaT,), (Missing(),)):
                result = report(carrier((*actual, *extra)), carrier((*predicted, *extra)))
                assert result["accuracy"] == 3 / (3 + len(extra))
                expected = groups + ([(None, 1)] if extra else [])
                assert [(r["label"], r["support"]) for r in result["per_class"]] == expected
                assert counts(result) == [
                    (n, n, n, 1.) if label is not None else (1, 1, 0, 0.)
                    for label, n in expected]


@pytest.mark.parametrize("truth,guess", [
    (np.array([1], dtype="m8[D]"), np.array([1], dtype="M8[D]")),
    (np.array([1], dtype="m8[M]"), np.array([30], dtype="m8[D]")),
    (np.array([datetime(2024, 1, 1)], dtype=object),
     np.array([datetime(2024, 1, 1, tzinfo=timezone.utc)], dtype=object)),
])
def test_incompatible_domains_are_not_silently_counted(report, truth, guess):
    with pytest.raises(ValueError, match="incompatible units"):
        report(truth, guess)


@pytest.mark.parametrize("a,b", [
    (date(1800, 1, 1), date(2024, 1, 1)),
    (timedelta(days=-999999999), timedelta(days=999999999)),
    (np.datetime64(-(2**63) + 1, "Y"), np.datetime64(2**63 - 1, "Y")),
])
def test_public_temporal_labels_round_trip_in_json(report, a, b):
    result = report(np.array([a, b], dtype=object), np.array([a, a], dtype=object))
    assert counts(result) == [(1, 2, 1, 1.), (1, 0, 0, 0.)]
    assert result["per_class"][0]["label"] != result["per_class"][1]["label"]
    assert result["accuracy"] == result["balanced_accuracy"] == .5


def test_aware_pandas_timestamp_and_python_datetime_use_same_instant(report):
    pd = pytest.importorskip("pandas")
    utc = pd.Timestamp("2024-01-01", tz="UTC")
    near = utc + pd.Timedelta(1, "ns")
    equal = datetime(2023, 12, 31, 19, tzinfo=timezone(-timedelta(hours=5)))
    result = report(np.array([utc, near], dtype=object), np.array([equal, equal], dtype=object))
    assert counts(result) == [(1, 2, 1, 1.), (1, 0, 0, 0.)]
    assert result["per_class"][0]["label"] != result["per_class"][1]["label"]
    assert result["accuracy"] == result["balanced_accuracy"] == .5


@pytest.mark.parametrize("base,earlier,later", [
    (date, (2024, 1, 1), (2024, 1, 2)),
    (datetime, (2024, 1, 1), (2024, 1, 2)),
    (timedelta, (1,), (2,)),
])
@pytest.mark.parametrize("carrier", [lambda v: np.array(v, dtype=object), sparse])
def test_temporal_subclasses_keep_public_scalar_comparisons(report, base, earlier, later, carrier):
    class Derived(base):
        pass

    actual = carrier([base(*earlier), base(*earlier)])
    guesses = carrier([Derived(*earlier), Derived(*later)])
    for truth, predicted, expected_support in [(actual, guesses, [2, 0]), (guesses, actual, [1, 1])]:
        result = report(truth, predicted)
        assert result["accuracy"] == result["balanced_accuracy"] == .5
        assert [r["support"] for r in result["per_class"]] == expected_support
        assert [r["correct"] for r in result["per_class"]] == [1, 0]


def test_active_date_subclass_comparison_is_preserved(report):
    # Its civil-date identity is one day ahead of the value stored by date.
    class Shifted(date):
        @staticmethod
        def comparison_day(value):
            return value.toordinal() + isinstance(value, Shifted)

        def __eq__(self, other):
            return self.comparison_day(self) == self.comparison_day(other)

        def __ne__(self, other):
            return not self == other

        def __lt__(self, other):
            return self.comparison_day(self) < self.comparison_day(other)

        def __gt__(self, other):
            return self.comparison_day(self) > self.comparison_day(other)

    known = np.array([date(2024, 1, 2), date(2024, 1, 4)], dtype=object)
    custom = np.array([Shifted(2024, 1, 1), Shifted(2024, 1, 3)], dtype=object)
    for actual, predicted in ((known, custom), (custom, known)):
        result = report(actual, predicted)
        assert result["accuracy"] == result["balanced_accuracy"] == 1.
        assert [r["correct"] for r in result["per_class"]] == [1, 1]


def test_python_dates_and_native_days_use_scalar_date_equality(report):
    boxed = np.array([date(2024, 1, 1), date(2024, 1, 2)], dtype=object)
    native = np.array(["2024-01-01", "2024-01-03"], dtype="datetime64[D]")
    for actual in (boxed, sparse(boxed)):
        for truth, predicted in ((actual, native), (native, actual)):
            result = report(truth, predicted)
            assert result["accuracy"] == result["balanced_accuracy"] == .5
            assert [r["correct"] for r in result["per_class"]] == [1, 0, 0]
            labels = [r["label"] for r in result["per_class"]]
            dates = [str(np.array(v["ticks"], dtype=v["dtype"])) if isinstance(v, dict) else v for v in labels]
            assert dates == ["2024-01-01", "2024-01-02", "2024-01-03"]
            counts(result)


@pytest.mark.parametrize("carrier", [lambda v: np.array(v, dtype=object), sparse])
@pytest.mark.parametrize("subclass", [False, True])
def test_python_civil_date_coexists_with_week_and_month(report, carrier, subclass):
    class CivilDate(date):
        pass

    jan8 = (CivilDate if subclass else date)(2024, 1, 8)
    month = np.datetime64("2024-01", "M")
    week = np.datetime64("2023-12-28", "W")
    truth, guesses = carrier([jan8, month]), carrier([jan8, week])
    for actual, predicted in ((truth, guesses), (guesses, truth)):
        summary = report(actual, predicted)
        assert summary["accuracy"] == summary["balanced_accuracy"] == .5
        assert sorted(counts(summary), key=lambda r: (r[0], r[1])) == [
            (0, 1, 0, None), (1, 0, 0, 0.), (1, 1, 1, 1.)]


def new_york():
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
    try:
        return ZoneInfo.no_cache("America/New_York")
    except ZoneInfoNotFoundError:
        pytest.skip("America/New_York timezone data is unavailable")


def test_python_datetime_fold_membership_uses_scalar_equality(report):
    early = datetime(2024, 11, 3, 1, 30, fold=0, tzinfo=new_york())
    late = early.replace(fold=1)
    utc_early, utc_late = (d.astimezone(timezone.utc) for d in (early, late))
    assert early == late and early != utc_early and late != utc_late
    truth = np.array([early, late, utc_early, utc_late], dtype=object)
    pred = np.array([utc_early, early, late, utc_late], dtype=object)
    expected = [(2, 2, 1, .5), (1, 1, 0, 0.), (1, 1, 1, 1.)]
    for actual, predicted in ((truth, pred), (pred, truth)):
        result = report(actual, predicted)
        assert result["accuracy"] == result["balanced_accuracy"] == .5
        assert sorted(counts(result)) == sorted(expected)


def test_civil_date_and_python_datetime_cannot_bridge_disjoint_classes(report):
    # Python distinguishes these classes. Treating a common native midnight
    # as equal to both would double count it.
    rows = np.array([date(2024, 1, 1), datetime(2024, 1, 1),
                     np.datetime64("2024-01-01", "D")], dtype=object)
    with pytest.raises(ValueError, match="disjoint report classes"):
        report(rows, rows[::-1])
    result = report(rows[:2], rows[1::-1])
    assert result["accuracy"] == result["balanced_accuracy"] == 0.
    assert counts(result) == [(1, 1, 0, 0.), (1, 1, 0, 0.)]


@pytest.mark.parametrize("carrier", [lambda v: np.array(v, dtype=object), sparse])
def test_python_and_pandas_repeated_hour_must_define_disjoint_classes(report, carrier):
    from itertools import permutations
    pd = pytest.importorskip("pandas")
    early = datetime(2024, 11, 3, 1, 30, fold=0, tzinfo=new_york())
    late = early.replace(fold=1)
    first, second = pd.Timestamp(early), pd.Timestamp(late)
    assert first != second and first == early and early == second
    for order in permutations([early, late, first, second]):
        with pytest.raises(ValueError, match="disjoint report classes"):
            report(carrier(order), carrier(order[::-1]))


def test_distinct_classes_with_the_same_printed_time_cannot_be_published(report):
    early = datetime(2024, 11, 3, 1, 30, tzinfo=new_york())
    other = early.replace(tzinfo=new_york())
    assert early != other and early.isoformat() == other.isoformat()
    truth = np.array([early, other], dtype=object)
    with pytest.raises(ValueError, match="distinct published report labels"):
        report(truth, truth[::-1])


def test_distinct_state_on_a_custom_date_with_identical_text(report):
    class TaggedDate(date):
        def __new__(cls, tag):
            value = date.__new__(cls, 2024, 1, 8)
            value.tag = tag
            return value

        def __eq__(self, other):
            return type(other) is TaggedDate and self.tag == other.tag

        def __ne__(self, other):
            return not self == other

        def __lt__(self, other):
            return self.tag < other.tag if type(other) is TaggedDate else super().__lt__(other)

        def __gt__(self, other):
            return self.tag > other.tag if type(other) is TaggedDate else super().__gt__(other)

    truth = np.array([TaggedDate("A"), TaggedDate("B"), np.datetime64("2024-01-29", "D")], dtype=object)
    with pytest.raises(ValueError, match="distinct published report labels"):
        report(truth, truth[[1, 0, 2]])


def test_writer_preserves_temporal_labels_of_same_passive_subclass(report, tmp_path):
    from search_support import write_json_atomic

    class CivilDate(date):
        pass

    truth = np.array([CivilDate(2024, 1, 1), CivilDate(2024, 1, 2)], dtype=object)
    result = report(truth, truth)
    path = tmp_path / "held-out.json"
    write_json_atomic(path, {"held_out_classification": result})
    saved = json.loads(path.read_text())["held_out_classification"]
    assert saved["accuracy"] == saved["balanced_accuracy"] == 1.
    assert [r["label"]["value"] for r in saved["per_class"]] == ["2024-01-01", "2024-01-02"]
    assert all(r["label"]["type"].endswith(".CivilDate") for r in saved["per_class"])
