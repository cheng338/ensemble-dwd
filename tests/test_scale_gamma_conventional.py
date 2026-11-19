"""Conventional scale-gamma arithmetic and bounded sklearn agreement checks.

The exact oracle uses the represented float64 variance returned by NumPy. It
does not claim that NumPy's variance equals the exact variance of input data.
The separate scalar panel does not assert that every supplied variance can be
obtained from a real array without NumPy's variance calculation overflowing.
"""

from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.svm import SVC

from ensemble_dwd import EnsembleKernelDWD, EnsembleKernelSVM


UNIT_ROUNDOFF = Fraction(1, 2**53)
MIN_SUBNORMAL = Fraction.from_float(math.ulp(0.0))
FAMILIES = ("dwd", "svm")


def evidence(name, record):
    directory = os.environ.get("IR43_EVIDENCE_DIR")
    if directory:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / (name + ".json")).open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")


def exact_gamma(dimension, variance):
    return Fraction(1, dimension) / Fraction.from_float(variance)


def rational(value):
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def assert_bounded(value, exact):
    assert math.isfinite(value) and value >= 0
    error = abs(Fraction.from_float(value) - exact)
    # Two correctly rounded elementary operations have a normal relative
    # error bounded by 2u/(1-2u). This slightly looser bound also includes two
    # absolute minimum-subnormal units for gradual-underflow rounding in this
    # finite panel. Intermediate overflow is classified separately below.
    allowance = 4 * UNIT_ROUNDOFF * abs(exact) + 2 * MIN_SUBNORMAL
    assert error <= allowance, (value.hex(), rational(error), rational(allowance))
    return {"hex": value.hex(), "absolute_error": rational(error),
            "absolute_allowance": rational(allowance)}


def resolve(family, X, gamma="scale"):
    if family == "dwd":
        model = EnsembleKernelDWD(kernel="rbf", gamma=gamma)
        value, keywords = model._kernel_parameters(X)
        assert keywords["gamma"] == value
    else:
        model = EnsembleKernelSVM(kernel="rbf", gamma=gamma)
        value = model._resolve_gamma(X)
    assert model.gamma is gamma
    return value


def real_array(case):
    if case == "ordinary":
        return np.random.default_rng(43).standard_normal((7, 3))
    if case == "unit-range":
        return np.arange(21, dtype=np.float64).reshape(7, 3) / 20
    if case == "offset":
        return 2.0**30 + np.arange(21, dtype=np.float64).reshape(7, 3) / 8
    if case == "large-finite-variance":
        return np.array([[-2.0**500] * 3, [2.0**500] * 3])
    assert case == "tiny-representable-variance"
    return np.array([[-2.0**-513] * 16, [2.0**-513] * 16])


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("case", ["ordinary", "unit-range", "offset",
                                  "large-finite-variance", "tiny-representable-variance"])
def test_real_array_scale_uses_product_order_and_exact_represented_variance_bound(family, case):
    X = real_array(case)
    original = X.tobytes()
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        variance = float(X.var())
    assert 0 < variance < math.inf
    value = resolve(family, X)
    assert value == 1.0 / (X.shape[1] * variance)
    bound = assert_bounded(value, exact_gamma(X.shape[1], variance))
    assert X.tobytes() == original
    if case == "tiny-representable-variance":
        assert math.isinf(1.0 / variance)
        assert value == 2.0**1022
    evidence("real-" + family + "-" + case,
             {"scope": "actual NumPy array variance and estimator resolver; no full fit",
              "shape": list(X.shape), "input_sha256": hashlib.sha256(original).hexdigest(),
              "variance_hex": variance.hex(), "result": bound})


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("constant", [0.0, 3.5])
def test_true_constant_fallback_is_one(family, constant):
    X = np.full((7, 3), constant)
    assert float(X.var()) == 0.0 and X.min() == X.max()
    assert resolve(family, X) == 1.0


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("kind,amplitude", [
    ("variance-underflow", 2.0**-540),
    ("variance-overflow", 2.0**600),
    ("gamma-overflow", 2.0**-520),
])
def test_real_nonconstant_range_failures_are_rejected(family, kind, amplitude):
    X = np.array([[-amplitude] * 3, [amplitude] * 3])
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        variance = float(X.var())
    assert X.min() != X.max()
    if kind == "variance-underflow":
        assert variance == 0
    elif kind == "variance-overflow":
        assert math.isinf(variance)
    else:
        assert variance > 0 and math.isfinite(variance)
        assert math.isinf(1.0 / (X.shape[1] * variance))
    messages = {
        ("dwd", "variance-underflow"): "Feature variance underflowed;",
        ("dwd", "variance-overflow"): "Feature variance overflowed;",
        ("dwd", "gamma-overflow"): "Scaled gamma overflowed;",
        ("svm", "variance-underflow"): "Feature variance is outside float64 range;",
        ("svm", "variance-overflow"): "Feature variance is outside float64 range;",
        ("svm", "gamma-overflow"): "Scaled gamma is outside float64 range;",
    }
    with pytest.raises(ValueError, match=messages[family, kind]):
        resolve(family, X)
    evidence("refusal-" + family + "-" + kind,
             {"scope": "actual NumPy array; refusal before fitting",
              "variance_hex": variance.hex(), "message_prefix": messages[family, kind]})


class NoVarianceArray(np.ndarray):
    def var(self, *args, **kwargs):
        raise AssertionError("Auto and numeric gamma must not inspect variance")


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("gamma", ["auto", np.float32(0.125)])
def test_auto_and_numeric_gamma_do_not_depend_on_variance(family, gamma):
    X = np.array([[-2.0**600] * 3, [2.0**600] * 3]).view(NoVarianceArray)
    expected = 1.0 / 3 if isinstance(gamma, str) else float(gamma)
    assert resolve(family, X, gamma) == expected


def test_seed43_fixed_panel_has_one_ulp_order_difference_without_mathematical_disagreement():
    rng = np.random.default_rng(43)
    rows = []
    differing = []
    one_ulp = []
    # This is the original fixed, bounded 20-array seed43 panel. Evaluate all
    # members; do not stop or expand a search upon observing a difference.
    for index in range(20):
        X = rng.standard_normal((7, 3))
        variance = float(X.var())
        split = (1.0 / X.shape[1]) / variance
        product = 1.0 / (X.shape[1] * variance)
        assert resolve("dwd", X) == resolve("svm", X) == product
        inverse_first = (1.0 / variance) / X.shape[1]
        exact = exact_gamma(X.shape[1], variance)
        split_bound = assert_bounded(split, exact)
        product_bound = assert_bounded(product, exact)
        inverse_bound = assert_bounded(inverse_first, exact)
        if split != product:
            differing.append(index)
            if abs(split - product) == math.ulp(product):
                one_ulp.append(index)
        rows.append({"index": index, "input_sha256": hashlib.sha256(X.tobytes()).hexdigest(),
                     "variance_hex": variance.hex(), "split": split_bound,
                     "product": product_bound, "inverse_first": inverse_bound})
    assert one_ulp, "The original seed43 one-ULP counterexample was not reproduced"
    evidence("seed43-fixed-panel", {"scope": "actual NumPy arrays and both resolvers",
                                    "different_indices": differing,
                                    "one_ulp_indices": one_ulp, "rows": rows})


@pytest.mark.parametrize("case", ["ordinary", "unit-range", "rounding-witness"])
def test_actual_sklearn_svc_resolves_identical_numeric_gamma_on_same_full_fold(case):
    if case == "rounding-witness":
        # Index 3 is a retained one-ULP witness in the original seed43 panel
        # on both approved hosts. Generate this fixed index, not a new search.
        rng = np.random.default_rng(43)
        for _ in range(4):
            X = rng.standard_normal((7, 3))
    else:
        X = real_array(case)
    y = np.where(np.arange(len(X)) % 2, 1, -1)
    original = X.tobytes()
    variance = float(X.var())
    expected = 1.0 / (X.shape[1] * variance)
    split = (1.0 / X.shape[1]) / variance
    if case == "rounding-witness":
        assert abs(expected - split) == math.ulp(expected)
    model = SVC(kernel="rbf", gamma="scale", C=1.0, cache_size=16.0).fit(X, y)
    assert model.fit_status_ == 0
    assert np.isfinite(model.decision_function(X)).all()
    assert float(model._gamma).hex() == expected.hex()
    assert resolve("dwd", X).hex() == expected.hex()
    assert resolve("svm", X).hex() == expected.hex()
    assert X.tobytes() == original
    evidence("actual-svc-" + case,
             {"scope": "actual sklearn SVC fit; ensemble gamma resolvers on same full fold",
              "input_sha256": hashlib.sha256(original).hexdigest(),
              "variance_hex": variance.hex(), "product_hex": expected.hex(),
              "split_hex": split.hex(), "sklearn_gamma_hex": float(model._gamma).hex(),
              "sklearn_fit_status": int(model.fit_status_)})


@pytest.mark.parametrize("case", ["dwd-linear", "dwd-callable", "svm-linear"])
def test_unused_gamma_does_not_read_variance_or_change_kernel_keywords(case):
    X = np.array([[-2.0**600] * 3, [2.0**600] * 3]).view(NoVarianceArray)
    if case == "svm-linear":
        model = EnsembleKernelSVM(kernel="linear", gamma="scale")
        assert model._resolve_gamma(X) == 1.0
    else:
        kernel = "linear" if case == "dwd-linear" else lambda A, B: A @ B.T
        keywords = {"sentinel": 5}
        model = EnsembleKernelDWD(kernel=kernel, gamma="scale", kernel_kws=keywords)
        value, resolved = model._kernel_parameters(X)
        assert value is None and resolved == keywords
        assert resolved is not keywords
        assert keywords == {"sentinel": 5}
    assert model.gamma == "scale"


SCALAR_PANEL = [
    ("reciprocal-overflow", 16, 2.0**-1026),
    ("subnormal-one", 1, sys.float_info.max),
    ("product-overflow-three", 3, sys.float_info.max),
    ("unrepresentable-exact-result", 1, math.ulp(0.0)),
]


@pytest.mark.parametrize("name,dimension,variance", SCALAR_PANEL,
                         ids=[case[0] for case in SCALAR_PANEL])
def test_scalar_orders_against_exact_oracle_with_explicit_overflow_classification(name, dimension, variance):
    exact = exact_gamma(dimension, variance)
    try:
        nearest = float(exact)
    except OverflowError:
        nearest = math.inf
    product_denominator = dimension * variance
    reciprocal = 1.0 / variance
    values = {"split": (1.0 / dimension) / variance,
              "product": 1.0 / product_denominator,
              "inverse_first": reciprocal / dimension}
    outcomes = {}
    for order, value in values.items():
        if math.isinf(nearest):
            assert math.isinf(value)
            outcome = {"classification": "exact result outside float64 range", "hex": value.hex()}
        elif order == "inverse_first" and math.isinf(reciprocal):
            assert math.isinf(value) and math.isfinite(values["split"])
            outcome = {"classification": "avoidable intermediate reciprocal overflow", "hex": value.hex()}
        elif order == "product" and math.isinf(product_denominator) and nearest != 0:
            assert value == 0.0 and values["split"] > 0.0
            outcome = {"classification": "intermediate product overflow loses representable result",
                       "hex": value.hex()}
        else:
            outcome = {"classification": "within conservative rounding allowance",
                       **assert_bounded(value, exact)}
        outcomes[order] = outcome
    evidence("scalar-" + name,
             {"scope": "scalar arithmetic only; not a claim of accepted X.var or a full fit",
              "dimension": dimension, "variance_hex": variance.hex(),
              "exact_gamma": rational(exact), "nearest_float_hex": nearest.hex(),
              "outcomes": outcomes, "universal_best_order_claim": False})
