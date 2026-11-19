"""Binary64 ratio sizing without changing public parameter identities."""

from fractions import Fraction
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone

from dwd.utils import parameters_equal
from ensemble_dwd import EnsembleKernelDWD, EnsembleKernelSVM
from ensemble_dwd import estimator as dwd_module
from ensemble_dwd import svm as svm_module


FAMILIES = [(EnsembleKernelDWD, dwd_module), (EnsembleKernelSVM, svm_module)]


def binary64_count(n, ratio):
    """Round an independent exact product once, then take its floor.

    Conversion of the population to binary64 is part of ordinary Python
    int-times-float multiplication, including the simulated n > 2**53 cases.
    This oracle does not call either NumPy multiplication or the sizing helper.
    """
    product = Fraction.from_float(float(n)) * Fraction.from_float(float(ratio))
    return math.floor(float(product))


ARITHMETIC = [
    (10, .3), (100, .29), (20, .5), (17, 1),
    (10, np.float16(.2)), (25, np.float16(.2)),
    (10, np.float32(.7)), (100, np.float32(.7)),
    (200, np.float32(.7)), (200, np.float16(.1)),
    (2051, np.float16(.5)), (65536, np.float16(.5)),
    (65536, np.float16(1)), (257, np.int8(1)),
    (257, np.uint8(1)), (100, np.float64(.29)),
    (10, np.nextafter(.3, 0.)), (10, np.nextafter(.3, 1.)),
    (100, np.nextafter(.29, 0.)), (100, np.nextafter(.29, 1.)),
    (31, np.nextafter(1., 0.)), (31, np.nextafter(0., 1.)),
    (2**24 + 1, np.float32(1)), (2**24 + 3, np.float32(.5)),
    (2**53 + 1, 1.), (2**53 + 3, .5), (2**63 - 1, .5),
]


@pytest.mark.parametrize("n,ratio", ARITHMETIC,
                         ids=[f"case-{i:02d}" for i in range(len(ARITHMETIC))])
def test_counts_follow_binary64_product(n, ratio):
    # Huge populations exercise arithmetic only; no corresponding array exists.
    assert dwd_module._fraction_rows(n, ratio, "theta") == binary64_count(n, ratio)


def test_python_float_semantics_and_fixed_promotion_boundaries():
    # Exact rational flooring would change 10 * .3 to 2; preserve Python's 3.
    assert dwd_module._fraction_rows(10, .3, "theta") == 3
    assert dwd_module._fraction_rows(100, .29, "theta") == 28
    assert dwd_module._fraction_rows(10, np.float32(.7), "theta") == 6
    assert dwd_module._fraction_rows(10, np.float16(.2), "theta") == 1
    assert dwd_module._fraction_rows(2051, np.float16(.5), "theta") == 1025


class ReachedFinal(Exception):
    """Stop after selecting the actual rows, before any numerical learner."""


class IdentityPermutation:
    def permutation(self, n):
        return np.arange(n)


def setup_probe(factory, module, n, M, theta, ratio, monkeypatch,
                *, labels=None, previous_fit=False):
    X = np.column_stack((np.arange(n, dtype=np.float64), np.zeros(n)))
    y = np.arange(n) % 2 if labels is None else np.asarray(labels)
    model = factory(M=M, theta=theta, sv_selection_ratio=ratio,
                    kernel="linear", random_state=19, cache_dir=None)
    original = {"theta": theta, "sv_selection_ratio": ratio}
    observed = {"sample_calls": 0, "base_calls": 0, "final_calls": 0}
    sampler = module._sample_subsets

    def sample(encoded, size, count, rng):
        observed["sample_calls"] += 1
        observed["subset_size"] = int(size)
        subsets = sampler(encoded, size, count, rng)
        observed["subset_sizes"] = [len(rows) for rows in subsets]
        observed["subset_rows"] = [rows.tolist() for rows in subsets] if n < 300 else None
        return subsets

    def base(*args):
        observed["base_calls"] += 1
        return ({"order": np.arange(n), "consensus": np.zeros(n),
                 "distances": np.ones(n)}, False, None, "disabled")

    class Final:
        def fit(self, selected_X, selected_y):
            observed["final_calls"] += 1
            observed["selected_size"] = len(selected_y)
            observed["selected_rows"] = selected_X[:, 0].astype(int).tolist() if n < 300 else None
            raise ReachedFinal("selection boundary reached without numerical fit")

    if previous_fit:
        model.final_estimator_ = object()
        model.selected_indices_ = np.array([0, 1])
    with monkeypatch.context() as context:
        context.setattr(module, "_private_random_state", lambda seed: IdentityPermutation())
        context.setattr(module, "_sample_subsets", sample)
        context.setattr(model, "_base_result", base)
        context.setattr(model, "_make_estimator", lambda *args, **kwargs: Final())
        try:
            model.fit(X, y)
        except ReachedFinal:
            observed["outcome"] = "final_boundary"
        except (ValueError, TypeError, OverflowError, FloatingPointError) as error:
            observed.update(outcome=type(error).__name__, message=str(error))
        else:
            raise AssertionError("A setup probe unexpectedly completed numerical work")
    for key, value in original.items():
        assert getattr(model, key) is value
        assert model.get_params(deep=False)[key] is value
    assert not hasattr(model, "final_estimator_")
    assert not hasattr(model, "selected_indices_")
    return observed, model


PROBES = [
    (10, 1, np.float32(.7), 1.),
    (100, 1, .25, np.float32(.7)),
    (200, 1, np.float16(.1), np.float16(.7)),
    (10, 1, .5, np.float16(.2)),
    (2051, 2, np.float16(.5), 1.),
    (65536, 1, np.float16(1), np.float16(1)),
    (10, 1, .3, .3),
    (100, 1, .25, .29),
    (20, 1, np.nextafter(0., 1.), 1.),
]


@pytest.mark.parametrize("factory,module", FAMILIES, ids=["dwd", "svm"])
@pytest.mark.parametrize("n,M,theta,ratio", PROBES,
                         ids=[f"route-{i:02d}" for i in range(len(PROBES))])
def test_public_setup_uses_current_population(factory, module, n, M, theta, ratio,
                                             monkeypatch):
    actual, _ = setup_probe(factory, module, n, M, theta, ratio, monkeypatch,
                            previous_fit=True)
    retained = binary64_count(n, ratio)
    subset = max(2, binary64_count(n, theta))
    if retained < 2:
        assert actual["outcome"] == "IneligibleSelectionError"
        assert actual["sample_calls"] == actual["base_calls"] == actual["final_calls"] == 0
    else:
        assert actual["outcome"] == "final_boundary"
        assert actual["sample_calls"] == actual["base_calls"] == actual["final_calls"] == 1
        assert actual["subset_size"] == subset
        assert actual["selected_size"] == retained
        assert actual["subset_sizes"] == [subset] * M
        if n < 300:
            assert actual["subset_rows"] == [list(range(i * subset, (i + 1) * subset))
                                              for i in range(M)]
            assert actual["selected_rows"] == list(range(retained))


INVALID = [True, np.bool_(True), 0., -0., -.1, 1.1, np.nan, np.inf, -np.inf]


@pytest.mark.parametrize("factory,module", FAMILIES, ids=["dwd", "svm"])
@pytest.mark.parametrize("name", ["theta", "sv_selection_ratio"])
@pytest.mark.parametrize("value", INVALID, ids=[f"invalid-{i}" for i in range(len(INVALID))])
def test_invalid_ratio_refuses_before_sampling_cache_or_fit(factory, module, name,
                                                           value, monkeypatch):
    kwargs = {"theta": .5, "ratio": .5}
    kwargs["theta" if name == "theta" else "ratio"] = value
    actual, _ = setup_probe(factory, module, 10, 1, monkeypatch=monkeypatch, **kwargs)
    assert actual["outcome"] in ("ValueError", "TypeError")
    assert actual["sample_calls"] == actual["base_calls"] == actual["final_calls"] == 0


@pytest.mark.parametrize("factory,module", FAMILIES, ids=["dwd", "svm"])
@pytest.mark.parametrize("boundary", ["above_one", "tiny_positive"])
def test_longdouble_domain_precedes_narrowing(factory, module, boundary, monkeypatch):
    value = (np.nextafter(np.longdouble(1), np.longdouble(2)) if boundary == "above_one"
             else np.nextafter(np.longdouble(0), np.longdouble(1)))
    # Works whether longdouble has the same or greater precision than float64.
    if boundary == "above_one":
        assert value > 1
        actual, _ = setup_probe(factory, module, 10, 1, value, 1., monkeypatch)
        assert actual["outcome"] == "ValueError"
        assert actual["sample_calls"] == actual["base_calls"] == actual["final_calls"] == 0
    else:
        assert value > 0
        actual, _ = setup_probe(factory, module, 10, 1, value, 1., monkeypatch)
        assert actual["outcome"] == "final_boundary"
        assert actual["subset_size"] == 2
        selection, _ = setup_probe(factory, module, 10, 1, .5, value, monkeypatch)
        assert selection["outcome"] == "IneligibleSelectionError"
        assert selection["sample_calls"] == selection["base_calls"] == selection["final_calls"] == 0


@pytest.mark.parametrize("factory,module", FAMILIES, ids=["dwd", "svm"])
def test_validate_options_override_still_returns_none(factory, module, monkeypatch):
    class Custom(factory):
        validations = 0

        def _validate_options(self):
            self.validations += 1
            assert super()._validate_options() is None
            return None

    actual, model = setup_probe(Custom, module, 10, 1, np.float32(.7), 1., monkeypatch)
    assert actual["outcome"] == "final_boundary"
    assert model.validations == 1


@pytest.mark.parametrize("factory,module", FAMILIES, ids=["dwd", "svm"])
def test_single_class_selection_is_not_repaired(factory, module, monkeypatch):
    actual, _ = setup_probe(factory, module, 10, 1, 1., .3, monkeypatch,
                            labels=np.array([0] * 5 + [1] * 5))
    assert actual["outcome"] == "IneligibleSelectionError"
    assert actual["sample_calls"] == actual["base_calls"] == 1
    assert actual["final_calls"] == 0


@pytest.mark.parametrize("factory,module", FAMILIES, ids=["dwd", "svm"])
@pytest.mark.parametrize("value", [np.float16(.5), np.float32(.7), np.float64(.3)])
def test_parameter_types_and_cache_identity_survive(factory, module, value):
    model = factory(theta=value, sv_selection_ratio=value)
    copied = clone(model)
    assert model.theta is value and model.sv_selection_ratio is value
    assert type(copied.theta) is type(value)
    assert type(copied.sv_selection_ratio) is type(value)
    assert copied.theta == value and copied.sv_selection_ratio == value
    assert not parameters_equal(value, float(value))
    assert not parameters_equal({"theta": value}, {"theta": float(value)})
    assert model.set_params(theta=value, sv_selection_ratio=value) is model
    assert model.theta is value and model.sv_selection_ratio is value


def real_data(n):
    permutation = np.random.RandomState(19).permutation(n)
    y = np.empty(n, dtype=int)
    y[permutation] = np.arange(n) % 2
    signed = 2 * y - 1
    X = np.column_stack((signed * (1. + (np.arange(n) % 7) / 32.),
                         (np.arange(n) % 5) / 32.))
    return X, y


@pytest.mark.parametrize("factory,module", FAMILIES, ids=["dwd", "svm"])
@pytest.mark.parametrize("n,theta,ratio", [
    (10, np.float32(.7), 1.),
    (100, .25, np.float32(.7)),
    (200, np.float16(.1), np.float16(.7)),
], ids=["ten-subset", "hundred-selection", "twohundred-subset"])
def test_real_public_fits_match_normalized_counterpart(factory, module, n, theta, ratio):
    X, y = real_data(n)
    before_X, before_y = X.copy(), y.copy()
    error_policy, error_callback = np.geterr(), np.geterrcall()
    options = dict(M=1, theta=theta, sv_selection_ratio=ratio,
                   kernel="rbf", gamma=.2, random_state=19, cache_dir=None)
    if factory is EnsembleKernelDWD:
        # Fixed updates test ratio parity, not optimizer convergence.
        options.update(lambd=.1, q=1., stopping="fixed", max_iter=8)
    else:
        options.update(C=1., tol=1e-6, max_iter=-1, cache_size=32.)
    model = factory(**options)
    counterpart = clone(model).set_params(theta=float(theta), sv_selection_ratio=float(ratio))
    model.fit(X, y)
    counterpart.fit(X, y)
    for fitted in (model, counterpart):
        assert len(fitted.subset_indices_[0]) == max(2, binary64_count(n, theta))
        assert len(fitted.selected_indices_) == binary64_count(n, ratio)
    assert_array_equal(model.subset_indices_[0], counterpart.subset_indices_[0])
    assert_array_equal(model.selected_indices_, counterpart.selected_indices_)
    assert_allclose(model.consensus_scores_, counterpart.consensus_scores_, rtol=1e-12, atol=1e-13)
    scores = model.decision_function(X)
    normalized_scores = counterpart.decision_function(X)
    assert np.isfinite(scores).all() and np.isfinite(normalized_scores).all()
    assert_allclose(scores, normalized_scores, rtol=1e-12, atol=1e-13)
    assert_array_equal(model.predict(X), counterpart.predict(X))
    assert model.theta is theta and model.sv_selection_ratio is ratio
    assert type(model.get_params()["theta"]) is type(theta)
    assert type(clone(model).theta) is type(theta)
    assert_array_equal(X, before_X)
    assert_array_equal(y, before_y)
    assert np.geterr() == error_policy and np.geterrcall() is error_callback
