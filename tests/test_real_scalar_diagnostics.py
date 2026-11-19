"""Named unsupported-scalar errors without widening scalar or sizing support."""
from decimal import Decimal
from fractions import Fraction
from numbers import Real
from types import SimpleNamespace

import numpy as np
import pytest

from ensemble_dwd import EnsembleKernelDWD, EnsembleKernelSVM
from ensemble_dwd import estimator as dwd_module
from ensemble_dwd import svm as svm_module


FAMILIES = [(EnsembleKernelDWD, dwd_module), (EnsembleKernelSVM, svm_module)]
UNSUPPORTED = [Fraction(1, 2), Decimal("0.5"), Decimal("NaN")]


def named_message(name, value):
    return f"{name} must be a supported real scalar; got {type(value).__name__}."


def refuse_before_work(factory, module, name, value, monkeypatch, tmp_path):
    options = dict(M=1, theta=0.5, sv_selection_ratio=0.5, kernel="linear",
                   random_state=19, cache_dir=str(tmp_path / "uncreated-cache"))
    options[name] = value
    model = factory(**options)
    assert getattr(model, name) is value
    assert model.get_params(deep=False)[name] is value
    reached = []

    def forbidden(*args, **kwargs):
        reached.append("sampling/cache/base/final work")
        raise AssertionError("An invalid scalar reached downstream work")

    for attr in ("_sample_subsets", "_cached_base_stage", "_runtime_provenance"):
        monkeypatch.setattr(module, attr, forbidden)
    for attr in ("_base_result", "_base_stage", "_make_estimator"):
        monkeypatch.setattr(model, attr, forbidden)
    X = np.arange(16, dtype=float).reshape(8, 2)
    y = np.arange(8) % 2
    with pytest.raises(ValueError) as captured:
        model.fit(X, y)
    assert type(captured.value) is ValueError
    assert str(captured.value) == named_message(name, value)
    if isinstance(value, Fraction):
        assert isinstance(captured.value.__cause__, TypeError)
    else:
        assert captured.value.__cause__ is None
    assert reached == []
    assert not (tmp_path / "uncreated-cache").exists()
    assert not hasattr(model, "final_estimator_")
    assert getattr(model, name) is value
    assert model.get_params(deep=False)[name] is value


@pytest.mark.parametrize("factory,module", FAMILIES, ids=["dwd", "svm"])
@pytest.mark.parametrize("name", ["theta", "sv_selection_ratio"])
@pytest.mark.parametrize("value", UNSUPPORTED, ids=["fraction", "decimal", "decimal-nan"])
def test_public_ratio_errors_name_parameter_before_work(factory, module, name, value,
                                                       monkeypatch, tmp_path):
    refuse_before_work(factory, module, name, value, monkeypatch, tmp_path)


@pytest.mark.parametrize("name", ["lambd", "q", "tol", "obj_tol", "gamma"])
@pytest.mark.parametrize("value", [Fraction(1, 2), Decimal("NaN")], ids=["fraction", "decimal-nan"])
def test_shared_dwd_scalar_options_have_named_errors_before_work(name, value, monkeypatch, tmp_path):
    refuse_before_work(EnsembleKernelDWD, dwd_module, name, value, monkeypatch, tmp_path)


@pytest.mark.parametrize("value", [True, np.bool_(True), complex(0.5), "0.5", None, np.array(0.5)],
                         ids=["bool", "numpy-bool", "complex", "str", "none", "zero-d-array"])
def test_non_real_rejection_does_not_call_isfinite(monkeypatch, value):
    def forbidden(value):
        raise AssertionError("Unsupported type reached numpy.isfinite")

    monkeypatch.setattr(dwd_module, "np", SimpleNamespace(bool_=np.bool_, isfinite=forbidden))
    with pytest.raises(ValueError) as captured:
        dwd_module._finite_real(value, "theta")
    assert str(captured.value) == named_message("theta", value)
    assert captured.value.__cause__ is None


class PlainFloat(float):
    """An otherwise ordinary float subclass remains supported."""


@pytest.mark.parametrize("value", [1, 0.5, np.int64(1), np.uint64(1), np.float16(0.5),
                                   np.float32(0.5), np.float64(0.5), np.longdouble(0.5), PlainFloat(0.5)],
                         ids=["int", "float", "numpy-int", "numpy-uint", "float16", "float32",
                              "float64", "longdouble", "float-subclass"])
def test_supported_real_scalars_keep_existing_float_result(value):
    result = dwd_module._finite_real(value, "lambd")
    assert type(result) is float
    assert result == float(value)


@pytest.mark.parametrize("value,allow_zero,accepted", [
    (-0.0, False, False), (-0.0, True, True), (-0.5, True, False),
    (np.nan, False, False), (np.inf, True, False),
], ids=["negative-zero-positive", "negative-zero-allowed", "negative", "nan", "infinity"])
def test_numeric_domain_messages_and_signed_zero_are_unchanged(value, allow_zero, accepted):
    if accepted:
        result = dwd_module._finite_real(value, "obj_tol", allow_zero=allow_zero)
        assert result == 0.0 and np.signbit(result)
    else:
        with pytest.raises(ValueError) as captured:
            dwd_module._finite_real(value, "obj_tol", allow_zero=allow_zero)
        domain = "nonnegative" if allow_zero else "positive"
        assert str(captured.value) == f"obj_tol must be a finite {domain} number."
        assert captured.value.__cause__ is None


@pytest.mark.parametrize("factory,name,accepted", [
    (EnsembleKernelDWD, "lambd", False), (EnsembleKernelDWD, "q", False),
    (EnsembleKernelDWD, "tol", False), (EnsembleKernelDWD, "obj_tol", True),
    (EnsembleKernelDWD, "gamma", True), (EnsembleKernelSVM, "gamma", False),
    (EnsembleKernelSVM, "coef0", True),
], ids=["dwd-lambda", "dwd-q", "dwd-tol", "dwd-obj-tol", "dwd-gamma", "svm-gamma", "svm-coef0"])
def test_public_options_keep_existing_zero_domains(factory, name, accepted):
    value = np.float32(0)
    model = factory(**{name: value})
    if accepted:
        assert model._validate_options() is None
    else:
        with pytest.raises(ValueError) as captured:
            model._validate_options()
        assert str(captured.value) == f"{name} must be a finite positive number."
    assert getattr(model, name) is value
    assert model.get_params(deep=False)[name] is value


@pytest.mark.parametrize("name", ["lambd", "q", "tol", "obj_tol", "gamma"])
def test_successful_dwd_option_validation_preserves_original_scalar(name):
    value = np.float32(0.5)
    model = EnsembleKernelDWD(**{name: value})
    assert model._validate_options() is None
    assert getattr(model, name) is value
    assert model.get_params(deep=False)[name] is value


def probe_real(monkeypatch, phase=None, error=None, finite_result=True):
    """Exercise the exception boundary without changing numpy process-wide."""
    calls = []

    def step(name):
        calls.append(name)
        if name == phase:
            raise error

    class Probe:
        def __lt__(self, other):
            step("lt")
            return False

        def __eq__(self, other):
            step("eq")
            return False

        def __gt__(self, other):
            step("gt")
            return False

        def __float__(self):
            step("float")
            return 0.5

    # This test-local class is the only new virtual Real; production ABC rules
    # and the package's imported Real object are not replaced.
    Real.register(Probe)
    value = Probe()

    def isfinite(actual):
        assert actual is value
        step("isfinite")
        return finite_result

    monkeypatch.setattr(dwd_module, "np", SimpleNamespace(bool_=np.bool_, isfinite=isfinite))
    return value, calls


def test_only_isfinite_type_error_is_named_and_retains_cause(monkeypatch):
    error = TypeError("specific finiteness-dispatch failure")
    value, calls = probe_real(monkeypatch, "isfinite", error)
    with pytest.raises(ValueError) as captured:
        dwd_module._finite_real(value, "q")
    assert str(captured.value) == named_message("q", value)
    assert captured.value.__cause__ is error
    assert calls == ["isfinite"]


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError, KeyboardInterrupt, SystemExit])
def test_non_type_error_from_isfinite_propagates_unchanged(monkeypatch, error_type):
    error = error_type("finiteness runtime failure")
    value, calls = probe_real(monkeypatch, "isfinite", error)
    with pytest.raises(error_type) as captured:
        dwd_module._finite_real(value, "tol")
    assert captured.value is error
    assert calls == ["isfinite"]


@pytest.mark.parametrize("phase,error_type", [
    ("lt", TypeError), ("eq", TypeError), ("float", TypeError), ("float", ValueError),
    ("lt", RuntimeError), ("eq", KeyboardInterrupt), ("float", KeyboardInterrupt),
], ids=["lt-type", "eq-type", "float-type", "float-value", "lt-runtime", "eq-interrupt", "float-interrupt"])
def test_comparison_and_float_conversion_errors_are_not_relabelled(monkeypatch, phase, error_type):
    error = error_type("failure outside finiteness dispatch")
    value, calls = probe_real(monkeypatch, phase, error)
    with pytest.raises(error_type) as captured:
        dwd_module._finite_real(value, "lambd")
    assert captured.value is error
    ordered = ["isfinite", "lt", "eq", "float"]
    assert calls == ordered[:ordered.index(phase) + 1]


def test_finite_result_truth_error_is_outside_the_conversion_catch(monkeypatch):
    error = TypeError("unexpected finiteness result")

    class TruthFailure:
        def __bool__(self):
            raise error

    value, calls = probe_real(monkeypatch, finite_result=TruthFailure())
    with pytest.raises(TypeError) as captured:
        dwd_module._finite_real(value, "gamma")
    assert captured.value is error
    assert calls == ["isfinite"]


def test_fraction_upper_bound_comparison_error_remains_original(monkeypatch):
    error = TypeError("failure in original scalar upper-bound comparison")
    value, calls = probe_real(monkeypatch, "gt", error)
    with pytest.raises(TypeError) as captured:
        dwd_module._fraction(value, "theta")
    assert captured.value is error
    assert calls == ["isfinite", "lt", "eq", "float", "gt"]


def test_huge_python_integer_is_reported_as_unsupported_scalar():
    value = 10 ** 100
    with pytest.raises(ValueError) as captured:
        dwd_module._finite_real(value, "lambd")
    assert str(captured.value) == named_message("lambd", value)
    assert isinstance(captured.value.__cause__, TypeError)
