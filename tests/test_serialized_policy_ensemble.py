"""Current missing requests and immutable fitted scoring policy.

These regression checks use real production classes. Deleting requested
fields is a synthetic schema probe, not evidence that every historical release
had that schema. The one hash-pinned saved Ensemble is a genuine prior
corrected-baseline object; it is observed without refitting historical work.

Fixed scope: 57 pytest cases, 30 tiny Ensemble fit calls, 60 constituent DWD
fits maximum (M=1). Serialization, cache payload, and historical cases add no
numerical fits. Trusted saved fixtures travel with the standalone test bundle.
"""
from copy import deepcopy
import hashlib
import io
from itertools import combinations
import json
from pathlib import Path
import pickle

import joblib
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline

from dwd.gen_kern_dwd import KernGDWD
from ensemble_dwd import EnsembleKernelDWD, OvOClassifier
import ensemble_dwd.estimator as estimator_module


CURRENT = dict(affine_computation="joint", acceleration="auto",
               residual_check_order="auto")
EXPLICIT_OLD = dict(affine_computation="standard", acceleration=None,
                    residual_check_order="refinement_first")
NAMES = tuple(CURRENT)
MISSING = [names for count in (1, 2, 3) for names in combinations(NAMES, count)]
FORMATS = ("memory", "pickle4", "pickle5", "joblib")
X = np.array([[-1., -.5], [-.75, .25], [-.5, -.75], [-.25, .5],
              [0., -.25], [.25, .75], [.5, -1.], [.75, .5],
              [1., 0.], [1.25, -.5], [1.5, .75], [1.75, .25]])
Y = np.tile([-3, 7], 6)
QUERY = np.array([[-.875, -.375], [.125, .625], [1.375, .5]])


def model(**overrides):
    options = dict(M=1, theta=1., sv_selection_ratio=1., lambd=.1,
                   gamma=.25, max_iter=3, stopping="fixed", random_state=29,
                   prediction_batch_size=None, base_jobs=1)
    options.update(overrides)
    return EnsembleKernelDWD(**options)


def remove_requested(subject, names=NAMES):
    for name in names:
        assert name in vars(subject)
        delattr(subject, name)
    return subject


def roundtrip(subject, fmt):
    if fmt == "memory":
        return subject
    if fmt.startswith("pickle"):
        return pickle.loads(pickle.dumps(subject, protocol=int(fmt[-1])))
    stream = io.BytesIO()
    joblib.dump(subject, stream, compress=0, protocol=5)
    stream.seek(0)
    return joblib.load(stream)


def requests(subject):
    parameters = subject.get_params(deep=False)
    return {name: parameters[name] for name in NAMES}


def concrete(parameters, backend):
    return dict(
        affine_computation=parameters["affine_computation"],
        acceleration=(None if backend == "lbfgs" else "restart")
        if parameters["acceleration"] == "auto" else parameters["acceleration"],
        residual_check_order=("adaptive" if backend in ("auto", "cholesky")
                              else "refinement_first")
        if parameters["residual_check_order"] == "auto"
        else parameters["residual_check_order"],
    )


def learners(subject):
    return [*subject.estimators_, subject.final_estimator_]


def learner_state(subject):
    return [dict(dual_coef=np.asarray(item.dual_coef_).copy(),
                 intercept=np.asarray(item.intercept_).copy(),
                 history=np.asarray(item.objective_history_).copy(),
                 affine=getattr(item, "affine_computation_", "standard"),
                 acceleration=item.acceleration_,
                 residual=item.residual_check_order_)
            for item in learners(subject)]


def assert_same_stored_state(before, after):
    assert len(before) == len(after)
    for a, b in zip(before, after):
        for field in ("dual_coef", "intercept", "history"):
            assert a[field].dtype == b[field].dtype
            assert_array_equal(a[field], b[field])
        for field in ("affine", "acceleration", "residual"):
            assert a[field] == b[field]


def assert_current_fit(subject):
    expected = concrete(CURRENT, subject.backend)
    assert requests(subject) == CURRENT
    assert subject.affine_computation_ == "joint"
    assert subject.requested_acceleration_ == "auto"
    assert subject.requested_residual_check_order_ == "auto"
    assert len(subject.estimators_) == 1
    for item in learners(subject):
        assert {name: getattr(item, name) for name in NAMES} == expected
        assert item.affine_computation_ == "joint"
        assert item.acceleration_ == expected["acceleration"]
        assert item.residual_check_order_ == expected["residual_check_order"]
        assert item.initialization == "zero"


def assert_same_fresh_fit(a, b):
    # Independent fits may round different eigenvectors identically useful
    # ways. This ordinary-fixture comparison does not require eigenbasis byte
    # identity and does not replace any solver acceptance criterion.
    assert_current_fit(a)
    assert_current_fit(b)
    assert_array_equal(a.subset_indices_, b.subset_indices_)
    assert_array_equal(np.sort(a.selected_indices_), np.arange(len(X)))
    assert_array_equal(np.sort(b.selected_indices_), np.arange(len(X)))
    assert_allclose(a.consensus_scores_, b.consensus_scores_, rtol=2e-11, atol=2e-12)
    assert_allclose(a.decision_function(QUERY), b.decision_function(QUERY),
                    rtol=2e-11, atol=2e-12)
    assert_array_equal(a.predict(QUERY), b.predict(QUERY))


@pytest.mark.parametrize("missing", MISSING, ids=lambda names: "+".join(names))
@pytest.mark.parametrize("fmt", FORMATS)
def test_each_missing_subset_uses_current_defaults_without_rewriting_explicit_values(missing, fmt):
    subject = remove_requested(model(**EXPLICIT_OLD), missing)
    restored = roundtrip(subject, fmt)
    expected = {name: CURRENT[name] if name in missing else EXPLICIT_OLD[name]
                for name in NAMES}
    assert requests(restored) == expected
    # A is a class fallback, not a deserialization rewrite.
    assert all(name not in vars(restored) for name in missing)
    for name in set(NAMES) - set(missing):
        assert vars(restored)[name] == EXPLICIT_OLD[name]
    copied = clone(restored)
    assert requests(copied) == expected
    assert all(name in vars(copied) for name in NAMES)
    assert not any(name.endswith("_") for name in vars(copied))
    assert (copied.stopping, copied.max_iter, copied.obj_tol, copied.tol) == (
        "fixed", 3, 1e-5, 1e-6)


@pytest.mark.parametrize("parameters", [EXPLICIT_OLD, CURRENT], ids=["explicit-old", "explicit-current"])
@pytest.mark.parametrize("fmt", FORMATS[1:])
def test_explicit_policies_survive_every_serializer_and_clone(parameters, fmt):
    original = model(**parameters)
    restored = roundtrip(original, fmt)
    assert requests(restored) == parameters
    assert requests(clone(restored)) == parameters
    for name in NAMES:
        assert name in vars(restored)
    assert requests(original) == parameters


@pytest.mark.parametrize("backend", ["auto", "spectral", "lbfgs"])
@pytest.mark.parametrize("fmt", ["memory", "pickle5"])
def test_missing_requests_forward_current_concrete_backend_compatible_controls(monkeypatch, backend, fmt):
    calls = []

    def observe_constructor(**kwargs):
        calls.append(deepcopy(kwargs))
        return KernGDWD(**kwargs)

    monkeypatch.setattr(estimator_module, "KernGDWD", observe_constructor)
    subject = roundtrip(remove_requested(model(backend=backend, **EXPLICIT_OLD)), fmt)
    item = subject._make_estimator({"gamma": .25})
    assert len(calls) == 1
    expected = concrete(CURRENT, backend)
    assert {name: calls[0][name] for name in NAMES} == expected
    assert {name: getattr(item, name) for name in NAMES} == expected
    assert calls[0]["initialization"] == "zero"
    assert calls[0]["backend"] == backend
    assert calls[0]["max_iter"] == 3


@pytest.mark.parametrize("missing", MISSING, ids=lambda names: "+".join(names))
def test_nested_ovo_pipeline_clone_and_grid_materialize_effective_requests(missing):
    template = remove_requested(model(**EXPLICIT_OLD), missing)
    expected = {name: CURRENT[name] if name in missing else EXPLICIT_OLD[name]
                for name in NAMES}
    wrapper = Pipeline([("pair", OvOClassifier(template, n_jobs=1))])
    copied = clone(wrapper)
    assert requests(copied.named_steps["pair"].estimator) == expected
    for name in NAMES:
        assert wrapper.get_params()["pair__estimator__" + name] == expected[name]
    grid = ParameterGrid({"pair__estimator__affine_computation": ["standard", "joint"]})
    for entry in grid:
        variant = clone(wrapper).set_params(**entry)
        desired = {**expected, "affine_computation": entry["pair__estimator__affine_computation"]}
        assert requests(variant.named_steps["pair"].estimator) == desired
    assert requests(template) == expected
    assert all(name not in vars(template) for name in missing)


def test_real_cache_storage_separates_historical_request_keys_from_current_missing_requests(monkeypatch, tmp_path):
    # Real hashing, file checksum, locking and joblib payload operations; the
    # payload is explicitly synthetic and no numerical base stage is called.
    calls = []
    provenance = {"synthetic_unit_contract": "fixed-source-runtime"}
    monkeypatch.setattr(estimator_module, "_runtime_provenance", lambda cls: provenance.copy())

    def base_stage(self, *args):
        item = {"unit_payload": len(calls), "requests": requests(self)}
        calls.append(item)
        return item

    monkeypatch.setattr(EnsembleKernelDWD, "_base_stage", base_stage)

    def get(subject):
        return subject._base_result(X, Y, np.unique(Y), [np.arange(len(X))], {"gamma": .25})

    explicit_old = model(cache_dir=tmp_path, **EXPLICIT_OLD)
    old_value, old_hit, old_key, old_status = get(explicit_old)
    assert (old_hit, old_status) == (False, "miss")
    missing = remove_requested(model(cache_dir=tmp_path, **EXPLICIT_OLD))
    new_value, new_hit, new_key, new_status = get(missing)
    assert (new_hit, new_status) == (False, "miss")
    assert new_key != old_key and new_value != old_value
    matching_value, hit, key, status = get(model(cache_dir=tmp_path))
    assert (hit, key, status) == (True, new_key, "hit")
    assert matching_value == new_value
    old_again, hit, key, status = get(clone(explicit_old))
    assert (hit, key, status) == (True, old_key, "hit")
    assert old_again == old_value
    assert len(calls) == 2
    for mask in MISSING:
        partial = remove_requested(model(cache_dir=tmp_path, **EXPLICIT_OLD), mask)
        explicit = model(cache_dir=tmp_path, **requests(partial))
        first, _, first_key, _ = get(partial)
        second, hit, second_key, _ = get(explicit)
        assert first_key == second_key and hit
        assert first == second
        assert first_key != old_key


@pytest.mark.parametrize("q", [1., 2.3])
@pytest.mark.parametrize("backend,fmt", [("auto", "pickle4"), ("spectral", "pickle5"), ("lbfgs", "joblib")])
def test_real_missing_parent_and_learner_requests_preserve_predictions_then_refit_current(monkeypatch, q, backend, fmt, record_property):
    # Three Ensemble fits per case: explicit-old, missing/refit, fresh-current.
    # Each fit is M=1 plus a final DWD fit. No historical saved fit is replayed.
    old = model(q=q, backend=backend, **EXPLICIT_OLD).fit(X, Y)
    before = learner_state(old)
    scores, labels = old.decision_function(QUERY), old.predict(QUERY)
    remove_requested(old)
    for item in learners(old):
        remove_requested(item)
    restored = roundtrip(old, fmt)
    assert requests(restored) == CURRENT
    assert_same_stored_state(before, learner_state(restored))
    assert_array_equal(restored.decision_function(QUERY), scores)
    assert_array_equal(restored.predict(QUERY), labels)
    assert restored.affine_computation_ == "standard"
    # Requested fields on all children now resolve to current defaults, but
    # their stored fitted affine metadata remains the scoring authority.
    for item in learners(restored):
        assert item.affine_computation == "joint"
        assert item.affine_computation_ == "standard"
    # A genuine standard-scoring state can also predate fitted affine
    # metadata. Its absence remains standard; changing requested defaults
    # must not reinterpret coefficients as a joint-scoring historical fit.
    del restored.affine_computation_
    for item in learners(restored):
        del item.affine_computation_
    restored = roundtrip(restored, fmt)
    assert "affine_computation_" not in vars(restored)
    assert all("affine_computation_" not in vars(item) for item in learners(restored))
    assert requests(restored) == CURRENT
    assert_same_stored_state(before, learner_state(restored))
    assert_array_equal(restored.decision_function(QUERY), scores)
    assert_array_equal(restored.predict(QUERY), labels)
    assert requests(clone(restored)) == CURRENT
    assert not hasattr(clone(restored), "final_estimator_")
    forwarded = []

    def observe_constructor(**kwargs):
        forwarded.append(deepcopy(kwargs))
        return KernGDWD(**kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(estimator_module, "KernGDWD", observe_constructor)
        restored.fit(X, Y)
        fresh = model(q=q, backend=backend).fit(X, Y)
    assert len(forwarded) == 4
    assert forwarded[:2] == forwarded[2:]
    expected = concrete(CURRENT, backend)
    assert all({name: row[name] for name in NAMES} == expected for row in forwarded)
    record_property("ordinary_equivalence", json.dumps(dict(
        q=q, backend=backend, serialization=fmt, ensemble_fits=3, dwd_fits=6,
        restored_scores=restored.decision_function(QUERY).tolist(),
        fresh_scores=fresh.decision_function(QUERY).tolist(),
        restored_iterations=[item.n_iter_ for item in learners(restored)],
        fresh_iterations=[item.n_iter_ for item in learners(fresh)])))
    assert_same_fresh_fit(restored, fresh)


@pytest.mark.parametrize("q", [1., 2.3])
def test_real_cv_from_missing_template_matches_fresh_current_default(q):
    # One candidate, two stratified folds and one selected refit per search:
    # 2 searches * 3 Ensemble fits * 2 constituent DWD fits = 12 per case.
    missing = remove_requested(model(q=q, max_iter=2, **EXPLICIT_OLD))
    if q == 2.3:
        missing = roundtrip(missing, "pickle5")
    searches = []
    for subject in (missing, model(q=q, max_iter=2)):
        search = GridSearchCV(subject, {"lambd": [.1]}, cv=2, n_jobs=1,
                              error_score="raise", return_train_score=True).fit(X, Y)
        searches.append(search)
        assert_current_fit(search.best_estimator_)
    a, b = searches
    assert a.best_params_ == b.best_params_ == {"lambd": .1}
    for key in ("mean_test_score", "mean_train_score", "split0_test_score", "split1_test_score"):
        assert_array_equal(a.cv_results_[key], b.cv_results_[key])
    assert_same_fresh_fit(a.best_estimator_, b.best_estimator_)


def _decode(value):
    if isinstance(value, dict) and set(value) == {"list"}:
        return [_decode(item) for item in value["list"]]
    if isinstance(value, dict) and set(value) == {"float_hex"}:
        return float.fromhex(value["float_hex"])
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    raise ValueError("Unexpected pinned fixture encoding")


def test_genuine_saved_corrected_baseline_ensemble_preserves_explicit_request_and_fitted_readout(monkeypatch, record_property):
    root = Path(__file__).resolve().parent / "fixtures" / "serialized_policy"
    paths = {
        "ensemble-actual.p4": "705389cc165dd59986d2c5f269bf4bb667fda14f0e131b28141a879e7b9d662f",
        "ensemble-actual.p5": "46650bc6ccfa4d430696226bddcd257a884ebd6bae60163e8dfec6f0fa793011",
        "ensemble-actual.input.json": "c50e8f67a69814e2bad2d317e30cd82661448be0e3f5805bdac759d259652170",
    }
    for name, digest in paths.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    artifact = "ensemble-actual.p5"
    try:
        original = pickle.loads((root / artifact).read_bytes())
    except ModuleNotFoundError as error:
        if error.name != "numpy._core.numeric":
            raise
        # NumPy 1.26.4 reads the protocol-4 copy from the same fitted model.
        artifact = "ensemble-actual.p4"
        original = pickle.loads((root / artifact).read_bytes())
    assert type(original) is EnsembleKernelDWD
    assert vars(original)["affine_computation"] == "auto"
    raw = json.loads((root / "ensemble-actual.input.json").read_text(encoding="utf-8"))
    encoded = raw["inputs"]["query"]
    query = np.asarray(_decode(encoded["values"]), dtype=encoded["dtype"]).reshape(encoded["shape"])
    before = learner_state(original)
    scores, labels = original.decision_function(query), original.predict(query)
    assert np.isfinite(scores).all()
    assert requests(clone(original)) == requests(original)
    # An explicit retired alias is not a missing field; retain it and reject
    # a requested new fit clearly, without constructing a numerical learner.
    forbidden = clone(original)
    with monkeypatch.context() as scoped:
        scoped.setattr(estimator_module, "KernGDWD", lambda **kwargs: pytest.fail("retired explicit alias reached a learner"))
        with pytest.raises(ValueError, match="affine_computation.*auto.*removed"):
            forbidden.fit(X, Y)
    remove_requested(original)
    restored = roundtrip(original, "pickle5")
    assert requests(restored) == CURRENT
    assert_same_stored_state(before, learner_state(restored))
    assert_array_equal(restored.decision_function(query), scores)
    assert_array_equal(restored.predict(query), labels)
    for name, digest in paths.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    record_property("saved_fixture_scope", json.dumps(dict(
        origin="prior corrected-baseline Ensemble, not arbitrary old releases",
        fit_calls=0, scores=scores.tolist(), labels=labels.tolist(),
        artifact=artifact, artifact_sha256=paths[artifact], explicit_auto_preserved_until_removed=True)))
