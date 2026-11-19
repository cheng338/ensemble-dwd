"""No-fit lifecycle experiments for the rejected alternatives and candidate A."""
import pickle
from contextlib import contextmanager

import pytest
from sklearn.base import BaseEstimator, clone
from ensemble_dwd.estimator import EnsembleKernelDWD

CURRENT = dict(affine_computation='joint', acceleration='auto', residual_check_order='auto')
OLD = dict(affine_computation='standard', acceleration=None, residual_check_order='refinement_first')
CLASSES = [EnsembleKernelDWD]

def missing(cls):
    obj = cls()
    for key in CURRENT:
        del obj.__dict__[key]
    return obj

def policies(obj):
    return {key: obj.get_params(deep=False)[key] for key in CURRENT}

def migrate_on_load_only(self, state):
    BaseEstimator.__setstate__(self, state)
    for key, value in CURRENT.items():
        self.__dict__.setdefault(key, value)

def reject_missing_requests(obj):
    absent = sorted(set(CURRENT) - set(obj.__dict__))
    if absent:
        raise ValueError('Explicit migration required: ' + ', '.join(absent))

@pytest.mark.parametrize('cls', CLASSES)
def test_baseline_fallback_materializes_in_clone(cls, monkeypatch):
    # Reproduce the exact baseline class attributes without invoking any solver.
    for key, value in OLD.items():
        monkeypatch.setattr(cls, key, value)
    obj = missing(cls)
    assert policies(obj) == OLD
    copied = clone(obj)
    assert policies(copied) == OLD
    assert all(key in copied.__dict__ for key in OLD)

@pytest.mark.parametrize('cls', CLASSES)
def test_a_covers_live_clone_and_deserialized_objects(cls):
    obj = missing(cls)
    assert policies(obj) == CURRENT
    assert policies(clone(obj)) == CURRENT
    assert policies(pickle.loads(pickle.dumps(obj, protocol=5))) == CURRENT
    assert not set(CURRENT).intersection(obj.__dict__)

@pytest.mark.parametrize('cls', CLASSES)
def test_b_load_only_migration_misses_live_object_and_prior_clone(cls, monkeypatch):
    for key, value in OLD.items():
        monkeypatch.setattr(cls, key, value)
    monkeypatch.setattr(cls, '__setstate__', migrate_on_load_only)
    obj = missing(cls)
    prior_clone = clone(obj)
    assert policies(obj) == OLD
    assert policies(prior_clone) == OLD
    loaded = pickle.loads(pickle.dumps(obj, protocol=5))
    assert policies(loaded) == CURRENT
    # Already materialized values are indistinguishable from explicit choices.
    assert policies(pickle.loads(pickle.dumps(prior_clone, protocol=5))) == OLD
    assert not set(CURRENT).intersection(obj.__dict__)

@pytest.mark.parametrize('cls', CLASSES)
def test_c_rejection_adds_a_barrier_where_current_defaults_are_defined(cls):
    obj = missing(cls)
    with pytest.raises(ValueError, match='Explicit migration required'):
        reject_missing_requests(obj)
    assert policies(obj) == CURRENT
    explicit = cls(**OLD)
    reject_missing_requests(explicit)
    assert policies(explicit) == OLD

@pytest.mark.parametrize('cls', CLASSES)
def test_a_does_not_guess_intent_from_previously_materialized_choices(cls, monkeypatch):
    with monkeypatch.context() as patch:
        for key, value in OLD.items():
            patch.setattr(cls, key, value)
        previous_clone = clone(missing(cls))
    explicit = cls(**OLD)
    assert policies(previous_clone) == policies(explicit) == OLD
    assert policies(clone(previous_clone)) == OLD

@pytest.mark.parametrize('cls', CLASSES)
def test_a_preserves_an_explicit_subclass_class_policy(cls):
    custom = type('ExplicitPolicySubclass', (cls,), dict(OLD))
    obj = missing(custom)
    assert policies(obj) == OLD
    assert policies(clone(obj)) == OLD
