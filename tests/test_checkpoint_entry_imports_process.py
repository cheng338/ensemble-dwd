"""Real loky entry must not multiply estimator-bearing probe hooks.

These adversarial first-call-only hooks detect repeated serialization or worker
inspection without fitting models. Neither joblib nor runtime fingerprints are
replaced. Two submitted tasks do not prove two distinct workers executed them.
"""
import json
import os
from pathlib import Path
import uuid

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from threadpoolctl import threadpool_info

from test_execution_runtime_roles import cp


class ProcessHookClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, journal, origin_pid, hook, random_state=31):
        self.journal = journal
        self.origin_pid = origin_pid
        self.hook = hook
        self.random_state = random_state

    def _event(self, name):
        root = Path(self.journal)
        row = {'event': name, 'pid': os.getpid(), 'origin_pid': self.origin_pid}
        (root / (name + '-' + uuid.uuid4().hex + '.json')).write_text(json.dumps(row), encoding='utf-8')
        # Exclusive creation avoids a shared counter race or timing assumption.
        try:
            with (root / (name + '.first')).open('x', encoding='utf-8') as stream:
                stream.write(str(os.getpid()))
        except FileExistsError:
            raise RuntimeError('custom estimator second ' + name + ' refused') from None

    def __getstate__(self):
        if self.hook == 'serialize':
            self._event('serialize')
        return dict(self.__dict__)

    def get_params(self, deep=True):
        if self.hook == 'get_params' and os.getpid() != self.origin_pid:
            self._event('get_params')
        return super().get_params(deep=deep)

    def fit(self, X, y):
        self._event('unexpected-fit')
        raise AssertionError('The process-hook test must never fit')

    def predict(self, X):
        self._event('unexpected-predict')
        raise AssertionError('The process-hook test must never predict')


def native_limits():
    return sorted((row['filepath'], row['user_api'], row['num_threads']) for row in threadpool_info())


@pytest.mark.parametrize('hook', ['serialize', 'get_params'])
def test_entry_admits_single_custom_hook_without_model_work(tmp_path, hook):
    journal = tmp_path / 'events'
    journal.mkdir()
    estimator = ProcessHookClassifier(str(journal), os.getpid(), hook)
    X = np.array([[-2.], [-1.], [1.], [2.]])
    y = np.array([0, 0, 1, 1])
    folds = [(np.array([0, 2]), np.array([1, 3])),
             (np.array([1, 3]), np.array([0, 2]))]
    store = cp.CheckpointedCV(tmp_path / 'run', estimator, X, y, folds,
                             jobs=2, reuse_equivalent_folds=False)
    env_before, numpy_before, pools_before = dict(os.environ), np.geterr(), native_limits()
    with store:
        assert (store.folder / 'identity.json').exists()
    events = [json.loads(path.read_text()) for path in journal.glob(hook + '-*.json')]
    assert len(events) == 1
    if hook == 'get_params':
        assert events[0]['pid'] != os.getpid()
    assert not list(journal.glob('unexpected-*'))
    assert not list((store.folder / 'folds').glob('*.json'))
    assert dict(os.environ) == env_before and np.geterr() == numpy_before
    assert all(row in native_limits() for row in pools_before)
    assert json.loads((store.folder / 'owner.json').read_text())['state'] == 'inactive'
    assert store.lease is None
    with cp._Lease(store.folder / 'session.lock'):
        pass
