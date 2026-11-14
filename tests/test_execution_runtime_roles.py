"""Closed-schema unit guards. Synthetic pool records are not native-load proof."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from runtime_test_support import synthetic_runtime

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
sys.path.insert(0, str(EXAMPLES))
import cv_checkpoint as cp


def records():
    worker = synthetic_runtime()
    parent = deepcopy(worker)
    extra = deepcopy(worker['native_pools'][0])
    extra.update(filepath=str(Path(__file__).resolve().with_name('__synthetic_extra__')),
                 sha256='e' * 64, version='extra')
    parent['native_pools'].append(extra)
    parent['files_sha256'], parent['file_count'] = 'e' * 64, 3
    return worker, parent


def compatible(worker, parent):
    copies = deepcopy((worker, parent))
    cp._require_execution_compatible(cp._stable(worker), cp._stable(parent))
    assert (worker, parent) == copies


def test_only_parent_extra_is_admitted_without_rewriting_full_identity():
    worker, parent = records()
    compatible(worker, parent)
    with pytest.raises(cp.CheckpointError, match='Execution native pools are not contained'):
        compatible(parent, worker)
    assert cp._stable(worker) != cp._stable(parent)
    equal_pools = deepcopy(worker)
    equal_pools['files_sha256'] = 'f' * 64
    with pytest.raises(cp.CheckpointError, match='full artifact identities are inconsistent'):
        compatible(worker, equal_pools)
    inflated = deepcopy(parent)
    inflated['file_count'] += 1
    with pytest.raises(cp.CheckpointError, match='artifact counts are inconsistent'):
        compatible(worker, inflated)
    # Deliberately malformed encoded scalar: Python True == 1 must not admit it.
    malformed = cp._stable(worker)
    fields = dict(malformed[1])
    native = dict(fields['native_pools'][1][0][1])
    native['num_threads'][0], native['num_threads'][1] = 'int', True
    with pytest.raises(cp.CheckpointError, match='requires native threads equal to one'):
        cp._require_execution_compatible(malformed, cp._stable(parent))
    typed_worker = synthetic_runtime(packages={'counter': 1})
    typed_parent = cp._stable(typed_worker)
    malformed_common = deepcopy(typed_parent)
    common = dict(malformed_common[1])
    counter = dict(common['packages'][1])['counter']
    counter[1] = True
    with pytest.raises(cp.CheckpointError, match='common runtime differs: packages'):
        cp._require_execution_compatible(malformed_common, typed_parent)


@pytest.mark.parametrize('field,value', [
    ('python', 'different-python'), ('packages', {'different': 'source'}),
    ('package_native', {'different': 'accelerator'}),
    ('common_files_sha256', 'd' * 64), ('common_file_count', 2),
])
def test_common_runtime_changes_are_not_coordinator_extras(field, value):
    worker, parent = records()
    worker[field] = value
    with pytest.raises(cp.CheckpointError, match='Execution and coordinator common runtime differs: ' + field):
        compatible(worker, parent)


@pytest.mark.parametrize('field,value', [
    ('version', 'different'), ('architecture', 'different'), ('sha256', 'f' * 64),
    ('num_threads', 2),
])
def test_entire_native_metadata_and_count_remain_exact(field, value):
    worker, parent = records()
    worker['native_pools'][0][field] = value
    reason = ('requires native threads equal to one' if field == 'num_threads'
              else 'Execution native pools are not contained')
    with pytest.raises(cp.CheckpointError, match=reason):
        compatible(worker, parent)


def test_pool_multiplicity_is_not_set_membership():
    worker, parent = records()
    worker['native_pools'].append(deepcopy(worker['native_pools'][0]))
    with pytest.raises(cp.CheckpointError, match='Execution native pools are not contained'):
        compatible(worker, parent)
    parent['native_pools'].append(deepcopy(parent['native_pools'][0]))
    compatible(worker, parent)


@pytest.mark.parametrize('mutation', ['legacy', 'duplicate', 'unknown', 'empty-blas'])
def test_malformed_or_unproved_role_identity_refuses(mutation):
    worker, parent = records()
    encoded = cp._stable(worker)
    if mutation == 'legacy':
        encoded[1] = [row for row in encoded[1] if row[0] != 'runtime_policy']
        message = 'Unsupported execution-role runtime policy'
    elif mutation == 'duplicate':
        encoded[1].append(deepcopy(encoded[1][0]))
        message = 'Malformed or duplicate runtime identity field'
    elif mutation == 'unknown':
        encoded[1].append(['unexpected', cp._stable('value')])
        message = 'Unsupported execution-role runtime policy'
    else:
        worker['native_pools'] = []
        encoded = cp._stable(worker)
        message = 'Loaded BLAS identity is unavailable'
    with pytest.raises(cp.CheckpointError, match=message):
        cp._require_execution_compatible(encoded, cp._stable(parent))


def unit_session(tmp_path, monkeypatch, *, jobs=2, resume=False):
    runtime = synthetic_runtime()
    monkeypatch.setattr(cp, 'runtime_fingerprint', lambda **kwargs: deepcopy(runtime))
    X = np.arange(8, dtype=float).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    folds = [(np.array([0, 2, 4, 6]), np.array([1, 3, 5, 7])),
             (np.array([1, 3, 5, 7]), np.array([0, 2, 4, 6]))]
    def forbidden_fit(*args, **kwargs):
        raise AssertionError('This metadata-only unit fixture must not fit')
    monkeypatch.setattr(DummyClassifier, 'fit', forbidden_fit)
    return cp.CheckpointedCV(tmp_path / 'run', DummyClassifier(random_state=31), X, y,
                            folds, jobs=jobs, resume=resume, reuse_equivalent_folds=False), runtime


def test_probe_failure_publishes_no_identity_or_session(tmp_path, monkeypatch):
    session, runtime = unit_session(tmp_path, monkeypatch)
    error = RuntimeError('controlled failed execution probe')
    def fail(*args, **kwargs):
        raise error
    events = []
    class FailingAuthorityPool:
        def __init__(self, **options):
            assert options == dict(n_jobs=2, pre_dispatch=2, batch_size=1, return_as='generator')
            self.active = False
        def __enter__(self):
            assert not self.active
            self.active = True
            return self
        def __exit__(self, *unused):
            assert self.active
            self.active = False
            events.append('closed')
        def __call__(self, pending):
            assert self.active
            tasks = list(pending)
            assert len(tasks) == 1
            function, args, kwargs = tasks[0]
            if function is cp._execution_import_probe:
                assert args == () and kwargs == {}
                events.append('import')
                return (function(*args, **kwargs) for unused in range(1))
            assert function is cp._execution_runtime_probe
            assert len(args) == 1 and args[0] is session.estimator and kwargs == {}
            events.append('authority')
            return (fail() for unused in range(1))
    monkeypatch.setattr(cp, 'Parallel', FailingAuthorityPool)
    with pytest.raises(RuntimeError) as caught:
        with session:
            raise AssertionError('Unexpected entry')
    assert caught.value is error
    assert events == ['import', 'authority', 'closed']
    assert session.lease is None
    assert not (session.folder / 'identity.json').exists()
    assert not list((session.folder / 'sessions').glob('*.json'))
    assert not list((session.folder / 'folds').glob('*.json'))


def test_saved_execution_pin_is_authenticated_before_no_probe_resume(tmp_path, monkeypatch):
    session, runtime = unit_session(tmp_path, monkeypatch, jobs=1)
    with session:
        pass
    path = session.folder / 'identity.json'
    record = json.loads(path.read_text(encoding='utf-8'))
    record['payload']['identity']['runtime'][1].append(['corrupt', cp._stable(True)])
    path.write_text(json.dumps(record), encoding='utf-8')
    def forbidden(*args, **kwargs):
        raise AssertionError('Resume must not probe, including corrupt identity')
    monkeypatch.setattr(cp, '_probe_execution_runtime', forbidden)
    resumed, _ = unit_session(tmp_path, monkeypatch, resume=True)
    with pytest.raises(cp.CheckpointError, match='Saved identity checksum or structure mismatch'):
        with resumed:
            raise AssertionError('Unexpected entry')


def test_rechecksummed_session_tamper_cannot_change_active_authority(tmp_path, monkeypatch):
    session, runtime = unit_session(tmp_path, monkeypatch, jobs=1)
    with session:
        path = session.folder / 'sessions' / (session.token + '.json')
        record = json.loads(path.read_text(encoding='utf-8'))
        record['payload']['execution_runtime_sha256'] = 'f' * 64
        cp._write_receipt(path, record['payload'])
        with pytest.raises(cp.CheckpointError, match='Coordinator session or execution runtime identity changed'):
            session.check_runtime()
    assert not list((session.folder / 'folds').glob('*.json'))
