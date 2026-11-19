"""Resume diagnostics retain authentic receipts and never repeat saved fits."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
import cv_checkpoint as checkpoint
from runtime_test_support import synthetic_runtime
from search_support import _io_path


class TinyDiagnosticClassifier(ClassifierMixin, BaseEstimator):
    fits = 0
    predictions = 0

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, y):
        type(self).fits += 1
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        type(self).predictions += 1
        return (np.asarray(X)[:, 0] >= 0).astype(np.int64)


@pytest.fixture
def session_factory(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint, 'runtime_fingerprint',
                        lambda **kwargs: synthetic_runtime(source='resume-diagnostics'))
    TinyDiagnosticClassifier.fits = TinyDiagnosticClassifier.predictions = 0
    X = np.asarray([[-4.], [-3.], [-2.], [-1.], [1.], [2.], [3.], [4.]])
    y = (X[:, 0] >= 0).astype(np.int64)
    folds = [(np.array([0, 2, 4, 6]), np.array([1, 3, 5, 7])),
             (np.array([1, 3, 5, 7]), np.array([0, 2, 4, 6]))]

    def create(*, resume=False, **changes):
        options = dict(run_dir=_io_path(tmp_path) / 'run', estimator=TinyDiagnosticClassifier(),
                       X=X, y=y, folds=folds, provenance={'policy': 'SECRET_ORIGINAL'},
                       jobs=1, error_policy='raise', reuse_equivalent_folds=False, resume=resume)
        options.update(changes)
        return checkpoint.CheckpointedCV(**options)

    return create


def saved_json(folder):
    return {path.relative_to(folder).as_posix(): path.read_bytes()
            for path in folder.rglob('*.json') if path.is_file()}


def test_valid_resume_preserves_completed_folds_without_extra_entry_work(session_factory, monkeypatch):
    with session_factory() as first:
        initial = first.search({})
    preserved = {name: data for name, data in saved_json(first.folder).items()
                 if name == 'identity.json' or name.startswith(('folds/', 'attempts/'))}
    assert TinyDiagnosticClassifier.fits == TinyDiagnosticClassifier.predictions == 2
    original_read, original_digest = checkpoint._read_json, checkpoint._digest
    calls = {'read': 0, 'digest': 0}

    def read(path):
        calls['read'] += 1
        return original_read(path)

    def digest(value):
        calls['digest'] += 1
        return original_digest(value)

    resumed = session_factory(resume=True)
    with monkeypatch.context() as local:
        local.setattr(checkpoint, '_read_json', read)
        local.setattr(checkpoint, '_digest', digest)
        local.setattr(checkpoint, '_identity_mismatch_details',
                      lambda *args: pytest.fail('Healthy resume must not format differences'))
        with resumed:
            assert calls == {'read': 2, 'digest': 6}
            replay = resumed.search({})
    assert replay.checkpoint_summary_['new_folds'] == 0
    assert replay.checkpoint_summary_['reused_folds'] == 2
    assert TinyDiagnosticClassifier.fits == TinyDiagnosticClassifier.predictions == 2
    np.testing.assert_array_equal(initial.cv_results_['mean_test_score'], replay.cv_results_['mean_test_score'])
    assert all(saved_json(first.folder)[name] == data for name, data in preserved.items())


@pytest.mark.parametrize('damage', ['checksum', 'foreign', 'both'])
def test_generic_receipt_uses_integrity_first_neutral_diagnostics(tmp_path, damage):
    path = _io_path(tmp_path) / 'receipt.json'
    checkpoint._write_receipt(path, {'identity_sha256': 'a' * 64, 'value': 'SECRET_VALUE'})
    record = json.loads(path.read_text(encoding='utf-8'))
    if damage in ('foreign', 'both'):
        record['payload']['identity_sha256'] = 'b' * 64
    record['sha256'] = checkpoint._digest(record['payload']) if damage == 'foreign' else '0' * 64
    path.write_text(json.dumps(record), encoding='utf-8')
    before = path.read_bytes()
    with pytest.raises(checkpoint.CheckpointError) as caught:
        checkpoint._read_receipt(path, 'a' * 64)
    category = 'identity' if damage == 'foreign' else 'checksum'
    assert str(caught.value).startswith('Checkpoint ' + category + ' mismatch:')
    assert 'Run identity' not in str(caught.value) and 'SECRET_VALUE' not in str(caught.value)
    assert path.read_bytes() == before


@pytest.mark.parametrize('corrupt', [False, True])
def test_changed_run_or_corrupt_manifest_refuses_before_work(session_factory, corrupt):
    with session_factory() as first:
        first.search({})
    manifest = first.folder / 'identity.json'
    if corrupt:
        record = json.loads(manifest.read_text(encoding='utf-8'))
        record['sha256'] = '0' * 64
        manifest.write_text(json.dumps(record), encoding='utf-8')
    before = saved_json(first.folder)
    rejected = session_factory(resume=True, provenance={'policy': 'SECRET_CHANGED'})
    with pytest.raises(checkpoint.CheckpointError) as caught:
        with rejected:
            pytest.fail('Invalid resume was admitted')
    message = str(caught.value)
    if corrupt:
        assert message == 'Saved identity checksum or structure mismatch'
    else:
        assert message.startswith('Run identity changed; use a new run directory')
        assert '"provenance"' in message
    assert 'SECRET_' not in message and saved_json(first.folder) == before
    assert rejected.lease is None
    assert TinyDiagnosticClassifier.fits == TinyDiagnosticClassifier.predictions == 2


@pytest.mark.parametrize('damage', ['checksum', 'same-hash-changed-body'])
def test_manifest_second_read_still_refuses_changes(session_factory, monkeypatch, damage):
    with session_factory() as first:
        first.search({})
    original_saved_identity = checkpoint._saved_identity
    mutated = []

    def read_then_change(path, **options):
        result = original_saved_identity(path, **options)
        record = json.loads(path.read_text(encoding='utf-8'))
        if damage == 'checksum':
            record['sha256'] = '0' * 64
        else:
            record['payload']['identity']['provenance'] = checkpoint._stable({'policy': 'SECRET_INCONSISTENT'})
            record['sha256'] = checkpoint._digest(record['payload'])
        path.write_text(json.dumps(record), encoding='utf-8')
        mutated.append(path.read_bytes())
        return result

    before = saved_json(first.folder)
    before.pop('identity.json')
    monkeypatch.setattr(checkpoint, '_saved_identity', read_then_change)
    rejected = session_factory(resume=True)
    with pytest.raises(checkpoint.CheckpointError) as caught:
        with rejected:
            pytest.fail('A changed second manifest read was admitted')
    prefix = ('Checkpoint checksum mismatch:' if damage == 'checksum'
              else 'Checkpoint identity metadata inconsistent:')
    assert str(caught.value).startswith(prefix)
    assert 'SECRET_' not in str(caught.value) and len(mutated) == 1
    assert (first.folder / 'identity.json').read_bytes() == mutated[0]
    after = saved_json(first.folder)
    after.pop('identity.json')
    assert after == before and rejected.lease is None
    assert TinyDiagnosticClassifier.fits == TinyDiagnosticClassifier.predictions == 2


@pytest.mark.parametrize('old,new', [(True, 1), (1, 1.0), (-0.0, 0.0)])
def test_field_diagnostics_preserve_canonical_types(old, new):
    assert checkpoint._identity_mismatch_details({'typed': old}, {'typed': new}) == \
           ' (differing identity fields: "typed")'


def test_field_diagnostics_are_bounded_and_values_remain_private():
    keys = [chr(0x10000 + index) * 40 for index in range(6)]
    detail = checkpoint._identity_mismatch_details(
        {key: 'SECRET_OLD' for key in keys}, {key: 'SECRET_NEW' for key in keys})
    assert len(detail) <= 512 and 'additional details truncated' in detail
    assert 'SECRET_' not in detail and not any(char in detail for char in '\r\n\t')
    assert checkpoint._identity_mismatch_details({'data': {'X': 1}}, {'data': {'X': 2}}) == \
           ' (differing identity fields: "data")'
