"""Focused optional-payload recovery and strict-boundary checks (pytest)."""
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modules(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    examples = root / 'examples'
    storage = _load(examples / '_pair_store.py', '_ir24_payload_storage')
    # Bind this private module to its own storage copy only during import.
    # Existing test fixtures and imported production modules are not reloaded.
    with monkeypatch.context() as importing:
        importing.syspath_prepend(str(examples))
        importing.setitem(sys.modules, '_pair_store', storage)
        recovery = _load(examples / 'pair_recovery.py', '_ir24_payload_recovery')
    error = type('CheckpointError', (RuntimeError,), {})
    monkeypatch.setattr(recovery, 'checkpoint', lambda: SimpleNamespace(CheckpointError=error))
    return storage, recovery, error


def _store(storage, root, owner=lambda: None):
    return storage.PairStore(root, {'fold': 0, 'data': 'fixed'},
                             snapshot=deepcopy, assert_owner=owner)


def _seed(storage, root, owner=lambda: None):
    store = _store(storage, root, owner)
    with store.session() as session:
        _, event = session.get_or_compute({'pair': 0}, lambda: {'value': 7}, lambda model: None)
        session.get_or_compute({'pair': 1}, lambda: {'value': 9}, lambda model: None)
    return store, store._owned_file(event['key'] + '.pkl'), store._owned_file(event['key'] + '.json')


def _hashes(storage, root):
    return {path.name: storage.digest(path.read_bytes())
            for path in storage._io_path(root).iterdir() if path.is_file()}


@pytest.mark.parametrize('damage', ['missing', 'truncated', 'bitflip', 'read_error'])
def test_only_damaged_pair_refits_without_loading_or_rewriting(modules, tmp_path, monkeypatch, damage):
    storage, _, _ = modules
    store, payload, _ = _seed(storage, tmp_path / 'pairs')
    if damage == 'missing':
        payload.unlink()
    elif damage == 'truncated':
        payload.write_bytes(payload.read_bytes()[:-1])
    elif damage == 'bitflip':
        data = bytearray(payload.read_bytes())
        data[-1] ^= 1
        payload.write_bytes(data)
    before = _hashes(storage, store.root)
    original_read, original_load = Path.read_bytes, storage.pickle.loads
    loaded = []

    def read(path):
        if damage == 'read_error' and path.name == payload.name:
            raise PermissionError('controlled payload read failure')
        return original_read(path)

    def load(data):
        loaded.append(data)
        return original_load(data)

    compute = Mock(return_value={'value': 7})
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'read_bytes', read)
        patch.setattr(storage.pickle, 'loads', load)
        patch.setattr(store, '_atomic_write', Mock(side_effect=AssertionError('recovery wrote cache')))
        with store.session() as session:
            model, event = session.get_or_compute({'pair': 0}, compute, lambda model: None)
            sibling, sibling_event = session.get_or_compute(
                {'pair': 1}, Mock(side_effect=AssertionError('healthy pair refit')), lambda model: None)
    assert model == {'value': 7} and sibling == {'value': 9}
    compute.assert_called_once_with()
    assert len(loaded) == 1
    assert sibling_event['status'] == 'reused'
    assert event['status'] == 'fitted_unsaved' and event['cache_committed'] is False
    assert event['load_seconds'] == event['save_seconds'] == event['historical_fit_seconds'] == 0
    assert event['failed_load_seconds'] >= 0 and event['fit_seconds'] >= 0
    assert event['cache_reason'] == ('payload_read_failure' if damage in ('missing', 'read_error')
                                     else 'payload_integrity_failure')
    assert _hashes(storage, store.root) == before


def test_repeated_unfinished_attempts_preserve_entry_and_count_new_fits(modules, tmp_path):
    storage, _, _ = modules
    store, payload, _ = _seed(storage, tmp_path / 'pairs')
    payload.write_bytes(b'broken')
    before = _hashes(storage, store.root)
    compute = Mock(return_value={'value': 7})
    for _ in range(2):
        with store.session() as session:
            _, event = session.get_or_compute({'pair': 0}, compute, lambda model: None)
            assert event['status'] == 'fitted_unsaved'
            assert event['historical_fit_seconds'] == 0
            session.get_or_compute({'pair': 1}, Mock(side_effect=AssertionError('sibling refit')),
                                   lambda model: None)
        assert _hashes(storage, store.root) == before
    assert compute.call_count == 2


@pytest.mark.parametrize('damage', ['envelope', 'parent', 'key', 'filename', 'bytes_bool',
                                  'bytes_negative', 'hash_shape', 'history_bool', 'schema_bool', 'missing_state'])
def test_manifest_and_identity_failures_never_refit(modules, tmp_path, monkeypatch, damage):
    storage, _, _ = modules
    store, payload, receipt = _seed(storage, tmp_path / 'pairs')
    saved, _ = storage._read_envelope(receipt)
    if damage == 'envelope':
        receipt.write_bytes(b'{"payload":')
    else:
        if damage == 'parent':
            saved['parent_sha256'] = '0' * 64
        elif damage == 'key':
            saved['key_identity'] = {'pair': 99}
        elif damage == 'filename':
            saved['model_file'] = '../other.pkl'
        elif damage == 'bytes_bool':
            saved['model_bytes'] = True
        elif damage == 'bytes_negative':
            saved['model_bytes'] = -1
        elif damage == 'hash_shape':
            saved['model_sha256'] = 'invalid'
        elif damage == 'history_bool':
            saved['historical_fit_seconds'] = True
        elif damage == 'schema_bool':
            saved['schema'] = True
        else:
            del saved['mathematical_state']
        receipt.write_bytes(storage._envelope(saved))
    payload.write_bytes(b'also broken')
    before = _hashes(storage, store.root)
    monkeypatch.setattr(storage.pickle, 'loads', Mock(side_effect=AssertionError('invalid load')))
    with store.session() as session, pytest.raises(storage.StoreMismatch):
        session.get_or_compute({'pair': 0}, Mock(side_effect=AssertionError('invalid refit')),
                               lambda model: None)
    assert _hashes(storage, store.root) == before


@pytest.mark.parametrize('kind', ['directory', 'symlink', 'fifo', 'reparse'])
def test_nonregular_namespace_never_becomes_cache_miss(modules, tmp_path, monkeypatch, kind):
    storage, _, _ = modules
    store, payload, _ = _seed(storage, tmp_path / 'pairs')
    if kind == 'reparse':
        original = Path.lstat

        def lstat(path):
            if path.name == payload.name:
                return SimpleNamespace(st_mode=stat.S_IFREG,
                                       st_file_attributes=getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 1024))
            return original(path)

        monkeypatch.setattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT',
                            getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 1024), raising=False)
        monkeypatch.setattr(Path, 'lstat', lstat)
    else:
        payload.unlink()
        if kind == 'directory':
            payload.mkdir()
        elif kind == 'symlink':
            target = tmp_path / 'external.pkl'
            target.write_bytes(b'not a trusted payload')
            try:
                payload.symlink_to(target)
            except (OSError, NotImplementedError):
                pytest.skip('unprivileged symlinks unavailable')
        else:
            if not hasattr(os, 'mkfifo'):
                pytest.skip('FIFO is not available on this platform')
            os.mkfifo(payload)
    with store.session() as session, pytest.raises(storage.StoreMismatch):
        session.get_or_compute({'pair': 0}, Mock(side_effect=AssertionError('namespace refit')),
                               lambda model: None)


def test_namespace_replacement_during_failed_read_never_refits(modules, tmp_path, monkeypatch):
    storage, _, _ = modules
    store, payload, _ = _seed(storage, tmp_path / 'pairs')
    original_read = Path.read_bytes

    def read(path):
        if path.name == payload.name:
            path.unlink()
            path.mkdir()
            return b'bad read before namespace replacement'
        return original_read(path)

    monkeypatch.setattr(Path, 'read_bytes', read)
    with store.session() as session, pytest.raises(storage.StoreMismatch):
        session.get_or_compute({'pair': 0}, Mock(side_effect=AssertionError('namespace refit')),
                               lambda model: None)


@pytest.mark.parametrize('failure', ['compute', 'validate', 'stale_before', 'stale_after'])
def test_fallback_never_hides_fit_validation_or_ownership_failure(modules, tmp_path, monkeypatch, failure):
    storage, _, _ = modules
    current = [True]

    def owner():
        if not current[0]:
            raise RuntimeError('stale owner')

    store, payload, _ = _seed(storage, tmp_path / 'pairs', owner)
    payload.write_bytes(b'broken')
    original_read = Path.read_bytes
    error = ArithmeticError('original computation failure')

    def read(path):
        data = original_read(path)
        if failure == 'stale_before' and path.name == payload.name:
            current[0] = False
        return data

    def compute():
        if failure == 'compute':
            raise error
        if failure == 'stale_after':
            current[0] = False
        return {'value': 7}

    def validate(model):
        if failure == 'validate':
            raise error

    counted = Mock(side_effect=compute)
    monkeypatch.setattr(Path, 'read_bytes', read)
    with store.session() as session, pytest.raises((RuntimeError, ArithmeticError)) as caught:
        session.get_or_compute({'pair': 0}, counted, validate)
    if failure in ('compute', 'validate'):
        assert caught.value is error
    assert counted.call_count == (0 if failure == 'stale_before' else 1)


@pytest.mark.parametrize('failure', ['deserialize', 'semantic', 'validate'])
def test_intact_payload_validation_remains_strict(modules, tmp_path, monkeypatch, failure):
    storage, _, _ = modules
    store, _, receipt = _seed(storage, tmp_path / 'pairs')
    if failure == 'deserialize':
        monkeypatch.setattr(storage.pickle, 'loads', Mock(side_effect=ValueError('bad pickle contract')))
    elif failure == 'semantic':
        saved, _ = storage._read_envelope(receipt)
        saved['mathematical_state'] = {'value': 99}
        receipt.write_bytes(storage._envelope(saved))
    error = ValueError('model validation failed')

    def validate(model):
        if failure == 'validate':
            raise error

    with store.session() as session, pytest.raises((storage.StoreMismatch, ValueError)) as caught:
        session.get_or_compute({'pair': 0}, Mock(side_effect=AssertionError('semantic refit')), validate)
    if failure == 'validate':
        assert caught.value is error


def _record(recovery):
    event = {'pair_index': 0, 'pair': [0, 1], 'status': 'fitted_unsaved',
             'work_origin': 'fitted', 'cache_committed': False,
             'fit_seconds': 2.0, 'load_seconds': 0.0, 'save_seconds': 0.0,
             'historical_fit_seconds': 0.0, 'failed_load_seconds': .25,
             'cache_reason': 'payload_integrity_failure', 'cache_error': 'size mismatch'}
    context = object.__new__(recovery.FoldContext)
    context.events, context.cleanup_status = [event], 'pending'
    return {'state': 'complete', 'pair_recovery': context.summary(),
            'diagnostics': {'pairwise_estimators': [{'pair': [0, 1], 'work_origin': 'fitted'}]}}


def test_failed_load_accounting_is_current_and_separate(modules):
    _, recovery, _ = modules
    record = _record(recovery)
    recovery.validate_summary(record)
    summary = record['pair_recovery']
    assert summary['pair_failed_load_seconds'] == .25
    assert summary['fitted_pairs'] == summary['fitted_unsaved_pairs'] == 1
    assert summary['recovered_pairs'] == summary['historical_reused_pair_fit_seconds'] == 0
    assert summary['pair_load_seconds'] == summary['pair_save_seconds'] == 0


@pytest.mark.parametrize('damage', ['negative', 'nan', 'bool', 'missing', 'reason',
                                  'error', 'summary', 'reused', 'saved', 'save_time'])
def test_failed_load_accounting_tampering_refuses(modules, damage):
    _, recovery, error = modules
    record = _record(recovery)
    summary, event = record['pair_recovery'], record['pair_recovery']['pairs'][0]
    if damage in ('negative', 'nan', 'bool'):
        event['failed_load_seconds'] = {'negative': -1, 'nan': float('nan'), 'bool': True}[damage]
    elif damage == 'missing':
        del event['failed_load_seconds']
    elif damage == 'reason':
        event['cache_reason'] = 'unrelated_failure'
    elif damage == 'error':
        event['cache_error'] = ''
    elif damage == 'summary':
        summary['pair_failed_load_seconds'] = 0.0
    elif damage == 'reused':
        event.update(status='reused', work_origin='recovered', cache_committed=True, fit_seconds=0.0)
    elif damage == 'saved':
        event.update(status='fitted_saved', cache_committed=True)
    else:
        event['save_seconds'] = 1.0
    with pytest.raises(error):
        recovery.validate_summary(record)


def test_ordinary_saved_and_reused_events_have_no_failure_fields(modules, tmp_path):
    storage, recovery, _ = modules
    store = _store(storage, tmp_path / 'pairs')
    with store.session() as session:
        _, fitted = session.get_or_compute({'pair': 0}, lambda: {'value': 7}, lambda model: None)
        _, reused = session.get_or_compute({'pair': 0}, Mock(side_effect=AssertionError('ordinary refit')),
                                          lambda model: None)
    for event in (fitted, reused):
        assert 'failed_load_seconds' not in event and 'cache_reason' not in event
    context = object.__new__(recovery.FoldContext)
    context.events, context.cleanup_status = [], 'pending'
    assert 'pair_failed_load_seconds' not in context.summary()
