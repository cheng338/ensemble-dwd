"""An unavailable optional pair cache must still return the validated result."""
from pathlib import Path
from unittest.mock import Mock

import pytest


class UnprintableFileError(FileNotFoundError):
    def __format__(self, spec):
        raise ValueError('cannot format storage failure')

    def __str__(self):
        raise ValueError('cannot stringify storage failure')


class CustomFormatError(OSError):
    def __format__(self, spec):
        assert spec == ''
        return 'formatted message'

    def __str__(self):
        return 'string message'


FALLBACK = 'UnprintableFileError: Error message could not be formatted'
KEY = {'pair': 0}


@pytest.fixture
def storage(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'examples'))
    import _pair_store
    return _pair_store


@pytest.fixture
def store(storage, tmp_path):
    return storage.PairStore(tmp_path / 'pairs', {'fold': 0}, snapshot=dict, assert_owner=lambda: None)


def check_fresh_result(session, reason, message=FALLBACK):
    model = {'value': 7}
    compute = Mock(return_value=model)
    validate = Mock()
    result, event = session.get_or_compute(KEY, compute, validate)
    assert result is model
    compute.assert_called_once_with()
    validate.assert_called_once_with(model)
    assert event['status'] == 'fitted_unsaved'
    assert event['cache_committed'] is False
    assert event['cache_reason'] == reason
    assert event['cache_error'] == message


@pytest.mark.parametrize('reason', ['storage_initialization_failure', 'parent_write_failure', 'reclaim_failure'])
def test_disabled_storage_preserves_fresh_result(storage, store, monkeypatch, reason):
    failure = Mock(side_effect=UnprintableFileError())
    if reason == 'storage_initialization_failure':
        monkeypatch.setattr(storage._Session, '_initialize', failure)
    elif reason == 'parent_write_failure':
        monkeypatch.setattr(store, '_atomic_write', failure)
    else:
        with store.session():
            pass
        store._owned_file('a' * 64 + '.pkl').write_bytes(b'orphan')
        monkeypatch.setattr(storage._Session, '_remove', failure)
    with store.session() as session:
        check_fresh_result(session, reason)
        cleanup = session.complete('a' * 64)
    failure.assert_called_once()
    assert cleanup['status'] == 'cleanup_skipped'
    assert cleanup['cache_reason'] == reason and cleanup['cache_error'] == FALLBACK


@pytest.mark.parametrize(('error', 'message'), [
    (UnprintableFileError(), FALLBACK),
    (CustomFormatError(), 'CustomFormatError: formatted message'),
], ids=['broken-format', 'custom-format'])
def test_write_failure_preserves_model_and_original_format_convention(store, monkeypatch, error, message):
    with store.session() as session:
        monkeypatch.setattr(store, '_atomic_write', Mock(side_effect=error))
        check_fresh_result(session, 'write_failure', message)


@pytest.mark.parametrize('operation', ['lstat', 'read_bytes'])
def test_payload_error_refits_once_and_preserves_committed_files(store, monkeypatch, operation):
    with store.session() as session:
        _, saved = session.get_or_compute(KEY, lambda: {'value': 1}, lambda model: None)
        payload = store._owned_file(saved['key'] + '.pkl')
        manifest = store._owned_file(saved['key'] + '.json')
        original_payload, original_manifest = payload.read_bytes(), manifest.read_bytes()
        # Inject the failure after the real namespace guard, at the optional load.
        guarded_path = Mock(wraps=payload)
        getattr(guarded_path, operation).side_effect = UnprintableFileError()
        owned_file = store._owned_file

        def resolve(name):
            path = owned_file(name)
            return guarded_path if path == payload else path

        monkeypatch.setattr(store, '_owned_file', resolve)
        check_fresh_result(session, 'payload_read_failure')
    assert payload.read_bytes() == original_payload
    assert manifest.read_bytes() == original_manifest


@pytest.mark.parametrize('operation', ['_remove', '_pair_bytes'])
def test_cleanup_error_preserves_authoritative_receipt_and_closed_session(store, monkeypatch, operation):
    with store.session() as session:
        _, saved = session.get_or_compute(KEY, lambda: {'value': 1}, lambda model: None)
        before = store._owned_file('identity.json').read_bytes()
        monkeypatch.setattr(session, operation, Mock(side_effect=UnprintableFileError()))
        event = session.complete('a' * 64)
        assert event['status'] == 'cleanup_incomplete'
        assert event['authoritative_receipt_sha256'] == 'a' * 64
        if operation == '_remove':
            assert event['cache_error'] == FALLBACK
            assert event['remaining_pair_bytes'] > 0 and not event['removed_files']
        else:
            assert event['remaining_bytes_error'] == FALLBACK
            assert event['remaining_pair_bytes'] is None
            assert set(event['removed_files']) == {saved['key'] + suffix for suffix in ('.json', '.pkl')}
        assert store._owned_file('identity.json').read_bytes() == before
        with pytest.raises(RuntimeError, match='already completed'):
            session.get_or_compute(KEY, lambda: pytest.fail('repeated fit'), lambda model: None)


@pytest.mark.parametrize('stage', ['compute', 'validate'])
def test_scientific_errors_remain_strict(store, stage):
    error = UnprintableFileError()
    compute = Mock(return_value={'value': 7}, side_effect=error if stage == 'compute' else None)
    validate = Mock(side_effect=error if stage == 'validate' else None)
    with store.session() as session, pytest.raises(UnprintableFileError) as caught:
        session.get_or_compute(KEY, compute, validate)
    assert caught.value is error
    compute.assert_called_once_with()
    assert validate.call_count == int(stage == 'validate')

