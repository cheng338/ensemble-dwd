"""Trusted-local pair storage for the optional checkpointed example workflow.

Checksums provide integrity checks, not safety for untrusted pickle producers.
The existing CV coordinator owns every access through its fold lease and token.
"""
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import threading
import time
import uuid


class StoreMismatch(RuntimeError):
    """Identity, committed-artifact integrity or namespace contract failed."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(value).hexdigest()


def _json_copy(value):
    """Reject identities that change meaning during their JSON round trip."""
    data = canonical(value)
    copied = json.loads(data)
    if copied != value:
        raise ValueError('Identity must consist of JSON values with string keys')
    return copied


def _io_path(path):
    path = Path(path).absolute()
    text = str(path)
    if os.name == 'nt' and not text.startswith('\\\\?\\'):
        if text.startswith('\\\\'):
            return Path('\\\\?\\UNC\\' + text[2:])
        return Path('\\\\?\\' + text)
    return path


def _envelope(payload):
    return canonical({'payload': payload, 'sha256': digest(canonical(payload))})


def _read_envelope(path):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise StoreMismatch('Duplicate JSON key in committed receipt')
            result[key] = value
        return result

    try:
        raw = _io_path(path).read_bytes()
        value = json.loads(raw, object_pairs_hook=unique_object,
                           parse_constant=lambda value: (_ for _ in ()).throw(
                               StoreMismatch('Nonfinite JSON token in committed receipt')))
        if not isinstance(value, dict) or set(value) != {'payload', 'sha256'}:
            raise StoreMismatch('Malformed checksummed receipt')
        if digest(canonical(value['payload'])) != value['sha256']:
            raise StoreMismatch('Receipt checksum mismatch')
        return value['payload'], raw
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise StoreMismatch('Unreadable committed receipt') from error


_ENTRY = re.compile(r'^[0-9a-f]{64}\.(?:pkl|json)$')
_PARTIAL = re.compile(r'^(?:[0-9a-f]{64}\.(?:pkl|json)|identity\.json)\.partial-[0-9a-f]{32}$')


class PairStore:
    """Trusted-local pair artifacts owned by the existing CV fold lease.

    The caller holds the coordinator's task lease for the entire session and
    supplies a live generation guard. No independent lock or takeover exists.
    The quota covers committed pair payloads and manifests in this fold only.
    """

    def __init__(self, root, identity, max_bytes=None, *, snapshot, assert_owner):
        if not isinstance(identity, dict):
            raise TypeError('Parent identity must be a JSON dictionary')
        if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 0):
            raise ValueError('max_bytes must be a nonnegative integer or None')
        if Path(root).is_symlink():
            raise StoreMismatch('Pair-store directory cannot be a symbolic link')
        self.root = Path(root).resolve()
        self._identity = _json_copy(identity)
        self.identity_sha256 = digest(canonical(self._identity))
        self.max_bytes = max_bytes
        self._snapshot, self._assert_owner = snapshot, assert_owner

    @property
    def identity(self):
        return _json_copy(self._identity)

    def _owned_file(self, name):
        if Path(name).name != name or name in ('', '.', '..'):
            raise StoreMismatch('Unsafe store filename')
        path = self.root / name
        resolved = path.resolve()
        if resolved.parent != self.root or path.is_symlink():
            raise StoreMismatch('Store file escaped its owned namespace')
        return _io_path(path)

    def _atomic_write(self, name, data):
        self._assert_owner()
        target = self._owned_file(name)
        partial_name = name + '.partial-' + uuid.uuid4().hex
        partial = self._owned_file(partial_name)
        with partial.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        self._assert_owner()
        os.replace(partial, target)

    @contextmanager
    def session(self):
        """Use the already owned task lease; never infer ownership from a file."""
        self._assert_owner()
        session = _Session(self, 0.0)
        try:
            try:
                _io_path(self.root).mkdir(parents=True, exist_ok=True)
                session._initialize()
            except OSError as error:
                # The authoritative fold lease remains held. Optional storage
                # initialization failure may retain ordinary validated fitting.
                session._disabled_error = f'{type(error).__name__}: {error}'
                session._disabled_reason = 'storage_initialization_failure'
            yield session
        finally:
            session._active = False


class _Session:
    def __init__(self, store, wait_seconds):
        self.store = store
        self.lock_wait_seconds = wait_seconds
        self._active = True
        self._pid = os.getpid()
        self._thread = threading.get_ident()
        self._completed = False
        self._disabled_error = None
        self._disabled_reason = None
        self.reclaimed_files = []

    def _check(self):
        if not self._active or self._pid != os.getpid() or self._thread != threading.get_ident():
            raise RuntimeError('Store session is inactive or used by another owner')
        self.store._assert_owner()

    def _list(self):
        self._check()
        return list(_io_path(self.store.root).iterdir())

    def _remove(self, name):
        # Re-resolve immediately before each deletion; no recursive deletion or
        # traversal through symlinks/reparse points is used.
        self._check()
        path = self.store._owned_file(name)
        if not path.is_file():
            raise StoreMismatch('Owned artifact is not a regular file')
        path.unlink()

    def _initialize(self):
        expected = {'schema': 1, 'kind': 'pair-store-parent',
                    'identity': self.store._identity,
                    'identity_sha256': self.store.identity_sha256}
        path = self.store._owned_file('identity.json')
        if path.exists():
            actual, _ = _read_envelope(path)
            if canonical(actual) != canonical(expected):
                raise StoreMismatch('Parent identity mismatch')
        else:
            if any(_ENTRY.fullmatch(path.name) for path in self._list()):
                raise StoreMismatch('Pair artifacts exist without parent identity')
            try:
                self.store._atomic_write('identity.json', _envelope(expected))
            except OSError as error:
                # We own the directory but cannot establish its cache identity.
                # This session may compute, but must neither load nor publish.
                self._disabled_error = f'{type(error).__name__}: {error}'
                self._disabled_reason = 'parent_write_failure'
                return
        # Only absent-manifest payloads and recognized partials are abandoned.
        # Invalid committed manifests are preserved and rejected on access.
        try:
            for path in self._list():
                orphan = _ENTRY.fullmatch(path.name) and path.suffix == '.pkl' and not self.store._owned_file(path.stem + '.json').exists()
                if _PARTIAL.fullmatch(path.name) or orphan:
                    self._remove(path.name)
                    self.reclaimed_files.append(path.name)
        except OSError as error:
            # Failure to reclaim optional artifacts must not prevent a valid
            # fresh fit. Do not read or add cache entries in this session.
            self._disabled_error = f'{type(error).__name__}: {error}'
            self._disabled_reason = 'reclaim_failure'

    def _pair_bytes(self):
        return sum(self.store._owned_file(path.name).stat().st_size
                   for path in self._list() if _ENTRY.fullmatch(path.name))

    def get_or_compute(self, key_identity, compute, validate):
        """Validate before return; never hide computation or validation errors."""
        self._check()
        if self._completed:
            raise RuntimeError('This session already completed its fold')
        if not isinstance(key_identity, dict):
            raise TypeError('Pair identity must be a JSON dictionary')
        key_identity = _json_copy(key_identity)
        binding = {'parent_sha256': self.store.identity_sha256,
                   'key_identity': key_identity}
        key = digest(canonical(binding))
        model_name, receipt_name = key + '.pkl', key + '.json'
        model_path = self.store._owned_file(model_name)
        receipt_path = self.store._owned_file(receipt_name)
        event = {'key': key, 'fit_seconds': 0.0, 'load_seconds': 0.0,
                 'save_seconds': 0.0, 'historical_fit_seconds': 0.0,
                 'lock_wait_seconds': self.lock_wait_seconds,
                 'timing_rule': 'fit_seconds is current compute only; historical_fit_seconds is prior work on reuse; lock_wait_seconds is repeated session context, not additive'}
        if self._disabled_error is None and receipt_path.exists():
            started = time.perf_counter()
            saved, _ = _read_envelope(receipt_path)
            if (not isinstance(saved, dict) or saved.get('schema') != 1
                    or saved.get('kind') != 'completed-pair'
                    or saved.get('parent_sha256') != self.store.identity_sha256
                    or canonical(saved.get('key_identity')) != canonical(key_identity)
                    or saved.get('model_file') != model_name):
                raise StoreMismatch('Pair identity or filename mismatch')
            historical = saved.get('historical_fit_seconds')
            if (type(historical) not in (int, float) or not math.isfinite(historical)
                    or historical < 0):
                raise StoreMismatch('Invalid historical fit time')
            try:
                data = model_path.read_bytes()
            except OSError as error:
                raise StoreMismatch('Committed pair payload is unreadable') from error
            if (len(data) != saved.get('model_bytes')
                    or digest(data) != saved.get('model_sha256')):
                raise StoreMismatch('Pair payload size or checksum mismatch')
            # Identity, filename, byte count and checksum all precede loading.
            try:
                model = pickle.loads(data)
            except Exception as error:
                raise StoreMismatch('Committed pair payload cannot be deserialized') from error
            validate(model)
            if canonical(self.store._snapshot(model)) != canonical(saved.get('mathematical_state')):
                raise StoreMismatch('Restored mathematical state mismatch')
            event.update(status='reused', historical_fit_seconds=historical,
                         model_bytes=len(data), load_seconds=time.perf_counter() - started,
                         cache_committed=True)
            return model, event

        started = time.perf_counter()
        model = compute()
        event['fit_seconds'] = time.perf_counter() - started
        # Validation is outside the optional I/O exception handler.
        validate(model)
        if self._disabled_error is not None:
            event.update(status='fitted_unsaved', cache_committed=False,
                         cache_reason=self._disabled_reason, cache_error=self._disabled_error)
            return model, event
        started = time.perf_counter()
        committed = False
        try:
            data = pickle.dumps(model, protocol=5)
            saved = {'schema': 1, 'kind': 'completed-pair', **binding,
                     'model_file': model_name, 'model_bytes': len(data),
                     'model_sha256': digest(data),
                     'mathematical_state': self.store._snapshot(model),
                     'historical_fit_seconds': event['fit_seconds']}
            receipt_data = _envelope(saved)
            event.update(model_bytes=len(data), manifest_bytes=len(receipt_data))
            used = self._pair_bytes()
            if self.store.max_bytes is not None and used + len(data) + len(receipt_data) > self.store.max_bytes:
                event.update(status='fitted_unsaved', cache_reason='quota',
                             cache_committed=False, pair_bytes_before=used)
            else:
                self.store._atomic_write(model_name, data)
                self.store._atomic_write(receipt_name, receipt_data)
                committed = True
                event.update(status='fitted_saved', cache_committed=True)
        except OSError as error:
            event.update(status='fitted_saved' if committed else 'fitted_unsaved',
                         cache_reason='post_commit_failure' if committed else 'write_failure',
                         cache_committed=committed,
                         cache_error=f'{type(error).__name__}: {error}')
        event['save_seconds'] = time.perf_counter() - started
        return model, event

    def complete(self, authoritative_receipt_sha256):
        """Reclaim only after the caller validates its authoritative fold.

        The owner guard is rechecked before every deletion. Parent identity and
        the coordinator's stable task lock are retained. No recursive deletion
        or following of symlinks is permitted.
        """
        self._check()
        if (type(authoritative_receipt_sha256) is not str or
                not re.fullmatch(r'[0-9a-f]{64}', authoritative_receipt_sha256)):
            raise StoreMismatch('Invalid authoritative receipt digest')
        started = time.perf_counter()
        removed = []
        event = {'status': 'cleaned',
                 'authoritative_receipt_sha256': authoritative_receipt_sha256,
                 'removed_files': removed}
        self._completed = True
        if self._disabled_error is not None:
            event.update(status='cleanup_skipped', cache_reason=self._disabled_reason,
                         cache_error=self._disabled_error, remaining_pair_bytes=None,
                         cleanup_seconds=time.perf_counter() - started)
            return event
        try:
            names = [path.name for path in self._list()
                     if _ENTRY.fullmatch(path.name) or _PARTIAL.fullmatch(path.name)]
            names.sort(key=lambda name: (not name.endswith('.json'), name))
            for name in names:
                self._remove(name)
                removed.append(name)
        except OSError as error:
            event.update(status='cleanup_incomplete', cache_error=f'{type(error).__name__}: {error}')
        try:
            event['remaining_pair_bytes'] = self._pair_bytes()
        except OSError as error:
            event.update(status='cleanup_incomplete', remaining_pair_bytes=None,
                         remaining_bytes_error=f'{type(error).__name__}: {error}')
        event['cleanup_seconds'] = time.perf_counter() - started
        return event
