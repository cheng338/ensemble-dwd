"""Checkpoint entry overlap controls; genuine loky tests use no fitted models."""
from contextlib import ExitStack
import functools
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from types import CodeType, FunctionType, SimpleNamespace

import pytest


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope='session')
def sources():
    from test_execution_runtime_roles import cp
    path = (Path(__file__).resolve().parents[1] / 'examples' / 'cv_checkpoint.py').resolve()
    assert Path(cp.__file__).resolve() == path
    source_pin = digest(path)
    yield {'path': str(path), 'sha256': source_pin, 'module': cp}
    assert digest(path) == source_pin


def extracted(config, namespace):
    """Run the actual method code with explicit, isolated control globals."""
    namespace.update(ExitStack=ExitStack, sys=sys, os=os)
    for class_name, methods in (
            ('_EntryImportPreparation', ('__init__', 'start', 'probe', 'close')),
            ('CheckpointedCV', ('__enter__', '__exit__'))):
        actual_class = getattr(config['module'], class_name)
        attributes = {'__module__': actual_class.__module__}
        for method in methods:
            actual = getattr(actual_class, method)
            assert isinstance(actual, FunctionType)
            assert actual.__closure__ is None and not actual.__code__.co_freevars
            controlled = FunctionType(actual.__code__, namespace,
                                      name=actual.__name__, argdefs=actual.__defaults__)
            controlled.__kwdefaults__ = (None if actual.__kwdefaults__ is None
                                         else dict(actual.__kwdefaults__))
            controlled.__annotations__ = dict(actual.__annotations__)
            controlled.__qualname__ = actual.__qualname__
            controlled.__module__ = actual.__module__
            attributes[method] = controlled
        namespace[class_name] = type(class_name, (), attributes)
    return namespace


class TestCheckpointError(RuntimeError):
    __test__ = False


@pytest.fixture
def harness(sources, tmp_path):
    events, faults = [], {}
    state = SimpleNamespace(parent_calls=0, active_limits=0, saved=None, instance=None)

    class Context:
        def __init__(self, name):
            self.name = name
        def __enter__(self):
            events.append(('enter', self.name))
            if self.name == 'limits':
                state.active_limits += 1
            return self
        def __exit__(self, *args):
            events.append(('exit', self.name))
            if self.name == 'limits':
                state.active_limits -= 1
            if self.name == 'limits' and faults.get('limit_exit'):
                raise faults['limit_exit']

    def fingerprint():
        state.parent_calls += 1
        events.append(('parent', state.parent_calls))
        if faults.get('parent'):
            raise faults['parent']
        return {'runtime': 'changed' if state.parent_calls > 1 and faults.get('post_drift') else 'same'}

    def imports():
        events.append(('import_body',))
        return 'wrong' if faults.get('import_result') else None

    def authority(estimator):
        events.append(('authority_body', estimator))
        return {'runtime': 'same'}

    class Results:
        def __init__(self, tasks, kind):
            self.tasks, self.kind, self.index = tasks, kind, 0
        def __iter__(self):
            return self
        def __next__(self):
            if self.index == len(self.tasks):
                raise StopIteration
            error = faults.get('drain_' + self.kind)
            if error:
                raise error
            function, args, kwargs = self.tasks[self.index]
            self.index += 1
            return function(*args, **kwargs)
        def close(self):
            events.append(('generator_close', self.kind))
            if faults.get('generator_close'):
                raise faults['generator_close']

    class Pool:
        def __init__(self, **kwargs):
            events.append(('pool_construct', kwargs))
            assert kwargs == dict(n_jobs=state.instance.jobs, pre_dispatch=state.instance.jobs,
                                  batch_size=1, return_as='generator')
        def __enter__(self):
            events.append(('pool_enter',))
            if faults.get('pool_enter'):
                raise faults['pool_enter']
            return self
        def __exit__(self, *args):
            events.append(('pool_exit',))
            if faults.get('pool_exit'):
                raise faults['pool_exit']
        def __call__(self, tasks):
            tasks = list(tasks)
            kind = 'imports' if all(task[0] is imports for task in tasks) else 'authority'
            events.append(('submit', kind, len(tasks)))
            if kind == 'authority':
                assert state.parent_calls == 1
            if faults.get('dispatch_' + kind):
                raise faults['dispatch_' + kind]
            return Results(tasks, kind)

    def legacy(estimator, *, jobs):
        events.append(('legacy_probe', jobs))
        return {'runtime': 'same'}

    def write_receipt(path, value):
        events.append(('write', path.name))
        if path.name == 'identity.json':
            state.saved = value['identity']

    ns = dict(CheckpointError=TestCheckpointError, Parallel=Pool,
        delayed=lambda function: lambda *args, **kwargs: (function, args, kwargs),
        parallel_config=lambda **kwargs: Context('config'), threadpool_limits=lambda **kwargs: Context('limits'),
        _execution_import_probe=imports, _execution_runtime_probe=authority, _probe_execution_runtime=legacy,
        _Lease=lambda path: Context('lease'), _stable=lambda value: value, _canonical=lambda value: value,
        runtime_fingerprint=fingerprint, _require_execution_compatible=lambda worker, parent: events.append(('compatible', worker == parent)),
        _saved_identity=lambda path: ({'runtime': {'runtime': 'same'}}, None),
        _read_receipt=lambda path, pin: {'identity': state.instance.identity},
        _write_receipt=write_receipt, write_json_atomic=lambda path, value, **kwargs: events.append(('write', path.name)),
        write_failure_json=lambda path, value, **kwargs: events.append(('inactive', path.name)),
        data_fingerprint=lambda *args: 'data', array_fingerprint=lambda value: 'array',
        _digest=lambda value: 'digest', _RUNTIME_ROLE_POLICY='role', SCHEMA=2,
        ACCURACY_AGGREGATION='exact-mean-seeds-within-fold-then-folds-v1',
        _now=lambda: 'now', uuid=SimpleNamespace(uuid4=lambda: SimpleNamespace(hex='token')))
    extracted(sources, ns)

    def make(jobs=3, resume=False, existing=False):
        instance = ns['CheckpointedCV']()
        instance.__dict__.update(folder=tmp_path/'cv-checkpoints', lease=None, token=None, identity=None,
            jobs=jobs, resume=resume, estimator='estimator', X='X', y='y', folds=[('train', 'validation')],
            error_policy='nan', provenance={}, pair_recovery=False, reuse_equivalent_folds=False, cv_seeds=None)
        state.instance = instance
        if existing:
            instance.folder.mkdir(parents=True, exist_ok=True)
            (instance.folder/'identity.json').write_text('{}')
        return instance
    return SimpleNamespace(ns=ns, events=events, faults=faults, state=state, make=make)


def test_exact_public_entry_order_and_task_counts(harness):
    context = harness.make()
    assert context.__enter__() is context
    events = harness.events
    assert [event for event in events if event[0] == 'submit'] == [('submit', 'imports', 2), ('submit', 'authority', 1)]
    assert events.index(('submit', 'imports', 2)) < events.index(('parent', 1)) < events.index(('submit', 'authority', 1))
    assert events.index(('parent', 2)) < events.index(('pool_exit',)) < events.index(('write', 'identity.json'))
    assert not any(event[0] == 'legacy_probe' for event in events)
    assert harness.state.active_limits == 0
    context.__exit__(None, None, None)
    assert context.lease is None


@pytest.mark.parametrize('jobs,resume,existing,error', [(1, False, False, None), (3, True, True, None),
    (3, False, True, 'already exists'), (3, True, False, 'No checkpoint')])
def test_serial_resume_and_existing_paths_do_not_prepare(harness, jobs, resume, existing, error):
    context = harness.make(jobs, resume, existing)
    if error:
        with pytest.raises(TestCheckpointError, match=error):
            context.__enter__()
    else:
        context.__enter__()
        context.__exit__(None, None, None)
    assert not any(event[0] in ('submit', 'pool_construct') for event in harness.events)
    assert harness.state.parent_calls == 1


@pytest.mark.parametrize('stage', ['pool_enter', 'dispatch_imports', 'drain_imports', 'dispatch_authority', 'drain_authority'])
def test_failure_never_retries_and_releases_lease(harness, stage):
    context = harness.make()
    error = RuntimeError(stage)
    harness.faults[stage] = error
    with pytest.raises(RuntimeError) as caught:
        context.__enter__()
    assert caught.value is error and context.lease is None
    assert harness.state.parent_calls == 1
    assert sum(event[0] == 'pool_enter' for event in harness.events) == 1
    assert sum(event[0] == 'pool_exit' for event in harness.events) == 1
    assert not any(event[0] == 'write' for event in harness.events)
    assert harness.state.active_limits == 0


@pytest.mark.parametrize('error', [MemoryError('preparation allocation'), OSError('preparation backend')])
def test_preparation_resource_error_waits_for_original_parent_check(harness, error):
    context = harness.make()
    harness.faults['pool_enter'] = error
    with pytest.raises(type(error)) as caught:
        context.__enter__()
    assert caught.value is error and harness.state.parent_calls == 1
    assert context.lease is None and harness.state.active_limits == 0


@pytest.mark.parametrize('startup', [None, 'pool_enter', 'dispatch_imports'])
def test_parent_error_has_precedence_over_preparation_and_cleanup(harness, startup):
    context = harness.make()
    parent = ValueError('original parent fingerprint failed')
    harness.faults.update(parent=parent, pool_exit=RuntimeError('pool cleanup failed'))
    if startup:
        harness.faults[startup] = RuntimeError('preparation failed first')
    else:
        harness.faults['generator_close'] = RuntimeError('generator cleanup failed')
    with pytest.raises(ValueError) as caught:
        context.__enter__()
    assert caught.value is parent and context.lease is None
    assert harness.state.active_limits == 0
    assert any('cleanup failed' in note for note in parent.__notes__)
    assert not any(event[0] == 'write' for event in harness.events)


@pytest.mark.parametrize('error', [KeyboardInterrupt('interrupt'), SystemExit(17)])
def test_start_interruption_cleans_without_delayed_parent_work(harness, error):
    context = harness.make()
    harness.faults['pool_enter'] = error
    with pytest.raises(type(error)) as caught:
        context.__enter__()
    assert caught.value is error and harness.state.parent_calls == 0
    assert context.lease is None and harness.state.active_limits == 0
    assert sum(event[0] == 'pool_exit' for event in harness.events) == 1


@pytest.mark.parametrize('fault', ['post_drift', 'import_result', 'pool_exit'])
def test_late_failure_creates_no_identity(harness, fault):
    context = harness.make()
    harness.faults[fault] = RuntimeError('pool exit') if fault == 'pool_exit' else True
    with pytest.raises(RuntimeError):
        context.__enter__()
    assert context.lease is None and not any(event[0] == 'write' for event in harness.events)
    assert harness.state.active_limits == 0


@pytest.mark.parametrize('later_error', [False, True])
def test_early_exists_oserror_is_only_hint(harness, monkeypatch, later_error):
    context = harness.make()
    original = Path.exists
    observations = []
    def observe(path):
        if path == context.folder/'identity.json':
            observations.append(harness.state.parent_calls)
            if len(observations) == 1 or later_error:
                raise PermissionError('manifest lookup')
        return original(path)
    monkeypatch.setattr(Path, 'exists', observe)
    if later_error:
        with pytest.raises(PermissionError):
            context.__enter__()
    else:
        context.__enter__()
        context.__exit__(None, None, None)
    assert observations[:2] == [0, 1]
    assert not any(event[0] == 'pool_construct' for event in harness.events)


def test_manifest_appearing_after_hint_retains_authoritative_refusal(harness, monkeypatch):
    context = harness.make()
    original = Path.exists
    calls = []
    def exists(path):
        if path == context.folder/'identity.json':
            calls.append(harness.state.parent_calls)
            return len(calls) > 1
        return original(path)
    monkeypatch.setattr(Path, 'exists', exists)
    with pytest.raises(TestCheckpointError, match='already exists'):
        context.__enter__()
    assert calls == [0, 1] and context.lease is None
    assert not any(event[:2] == ('submit', 'authority') for event in harness.events)


def _direct_code(code, name):
    matches = [value for value in code.co_consts
               if isinstance(value, CodeType) and value.co_name == name]
    assert len(matches) == 1
    return matches[0]


def test_control_extraction_matches_repository_definitions(sources):
    namespace = extracted(sources, {})
    module_code = compile(Path(sources['path']).read_bytes(), sources['path'], 'exec',
                          dont_inherit=True, optimize=sys.flags.optimize)
    for name, methods in (
            ('_EntryImportPreparation', ('__init__', 'start', 'probe', 'close')),
            ('CheckpointedCV', ('__enter__', '__exit__'))):
        actual_class = getattr(sources['module'], name)
        class_code = _direct_code(module_code, name)
        for method in methods:
            controlled = getattr(namespace[name], method)
            actual = getattr(actual_class, method)
            assert actual.__code__ == _direct_code(class_code, method)
            assert controlled.__code__ is actual.__code__
            assert controlled.__globals__ is namespace
            assert controlled.__globals__ is not actual.__globals__
            assert controlled.__defaults__ == actual.__defaults__
            assert controlled.__kwdefaults__ == actual.__kwdefaults__
            assert controlled.__annotations__ == actual.__annotations__
            assert controlled.__qualname__ == actual.__qualname__


def _record(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream)
    assert not path.exists(), 'unexpected repeated control event'
    temporary.replace(path)


def _real_import_task(directory):
    import psutil
    from threadpoolctl import threadpool_info
    folder = Path(directory)
    process = psutil.Process()
    pools = threadpool_info()
    assert all(pool['num_threads'] == 1 for pool in pools)
    _record(folder/'import-start.json', dict(pid=process.pid, create_time=process.create_time(), pools=pools))
    deadline = time.monotonic() + 30
    while not (folder/'release').exists():
        if time.monotonic() > deadline:
            raise RuntimeError('bounded synthetic import was not released')
        time.sleep(.01)
    return None


class SerializationMarker:
    def __init__(self, directory):
        self.directory = str(directory)
    def __getstate__(self):
        folder = Path(self.directory)
        assert (folder/'parent-qualified').is_file(), 'estimator serialized before parent proof'
        _record(folder/'serialized.json', dict(pid=os.getpid(), monotonic=time.monotonic()))
        return self.__dict__
    def __setstate__(self, state):
        self.__dict__.update(state)
        _record(Path(self.directory)/'deserialized.json', dict(pid=os.getpid(), monotonic=time.monotonic()))


def _real_authority_task(estimator):
    from threadpoolctl import threadpool_info
    folder = Path(estimator.directory)
    assert (folder/'parent-qualified').is_file()
    assert all(pool['num_threads'] == 1 for pool in threadpool_info())
    _record(folder/'authority.json', dict(pid=os.getpid(), monotonic=time.monotonic()))
    return {'synthetic_authority': 'same'}


def _real_namespace(sources, folder):
    from joblib import Parallel, delayed, parallel_config
    from threadpoolctl import threadpool_limits
    return extracted(sources, dict(CheckpointError=TestCheckpointError, Parallel=Parallel, delayed=delayed,
        parallel_config=parallel_config, threadpool_limits=threadpool_limits,
        _execution_import_probe=functools.partial(_real_import_task, str(folder)),
        _execution_runtime_probe=_real_authority_task,
        _probe_execution_runtime=lambda *args, **kwargs: pytest.fail('unexpected legacy path')))


def _wait(path):
    deadline = time.monotonic() + 30
    while not path.is_file():
        if time.monotonic() > deadline:
            raise AssertionError('real synthetic import never started')
        time.sleep(.01)
    return json.loads(path.read_text())


def _worker_identities(preparation):
    import psutil
    # Private inspection is confined to this diagnostic test, never production.
    processes = preparation.pool._backend._workers._processes
    return [{'pid': pid, 'create_time': psutil.Process(pid).create_time()} for pid in processes]


def _assert_closed(identities):
    import psutil
    for identity in identities:
        try:
            assert psutil.Process(identity['pid']).create_time() != identity['create_time'], identity
        except psutil.NoSuchProcess:
            pass


def _shutdown_test_executor():
    module = sys.modules.get('joblib.externals.loky.reusable_executor')
    executor = getattr(module, '_executor', None) if module is not None else None
    if executor is not None:
        executor.shutdown(wait=True, kill_workers=False)


def _parent_state():
    from threadpoolctl import threadpool_info
    return (dict(os.environ), sorted((pool['filepath'], pool['num_threads']) for pool in threadpool_info()))


@pytest.mark.parametrize('primary', [ValueError('parent refuses'), KeyboardInterrupt('parent interrupted')])
def test_real_pending_import_cancellation_has_native_closure(sources, tmp_path, primary):
    ns = _real_namespace(sources, tmp_path)
    before = _parent_state()
    preparation = ns['_EntryImportPreparation'](tmp_path/'identity.json', jobs=2, resume=False)
    identities = []
    try:
        preparation.start()
        assert preparation.failure is None
        witness = _wait(tmp_path/'import-start.json')
        identities = _worker_identities(preparation)
        assert any(row['pid'] == witness['pid'] and row['create_time'] == witness['create_time'] for row in identities)
        with pytest.raises(type(primary)) as caught:
            try:
                raise primary
            except BaseException as error:
                preparation.close(primary=error)
                raise
        assert caught.value is primary
        _assert_closed(identities)
        assert _parent_state() == before
        assert not (tmp_path/'authority.json').exists()
        from joblib import Parallel, delayed, parallel_config
        with parallel_config(backend='loky', inner_max_num_threads=1):
            assert Parallel(n_jobs=2)(delayed(abs)(-1) for _ in range(2)) == [1, 1]
        _record(tmp_path/'lifecycle-witness.json', dict(known_workers=identities, closed=True,
            authority_calls=0, subsequent_executor_usable=True, synthetic_ping_tasks=2))
    finally:
        (tmp_path/'release').touch(exist_ok=True)
        try:
            preparation.close()
        finally:
            _shutdown_test_executor()


def test_real_serialization_order_and_executor_usable_after_success(sources, tmp_path):
    identities = []
    before = None
    try:
        for index in range(2):
            folder = tmp_path/str(index)
            folder.mkdir()
            ns = _real_namespace(sources, folder)
            if before is None:
                before = _parent_state()
            preparation = ns['_EntryImportPreparation'](folder/'identity.json', jobs=2, resume=False)
            try:
                preparation.start()
                assert preparation.failure is None
                _wait(folder/'import-start.json')
                identities.extend(_worker_identities(preparation))
                assert not (folder/'serialized.json').exists()
                (folder/'parent-qualified').touch()
                (folder/'release').touch()
                assert preparation.probe(SerializationMarker(folder)) == {'synthetic_authority': 'same'}
                assert json.loads((folder/'serialized.json').read_text())['pid'] == os.getpid()
                assert json.loads((folder/'authority.json').read_text())['pid'] != os.getpid()
                assert (folder/'deserialized.json').is_file()
            finally:
                (folder/'release').touch(exist_ok=True)
                preparation.close()
            assert _parent_state() == before
        _record(tmp_path/'reuse-witness.json', dict(successful_entries=2, serialized_payloads=2, fitted_models=0))
    finally:
        _shutdown_test_executor()
        _assert_closed(identities)


def test_real_effective_sequential_backend_preserves_proof_order(sources, tmp_path):
    process = multiprocessing.current_process()
    previous = process.daemon
    ns = _real_namespace(sources, tmp_path)
    preparation = ns['_EntryImportPreparation'](tmp_path/'identity.json', jobs=2, resume=False)
    (tmp_path/'release').touch()
    try:
        process.daemon = True  # Process-local diagnostic control, restored below.
        with pytest.warns(UserWarning, match='n_jobs=1'):
            preparation.start()
        assert preparation.failure is None
        assert not (tmp_path/'import-start.json').exists()
        (tmp_path/'parent-qualified').touch()
        assert preparation.probe(SerializationMarker(tmp_path)) == {'synthetic_authority': 'same'}
        assert json.loads((tmp_path/'authority.json').read_text())['pid'] == os.getpid()
        assert not (tmp_path/'serialized.json').exists()
    finally:
        try:
            preparation.close()
        finally:
            try:
                process.daemon = previous
            finally:
                _shutdown_test_executor()
