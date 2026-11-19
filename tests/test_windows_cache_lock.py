"""Real Windows process contention on new and existing cache lock files."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


def worker(mode, folder, owner_pid, ordinal='0'):
    """Tiny subprocess payload; numerical arrays are never materialized here."""
    from ensemble_dwd._cache import _cached_base_stage, _key_lock
    folder = Path(folder)
    if mode in ('wait', 'owner'):
        (folder / ('ready-' + ordinal)).write_text('ready', encoding='utf-8')
        with _key_lock(folder / 'key.lock'):
            (folder / ('entered-' + ordinal)).write_text('entered', encoding='utf-8')
            if mode == 'owner':
                # The test explicitly terminates this exact owned child.
                time.sleep(90)
        print(json.dumps({'entered': True}), flush=True)
        return
    import psutil
    # parent() checks for PID reuse when it obtains the direct parent.
    # is_running() subsequently checks the identity of that same process.
    owner = psutil.Process().parent()
    if owner is None or owner.pid != int(owner_pid):
        raise RuntimeError('Cache worker lost its parent before readiness.')
    (folder / ('ready-' + ordinal)).write_text('ready', encoding='utf-8')
    start = folder / 'start'
    next_owner_check = time.monotonic() + .5
    while not start.is_file():
        # The parent owns the readiness deadline, including for early arrivals.
        if time.monotonic() >= next_owner_check:
            try:
                alive = owner.is_running() and owner.status() != psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                alive = False
            if not alive:
                if not start.is_file():
                    raise RuntimeError('Cache worker lost its parent before start.')
                break
            next_owner_check = time.monotonic() + .5
        time.sleep(.01)

    def compute():
        with (folder / 'computations.txt').open('a', encoding='utf-8') as stream:
            stream.write('computed\n')
            stream.flush()
        time.sleep(.1)
        return {'answer': 37, 'values': [1, 2, 3]}

    value, hit, key = _cached_base_stage(folder / 'cache', {'shared': 'cold-key'}, compute)
    print(json.dumps({'value': value, 'hit': hit, 'key': key}), flush=True)


@unittest.skipUnless(os.name == 'nt', 'Windows mandatory byte-range locking regression')
class WindowsCacheLockTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.processes = []
        self.addCleanup(self.cleanup_processes)

    def cleanup_processes(self):
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
            process.communicate(timeout=15)

    def spawn(self, mode, ordinal='0'):
        from process_test_support import popen_python

        environment = os.environ.copy()
        environment['PYTHONPATH'] = os.pathsep.join(sys.path)
        for name in ('MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                     'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            environment[name] = '1'
        process = popen_python(
            [str(Path(__file__).resolve()), '--worker', mode, str(self.folder), str(os.getpid()), str(ordinal)],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment,
            creationflags=subprocess.CREATE_NO_WINDOW)
        self.processes.append(process)
        return process

    def wait_for(self, paths, timeout=90):
        started = time.monotonic()
        while not all(path.exists() for path in paths):
            for process in self.processes:
                code = process.poll()
                if code is not None:
                    ordinal = process.args[-1]
                    ready = (self.folder / ('ready-' + ordinal)).exists()
                    missing = [path.name for path in paths if not path.exists()]
                    stdout, stderr = process.communicate(timeout=5)
                    self.fail(f'Worker {ordinal} (PID {process.pid}) exited with code {code}, '
                              f'ready={ready}, missing={missing}.\n'
                              f'stderr: {stderr[-4000:]}\nstdout: {stdout[-1000:]}')
            if time.monotonic() - started >= timeout:
                missing = [path.name for path in paths if not path.exists()]
                self.fail(f'Parent waited {timeout}s for worker readiness. Missing: {missing}')
            time.sleep(.01)

    def result(self, process):
        stdout, stderr = process.communicate(timeout=30)
        self.assertEqual(process.returncode, 0, stderr + stdout)
        return json.loads(stdout)

    def test_empty_file_lock_blocks_contender_without_any_prelock_write(self):
        import msvcrt
        path = self.folder / 'key.lock'
        with path.open('a+b') as held:
            held.seek(0)
            # Locking the region beyond EOF is supported by Windows. Holding
            # it makes the former pre-lock initialization flush fail reliably.
            msvcrt.locking(held.fileno(), msvcrt.LK_NBLCK, 1)
            process = self.spawn('wait')
            try:
                self.wait_for([self.folder / 'ready-0'])
                with self.assertRaises(subprocess.TimeoutExpired):
                    process.communicate(timeout=.3)
                self.assertFalse((self.folder / 'entered-0').exists())
                self.assertEqual(path.stat().st_size, 0)
            finally:
                held.seek(0)
                msvcrt.locking(held.fileno(), msvcrt.LK_UNLCK, 1)
        self.assertTrue(self.result(process)['entered'])
        self.assertEqual(path.stat().st_size, 0)

    def test_existing_nonempty_lock_file_remains_compatible(self):
        from ensemble_dwd._cache import _key_lock
        path = self.folder / 'key.lock'
        path.write_bytes(b'0')
        with _key_lock(path):
            process = self.spawn('wait')
            self.wait_for([self.folder / 'ready-0'])
            with self.assertRaises(subprocess.TimeoutExpired):
                process.communicate(timeout=.3)
        self.assertTrue(self.result(process)['entered'])
        self.assertEqual(path.read_bytes(), b'0')

    def test_os_releases_lock_after_owned_worker_exits_unexpectedly(self):
        owner = self.spawn('owner', 'owner')
        self.wait_for([self.folder / 'entered-owner'])
        waiter = self.spawn('wait', 'waiter')
        self.wait_for([self.folder / 'ready-waiter'])
        with self.assertRaises(subprocess.TimeoutExpired):
            waiter.communicate(timeout=.3)
        owner.terminate()
        owner.communicate(timeout=15)
        self.assertTrue(self.result(waiter)['entered'])

    def test_32_cold_key_processes_compute_once_and_read_identical_payload(self):
        processes = [self.spawn('cache', str(index)) for index in range(32)]
        self.wait_for([self.folder / ('ready-' + str(index)) for index in range(32)])
        (self.folder / 'start').touch(exist_ok=False)
        rows = [self.result(process) for process in processes]
        self.assertEqual((self.folder / 'computations.txt').read_text(encoding='utf-8').splitlines(), ['computed'])
        self.assertEqual(sum(not row['hit'] for row in rows), 1)
        self.assertEqual(sum(row['hit'] for row in rows), 31)
        self.assertEqual(len({row['key'] for row in rows}), 1)
        self.assertTrue(all(row['value'] == {'answer': 37, 'values': [1, 2, 3]} for row in rows))
        lock_files = list((self.folder / 'cache' / 'base-stage-v2').glob('*.lock'))
        self.assertEqual(len(lock_files), 1)
        self.assertEqual(lock_files[0].stat().st_size, 0)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        worker(*sys.argv[2:])
    else:
        unittest.main()
