"""Bounded Windows Wait Chain Traversal for the instrumented deep-temp probe."""
from contextlib import ExitStack
import ctypes as C
from ctypes import wintypes as W
import json
import os
from pathlib import Path
import subprocess
import sys


def record(path, **fields):
    with open(path, "a", encoding="utf-8") as stream:
        print(json.dumps({"native_wait": fields}), file=stream)


class LockInfo(C.Structure):
    _fields_ = [("name", W.WCHAR * 128), ("timeout", C.c_int64), ("alertable", W.BOOL)]


class ThreadInfo(C.Structure):
    _fields_ = [(name, W.DWORD) for name in ("pid", "tid", "wait_time", "switches")]


class Detail(C.Union):
    _fields_ = [("lock", LockInfo), ("thread", ThreadInfo)]


class WaitNode(C.Structure):
    _fields_ = [("kind", C.c_int), ("status", C.c_int), ("detail", Detail)]


KINDS = ("", "CriticalSection", "SendMessage", "Mutex", "ALPC", "COM", "ThreadWait",
         "ProcessWait", "Thread", "ComActivation", "Unknown", "SocketIo", "SmbIo")
STATES = ("", "NoAccess", "Running", "Blocked", "PidOnly", "PidOnlyRpcss", "Owned",
          "NotOwned", "Abandoned", "Unknown", "Error")


def bind(dll, name, result, *args):
    fn = getattr(dll, name)
    fn.restype, fn.argtypes = result, args
    return fn


def inspect(parent_pid, root_pid, birth, psutil_dir, log):
    sys.path.insert(0, psutil_dir)
    import psutil

    # wct.h uses two DWORD-sized enums and a naturally aligned native union.
    if (C.sizeof(W.DWORD), C.sizeof(W.WCHAR), C.sizeof(W.BOOL), LockInfo.timeout.offset,
            WaitNode.detail.offset, C.sizeof(WaitNode)) != (4, 2, 4, 256, 8, 280):
        raise RuntimeError("Unexpected WAITCHAIN_NODE_INFO layout")
    kernel = C.WinDLL("kernel32", use_last_error=True)
    wct = C.WinDLL("advapi32", use_last_error=True)
    open_proc = bind(kernel, "OpenProcess", W.HANDLE, W.DWORD, W.BOOL, W.DWORD)
    open_thread = bind(kernel, "OpenThread", W.HANDLE, W.DWORD, W.BOOL, W.DWORD)
    owner = bind(kernel, "GetProcessIdOfThread", W.DWORD, W.HANDLE)
    wait = bind(kernel, "WaitForSingleObject", W.DWORD, W.HANDLE, W.DWORD)
    close = bind(kernel, "CloseHandle", W.BOOL, W.HANDLE)
    start = bind(wct, "OpenThreadWaitChainSession", W.HANDLE, W.DWORD, C.c_void_p)
    finish = bind(wct, "CloseThreadWaitChainSession", None, W.HANDLE)
    query = bind(wct, "GetThreadWaitChain", W.BOOL, W.HANDLE, C.c_size_t, W.DWORD, W.DWORD,
                 C.POINTER(W.DWORD), C.POINTER(WaitNode), C.POINTER(W.BOOL))
    root = psutil.Process(root_pid)
    with ExitStack() as cleanup:
        root_handle = open_proc(0x101000, False, root_pid)  # query-limited | synchronize
        if not root_handle:
            raise C.WinError(C.get_last_error())
        cleanup.callback(close, root_handle)
        if (root.ppid() != parent_pid or root.create_time() != birth
                or wait(root_handle, 0) != 258):
            raise RuntimeError("Original probe process no longer live")
        args = root.cmdline()
        if not ("-c" in args and "from path_support import extended_temporary_environment"
                in args[args.index("-c") + 1]):
            raise RuntimeError("Original probe command not identified")
        session = start(0, None)
        if not session:
            raise C.WinError(C.get_last_error())
        cleanup.callback(finish, session)
        # The venv redirector starts the interpreter as its direct child.
        for proc in [root, *root.children()]:
            with ExitStack() as hold:
                h = open_proc(0x101000, False, proc.pid)
                if not h:
                    continue
                hold.callback(close, h)
                if (not proc.is_running() or wait(root_handle, 0) != 258
                        or wait(h, 0) != 258 or proc.name().lower() != "python.exe"):
                    continue
                if proc.pid != root_pid and (proc.ppid() != root_pid or proc.create_time() < birth):
                    continue
                for entry in proc.threads():
                    with ExitStack() as pin:
                        th = open_thread(0x100800, False, entry.id)  # query-limited | synchronize
                        if not th:
                            continue
                        pin.callback(close, th)
                        if owner(th) != proc.pid or wait(th, 0) != 258 or wait(h, 0) != 258:
                            continue
                        nodes = (WaitNode * 16)()
                        count, cycle = W.DWORD(16), W.BOOL()
                        # No out-of-process traversal past the first blocking object.
                        ok = query(session, 0, 0, entry.id, C.byref(count), nodes, C.byref(cycle))
                        if not ok:
                            record(log, pid=proc.pid, tid=entry.id, winerror=C.get_last_error())
                            continue
                        if wait(th, 0) != 258 or wait(h, 0) != 258:
                            continue
                        chain = []
                        for node in nodes[:count.value]:
                            kind, status = node.kind, node.status
                            item = {"kind": KINDS[kind] if 0 < kind < len(KINDS) else kind,
                                    "state": STATES[status] if 0 < status < len(STATES) else status}
                            if kind == 8:
                                item.update(pid=node.detail.thread.pid, tid=node.detail.thread.tid)
                            elif node.detail.lock.name:
                                item["object"] = node.detail.lock.name
                            chain.append(item)
                        record(log, pid=proc.pid, tid=entry.id, cycle=bool(cycle), nodes=chain)


def sample_waits(root_pid, log_path):
    """Sample in a separate worker so an unresponsive OS query cannot delay the test."""
    try:
        import _winapi
        import psutil

        root = psutil.Process(root_pid)
        if root.ppid() != os.getpid():
            raise RuntimeError("Expected a direct probe child")
        executable = sys._base_executable
        args = [executable, "-I", "-S", "-B", str(Path(__file__).resolve()),
                str(os.getpid()), str(root_pid), repr(root.create_time()),
                str(Path(psutil.__file__).resolve().parent.parent), str(Path(log_path).resolve())]
        # The observer wraps subprocess.Popen. Launch independently of that hook.
        proc, thread, _, _ = _winapi.CreateProcess(
            executable, subprocess.list2cmdline(args), None, None, False,
            0x08000000, None, None, subprocess.STARTUPINFO())
        _winapi.CloseHandle(thread)
        try:
            if _winapi.WaitForSingleObject(proc, 7000) == _winapi.WAIT_TIMEOUT:
                _winapi.TerminateProcess(proc, 1)  # Only our sampler, by its owned handle.
                _winapi.WaitForSingleObject(proc, 500)
                record(log_path, sampler="deadline")
            elif _winapi.GetExitCodeProcess(proc):
                record(log_path, sampler="worker failed")
        finally:
            _winapi.CloseHandle(proc)
    except Exception as exc:
        record(log_path, sampler="unavailable", error=type(exc).__name__, detail=str(exc))


if __name__ == "__main__":
    parent, pid, birth, psutil_dir, logfile = sys.argv[1:]
    try:
        inspect(int(parent), int(pid), float(birth), psutil_dir, logfile)
    except Exception as exc:
        record(logfile, sampler="unavailable", error=type(exc).__name__, detail=str(exc))
