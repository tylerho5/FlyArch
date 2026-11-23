"""Run lock and run-directory control files for training runs.

Stdlib-only and Python 3.9 compatible, so the host launcher, ``train/`` and
``scripts/stop_run.py`` can all import it. A run directory holds ``.lock``
(held while a process trains), ``state.json`` (the run's last status) and
``stop-request.json`` (a pause or cancel request for the live run).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import socket
import threading
import time
from pathlib import Path
from typing import Any, Optional

LOCK_FILE = ".lock"
STATE_FILE = "state.json"
REQUEST_FILE = "stop-request.json"
REASON_MAX = 200
END_STATUSES = ("finished", "stopped_early", "crashed", "paused", "cancelled")
REQUEST_STATUSES = ("paused", "cancelled")
REPLACE_ATTEMPTS = 50
REPLACE_WAIT_S = 0.02

log = logging.getLogger(__name__)


class RunLocked(RuntimeError):
    """Another process holds the run directory's lock."""


def _lock_fd(fd: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_fd(fd: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


def _holder(path: Path) -> str:
    try:
        info = json.loads(path.read_text())
        return f" (pid {info['pid']} on {info['host']})"
    except (OSError, ValueError, KeyError, TypeError):
        return ""


class RunLock:
    """Exclusive, non-blocking lock on ``run_dir/.lock``; never deletes the file."""

    def __init__(self, run_dir: Path, name: str = LOCK_FILE) -> None:
        self.path = Path(run_dir) / name
        self._fd: Optional[int] = None

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            _lock_fd(fd)
        except OSError:
            os.close(fd)
            raise RunLocked(
                f"run directory {self.path.parent} is locked by another process"
                + _holder(self.path)
            ) from None
        try:
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, json.dumps({"pid": os.getpid(), "host": socket.gethostname()}).encode())
        except OSError:
            pass
        self._fd = fd
        return self

    def __exit__(self, *exc: object) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            _unlock_fd(fd)
        finally:
            os.close(fd)


@contextlib.contextmanager
def blocking_lock(path: Path, poll_s: float = 0.1):
    """Exclusive lock on ``path`` that waits for its holder; never deletes the file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        while True:
            try:
                _lock_fd(fd)
                break
            except OSError:
                time.sleep(poll_s)
        try:
            yield
        finally:
            _unlock_fd(fd)
    finally:
        os.close(fd)


def is_locked(run_dir: Path, name: str = LOCK_FILE) -> bool:
    path = Path(run_dir) / name
    try:
        fd = os.open(path, os.O_RDWR if os.name == "nt" else os.O_RDONLY)
    except OSError:
        return False
    try:
        try:
            _lock_fd(fd)
        except OSError:
            return True
        _unlock_fd(fd)
        return False
    finally:
        os.close(fd)


def clean_reason(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    text = text.replace("\x00", "").strip()[:REASON_MAX].strip()
    return text or None


def replace(src: Path, dst: Path) -> None:
    """``os.replace`` that waits out a Windows reader holding ``dst`` or ``src`` open."""
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if os.name != "nt" or attempt == REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(REPLACE_WAIT_S)


def write_json(path: Path, record: dict, indent: Optional[int] = None) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(record, indent=indent))
        replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def read_json(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def read_state(run_dir: Path) -> Optional[dict]:
    return read_json(Path(run_dir) / STATE_FILE)


def write_state(run_dir: Path, status: str, **fields: Any) -> None:
    record = {"status": status, "t": time.time()}
    record.update({k: v for k, v in fields.items() if v is not None})
    write_json(Path(run_dir) / STATE_FILE, record)


def read_request(run_dir: Path) -> Optional[dict]:
    request = read_json(Path(run_dir) / REQUEST_FILE)
    if request is None or request.get("status") not in REQUEST_STATUSES:
        return None
    return request


def write_request(run_dir: Path, status: str, reason: Optional[str] = None) -> None:
    if status not in REQUEST_STATUSES:
        raise ValueError(f"request status must be one of {REQUEST_STATUSES}, got {status!r}")
    record: dict = {"status": status, "requested_t": time.time()}
    reason = clean_reason(reason)
    if reason is not None:
        record["reason"] = reason
    write_json(Path(run_dir) / REQUEST_FILE, record)


def clear_request(run_dir: Path) -> None:
    try:
        (Path(run_dir) / REQUEST_FILE).unlink()
    except FileNotFoundError:
        pass


class RunStopped(BaseException):
    """A run must stop now; ``status`` is the state it ends in."""

    def __init__(self, status: str, *, signum: Optional[int] = None,
                 reason: Optional[str] = None) -> None:
        super().__init__(status)
        self.status = status
        self.signum = signum
        self.reason = reason
        self.resume_epoch: Optional[int] = None
        self.run_dir: Optional[Path] = None


def stop_signals() -> tuple:
    if os.name == "nt":
        return (signal.SIGINT, signal.SIGBREAK)
    return (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


class StopSignals:
    """Turns the first stop signal into ``RunStopped("paused")`` on the main thread."""

    def __init__(self) -> None:
        self.fired: Optional[int] = None
        self._held = False
        self._ignoring = False
        self._depth = 0
        self._previous: dict = {}

    def __enter__(self) -> "StopSignals":
        if threading.current_thread() is threading.main_thread():
            for signum in stop_signals():
                if signal.getsignal(signum) == signal.SIG_IGN:
                    continue
                self._previous[signum] = signal.signal(signum, self._handle)
        return self

    def __exit__(self, *exc: object) -> None:
        self.ignore()
        previous, self._previous = self._previous, {}
        for signum, handler in previous.items():
            signal.signal(signum, handler)

    def ignore(self) -> None:
        self._ignoring = True
        self._held = False

    def _handle(self, signum: int, frame: object) -> None:
        if self._ignoring or self.fired is not None:
            log.warning("signal %s ignored (first signal: %s)", signum, self.fired)
            return
        self.fired = signum
        if self._depth > 0:
            self._held = True
            return
        raise RunStopped("paused", signum=signum)

    @contextlib.contextmanager
    def deferred(self):
        self._depth += 1
        try:
            yield
        finally:
            self._depth -= 1
        if self._depth == 0 and self._held:
            self._held = False
            raise RunStopped("paused", signum=self.fired)
