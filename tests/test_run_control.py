from __future__ import annotations

import json
import logging
import multiprocessing
import os
import signal
import sys
import threading
import time
import types

import pytest

from src import run_control
from src.run_control import (
    REASON_MAX,
    RunLock,
    RunLocked,
    RunStopped,
    StopSignals,
    clean_reason,
    clear_request,
    is_locked,
    read_request,
    read_state,
    stop_signals,
    write_request,
    write_state,
)


def _try_lock(run_dir, queue):
    try:
        with RunLock(run_dir):
            queue.put(("ok", ""))
    except RunLocked as exc:
        queue.put(("locked", str(exc)))


def _probe(run_dir, queue):
    queue.put(is_locked(run_dir))


def _in_child(target, run_dir):
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=target, args=(run_dir, queue))
    proc.start()
    result = queue.get(timeout=30)
    proc.join(30)
    return result


def test_second_process_is_refused_then_succeeds_after_release(tmp_path):
    with RunLock(tmp_path):
        outcome, message = _in_child(_try_lock, tmp_path)
        assert outcome == "locked"
        assert str(os.getpid()) in message
        assert str(tmp_path) in message
    outcome, _ = _in_child(_try_lock, tmp_path)
    assert outcome == "ok"
    assert (tmp_path / ".lock").exists()


def test_blocking_lock_waits_for_the_holder(tmp_path, monkeypatch):
    attempts = []
    lock_fd = run_control._lock_fd

    def busy_then_free(fd):
        attempts.append(fd)
        if len(attempts) <= 3:
            raise OSError("busy")
        lock_fd(fd)

    monkeypatch.setattr(run_control, "_lock_fd", busy_then_free)
    monkeypatch.setattr(run_control.time, "sleep", lambda seconds: None)
    with run_control.blocking_lock(tmp_path / "results" / ".lock"):
        assert len(attempts) == 4
    assert (tmp_path / "results" / ".lock").exists()


def test_is_locked_tracks_the_holder(tmp_path):
    assert _in_child(_probe, tmp_path) is False
    with RunLock(tmp_path):
        assert _in_child(_probe, tmp_path) is True
    assert _in_child(_probe, tmp_path) is False


def test_is_locked_creates_nothing(tmp_path):
    missing = tmp_path / "absent"
    assert is_locked(missing) is False
    assert not missing.exists()
    assert is_locked(tmp_path) is False
    assert not (tmp_path / ".lock").exists()


def test_windows_branch_maps_oserror_to_run_locked(tmp_path, monkeypatch):
    calls = []

    def locking(fd, mode, nbytes):
        calls.append((mode, nbytes))
        if mode == "NBLCK":
            raise OSError("locked")

    fake = types.SimpleNamespace(LK_NBLCK="NBLCK", LK_UNLCK="UNLCK", locking=locking)
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    lock = RunLock(tmp_path)
    monkeypatch.setattr(os, "name", "nt")
    with pytest.raises(RunLocked):
        lock.__enter__()
    assert calls == [("NBLCK", 1)]


def test_windows_branch_locks_and_unlocks(tmp_path, monkeypatch):
    calls = []
    fake = types.SimpleNamespace(
        LK_NBLCK="NBLCK",
        LK_UNLCK="UNLCK",
        locking=lambda fd, mode, nbytes: calls.append((mode, nbytes)),
    )
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    lock = RunLock(tmp_path)
    monkeypatch.setattr(os, "name", "nt")
    with lock:
        pass
    assert calls == [("NBLCK", 1), ("UNLCK", 1)]


def test_state_round_trip_drops_none(tmp_path):
    write_state(tmp_path, "paused", resume_epoch=4, reason=None)
    state = read_state(tmp_path)
    assert state["status"] == "paused"
    assert state["resume_epoch"] == 4
    assert "reason" not in state
    assert isinstance(state["t"], float)
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_read_state_missing_or_truncated(tmp_path):
    assert read_state(tmp_path) is None
    (tmp_path / "state.json").write_text('{"status": "pau')
    assert read_state(tmp_path) is None


def test_write_failure_leaves_no_tmp(tmp_path):
    with pytest.raises(TypeError):
        write_state(tmp_path, "paused", bad=object())
    assert list(tmp_path.iterdir()) == []


def test_request_rejects_end_status_and_replaces(tmp_path):
    with pytest.raises(ValueError):
        write_request(tmp_path, "finished")
    write_request(tmp_path, "paused", "first")
    write_request(tmp_path, "cancelled", "second")
    request = read_request(tmp_path)
    assert request["status"] == "cancelled"
    assert request["reason"] == "second"
    assert "requested_t" in request


def test_request_malformed_or_unknown(tmp_path):
    assert read_request(tmp_path) is None
    (tmp_path / "stop-request.json").write_text("{")
    assert read_request(tmp_path) is None
    (tmp_path / "stop-request.json").write_text(json.dumps({"status": "finished"}))
    assert read_request(tmp_path) is None


def test_clear_request_tolerates_missing(tmp_path):
    clear_request(tmp_path)
    write_request(tmp_path, "paused")
    clear_request(tmp_path)
    assert read_request(tmp_path) is None


def test_clean_reason():
    assert clean_reason("x" * 300) == "x" * REASON_MAX
    assert clean_reason("  a\x00b  ") == "ab"
    assert clean_reason("  \x00 ") is None
    assert clean_reason(None) is None
    assert run_control.END_STATUSES[0] == "finished"


posix_only = pytest.mark.skipif(os.name == "nt", reason="sends POSIX signals")


def _handlers():
    return {s: signal.getsignal(s) for s in stop_signals()}


def _kill(sig):
    os.kill(os.getpid(), sig)
    time.sleep(0.01)


@posix_only
def test_sigterm_raises_paused_and_restores_handlers():
    before = _handlers()
    with pytest.raises(RunStopped) as info:
        with StopSignals() as stops:
            _kill(signal.SIGTERM)
    assert info.value.status == "paused"
    assert info.value.signum == signal.SIGTERM
    assert stops.fired == signal.SIGTERM
    assert _handlers() == before


@posix_only
def test_inherited_sig_ign_is_left_alone():
    before = signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        with StopSignals() as stops:
            assert signal.getsignal(signal.SIGHUP) == signal.SIG_IGN
            _kill(signal.SIGHUP)
            assert stops.fired is None
        assert signal.getsignal(signal.SIGHUP) == signal.SIG_IGN
    finally:
        signal.signal(signal.SIGHUP, before)


@posix_only
def test_signal_in_deferred_is_held_until_outermost_exit():
    inner_done = False
    with pytest.raises(RunStopped) as info:
        with StopSignals() as stops:
            with stops.deferred():
                with stops.deferred():
                    _kill(signal.SIGINT)
                inner_done = True
                assert stops.fired == signal.SIGINT
                _kill(signal.SIGINT)
            pytest.fail("deferred exit did not raise")
    assert inner_done
    assert info.value.signum == signal.SIGINT


@posix_only
def test_ignore_logs_signals_and_drops_held_one(caplog):
    with caplog.at_level(logging.WARNING, logger="src.run_control"):
        with StopSignals() as stops:
            stops.ignore()
            _kill(signal.SIGTERM)
        assert len(caplog.records) == 1
        caplog.clear()
        with StopSignals() as stops:
            with stops.deferred():
                _kill(signal.SIGTERM)
                stops.ignore()
        assert caplog.records == []


@posix_only
def test_deferred_own_exception_wins_over_held_signal():
    with pytest.raises(ValueError):
        with StopSignals() as stops:
            with stops.deferred():
                _kill(signal.SIGTERM)
                raise ValueError("write failed")


@posix_only
def test_later_signal_is_logged_and_ignored(caplog):
    with caplog.at_level(logging.WARNING, logger="src.run_control"):
        with pytest.raises(RunStopped):
            with StopSignals():
                _kill(signal.SIGTERM)
        with StopSignals() as stops:
            with pytest.raises(RunStopped):
                _kill(signal.SIGTERM)
            caplog.clear()
            _kill(signal.SIGTERM)
            assert stops.fired == signal.SIGTERM
    assert len(caplog.records) == 1


@posix_only
def test_second_signal_while_held_is_logged_not_raised(caplog):
    with caplog.at_level(logging.WARNING, logger="src.run_control"):
        with pytest.raises(RunStopped) as info:
            with StopSignals() as stops:
                with stops.deferred():
                    _kill(signal.SIGINT)
                    _kill(signal.SIGTERM)
    assert info.value.signum == signal.SIGINT
    assert len(caplog.records) == 1


def test_off_main_thread_installs_nothing():
    before = _handlers()
    seen = {}

    def run():
        with StopSignals() as stops:
            seen["during"] = _handlers()
            with stops.deferred():
                pass
        seen["after"] = _handlers()

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    assert seen["during"] == before == seen["after"]


def test_stop_signals_per_platform(monkeypatch):
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(signal, "SIGBREAK", 21, raising=False)
    assert stop_signals() == (signal.SIGINT, 21)
    monkeypatch.undo()
    if os.name != "nt":
        assert stop_signals() == (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


def test_run_stopped_is_not_an_exception():
    stopped = RunStopped("cancelled", reason="x")
    assert not isinstance(stopped, Exception)
    assert (stopped.status, stopped.signum, stopped.reason) == ("cancelled", None, "x")
    assert stopped.resume_epoch is None and stopped.run_dir is None
