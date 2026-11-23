import multiprocessing
import os
import sys
import time
from pathlib import Path

import pytest

from scripts import stop_run
from src.run_control import (
    RunLock,
    clear_request,
    read_request,
    read_state,
    write_state,
)


@pytest.fixture
def run_dir(tmp_path):
    path = tmp_path / "run"
    path.mkdir()
    (path / "manifest.json").write_text("{}")
    return path


def _child(run_dir, ready, act, hold_s):
    with RunLock(Path(run_dir)):
        Path(ready).write_text("1")
        deadline = time.monotonic() + hold_s
        while time.monotonic() < deadline:
            request = read_request(Path(run_dir))
            if request is not None and act:
                write_state(Path(run_dir), request["status"], reason=request.get("reason"))
                clear_request(Path(run_dir))
                return
            time.sleep(0.05)


class _Live:
    def __init__(self, run_dir, tmp_path, act=True, hold_s=20.0):
        self.process = multiprocessing.get_context("spawn").Process(
            target=_child, args=(str(run_dir), str(tmp_path / "ready"), act, hold_s))
        self.ready = tmp_path / "ready"

    def __enter__(self):
        self.process.start()
        deadline = time.monotonic() + 30
        while not self.ready.exists():
            assert time.monotonic() < deadline, "child never took the lock"
            time.sleep(0.05)
        return self

    def __exit__(self, *exc):
        self.process.join(15)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(5)


def test_live_cancel_waits_and_reports(run_dir, tmp_path, capsys):
    with _Live(run_dir, tmp_path):
        code = stop_run.main([str(run_dir), "--cancel", "--reason", "lr too high"])
    out = capsys.readouterr().out
    assert code == 0
    assert "cancelled" in out
    state = read_state(run_dir)
    assert (state["status"], state["reason"]) == ("cancelled", "lr too high")
    assert read_request(run_dir) is None


def test_live_no_wait_leaves_request(run_dir, tmp_path, capsys):
    with _Live(run_dir, tmp_path, act=False, hold_s=1.0):
        started = time.monotonic()
        code = stop_run.main([str(run_dir), "--pause", "--no-wait"])
        elapsed = time.monotonic() - started
        request = read_request(run_dir)
    assert code == 0
    assert elapsed < 0.5
    assert request["status"] == "paused"
    assert "request" in capsys.readouterr().out


def test_live_timeout_leaves_request(run_dir, tmp_path, capsys):
    with _Live(run_dir, tmp_path, act=False, hold_s=3.0):
        code = stop_run.main([str(run_dir), "--cancel", "--timeout", "1"])
        request = read_request(run_dir)
    captured = capsys.readouterr()
    assert code == 1
    assert request["status"] == "cancelled"
    assert "after its current epoch" in captured.err


def test_request_left_when_lock_frees_cancel_continues(run_dir, tmp_path, capsys):
    write_state(run_dir, "paused")
    with _Live(run_dir, tmp_path, act=False, hold_s=1.0):
        code = stop_run.main([str(run_dir), "--cancel", "--reason", "late"])
    assert code == 0
    assert read_request(run_dir) is None
    assert read_state(run_dir)["status"] == "cancelled"


def test_request_left_when_lock_frees_pause_fails(run_dir, tmp_path, capsys):
    write_state(run_dir, "finished")
    with _Live(run_dir, tmp_path, act=False, hold_s=1.0):
        code = stop_run.main([str(run_dir), "--pause"])
    captured = capsys.readouterr()
    assert code == 1
    assert read_request(run_dir) is None
    assert "finished" in captured.out + captured.err


def test_cancel_already_cancelled_is_a_no_op(run_dir, capsys):
    write_state(run_dir, "cancelled", reason="first")
    assert stop_run.main([str(run_dir), "--cancel", "--reason", "second"]) == 0
    assert read_state(run_dir)["reason"] == "first"
    assert "already cancelled" in capsys.readouterr().out


def test_cancel_refuses_finished_by_state(run_dir, capsys):
    write_state(run_dir, "finished")
    assert stop_run.main([str(run_dir), "--cancel"]) == 1
    assert "finished" in capsys.readouterr().err
    assert read_state(run_dir)["status"] == "finished"


def test_pause_on_stopped_run_refused(run_dir, capsys):
    write_state(run_dir, "paused")
    assert stop_run.main([str(run_dir), "--pause"]) == 1
    err = capsys.readouterr().err
    assert "not running" in err and "paused" in err
    assert read_request(run_dir) is None


@pytest.mark.parametrize("recorded", [None, "paused", "crashed"])
def test_cancel_of_a_stopped_run_records_cancelled(run_dir, capsys, recorded):
    if recorded is not None:
        write_state(run_dir, recorded)
    assert stop_run.main([str(run_dir), "--cancel", "--reason", "x"]) == 0
    assert "run cancelled" in capsys.readouterr().out
    state = read_state(run_dir)
    assert (state["status"], state["reason"]) == ("cancelled", "x")


@pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                    reason="needs a non-root POSIX user")
def test_read_only_run_dir_exits_naming_path(run_dir, capsys):
    run_dir.chmod(0o500)
    try:
        code = stop_run.main([str(run_dir), "--cancel"])
    finally:
        run_dir.chmod(0o700)
    err = capsys.readouterr().err
    assert code == 1
    assert str(run_dir) in err
    assert "Traceback" not in err
    assert read_state(run_dir) is None


def test_bad_directories(tmp_path, run_dir, capsys):
    assert stop_run.main([str(tmp_path / "missing"), "--cancel"]) == 1
    assert str(tmp_path / "missing") in capsys.readouterr().err
    empty = tmp_path / "empty"
    empty.mkdir()
    assert stop_run.main([str(empty), "--cancel"]) == 1
    assert "manifest.json" in capsys.readouterr().err


def test_interrupt_while_waiting_prints_one_line(run_dir, tmp_path, capsys, monkeypatch,
                                                 ):
    def interrupt(_):
        raise KeyboardInterrupt

    with _Live(run_dir, tmp_path, act=False, hold_s=2.0):
        monkeypatch.setattr(stop_run.time, "sleep", interrupt)
        code = stop_run.main([str(run_dir), "--cancel"])
        monkeypatch.undo()
        request = read_request(run_dir)
    err = capsys.readouterr().err
    assert code == 130
    assert request["status"] == "cancelled"
    assert "after its current epoch" in err
    assert "Traceback" not in err
