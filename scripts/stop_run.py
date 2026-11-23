"""Pause or cancel a training run from its run directory."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.run_control import (  # noqa: E402
    RunLock,
    RunLocked,
    clean_reason,
    clear_request,
    is_locked,
    read_request,
    read_state,
    write_request,
    write_state,
)

POLL_S = 0.5
DEFAULT_TIMEOUT_S = 300.0
ENDED = ("finished", "stopped_early")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", metavar="RUN_DIR", type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--pause", action="store_true")
    action.add_argument("--cancel", action="store_true")
    parser.add_argument("--reason")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, metavar="SECONDS")
    return parser


def _fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def _recorded_end(run_dir: Path):
    state = read_state(run_dir)
    return state["status"] if state is not None and state.get("status") else None


def _cancel_stopped(run_dir: Path, reason) -> int:
    with RunLock(run_dir):
        recorded = _recorded_end(run_dir)
        if recorded in ENDED:
            return _fail(f"run already ended as {recorded}; nothing to cancel")
        if recorded == "cancelled":
            print("run is already cancelled")
            return 0
        write_state(run_dir, "cancelled", **({} if reason is None else {"reason": reason}))
    print("run cancelled")
    return 0


def _stop(args: argparse.Namespace) -> int:
    run_dir: Path = args.run_dir
    if not run_dir.is_dir():
        return _fail(f"{run_dir} is not a directory")
    if not (run_dir / "manifest.json").is_file():
        return _fail(f"{run_dir} has no manifest.json")
    if not os.access(run_dir, os.W_OK):
        return _fail(f"cannot write to {run_dir}")
    status = "cancelled" if args.cancel else "paused"
    reason = clean_reason(args.reason)

    if is_locked(run_dir):
        write_request(run_dir, status, reason)
        args.request_written = True
        print(f"wrote a {status} request to {run_dir}")
        if args.no_wait:
            return 0
        deadline = time.monotonic() + args.timeout
        while is_locked(run_dir):
            if time.monotonic() >= deadline:
                return _fail("timed out waiting for the run; the request stays and the run "
                             "will act on it after its current epoch")
            time.sleep(POLL_S)
        if read_request(run_dir) is None:
            state = read_state(run_dir)
            print(f"run is now {state['status'] if state else 'stopped'}")
            return 0
        clear_request(run_dir)
        if args.pause:
            return _fail(f"run ended before it saw the request (final state: "
                         f"{_recorded_end(run_dir) or 'unknown'})")
        return _cancel_stopped(run_dir, reason)

    if args.pause:
        return _fail(f"run is not running (recorded status: {_recorded_end(run_dir) or 'unknown'})")
    return _cancel_stopped(run_dir, reason)


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    args.request_written = False
    try:
        return _stop(args)
    except KeyboardInterrupt:
        _fail("interrupted; the request stays and the run will act on it after its "
              "current epoch" if args.request_written else "interrupted")
        return 130
    except RunLocked as exc:
        return _fail(str(exc))
    except OSError as exc:
        return _fail(f"cannot update {exc.filename or args.run_dir}: {exc.strerror or exc}")


if __name__ == "__main__":
    raise SystemExit(main())
