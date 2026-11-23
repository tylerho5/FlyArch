"""Labels for a training run: a title, the flags that changed, and git context.

Stdlib only: the queue launchers and host worker import this without the
training stack.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

TITLE_MAX = 80
SUBJECT_MAX = 120
TITLE_WARNING = (
    "warning: no --title; the run will be labelled by its settings. See README."
)


@dataclass(frozen=True)
class RunLabels:
    title: Optional[str] = None
    overrides: dict = field(default_factory=dict)


def clean_title(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    collapsed = " ".join(text.split())[:TITLE_MAX].strip()
    return collapsed or None


def _git(repo_root: Path, *args: str) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def git_branch(
    repo_root: Path, environ: Optional[Mapping[str, str]] = None
) -> Optional[str]:
    env = os.environ if environ is None else environ
    for branch in (
        _git(repo_root, "symbolic-ref", "--short", "-q", "HEAD"),
        (env.get("FLYARCH_GIT_BRANCH") or "").strip(),
    ):
        if branch and branch != "HEAD":
            return branch
    return None


def git_subject(
    repo_root: Path,
    commit: Optional[str],
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    env = os.environ if environ is None else environ
    subject = _git(repo_root, "log", "-1", "--format=%s", commit) if commit else None
    if not subject:
        subject = (env.get("FLYARCH_GIT_SUBJECT") or "").strip()
    return subject[:SUBJECT_MAX] if subject else None


def _as_list(value: Any) -> Any:
    return list(value) if isinstance(value, tuple) else value


def overrides(flags: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict:
    out = {}
    for key, value in flags.items():
        if key not in baseline:
            continue
        if _as_list(value) != _as_list(baseline[key]):
            out[key] = _as_list(value)
    return out
