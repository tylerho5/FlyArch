import os
import subprocess
import sys
from pathlib import Path

from src import run_labels
from src.run_labels import (
    SUBJECT_MAX,
    TITLE_MAX,
    clean_title,
    git_branch,
    git_subject,
    overrides,
)


def _git(cwd, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def _repo(path: Path, subject: str = "first commit", branch: str = "feat/x") -> str:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q", "-b", branch)
    _git(path, "commit", "-q", "--allow-empty", "-m", subject)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()


def test_clean_title_collapses_whitespace_and_cuts_to_80():
    assert clean_title("  a\tb\n\n c  ") == "a b c"
    assert clean_title("x" * 300) == "x" * TITLE_MAX
    assert TITLE_MAX == 80


def test_clean_title_empty_is_none():
    assert clean_title(None) is None
    assert clean_title(" \t\n ") is None


def test_git_subject_round_trips_non_ascii_text(tmp_path):
    subject = "§ ← ½"
    sha = _repo(tmp_path, subject=subject)
    assert git_subject(tmp_path, sha, {}) == subject


def test_git_subject_is_utf8_under_a_non_utf8_locale(tmp_path):
    subject = "§ ← ½"
    sha = _repo(tmp_path, subject=subject)
    code = (
        "import sys; from pathlib import Path; from src.run_labels import git_subject; "
        "sys.stdout.buffer.write(git_subject(Path(sys.argv[1]), sys.argv[2], {}).encode('utf-8'))"
    )
    env = {**os.environ, "LC_ALL": "C", "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0"}
    out = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), sha],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
        capture_output=True,
    ).stdout
    assert out.decode("utf-8") == subject


def test_git_branch_reads_the_checkout_branch(tmp_path):
    _repo(tmp_path)
    assert git_branch(tmp_path, {}) == "feat/x"


def test_git_branch_in_a_worktree_is_the_worktree_branch(tmp_path):
    main = tmp_path / "main"
    _repo(main)
    wt = tmp_path / "wt"
    _git(main, "worktree", "add", "-q", "-b", "feat/wt", str(wt))
    assert git_branch(wt, {}) == "feat/wt"
    assert git_branch(main, {}) == "feat/x"


def test_detached_checkout_falls_back_to_the_environment(tmp_path):
    commit = _repo(tmp_path)
    _git(tmp_path, "checkout", "-q", "--detach", commit)
    assert git_branch(tmp_path, {"FLYARCH_GIT_BRANCH": "feat/env"}) == "feat/env"
    assert git_branch(tmp_path, {"FLYARCH_GIT_BRANCH": "HEAD"}) is None
    assert git_branch(tmp_path, {"FLYARCH_GIT_BRANCH": ""}) is None


def test_detached_without_environment_is_none(tmp_path):
    commit = _repo(tmp_path)
    _git(tmp_path, "checkout", "-q", "--detach", commit)
    assert git_branch(tmp_path, {}) is None


def test_git_subject_is_the_named_commits_subject_cut_to_120(tmp_path):
    commit = _repo(tmp_path, subject="s" * 300)
    assert git_subject(tmp_path, commit, {}) == "s" * SUBJECT_MAX
    assert SUBJECT_MAX == 120
    short = _repo(tmp_path / "b", subject="short one")
    assert git_subject(tmp_path / "b", short, {}) == "short one"


def test_git_subject_without_git_uses_the_environment(tmp_path):
    env = {"FLYARCH_GIT_SUBJECT": "from env"}
    assert git_subject(tmp_path, "0" * 40, env) == "from env"
    assert git_subject(tmp_path, None, env) == "from env"
    assert git_subject(tmp_path, None, {}) is None


def test_overrides_keeps_only_changed_flags():
    flags = {"hidden": 64, "lr": 0.001}
    assert overrides(flags, {"hidden": 128, "lr": 0.001}) == {"hidden": 64}


def test_overrides_compares_tuples_as_lists():
    same = overrides({"dims": (1, 2)}, {"dims": [1, 2]})
    assert same == {}
    assert overrides({"dims": (1, 3)}, {"dims": [1, 2]}) == {"dims": [1, 3]}


def test_overrides_drops_fields_without_a_default():
    assert overrides({"epochs": 5}, {}) == {}


def test_title_warning_text():
    assert run_labels.TITLE_WARNING == (
        "warning: no --title; the run will be labelled by its settings. See README."
    )
