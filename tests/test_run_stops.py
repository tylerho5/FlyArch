"""A real training process stopped by a signal ends paused; a second fit on its run is locked out."""

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX signals")

torch = pytest.importorskip("torch")

from src.run_control import RunLocked, read_state

REPO_ROOT = Path(__file__).resolve().parent.parent
WAIT_S = 180.0


def _inputs(fx, suites_dir: Path) -> dict:
    from src.features import FeatureSchema, build_features
    from src.regions import build_region_graph
    from src.suites import build_suite, suite_negatives

    connections, type_idx = fx._synthetic_connections()
    graph = build_region_graph(connections, region="CX", dataset="f1")
    annotations = fx._synthetic_annotations(graph, type_idx)
    schema = FeatureSchema.from_f1(annotations, "CX")
    features = build_features(annotations, graph, schema, dataset="f1")
    cell_types = annotations.set_index("root_id")["cell_type"].reindex(graph.node_ids)
    suite = build_suite(graph, name="f1_val", types=cell_types,
                        centroids=fx._synthetic_centroids(graph.n_nodes))
    fx._write_committed_negatives(suites_dir, suite, suite_negatives(graph, suite))
    return dict(graph=graph, features=features, schema=schema)

CHILD = textwrap.dedent(
    """
    import sys
    import time
    sys.path[:0] = [{tests!r}, {root!r}]
    import conftest as fx
    from pathlib import Path
    from test_run_stops import _inputs
    from src.run_control import RunStopped
    from train import train as training
    from train.train import TrainConfig, fit

    run_dir = Path(sys.argv[1])
    release = Path(sys.argv[2])
    inputs = _inputs(fx, run_dir.parent / "suites")
    real_save = training._save_checkpoint

    def save(path, *args, **kwargs):
        real_save(path, *args, **kwargs)
        if path.name == "last.pt":
            print("SAVED", args[2], flush=True)
            deadline = time.monotonic() + 60
            while not release.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            if not release.exists():
                sys.exit(9)

    training._save_checkpoint = save
    config = TrainConfig(epochs=50, hidden=8, patience=50, run_dir=run_dir)
    print("READY", flush=True)
    try:
        fit(inputs["graph"], inputs["features"], None, seed=0, config=config, schema=inputs["schema"])
    except RunStopped as stop:
        print("STOPPED", stop.status, stop.resume_epoch, flush=True)
    """
)


def _wait_for(path, text, child, *, count=1):
    deadline = time.monotonic() + WAIT_S
    while time.monotonic() < deadline:
        if path.exists() and path.read_text().count(text) >= count:
            return
        if child.poll() is not None:
            pytest.fail(f"child exited early ({child.returncode}):\n{path.read_text()}")
        time.sleep(0.1)
    pytest.fail(f"timed out waiting for {text!r}:\n{path.read_text() if path.exists() else ''}")


@pytest.fixture
def child_run(tmp_path):
    runs = tmp_path / "runs"
    script = tmp_path / "child.py"
    script.write_text(CHILD.format(tests=str(REPO_ROOT / "tests"), root=str(REPO_ROOT)))
    log_path = tmp_path / "child.log"
    release = tmp_path / "release"
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    with open(log_path, "w") as log:
        child = subprocess.Popen(
            [sys.executable, str(script), str(runs), str(release)],
            stdout=log, stderr=subprocess.STDOUT, env=env, cwd=REPO_ROOT,
        )
        try:
            yield child, runs, log_path, release
        finally:
            if child.poll() is None:
                child.kill()
            child.wait()


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_a_signal_pauses_the_run_at_the_last_checkpoint(child_run, signum):
    child, runs, log_path, release = child_run
    _wait_for(log_path, "SAVED 0", child)
    child.send_signal(signum)
    release.touch()
    try:
        child.wait(timeout=60)
    except subprocess.TimeoutExpired:
        pytest.fail(f"child did not exit:\n{log_path.read_text()}")

    (run_dir,) = runs.iterdir()
    saved = torch.load(run_dir / "last.pt", weights_only=True)["epoch"]
    assert f"STOPPED paused {saved + 1}" in log_path.read_text()
    state = read_state(run_dir)
    assert (state["status"], state["resume_epoch"]) == ("paused", saved + 1)


def test_a_second_fit_on_a_live_run_is_refused(child_run):
    child, runs, log_path, release = child_run
    _wait_for(log_path, "SAVED 0", child)
    import conftest as fx
    from train.train import TrainConfig, fit

    inputs = _inputs(fx, runs.parent / "suites-parent")
    config = TrainConfig(epochs=50, hidden=8, patience=50, run_dir=runs)
    with pytest.raises(RunLocked, match="locked by another process"):
        fit(inputs["graph"], inputs["features"], None, seed=0, config=config,
            schema=inputs["schema"])
    assert child.poll() is None
    release.touch()
