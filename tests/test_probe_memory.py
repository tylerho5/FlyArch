import sys

import numpy as np
import pytest

pytest.importorskip("torch")

# The CPU branch of the probe's peak reading imports the POSIX-only ``resource``
# module; the CUDA path does not, so this only skips CPU probes.
posix_peak = pytest.mark.skipif(
    sys.platform == "win32", reason="the CPU memory probe reads the POSIX resource module"
)


@posix_peak
def test_probe_reports_each_draw_and_the_f1_val_pass(paired_inputs):
    from scripts.probe_memory import probe
    from src.feature_io import read_feature_artifacts
    from src.regions import RegionGraph
    from src.suites import Suite
    from train.train import TrainConfig

    processed, suite_path, suites_dir = paired_inputs
    graph = RegionGraph.read(processed)
    table, schema = read_feature_artifacts(processed, graph)
    config = TrainConfig(epochs=1, hidden=8, pair_batch=2, suites_dir=suites_dir)
    result = probe(graph, table, schema, Suite.read(suite_path), config, device="cpu", draws=3)

    assert len(result["steps"]) == 3
    assert all(step["ranked_rows"] <= 2 < step["hidden_edges"] for step in result["steps"])
    assert result["validation"]["peak"] > 0
    assert result["max_type_id"] == len(schema.type_vocab)


def test_cuda_probe_reports_and_applies_the_free_memory_cap(monkeypatch):
    from scripts import probe_memory

    calls = []
    free, total = 6 * 2**30, 8 * 2**30
    monkeypatch.setattr(probe_memory.torch.cuda, "mem_get_info", lambda device: (free, total))
    monkeypatch.setattr(probe_memory.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(probe_memory.torch.cuda, "set_per_process_memory_fraction",
                        lambda fraction, device: calls.append((fraction, device)))

    result = probe_memory._cap_cuda_allocator("cuda")

    assert result == {"free_at_start": free, "allocator_cap": free - 512 * 2**20}
    assert calls == [((free - 512 * 2**20) / total, 0)]


def test_cuda_peak_is_reserved_memory_and_keeps_allocated(monkeypatch):
    from scripts import probe_memory

    monkeypatch.setattr(probe_memory.torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(probe_memory.torch.cuda, "max_memory_reserved", lambda device=None: 70)
    monkeypatch.setattr(probe_memory.torch.cuda, "max_memory_allocated", lambda device=None: 40)
    memory = probe_memory._Memory("cuda")

    assert memory.peak() == 70
    assert memory.peak_allocated() == 40


def test_probe_sets_determinism_before_seed_and_model(monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from scripts import probe_memory
    from train import train as training

    events = []
    class Model:
        def __init__(self, *args, **kwargs):
            events.append("model")
        def to(self, device): return self
        def set_trained_types(self, ids): pass
        def train(self): pass

    class Memory:
        def __init__(self, device): pass
        def reset(self): pass
        def peak(self): return 1
        def peak_allocated(self): return None
        def total(self): return 10

    monkeypatch.setattr(training, "_resolve_device", lambda device: "cpu")
    monkeypatch.setitem(training._BACKENDS, "cpu", SimpleNamespace(deterministic=True))
    monkeypatch.setattr(training, "_use_deterministic_algorithms",
                        lambda: events.append("deterministic"))
    monkeypatch.setattr(probe_memory.torch, "manual_seed", lambda seed: events.append("seed"))
    monkeypatch.setattr(probe_memory, "_Memory", Memory)
    monkeypatch.setattr(probe_memory, "M1", Model)
    monkeypatch.setattr(training, "validate_m1_inputs", lambda *args:
                        SimpleNamespace(instances=[], tasks=[]))
    monkeypatch.setattr(training, "_masked_union", lambda instances: np.array([], dtype=int))
    monkeypatch.setattr(training, "_training_graph", lambda *args: object())
    monkeypatch.setattr(training, "_gap_sampler", lambda *args: None)
    monkeypatch.setattr(probe_memory, "pack_edges", lambda edges: [])
    monkeypatch.setattr(training, "_table_schema", lambda *args: object())
    monkeypatch.setattr(training, "_optimizer", lambda *args: object())
    monkeypatch.setattr(training, "_trained_type_ids", lambda *args: np.array([], dtype=int))
    monkeypatch.setattr(training, "_validation_scores", lambda *args: None)

    probe_memory.probe(
        SimpleNamespace(n_nodes=1, edges=[]), object(), object(), object(),
        training.TrainConfig(epochs=1, hidden=1, layers=1, encoder_dropout=0,
                             decoder_dropout=0, sibling_features=False, pair_batch=1,
                             suites_dir="suites"),
        device="cpu", draws=0,
    )

    assert events == ["deterministic", "seed", "model"]


@posix_peak
def test_probe_reports_the_distinct_endpoints_in_each_draw(paired_inputs, monkeypatch):
    from scripts.probe_memory import probe
    from src.feature_io import read_feature_artifacts
    from src.regions import RegionGraph
    from src.suites import Suite
    from train import train as training
    from train.train import TrainConfig

    processed, suite_path, suites_dir = paired_inputs
    graph = RegionGraph.read(processed)
    table, schema = read_feature_artifacts(processed, graph)
    config = TrainConfig(epochs=1, hidden=8, pair_batch=2, suites_dir=suites_dir)
    actual_batches = []
    make_batch = training._epoch_batch

    def capture_batch(*args, **kwargs):
        batch = make_batch(*args, **kwargs)
        actual_batches.append(batch)
        return batch

    monkeypatch.setattr(training, "_epoch_batch", capture_batch)
    result = probe(
        graph, table, schema, Suite.read(suite_path), config,
        device="cpu", draws=3, seed_set=True,
    )

    assert [step["seed_set"] for step in result["steps"]] == [
        np.unique(batch.pairs).size for batch in actual_batches
    ]


def test_probe_accepts_seed_set_flag():
    from scripts.probe_memory import build_argparser

    args = build_argparser().parse_args(
        ["--region", "AL", "--device", "cpu", "--seed-set"]
    )

    assert args.seed_set


def test_probe_refuses_a_region_that_disagrees_with_the_config():
    from scripts.probe_memory import main
    from train.region_config import CONFIG_DIR

    with pytest.raises(SystemExit) as refused:
        main(["--region", "cx", "--device", "cpu",
              "--config", str(CONFIG_DIR / "al.toml")])

    assert refused.value.code == 2


def test_probe_refuses_to_start_while_gpu_lock_is_held(tmp_path):
    from scripts.probe_memory import main
    from src.run_control import RunLock

    lock = RunLock(tmp_path / "runs", ".gpu.lock")
    lock.__enter__()
    try:
        with pytest.raises(SystemExit) as refused:
            main(["--region", "al", "--device", "cuda", "--gpu-lock",
                  str(tmp_path / "runs/.gpu.lock")])
        assert refused.value.code == 2
    finally:
        lock.__exit__(None, None, None)


def test_probe_json_includes_scientific_input_and_code_identity(monkeypatch, tmp_path, capsys):
    import json
    from types import SimpleNamespace
    from scripts import probe_memory
    from train import train as training
    from train.region_config import config_path, load_region_config

    monkeypatch.setattr(probe_memory.RegionGraph, "read", lambda path: SimpleNamespace(
        region="AL", digest=lambda: "graph"
    ))
    monkeypatch.setattr(probe_memory, "read_feature_artifacts", lambda path, graph: (
        SimpleNamespace(), SimpleNamespace(digest=lambda: "schema")
    ))
    monkeypatch.setattr(probe_memory.Suite, "read", lambda path: SimpleNamespace(
        digest=lambda: "suite"
    ))
    monkeypatch.setattr(probe_memory, "probe", lambda *args, **kwargs: {
        "steps": [{"peak": 3}], "validation": {"peak": 4}, "max_type_id": 2,
        "total": 10,
    })
    monkeypatch.setattr(training, "_feature_digest", lambda table, **_: "features")
    monkeypatch.setattr(training, "_code_digest", lambda: "code")

    assert probe_memory.main([
        "--region", "al", "--device", "cpu", "--data-root", str(tmp_path),
        "--suites-dir", str(tmp_path / "suites"), "--json",
    ]) == 0

    report = json.loads(capsys.readouterr().out)
    assert {key: report[key] for key in (
        "region", "device", "pair_batch", "code_digest", "graph_digest",
        "feature_digest", "schema_digest", "suite_digest",
    )} == {
        "region": "AL", "device": "cpu",
        "pair_batch": load_region_config(config_path("al")).train["pair_batch"],
        "code_digest": "code", "graph_digest": "graph", "feature_digest": "features",
        "schema_digest": "schema", "suite_digest": "suite",
    }
    assert report["config"]["suites_dir"] == (tmp_path / "suites").as_posix()
