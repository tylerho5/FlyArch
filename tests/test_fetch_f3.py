"""Seam: the bulk batches -> fetch_f3's connection and synapse outputs.

CPU-only, unlike tests/test_phase2_pipeline.py, so it runs without torch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.fetch_f3 import PARTNERS, POINTS, fetch_connections, fetch_synapses

N_BATCHES = 70  # past the 64-part compaction, so repeated pairs cross a compaction


class FakeBulk:
    def __init__(self, tables: dict[str, pd.DataFrame]) -> None:
        self.tables = tables

    def batches(self, name, columns):
        for i in range(N_BATCHES):
            yield i, N_BATCHES, self.tables[name][columns]


def hemisphere(x_nm):
    return np.where(x_nm > 4000, "L", "R")


def test_fetch_outputs_from_batches(tmp_path, capsys):
    population = np.array([1, 2, 3], dtype="int64")
    partners = pd.DataFrame({
        "body_pre": [1, 1, 1, 1, 1, 9, 2, 1],
        "body_post": [2, 2, 1, 3, 2, 2, 3, 3],
        "primary_post": ["aL(L)", "IB", "FB", "AB(R)", "IB", "FB", "ANm", None],
        "x_post": [0, 1000, 0, 0, 10, 0, 0, 0],  # voxels: 1000 -> 8000 nm (L), 10 -> 80 nm (R)
    })
    points = pd.DataFrame({
        "x": [600, 10, 1000, 5, 5], "y": [10, 20, 30, 40, 50], "z": [1, 2, 3, 4, 5],
        "kind": ["PreSyn", "PostSyn", "PostSyn", "PreSyn", "PreSyn"],
        "body": [1, 2, 3, 9, 1],
        "primary": ["SAD", "LAL(R)", "IB", "FB", "AB(L)"],
        "subprimary": ["AMMC(L)", "GA(R)", "<unspecified>", "<unspecified>", "<unspecified>"],
    })
    bulk = FakeBulk({PARTNERS: partners, POINTS: points})
    regions = ["MB", "INP", "CX", "PENP", "LX"]

    connections = fetch_connections(bulk, population, hemisphere, regions)
    got = {tuple(r[:3]): r[3] for r in connections.itertuples(index=False)}
    # The autapse, body 9 (not in the population), AB, the VNC ROI and the null ROI are gone.
    assert got == {(1, 2, "MB_VL_L"): N_BATCHES, (1, 2, "IB_L"): N_BATCHES, (1, 2, "IB_R"): N_BATCHES}
    output = capsys.readouterr().out
    assert f"AB: {N_BATCHES} synapse(s)" in output
    assert f"excluded ROI (no ROI): {N_BATCHES} synapse(s)" in output

    innervation = fetch_synapses(bulk, population, hemisphere, regions, tmp_path)
    counts = {tuple(r[:3]): r[3] for r in innervation.itertuples(index=False)}
    assert counts == {(1, "AMMC_L", "pre"): N_BATCHES, (2, "GA_R", "post"): N_BATCHES,
                      (3, "IB_L", "post"): N_BATCHES}
    synapses = pd.read_parquet(tmp_path / "synapses.parquet")
    assert len(synapses) == 3 * N_BATCHES
    first = synapses.drop_duplicates("neuropil").set_index("neuropil")
    assert first.loc["AMMC_L", ["x", "y", "z"]].tolist() == [4800, 80, 8]  # voxels * 8 nm


def test_a_transient_read_error_is_retried(monkeypatch):
    import scripts.fetch_f3 as fetch

    class Flaky:
        calls = 0

        def get_batch(self, i):
            Flaky.calls += 1
            if Flaky.calls == 1:
                raise OSError("Could not resolve hostname")
            return i

    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    assert fetch._get_batch(Flaky(), 7) == 7 and Flaky.calls == 2


def test_a_failed_fetch_leaves_the_previous_outputs_in_place(tmp_path, monkeypatch):
    import scripts.fetch_f3 as fetch
    from src import harmonize

    names = ("annotations.parquet", "connections.parquet", "synapses.parquet", "innervation.parquet")
    for name in names:
        (tmp_path / name).write_bytes(b"previous run")
    table = pd.DataFrame({"a": [1], "body": [1], "bodyId": [1], "predicted_nt": ["gaba"]})
    monkeypatch.setattr(fetch.Bulk, "read_all", lambda self, name: table)
    monkeypatch.setattr(harmonize, "f3_neuron_ids", lambda raw: [])
    monkeypatch.setattr(harmonize, "normalize_f3_annotations", lambda raw: (table, {}))
    monkeypatch.setattr(fetch, "fetch_connections", lambda *args: table)

    def fail(bulk, population, hemisphere, regions, out_dir):
        (out_dir / "synapses.parquet").write_bytes(b"truncated")
        raise OSError("stream died")

    monkeypatch.setattr(fetch, "fetch_synapses", fail)
    with pytest.raises(OSError, match="stream died"):
        fetch.main(["--out", str(tmp_path), "--from-bulk", str(tmp_path)])
    assert {name: (tmp_path / name).read_bytes() for name in names} == dict.fromkeys(names, b"previous run")


def test_verify_weights_flags_a_pair_that_sums_above_the_release(tmp_path, monkeypatch):
    import scripts.fetch_f3 as fetch

    pd.DataFrame({"pre_pt_root_id": [1, 1, 2], "post_pt_root_id": [2, 2, 3], "neuropil": "AL_L",
                  "syn_count": [3, 4, 5]}).to_parquet(tmp_path / "connections.parquet")

    def run(weights):
        batch = pd.DataFrame({"body_pre": [1, 2], "body_post": [2, 3], "weight": weights})
        monkeypatch.setattr(fetch.Bulk, "batches", lambda self, name, columns: iter([(0, 1, batch)]))
        return fetch.verify_weights(fetch.Bulk(tmp_path), tmp_path)

    assert run([9, 5]) == 0  # 7 of 9 for pair (1, 2): the rest sits in a dropped ROI
    assert run([6, 5]) == 1
