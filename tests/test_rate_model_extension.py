"""Extension: seeds of the original cells unchanged, decision rules."""
import numpy as np

from src import rate_model as rm


def _cell(z, p, d, status="defined", ps=0.01):
    return {"status": status, "reason": "", "z": z, "p": p, "d": d, "sens_complete": True,
            "z_flip": [z] * rm.K_FLIPS, "z_amp": {0.5: z, 2.0: z},
            "p_flip": [ps] * rm.K_FLIPS, "p_amp": {0.5: ps, 2.0: ps}}


def test_original_seed_indices_unchanged():
    for i, r in enumerate(("al", "lx", "cx")):
        for j, sp in enumerate(("f1", "f2")):
            assert rm._entropy(20261122, r, sp) == [20261122, i, j]
    assert rm._entropy(20261122, "lx", "f3") == [20261122, 1, 2]
    assert [rm._entropy(1, r, "f1")[1] for r in rm.EXT_REGIONS] == [3, 4, 5]


def test_original_draw_identical():
    rng_old = np.random.default_rng(np.random.SeedSequence([20261122, 1, 0], spawn_key=(1, 7)))
    assert rm.draw_rng(20261122, "lx", "f1", 7).integers(0, 1 << 30, 5).tolist() == rng_old.integers(0, 1 << 30, 5).tolist()


def test_lx_male_cns_rule():
    res = rm.decide_ext({("lx", "f3"): _cell(-2.1, 0.04, -0.3)}, lx_f1_z=-2.18)
    assert bool(res.iloc[0].confirmed) and res.iloc[0].verdict == "male-CNS LX directional test passed"
    for c in (_cell(2.1, 0.04, 0.3), _cell(-2.1, 0.06, -0.3), _cell(-2.1, 0.04, -0.01), _cell(np.nan, np.nan, np.nan, "undefined")):
        assert not bool(rm.decide_ext({("lx", "f3"): c}, lx_f1_z=-2.18).iloc[0].confirmed)


def test_extension_holm_and_replication():
    cells = {("lh", "f1"): _cell(-4, 0.001, -0.2), ("lh", "f2"): _cell(-2.5, 0.01, -0.1),
             ("mb", "f1"): _cell(-2, 0.06, -0.2), ("mb", "f2"): _cell(-3, 0.001, -0.1),
             ("penp", "f1"): _cell(3, 0.001, 0.2), ("penp", "f2"): _cell(-3, 0.001, -0.1)}
    res = rm.decide_ext(cells, lx_f1_z=-2.18).set_index(["region", "split"])
    assert res.loc[("lh", "f1"), "verdict"] == "effect declared"            # Holm 0.003, replicated
    assert res.loc[("mb", "f1"), "verdict"] == "no F1 effect"               # Holm-adjusted 0.06
    assert res.loc[("penp", "f1"), "verdict"] == "not replicated"           # F2 sign differs
    assert np.isclose(res.loc[("mb", "f1"), "p_holm"], 0.06) and np.isclose(res.loc[("lh", "f1"), "p_holm"], 0.003)


def test_direction_frozen_in_code():
    assert rm.LX_F1_Z_REGISTERED < 0
    assert bool(rm.decide_ext({("lx", "f3"): _cell(-2.1, 0.04, -0.3)}).iloc[0].confirmed)
    assert not bool(rm.decide_ext({("lx", "f3"): _cell(2.1, 0.04, 0.3)}).iloc[0].confirmed)


def test_ext_sensitivity_labels_use_p_not_z():
    r = rm.decide_ext({("lx", "f3"): _cell(-2.1, 0.04, -0.3, ps=0.03)}).iloc[0]   # |z| < 3 but replicates p < 0.05
    assert r.sign_label == "sign-robust" and r.amp_label == "amplitude-robust"
    r = rm.decide_ext({("lx", "f3"): _cell(-2.1, 0.04, -0.3, ps=0.2)}).iloc[0]
    assert r.sign_label == "sign-sensitive" and r.amp_label == "amplitude-sensitive"


def test_cli_plan_fixed_per_cell():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("cli", pathlib.Path(__file__).parents[1] / "scripts/rate_model_rewiring.py")
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    assert cli._plan("lh", "f1") == (2000, 500) and cli._plan("lx", "f3") == (5000, 2000) and cli._plan("al", "f1") == (5000, 2000)
