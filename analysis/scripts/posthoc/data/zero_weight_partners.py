"""Gated (single-masked) F2-confirm positives whose visible partner has zero visible strength on the replaced side,
so the degree-weighted sampler can never draw it as a negative (positive-only candidate)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import parse_args  # noqa: E402
A = parse_args(__doc__)
EV, DATA, OUT = A.repo_root, A.data_root, A.out
from src.regions import RegionGraph, REGIONS
from src.suites import Suite
from src.masks import apply_mask
rows = []
for code in REGIONS:
    g = RegionGraph.read(DATA / "processed/f2" / code.lower())
    s = Suite.read(EV / "data/suites/f2" / code.lower() / "f2_confirm.json")
    tot = zero = 0
    for inst in s.resolve(g):
        v = apply_mask(g, inst); m = v.is_masked
        e = v.hidden_edges[~v.both_masked()]
        vin, vout = v.visible_in_strength(), v.visible_out_strength()
        src_masked = m[e[:, 0]]
        # source masked -> target replaced (weighted by visible in-strength of candidates); else source replaced (out-strength)
        partner_w = np.where(src_masked, vin[e[:, 1]], vout[e[:, 0]])
        tot += len(e); zero += int((partner_w == 0).sum())
    rows.append(dict(region=code, gated_positives=tot, partner_zero_weight=zero, frac=round(zero / tot, 4)))
df = pd.DataFrame(rows); df.to_csv(OUT / "zero_weight_partners_f2_confirm.csv", index=False)
print(df.to_string(index=False))
