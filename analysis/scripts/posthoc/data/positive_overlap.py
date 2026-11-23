"""Share of F2-confirm gated (single-masked) positives that are also hidden edges in some F2-dev instance."""
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
    c = code.lower()
    g = RegionGraph.read(DATA / "processed/f2" / c)
    key = lambda e: e[:, 0].astype(np.int64) * g.n_nodes + e[:, 1]
    def views(name):
        s = Suite.read(EV / "data/suites/f2" / c / f"{name}.json")
        return [apply_mask(g, i) for i in s.resolve(g)]
    dev_hidden, dev_gated = set(), set()
    for v in views("f2_dev"):
        k = key(v.hidden_edges); bm = v.both_masked()
        dev_hidden.update(k.tolist()); dev_gated.update(k[~bm].tolist())
    tot = in_hidden = in_gated = 0; per = []
    for v in views("f2_confirm"):
        bm = v.both_masked(); k = key(v.hidden_edges)[~bm]
        a = np.isin(k, list(dev_hidden)).sum(); b = np.isin(k, list(dev_gated)).sum()
        tot += len(k); in_hidden += a; in_gated += b; per.append(a / len(k))
    rows.append(dict(region=code, confirm_gated_positives=tot, also_hidden_in_dev=in_hidden,
                     frac_also_hidden_in_dev=round(in_hidden / tot, 4), frac_also_gated_in_dev=round(in_gated / tot, 4),
                     per_instance_min=round(min(per), 3), per_instance_max=round(max(per), 3)))
df = pd.DataFrame(rows); df.to_csv(OUT / "confirm_positive_overlap_with_dev.csv", index=False)
print(df.to_string(index=False))
