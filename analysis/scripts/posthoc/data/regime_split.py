"""Gated-cohort F2-confirm MRR by mask regime (results/phase3/reference/*.json, 3-seed mean per instance),
and the F2-dev masked-innervation ablation split by regime (results/phase3/f2dev-diagnosis/f2dev-instances.csv)."""
import json, re
from pathlib import Path
import numpy as np, pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import parse_args  # noqa: E402
A = parse_args(__doc__)
EV, DATA, OUT = A.repo_root, A.data_root, A.out
rows = []
for p in sorted((EV / "results/phase3/reference").glob("[A-Z]*.json")):
    d = json.loads(p.read_text())
    per = pd.DataFrame(d["reference"]).mean(axis=1)  # instance x seed -> mean over seeds
    for inst, v in per.items():
        rows.append(dict(region=d["region"], instance=inst, regime=re.search(r"-(R01|R05|R10|T|S05)-", inst).group(1), gnn_mrr=v))
df = pd.DataFrame(rows)
piv = df.pivot_table(index="region", columns="regime", values="gnn_mrr", aggfunc="mean").round(3)
piv["n_instances"] = df.groupby("region").size()
piv.to_csv(OUT / "gated_cohort_confirm_mrr_by_regime.csv")
print("Gated cohort F2-confirm MRR by regime (3-seed mean, then instance mean):"); print(piv.to_string())
dev = pd.read_csv(EV / "results/phase3/f2dev-diagnosis/f2dev-instances.csv")
dev["regime"] = dev.instance.str.extract(r"-(R01|R05|R10|T|S05)-")[0]
dev["log2_unif_vs_real"] = np.log2(dev.ref_unif / dev.ref_real)
p2 = dev.pivot_table(index="region", columns="regime", values="log2_unif_vs_real", aggfunc="mean").round(3)
p2.to_csv(OUT / "f2dev_masked_innervation_effect_by_regime.csv")
print("\nF2-dev: mean log2(MRR uniform masked innervation / as scored), gated cohort, by regime:"); print(p2.to_string())
