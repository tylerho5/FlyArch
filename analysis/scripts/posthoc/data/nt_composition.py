"""Per-region neuron-level NT composition (graph nodes) for F1 (top_nt) and F2 (neurotransmitter_predicted)."""
import sys
from pathlib import Path
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import parse_args  # noqa: E402
A = parse_args(__doc__)
EV, DATA, OUT = A.repo_root, A.data_root, A.out
from src.regions import RegionGraph, REGIONS
from scripts.build_features import read_table
f1 = pd.read_csv(DATA / "raw/Supplemental_file1_neuron_annotations.tsv", sep="\t", usecols=["root_id", "top_nt", "cell_type"], dtype={"root_id": "int64"})
print("F1 annotation rows", len(f1), "top_nt values:", f1.top_nt.value_counts(dropna=False).to_dict())
print("F1 distinct cell_type", f1.cell_type.nunique(), "typed rows", int(f1.cell_type.notna().sum()))
meta = read_table(DATA / "raw/banc/banc_888_meta.feather")[["root_id", "neurotransmitter_predicted"]]
f1nt = f1.set_index("root_id")["top_nt"]; f2nt = meta.set_index("root_id")["neurotransmitter_predicted"]
rows = []
for code in REGIONS:
    for ds, nt in (("f1", f1nt), ("f2", f2nt)):
        g = RegionGraph.read(DATA / "processed" / ds / code.lower())
        v = nt.reindex(g.node_ids).fillna("<missing>").astype(str)
        frac = v.value_counts(normalize=True)
        row = dict(region=code, dataset=ds, n_neurons=g.n_nodes)
        row.update({k: round(float(x), 4) for k, x in frac.items()})
        rows.append(row)
df = pd.DataFrame(rows).fillna(0.0)
order = ["region", "dataset", "n_neurons", "acetylcholine", "glutamate", "gaba", "dopamine", "serotonin", "octopamine", "histamine", "tyramine", "<missing>"]
df = df[[c for c in order if c in df.columns] + [c for c in df.columns if c not in order]]
df.to_csv(OUT / "nt_composition_by_region.csv", index=False)
print(df.to_string(index=False))
