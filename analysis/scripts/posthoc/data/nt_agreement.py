"""F2 neurotransmitter_predicted vs F1 top_nt of the FAFB-matched neuron (proofread-tier F2)."""
import sys
from pathlib import Path
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import parse_args  # noqa: E402
A = parse_args(__doc__)
EV, DATA, OUT = A.repo_root, A.data_root, A.out
from scripts.build_features import read_table
from src.harmonize import banc_neuron_ids
f1 = pd.read_csv(DATA / "raw/Supplemental_file1_neuron_annotations.tsv", sep="\t", usecols=["root_id", "top_nt", "cell_type", "super_class"], dtype={"root_id": "int64"}).set_index("root_id")
meta = read_table(DATA / "raw/banc/banc_888_meta.feather")
ids = banc_neuron_ids(meta)
m = meta[meta.root_id.isin(ids)][["root_id", "fafb_match", "neurotransmitter_predicted", "super_class"]].copy()
m = m[m.fafb_match.notna()]
m["fafb"] = pd.to_numeric(m.fafb_match, errors="coerce").astype("Int64")
m = m[m.fafb.notna()]
m["f1_nt"] = f1["top_nt"].reindex(m.fafb.astype("int64")).to_numpy()
m["f1_super"] = f1["super_class"].reindex(m.fafb.astype("int64")).to_numpy()
both = m.dropna(subset=["f1_nt", "neurotransmitter_predicted"])
print("matched proofread-tier F2 neurons with both NT:", len(both))
print("agreement overall:", round((both.f1_nt == both.neurotransmitter_predicted).mean(), 4))
ct = pd.crosstab(both.f1_nt, both.neurotransmitter_predicted, margins=True)
ct.to_csv(OUT / "nt_crosstab_f1match_vs_f2.csv")
print(ct.to_string())
print("\nagreement by F1 super_class:")
print(both.assign(agree=both.f1_nt == both.neurotransmitter_predicted).groupby("f1_super").agree.agg(["mean", "size"]).round(3).to_string())
# Kenyon cells
kc = both[f1["cell_type"].reindex(both.fafb.astype("int64")).fillna("").str.startswith("KC").to_numpy()]
print("\nKC-matched: F1 top_nt", kc.f1_nt.value_counts().to_dict(), "F2", kc.neurotransmitter_predicted.value_counts().to_dict())
