"""F1: per-region all-pair within-region synapse totals (sizes.md 'Synapses (all)'), autapse check, global >=5 count."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.feather as pf
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import parse_args  # noqa: E402
A = parse_args(__doc__)
EV, DATA, OUT = A.repo_root, A.data_root, A.out
from src.regions import REGIONS, neuropil_to_region
t = pf.read_table(DATA / "raw/proofread_connections_783.feather", columns=["pre_pt_root_id", "post_pt_root_id", "neuropil", "syn_count"])
df = t.to_pandas(); del t
print("rows", len(df), "autapse rows", int((df.pre_pt_root_id == df.post_pt_root_id).sum()), "distinct neuropils", df.neuropil.nunique())
glob = df.groupby(["pre_pt_root_id", "post_pt_root_id"], sort=False).syn_count.sum()
print("global pairs >=5:", int((glob >= 5).sum()), "total synapses:", int(df.syn_count.sum()))
del glob
reg = df.neuropil.map(neuropil_to_region())
out = df.assign(region=reg).groupby("region", dropna=False).syn_count.sum()
print(out.to_string())
ids = np.load(DATA / "raw/proofread_root_ids_783.npy"); print("proofread_root_ids", len(ids))
