"""Per-suite instance stats and dev/confirm overlap from committed suites + local graphs."""
import json, sys
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
inst_rows, summ = [], []
for code in REGIONS:
    c = code.lower()
    for ds, names in (("f1", ("f1_val",)), ("f2", ("f2_dev", "f2_confirm", "f2_side")), ("f3", ("f3_eval",))):
        gdir = DATA / "processed" / ds / c
        g = RegionGraph.read(gdir) if gdir.exists() else None
        masked_sets = {}
        for name in names:
            p = EV / "data/suites" / ds / c / f"{name}.json"
            if not p.exists():
                continue
            s = Suite.read(p)
            masked_sets[name] = [set(i.masked_root_ids.tolist()) for i in s.instances]
            for inst in (s.resolve(g) if g is not None else []):
                v = apply_mask(g, inst)
                bm = v.both_masked()
                inst_rows.append(dict(region=code, dataset=ds, suite=name, instance=inst.instance_id, regime=inst.regime,
                                      cell_type=inst.params.get("cell_type"), n_masked=inst.n_masked,
                                      frac_masked=round(inst.n_masked / g.n_nodes, 4),
                                      hidden_edges=len(bm), single_masked=int((~bm).sum()), both_masked=int(bm.sum())))
            if g is None:
                for inst in s.instances:
                    inst_rows.append(dict(region=code, dataset=ds, suite=name, instance=inst.instance_id, regime=inst.regime,
                                          cell_type=inst.params.get("cell_type"), n_masked=len(inst.masked_root_ids)))
        if ds == "f2" and g is not None:
            dev, conf = masked_sets["f2_dev"], masked_sets["f2_confirm"]
            udev, uconf = set().union(*dev), set().union(*conf)
            # pairwise dev x confirm overlap
            inter = [len(a & b) for a in dev for b in conf]
            conf_pairs = [len(conf[i] & conf[j]) / max(1, min(len(conf[i]), len(conf[j]))) for i in range(len(conf)) for j in range(i + 1, len(conf))]
            summ.append(dict(region=code, n_nodes=g.n_nodes,
                             dev_union=len(udev), dev_union_frac=round(len(udev) / g.n_nodes, 4),
                             confirm_union=len(uconf), confirm_union_frac=round(len(uconf) / g.n_nodes, 4),
                             dev_confirm_union_overlap=len(udev & uconf),
                             confirm_roots_also_in_dev_frac=round(len(udev & uconf) / len(uconf), 4),
                             dev_confirm_pairs_sharing_any=int(sum(x > 0 for x in inter)), dev_confirm_pairs=len(inter),
                             max_dev_confirm_shared=int(max(inter)),
                             confirm_pairwise_overlap_median=round(float(np.median(conf_pairs)), 4),
                             confirm_pairwise_overlap_max=round(float(np.max(conf_pairs)), 4)))
inst = pd.DataFrame(inst_rows); inst.to_csv(OUT / "suite_instances.csv", index=False)
sm = pd.DataFrame(summ); sm.to_csv(OUT / "dev_confirm_overlap.csv", index=False)
print(inst.groupby(["dataset", "suite"]).agg(n_inst=("instance", "count"), regions=("region", "nunique")).to_string())
print(inst[inst.dataset != "f3"].groupby(["suite", "regime"]).agg(n=("instance", "count"), med_masked=("n_masked", "median"), med_single=("single_masked", "median"), min_single=("single_masked", "min")).to_string())
print(sm.to_string(index=False))
print(inst[inst.regime == "T"][["region", "suite", "cell_type", "n_masked", "single_masked", "both_masked"]].to_string(index=False))
