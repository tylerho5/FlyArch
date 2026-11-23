"""Single-masked positive counts per F2-confirm instance, all 12 regions (graph + frozen suite).
Joined with the reference 3-seed model MRR to show how small T instances behave.
"""
import json
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import parse_args  # noqa: E402
A = parse_args(__doc__)
EV, DATA, OUT = A.repo_root, A.data_root, A.out
from src.regions import RegionGraph  # noqa: E402
from src.suites import Suite  # noqa: E402
from src.masks import apply_mask  # noqa: E402

REGIONS = ['CX', 'AL', 'LH', 'MB', 'LX', 'SNP', 'INP', 'VMNP', 'VLNP', 'PENP', 'GNG', 'OL']
rows = []
for r in REGIONS:
    graph = RegionGraph.read(DATA / 'processed/f2' / r.lower())
    suite = Suite.read(A.suites / 'f2' / r.lower() / 'f2_confirm.json')
    suite.validate(graph)
    ref = pd.DataFrame(json.load(open(f'{EV}/results/phase3/reference/{r}.json'))['reference']).mean(axis=1)
    for inst in suite.resolve(graph):
        view = apply_mask(graph, inst)
        single = int((~view.both_masked()).sum())
        rows.append(dict(region=r, instance=inst.instance_id, regime=inst.instance_id.split('-')[2],
                         n_masked=len(inst.masked), n_single_positives=single, model_mrr=ref[inst.instance_id]))
d = pd.DataFrame(rows)
d.to_csv(OUT / 'confirm_instance_positives.csv', index=False)
print(d.groupby('regime').agg(min_pos=('n_single_positives', 'min'), median_pos=('n_single_positives', 'median'),
                              max_pos=('n_single_positives', 'max'), n_mrr_eq_1=('model_mrr', lambda s: int((s == 1).sum()))).to_string())
print(d[d.n_single_positives < 20].to_string(index=False))
