"""CX F2-dev strata (skeleton Table S-strata) from per-positive ranks, plus random expectation.

Per-positive ranks: <results-root>/cx/scored/{95c4fd83 (gnn), a86ff9ed (baselines)}.positives.parquet
Direction (out = masked endpoint is the source) is rebuilt from the F2 CX graph + frozen suite with
src.masks.apply_mask, exactly as src/evaluate.py:300 does; positive_index indexes view.hidden_edges.
"""
import sys
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

SC = A.results_root / 'cx/scored'
gnn = pd.read_parquet(SC / '95c4fd830b2e4d4f81a9a4b065c31219.positives.parquet')
base = pd.read_parquet(SC / 'a86ff9ed321044ad920b46df34808c4a.positives.parquet')
gnn = gnn[gnn.seed.isin([0, 1, 2])]
base = base[base.model == 'xgboost']

graph = RegionGraph.read(DATA / 'processed/f2/cx')
suite = Suite.read(A.suites / 'f2/cx/f2_dev.json')
suite.validate(graph)  # graph digest must equal the suite's frozen digest
dirs = []
for inst in suite.resolve(graph):
    view = apply_mask(graph, inst)
    src_masked = view.is_masked[view.hidden_edges[:, 0]]
    dirs.append(pd.DataFrame({'mask_instance': inst.instance_id,
                              'positive_index': np.arange(len(view.hidden_edges)),
                              'direction': np.where(src_masked, 'out', 'in')}))
dirs = pd.concat(dirs)

def table(pos, label):
    pos = pos.merge(dirs, on=['mask_instance', 'positive_index'], how='left', validate='many_to_one')
    assert pos.direction.notna().all()
    pos['hit10'] = pos.reciprocal_rank >= 0.1  # rank <= 10
    # per positive: mean over seeds, then pool positives
    per = pos.groupby(['mask_instance', 'positive_index']).agg(
        rr=('reciprocal_rank', 'mean'), hit=('hit10', 'mean'), both=('both_masked', 'first'),
        direction=('direction', 'first'), matched=('type_matched', 'first')).reset_index()
    single = ~per.both
    strata = {
        ('masked', 'one'): single, ('masked', 'both'): per.both,
        ('direction', 'out'): single & (per.direction == 'out'), ('direction', 'in'): single & (per.direction == 'in'),
        ('type', 'matched'): single & per.matched, ('type', 'unmatched'): single & ~per.matched,
    }
    rows = [dict(axis=a, stratum=s, n=int(m.sum()), **{f'{label}_mrr': per.rr[m].mean(), f'{label}_r10': per.hit[m].mean()})
            for (a, s), m in strata.items()]
    return pd.DataFrame(rows)

t = table(base, 'xgb').merge(table(gnn, 'model'), on=['axis', 'stratum', 'n'])
print(t.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
t.to_csv(OUT / 'cx_strata.csv', index=False)

K = 100
print('random expected MRR', np.mean(1 / np.arange(1, K + 2)), 'R@10', 10 / (K + 1))
