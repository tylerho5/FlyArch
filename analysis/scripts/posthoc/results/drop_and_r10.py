"""Items 4-5: Recall@10 where local rows exist; F1-val -> F2 drop vs region properties.
Inputs: outputs of recompute.py (same dir), graph.json files under <data-root>/processed, committed suites,
results/experts.csv, registered confirm-score values (typed in below).
"""
import glob, json, os
import numpy as np
import pandas as pd
from scipy import stats

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import parse_args  # noqa: E402
A = parse_args(__doc__)
EV, DATA, OUT = A.repo_root, A.data_root / 'processed', A.out
REGIONS = ['CX', 'AL', 'LH', 'MB', 'LX', 'SNP', 'INP', 'VMNP', 'VLNP', 'PENP', 'GNG', 'OL']
pd.set_option('display.width', 250)

# --- Recall@10 / n_positives per region from local scored rows ---
roots = [str(A.results_root)]
seen, frames = set(), []
for root in roots:
    for f in sorted(glob.glob(root + '/*/scored/*.scores.parquet')):
        if os.path.basename(f) not in seen:
            seen.add(os.path.basename(f)); frames.append(pd.read_parquet(f))
sc = pd.concat(frames)
reg = pd.read_csv(f'{EV}/results/experts.csv')
cohort = reg.groupby('region').cohort.first()
sc = sc[sc.cohort.values == cohort.reindex(sc.region).values]
sc = sc[(sc.model != 'gnn_a4') | sc.seed.isin([0, 1, 2])]
r10 = (sc[sc.split.isin(['f2_dev', 'f2_confirm'])]
       .groupby(['region', 'split', 'model', 'mask_instance'])[['mrr', 'recall_at_10', 'n_positives']].mean()
       .groupby(['region', 'split', 'model']).agg(mrr=('mrr', 'mean'), recall_at_10=('recall_at_10', 'mean'),
                                                  n_inst=('mrr', 'size'), positives=('n_positives', 'sum')))
print(r10.to_string(float_format=lambda x: f'{x:.4f}'))
r10.to_csv(f'{OUT}/recall10_local.csv')

# --- region properties and drop table ---
s7 = {'CX': (0.7526, 19), 'AL': (0.8313, 18), 'LH': (0.4750, 19), 'MB': (0.6712, 20), 'LX': (0.9474, 19),
      'SNP': (0.6314, 20), 'INP': (0.8677, 20), 'VMNP': (1.0564, 19), 'VLNP': (0.5815, 19), 'PENP': (0.5076, 18),
      'GNG': (0.9660, 20), 'OL': (0.4123, 20)}
f1val = {  # 3-seed selection-instance means, from the Phase 1 and Phase 2 F1-val check records
    'CX': 0.734609, 'AL': 0.642301, 'LH': 0.621431, 'MB': 0.652753, 'LX': 0.661793, 'SNP': 0.625814,
    'INP': 0.750640, 'VMNP': 0.716323, 'VLNP': 0.812145, 'PENP': 0.714940, 'GNG': 0.627925, 'OL': 0.912030}
summ = pd.read_csv(f'{OUT}/summary_model_mrr.csv').set_index('region')
rows = []
for r in REGIONS:
    g1 = json.load(open(f'{DATA}/f1/{r.lower()}/graph.json'))
    g2 = json.load(open(f'{DATA}/f2/{r.lower()}/graph.json'))
    assert g1['digest'] == reg[reg.region == r].graph_digest.iloc[0], r
    assert g2['digest'] == json.load(open(f'{A.suites}/f2/{r.lower()}/f2_dev.json'))['graph_digest'], r
    rows.append(dict(region=r, f1_nodes=g1['n_nodes'], f1_edges=g1['n_edges'], f1_edges_per_node=g1['n_edges'] / g1['n_nodes'],
                     f2_threshold=g2['threshold'], f2_nodes=g2['n_nodes'], f2_edges=g2['n_edges'],
                     f2_f1_edge_ratio=g2['n_edges'] / g1['n_edges'],
                     f1val=f1val[r], f2dev=summ.loc[r, 'model_dev_diag'], f2conf=summ.loc[r, 'model_conf_ref'],
                     r_bar=s7[r][0], factor=2 ** s7[r][0], wins=s7[r][1]))
t = pd.DataFrame(rows)
t['drop_f1val_to_f2dev'] = t.f1val - t.f2dev
t['drop_f1val_to_f2conf'] = t.f1val - t.f2conf
t['rel_drop_conf'] = t.f2conf / t.f1val
print(t.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
t.to_csv(f'{OUT}/drop_vs_properties.csv', index=False)
print('\nSpearman rho with r_bar (n=12, descriptive only):')
for c in ['f1_nodes', 'f1_edges', 'f1_edges_per_node', 'f2_threshold', 'f2_f1_edge_ratio', 'f1val', 'f2conf',
          'drop_f1val_to_f2conf', 'rel_drop_conf']:
    rho, p = stats.spearmanr(t[c], t.r_bar)
    print(f'  {c:24s} rho={rho:+.3f} p={p:.3f}')
