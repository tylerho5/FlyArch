"""Recompute regional numbers from primary records (results agent, 2026-11-17).

Sources (read only):
  scored caches   <results-root>/<code>/scored/*.parquet
  reference JSON  <repo-root>/results/phase3/reference/<CODE>.json  (gate seeds 0-2 on f2_confirm, all 12 regions)
  f2dev diag CSV  <repo-root>/results/phase3/f2dev-diagnosis/f2dev-instances.csv (ref_real = 3-seed F2-dev MRR)
  delta records   <repo-root>/results/<code>/delta.json
Gate statistics use src/gate.py from the repo checkout.
Run: python recompute.py [--results-root DIR] [--out DIR]
"""
import glob, json, os, sys
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import parse_args  # noqa: E402
A = parse_args(__doc__)
EV, OUT = A.repo_root, A.out
from src.gate import PreregConfig, evaluate_gate, select_champion_baseline, calibrate_delta  # noqa: E402

REGIONS = ['CX', 'AL', 'LH', 'MB', 'LX', 'SNP', 'INP', 'VMNP', 'VLNP', 'PENP', 'GNG', 'OL']
REGIMES = ['R01', 'R05', 'R10', 'T', 'S05']
GATE_SEEDS = {0, 1, 2}
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)

def regime(instance_id):
    return instance_id.split('-')[2]

# ---------- load scored caches (dedupe by file name) ----------
roots = [str(A.results_root)]
seen, frames = {}, []
for root in roots:
    for f in sorted(glob.glob(root + '/*/scored/*.scores.parquet')):
        name = os.path.basename(f)
        if name in seen:
            continue
        seen[name] = f
        d = pd.read_parquet(f)
        d['file'] = f
        frames.append(d)
scored = pd.concat(frames, ignore_index=True)
delta = {r: json.load(open(f'{EV}/results/{r.lower()}/delta.json')) for r in REGIONS}
# keep each region's calibrated cohort only
scored = scored[scored.apply(lambda x: x.cohort == delta[x.region]['cohort'], axis=1)]
dup = scored.duplicated(['region', 'split', 'model', 'mask_instance', 'seed'], keep=False)
if dup.any():
    print('WARNING duplicate rows:', scored[dup].groupby(['region', 'split', 'model']).size())

def gnn_gate(d):
    return d[(d.model == 'gnn_a4') & d.seed.isin(GATE_SEEDS)]

lines = []
def out(s=''):
    print(s); lines.append(str(s))

# ---------- 1. gate recomputation where all confirm rows exist ----------
out('## Gate recomputation from local scored rows (src/gate.py evaluate_gate)')
gate_rows = []
for r in REGIONS:
    d = scored[scored.region == r]
    have = set(d[d.split == 'f2_confirm'].model)
    if not {'gnn_a4', 'configuration', 'logreg', 'xgboost'} <= have or 'f2_dev' not in set(d.split):
        continue
    frame = pd.concat([gnn_gate(d[d.split == 'f2_confirm']),
                       d[(d.split == 'f2_confirm') & (d.model != 'gnn_a4')],
                       d[(d.split == 'f2_dev') & (d.model != 'gnn_a4')]])
    v = evaluate_gate(frame, PreregConfig(delta=delta[r]['delta'], region=r))
    conf = d[d.split == 'f2_confirm']
    m = gnn_gate(conf).groupby('mask_instance').mrr.mean()
    c = conf[conf.model == v.champion_baseline].groupby('mask_instance').mrr.mean()
    ri = np.log2(m / c)
    gate_rows.append(dict(region=r, champion=v.champion_baseline, model_conf=m.mean(), champ_conf=c.mean(),
                          r_bar=v.r_bar, ci_low=v.ci_low, ci_high=v.ci_high, factor=2 ** v.r_bar,
                          wins=v.n_instances_positive, ties=int((ri == 0).sum()), p_sign=v.p_sign,
                          delta=v.delta, passed=v.passed, ci_method=v.ci_method))
gate_df = pd.DataFrame(gate_rows)
out(gate_df.to_string(index=False, float_format=lambda x: f'{x:.6g}'))
gate_df.to_csv(f'{OUT}/gate_recomputed.csv', index=False)

# ---------- 2. model confirm MRR from committed reference JSONs (all 12) ----------
out('\n## Model F2-confirm per-instance (results/phase3/reference/<CODE>.json, 3 gate seeds)')
ref_inst = {}
for r in REGIONS:
    js = json.load(open(f'{EV}/results/phase3/reference/{r}.json'))
    assert js['cohort'] == delta[r]['cohort'], r
    runs = js['reference']
    assert len(runs) == 3 and all(len(v) == 20 for v in runs.values()), r
    ref_inst[r] = pd.DataFrame(runs).mean(axis=1)  # index instance, 3-seed mean
# cross-check vs local scored rows
for r in REGIONS:
    d = scored[(scored.region == r) & (scored.split == 'f2_confirm')]
    if d.empty:
        continue
    m = gnn_gate(d).groupby('mask_instance').mrr.mean()
    out(f'{r}: max |reference - scored| per instance = {np.abs(m - ref_inst[r].reindex(m.index)).max():.2e}')

# ---------- 3. F2-dev ----------
diag = pd.read_csv(f'{EV}/results/phase3/f2dev-diagnosis/f2dev-instances.csv')
dev_rows = []
for r in REGIONS:
    row = dict(region=r, model_dev_diag=diag[diag.region == r].ref_real.mean(), n_dev=int((diag.region == r).sum()))
    d = scored[(scored.region == r) & (scored.split == 'f2_dev')]
    if (d.model == 'gnn_a4').any():
        g = gnn_gate(d)
        row['model_dev_scored'] = g.groupby('mask_instance').mrr.mean().mean()
        row['model_dev_R10_scored'] = g.groupby('mask_instance').recall_at_10.mean().mean()
        row['model_dev_5seed'] = d[d.model == 'gnn_a4'].groupby('mask_instance').mrr.mean().mean()
    for fam in ['configuration', 'logreg', 'xgboost']:
        b = d[d.model == fam]
        if not b.empty:
            row[f'{fam}_dev'] = b.groupby('mask_instance').mrr.mean().mean()
            row[f'{fam}_dev_R10'] = b.groupby('mask_instance').recall_at_10.mean().mean()
    if {'configuration', 'logreg', 'xgboost'} <= set(d.model):
        row['champion_dev'] = select_champion_baseline(d[d.model != 'gnn_a4'], PreregConfig(region=r))
    dev_rows.append(row)
dev_df = pd.DataFrame(dev_rows)
out('\n## F2-dev (model: diag CSV for all 12; scored rows where local)')
out(dev_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
dev_df.to_csv(f'{OUT}/f2dev_recomputed.csv', index=False)

# ---------- 4. F1-val ----------
f1_rows = []
for r in REGIONS:
    d = scored[(scored.region == r) & (scored.split == 'f1_val')]
    row = dict(region=r)
    g = d[d.model == 'gnn_a4']
    if not g.empty:
        per_seed = g[g.seed.isin(GATE_SEEDS)].pivot_table(index='mask_instance', columns='seed', values='mrr')
        sel = [i for i in per_seed.index if regime(i) != 'T']
        row['f1val_model_4inst'] = per_seed.loc[sel].mean().mean()
        row['f1val_model_5inst'] = per_seed.mean().mean()
        row['f1val_model_T_only'] = per_seed.loc[[i for i in per_seed.index if regime(i) == 'T']].mean().mean()
        if g.seed.nunique() == 5:
            s, dl = calibrate_delta(g, PreregConfig(region=r))
            row['s_recomputed'] = s; row['delta_recomputed'] = dl
            row['s_json'] = delta[r]['s']; row['delta_json'] = delta[r]['delta']
    for fam in ['configuration', 'logreg', 'xgboost']:
        b = d[d.model == fam]
        if not b.empty:
            sel = [i for i in b.mask_instance.unique() if regime(i) != 'T']
            row[f'{fam}_f1val_4inst'] = b[b.mask_instance.isin(sel)].groupby('mask_instance').mrr.mean().mean()
    f1_rows.append(row)
f1_df = pd.DataFrame(f1_rows)
out('\n## F1-val from local scored rows')
out(f1_df.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
f1_df.to_csv(f'{OUT}/f1val_recomputed.csv', index=False)

# ---------- 5. regime split ----------
reg_rows = []
for r in REGIONS:
    m = ref_inst[r]
    conf = scored[(scored.region == r) & (scored.split == 'f2_confirm')]
    champ = gate_df.set_index('region').champion.get(r)
    c = conf[conf.model == champ].groupby('mask_instance').mrr.mean() if champ else None
    for g in REGIMES:
        idx = [i for i in m.index if regime(i) == g]
        row = dict(region=r, regime=g, n=len(idx), model_mrr=m[idx].mean())
        if c is not None:
            row['champion'] = champ
            row['champion_mrr'] = c[idx].mean()
            row['mean_log2_ratio'] = np.log2(m[idx] / c[idx]).mean()
        reg_rows.append(row)
reg = pd.DataFrame(reg_rows)
pooled = reg.groupby('regime', sort=False).agg(model_mrr=('model_mrr', 'mean'), n_regions=('region', 'nunique')).reset_index()
pooled4 = reg.dropna(subset=['champion_mrr']).groupby('regime', sort=False).agg(
    model_mrr_4reg=('model_mrr', 'mean'), champion_mrr_4reg=('champion_mrr', 'mean'),
    mean_log2_ratio_4reg=('mean_log2_ratio', 'mean')).reset_index()
pooled = pooled.merge(pooled4, on='regime')
pooled.insert(0, 'region', 'POOLED')
reg_all = pd.concat([reg, pooled], ignore_index=True)
reg_all.to_csv(f'{OUT}/regime_split.csv', index=False)
out('\n## Regime split (model: 12 regions from reference JSONs; champion: local rows only)')
out(reg.pivot(index='region', columns='regime', values='model_mrr')[REGIMES].reindex(REGIONS).to_string(float_format=lambda x: f'{x:.4f}'))
out(pooled.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
out(reg.dropna(subset=['champion_mrr']).to_string(index=False, float_format=lambda x: f'{x:.4f}'))

# ---------- 6. summary per region ----------
summ = pd.DataFrame({'region': REGIONS})
summ['model_conf_ref'] = [ref_inst[r].mean() for r in REGIONS]
summ = summ.merge(dev_df[['region', 'model_dev_diag']], on='region')
summ['delta_json'] = [delta[r]['delta'] for r in REGIONS]
summ.to_csv(f'{OUT}/summary_model_mrr.csv', index=False)
out('\n## Model MRR summary')
out(summ.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
open(f'{OUT}/recompute_output.txt', 'w').write('\n'.join(lines))
