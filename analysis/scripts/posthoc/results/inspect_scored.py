import glob, pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import parse_args  # noqa: E402
A = parse_args(__doc__)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
roots = [str(A.results_root)]
for root in roots:
    for f in sorted(glob.glob(root + '/*/scored/*.scores.parquet')):
        d = pd.read_parquet(f)
        g = d.groupby(['region', 'split', 'model'], dropna=False).agg(
            n=('mrr', 'size'), seeds=('seed', 'nunique'), cohort=('cohort', 'first'),
            milestone=('milestone', 'first'), commit=('scoring_commit', 'first'))
        print('==', f.replace(root, '').split('scored')[0], f[-40:-16])
        print(g.to_string())
