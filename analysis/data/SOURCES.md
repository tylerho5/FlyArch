# Data sources for the figures and tables

Every number in `data/*.csv` comes from a committed project record or a
recomputation from scored rows. Paths are relative to the repository root.
`scripts/` regenerates most of these files; `README.md` lists which script
writes which CSV and where its inputs come from.

## regional_gate.csv

| column | source |
|---|---|
| `f1_nodes`, `f1_edges`, `f2_threshold` | `data/processed/f1/<r>/` region graphs and the frozen F2 suite thresholds (`data/suites/f2/<r>/*.json`), recounted |
| `f1val_model` | `results/experts.csv` and the run manifests; mean over the four non-typed F1-val instances and gate seeds 0-2 (`train/train.py` selection set) |
| `f2dev_model` | `results/phase3/f2dev-diagnosis/f2dev-instances.csv`, column `ref_real`, mean over 15 instances |
| `f2dev_champion` | `gate_rows`: recomputed from the gate's scored baseline rows (`results/<r>/scored/`, in the Zenodo deposit ZENODO_RECORD_TBD); `pr195_body`: PR #195 description; `transfer_report`: `analysis/data/transfer_matrix/` (a separate baseline scoring on another machine, PR #226). The gate rows for the other eight regions were not available when the table was built. |
| `f2conf_model` | `results/phase3/reference/<R>.json`, mean over 20 instances of the three-seed instance means |
| `champion`, `delta`, `rbar`, `ci_lo`, `ci_hi`, `wins`, `n_nonzero`, `p_sign`, `verdict` | the registered confirm-score records; `delta` also `results/<r>/delta.json`. Recomputed with `src/gate.py` for CX, LH, LX and MB (exact match). |

## regime_split.csv

Model F2-confirm MRR by mask regime: mean over the four instances of each
regime of the three-seed instance mean, from `results/phase3/reference/<R>.json`.
Computed for this paper; not registered in advance (the split was added to the
preregistration's exploratory list on 2026-11-17, after every confirmatory read).
