# Paper analysis

Post-hoc analysis scripts and the tables and figures built from them. All
paths come from the command line, so the scripts run against data you
downloaded, not against the authors' machines. These analyses are exploratory
unless the paper says otherwise; the registered gate decides what counts as
confirmatory.

## Layout

- `data/` holds the CSVs the paper's figures and tables read. `data/SOURCES.md` says where each column comes from.
- `scripts/make_figures.py` draws the figures from `data/`. It needs only matplotlib, numpy and pandas.
- `scripts/posthoc/` holds the scripts that regenerate those CSVs from the repo, the connectome data and the scored rows.
- `scripts/common.py` is the shared argument parser.

## Common arguments

| Argument | Default | Meaning |
|---|---|---|
| `--repo-root` | this repo | `src/`, `data/suites/`, `results/experts.csv`, `results/phase3/` |
| `--data-root` | `<repo-root>/data` | `raw/`, `interim/`, `processed/`, `runs/` |
| `--results-root` | `<repo-root>/results` | `<region>/scored/*.parquet` |
| `--out` | `analysis/out/` (gitignored) | where scripts write CSVs and intermediates |

`make_figures.py` takes `--data` (CSV directory, default `analysis/data`) and
`--out` (default `analysis/figures`, gitignored; LaTeX tables go to `--out/tables`).

Scripts run from any directory, for example
`python analysis/scripts/posthoc/results/recompute.py --results-root /path/to/results`.
They import `src/` from `--repo-root`, so use the project's Python environment.

## Where the inputs come from

- **Repo (committed):** frozen suites in `data/suites/`, `results/experts.csv`, `results/<region>/delta.json`, `results/phase3/`, `analysis/data/transfer_matrix/`.
- **Scored rows** (`<region>/scored/*.scores.parquet` and `*.positives.parquet`): the Zenodo deposit ZENODO_RECORD_TBD, or regenerate with `scripts/score_run.py` and `scripts/run_baselines.py`.
- **Connectome data** (`raw/`, `interim/`, `processed/`): rebuild from the three public releases, or take the processed graphs from the deposit.
- **Checkpoints** (`runs/`, only `specificity/extract.py`): the deposit.

## Scripts

| Script | Writes (in `--out`) | Needs |
|---|---|---|
| `make_figures.py` | `fig_gate.pdf` (`regional_gate.csv`), `fig_regime.pdf` (`regime_split.csv`), `fig_transfer.pdf` (`transfer_ratio.csv`), `tables/transfer_raw.tex`, `tables/transfer_oov.tex` | `data/` only |
| `posthoc/results/recompute.py` | `gate_recomputed.csv`, `f2dev_recomputed.csv`, `f1val_recomputed.csv`, `regime_split.csv`, `summary_model_mrr.csv`, `recompute_output.txt` (gate statistics from `src/gate.py`) | scored rows, repo records |
| `posthoc/results/drop_and_r10.py` | `recall10_local.csv`, `drop_vs_properties.csv` | `recompute.py` output in `--out`, scored rows, `processed/f1` and `processed/f2` |
| `posthoc/results/confirm_positives.py` | `confirm_instance_positives.csv` | `processed/f2`, suites |
| `posthoc/results/cx_strata.py` | `cx_strata.csv` (CX F2-dev strata by direction) | scored CX positives, `processed/f2/cx` |
| `posthoc/results/inspect_scored.py` | prints a summary of every scored file | scored rows |
| `posthoc/data/region_table.py`, `suite_table.py` | `region_table.csv`, `region_overlap_<ds>.csv`, `suite_instances.csv`, `dev_confirm_overlap.csv` | `processed/`, `interim/`, suites |
| `posthoc/data/positive_overlap.py` | `confirm_positive_overlap_with_dev.csv` | `processed/f2`, suites |
| `posthoc/data/zero_weight_partners.py` | `zero_weight_partners_f2_confirm.csv` | `processed/f2`, suites |
| `posthoc/data/positive_only_sensitivity.py` | `positive_only_sensitivity_{summary,instances}.csv` | scored rows (CX, LH, LX, MB), `processed/f2`, suites |
| `posthoc/data/regime_split.py` | `gated_cohort_confirm_mrr_by_regime.csv`, `f2dev_masked_innervation_effect_by_regime.csv` | repo records only |
| `posthoc/data/typemate_heuristic.py [CODES]` | `typemate_<code>.csv` (skips existing files) | `processed/f1`, `processed/f2`, `interim/f1`, suites |
| `posthoc/data/typemate_summary.py` | `typemate_summary.csv`, `typemate_instances_all.csv` (needs `typemate_<code>.csv` for all regions) | scored rows, repo records |
| `posthoc/data/typemate_rows.py` | `typemate_rows_confirm_4regions.csv` | scored rows (CX, LH, LX, MB), `processed/f2`, suites |
| `posthoc/data/build_f1_lookup_inputs.py` | `f1_ids_sorted.npy`, `f1_pairs_<region>.npz`, `f1_lookup_inputs.sha256` (`--raw`, `--processed` default to `data/`) | `raw/proofread_*_783.*`, `processed/f1`. Compare the checksum file with `posthoc/data/f1_lookup_inputs.sha256`. |
| `posthoc/data/match_transfer.py [CODES]` | `match_<code>.csv`, `match_rows_<code>.parquet` | `build_f1_lookup_inputs.py` output and `typemate_heuristic.py` imports, `raw/banc/banc_888_meta.feather` |
| `posthoc/data/match_summary.py` | `match_summary.csv`, `match_restricted_rows.csv`, `match_by_regime.csv` | `match_transfer.py` output, scored rows |
| `posthoc/data/innervation_share.py f1\|f2` | `innervation_share_<f1\|f2>.csv` | `raw/` neuropil-count files, `processed/` |
| `posthoc/data/nt_composition.py`, `nt_agreement.py` | `nt_composition_by_region.csv`, `nt_crosstab_f1match_vs_f2.csv` | `raw/Supplemental_file1_neuron_annotations.tsv`, `raw/banc/banc_888_meta.feather` |
| `posthoc/data/f1_totals.py` | prints F1 synapse totals | `raw/proofread_connections_783.feather` |
| `posthoc/specificity/extract.py` | `matrix_*.csv`, `baselines.csv`, `cells.csv`, `region_stats.csv`, `pair_predictors.csv` (`data/transfer_raw_mrr.csv`, `transfer_ratio.csv` and `transfer_masked_oov.csv` hold the same values as `matrix_raw`, `matrix_home_norm` and `matrix_oov`) | `analysis/data/transfer_matrix/`, `processed/`, `runs/` checkpoints, torch |
| `posthoc/specificity/analyze.py`, `extra.py` | `diagonal_dominance.csv`, `two_way_decomposition.csv`, `correlations_*.csv`, `cell_table.csv`, `asymmetry_pairs.csv`, `analysis_output.txt`, `extra_output.txt` | `extract.py` output in `--out`. `analyze.py --skeleton FILE.tex` cross-checks the manuscript's transfer table. |

Run `extract.py` before `analyze.py` before `extra.py`; `typemate_heuristic.py`
before `typemate_summary.py`; `build_f1_lookup_inputs.py`, then
`typemate_heuristic.py`, `match_transfer.py` and `match_summary.py` in that order.
`regional_gate.csv` is assembled by hand from the registered confirm-score records and the
records listed in `data/SOURCES.md`.
