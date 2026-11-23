# Dataset download scripts — usage

`sync_flywire_dataset.py` downloads F1 (FlyWire v783) and `sync_banc_dataset.py`
downloads F2 (BANC v888). F3 (male CNS v1.0) comes from `fetch_f3.py`, and
`fetch_checkpoints.py` downloads the gate-cohort checkpoints.

Each is stdlib-only Python **3.9+** (use 3.13 if you have it, 3.9 is EOL) and runs
identically on **Windows** and **macOS**. No `pip install` needed (if your
interpreter has no CA store and you see `CERTIFICATE_VERIFY_FAILED`, the fix is
`python -m pip install certifi`). Downloads are resumable and every file is checked
against a pinned size and sha256. Nothing is uploaded.

## Run

```bash
# F1: download -> verify (~10.6 GB)
python sync_flywire_dataset.py

# F2: download -> verify (~23 GB), into data/raw/banc/
python sync_banc_dataset.py

# Individual steps (either script)
python sync_flywire_dataset.py --step download   # fetch into data/raw/ (resumable)
python sync_flywire_dataset.py --step verify     # size + checksum every file

# Checkpoints: 36 best.pt files (~340 MB) into data/runs/<run_id>/best.pt
python fetch_checkpoints.py
```

On Windows use `py sync_flywire_dataset.py` if `python` isn't on PATH.

`fetch_checkpoints.py` reads the Zenodo record id from `ZENODO_RECORD` in
`sync_banc_dataset.py` and stops with a message while it is unset.
`sync_banc_dataset.py` downloads the synapse table from Harvard Dataverse and
prints the bucket URLs of the current meta, edgelist and per-neuron files, whose
pinned bytes are not publicly hosted. Files placed in `data/raw/banc/` by hand
are checked against the pinned sha256, which the current meta file does not
match.

## What they download

| Source | Files |
|---|---|
| Zenodo 10676866 (v783.0) | `flywire_synapses_783.feather` (9.5 GB), `proofread_connections_783.feather`, `proofread_root_ids_783.npy`, `per_neuron_neuropil_count_{pre,post}_783.feather` |
| flyconnectome/flywire_annotations @ v3.1.0 | `Supplemental_file1_neuron_annotations.tsv` + 4 supplemental files |
| Zenodo deposit (`ZENODO_RECORD`, not yet published) | The 36 gate-cohort `best.pt` files. |
| Harvard Dataverse, datafile 13992863 | BANC `banc_888_synapses_v2_enriched.parquet` (17 GB) |

F1 files are verified against Zenodo's md5 or pinned sha256 (annotations); BANC and
checkpoint files against pinned sha256.

## Other scripts

`build_region.py --config train/configs/<code>.toml --dataset {f1,f2,f3}` builds
one region's graph, features, suite inputs and suites (`f1_val`; `f2_dev`,
`f2_confirm` and `f2_side`; the exploratory `f3_eval`) into the data layout, calling
`extract_region`'s graph builder, `build_features.py`,
`prepare_suite_inputs.py` and `build_suites.py` in-process. Its `extras` step
runs `build_node_extras.py` for F1, writing `extras.npz` beside the graph for
the `annotation_groups` model switch; F2 and F3 skip it, since the file aligns
F1 root ids from the FlyWire annotation table. Raw files sit
under `--raw-dir` (default `<data-root>/raw`, with `--data-root` defaulting to
`$FLYARCH_DATA_ROOT`, then `data`) at their pinned names: F1's at the top, BANC's
under `banc/`, F3's under `f3/` (written by `fetch_f3.py`). `--steps` picks steps; `--report` prints the pre-build table
(cell-type sizes, F2's type-match rate, and each typed draw with its size,
its single-masked positives and, for F2, the nodes BANC names by the drawn
type that the mask leaves visible) before any suite is written. Each F2 suite
gets that last count per T instance in `<suite>.same_name.json`, committed
with it. Build F1 first.
F2 goes calibrate, commit, build: `--dataset f2 --calibrate` prints the
threshold and its curve and writes nothing; a person commits it as
`f2_threshold` in one
commit; then the F2 build runs, and refuses to while `f2_threshold` is
absent. It needs the training dependencies, since it reads the region
config through `train/`.

`build_node_extras.py [--annotations TSV] PROCESSED_DIR [PROCESSED_DIR ...]`
writes `extras.npz` into each processed region directory: per-neuron annotation
groups aligned to the graph's node order, which `--annotation-groups` reads.
It defaults to `data/raw/Supplemental_file1_neuron_annotations.tsv`, so the
arrays are F1's; `build_region.py`'s `extras` step runs it for F1.

`stop_run.py RUN_DIR (--pause | --cancel) [--reason TEXT] [--no-wait]
[--timeout SECONDS]` pauses or cancels a training run from its run directory
(`data/runs/<run_id>`). For a live run it writes `stop-request.json` and waits
up to 300 s by default for the run to act on it; a stopped run can be
cancelled directly. It uses only the standard library and `src/`, needs no
training dependencies, and exits 1 on a refusal, a timeout or a run
directory it cannot write.
