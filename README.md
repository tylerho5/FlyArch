# FlyArch

Code, frozen evaluation suites and results for the paper *Predicting the
synaptic partners of masked neurons in a second fly connectome with
region-specific graph models* (Hassan, Ho and Aslam).

## What the study tests

A graph neural network "expert" is trained for each of 12 brain regions on
one female *Drosophila* connectome, FlyWire v783 (F1). Each expert is then
tested on a second female brain, BANC v888 (F2), which is never used for
training: neurons are removed from the same region of F2, and the expert
ranks each removed neuron's true synaptic partners against sampled
negatives. The male CNS v1.0 connectome (F3) is an exploratory cross-sex
test.

One pre-registered comparison per region decides the result: the expert
against the best tuned non-graph baseline on 20 frozen F2 mask instances
(the F2-confirm suite). A region passes only if the bootstrap confidence
bound on the mean log2 MRR ratio clears a measured noise floor (`delta`)
and a sign test also passes. `src/gate.py` implements the rule.

## Result

- Phase 1, the Central Complex (CX): passed.
- Phase 2, the other 11 regions (AL, GNG, INP, LH, LX, MB, OL, PENP, SNP,
  VLNP, VMNP): all passed.
- Phase 3, one integrated whole-brain system: failed. It beat all 17
  opponents on edges that cross region boundaries, and passed the regional
  guard against the Phase 2 experts in 0 of 12 regions. Under the registered
  rules the phase stopped there, and it is reported as a negative result.

The registered suite digests and Phase 2 verdicts are in
`results/registered.json`.

## Install

Python 3.11 or newer. The tests run on CPU. This is the recipe the test
workflow uses (`.github/workflows/tests.yml`):

```bash
python -m venv .venv
grep -vE '^(--find-links|pyg-lib|nvidia-nccl)' requirements/training.in > cpu-requirements.txt
.venv/bin/python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r cpu-requirements.txt
.venv/bin/python -m pytest -q
```

For GPU training, build the container from `docker/training.Dockerfile`
(PyTorch 2.7.1, CUDA 12.8, `requirements/training.lock` installed with
hashes) through `compose.training.yml`, or install the CUDA build of torch
2.7 in the virtual environment.

## Data

The scripts below download into `data/raw/` and check every file against a
pinned size and checksum. They need only the Python standard library
([`scripts/README.md`](scripts/README.md)).

| Script | What it fetches |
|---|---|
| `scripts/sync_flywire_dataset.py` | F1: FlyWire v783 from Zenodo record 10676866 and the annotations release v3.1.0 (about 10.6 GB) |
| `scripts/sync_banc_dataset.py` | F2: BANC v888, the pinned files (see below) |
| `scripts/fetch_f3.py` | F3: male CNS v1.0 from its public release |
| `scripts/fetch_checkpoints.py` | The 36 gate-cohort `best.pt` checkpoints (three gate seeds per region, about 340 MB) from the Zenodo deposit |

The F2 results were computed on BANC files pinned on 2026-10-11 (sizes and
sha256 in `MANIFEST` in `scripts/sync_banc_dataset.py`). BANC's
bucket rewrites files in place, and it replaced `banc_888_meta.feather` on
2026-11-11, so the pinned meta file can no longer be downloaded. The current
file differs from it only in the soma `side` column, which BANC filled in for
14,901 neurons that had none. No neuron changed side, and cell types, FAFB
matches and the proofread set are unchanged. `side` is a model input, so F2
scores computed from the current file can differ slightly from the reported
ones. On AL, the most affected region (16% of its F2 nodes), the mean F2-dev
MRR of the three gate checkpoints moved from 0.5833 to 0.5818.
`scripts/sync_banc_dataset.py` downloads the synapse table from Harvard
Dataverse and prints the bucket URLs of the current meta, edgelist and
per-neuron files. Files placed in `data/raw/banc/` by hand are checked
against the pinned checksums, which the current meta file does not match.

The gate checkpoints will be deposited on Zenodo. Until the record id replaces
`ZENODO_RECORD_TBD`, `scripts/fetch_checkpoints.py` stops with a message
naming the constant.

The frozen mask suites and their negatives are committed in `data/suites/`.
The registered runs' manifests are in `results/manifests/`, the expert
registry is `results/experts.csv`, and each region's calibration is
`results/<code>/delta.json`.

## Reproducing a region

`<code>` is the lower-case region code (`cx`, `al`, ...). Each region's
recipe is `train/configs/<code>.toml`. Data paths default to `data/`, or
`$FLYARCH_DATA_ROOT` when set. Each script's `--help` lists its options.

1. Build the region's graphs, features and suites, F1 first:

   ```bash
   python scripts/build_region.py --config train/configs/<code>.toml --dataset f1
   python scripts/build_region.py --config train/configs/<code>.toml --dataset f2
   ```

   The F2 build reads the committed `f2_threshold`, and the suites it
   writes must match the committed ones in `data/suites/`.

2. Train the five calibration seeds:

   ```bash
   for seed in 0 1 2 3 4; do
     python scripts/train_m1.py --config train/configs/<code>.toml \
       --runs-dir data/runs --seed "$seed" --device cuda \
       --title "<code> gate cohort" || break
   done
   ```

   To rescore the registered experts instead, fetch the gate checkpoints
   with `scripts/fetch_checkpoints.py`. The scoring scripts find runs
   through `data/runs/<run_id>/manifest.json`, so copy each run's manifest
   there from `results/manifests/<run_id>.json`.

3. Calibration is frozen. `run_gate.py calibrate` refuses to run for a
   registered region, because its `delta.json` is committed and listed in
   `results/release-evidence.json`. The cohort digest to pass below is in
   that `delta.json`.

4. Score the expert and fit and score the three baselines on F2-confirm:

   ```bash
   python scripts/score_run.py --config train/configs/<code>.toml \
     --run data/runs/<run-id> --suite data/suites/f2/<code>/f2_confirm.json \
     --milestone replay-1
   python scripts/run_baselines.py --config train/configs/<code>.toml \
     --suite data/suites/f2/<code>/f2_confirm.json --milestone replay-1
   ```

   Pass `--run` once for each gate seed. `run_baselines.py` refits the
   baselines on F1 and selects the champion on F2-dev itself.

5. Run the gate:

   ```bash
   python scripts/run_gate.py --config train/configs/<code>.toml gate --cohort <cohort>
   ```

   It prints the region's confirm-score record.

Retrained checkpoints reproduce the registered numbers only on the same
hardware and software stack: CUDA runs are deterministic on one GPU and
stack, and MPS runs are not.

## Reproducing Phase 3

The gate's inputs are committed in `results/phase3/confirm-gate-1.json`.
This command recomputes the registered FAIL (boundary comparison 17 of 17,
regional guard 0 of 12):

```bash
python scripts/phase3_gate.py --components results/phase3/confirm-gate-1.json
```

## Calibration evidence without tags

During the study each region's `delta.json` was frozen by a Git tag
(`delta-<code>-*`) before any confirm data was read. This
repository has no tags. `results/release-evidence.json` lists, for each of
the 13 calibrations, the original tag name, tag commit, cohort and the
`delta.json` SHA-256. With no local tag, confirm scoring and the gate accept
a `delta.json` only if its bytes match that entry. This verifies the
published calibration bytes. It does not prove when they were fixed.

## Code digest

Each run records a digest of the training code (`_code_digest()` in
`train/train.py`, over the files in `_CODE_FILES`), and that digest enters
its run ID. At this release it is:

```
dae2fd0a98d1369900e5a672b8cc62903ceeb0f08acf221b8d87c50fc05967794bbfe59c4f01969ab9cc4e2fd5b61362cb7864fbf77d67728d645c9084373de8
```

The registered runs were trained in the development repository and record
that repository's digests. Removing the development tooling changed
`train/train.py`, so a run trained from this repository gets a different
run ID from the registered run with the same settings.

## Licenses

- Code: MIT ([`LICENSE`](LICENSE)).
- `data/suites/`: CC BY-NC 4.0, with attribution to the three source
  datasets ([`data/suites/LICENSE`](data/suites/LICENSE)).

The connectome datasets keep their own licenses: FlyWire v783 CC BY-NC 4.0,
BANC v888 CC BY 4.0, male CNS v1.0 CC BY.

## Citation

Cite the paper as given in [`CITATION.cff`](CITATION.cff), along with the
three datasets it lists.
