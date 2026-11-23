"""Fetch F3 (FlyEM male-cns:v1.0) for every registry region from the bulk release.

    py -3.13 scripts/fetch_f3.py
    py -3.13 scripts/fetch_f3.py --regions CX AL --out data/raw/f3
    py -3.13 scripts/fetch_f3.py --from-bulk /path/to/f3-bulk
    py -3.13 scripts/fetch_f3.py --verify-neuprint
    py -3.13 scripts/fetch_f3.py --verify-weights

Streams the public bulk feathers (``gs://flyem-male-cns/v1.0/connectome-data/
flat-connectome/``, no token) one record batch at a time with column
projection, and writes four files under ``data/raw/f3/``:

* ``connections.parquet``: ``pre_pt_root_id``, ``post_pt_root_id``,
  ``neuropil`` (registry label), ``syn_count``. Counted from syn-partners rows
  grouped by (pre, post, primary ROI); the bulk release has no per-ROI weight
  table. Population neurons at both ends, no autapses.
* ``synapses.parquet``: ``root_id``, ``x``, ``y``, ``z`` (int32 nm, voxel * 8,
  the release's own axes), ``neuropil``, one row per synapse end.
* ``innervation.parquet``: ``root_id``, ``neuropil``, ``kind``, ``count``.
* ``annotations.parquet``: the release annotation table with ``predicted_nt``
  joined on; raw, like F2's. ``build_features.f3_annotations`` normalizes it.

``--verify-neuprint`` instead compares ``Client.meta()["roiInfo"]`` with the ROI
tables (needs ``NEUPRINT_TOKEN`` in ``.env`` and ``pip install neuprint-python``).

``--verify-weights`` checks ``connections.parquet`` against the release's
``connectome-weights`` (1 GB, all ROIs): per pair, the per-ROI weights summed
must not exceed the release weight, and equal it unless some of the pair's
synapses sit in a dropped ROI. Run it after a full fetch.

Side convention: Janelia ``(L)``/``(R)`` name the fly's left and right, and the
registry's ``_L``/``_R`` (FlyWire v783) do the same, so paired names map
``X(L)`` -> ``X_L`` with no swap. ``src.harmonize.f3_hemisphere`` applies the
same convention to IB, which is unpaired in F3.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.fs as pafs  # noqa: E402
import pyarrow.ipc as ipc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from src.regions import REGIONS, region_labels  # noqa: E402

NEUPRINT_SERVER = "neuprint.janelia.org"
NEUPRINT_DATASET = "male-cns:v1.0"
BULK_URI = "flyem-male-cns/v1.0/connectome-data/flat-connectome/"
PARTNERS = "syn-partners-male-cns-v1.0-minconf-0.5.feather"
POINTS = "syn-points-male-cns-v1.0-minconf-0.5.feather"
ANNOTATIONS = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
WEIGHTS = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS = "body-neurotransmitters-male-cns-v1.0.feather"
VOXEL_NM = 8
PREFETCH = 6
RETRIES = 5  # per batch; a multi-hour GCS stream meets transient DNS/network errors


def _paired(base: str, label: str | None = None) -> dict[str, str]:
    label = label or base
    return {f"{base}(L)": f"{label}_L", f"{base}(R)": f"{label}_R"}


_PAIRED_BASES = (
    "AL", "AME", "AOTU", "ATL", "AVLP", "BU", "CAN", "CRE", "EPA", "FLA", "GOR", "ICL",
    "IPS", "LA", "LAL", "LH", "LO", "LOP", "ME", "PLP", "PVLP", "SCL", "SIP", "SLP", "SMP",
    "SPS", "VES", "WED", "AMMC", "GA",
)

#: Janelia ROI name (primary or subprimary) -> registry label. Many-to-one:
#: CA -> MB_CA, PED -> MB_PED, a'L/aL -> MB_VL, b'L/bL/gL -> MB_ML.
JANELIA_ROI_TO_LABEL: dict[str, str] = {
    **{k: v for b in _PAIRED_BASES for k, v in _paired(b).items()},
    **_paired("CA", "MB_CA"),
    **_paired("PED", "MB_PED"),
    **{k: v for b in ("a'L", "aL") for k, v in _paired(b, "MB_VL").items()},
    **{k: v for b in ("b'L", "bL", "gL") for k, v in _paired(b, "MB_ML").items()},
    "LAL(-GA)(L)": "LAL_L",
    "LAL(-GA)(R)": "LAL_R",
    "SAD(-AMMC)": "SAD",
    **{n: n for n in ("EB", "FB", "PB", "NO", "GNG", "PRW", "SAD")},
}

#: Subprimary names that replace the primary ROI of a synapse point. AMMC and
#: GA exist in F3 only as subprimaries; each sits in its primary parent's region.
SUBPRIMARY_ROIS = frozenset(
    {"AMMC(L)", "AMMC(R)", "SAD(-AMMC)", "GA(L)", "GA(R)", "LAL(-GA)(L)", "LAL(-GA)(R)"}
)

#: Unpaired in F3, paired in the registry: split per synapse by hemisphere.
SPLIT_BY_HEMISPHERE = {"IB"}


def _with_sides(*bases: str) -> set[str]:
    return {n for b in bases for n in (b, f"{b}(L)", f"{b}(R)")}


#: Dropped on purpose: VNC, neck, nerves, catch-alls, and AB (F1 has no
#: counterpart; the alternative is mapping it to FB).
EXCLUDED_ROIS = frozenset(
    {"AB(L)", "AB(R)", "<unspecified>", "CentralBrain-unspecified", "CV", "CV-unspecified",
     "VNC-unspecified", "Optic-unspecified(L)", "Optic-unspecified(R)", "NotPrimary", "CRN", "ANm"}
    | _with_sides("HTct(UTct-T3)", "IntTct", "LTct", "NTct(UTct-T1)", "WTct(UTct-T2)", "Ov", "AbNT")
    | _with_sides(*(f"LegNp(T{i})" for i in (1, 2, 3)))
    | _with_sides(*(f"mVAC(T{i})" for i in (1, 2, 3)))
    | _with_sides(*(f"AbN{i}" for i in (1, 2, 3, 4)))
    | _with_sides("ADMN", "CvN", "DMetaN", "DProN", "MesoAN", "MesoLN", "MetaLN", "PDMN",
                  "PrN", "ProAN", "ProCN", "ProLN", "VProN")
)


def log(message: str) -> None:
    print(f"[fetch-f3] {message}", flush=True)


def load_env(path: Path) -> dict:
    env: dict = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip()
    except FileNotFoundError:
        pass
    return env


def translate_rois(roi_conn: pd.DataFrame) -> pd.DataFrame:
    """neuPrint-shaped ROI connections -> the registry's connections table.

    ``roi_conn`` has ``bodyId_pre``, ``bodyId_post``, ``roi``, ``weight`` and a
    ``side`` column (``"L"``/``"R"``) on rows whose ROI is in
    ``SPLIT_BY_HEMISPHERE``; it picks IB_L or IB_R and a missing side raises.
    Excluded ROIs are dropped, an ROI in neither table raises and is named, and
    duplicate rows are summed.
    """
    roi = roi_conn["roi"].astype(str)
    unknown = sorted(set(roi) - set(JANELIA_ROI_TO_LABEL) - EXCLUDED_ROIS - SPLIT_BY_HEMISPHERE)
    if unknown:
        raise ValueError(
            f"unmapped Janelia ROI(s): {unknown}; add them to JANELIA_ROI_TO_LABEL "
            "or EXCLUDED_ROIS"
        )
    label = roi.map(JANELIA_ROI_TO_LABEL)
    split = roi.isin(SPLIT_BY_HEMISPHERE)
    if split.any():
        side = roi_conn["side"] if "side" in roi_conn else pd.Series(None, index=roi_conn.index)
        if not side[split].isin(["L", "R"]).all():
            raise ValueError(f"{sorted(SPLIT_BY_HEMISPHERE)} rows need a side of 'L' or 'R'")
        label[split] = roi[split] + "_" + side[split]
    out = pd.DataFrame({
        "pre_pt_root_id": roi_conn["bodyId_pre"].astype("int64"),
        "post_pt_root_id": roi_conn["bodyId_post"].astype("int64"),
        "neuropil": label,
        "syn_count": roi_conn["weight"].astype("int64"),
    })[label.notna()]
    keys = ["pre_pt_root_id", "post_pt_root_id", "neuropil"]
    return out.groupby(keys, as_index=False)["syn_count"].sum()


class Bulk:
    """Column-projected, batch-at-a-time reader over the bulk feathers."""

    def __init__(self, local_dir: Path | None) -> None:
        self.local_dir = local_dir
        self.gcs = None if local_dir else pafs.GcsFileSystem(anonymous=True)

    def _open(self, name: str, columns: list[str] | None = None):
        if self.local_dir:
            source = pa.memory_map(str(self.local_dir / name))
        else:
            source = self.gcs.open_input_file(BULK_URI + name)
        if columns is None:
            return ipc.open_file(source)
        schema = ipc.open_file(source).schema
        idx = [schema.get_field_index(c) for c in columns]
        return ipc.open_file(source, options=ipc.IpcReadOptions(included_fields=idx))

    def read_all(self, name: str) -> pd.DataFrame:
        return self._open(name).read_all().to_pandas()

    def batches(self, name: str, columns: list[str]):
        """Yield ``(index, total, DataFrame)``; ``PREFETCH`` batches fetched in parallel."""
        readers = [self._open(name, columns) for _ in range(PREFETCH)]
        total = readers[0].num_record_batches
        with ThreadPoolExecutor(PREFETCH) as pool:
            for start in range(0, total, PREFETCH):
                idx = range(start, min(start + PREFETCH, total))
                futures = [pool.submit(_get_batch, readers[k], i) for k, i in enumerate(idx)]
                for i, fut in zip(idx, futures):
                    yield i, total, fut.result().to_pandas()


def _get_batch(reader, i: int):
    """``reader.get_batch(i)``, retried with backoff on I/O errors."""
    for attempt in range(RETRIES):
        try:
            return reader.get_batch(i)
        except OSError as exc:
            if attempt == RETRIES - 1:
                raise
            log(f"batch {i}: {exc.__class__.__name__}, retry {attempt + 1}/{RETRIES - 1}")
            time.sleep(30 * (attempt + 1))


def _compact(frames: list[pd.DataFrame], keys: list[str], value: str) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True).groupby(keys, as_index=False)[value].sum()


def fetch_connections(bulk, population, hemisphere, regions) -> pd.DataFrame:
    wanted = {label for r in regions for label in region_labels(r)}
    keys = ["pre_pt_root_id", "post_pt_root_id", "neuropil"]
    parts: list[pd.DataFrame] = []
    excluded: dict[str, int] = {}
    for i, total, df in bulk.batches(PARTNERS, ["body_pre", "body_post", "primary_post", "x_post"]):
        pre, post = df["body_pre"].to_numpy(), df["body_post"].to_numpy()
        df = df[np.isin(pre, population) & np.isin(post, population) & (pre != post)]
        no_roi = df["primary_post"].isna().to_numpy()
        if no_roi.any():
            excluded["(no ROI)"] = excluded.get("(no ROI)", 0) + int(no_roi.sum())
            df = df[~no_roi]
        if df.empty:
            continue
        roi = df["primary_post"].astype(str).to_numpy()
        for name, n in pd.Series(roi[np.isin(roi, list(EXCLUDED_ROIS))]).value_counts().items():
            excluded[name] = excluded.get(name, 0) + int(n)
        side = np.full(len(df), "", dtype=object)
        ib = np.isin(roi, list(SPLIT_BY_HEMISPHERE))
        if ib.any():
            side[ib] = hemisphere(df["x_post"].to_numpy()[ib].astype("float64") * VOXEL_NM)
        grouped = (pd.DataFrame({"bodyId_pre": df["body_pre"].to_numpy(),
                                 "bodyId_post": df["body_post"].to_numpy(), "roi": roi, "side": side})
                   .groupby(["bodyId_pre", "bodyId_post", "roi", "side"], as_index=False)
                   .size().rename(columns={"size": "weight"}))
        out = translate_rois(grouped)
        parts.append(out[out["neuropil"].isin(wanted)])
        if len(parts) >= 64:
            parts = [_compact(parts, keys, "syn_count")]
        if i % 200 == 0:
            log(f"partners batch {i}/{total}")
    connections = _compact(parts, keys, "syn_count")
    for name, n in sorted(excluded.items(), key=lambda kv: -kv[1]):
        log(f"excluded ROI {name}: {n} synapse(s) between population neurons")
    ab = excluded.get("AB(L)", 0) + excluded.get("AB(R)", 0)
    cx = int(connections.loc[connections["neuropil"].isin(region_labels("CX")), "syn_count"].sum())
    if ab + cx:
        log(f"AB: {ab} synapse(s), {ab / (ab + cx):.2%} of AB + retained CX ({ab + cx})")
    return connections


def fetch_synapses(bulk, population, hemisphere, regions, out_dir: Path) -> pd.DataFrame:
    """Write synapses.parquet incrementally; return the innervation table."""
    wanted = {label for r in regions for label in region_labels(r)}
    keys = ["root_id", "neuropil", "kind"]
    schema = pa.schema([("root_id", pa.int64()), ("x", pa.int32()), ("y", pa.int32()),
                        ("z", pa.int32()), ("neuropil", pa.string())])
    counts: list[pd.DataFrame] = []
    with pq.ParquetWriter(out_dir / "synapses.parquet", schema) as writer:
        for i, total, df in bulk.batches(
                POINTS, ["x", "y", "z", "kind", "body", "primary", "subprimary"]):
            df = df[np.isin(df["body"].to_numpy(), population)]
            kind = df["kind"].astype(str).map({"PreSyn": "pre", "PostSyn": "post"})
            df, kind = df[kind.notna().to_numpy()], kind.dropna().to_numpy()
            if df.empty:
                continue
            primary = df["primary"].astype(str).to_numpy()
            sub = df["subprimary"].astype(str).to_numpy()
            roi = np.where(np.isin(sub, list(SUBPRIMARY_ROIS)), sub, primary)
            label = pd.Series(roi).map(JANELIA_ROI_TO_LABEL).to_numpy(dtype=object)
            ib = roi == "IB"
            if ib.any():
                label[ib] = "IB_" + np.asarray(
                    hemisphere(df["x"].to_numpy()[ib].astype("float64") * VOXEL_NM), dtype=object)
            ok = pd.Series(label).isin(wanted).to_numpy()
            if not ok.any():
                continue
            table = pd.DataFrame({
                "root_id": df["body"].to_numpy()[ok],
                **{c: (df[c].to_numpy()[ok].astype("int64") * VOXEL_NM).astype("int32")
                   for c in ("x", "y", "z")},
                "neuropil": label[ok],
            })
            writer.write_table(pa.Table.from_pandas(table, schema=schema, preserve_index=False))
            counts.append(table.assign(kind=kind[ok]).groupby(keys, as_index=False)
                          .size().rename(columns={"size": "count"}))
            if len(counts) >= 64:
                counts = [_compact(counts, keys, "count")]
            if i % 200 == 0:
                log(f"points batch {i}/{total}")
    return _compact(counts, keys, "count")


def verify_neuprint(env_path: Path) -> int:
    token = load_env(env_path).get("NEUPRINT_TOKEN") or ""
    if not token:
        log(f"NEUPRINT_TOKEN not set in {env_path}; get one at https://neuprint.janelia.org")
        return 1
    from neuprint import Client

    client = Client(NEUPRINT_SERVER, dataset=NEUPRINT_DATASET, token=token)
    known = set(JANELIA_ROI_TO_LABEL) | EXCLUDED_ROIS | SPLIT_BY_HEMISPHERE
    missing = sorted(set(client.meta["roiInfo"]) - known)
    log(f"{len(missing)} neuPrint ROI name(s) in neither table: {missing}")
    return 1 if missing else 0


def verify_weights(bulk, out: Path) -> int:
    """Per-ROI weights summed per pair vs ``connectome-weights``; fails if any pair sums above."""
    ours = (pd.read_parquet(out / "connections.parquet")
            .groupby(["pre_pt_root_id", "post_pt_root_id"])["syn_count"].sum().rename("ours").reset_index()
            .rename(columns={"pre_pt_root_id": "body_pre", "post_pt_root_id": "body_post"}))
    parts = [ours.merge(df, on=["body_pre", "body_post"])
             for _, _, df in bulk.batches(WEIGHTS, ["body_pre", "body_post", "weight"])]
    merged = pd.concat(parts)
    n, equal = len(ours), int((merged["ours"] == merged["weight"]).sum())
    above, missing = int((merged["ours"] > merged["weight"]).sum()), n - len(merged)
    log(f"{n} pairs: {equal} equal to the release weight, {n - equal - above - missing} below "
        f"(dropped-ROI synapses), {above} above, {missing} absent from connectome-weights")
    log(f"synapses: {int(merged['ours'].sum())} of {int(merged['weight'].sum())} release weight "
        f"({merged['ours'].sum() / merged['weight'].sum():.2%})")
    return 1 if above or missing else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--regions", nargs="+", default=list(REGIONS), choices=sorted(REGIONS))
    parser.add_argument("--from-bulk", type=Path, default=None,
                        help="local directory holding the same bulk feather files")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "raw" / "f3")
    parser.add_argument("--verify-neuprint", action="store_true")
    parser.add_argument("--verify-weights", action="store_true")
    parser.add_argument("--env", type=Path, default=REPO_ROOT / ".env")
    args = parser.parse_args(argv)

    if args.verify_neuprint:
        return verify_neuprint(args.env)

    from src import harmonize

    bulk = Bulk(args.from_bulk)
    if args.verify_weights:
        return verify_weights(bulk, args.out)
    stage = args.out / ".partial"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)

    raw = bulk.read_all(ANNOTATIONS)
    population = np.asarray(harmonize.f3_neuron_ids(raw), dtype="int64")
    log(f"population: {len(population)} neurons")

    nt = bulk.read_all(NEUROTRANSMITTERS).drop_duplicates("body").set_index("body")["predicted_nt"]
    raw["predicted_nt"] = raw["bodyId"].map(nt)
    raw.to_parquet(stage / "annotations.parquet", index=False)
    _, unmapped = harmonize.normalize_f3_annotations(raw)  # logged only; the build normalizes
    for field, values in unmapped.items():
        for value, n in sorted(values.items(), key=lambda kv: -kv[1]):
            log(f"unmapped {field} {value!r}: {n} neuron(s)")

    hemisphere = harmonize.f3_hemisphere
    connections = fetch_connections(bulk, population, hemisphere, args.regions)
    connections.to_parquet(stage / "connections.parquet", index=False)
    log(f"connections: {len(connections)} rows")

    innervation = fetch_synapses(bulk, population, hemisphere, args.regions, stage)
    innervation.to_parquet(stage / "innervation.parquet", index=False)
    for path in stage.iterdir():
        os.replace(path, args.out / path.name)
    stage.rmdir()
    log(f"innervation: {len(innervation)} rows; wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
