"""Render the cross-region transfer matrix CSVs as a markdown page and SVG heatmaps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# train.region_config pulls torch through train.train, so the tuple is copied from there;
# tests/test_region_config.py keeps the two equal.
TRAINING_ORDER = (
    "CX", "AL", "LH", "MB", "LX", "SNP", "INP", "VMNP", "VLNP", "PENP", "GNG", "OL",
)
SPLITS = {"f2_dev": "f2", "f1_val": "f1"}
NEEDED = ("expert_region", "region", "variant", "mask_instance", "seed", "degree_setting", "mrr", "model",
          "split", "dataset", "masked_oov_frac")
FIRST_LOOK = (("AL", "LH"), ("LH", "AL"), ("CX", "LX"), ("LX", "CX"))
FRAMES = (
    ("raw", "Raw MRR (diagonal = home, off-diagonal = transfer)"),
    ("home_norm", "MRR divided by the target's home MRR"),
    ("ablated_norm", "MRR divided by the target's innervation-ablated home MRR"),
    ("base_norm", "(MRR - baseline) / (home - baseline)"),
    ("geo", "Geometry-ablated MRR (diagonal = ablated_geo, off-diagonal = transfer_geo)"),
    ("oov", "Fraction of masked target nodes whose cell type the expert reads as out-of-vocabulary"),
)
CAVEATS = (
    "Off-diagonal cells show region j's geometry z-scored inside j's own frame, its "
    "neurotransmitters, and its super class, class, side, flow and cell type remapped by name "
    "into expert i's vocabularies. They do not show innervation, which is uniform, and they read "
    "cell types that i never trained on as out-of-vocabulary with zero sibling features. The "
    "`transfer_geo` cells zero real within-region position along with the frame mismatch, so "
    "they are a sensitivity check, not a correction. The F1-val control is not inductive: region "
    "j's F1-val neurons can sit in region i's F1 training graph."
)


def _cells(df: pd.DataFrame) -> pd.DataFrame:
    """Mean MRR per (expert_region, region, variant) over seeds, then instances."""
    df = df[df["degree_setting"] == "degree_given"]
    df = df.assign(baseline=df["variant"] == "baseline")
    keys = ["expert_region", "region", "variant"]
    seeds = df.groupby(["expert_region", "baseline"])["seed"].nunique().rename("seeds")
    per_instance = (df.groupby(keys + ["baseline", "mask_instance"])["mrr"].agg(["mean", "size"])
                    .reset_index().join(seeds, on=["expert_region", "baseline"]))
    short = per_instance[per_instance["size"] != per_instance["seeds"]]
    if len(short):
        r = short.iloc[0]
        raise ValueError(f"{r['expert_region']}->{r['region']} {r['variant']} instance "
                         f"{r['mask_instance']} has {r['size']} seeds, expected {r['seeds']}")
    cells = per_instance.groupby(keys)["mean"].agg(["mean", "size"])
    cells.columns = ["mrr", "n_instances"]
    cells["oov"] = df.groupby(keys)["masked_oov_frac"].mean()
    instances = per_instance.groupby(keys)["mask_instance"].agg(frozenset)
    home = {j: s for (i, j, v), s in instances.items() if i == j and v == "home"}
    for (i, j, variant), got in instances.items():
        if j not in home:
            raise ValueError(f"no home cell for target {j}")
        if got != home[j]:
            raise ValueError(f"{i}->{j} {variant} scores a different instance set than {j} home")
    return cells.reset_index()


def compute(df: pd.DataFrame, split: str) -> dict[str, pd.DataFrame]:
    """Return the matrices (rows = expert, cols = target), the baselines and the cell table."""
    missing = [c for c in NEEDED if c not in df.columns]
    if missing:
        raise ValueError(f"transfer CSV lacks columns {missing}")
    if set(df["split"]) != {split} or set(df["dataset"]) != {SPLITS[split]}:
        raise ValueError(f"{split}.csv holds rows of split {sorted(set(df['split']))}")
    cells = _cells(df)
    present = set(df["expert_region"]) | set(df["region"])
    if present - set(TRAINING_ORDER):
        raise ValueError(f"unknown regions {sorted(present - set(TRAINING_ORDER))}")
    order = [r for r in TRAINING_ORDER if r in present]
    indexed = cells.set_index(["expert_region", "region", "variant"])
    lookup = indexed["mrr"]

    def matrix(diag: str, off: str, column: pd.Series = lookup) -> pd.DataFrame:
        return pd.DataFrame(
            [[column.get((i, j, diag if i == j else off), np.nan) for j in order] for i in order],
            index=order, columns=order, dtype=float)

    def diagonal(variant: str) -> pd.Series:
        return pd.Series([lookup.get((j, j, variant), np.nan) for j in order], index=order)

    raw, geo = matrix("home", "transfer"), matrix("ablated_geo", "transfer_geo")
    home, ablated, base = diagonal("home"), diagonal("ablated"), diagonal("baseline")
    if home.isna().any() or base.isna().any():
        raise ValueError("every target needs a home cell and a baseline cell")
    gap = (home - base).where(home > base)
    names = df[df["variant"] == "baseline"].drop_duplicates("region").set_index("region")["model"]
    return {
        "raw": raw,
        "home_norm": raw.div(home.where(home != 0)),
        "ablated_norm": raw.div(ablated.where(ablated != 0)),
        "base_norm": raw.sub(base, axis=1).div(gap),
        "geo": geo,
        "oov": matrix("home", "transfer", indexed["oov"]),
        "baselines": pd.DataFrame({"model": names.reindex(order), "mrr": base}),
        "cells": cells,
    }


def _table(frame: pd.DataFrame, raw: pd.DataFrame) -> str:
    lines = ["| expert \\ target | " + " | ".join(frame.columns) + " |",
             "|---" * (len(frame.columns) + 1) + "|"]
    for i in frame.index:
        row = [f"{frame.loc[i, j]:.3f}" if pd.notna(frame.loc[i, j])
               else ("undefined" if pd.notna(raw.loc[i, j]) else "not scored")
               for j in frame.columns]
        lines.append(f"| {i} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def _svg(frame: pd.DataFrame) -> str:
    size, left, top = 56, 50, 40
    vals = frame.to_numpy(dtype=float)
    lo, hi = (np.nanmin(vals), np.nanmax(vals)) if np.isfinite(vals).any() else (0.0, 1.0)
    span = (hi - lo) or 1.0
    n = len(frame)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{left + size * n + 20}" '
           f'height="{top + size * n + 60}" font-family="sans-serif" font-size="12">',
           '<defs><linearGradient id="scale"><stop offset="0" stop-color="rgb(235,242,250)"/>'
           '<stop offset="1" stop-color="rgb(20,70,140)"/></linearGradient></defs>',
           f'<text x="{left}" y="14">target region</text>',
           f'<text x="4" y="{top - 8}">expert</text>']
    for k, name in enumerate(frame.columns):
        out.append(f'<text x="{left + k * size + size / 2}" y="{top - 8}" '
                   f'text-anchor="middle">{name}</text>')
    for r, i in enumerate(frame.index):
        y = top + r * size
        out.append(f'<text x="{left - 6}" y="{y + size / 2 + 4}" text-anchor="end">{i}</text>')
        for k, j in enumerate(frame.columns):
            x, v = left + k * size, frame.loc[i, j]
            if pd.isna(v):
                fill, ink, label = "#bdbdbd", "#000", ""
            else:
                t = (v - lo) / span
                fill = f"rgb({round(235 - 215 * t)},{round(242 - 172 * t)},{round(250 - 110 * t)})"
                ink, label = ("#fff" if t > 0.55 else "#000"), f"{v:.2f}"
            out.append(f'<rect x="{x}" y="{y}" width="{size}" height="{size}" fill="{fill}" stroke="#fff"/>')
            out.append(f'<text x="{x + size / 2}" y="{y + size / 2 + 4}" text-anchor="middle" '
                       f'fill="{ink}">{label}</text>')
    ly = top + n * size + 20
    out += [f'<rect x="{left}" y="{ly}" width="160" height="14" fill="url(#scale)"/>',
            f'<text x="{left}" y="{ly + 30}">{lo:.3f}</text>',
            f'<text x="{left + 160}" y="{ly + 30}" text-anchor="end">{hi:.3f}</text>',
            f'<rect x="{left + 190}" y="{ly}" width="14" height="14" fill="#bdbdbd"/>',
            f'<text x="{left + 210}" y="{ly + 12}">not scored</text>', "</svg>"]
    return "\n".join(out) + "\n"


def _first_look(raw: pd.DataFrame) -> str:
    lines = []
    for i, j in list(FIRST_LOOK) + [(i, "OL") for i in raw.index if i != "OL"]:
        scored = i in raw.index and j in raw.columns and pd.notna(raw.loc[i, j])
        lines.append(f"- {i} to {j}: " + (f"{raw.loc[i, j]:.3f}" if scored else "not scored"))
    return "\n".join(lines)


def render(split: str, result: dict[str, pd.DataFrame], svg_name: str) -> str:
    raw, base, cells = result["raw"], result["baselines"], result["cells"]
    parts = [f"## {split}", f"![raw MRR matrix, {split}]({svg_name})", "### First cells to inspect",
             _first_look(raw)]
    for key, title in FRAMES:
        parts += [f"### {title}", _table(result[key], raw)]
    parts += ["### Champion baselines",
              "| target | model | MRR |\n|---|---|---|\n" + "\n".join(
                  f"| {j} | {base.loc[j, 'model']} | {base.loc[j, 'mrr']:.3f} |" for j in base.index),
              "### Cells",
              "| expert | target | variant | n_instances | MRR |\n|---|---|---|---|---|\n" + "\n".join(
                  f"| {r.expert_region} | {r.region} | {r.variant} | {r.n_instances} | {r.mrr:.3f} | {r.oov:.3f} |"
                  for r in cells.itertuples())]
    return "\n\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in", dest="src", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        found = [s for s in SPLITS if (args.src / f"{s}.csv").is_file()]
        if not found:
            raise ValueError(f"no f2_dev.csv or f1_val.csv under {args.src}")
        if args.out.exists():
            raise ValueError(f"{args.out} already exists")
        sections, svgs = [], {}
        for split in found:
            result = compute(pd.read_csv(args.src / f"{split}.csv"), split)
            svg_name = f"{args.out.stem}-{split}.svg"
            svgs[svg_name] = _svg(result["raw"])
            sections.append(render(split, result, svg_name))
        intro = ("# Transfer matrix\n\nAhmadh Hassan\n\n"
                 "Each frozen region expert scores every other region's suite. The matrix is "
                 "exploratory and feeds no gate. Rows are the expert's region, "
                 "columns the target region. A cell is the degree-given MRR averaged over 3 seeds, "
                 f"then over mask instances.\n\n{CAVEATS}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        for name, text in svgs.items():
            (args.out.parent / name).write_text(text, encoding="utf-8")
        args.out.write_text(intro + "\n\n" + "\n\n".join(sections) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError) as exc:
        print(f"transfer_report failed: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {args.out} for {', '.join(found)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
