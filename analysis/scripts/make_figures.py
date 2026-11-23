"""Draw the paper's data figures from the CSVs in analysis/data.

Run: python analysis/scripts/make_figures.py [--data DIR] [--out DIR]
Figures go to --out (default analysis/figures); LaTeX tables to --out/tables.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANALYSIS = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
ap.add_argument("--data", type=Path, default=ANALYSIS / "data", help="directory with the figure CSVs")
ap.add_argument("--out", type=Path, default=ANALYSIS / "figures")
args = ap.parse_args()
DATA, FIGS = args.data, args.out

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": INK,
    "xtick.major.width": 0.6,
    "ytick.major.size": 0,
    "legend.frameon": False,
    "pdf.fonttype": 42,
})


def _rows(ax, labels):
    y = np.arange(len(labels))[::-1]
    ax.set_yticks(y, labels)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    return y


def fig_gate():
    g = pd.read_csv(DATA / "regional_gate.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.7), sharey=True,
                               gridspec_kw={"width_ratios": [1.15, 1]})
    y = _rows(a, g.region)
    a.scatter(g.f1val_model, y, s=22, marker="o", color=BLUE, label="Model, FlyWire validation", zorder=3)
    a.scatter(g.f2dev_model, y, s=22, marker="s", color=ORANGE, label="Model, BANC dev", zorder=3)
    a.scatter(g.f2conf_model, y, s=26, marker="D", color=AQUA, label="Model, BANC test", zorder=3)
    a.scatter(g.f2dev_champion, y, s=26, marker="^", facecolor="none", edgecolor=MUTED,
              linewidth=0.9, label="Tuned baseline, BANC dev", zorder=3)
    a.scatter(g.f2conf_match, y, s=26, marker="x", color=INK, linewidth=1.0,
              label="Matched-neuron lookup, BANC test", zorder=3)
    a.set_xlim(0.15, 0.95)
    a.set_xlabel("Mean reciprocal rank")
    a.set_title("a", loc="left", fontweight="bold", fontsize=9)
    a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=2, fontsize=7,
             handletextpad=0.3, columnspacing=0.8)

    _rows(b, g.region)
    b.hlines(y, g.ci_lo, g.ci_hi, color=BLUE, linewidth=1.6, zorder=2)
    b.scatter(g.rbar, y, s=24, color=BLUE, zorder=3, label="Mean log$_2$ ratio, 95% BCa interval")
    b.scatter(g["delta"], y, marker="|", s=90, color=INK, linewidth=1.2, zorder=4,
              label=r"Noise floor $\delta$")
    b.axvline(0, color=MUTED, linewidth=0.6)
    b.set_xlim(-0.05, 1.55)
    b.set_xlabel(r"log$_2$(MRR$_{\mathrm{model}}$ / MRR$_{\mathrm{baseline}}$), BANC test")
    b.set_title("b", loc="left", fontweight="bold", fontsize=9)
    b.legend(loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=1, fontsize=7, handletextpad=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_gate.pdf")
    plt.close(fig)


def fig_regime():
    r = pd.read_csv(DATA / "regime_split.csv")
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    y = _rows(ax, r.region)
    others = ["R01", "R05", "R10", "S05"]
    lo, hi = r[others].min(axis=1), r[others].max(axis=1)
    ax.hlines(y, lo, hi, color=BLUE, linewidth=4, alpha=0.35, zorder=2,
              label="Range over R01, R05, R10, S05")
    ax.scatter(r[others].mean(axis=1), y, s=16, color=BLUE, zorder=3,
               label="Mean over R01, R05, R10, S05")
    ax.scatter(r["T"], y, s=26, marker="D", color=ORANGE, zorder=4, label="Typed (T)")
    ax.set_xlim(0.28, 0.72)
    ax.set_xlabel("Model MRR, BANC test\n(four instances per regime)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=1, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_regime.pdf")
    plt.close(fig)


def fig_transfer():
    path = DATA / "transfer_ratio.csv"
    if not path.exists():
        return
    m = pd.read_csv(path, index_col=0)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(m.values, cmap="Blues", vmin=0.2, vmax=1.0)
    ax.set_xticks(range(m.shape[1]), m.columns, rotation=90)
    ax.set_yticks(range(m.shape[0]), m.index)
    ax.set_xlabel("Target region (BANC dev suite)")
    ax.set_ylabel("Training region of the model")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.2,
                    color="white" if v > 0.7 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Visiting-model MRR / home-model MRR", fontsize=7)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_transfer.pdf")
    plt.close(fig)


def matrix_table(csv, out, fmt):
    m = pd.read_csv(DATA / csv, index_col=0)
    lines = [r"\begin{tabular}{@{}l" + "c" * m.shape[1] + "@{}}", r"\toprule",
             "Model & " + " & ".join(m.columns) + r" \\", r"\midrule"]
    for name, row in m.iterrows():
        cells = [rf"\textbf{{{fmt(v)}}}" if col == name else fmt(v) for col, v in row.items()]
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (FIGS / "tables").mkdir(parents=True, exist_ok=True)
    (FIGS / "tables" / out).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_gate()
    fig_regime()
    fig_transfer()
    matrix_table("transfer_raw_mrr.csv", "transfer_raw.tex", lambda v: f"{v:.3f}".lstrip("0"))
    matrix_table("transfer_masked_oov.csv", "transfer_oov.tex", lambda v: f"{v:.2f}")
