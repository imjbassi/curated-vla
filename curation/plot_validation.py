"""Figure for the README: do detector flags predict hand-labeled failure?

Left: failure rate among episodes each detector flagged vs did not flag.
Right: hand-labeled failure rate by source.

Usage:
    python -m curation.plot_validation LABEL_DIR --out docs/img/validation.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import fisher_exact

from curation.validate import load_labels

# Reference palette (dataviz skill): surface, inks, series-1 blue, muted gray.
SURFACE, INK, INK2, MUTED, GRID, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#2a78d6"
DETECTORS = {"flag_idle_start": "Idle at start", "flag_idle_end": "Idle at end", "flag_spike": "Spike (glitch)",
             "flag_truncation": "Truncation", "flag_flailing": "Flailing"}


def load(label_dir: Path) -> pd.DataFrame:
    df = pd.read_parquet(label_dir / "sample_key.parquet").merge(load_labels(label_dir), on="label_id")
    df = df[[(label_dir / "clips" / f"{i}.mp4").stat().st_size >= 1000 for i in df["label_id"]]]
    df = df[df["completed"] != "unclear"].copy()
    df["flag_idle_start"] = df["idle_start_s"] > 3.0
    df["flag_idle_end"] = df["idle_end_s"] > 3.0
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    df = load(args.label_dir)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.patch.set_facecolor(SURFACE)

    rows = []
    for col, name in DETECTORS.items():
        a, b = df[df[col]]["lab_failed"], df[~df[col]]["lab_failed"]
        p = fisher_exact([[a.sum(), (~a).sum()], [b.sum(), (~b).sum()]])[1]
        rows.append((name, a.mean(), b.mean(), len(a), p))
    for i, (name, fr, base, n, p) in enumerate(reversed(rows)):
        ax.plot([base, fr], [i, i], color=GRID, lw=2, zorder=1)
        ax.scatter(base, i, s=70, color=MUTED, zorder=2, edgecolor=SURFACE, linewidth=2)
        ax.scatter(fr, i, s=70, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=2)
        sig = f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}"
        ax.text(1.02, i, f"{fr:.0%} of {n} flagged  ·  {sig}", va="center", fontsize=9,
                color=INK if p < 0.05 else INK2, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
    ax.set_xlim(0, 0.8)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Hand-labeled failure rate")
    ax.set_title("Failure rate when a detector fires vs. when it doesn't", loc="left", fontsize=11, color=INK, pad=24)
    ax.scatter([], [], s=50, color=BLUE, label="flagged")
    ax.scatter([], [], s=50, color=MUTED, label="not flagged")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9, handletextpad=0.3)

    by_src = df.groupby("source")["lab_failed"].agg(["mean", "size"]).sort_values("mean")
    y = range(len(by_src))
    ax2.barh(list(y), by_src["mean"], color=BLUE, height=0.55)
    for i, (m, n) in enumerate(zip(by_src["mean"], by_src["size"])):
        ax2.text(m + 0.015, i, f"{m:.0%}  (n={n})", va="center", fontsize=9, color=INK2)
    ax2.set_yticks(list(y), by_src.index)
    ax2.set_xlim(0, 0.8)
    ax2.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax2.set_xlabel("Hand-labeled failure rate")
    ax2.set_title("Failure rate by source", loc="left", fontsize=11, color=INK, pad=24)

    for a in (ax, ax2):
        a.set_facecolor(SURFACE)
        a.grid(axis="x", color=GRID, lw=0.8)
        a.set_axisbelow(True)
        for s in ("top", "right", "left"):
            a.spines[s].set_visible(False)
        a.spines["bottom"].set_color("#c3c2b7")
        a.tick_params(length=0)
    fig.text(0.01, 0.01, f"{len(df)} hand-labeled episodes (8 sources); 'unclear' excluded. "
             "Detector flags use initial thresholds.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 0.97, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
