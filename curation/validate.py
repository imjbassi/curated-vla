"""Validate detectors against hand labels.

For each detector/label pair reports precision, recall, F1 and Cohen's kappa on
the labeled sample, then the same within episode-length tertiles (a detector
that only works across lengths is measuring length, not quality).

Precision/recall are also reported reweighted to the population: the sample
over-represents flagged episodes (half flagged per source), so each labeled
episode is weighted by (population share of its stratum) / (sample share).

Usage:
    python -m curation.validate LABEL_DIR [--vlm AUDIT/vlm.parquet] [--scores AUDIT/scores.parquet]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

# detector flag column -> hand-label problem
PAIRS = {
    "flag_truncation": "truncated",
    "flag_idle_start": "idle_start",
    "flag_idle_end": "idle_end",
    "flag_flailing": "flailing",
    "flag_spike": "glitch",
}


def load_labels(label_dir: Path) -> pd.DataFrame:
    latest = {}
    for line in (label_dir / "labels.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            latest[row["label_id"]] = row
    lab = pd.DataFrame(latest.values())
    for p in set(PAIRS.values()) | {"wrong_task"}:
        lab[f"lab_{p}"] = lab["problems"].apply(lambda ps, p=p: p in ps)
    lab["lab_failed"] = lab["completed"] == "no"
    lab["lab_bad"] = lab["lab_failed"] | lab[[f"lab_{p}" for p in PAIRS.values()]].any(axis=1)
    return lab


def weighted_pr(y: np.ndarray, f: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    tp = (w * (y & f)).sum()
    prec = tp / max((w * f).sum(), 1e-12)
    rec = tp / max((w * y).sum(), 1e-12)
    return prec, rec


def evaluate(df: pd.DataFrame, flag: str, label: str) -> dict:
    y, f, w = df[label].to_numpy(bool), df[flag].to_numpy(bool), df["w"].to_numpy()
    tp, fp, fn = (y & f).sum(), (~y & f).sum(), (y & ~f).sum()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    wp, wr = weighted_pr(y, f, w)
    kappa = cohen_kappa_score(y, f) if y.any() and f.any() and not (y.all() or f.all()) else np.nan
    return dict(n=len(df), label_pos=int(y.sum()), flag_pos=int(f.sum()), precision=prec, recall=rec,
                f1=2 * prec * rec / max(prec + rec, 1e-12), kappa=kappa, pop_precision=wp, pop_recall=wr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label_dir", type=Path)
    parser.add_argument("--scores", type=Path, help="full audit scores, for population reweighting")
    parser.add_argument("--vlm", type=Path)
    args = parser.parse_args()

    key = pd.read_parquet(args.label_dir / "sample_key.parquet")
    df = key.merge(load_labels(args.label_dir), on="label_id")
    df["flag_idle_start"] = df["idle_start_s"] > 3.0
    df["flag_idle_end"] = df["idle_end_s"] > 3.0

    # population weights: stratum = (source, flag_any)
    df["w"] = 1.0
    if args.scores:
        pop = pd.read_parquet(args.scores, columns=["source", "flag_any"])
        pop_share = pop.groupby(["source", "flag_any"]).size() / pop.groupby("source").size()
        samp_share = df.groupby(["source", "flag_any"]).size() / df.groupby("source").size()
        ratio = (pop_share / samp_share).rename("w")
        df = df.drop(columns="w").join(ratio, on=["source", "flag_any"])

    rows = [dict(detector=f, label=l, **evaluate(df, f, f"lab_{l}")) for f, l in PAIRS.items()]
    rows.append(dict(detector="flag_any", label="any problem or failed", **evaluate(df, "flag_any", "lab_bad")))
    print("## Detector vs hand label (labeled sample)\n")
    print(pd.DataFrame(rows).round(3).to_markdown(index=False))

    print("\n## Within length tertiles (F1)\n")
    df["len_bin"] = pd.qcut(df["length"].rank(method="first"), 3, labels=["short", "mid", "long"])
    tert = []
    for f, l in PAIRS.items():
        r = {"detector": f}
        for b, g in df.groupby("len_bin", observed=True):
            r[str(b)] = evaluate(g, f, f"lab_{l}")["f1"]
        tert.append(r)
    print(pd.DataFrame(tert).round(3).to_markdown(index=False))

    if args.vlm:
        v = df.merge(pd.read_parquet(args.vlm), on="key")
        v = v[v["completed"] != "unclear"]
        auc = roc_auc_score(v["completed"] == "yes", v["vlm_p_yes"])
        print(f"\n## VLM judge\n\nAUROC vs hand-labeled completion: {auc:.3f} (n={len(v)})")
        for t in (0.3, 0.5, 0.7):
            r = evaluate(v.assign(flag_vlm=v["vlm_p_yes"] < t), "flag_vlm", "lab_failed")
            print(f"- flag if p_yes < {t}: precision {r['precision']:.2f}, recall {r['recall']:.2f}, kappa {r['kappa']:.2f}")

    print("\n## Hand-label base rates (sample)\n")
    print(df[[c for c in df.columns if c.startswith("lab_")]].mean().round(3).to_markdown())


if __name__ == "__main__":
    main()
