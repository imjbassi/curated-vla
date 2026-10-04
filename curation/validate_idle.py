"""Validate the idle detectors against the idle-only labeling pass.

Reports, for idle at the start and at the end: precision/recall/kappa of the
"> 3 s still" flag, AUROC of the raw idle-seconds score, and the threshold that
best matches the labels (with the caveat that it is tuned on these labels).

Usage:
    python -m curation.validate_idle IDLE_DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("idle_dir", type=Path)
    args = parser.parse_args()
    latest = {}
    for line in (args.idle_dir / "labels.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            latest[r["label_id"]] = r
    df = pd.read_parquet(args.idle_dir / "sample_key.parquet").merge(pd.DataFrame(latest.values()), on="label_id")
    print(f"{len(df)} labeled\n")
    rows = []
    for side in ("start", "end"):
        y = (df[f"idle_{side}"] == "yes").to_numpy()
        score = df[f"idle_{side}_s"].to_numpy()
        f = score > 3.0
        tp, fp, fn = (y & f).sum(), (~y & f).sum(), (y & ~f).sum()
        best = max(np.unique(score), key=lambda t: cohen_kappa_score(y, score > t) if 0 < (score > t).sum() < len(y) else -1)
        rows.append(dict(side=side, label_pos=int(y.sum()), flagged=int(f.sum()),
                         precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                         kappa=cohen_kappa_score(y, f), auroc=roc_auc_score(y, score) if 0 < y.sum() < len(y) else np.nan,
                         best_threshold_s=round(float(best), 2), kappa_at_best=cohen_kappa_score(y, score > best)))
    print(pd.DataFrame(rows).round(3).to_markdown(index=False))


if __name__ == "__main__":
    main()
