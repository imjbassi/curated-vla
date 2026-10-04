"""Idle-only labeling pass: like-for-like check of the idle detectors.

The first labeling pass marked "idle" only when the robot never moved, so it could
not test the detectors' definition (> 3 s still at the start or end). This draws a
fresh blind sample, half flagged by an idle detector and half not, from the
sources where idle fires, excluding episodes already labeled.

Usage:
    python -m curation.sample_idle --scores AUDIT/scores.parquet --exclude LABELS/sample_key.parquet --out IDLE_DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from curation.sample_for_labeling import clip_episode, has_video

SEED = 2000
IDLE_S = 3.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=40)
    args = parser.parse_args()

    s = pd.read_parquet(args.scores)
    s = s[has_video(s) & ~s["key"].isin(pd.read_parquet(args.exclude)["key"])]
    s["flag_idle_start"] = s["idle_start_s"] > IDLE_S
    s["flag_idle_end"] = s["idle_end_s"] > IDLE_S
    pool = s[s["source"].isin(s.loc[s["flag_idle_start"] | s["flag_idle_end"], "source"].unique())]

    k = args.n // 4
    parts = [
        pool[pool["flag_idle_start"]].sample(k, random_state=SEED),
        pool[pool["flag_idle_end"] & ~pool["flag_idle_start"]].sample(k, random_state=SEED),
    ]
    taken = pd.concat(parts)["key"]
    clean = pool[~pool["flag_idle_start"] & ~pool["flag_idle_end"] & ~pool["key"].isin(taken)]
    parts.append(clean.sample(args.n - 2 * k, random_state=SEED))
    sample = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    sample["label_id"] = [f"idle{i:02d}" for i in range(len(sample))]

    (args.out / "clips").mkdir(parents=True, exist_ok=True)
    sample.to_parquet(args.out / "sample_key.parquet")
    manifest = []
    for _, row in sample.iterrows():
        clip_episode(row, args.out / "clips")
        manifest.append(dict(label_id=row["label_id"], task=row["task"], duration_s=round(row["duration_s"], 1),
                             fps=row["fps"]))
        print(row["label_id"], row["key"], flush=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(sample.groupby("source")[["flag_idle_start", "flag_idle_end"]].sum())


if __name__ == "__main__":
    main()
