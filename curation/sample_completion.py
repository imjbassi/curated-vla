"""Fresh completion-only labeling pass to confirm the failure detector.

The 8-frame judge configuration was chosen on the first 187 labels, so its AUROC
there is optimistic. This draws new episodes (equal per source, uniformly at
random within each, excluding every previously labeled episode) for a held-out
check. Pre-registered config: Qwen3-VL-8B, 4-bit, 8 frames.

Usage:
    python -m curation.sample_completion --scores AUDIT/scores.parquet --exclude A.parquet B.parquet --out DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from curation.sample_for_labeling import clip_episode, has_video

SEED = 3000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    s = pd.read_parquet(args.scores)
    seen = set().union(*(set(pd.read_parquet(p)["key"]) for p in args.exclude))
    s = s[has_video(s) & ~s["key"].isin(seen)]
    sources = sorted(s["source"].unique())
    sizes = [args.n // len(sources) + (i < args.n % len(sources)) for i in range(len(sources))]
    sample = pd.concat([s[s["source"] == src].sample(k, random_state=SEED) for src, k in zip(sources, sizes)])
    sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)
    sample["label_id"] = [f"c{i:03d}" for i in range(len(sample))]

    (args.out / "clips").mkdir(parents=True, exist_ok=True)
    sample.to_parquet(args.out / "sample_key.parquet")
    manifest = []
    for _, row in sample.iterrows():
        try:
            clip_episode(row, args.out / "clips")
        except Exception as e:
            print(f"clip failed {row['key']}: {type(e).__name__}", flush=True)
        manifest.append(dict(label_id=row["label_id"], task=row["task"], duration_s=round(row["duration_s"], 1),
                             fps=row["fps"]))
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(sample["source"].value_counts().to_dict())


if __name__ == "__main__":
    main()
