"""Fill key frames for the labeled sample from its local clips.

The labeling clips already hold each sampled episode's main camera, so the VLM
judge can be validated on them without waiting for the full frame extraction.

Usage:
    python -m curation.frames_from_clips LABEL_DIR --out FRAMES_DIR
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from curation.frames import POSITIONS, _duration, _grab


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    key = pd.read_parquet(args.label_dir / "sample_key.parquet")
    n = 0
    for r in key.itertuples():
        d = args.out / r.source / (r.subset or "_")
        d.mkdir(parents=True, exist_ok=True)
        targets = {name: d / f"{r.episode_index:06d}_{name}.jpg" for name in POSITIONS}
        if all(p.exists() for p in targets.values()):
            continue
        clip = str(args.label_dir / "clips" / f"{r.label_id}.mp4")
        try:
            dur = _duration(clip)
        except (ValueError, IndexError):
            print(f"skip {r.label_id} ({r.key}): unreadable clip")
            continue
        for name, frac in POSITIONS.items():
            _grab(clip, min(frac * dur, dur - 0.1), targets[name])
        n += 1
    print(f"filled {n} episodes from clips")


if __name__ == "__main__":
    main()
