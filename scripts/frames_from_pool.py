"""Export key frames of every pool episode exactly as the policy sees them.

Goes through PoolDataset (main camera, transcoded 256 px videos, aspect-preserving pad), so the
camera choice matches the index (curation.frames re-picks the camera from the Hub metadata and
can disagree for sub-datasets whose main camera was overridden). Output:

    OUT/<source>/<subset or _>/<episode:06d>_{first,mid,last}.jpg   (default)
    OUT/<source>/<subset or _>/<episode:06d>_f0..f{N-1}.jpg          (--n-frames N, evenly spaced)

Existing files are skipped, so the export is resumable.

Usage:
    python scripts/frames_from_pool.py ~/cvla/pool/index_all.pkl --video-root ~/cvla/pool_video --out ~/cvla/frames
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from pretrain.pool import PoolDataset


class KeyFrames(Dataset):
    """Wraps PoolDataset: item j -> (frame image, output path) for the j-th requested key frame."""

    def __init__(self, pool: PoolDataset, out: Path, n_frames: int):
        self.pool, self.items = pool, []
        names = ["first", "mid", "last"] if n_frames == 3 else [f"f{k}" for k in range(n_frames)]
        fracs = np.linspace(0.0, 1.0, n_frames)
        for e, row in enumerate(pool.episodes.itertuples()):
            sub = pool.subsets[row.sub]
            d = out / sub.source / (sub.subset or "_")
            n = int(row.length)
            for name, frac in zip(names, fracs):
                path = d / f"{int(row.episode_index):06d}_{name}.jpg"
                if not path.exists():
                    self.items.append((int(pool.cum[e]) + min(int(round(frac * (n - 1))), n - 1), str(path)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, j: int):
        i, path = self.items[j]
        return self.pool[i]["observation.images.camera1"], path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("index", type=Path)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-frames", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()
    pool = PoolDataset(args.index, image_size=args.image_size, video_root=args.video_root)
    ds = KeyFrames(pool, args.out, args.n_frames)
    print(f"{len(pool.episodes)} episodes, {len(ds)} frames to export", flush=True)
    for d in {Path(p).parent for _, p in ds.items}:
        d.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(ds, batch_size=32, num_workers=args.workers, collate_fn=lambda b: b)
    t0, done = time.time(), 0
    for batch in loader:
        for img, path in batch:
            Image.fromarray(img.permute(1, 2, 0).numpy()).save(path, quality=90)
        done += len(batch)
        if done % 3200 == 0:
            print(f"{done}/{len(ds)} ({done / (time.time() - t0):.0f} frames/s)", flush=True)
    print(f"done: {done} frames in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
