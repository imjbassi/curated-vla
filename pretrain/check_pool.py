"""Sanity checks for PoolDataset: shapes, normalization, padding, frames, throughput.

Usage: python -m pretrain.check_pool INDEX [--workers 8] [--batches 30] [--sheet OUT.png]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pretrain.pool import PoolDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batches", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sheet", type=Path)
    parser.add_argument("--video-root", type=Path, help="transcoded videos (pretrain/transcode.py)")
    args = parser.parse_args()

    ds = PoolDataset(args.index, video_root=args.video_root)
    print(f"{len(ds.episodes)} episodes, {len(ds)} samples")

    # 1. normalization: per-source action/state should be ~N(0, 1) on live dims
    for s, g in ds.episodes.groupby(ds.episodes["sub"].map(lambda k: ds.subsets[k].source)):
        rows = np.concatenate([np.arange(r.start, r.start + r.length) for r in g.itertuples()][:200])
        a = ds.actions[rows].astype(np.float32)
        dim = ds.subsets[g["sub"].iloc[0]].action_dim
        print(f"  {s:22s} action mean {np.abs(a[:, :dim].mean(0)).max():.2f} (max |mean|), "
              f"std {a[:, :dim].std(0).min():.2f}-{a[:, :dim].std(0).max():.2f}, pad dims zero: {np.all(a[:, dim:] == 0)}")

    # 2. one sample near an episode end: padding mask
    e = ds.episodes.iloc[0]
    item = ds[int(e["length"]) - 5]
    print("sample keys:", {k: (tuple(v.shape) if torch.is_tensor(v) else v) for k, v in item.items()})
    print(f"  near end: action_is_pad sum = {int(item['action_is_pad'].sum())} (expect 45)")
    img = item["observation.images.camera1"]
    print(f"  camera1 range [{img.min():.2f}, {img.max():.2f}]")

    # 3. contact sheet of camera1/camera2 from random samples across sources
    if args.sheet:
        from PIL import Image

        rng = np.random.default_rng(0)
        tiles = []
        for sid in sorted(ds.episodes["sub"].unique())[:8]:
            rows = ds.episodes.index[ds.episodes["sub"] == sid]
            ep = int(rng.choice(rows))
            it = ds[int(ds.cum[ep] + ds.episodes.iloc[ep]["length"] // 2)]
            pair = torch.cat([it["observation.images.camera1"], it["observation.images.camera2"]], dim=2)
            tiles.append((pair.permute(1, 2, 0).numpy()).astype(np.uint8))
        Image.fromarray(np.concatenate(tiles, axis=0)).save(args.sheet)
        print(f"  wrote {args.sheet}")

    # 4. throughput
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        persistent_workers=True, prefetch_factor=4)
    it = iter(loader)
    next(it)
    t = time.time()
    for _ in range(args.batches):
        next(it)
    dt = time.time() - t
    print(f"throughput: {args.batches * args.batch_size / dt:.0f} samples/s with {args.workers} workers "
          f"(training needs ~54)")


if __name__ == "__main__":
    main()
