"""Training-throughput benchmark: GPU ceiling (fixed batch) vs full pipeline (DataLoader).

Usage:
    python -m pretrain.bench --index POOL.pkl --video-root VIDEOS [--workers 6 8 10] [--steps 60]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pretrain.pool import PoolDataset
from pretrain.train import CAMERAS, make_policy


def step_fn(policy, pre, optimizer, bf16: bool = False):
    def run(batch):
        batch = pre(batch)
        for k in CAMERAS:
            batch[k] = batch[k].float() / 255.0
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            loss, _ = policy.forward(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        optimizer.step()
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--workers", type=int, nargs="+", default=[6, 8, 10])
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--main-threads", type=int, default=0, help="torch threads in the main process (0 = default)")
    parser.add_argument("--bf16", action="store_true", help="bf16 autocast (no GradScaler needed)")
    args = parser.parse_args()
    if args.main_threads:
        torch.set_num_threads(args.main_threads)

    ds = PoolDataset(args.index, video_root=args.video_root)
    policy, pre, _ = make_policy(256, "cuda")
    policy.train()
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-5)
    run = step_fn(policy, pre, optimizer, args.bf16)

    # GPU ceiling: one fixed batch, no loading
    fixed = next(iter(DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4)))
    for _ in range(5):
        run({k: (v.clone() if torch.is_tensor(v) else list(v)) for k, v in fixed.items()})
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(args.steps):
        run({k: (v.clone() if torch.is_tensor(v) else list(v)) for k, v in fixed.items()})
    torch.cuda.synchronize()
    print(f"GPU ceiling (fixed batch): {args.steps * args.batch_size / (time.time() - t):.1f} samples/s", flush=True)

    for w in args.workers:
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=w, persistent_workers=True,
                            prefetch_factor=4, pin_memory=True)
        it = iter(loader)
        for _ in range(8):  # warm up workers and decoders
            run(next(it))
        torch.cuda.synchronize()
        t, wait = time.time(), 0.0
        for _ in range(args.steps):
            t1 = time.time()
            b = next(it)
            wait += time.time() - t1
            run(b)
        torch.cuda.synchronize()
        el = time.time() - t
        print(f"workers={w:2d}: {args.steps * args.batch_size / el:.1f} samples/s, waiting on data {wait / el:.0%}",
              flush=True)
        del it, loader


if __name__ == "__main__":
    main()
