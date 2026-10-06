"""Pretrain SmolVLA (VLM-initialized) on the mixed public pool.

Thin loop around LeRobot's SmolVLA policy, because lerobot-train cannot train on
several datasets (see docs/phase2_design.md). Checkpoints are saved in LeRobot's
`pretrained_model` format, so post-training uses stock lerobot-train:

    lerobot-train --policy.path=RUN/checkpoints/last/pretrained_model --dataset.repo_id=lerobot/libero ...

Pool data is already normalized per sub-dataset, so the saved normalizer stats are
mean 0 / std 1 (identity); lerobot-train replaces them with the post-training
dataset's stats.

Usage:
    python -m pretrain.train --index POOL.pkl --video-root VIDEOS --out RUN \
        --samples 5000000 [--keys KEYS.parquet] [--batch-size 32]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from pretrain.pool import CHUNK, MAX_DIM, EpisodeBlockSampler, PoolDataset

CAMERAS = ["observation.images.camera1", "observation.images.camera2"]


def make_policy(image_size: int, device: str) -> tuple[SmolVLAPolicy, object, object]:
    cfg = SmolVLAConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(MAX_DIM,)),
            **{k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, image_size, image_size)) for k in CAMERAS},
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(MAX_DIM,))},
        normalization_mapping={"VISUAL": NormalizationMode.IDENTITY, "STATE": NormalizationMode.MEAN_STD,
                               "ACTION": NormalizationMode.MEAN_STD},
        chunk_size=CHUNK,
        n_action_steps=CHUNK,
        load_vlm_weights=True,
        device=device,
        push_to_hub=False,
    )
    identity = {"mean": torch.zeros(MAX_DIM), "std": torch.ones(MAX_DIM)}
    stats = {"observation.state": identity, "action": identity}
    pre, post = make_smolvla_pre_post_processors(cfg, dataset_stats=stats)
    return SmolVLAPolicy(cfg).to(device), pre, post


def save(out: Path, step: int, policy, pre, post, optimizer, scheduler, args) -> None:
    ckpt = out / "checkpoints" / f"{step:07d}"
    model_dir = ckpt / "pretrained_model"
    policy.save_pretrained(model_dir)
    pre.save_pretrained(model_dir)
    post.save_pretrained(model_dir)
    torch.save({"step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
               ckpt / "training_state.pt")
    (ckpt / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    last = out / "checkpoints" / "last"
    if last.is_symlink() or last.exists():
        last.unlink()
    last.symlink_to(ckpt.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--keys", type=Path, help="parquet/csv with a 'key' column: episodes to train on")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5_000_000, help="training budget in samples")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8, help="8 was fastest on 12 cores (pretrain/bench.py)")
    parser.add_argument("--block", type=int, default=4,
                        help="frames per episode visit; 4 keeps the GPU fed on the full pool (pretrain/bench.py)")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=5000, help="steps")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    keys = None
    if args.keys:
        k = pd.read_parquet(args.keys) if args.keys.suffix == ".parquet" else pd.read_csv(args.keys)
        keys = set(k["key"])
    ds = PoolDataset(args.index, keys=keys, image_size=args.image_size, video_root=args.video_root)
    steps = math.ceil(args.samples / args.batch_size)
    assert args.batch_size % args.block == 0, "batch size must be a multiple of --block"
    sampler = EpisodeBlockSampler(ds, steps * args.batch_size, block=args.block, seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                        persistent_workers=True, prefetch_factor=4, pin_memory=True, drop_last=True)
    print(f"pool: {len(ds.episodes)} episodes, {len(ds)} samples; budget {args.samples} samples = {steps} steps",
          flush=True)

    policy, pre, post = make_policy(args.image_size, "cuda")
    policy.train()
    opt_cfg = policy.config.get_optimizer_preset()
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=opt_cfg.lr,
                                  betas=opt_cfg.betas, eps=opt_cfg.eps, weight_decay=opt_cfg.weight_decay)
    sched_cfg = policy.config.get_scheduler_preset()
    sched_cfg.num_decay_steps = steps  # cosine over the whole budget
    scheduler = sched_cfg.build(optimizer, steps)
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"trainable params {n_train / 1e6:.0f}M of {sum(p.numel() for p in policy.parameters()) / 1e6:.0f}M",
          flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    log = open(args.out / "train_log.jsonl", "a")
    t0, step, data_t = time.time(), 0, 0.0
    t_fetch = time.time()
    for batch in loader:
        data_t += time.time() - t_fetch
        batch = pre(batch)
        for k in CAMERAS:  # uint8 from the loader -> [0, 1] on the GPU
            batch[k] = batch[k].float() / 255.0
        loss, info = policy.forward(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), opt_cfg.grad_clip_norm)
        optimizer.step()
        scheduler.step()
        step += 1
        if step % args.log_every == 0 or step == 1:
            el = time.time() - t0
            rec = dict(step=step, samples=step * args.batch_size, loss=round(loss.item(), 4),
                       grad_norm=round(float(grad_norm), 3), lr=scheduler.get_last_lr()[0],
                       samples_per_s=round(step * args.batch_size / el, 1), data_wait_frac=round(data_t / el, 3),
                       mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec, flush=True)
        if step % args.save_every == 0 or step == steps:
            save(args.out, step, policy, pre, post, optimizer, scheduler, args)
        t_fetch = time.time()
    print(f"done: {step} steps in {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
