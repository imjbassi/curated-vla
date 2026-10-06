"""Pretraining pool: per-source-normalized episodes from all Phase 1 datasets.

LeRobot 0.6.1 cannot train on several datasets at once, and merging them pools
normalization statistics across robots (see docs/phase2_design.md). This module
builds one index over every episode and serves SmolVLA training samples:

  observation.state                       (32,)  normalized per sub-dataset, zero-padded
  action                                  (50, 32) next 50 actions, same normalization
  action_is_pad                           (50,)  True past the episode end
  observation.images.camera1              (3, H, W) uint8 main third-person view (divide by 255 on GPU)
  observation.images.camera2              (3, H, W) uint8 wrist view, zeros when absent
  observation.images.camera2_padding_mask ()     False when the wrist view is absent
  task                                    str

Build once (reads parquet + metadata only, ~minutes):
    python -m pretrain.pool build --out ~/cvla/pool/index.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset, Sampler

from curation.cameras import _community_mappings, main_camera
from curation.datasets import SOURCES
from curation.load import find_datasets, load_dataset
from curation.sample_for_labeling import NO_VIDEO_SUBSETS

MAX_DIM = 32
CHUNK = 50
MIN_LEN = 10
WRIST = re.compile(r"wrist|hand|gripper|eye", re.I)
DEPTH = re.compile(r"depth", re.I)


@dataclass
class SubsetInfo:
    source: str
    subset: str
    root: str  # local snapshot dir of the sub-dataset
    fps: float
    v2: bool
    video_path: str  # info.json template
    chunks_size: int
    main_key: str
    wrist_key: str | None
    action_dim: int
    state_dim: int
    # v3 only: episode_index -> (chunk, file, from_ts) per camera key
    v3_video: dict = field(default_factory=dict)


def _wrist_camera(info: dict, repo_id: str, subset: str, main_key: str) -> str | None:
    keys = [k for k, v in info["features"].items() if v["dtype"] == "video" and k != main_key]
    original = _community_mappings(repo_id).get(subset, {}) if subset else {}
    for k in keys:
        desc = f"{k.split('.')[-1]} {original.get(k.split('.')[-1], '')}"
        if WRIST.search(desc) and not DEPTH.search(desc):
            return k
    return None


def _norm_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0  # constant dims (e.g. roboturk's zero state) map to 0
    return mean.astype(np.float32), std.astype(np.float32)


def build(out: Path, names: list[str] | None = None) -> None:
    subsets: list[SubsetInfo] = []
    episodes = []  # dicts with offsets into the big arrays
    actions, states = [], []
    offset = 0
    for src in SOURCES:
        if names and src.name not in names:
            continue
        root = Path(snapshot_download(src.repo_id, repo_type="dataset", allow_patterns=["*meta/*", "meta/*"]))
        for ds_root in find_datasets(root):
            subset = str(ds_root.relative_to(root)) if ds_root != root else ""
            if (src.name, subset) in NO_VIDEO_SUBSETS:
                continue
            info = json.loads((ds_root / "meta" / "info.json").read_text())
            eps = [e for e in load_dataset(ds_root, src.name, subset) if e.length >= MIN_LEN]
            if not eps:
                continue
            main_key = main_camera(info, src.repo_id, subset)
            wrist_key = _wrist_camera(info, src.repo_id, subset, main_key)
            sub = SubsetInfo(src.name, subset, str(ds_root), float(info["fps"]),
                             info["codebase_version"].startswith("v2"), info["video_path"],
                             int(info.get("chunks_size", 1000)), main_key, wrist_key,
                             eps[0].action.shape[1], 0 if eps[0].state is None else eps[0].state.shape[1])
            if not sub.v2:
                meta = pd.concat([pd.read_parquet(f) for f in sorted((ds_root / "meta" / "episodes").rglob("*.parquet"))])
                for k in [main_key, wrist_key]:
                    if k is None:
                        continue
                    sub.v3_video[k] = {
                        int(r["episode_index"]): (int(r[f"videos/{k}/chunk_index"]), int(r[f"videos/{k}/file_index"]),
                                                  float(r[f"videos/{k}/from_timestamp"]))
                        for _, r in meta.iterrows()}
            a_all = np.concatenate([e.action for e in eps])
            a_mean, a_std = _norm_stats(a_all)
            if sub.state_dim:
                s_mean, s_std = _norm_stats(np.concatenate([e.state for e in eps]))
            sub_id = len(subsets)
            subsets.append(sub)
            for e in eps:
                a = (e.action - a_mean) / a_std
                s = ((e.state - s_mean) / s_std) if sub.state_dim else np.zeros((e.length, 0), np.float32)
                actions.append(np.pad(a, ((0, 0), (0, MAX_DIM - a.shape[1]))).astype(np.float16))
                states.append(np.pad(s, ((0, 0), (0, MAX_DIM - s.shape[1]))).astype(np.float16))
                episodes.append(dict(key=e.key, sub=sub_id, episode_index=e.episode_index, task=e.task,
                                     start=offset, length=e.length, t0=float(e.timestamp[0])))
                offset += e.length
            print(f"{src.name}/{subset or '_'}: {len(eps)} episodes, action {sub.action_dim}D, "
                  f"cams main={main_key.split('.')[-1]} wrist={(wrist_key or '-').split('.')[-1]}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        # subsets as plain dicts: the build runs as __main__, so pickled classes would not resolve elsewhere
        pickle.dump(dict(subsets=[asdict(s) for s in subsets], episodes=pd.DataFrame(episodes),
                         actions=np.concatenate(actions), states=np.concatenate(states)), f, protocol=5)
    print(f"wrote {out}: {len(episodes)} episodes, {offset} frames, {len(subsets)} sub-datasets")


def _resize_pad(img: torch.Tensor, size: int) -> torch.Tensor:
    """(3, H, W) float -> (3, size, size), aspect preserved, zero padding."""
    _, h, w = img.shape
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    img = torch.nn.functional.interpolate(img[None], size=(nh, nw), mode="bilinear", align_corners=False)[0]
    out = torch.zeros(3, size, size)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[:, top:top + nh, left:left + nw] = img
    return out


@lru_cache(maxsize=64)
def _decoder(path: str, pid: int, seek_mode: str = "approximate"):
    """Per-process decoder cache. The pid is part of the key: a decoder opened in the parent and
    inherited by forked DataLoader workers shares FFmpeg state and corrupts reads.

    One FFmpeg thread per decoder: frames are small (256 px), so multi-threaded decode only adds
    thread overhead (12 ms vs 1.4 ms per frame) and oversubscribes the cores across workers."""
    from torchcodec.decoders import VideoDecoder

    return VideoDecoder(path, seek_mode=seek_mode, num_ffmpeg_threads=1)


class PoolDataset(Dataset):
    def __init__(self, index_path: Path, keys: set[str] | None = None, image_size: int = 256, chunk: int = CHUNK,
                 video_root: Path | None = None):
        # video_root: transcoded copies (pretrain/transcode.py), used when present; else the Hub originals
        self.video_root = Path(video_root) if video_root else None
        with open(index_path, "rb") as f:
            idx = pickle.load(f)
        self.subsets = [SubsetInfo(**s) for s in idx["subsets"]]
        eps = idx["episodes"]
        if keys is not None:
            eps = eps[eps["key"].isin(keys)]
        self.episodes = eps.reset_index(drop=True)
        self.actions, self.states = idx["actions"], idx["states"]
        self.image_size, self.chunk = image_size, chunk
        lengths = self.episodes["length"].to_numpy()
        self.cum = np.concatenate([[0], np.cumsum(lengths)])

    def __len__(self) -> int:
        return int(self.cum[-1])

    def _video(self, sub: SubsetInfo, key: str, ep: int) -> tuple[str, float]:
        if sub.v2:
            rel, from_ts = sub.video_path.format(episode_chunk=ep // sub.chunks_size, video_key=key, episode_index=ep), 0.0
        else:
            chunk, file, from_ts = sub.v3_video[key][ep]
            rel = sub.video_path.format(video_key=key, chunk_index=chunk, file_index=file)
        if self.video_root is not None:
            fast = self.video_root / sub.source / (sub.subset or "_") / rel
            if fast.exists():
                return str(fast), from_ts
        return str(Path(sub.root) / rel), from_ts

    def _frame(self, sub: SubsetInfo, key: str, ep: int, t: float) -> torch.Tensor:
        path, from_ts = self._video(sub, key, ep)
        try:
            dec = _decoder(path, os.getpid())
            ts = min(from_ts + t, dec.metadata.duration_seconds - 1e-3) if dec.metadata.duration_seconds else from_ts + t
            frame = dec.get_frames_played_at(seconds=[max(ts, 0.0)]).data[0]  # (3, H, W) uint8
        except RuntimeError:  # retry once with a fresh, exact-seek decoder
            dec = _decoder(path, os.getpid(), "exact")
            ts = min(from_ts + t, dec.metadata.duration_seconds - 1e-3) if dec.metadata.duration_seconds else from_ts + t
            frame = dec.get_frames_played_at(seconds=[max(ts, 0.0)]).data[0]
        # uint8 out: 4x less DataLoader IPC than float32; the training loop converts on the GPU
        return (_resize_pad(frame.float(), self.image_size)).round().clamp(0, 255).to(torch.uint8)

    def __getitem__(self, i: int) -> dict:
        e = int(np.searchsorted(self.cum, i, side="right") - 1)
        row = self.episodes.iloc[e]
        t = int(i - self.cum[e])
        sub = self.subsets[row["sub"]]
        start, n = int(row["start"]), int(row["length"])
        idx = np.arange(t, t + self.chunk)
        pad = idx >= n
        a = self.actions[start + np.minimum(idx, n - 1)].astype(np.float32)
        a[pad] = 0.0
        ep = int(row["episode_index"])
        ts = t / sub.fps
        item = {
            "observation.state": torch.from_numpy(self.states[start + t].astype(np.float32)),
            "action": torch.from_numpy(a),
            "action_is_pad": torch.from_numpy(pad),
            "observation.images.camera1": self._frame(sub, sub.main_key, ep, ts),
            "task": row["task"] or "complete the task",
        }
        if sub.wrist_key is not None:
            item["observation.images.camera2"] = self._frame(sub, sub.wrist_key, ep, ts)
            item["observation.images.camera2_padding_mask"] = torch.tensor(True)
        else:
            item["observation.images.camera2"] = torch.zeros(3, self.image_size, self.image_size, dtype=torch.uint8)
            item["observation.images.camera2_padding_mask"] = torch.tensor(False)
        return item


class EpisodeBlockSampler(Sampler[int]):
    """Draws frames in blocks of `block` from the same episode.

    Community episodes each have their own video file, so uniform frame sampling
    opens two new files per sample and starves the GPU (24 samples/s on the full
    pool). Each block picks an anchor frame uniformly over all frames (so episodes
    are chosen in proportion to length) plus block-1 more frames uniformly from the
    same episode; every frame's marginal probability stays uniform, while the
    worker reuses the episode's open decoders. Keep batch_size a multiple of block
    so a block is never split across workers.
    """

    def __init__(self, ds: "PoolDataset", num_samples: int, block: int = 4, seed: int = 0):
        self.ds, self.num_samples, self.block, self.seed = ds, num_samples, block, seed

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        cum = self.ds.cum
        total = int(cum[-1])
        n = 0
        while n < self.num_samples:
            anchors = rng.integers(0, total, size=4096)
            eps = np.searchsorted(cum, anchors, side="right") - 1
            for a, e in zip(anchors, eps):
                start, length = int(cum[e]), int(cum[e + 1] - cum[e])
                yield int(a)
                for t in rng.integers(0, length, size=self.block - 1):
                    yield start + int(t)
                n += self.block
                if n >= self.num_samples:
                    return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--sources", nargs="*", help="default: all Phase 1 sources")
    args = parser.parse_args()
    if args.cmd == "build":
        build(args.out, args.sources)


if __name__ == "__main__":
    main()
