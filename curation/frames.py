"""Extract a few key frames per episode (main camera) for the VLM success judge.

Streams videos: for per-episode files (v2.1 community datasets) each video is
downloaded, sampled, and deleted, so disk use stays small even though the full
community videos are ~520 GB. Shared multi-episode files (v3.0) are kept in the
HF cache since they are small for the OXE datasets used here.

Output: FRAMES_DIR/<source>/<subset>/<episode>_{first,mid,last}.jpg (256 px tall), or
<episode>_f0..f{N-1}.jpg with --n-frames N (evenly spaced, for the multi-frame judge).

Usage:
    python -m curation.frames --scores AUDIT/scores.parquet --out FRAMES_DIR [--sources ...]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from curation.datasets import BY_NAME
from curation.cameras import main_camera
from curation.sample_for_labeling import has_video

POSITIONS = {"first": 0.0, "mid": 0.5, "last": 1.0}


def positions(n_frames: int) -> dict[str, float]:
    """3 -> first/mid/last (original layout); otherwise f0..f{n-1} evenly spaced."""
    if n_frames == 3:
        return POSITIONS
    return {f"f{k}": k / (n_frames - 1) for k in range(n_frames)}


def _grab(video: str, t: float, out: Path) -> None:
    # Seeking right to the end can yield no frame on short / low-fps videos: back off until one decodes.
    for back in (0.0, 0.3, 0.6, 1.0, 1.5):
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{max(t - back, 0):.3f}", "-i", video,
                        "-frames:v", "1", "-vf", "scale=-2:256", "-q:v", "3", str(out)], check=True)
        if out.exists():
            return


def _duration(video: str) -> float:
    """Container duration, falling back to stream duration, then packet count / frame rate."""
    for entry in ("format=duration", "stream=duration"):
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", entry,
                            "-of", "csv=p=0", video], capture_output=True, text=True, check=True)
        try:
            return float(r.stdout.strip().splitlines()[0])
        except (ValueError, IndexError):
            pass
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                        "stream=nb_read_packets,r_frame_rate", "-of", "csv=p=0", video],
                       capture_output=True, text=True, check=True)
    rate, packets = r.stdout.strip().split(",")
    num, den = rate.split("/")
    return int(packets) * int(den) / int(num)


def extract_subset(source: str, subset: str, episodes: list[int], out_root: Path, n_frames: int = 3) -> int:
    pos = positions(n_frames)
    src = BY_NAME[source]
    prefix = f"{subset}/" if subset else ""
    info = json.loads(Path(hf_hub_download(src.repo_id, f"{prefix}meta/info.json", repo_type="dataset")).read_text())
    key = main_camera(info, src.repo_id, subset)
    out_dir = out_root / source / (subset or "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    v2 = info["codebase_version"].startswith("v2")
    meta = None
    if not v2:
        files = sorted(Path(hf_hub_download(src.repo_id, "meta/info.json", repo_type="dataset")).parent.glob("episodes/*/*.parquet"))
        meta = pd.concat([pd.read_parquet(f) for f in files]).set_index("episode_index")
    for ep in episodes:
        targets = {name: out_dir / f"{ep:06d}_{name}.jpg" for name in pos}
        if all(p.exists() for p in targets.values()):
            continue
        try:
            if v2:
                rel = info["video_path"].format(episode_chunk=ep // info.get("chunks_size", 1000), video_key=key,
                                                episode_index=ep)
                with tempfile.TemporaryDirectory() as tmp:
                    video = hf_hub_download(src.repo_id, prefix + rel, repo_type="dataset", local_dir=tmp)
                    dur = _duration(video)
                    for name, frac in pos.items():
                        _grab(video, min(frac * dur, dur - 0.1), targets[name])
                    os.remove(video)
            else:
                m = meta.loc[ep]
                rel = info["video_path"].format(video_key=key, chunk_index=int(m[f"videos/{key}/chunk_index"]),
                                                file_index=int(m[f"videos/{key}/file_index"]))
                video = hf_hub_download(src.repo_id, rel, repo_type="dataset")
                t0, t1 = float(m[f"videos/{key}/from_timestamp"]), float(m[f"videos/{key}/to_timestamp"])
                for name, frac in pos.items():
                    _grab(video, min(t0 + frac * (t1 - t0), t1 - 0.1), targets[name])
        except Exception as e:  # one bad episode should not stop its sub-dataset
            print(f"FAILED {source}/{subset or '_'}/{ep}: {type(e).__name__}: {str(e)[:150]}", flush=True)
            continue
        done += 1
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sources", nargs="*")
    parser.add_argument("--shard", default="0/1", help="i/n: process every n-th sub-dataset starting at i")
    parser.add_argument("--n-frames", type=int, default=3, help="3 = first/mid/last; otherwise evenly spaced f0..")
    args = parser.parse_args()
    shard_i, shard_n = map(int, args.shard.split("/"))
    df = pd.read_parquet(args.scores, columns=["source", "subset", "episode_index"])
    df = df[has_video(df)]
    if args.sources:
        df = df[df["source"].isin(args.sources)]
    for k, ((source, subset), g) in enumerate(df.groupby(["source", "subset"], sort=False)):
        if k % shard_n != shard_i:
            continue
        try:
            n = extract_subset(source, subset, sorted(g["episode_index"].tolist()), args.out, args.n_frames)
            print(f"{source}/{subset or '_'}: {n} episodes", flush=True)
        except Exception as e:  # one broken sub-dataset should not stop the run
            print(f"FAILED {source}/{subset or '_'}: {type(e).__name__}: {str(e)[:200]}", flush=True)


if __name__ == "__main__":
    main()
