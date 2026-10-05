"""Transcode the pool's camera videos for fast random access during pretraining.

The Hub videos are AV1 with long keyframe intervals: a random frame read had a
median of 21 ms and a p90 of 197 ms, capping data loading at ~25 samples/s on
12 cores (training needs ~54). H.264 at 256 px with a keyframe every 10 frames
bounds each read at ~1-2 ms and halves the size. Timestamps are preserved, so
v3 episode offsets (from_timestamp) stay valid.

Only the cameras PoolDataset uses (main + wrist) are transcoded. Output mirrors
the source layout under OUT/<source>/<subset>/<relative video path>.

Usage:
    python -m pretrain.transcode INDEX --out ~/cvla/pool_video [--jobs 12] [--sources ...]
"""

from __future__ import annotations

import argparse
import pickle
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from pretrain.pool import SubsetInfo


def transcode(src: str, dst: str) -> tuple[str, bool]:
    d = Path(dst)
    if d.exists() and d.stat().st_size > 0:
        return dst, True
    d.parent.mkdir(parents=True, exist_ok=True)
    tmp = d.with_suffix(".tmp.mp4")
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-threads", "1", "-i", src,
                        "-vf", "scale=256:256:force_original_aspect_ratio=decrease:force_divisible_by=2",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-g", "10", "-pix_fmt", "yuv420p",
                        "-fps_mode", "passthrough", "-an", str(tmp)], capture_output=True, text=True)
    if r.returncode != 0 or not tmp.exists():
        return f"{dst}: {r.stderr.strip()[:200]}", False
    tmp.rename(d)
    return dst, True


def video_files(sub: SubsetInfo, episodes) -> set[str]:
    """Relative paths of every video file the pool reads for this sub-dataset."""
    rels = set()
    for key in [sub.main_key, sub.wrist_key]:
        if key is None:
            continue
        if sub.v2:
            for ep in episodes:
                rels.add(sub.video_path.format(episode_chunk=ep // sub.chunks_size, video_key=key, episode_index=ep))
        else:
            for chunk, file, _ in set(sub.v3_video[key].values()):
                rels.add(sub.video_path.format(video_key=key, chunk_index=chunk, file_index=file))
    return rels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("index", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--sources", nargs="*")
    args = parser.parse_args()
    idx = pickle.load(open(args.index, "rb"))
    subsets = [SubsetInfo(**s) for s in idx["subsets"]]
    eps = idx["episodes"]
    tasks = []
    for sid, sub in enumerate(subsets):
        if args.sources and sub.source not in args.sources:
            continue
        for rel in sorted(video_files(sub, eps.loc[eps["sub"] == sid, "episode_index"].tolist())):
            src = Path(sub.root) / rel
            if src.exists():
                tasks.append((str(src), str(args.out / sub.source / (sub.subset or "_") / rel)))
    print(f"{len(tasks)} video files to transcode", flush=True)
    done = failed = 0
    with ProcessPoolExecutor(args.jobs) as pool:
        for f in as_completed([pool.submit(transcode, s, d) for s, d in tasks]):
            msg, ok = f.result()
            done += ok
            failed += not ok
            if not ok:
                print("FAILED", msg, flush=True)
            if (done + failed) % 500 == 0:
                print(f"{done + failed}/{len(tasks)} ({failed} failed)", flush=True)
    print(f"finished: {done} ok, {failed} failed")


if __name__ == "__main__":
    main()
