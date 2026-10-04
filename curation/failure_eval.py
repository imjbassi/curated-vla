"""Compare VLM failure-detector configurations on the hand-labeled sample.

Each config = (model, number of frames, quantization). Frames are sampled evenly
from each labeled episode's local clip. Reports AUROC for hand-labeled task
completion ('unclear' and empty clips excluded), overall and per source.

Usage:
    python -m curation.failure_eval LABEL_DIR --configs 4b-2 4b-8 8b4-8 [--out RESULTS.csv]
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from curation.frames import _duration
from curation.validate import load_labels
from curation.vlm_judge import Judge

CONFIGS = {
    "4b-2": ("Qwen/Qwen3-VL-4B-Instruct", 2, False),
    "4b-8": ("Qwen/Qwen3-VL-4B-Instruct", 8, False),
    "8b4-2": ("Qwen/Qwen3-VL-8B-Instruct", 2, True),
    "8b4-8": ("Qwen/Qwen3-VL-8B-Instruct", 8, True),
}


def frames_from_clip(clip: Path, n: int, cache: Path) -> list[Image.Image]:
    cache.mkdir(parents=True, exist_ok=True)
    out = [cache / f"{clip.stem}_{n}_{k}.jpg" for k in range(n)]
    if not all(p.exists() for p in out):
        dur = _duration(str(clip))
        for k, p in enumerate(out):
            t = min(dur * k / (n - 1), dur - 0.1)
            # Seeking right to the end can yield no frame on short / low-fps clips: back off until one decodes.
            for back in (0.0, 0.3, 0.6, 1.0, 1.5):
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{max(t - back, 0):.3f}", "-i", str(clip),
                                "-frames:v", "1", "-vf", "scale=-2:256", "-q:v", "3", str(p)], check=True)
                if p.exists():
                    break
    return [Image.open(p) for p in out]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label_dir", type=Path)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    df = pd.read_parquet(args.label_dir / "sample_key.parquet").merge(load_labels(args.label_dir), on="label_id")
    df = df[[(args.label_dir / "clips" / f"{i}.mp4").stat().st_size >= 1000 for i in df["label_id"]]]
    df = df[df["completed"] != "unclear"].reset_index(drop=True)
    y = (df["completed"] == "yes").to_numpy()
    cache = args.label_dir / "frames_eval"

    results = pd.DataFrame({"label_id": df["label_id"], "source": df["source"], "completed": df["completed"]})
    summary = []
    for name in args.configs:
        model_id, n, q4 = CONFIGS[name]
        judge = Judge(model_id, quant4=q4)
        t0 = time.time()
        scores = [judge.p_yes_frames(frames_from_clip(args.label_dir / "clips" / f"{r.label_id}.mp4", n, cache), r.task)
                  for r in df.itertuples()]
        sec = (time.time() - t0) / len(df)
        results[name] = scores
        per_src = {}
        for s, g in results.groupby("source"):
            yy = g["completed"] == "yes"
            per_src[s] = round(roc_auc_score(yy, g[name]), 2) if 0 < yy.sum() < len(g) else None
        summary.append(dict(config=name, auroc=round(roc_auc_score(y, scores), 3), sec_per_episode=round(sec, 3),
                            peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1), **per_src))
        print(summary[-1], flush=True)
        del judge
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print("\n" + pd.DataFrame(summary).to_markdown(index=False))
    if args.out:
        results.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
