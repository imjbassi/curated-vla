"""Draw the hand-labeling sample and cut one video clip per episode.

Sample design (blind validation of the detectors):
  - equal episodes per source;
  - within a source, half drawn from episodes any detector flagged and half from
    unflagged ones, so both precision and recall can be estimated (reweight by
    the true flag rate when reporting population numbers);
  - order shuffled; detector outputs are NOT written to the labeling manifest.

Usage:
    python -m curation.sample_for_labeling --scores AUDIT/scores.parquet --out LABEL_DIR [--n 200]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from curation.cameras import main_camera
from curation.datasets import BY_NAME

SEED = 1000


NO_VIDEO_SUBSETS = {
    # community_v2 sub-dataset with tabular data but no video files on the Hub (774 episodes)
    ("community_v2", "Yotofu/so100_sweeper_shoes"),
}


def has_video(scores: pd.DataFrame) -> pd.Series:
    return ~pd.Series(list(zip(scores["source"], scores["subset"])), index=scores.index).isin(NO_VIDEO_SUBSETS)


def draw_sample(scores: pd.DataFrame, n: int) -> pd.DataFrame:
    scores = scores[has_video(scores)]
    sources = sorted(scores["source"].unique())
    per_source = n // len(sources)
    picks = []
    for s in sources:
        g = scores[scores["source"] == s]
        flagged, clean = g[g["flag_any"]], g[~g["flag_any"]]
        k_flag = min(per_source // 2, len(flagged))
        k_clean = min(per_source - k_flag, len(clean))
        picks += [flagged.sample(k_flag, random_state=SEED), clean.sample(k_clean, random_state=SEED)]
    return pd.concat(picks).sample(frac=1, random_state=SEED).reset_index(drop=True)



def clip_episode(row: pd.Series, out_dir: Path) -> Path:
    """Fetch the episode's main-camera video and cut it to out_dir/<id>.mp4 (H.264, browser-playable)."""
    src = BY_NAME[row["source"]]
    prefix = f"{row['subset']}/" if row["subset"] else ""
    info = json.loads(Path(hf_hub_download(src.repo_id, f"{prefix}meta/info.json", repo_type="dataset")).read_text())
    key = main_camera(info, src.repo_id, row["subset"])
    ep = int(row["episode_index"])
    out = out_dir / f"{row['label_id']}.mp4"
    if out.exists():
        return out
    if info["codebase_version"].startswith("v2"):
        chunk = ep // info.get("chunks_size", 1000)
        rel = info["video_path"].format(episode_chunk=chunk, video_key=key, episode_index=ep)
        video = hf_hub_download(src.repo_id, prefix + rel, repo_type="dataset")
        start, end = 0.0, None
    else:
        meta_files = sorted(Path(hf_hub_download(src.repo_id, "meta/info.json", repo_type="dataset")).parent.glob("episodes/*/*.parquet"))
        meta = pd.concat([pd.read_parquet(f, columns=None) for f in meta_files])
        m = meta[meta["episode_index"] == ep].iloc[0]
        rel = info["video_path"].format(video_key=key, chunk_index=int(m[f"videos/{key}/chunk_index"]),
                                        file_index=int(m[f"videos/{key}/file_index"]))
        video = hf_hub_download(src.repo_id, rel, repo_type="dataset")
        start, end = float(m[f"videos/{key}/from_timestamp"]), float(m[f"videos/{key}/to_timestamp"])
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{start:.3f}", "-i", video]
    if end is not None:
        cmd += ["-t", f"{end - start:.3f}"]
    cmd += ["-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=-2:360", "-preset", "veryfast", str(out)]
    subprocess.run(cmd, check=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()
    scores = pd.read_parquet(args.scores)
    sample = draw_sample(scores, args.n)
    sample["label_id"] = [f"ep{i:03d}" for i in range(len(sample))]
    (args.out / "clips").mkdir(parents=True, exist_ok=True)

    # Private key (with detector outputs) kept apart from the blind manifest shown to the labeler.
    sample.to_parquet(args.out / "sample_key.parquet")
    manifest = []
    for _, row in sample.iterrows():
        clip_episode(row, args.out / "clips")
        manifest.append(dict(label_id=row["label_id"], task=row["task"], duration_s=round(row["duration_s"], 1),
                             fps=row["fps"]))
        print(f"{row['label_id']} {row['key']}", flush=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
