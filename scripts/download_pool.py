"""Download the full pretraining pool (data + videos) for all Phase 1 sources.

Files go to the Hugging Face cache inside WSL, whose disk lives on D: (see
docs/phase0.md), so the ~550 GB of video does not touch C:. Resumable: re-running
skips files already downloaded.

Usage:
    python scripts/download_pool.py [SOURCE ...] [--workers 8]
"""

from __future__ import annotations

import argparse
import time

from huggingface_hub import snapshot_download

from curation.datasets import BY_NAME, SOURCES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", default=[s.name for s in SOURCES])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    for name in args.names:
        src = BY_NAME[name]
        for attempt in range(5):  # retry on rate limits / transient network errors
            try:
                t = time.time()
                path = snapshot_download(src.repo_id, repo_type="dataset", max_workers=args.workers)
                print(f"{name}\tdone in {(time.time() - t) / 60:.0f} min\t{path}", flush=True)
                break
            except Exception as e:
                print(f"{name}\tattempt {attempt + 1} failed: {type(e).__name__}: {str(e)[:150]}", flush=True)
                time.sleep(60 * (attempt + 1))


if __name__ == "__main__":
    main()
