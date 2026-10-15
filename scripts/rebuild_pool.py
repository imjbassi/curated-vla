"""Rebuild pool_video from the Hub without the full raw download.

The pool index (pretrain/pool.py build) records each sub-dataset's cache root, which
pins the Hub revision. This downloads only the video files the pool reads (main +
wrist camera, every indexed episode) at those revisions, into the standard HF cache
so the recorded roots resolve, then runs pretrain/transcode.py on them. Resumable:
already downloaded files and already transcoded outputs are skipped.

Written when the original pool_video (and its raw downloads) were lost with a failing
drive; the rebuilt files are byte-for-byte the same inputs, transcoded the same way.

Usage:
    python scripts/rebuild_pool.py ~/cvla/pool/index_all.pkl --out ~/cvla/pool_video [--sources ...] [--workers 16]
"""

from __future__ import annotations

import argparse
import pickle
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

from curation.datasets import BY_NAME
from pretrain.pool import SubsetInfo
from pretrain.transcode import video_files


def fetch(repo_id: str, revision: str, rel: str) -> tuple[str, str | None]:
    for attempt in range(5):
        try:
            hf_hub_download(repo_id, rel, repo_type="dataset", revision=revision)
            return rel, None
        except HfHubHTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return rel, "404"
            time.sleep(10 * (attempt + 1))
        except Exception as e:  # transient network errors
            err = f"{type(e).__name__}: {str(e)[:100]}"
            time.sleep(10 * (attempt + 1))
    return rel, err


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("index", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sources", nargs="*")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--jobs", type=int, default=12, help="transcode processes")
    args = parser.parse_args()
    idx = pickle.load(open(args.index, "rb"))
    eps = idx["episodes"]
    subsets = [SubsetInfo(**s) for s in idx["subsets"]]
    by_source: dict[str, tuple[str, set[str]]] = {}
    for sid, sub in enumerate(subsets):
        if args.sources and sub.source not in args.sources:
            continue
        m = re.search(r"/snapshots/([0-9a-f]{40})", sub.root)
        if not m:
            sys.exit(f"{sub.source}/{sub.subset}: root does not pin a revision: {sub.root}")
        rev, files = by_source.setdefault(sub.source, (m.group(1), set()))
        assert rev == m.group(1), f"{sub.source}: mixed revisions"
        prefix = f"{sub.subset}/" if sub.subset else ""
        files |= {prefix + r for r in video_files(sub, eps.loc[eps["sub"] == sid, "episode_index"].tolist())}

    for source, (rev, files) in by_source.items():
        repo_id = BY_NAME[source].repo_id
        print(f"== {source}: {len(files)} video files from {repo_id}@{rev[:8]}", flush=True)
        t = time.time()
        missing = []
        with ThreadPoolExecutor(args.workers) as ex:
            futures = [ex.submit(fetch, repo_id, rev, rel) for rel in sorted(files)]
            for i, f in enumerate(as_completed(futures), 1):
                rel, err = f.result()
                if err:
                    missing.append((rel, err))
                if i % 1000 == 0:
                    print(f"   {i}/{len(files)} downloaded ({(time.time() - t) / 60:.0f} min)", flush=True)
        print(f"   download done in {(time.time() - t) / 60:.0f} min, {len(missing)} missing", flush=True)
        for rel, err in missing[:20]:
            print(f"   MISSING {rel}: {err}", flush=True)
        subprocess.run([sys.executable, "-m", "pretrain.transcode", str(args.index), "--out", str(args.out),
                        "--jobs", str(args.jobs), "--sources", source], check=False)


if __name__ == "__main__":
    main()
