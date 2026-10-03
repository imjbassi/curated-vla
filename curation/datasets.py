"""Phase 1 audit datasets and download helpers.

Community datasets are collections of per-contributor LeRobot datasets (v2.1);
Open X-Embodiment ports are single LeRobot datasets (v3.0). Downloads fetch
tabular data and metadata only; videos are fetched per episode when needed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download


@dataclass(frozen=True)
class Source:
    name: str
    repo_id: str
    robot: str
    collection: bool  # True: repo holds many sub-datasets, one per contributor/task


SOURCES = [
    Source("community_v1", "HuggingFaceVLA/community_dataset_v1", "so100", True),
    Source("community_v2", "HuggingFaceVLA/community_dataset_v2", "so100", True),
    Source("taco_play", "lerobot/taco_play", "franka", False),
    Source("jaco_play", "lerobot/jaco_play", "jaco", False),
    Source("berkeley_autolab_ur5", "lerobot/berkeley_autolab_ur5", "ur5", False),
    Source("utaustin_mutex", "lerobot/utaustin_mutex", "franka", False),
    Source("roboturk", "lerobot/roboturk", "sawyer", False),
    Source("fmb", "lerobot/fmb", "franka", False),
]
BY_NAME = {s.name: s for s in SOURCES}


def download_tabular(source: Source) -> Path:
    """Download parquet data + metadata (no videos). Returns the local snapshot root."""
    patterns = ["*/meta/*", "*/data/*"] if source.collection else ["meta/*", "data/*"]
    root = snapshot_download(source.repo_id, repo_type="dataset", allow_patterns=patterns + ["README.md"])
    return Path(root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", default=[s.name for s in SOURCES])
    args = parser.parse_args()
    for name in args.names:
        root = download_tabular(BY_NAME[name])
        print(f"{name}\t{root}")


if __name__ == "__main__":
    main()
