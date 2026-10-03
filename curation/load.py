"""Load LeRobot datasets (v2.1 or v3.0 layout) into per-episode arrays.

Reads parquet directly rather than through `LeRobotDataset`, so both the v2.1
community datasets and the v3.0 OXE ports load the same way, and videos are
never touched.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SUCCESS_COLUMNS = ("next.success", "is_success", "success")


@dataclass
class Episode:
    source: str
    subset: str  # sub-dataset path within a collection, "" for single datasets
    episode_index: int
    fps: float
    task: str
    action: np.ndarray  # (T, action_dim)
    state: np.ndarray | None  # (T, state_dim)
    timestamp: np.ndarray  # (T,)
    success: bool | None = None  # dataset-provided label, if any
    extra: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.action)

    @property
    def duration_s(self) -> float:
        return self.length / self.fps

    @property
    def key(self) -> str:
        return f"{self.source}/{self.subset}/{self.episode_index}" if self.subset else f"{self.source}/{self.episode_index}"


def find_datasets(root: Path) -> list[Path]:
    """All dataset roots (directories containing meta/info.json) under root."""
    return sorted(p.parent.parent for p in root.rglob("meta/info.json"))


_TF_REPR = re.compile(r'^tf\.Tensor\(b["\'](.*)["\'],\s*shape=\(\),\s*dtype=string\)$', re.S)


def clean_task(task: str) -> str:
    """Strip TensorFlow repr wrappers left by some OXE conversions (e.g. utaustin_mutex)."""
    m = _TF_REPR.match(task.strip())
    if m:
        task = m.group(1).encode().decode("unicode_escape")
    return " ".join(task.split())


def _tasks(meta: Path) -> dict[int, str]:
    if (meta / "tasks.jsonl").exists():  # v2.1
        rows = [json.loads(line) for line in (meta / "tasks.jsonl").read_text().splitlines() if line.strip()]
        return {r["task_index"]: r["task"] for r in rows}
    if (meta / "tasks.parquet").exists():  # v3.0: task string is the index
        df = pd.read_parquet(meta / "tasks.parquet")
        return {int(i): str(t) for t, i in zip(df.index, df["task_index"])}
    return {}


def _column(df: pd.DataFrame, name: str) -> np.ndarray | None:
    if name not in df.columns:
        return None
    return np.stack(df[name].to_numpy()).astype(np.float32)


def load_dataset(ds_root: Path, source: str, subset: str = "") -> list[Episode]:
    info = json.loads((ds_root / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    tasks = _tasks(ds_root / "meta")
    files = sorted((ds_root / "data").rglob("*.parquet"))
    if not files:
        return []
    wanted = {"action", "observation.state", "timestamp", "episode_index", "frame_index", "task_index", *SUCCESS_COLUMNS}
    frames = []
    for f in files:
        cols = [c for c in pq.read_schema(f).names if c in wanted]
        frames.append(pd.read_parquet(f, columns=cols))
    df = pd.concat(frames, ignore_index=True)
    if "action" not in df.columns:
        return []

    success_col = next((c for c in SUCCESS_COLUMNS if c in df.columns), None)
    episodes = []
    for ep_idx, g in df.groupby("episode_index", sort=True):
        if "frame_index" in g.columns:
            g = g.sort_values("frame_index")
        task_idx = int(g["task_index"].iloc[0]) if "task_index" in g.columns else -1
        success = None
        if success_col is not None:
            success = bool(np.asarray(g[success_col].to_numpy(), dtype=bool).any())
        episodes.append(
            Episode(
                source=source,
                subset=subset,
                episode_index=int(ep_idx),
                fps=fps,
                task=clean_task(tasks.get(task_idx, "")),
                action=_column(g, "action"),
                state=_column(g, "observation.state"),
                timestamp=g["timestamp"].to_numpy(dtype=np.float64) if "timestamp" in g.columns else np.arange(len(g)) / fps,
                success=success,
                extra={"robot_type": info.get("robot_type"), "codebase_version": info.get("codebase_version")},
            )
        )
    return episodes


def load_source(root: Path, source: str) -> list[Episode]:
    """Load every dataset under a downloaded snapshot root."""
    episodes = []
    for ds_root in find_datasets(root):
        subset = str(ds_root.relative_to(root)) if ds_root != root else ""
        episodes.extend(load_dataset(ds_root, source, subset))
    return episodes
