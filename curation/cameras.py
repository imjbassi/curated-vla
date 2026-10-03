"""Pick the main (third-person) camera of a dataset for frames, clips and the VLM judge.

Community datasets renamed cameras to generic image/image2/image3; their
PROCESSING_SUMMARY.json records each sub-dataset's original names (top, wrist,
laptop, phone, ...), which we use to avoid wrist cameras. Other datasets are
judged by key name alone.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

from huggingface_hub import hf_hub_download

NOT_MAIN = re.compile(r"wrist|hand|gripper|eye|depth|ego", re.I)
PREFERRED = ["top", "front", "static", "third", "side", "overhead", "laptop", "phone", "main", "image"]


@lru_cache(maxsize=4)
def _community_mappings(repo_id: str) -> dict[str, dict[str, str]]:
    """{original_id: {generic short key: original short key}} from the processing summary."""
    try:
        summary = json.load(open(hf_hub_download(repo_id, "PROCESSING_SUMMARY.json", repo_type="dataset")))
    except Exception:
        return {}
    out = {}
    for ds_id, d in summary.get("datasets", {}).items():
        m = (d.get("preprocessing_applied", {}).get("feature_remapping", {}) or {}).get("mapping_applied") or {}
        out[ds_id] = {new.split(".")[-1]: orig.split(".")[-1] for orig, new in m.items() if new and "image" in new}
    return out


def main_camera(info: dict, repo_id: str = "", subset: str = "") -> str:
    keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    original = _community_mappings(repo_id).get(subset, {}) if subset else {}

    def describe(k: str) -> str:
        short = k.split(".")[-1]
        return f"{short} {original.get(short, '')}"

    def rank(k: str) -> tuple:
        desc = describe(k)
        pref = next((i for i, p in enumerate(PREFERRED) if p in desc.lower()), len(PREFERRED))
        return (bool(NOT_MAIN.search(desc)), pref, k)

    return sorted(keys, key=rank)[0]
