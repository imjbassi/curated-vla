"""Rebuild a run_protocol manifest from existing eval directories.

Reads each finished eval's own log for its suite and task id, so tasks whose
'done' line was never written (the protocol stopped after the eval wrote its
results) are still counted. Keeps the latest result per (suite, task).

Usage: python eval/rebuild_manifest.py TAG [--seed 1000]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

EVAL = Path.home() / "cvla" / "outputs" / "eval"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    found = {}
    for d in sorted(EVAL.glob(f"*_{args.tag}_seed{args.seed}_*")):
        log = (d / "eval.log").read_text(errors="replace") if (d / "eval.log").exists() else ""
        m = re.search(r"'task_group': '(\w+)', 'task_id': (\d+)", log)
        if m and "'pc_success'" in log:
            found[(m.group(1), int(m.group(2)))] = d  # sorted by time: later results win
    out = EVAL / f"manifest_{args.tag}_seed{args.seed}.tsv"
    with open(out, "w") as f:
        for (suite, task), d in sorted(found.items()):
            f.write(f"{suite}\t{task}\t{d}/\n")
    print(f"{out}: {len(found)} tasks")
    for suite in sorted({s for s, _ in found}):
        print(f"  {suite}: tasks {sorted(t for s, t in found if s == suite)}")


if __name__ == "__main__":
    main()
