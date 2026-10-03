"""Summarize LIBERO eval runs into a per-suite success table.

Usage: python eval/summarize.py RUN_DIR [RUN_DIR ...]

Reads eval_info.json if lerobot-eval wrote one, otherwise parses eval.log.
"""

import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def per_task_results(run_dir: Path) -> list[dict]:
    info = run_dir / "eval_info.json"
    if info.exists():
        data = json.loads(info.read_text())
        if "per_task" in data:
            return data["per_task"]
    log = (run_dir / "eval.log").read_text(errors="replace")
    match = re.search(r"Aggregated Metrics for per_task:\n.*?(\[\{.*\}\])\n", log)
    if not match:
        raise ValueError(f"no per-task metrics found in {run_dir}")
    return ast.literal_eval(match.group(1))


def main() -> None:
    successes: dict[str, list[bool]] = defaultdict(list)
    for arg in sys.argv[1:]:
        for task in per_task_results(Path(arg)):
            successes[task["task_group"]].extend(task["metrics"]["successes"])

    rows = [s for s in SUITES if s in successes] + sorted(set(successes) - set(SUITES))
    print(f"{'suite':<16}{'success %':>10}{'episodes':>10}")
    total = []
    for suite in rows:
        vals = successes[suite]
        total.extend(vals)
        print(f"{suite:<16}{100 * sum(vals) / len(vals):>10.1f}{len(vals):>10}")
    suite_means = [100 * sum(successes[s]) / len(successes[s]) for s in rows]
    print(f"{'mean of suites':<16}{sum(suite_means) / len(suite_means):>10.1f}{len(total):>10}")


if __name__ == "__main__":
    main()
