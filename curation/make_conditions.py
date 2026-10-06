"""Build the Phase 3 pretraining conditions as episode lists over the pool.

Filters are exactly the Phase 1 validated ones:
  idle      : still > 3 s at the start or the end (idle_start_s / idle_end_s)
  failure   : 8-frame Qwen3-VL-8B judge P(yes) < 0.05, except taco_play, whose fixed
              4.4 s windows look unfinished and are over-flagged (docs/phase1.md)

Conditions (same frame budget for A and B):
  B curated : every pool episode not flagged
  A random  : uniformly random pool episodes until their frames reach |B|
  C full    : the whole pool

Usage:
    python -m curation.make_conditions --index POOL.pkl --scores AUDIT/scores.parquet \
        --vlm AUDIT/vlm8.parquet --out CONDITIONS_DIR [--seed 1000]
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

IDLE_S = 3.0
FAIL_P = 0.05
NO_FAILURE_FILTER = {"taco_play"}


def flags(pool: pd.DataFrame, scores: pd.DataFrame, vlm: pd.DataFrame) -> pd.DataFrame:
    df = pool.merge(scores[["key", "idle_start_s", "idle_end_s"]], on="key", how="left")
    df = df.merge(vlm.drop_duplicates("key", keep="last")[["key", "vlm_p_yes"]], on="key", how="left")
    df["flag_idle"] = (df["idle_start_s"] > IDLE_S) | (df["idle_end_s"] > IDLE_S)
    df["flag_failure"] = (df["vlm_p_yes"] < FAIL_P) & ~df["source"].isin(NO_FAILURE_FILTER)
    df["flag_any"] = df["flag_idle"] | df["flag_failure"]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--vlm", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    idx = pickle.load(open(args.index, "rb"))
    pool = idx["episodes"][["key", "sub", "length"]].copy()
    pool["source"] = pool["sub"].map(lambda s: idx["subsets"][s]["source"])
    df = flags(pool, pd.read_parquet(args.scores), pd.read_parquet(args.vlm))
    missing = df["vlm_p_yes"].isna().sum()
    if missing:
        print(f"note: {missing} pool episodes have no failure score (not flagged for failure)")

    b = df[~df["flag_any"]]
    target = int(b["length"].sum())
    order = df.sample(frac=1.0, random_state=args.seed)
    a = order[order["length"].cumsum().shift(fill_value=0) < target]
    conds = {"A_random": a, "B_curated": b, "C_full": df}

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, c in conds.items():
        c[["key"]].to_parquet(args.out / f"{name}_seed{args.seed}.parquet")
        share = (c.groupby("source")["length"].sum() / c["length"].sum() * 100).round(1)
        rows.append(dict(condition=name, episodes=len(c), frames=int(c["length"].sum()),
                         flagged_frac=round(float(c["flag_any"].mean()), 3), **share.to_dict()))
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out / f"summary_seed{args.seed}.csv", index=False)
    print(summary.to_markdown(index=False))
    print("\npool flag rates (episodes):", df[["flag_idle", "flag_failure", "flag_any"]].mean().round(3).to_dict())


if __name__ == "__main__":
    main()
