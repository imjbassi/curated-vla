# Phase 1: Data audit

Status: in progress (started 2026-10-03). Detectors built and run on all sources; **hand-label validation pending** (200 episodes), VLM judge pending full run.

## Datasets

8 sources, 6 robot types, 34,963 episodes / 16.5M frames (episodes ≥ 10 frames).

| Source | Hub repo | Robot | Why |
|---|---|---|---|
| community_v1 | `HuggingFaceVLA/community_dataset_v1` | SO-100 | SmolVLA's pretraining set; already filtered by HF at the *dataset* level (fps, min episodes, qualitative video review) |
| community_v2 | `HuggingFaceVLA/community_dataset_v2` | SO-100 | Larger, less-filtered follow-on: the contrast case |
| taco_play | `lerobot/taco_play` | Franka | Play data |
| jaco_play | `lerobot/jaco_play` | Jaco | Different arm, teleop play |
| berkeley_autolab_ur5 | `lerobot/berkeley_autolab_ur5` | UR5 | Different arm |
| utaustin_mutex | `lerobot/utaustin_mutex` | Franka | 50 tasks, multi-step |
| roboturk | `lerobot/roboturk` | Sawyer | Crowd-sourced teleop |
| fmb | `lerobot/fmb` | Franka | Precise assembly/insertion |

Reproduce: `python -m curation.audit --out OUT` (downloads parquet + metadata only, ~1 GB).

## Dataset stats

| source | sub-datasets | episodes | frames | fps | median len (s) | p5–p95 len (s) | action dim | tasks | success labels |
|---|---|---|---|---|---|---|---|---|---|
| community_v1 | 128 | 11,108 | 5.11M | 30/50 | 13.3 | 7.0–26.9 | 6 | 109 | none |
| community_v2 | 324 | 12,920 | 10.12M | 5–30 (7 values) | 14.4 | 6.9–175 | 6/7/12/14 | 242 | none |
| taco_play | 1 | 3,603 | 0.24M | 15 | 4.4 | 4.4–4.4 | 7 | 406 | none |
| jaco_play | 1 | 1,085 | 0.08M | 10 | 7.0 | 3.9–11.2 | 7 | 89 | none |
| berkeley_autolab_ur5 | 1 | 1,000 | 0.10M | 5 | 19.6 | 14.6–24.6 | 7 | 5 | none |
| utaustin_mutex | 1 | 1,500 | 0.36M | 20 | 9.1 | 6.5–22.1 | 7 | 50 | none |
| roboturk | 1 | 1,943 | 0.19M | 10 | 8.9 | 2.0–17.4 | 7 | 3 | none |
| fmb | 1 | 1,804 | 0.34M | 10 | 15.6 | 10.2–32.4 | 7 | 6 | none |

**No source has per-episode success labels**, so "possible failure" must come from the VLM judge (or hand labels). OXE `next.reward` is 1 on the final step by convention and is not a success label.

### Dataset-level findings

- **taco_play episodes are fixed 66-frame (4.4 s) windows** cut from continuous play. Length-based reasoning does not apply, and "truncation" is by construction. Treat taco_play separately or exclude it from truncation filtering.
- **roboturk's `observation.state` is all zeros** in this port. Motion must come from (integrated) actions. It also contains 1-frame episodes (dropped by the ≥ 10-frame filter).
- **community_v2 is heterogeneous**: 7 different frame rates, 4 action dimensionalities (6/7/12/14, i.e. not only single SO-100 arms), and a long tail of very long episodes (p95 = 175 s).
- **Binary grippers** (taco, ur5, roboturk, fmb): a 2-valued dimension; must be excluded from motion metrics or every grasp reads as a teleport.
- **Community camera names are generic** (`image`, `image2`, `image3`); original names (top/wrist/laptop/phone…) are recoverable from each collection's `PROCESSING_SUMMARY.json` (`curation/cameras.py`).

## Detectors (`curation/detectors.py`)

Computed from robot state (or integrated delta actions when the state is unusable), normalized per sub-dataset by the typical per-step change of each continuous dimension, so "1" = a typical step for that robot.

| Detector | Score | Initial flag | Failure mode |
|---|---|---|---|
| Idle start / end | seconds motionless before first / after last moving frame | > 3 s | idle time |
| Truncation | median speed over final 5% (excl. last frame) ÷ median moving speed | ≥ 1.0 | truncation |
| Spike | max single-step motion ÷ its neighbors' (±3 steps) | > 10 | recording glitch |
| Flailing | mean normalized jerk over moving frames, robust z within source | z > 3.5 | flailing |
| VLM judge | P(yes) "instruction completed?" from first+last frames, Qwen3-VL-4B | TBD from labels | possible failure |

Design iterations (each caught by inspecting flagged episodes, not by tuning on outcomes):

1. **Idle as a fraction of the episode was length-confounded** (ρ = −0.66 with length on jaco_play, −0.97 on fmb). Switched to seconds (ρ ≈ 0.05).
2. **Top "truncation" episodes in jaco_play were last-frame state jumps**, not truncations. Truncation now excludes the final frame; jumps get their own *spike* detector.
3. **fmb's "spikes" were fast-but-smooth transport segments** between slow insertions. Spike is now relative to neighboring steps, not the episode median (fmb: 47.5% → 0.1% flagged).
4. Binary gripper dims excluded; radian Euler angles unwrapped at ±π.

## Flag rates (initial thresholds, % of episodes)

| source | idle | truncation | spike | flailing | any |
|---|---|---|---|---|---|
| community_v1 | 18.4 | 5.1 | 0.1 | 1.6 | 24.2 |
| community_v2 | 28.1 | 3.1 | 6.2 | 7.4 | 36.6 |
| taco_play | 0.0 | 54.9 | 0.0 | 1.2 | 55.4 |
| jaco_play | 0.0 | 35.2 | 2.0 | 1.0 | 37.2 |
| berkeley_autolab_ur5 | 0.0 | 50.4 | 0.0 | 0.2 | 50.6 |
| utaustin_mutex | 0.0 | 21.3 | 0.0 | 0.0 | 21.3 |
| roboturk | 0.1 | 45.1 | 0.7 | 4.8 | 48.2 |
| fmb | 0.0 | 13.0 | 0.1 | 0.1 | 13.1 |

Early signals, **not yet validated**:
- The less-filtered community_v2 trips every motion detector more than HF-filtered v1 (spike 6.2% vs 0.1%, flailing 7.4% vs 1.6%, idle 28% vs 18%).
- Truncation fires on 13–55% of OXE episodes but only 3–5% of community episodes. Likely an OXE recording convention (episodes stop the instant the task completes) rather than a quality failure; hand labels will decide whether truncation is meaningful for OXE.

## Length confound

Spearman ρ of each score with episode length, overall and within length quintiles (`curation.audit` writes the full table). Summary of what needs care:

- **Idle fraction**: strongly confounded everywhere (ρ −0.66 to −0.99); not used for flags.
- **Idle seconds**: modest overall correlation in some sources (ur5 0.72, community 0.27–0.41), but much weaker within length quintiles (≤ 0.25). Longer episodes do contain more hesitation; validation will check it within length bins.
- **Spike**: mild positive correlation (max over more steps); within-quintile ρ ≤ 0.17.
- **Truncation, jerk**: within-quintile |ρ| ≤ 0.17 everywhere.

## Validation (pending)

200 episodes, 25 per source, half flagged / half unflagged by any detector, shuffled; detector outputs hidden from the labeler. Labels: task completed (yes/no/unclear) + truncated / idle start / idle end / flailing / glitch / wrong task.

```bash
python -m curation.sample_for_labeling --scores AUDIT/scores.parquet --out LABELS --n 200
python -m curation.label_server LABELS   # open http://localhost:8765
python -m curation.validate LABELS --scores AUDIT/scores.parquet --vlm AUDIT/vlm.parquet
```

`validate.py` reports precision/recall/F1/κ per detector, population-reweighted precision/recall, F1 within length tertiles, and VLM AUROC vs hand-labeled completion.
