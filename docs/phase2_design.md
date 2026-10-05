# Phase 2 design: pretraining on a mixed public pool

Status: design (2026-10-05). Pool download in progress (~550 GB, into WSL on D:).

## Constraint found

LeRobot 0.6.1 cannot pretrain on several datasets at once:

- `lerobot-train` with a list of `repo_id`s raises `NotImplementedError("The MultiLeRobotDataset isn't supported for now.")`.
- `aggregate_datasets` requires identical features across inputs and **pools normalization statistics** across them. Our pool mixes SO-100 joint angles in degrees, Franka/UR5/Sawyer end-effector deltas in meters, and 6/7/12/14-D actions, so pooled stats would be wrong for every robot.
- The community collections are LeRobot v2.1; 0.6.1 reads v3.0 only. Converting 452 sub-datasets rewrites ~520 GB of video.

## Approach

Pretraining uses our own data layer and a thin training loop around LeRobot's SmolVLA policy. Post-training and evaluation stay on stock `lerobot-train` / `lerobot-eval` (single LIBERO dataset), so the Phase 0 harness and the Phase 2 `smolvla_base` control are unchanged.

**`PoolDataset`** (torch `Dataset`), reading the Hub files directly in both layouts (as `curation/load.py` already does):

| Concern | Rule |
|---|---|
| Actions / state | **Normalized per sub-dataset** (mean/std computed from that sub-dataset), then zero-padded to 32-D, matching SmolVLA's `max_action_dim` / `max_state_dim`. |
| Action chunk | 50 future actions at the dataset's native rate (SmolVLA `chunk_size`); padded and masked past the episode end. |
| Cameras | Main third-person view → `camera1` (same picker as Phase 1, `curation/cameras.py`); a wrist view → `camera2` when present; missing views left empty (SmolVLA supports a subset). |
| Language | Cleaned task string (`curation.load.clean_task`). |
| Video decode | torchcodec seek per sampled frame (v2.1: per-episode file; v3.0: shared file + episode offset). |
| Episodes | Driven by an **episode list**, so each condition is just a list. |

**Conditions as episode lists** (Phase 3), all at the same frame budget:

| Condition | Episode list |
|---|---|
| A: random | uniform random episodes from the pool until the budget is met |
| B: curated | pool minus episodes flagged by validated filters (idle > 3 s at start/end; failure judge P(yes) < 0.05, except taco_play), then random until the same budget |
| C: full | all pool episodes |

Filters used are exactly the Phase 1 validated ones; truncation, flailing and spike are excluded.

**Exclusions from the pool**: the community_v2 sub-dataset with no videos (774 episodes); episodes < 10 frames; any episode whose video fails to decode at load time (logged).

## Training

- Model: `--policy.type=smolvla --policy.load_vlm_weights=true` (SmolVLM2-500M backbone, robotics weights ours).
- Budget: 5M samples per pretraining run (~26 h at 54 samples/s, batch 32, fp32), per the Phase 0 estimate. Revisit bf16 AMP (broken in 0.6.1) for a ~1.5–2× speedup.
- Post-training: LIBERO via stock `lerobot-train`, identical settings for our model and the `smolvla_base` control.
- Control gate: if `smolvla_base` post-trained with our settings does not land near published numbers (see README), stop and debug before Phase 3.

## Build order

1. `PoolDataset` + unit checks (shapes, normalization, padding masks, camera mapping) on the small OXE sources while community videos download.
2. Throughput check: data loading must keep up with ~54 samples/s.
3. Pretraining loop (SmolVLA forward/backward, cosine LR, checkpointing in LeRobot's `pretrained_model` format so `lerobot-train --policy.path` can post-train from it).
4. Short end-to-end smoke run: pretrain a few thousand steps → post-train on LIBERO → evaluate.
5. `smolvla_base` control post-training.
