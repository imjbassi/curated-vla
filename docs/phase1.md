# Phase 1: Data audit

Status (2026-10-04): audit and validation done. **Two filters validated**: idle (like-for-like pass: start κ = 1.00, end κ = 0.75) and a failure judge (Qwen3-VL-8B, 8 frames: held-out AUROC 0.93, 95% CI 0.82–1.00 on 100 fresh episodes). Truncation, flailing and spike did not validate. Population scoring with the failure judge is running.

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

## Validation

200 episodes, 25 per source. Within each source, half come from flagged episodes, spread round-robin across the detectors that fire there (so the sample has 14 idle, 60 truncation, 18 spike and 29 flailing flags), and half from unflagged ones; order shuffled. Detector outputs were hidden from the labeler. Labels: task completed (yes/no/unclear) + truncated / idle start / idle end / flailing / glitch / wrong task. Raw labels and outputs: [`labels/phase1/`](../labels/phase1/).

One episode (`ep168`, berkeley_autolab_ur5/613) was excluded: its clip rendered empty, so its label ("unclear", "glitch") describes our tooling, not the data. 199 remain; 186 after excluding "unclear".

### Hand labels

| | count |
|---|---|
| Task completed: yes / no / unclear | 150 / 37 / 13 |
| Truncated / idle start / idle end / flailing / glitch / wrong task | 1 / 0 / 0 / 0 / 0* / 5 |

\*The one glitch label was ep168 (excluded).

Failure rate by source (186 episodes): fmb 60%, roboturk 31%, community_v2 29%, community_v1 20%, utaustin_mutex 16%, taco_play 4%, jaco_play 0%, berkeley_autolab_ur5 0%.

### Detectors vs the problem they target

| detector | label | flagged | label positives | precision | recall | κ |
|---|---|---|---|---|---|---|
| truncation | truncated | 59 | 1 | 0.02 | 1.00 | 0.02 |
| idle start | idle start | 8 | 0 | 0 | – | – |
| idle end | idle end | 10 | 0 | 0 | – | – |
| flailing | flailing | 29 | 0 | 0 | – | – |
| spike | glitch | 18 | 0 | 0 | – | – |

The labeler perceived almost none of the problems these detectors flag.

### Do flags predict failure?

| detector | flagged | fail rate flagged | fail rate unflagged | odds ratio | p (Fisher) |
|---|---|---|---|---|---|
| **idle start** | 8 | **62%** | 18% | **7.7** | **0.008** |
| idle end | 9 | 44% | 19% | 3.5 | 0.078 |
| spike | 17 | 29% | 19% | 1.8 | 0.34 |
| truncation | 57 | 18% | 21% | 0.8 | 0.69 |
| flailing | 27 | 15% | 21% | 0.7 | 0.61 |
| any | 91 | 23% | 17% | 1.5 | 0.36 |

With 6 tests, idle-start survives a Bonferroni correction only marginally (0.008 × 6 ≈ 0.05), and rests on 8 flagged episodes. Treat it as promising, not established.

### VLM judge

Qwen3-VL-4B, first + last frame + instruction, P(yes) for "completed?": **AUROC 0.69** (n = 186). Badly calibrated: median P(yes) is 0.03, so any usable threshold is very low (P < 0.01 flags 37% of episodes, with precision 0.31 and recall 0.57 for failures). Per source: fmb 0.85, community_v2 0.73, community_v1 0.63, utaustin_mutex 0.33 (worse than chance); roboturk and taco_play 1.00 but on 5 and 1 failures.

### VLM judge on the full population

Run on all 34,156 episodes with frames (`curation/vlm_judge.py`, ~1 h on the 4070). 807 episodes have no frames: the 774-episode community_v2 sub-dataset with no videos, plus 33 episodes whose videos are missing or unreadable.

| source | episodes | judged | median P(yes) | % P(yes) < 0.01 | hand-labeled failure rate |
|---|---|---|---|---|---|
| community_v1 | 11,108 | 11,108 | 0.037 | 31.9 | 20% |
| community_v2 | 12,920 | 12,123 | 0.029 | 39.9 | 29% |
| taco_play | 3,603 | 3,602 | 0.060 | 24.2 | 4% |
| jaco_play | 1,085 | 1,085 | 0.679 | 6.7 | 0% |
| berkeley_autolab_ur5 | 1,000 | 993 | 0.003 | **62.9** | **0%** |
| utaustin_mutex | 1,500 | 1,498 | 0.029 | 37.7 | 16% |
| roboturk | 1,943 | 1,943 | 0.029 | 34.0 | 31% |
| fmb | 1,804 | 1,804 | 0.076 | **21.6** | **60%** |

Across sources, the judge's flag rate does not track the hand-labeled failure rate: it flags 63% of berkeley_autolab_ur5 (0 labeled failures) and only 22% of fmb (60% labeled failures). Its scores largely reflect the scene and camera, not task success. **Not usable as a filter.**

### Idle detectors: like-for-like check (2026-10-04)

Because the first pass used a stricter idle definition, a second blind pass labeled idle with the detector's own definition: "completely still for more than ~3 s at the start / end". 40 fresh episodes from the sources where idle fires (community_v1 / v2), 20 flagged by an idle detector and 20 not, shuffled, played at 1× (`curation/sample_idle.py`, `label_server --page idle`). Labels: [`labels/phase1_idle/`](../labels/phase1_idle/).

| detector | labeled positives | flagged | precision | recall | κ | AUROC (raw seconds) |
|---|---|---|---|---|---|---|
| idle at start (> 3 s) | 10 | 10 | **1.00** | **1.00** | **1.00** | 1.00 |
| idle at end (> 3 s) | 11 | 11 | 0.82 | 0.82 | 0.75 | 0.98 |

All four idle-end disagreements are within 0.8 s of the threshold (detector measured 2.2, 2.5, 3.2 and 3.4 s): borderline calls against an eyeballed "about 3 seconds", not detector errors. **Both idle detectors are validated** at the 3 s threshold.

### Failure detector iteration (2026-10-04)

`python -m curation.failure_eval LABELS --configs ...`: frames sampled evenly from each labeled clip; AUROC vs hand-labeled completion on the same 187 episodes ("unclear" and the empty clip excluded). Per-source AUROC is blank where a source had no labeled failures.

| config | AUROC | community_v1 | community_v2 | fmb | roboturk | taco_play | utaustin_mutex | s/episode | peak VRAM |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-4B, 2 frames | 0.73 | 0.63 | 0.75 | 0.78 | 1.00 | 0.86 | 0.30 | 0.31 | 9.1 GB |
| Qwen3-VL-4B, 8 frames | 0.83 | 0.91 | 0.82 | 0.99 | 0.71 | 0.50 | 0.31 | 0.89 | 9.4 GB |
| Qwen3-VL-8B (4-bit), 2 frames | 0.77 | 0.78 | 0.71 | 0.92 | 0.91 | 0.73 | 0.38 | 0.21 | 7.0 GB |
| **Qwen3-VL-8B (4-bit), 8 frames** | **0.84** | 0.89 | 0.80 | 0.99 | 0.85 | 0.86 | 0.57 | 0.44 | 11.5 GB |

(The 2-frame 4B number here, 0.73, differs from the 0.69 above because frames now come from the labeling clips rather than separately extracted frames.)

- **More frames matter more than model size**: 2 → 8 frames adds ~0.07–0.09 AUROC for both models; 4B → 8B adds ~0.01–0.04.
- The 8B model is more even across sources (utaustin_mutex 0.57 vs 0.31, taco_play 0.86 vs 0.50).
- **Selection caveat:** the best config was picked on the same 187 labels it is scored on, so 0.84 is optimistic. It must be confirmed on fresh labels before being used as a Phase 3 filter.
- **Cost to apply everywhere:** the population frames are first/mid/last only, so 8-frame scoring needs a new frame pass over all videos (another multi-hour stream) plus ~4 h of GPU at 0.44 s/episode.

### Failure detector: held-out confirmation (2026-10-04)

The configuration was fixed in advance: **Qwen3-VL-8B, 4-bit, 8 frames**. A fresh completion-only pass labeled 100 new episodes, drawn uniformly at random, 12–13 per source, with no overlap with any earlier label (`curation/sample_completion.py`, `label_server --page completion`). Labels: [`labels/phase1_completion/`](../labels/phase1_completion/).

Labels: 80 yes, 9 no, 11 unclear. The 9 failures: fmb 6, community_v2 2, roboturk 1. **Random-sample failure rate: 10%** (9 / 89); the first sample's 19% was inflated by its oversampling of flagged episodes.

| config | AUROC (held-out) | 95% bootstrap CI |
|---|---|---|
| **Qwen3-VL-8B, 4-bit, 8 frames (pre-registered)** | **0.93** | 0.82 – 1.00 |
| Qwen3-VL-4B, 8 frames (secondary) | 0.93 | 0.82 – 1.00 |

Operating points for the 8B judge (flag = P(yes) below threshold):

| threshold | flagged | precision (failure) | recall (failure) |
|---|---|---|---|
| < 0.01 | 7% | 1.00 | 0.67 |
| < 0.05 | 15% | 0.54 | 0.78 |
| < 0.10 | 25% | 0.36 | 0.89 |

**Confirmed, with wide error bars.** The held-out AUROC is not lower than the selection-sample estimate (0.84), so the choice did not overfit. But only 9 failures support it, 6 from fmb, whose failures are easy to see, so the point estimate is likely optimistic and the lower bound (0.82) is the safer number. The 4B run needed 13.7 GB here (it spilled past VRAM and ran 3× slower), so the 8B in 4-bit is the practical choice.

### Failure judge on the full population (2026-10-05)

8 evenly spaced frames per episode (`curation.frames --n-frames 8`, 6 parallel shards), then the pre-registered judge (Qwen3-VL-8B, 4-bit) over all 34,183 episodes with video (~5 h on the 4070). Flag = P(yes) < 0.05. Labeled failure rate pools the first pass and the held-out pass ("unclear" excluded).

| source | scored | failure flag | idle flag | either | labeled failure rate (n) |
|---|---|---|---|---|---|
| community_v1 | 11,108 | 16.7% | 18.4% | 32.0% | 14% (36) |
| community_v2 | 12,146 | 18.0% | 28.1% | 40.9% | 26% (34) |
| taco_play | 3,602 | **24.8%** | 0% | 24.8% | **3% (33)** |
| jaco_play | 1,085 | 3.2% | 0% | 3.2% | 0% (37) |
| berkeley_autolab_ur5 | 997 | 5.2% | 0% | 5.2% | 0% (37) |
| utaustin_mutex | 1,498 | 19.3% | 0% | 19.3% | 11% (37) |
| roboturk | 1,943 | 23.3% | 0.1% | 23.3% | 25% (24) |
| fmb | 1,804 | 39.5% | 0% | 39.5% | 55% (38) |
| **all** | **34,183** | **19.3%** | **16.0%** | **32.4%** | |

- Unlike the first VLM version, **flag rates now track labeled failure rates across sources**, with one exception.
- **taco_play is over-flagged** (25% vs 3% labeled): its episodes are fixed 4.4 s windows cut from continuous play, which look unfinished. Exclude taco_play from failure filtering.
- **Idle and failure flags are nearly independent**: an idle-start flag barely changes the chance of a failure flag (20.8% vs 19.2%). The two filters remove different episodes.

### Conclusions

1. **Don't filter on truncation, flailing or spike as built.** No agreement with labels and no relationship to failure.
2. **Idle detectors are validated** (start κ = 1.00, end κ = 0.75 on a like-for-like pass), and idle-at-start is associated with failure (62% vs 18%, n = 8 flagged). Idle is the first filter ready for Phase 3.
3. **Task failure is the quality problem worth curating**, and it is concentrated by source. A failure detector good enough to filter on is the main open item: try a larger VLM, more frames (or video), and source-specific prompts, and validate on the same 186 labels.
4. **Labeling caveat (confirmed with the labeler, 2026-10-04):** the idle boxes were ticked only when the robot never moved during the whole episode, much stricter than the detector's "> 3 s still at the start/end". So the idle-start / idle-end agreement rows above are **not a valid test** of the idle detectors; they need re-scoring against labels that use the detector's definition. The other problem boxes (truncated, flailing, glitch) were used as named, so those detectors remain unvalidated. The idle-start → failure association is unaffected, since it uses the completion label.

```bash
python -m curation.sample_for_labeling --scores AUDIT/scores.parquet --out LABELS --n 200
python -m curation.label_server LABELS   # open http://localhost:8765
python -m curation.validate LABELS --scores AUDIT/scores.parquet --vlm AUDIT/vlm.parquet
```

`validate.py` reports precision/recall/F1/κ per detector, population-reweighted precision/recall, F1 within length tertiles, and VLM AUROC vs hand-labeled completion.
