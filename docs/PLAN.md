# Project plan

Built so each phase produces something useful even if later phases don't pan out.

## The goal

Train a small vision-language-action model without collecting any teleop data, and use it to answer one research question:

**Does curating public robot data by failure type let a smaller dataset match or beat a larger, uncurated one?**

## Core technical choices

| Decision | Choice | Why |
|---|---|---|
| Framework | Hugging Face LeRobot | Standard, well-documented, handles datasets, training, and eval |
| Architecture | SmolVLA-style (small VLM backbone plus action expert, roughly 450M params) | Same design family as π0, small enough for an RTX 4070 |
| Starting point | A pretrained vision-language model, not robotics-pretrained weights | So the robotics pretraining is genuinely ours |
| Pretraining data | Public LeRobot community datasets plus a slice of Open X-Embodiment | Real robot data, no teleop needed |
| Evaluation | LIBERO simulation benchmark | Standardized, closed-loop, widely reported |

Verify current versions and docs for each before starting; this ecosystem moves fast.

## Phase 0: Setup and reproduction (week 1)

**Goal:** prove the environment works before building anything.

1. Set up LeRobot in WSL2, confirm the 4070 is visible, check free VRAM.
2. Download a published SmolVLA checkpoint and run its LIBERO evaluation.
3. Compare success rates to the published ones.
4. Measure training throughput (samples per second) on a short run.

**Gate:** if published numbers can't be reproduced, stop and fix that first. Every later result depends on the evaluation being trustworthy.

**Output:** a compute estimate. From measured throughput, calculate how long each training run will take. If pretraining would take weeks locally, budget cloud GPU time now.

## Phase 1: Data audit (weeks 2–3)

**Goal:** understand what's actually in the public data.

1. Pick 5–10 datasets with different robots and tasks.
2. Compute basic stats: episode counts, lengths, control rates, action spaces, cameras, whether success labels exist.
3. Build quality detectors mapped to known failure modes:
   - **Truncation:** episodes that end abruptly mid-task
   - **Idle time:** long stretches of no motion at the start or end
   - **Flailing:** erratic, jerky motion
   - **Possible failure:** a VLM looks at the final frames and judges whether the task was completed
4. Validate the detectors by hand: label ~200 episodes and measure agreement with each detector.

**Trap to avoid:** quality metrics can secretly just measure episode length. Check every detector against duration, and test it within groups of similar-length episodes before trusting it.

**Output:** a data audit report and validated quality labels for every episode.

## Phase 2: Baseline model (weeks 4–5)

1. Handle mixed robots: normalize actions per dataset and pad to a common size, following LeRobot/SmolVLA conventions.
2. Pretrain on a random subset of the public data at a fixed, affordable size.
3. Post-train on LIBERO's public demonstrations.
4. Evaluate on LIBERO, closed-loop, multiple seeds.
5. **Training-pipeline control:** post-train the published `lerobot/smolvla_base` on LIBERO with exactly the same post-training settings as our baseline, and evaluate it the same way. If it lands near the published LIBERO numbers (see README for the paper vs community-reproduction gap), the training pipeline is verified. **If not, stop and debug before any Phase 3 curation runs.**

**Output:** our own robot foundation model with a real benchmark score, plus a verified training pipeline.

## Phase 3: Curation experiment (weeks 6–8)

Train at the same data size under different selection rules, everything else identical:

| Condition | Pretraining data |
|---|---|
| A | Random subset |
| B | Curated subset (flagged episodes removed) |
| C | Full uncurated pool (if compute allows) |

Same post-training, same evaluation, at least 3 seeds each.

- B beats A: curation helps at equal size.
- B matches C: curation lets us use less data for the same performance.
- No difference: a valid result, reported as such.

### Phase 3 design, fixed before any results (2026-10-07)

- **Conditions: A (random) vs B (curated), 3 seeds each** (6 models). C (full pool) and a source-matched random condition are deferred; add the source-matched condition only if A and B differ, to separate "removed bad episodes" from "changed the source mix" (see `docs/phase2_design.md`).
- **Filters**: only the Phase 1 validated ones: idle > 3 s at start or end; 8-frame failure judge P(yes) < 0.05 (not applied to taco_play).
- **Equal size**: A is drawn to the same number of frames as B (6.97M of the 12.2M-frame pool).
- **What a seed changes** (seeds 1000, 1001, 1002):
  - A: a *different* random episode subset (pairwise overlap ~57%) and the pretraining / post-training RNG. A's seed-to-seed spread therefore includes subset choice.
  - B: the same curated list for every seed; only pretraining / post-training RNG.
- **Training**: identical pretraining budget and post-training recipe for all 6 models.
- **Evaluation**: full 4-suite LIBERO protocol, evaluation seed fixed at 1000 for every model (paired: same initial states).
- **Primary outcome**: mean LIBERO success averaged over suites; report mean ± spread over seeds per condition, per-suite results, and the per-seed numbers. Null results are reported as such.
- **Cost**: ~40 GPU-hours per model (≈26 h pretraining, ≈10 h post-training, ≈4 h eval) → ~10 days on the 4070.

## Phase 4: Which filters matter (weeks 9–10)

1. Remove one failure type at a time to see which filter drives any improvement.
2. Vary filtering aggressiveness.
3. Check whether open-loop metrics predict closed-loop success.

## Phase 5: Write-up (weeks 11–12)

- **GitHub repo:** code, detectors, configs, one command to reproduce each result
- **Preprint:** follow-up to the demonstration-quality paper
- **Hugging Face release:** model weights and quality labels for the public datasets
- **Video:** curated vs uncurated rollouts side by side
- **Blog post** telling the story in plain language

## Risks

| Risk | Plan |
|---|---|
| Not enough compute | Measured in Phase 0. Shrink model or data, or rent cloud GPUs for pretraining only |
| Mixed datasets are hard to combine | Start with 2–3 similar datasets, expand later |
| Quality detectors are unreliable | Phase 1 manual validation catches this before it costs training time |
| Curation shows no benefit | Report it honestly; Phase 2 still yields a working model |
| Simulation-only results | State plainly that results are on LIBERO |
| Outside-work / IP constraints | Public code, public data, own hardware and time only |

## Rules

- Reproduce published numbers before trusting our own evaluation.
- Same data size, same post-training, and multiple seeds for every comparison.
- Fixed seeds, no cherry-picked rollouts in videos.
- Report null results as clearly as positive ones.

## Timeline

| Weeks | Milestone |
|---|---|
| 1 | Environment works, published numbers reproduced, compute estimated |
| 2–3 | Data audited, quality detectors validated |
| 4–5 | Baseline model, benchmarked |
| 6–8 | Curated vs random results |
| 9–10 | Which filters matter |
| 11–12 | Repo, preprint, model release, video |
