# curated-vla

A small vision-language-action (VLA) model, pretrained only on public robot data, built to answer one question:

> **Does curating public robot data by failure type let a smaller dataset match or beat a larger, uncurated one?**

This extends earlier work on demonstration quality. The output is a trained model, validated quality labels for public datasets, and a controlled curation experiment evaluated closed-loop on LIBERO.

## Approach

| Decision | Choice |
|---|---|
| Framework | [Hugging Face LeRobot](https://github.com/huggingface/lerobot) |
| Architecture | SmolVLA-style: small VLM backbone + action expert (~450M params) |
| Starting point | Pretrained VLM, not robotics-pretrained weights |
| Pretraining data | Public LeRobot community datasets + a slice of Open X-Embodiment |
| Evaluation | LIBERO simulation benchmark, closed-loop, multiple seeds |

## Experiment

Same data size, same post-training, at least 3 seeds per condition:

| Condition | Pretraining data |
|---|---|
| A | Random subset |
| B | Curated subset (flagged episodes removed) |
| C | Full uncurated pool (if compute allows) |

Quality detectors target four failure modes: **truncation**, **idle time**, **flailing**, and **likely task failure** (VLM judge on final frames). Every detector is validated against ~200 hand-labeled episodes and checked for confounding with episode length.

## Status

| Phase | Weeks | Milestone | Status |
|---|---|---|---|
| 0 | 1 | Environment works, published SmolVLA LIBERO numbers reproduced, compute estimated | ⏳ |
| 1 | 2–3 | Data audited, quality detectors validated | ⏳ |
| 2 | 4–5 | Baseline model pretrained, post-trained, benchmarked | ⏳ |
| 3 | 6–8 | Curated vs random results | ⏳ |
| 4 | 9–10 | Per-filter ablations and dose-response | ⏳ |
| 5 | 11–12 | Repo, preprint, model + label release, video | ⏳ |

Full plan: [docs/PLAN.md](docs/PLAN.md)

## Layout

```
configs/    training and eval configs
curation/   quality detectors and episode labeling
eval/       LIBERO evaluation and result aggregation
scripts/    setup, environment checks, utilities
docs/       plan, audit reports, write-up
```

## Getting started

```bash
python scripts/check_env.py
```

## Ground rules

- Reproduce published numbers before trusting our own evaluation.
- Same data size, same post-training, multiple seeds for every comparison.
- Fixed seeds; no cherry-picked rollouts in videos.
- Null results are reported as clearly as positive ones.
- Results are on LIBERO simulation only; no real-robot claims.
