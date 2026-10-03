# Phase 0: Setup and reproduction

Status: in progress (started 2026-10-02)

## Environment

| Item | Value |
|---|---|
| Host | Windows 11, Ryzen 5 7600 (6C/12T), 31 GB RAM |
| WSL2 | Ubuntu 24.04, 15.5 GB RAM visible (WSL default: half of host) |
| GPU | RTX 4070, 12 GB (≈10.8 GB free with desktop running), driver 610.62 |
| Python / PyTorch | 3.12.3 / 2.11.0+cu130 |
| LeRobot | 0.6.1 (`lerobot[smolvla,libero]`) |
| Rendering | `MUJOCO_GL=egl` |

Setup: `bash scripts/setup_wsl.sh` (needs `sudo apt-get install -y ffmpeg git-lfs` once).

### Gotchas hit

- **LIBERO prompts on first import** for a dataset path, which crashes non-interactive evals with `EOFError`. `setup_wsl.sh` pre-writes `~/.libero/config.yaml`.
- **Async env workers cost ~2 GB RAM each.** `--eval.batch_size=10` with async envs OOM-killed WSL at 15.5 GB. Sync vector envs (`--eval.use_async_envs=false`) batch policy inference in one process and fit comfortably.
- **The published checkpoint replans every step** (`n_action_steps=1`, 10 flow-matching steps). This matches the paper's simulation protocol ("predicting a new action after each executed action"), so it is kept; it makes inference the eval bottleneck.

## Reproduction target

SmolVLA paper ([arXiv 2506.01844](https://arxiv.org/abs/2506.01844)), Table 2, 10 trials per task:

| Model | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| SmolVLA (0.45B), paper | 90 | 96 | 92 | 71 | 87.3 |
| `HuggingFaceVLA/smolvla_libero`, ours | | | | | |

## Eval speed (RTX 4070)

| Setting | Sec / episode | Full protocol (400 eps) |
|---|---|---|
| batch 1 | ~70 | ~7.7 h |
| batch 10, sync envs | ~36 | ~4 h |

## Training throughput

TBD

## Compute estimate

TBD
