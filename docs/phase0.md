# Phase 0: Setup and reproduction

Status: in progress (started 2026-10-02). **Gate not yet passed**: our LIBERO numbers for the public SmolVLA checkpoint are below published ones; diagnosis below.

## Environment

| Item | Value |
|---|---|
| Host | Windows 11, Ryzen 5 7600 (6C/12T, has iGPU), 31 GB RAM |
| WSL2 | Ubuntu 24.04, 15.5 GB RAM visible (WSL default: half of host) |
| GPU | RTX 4070, 12 GB (≈10.8 GB free with desktop running), driver 610.62 |
| Python / PyTorch | 3.12.3 / 2.11.0+cu130 |
| LeRobot | 0.6.1 (`lerobot[smolvla,libero]`) |
| Simulator | MuJoCo 3.8.1, robosuite 1.4.0, hf-libero 0.1.4 |
| Rendering | `MUJOCO_GL=egl` → Mesa **llvmpipe (CPU)** by default; see below |

Setup: `bash scripts/setup_wsl.sh` (needs `sudo apt-get install -y ffmpeg git-lfs` once).

### Gotchas hit

- **LIBERO prompts on first import** for a dataset path, which crashes non-interactive evals with `EOFError`. `setup_wsl.sh` pre-writes `~/.libero/config.yaml`.
- **Async env workers cost ~2 GB RAM each.** `--eval.batch_size=10` with async envs OOM-killed WSL at 15.5 GB. Sync vector envs (`--eval.use_async_envs=false`) batch policy inference in one process and fit comfortably. Batching does not change results (task 0 of Object: 6/10 batched vs 5/10 serial).
- **EGL in WSL falls back to CPU rendering** (llvmpipe). Mesa's D3D12 driver renders on the GPU: `GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` (without the adapter name it picks the Ryzen iGPU). Not yet adopted for evals; pixel output may differ slightly from llvmpipe.
- **The published checkpoint replans every step** (`n_action_steps=1`, 10 flow-matching steps). This matches the paper's simulation protocol ("predicting a new action after each executed action").

## Reproduction target

| Source | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| SmolVLA (0.45B), paper Table 2 | 90 | 96 | 92 | 71 | 87.3 |
| Community, same checkpoint, LeRobot 0.5.1, MuJoCo 3.3.2, `n_action_steps=10` ([lerobot#3264](https://github.com/huggingface/lerobot/issues/3264)) | 63 | 93 | 81 | 56 | 73.3 |
| Ours, LeRobot 0.6.1, MuJoCo 3.8.1, `n_action_steps=1` (checkpoint default), seed 1000 | | 81 | | | |

The paper's own ablation (Table 13) reports ~80–83% average at 1–10 action steps, below the 87.3 headline, so the headline is likely not reachable from the public checkpoint. Our target is to match the community reproduction on the same checkpoint.

## Diagnosis log (LIBERO-Object, 100 episodes, seed 1000)

| Change from default | Success % | Notes |
|---|---|---|
| none (`n_action_steps=1`) | 81 | ~55 min |
| `n_action_steps=10` | 79 | not the cause; GPU idle, rendering-bound |
| MuJoCo 3.3.2 | running | |

Ruled out by inspection: images not flipped (rollout videos look correct, policy targets the right objects); batching.

Remaining suspects:
- **MuJoCo version** (3.8.1 vs community 3.3.2): contact physics changes across versions.
- **Render resolution**: LeRobot's LIBERO env rendered at 256×256 when this checkpoint was made (Sept 2025); since [0699b46d](https://github.com/huggingface/lerobot/commit/0699b46d) (Oct 2025) it renders at 360×360. Training data is 256×256. Test with `--env.observation_height=256 --env.observation_width=256`.
- LeRobot env changes after 0.5.1: fps plumbing (#4124), reset-on-termination (#4273), env lifecycle (#4194).

## Eval speed (RTX 4070, llvmpipe rendering)

| Setting | Full protocol (400 eps) |
|---|---|
| batch 1, `n_action_steps=1` | ~7.7 h |
| batch 10 sync, `n_action_steps=1` | ~4 h (inference-bound) |
| batch 10 sync, `n_action_steps=10` | ~2.8 h (render-bound) |

## Training throughput

TBD (GPU busy with eval diagnosis)

## Compute estimate

TBD
