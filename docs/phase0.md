# Phase 0: Setup and reproduction

Status: complete (2026-10-02 → 2026-10-03), one eval seed.

## Summary

- **Eval pipeline matches an independent reproduction** of the public `HuggingFaceVLA/smolvla_libero` checkpoint: **72.2%** average vs the community's 73.3%. It does **not** reach the SmolVLA paper's 87.3%; the community run and the paper's own ablation table both indicate that gap lies in the public checkpoint vs paper, not in our eval.
- **One real eval bug found and fixed:** current LeRobot renders LIBERO cameras at 360×360, but the checkpoint and dataset are 256×256. Rendering at 256 raised LIBERO-Object from 81% to 88–90%.
- **Training:** 54 samples/s at batch 32 (fp32, VLM frozen) on the 4070; batch 64 does not fit; bf16 AMP is broken in LeRobot 0.6.1.
- **Compute:** paper-scale pretraining (~11 days per run) is not feasible locally; a 5M-sample pretraining budget (~26 h per run, ~53 GPU-hours per curation condition) is.

**Gate decision (2026-10-03): passed.** Matching the community reproduction verifies the evaluation harness, which is what the gate is for. Conditions: (1) README states the paper vs community gap; (2) the training pipeline is verified by a short sanity run now and by a `smolvla_base` LIBERO post-training control in Phase 2, with a stop-and-debug rule before Phase 3.

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
- **Sync batch 10 on LIBERO-Long OOM-kills WSL.** 10 LIBERO envs in one process, each with its own MuJoCo instance and llvmpipe renderer, reach ~15 GB on Long's larger scenes (~12 GB on Object). `run_protocol.sh` now runs each task in its own process, and Long uses `BATCH_SIZE=5` (peak ~9.5 GB). Raising the WSL memory cap in `%USERPROFILE%\.wslconfig` (host has 31 GB) would allow batch 10 again.
- **Async env workers cost ~2 GB RAM each.** `--eval.batch_size=10` with async envs OOM-killed WSL at 15.5 GB. Sync vector envs (`--eval.use_async_envs=false`) batch policy inference in one process and fit comfortably. Batching does not change results (task 0 of Object: 6/10 batched vs 5/10 serial).
- **EGL in WSL falls back to CPU rendering** (llvmpipe). Mesa's D3D12 driver renders on the GPU: `GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` (without the adapter name it picks the Ryzen iGPU). Not yet adopted for evals; pixel output may differ slightly from llvmpipe.
- **The published checkpoint replans every step** (`n_action_steps=1`, 10 flow-matching steps). This matches the paper's simulation protocol ("predicting a new action after each executed action").

## Reproduction target

| Source | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| SmolVLA (0.45B), paper Table 2 | 90 | 96 | 92 | 71 | 87.3 |
| Community, same checkpoint, LeRobot 0.5.1, MuJoCo 3.3.2, `n_action_steps=10` ([lerobot#3264](https://github.com/huggingface/lerobot/issues/3264)) | 63 | 93 | 81 | 56 | 73.3 |
| **Ours**, LeRobot 0.6.1, render 256×256, `n_action_steps=1` (checkpoint default), seed 1000 | **75** | **90** | **78** | **46** | **72.2** |

Per-suite vs community: Spatial +12, Object −3, Goal −3, Long −10 (binomial SE ≈ 4–5 pp per suite at n=100, so only Spatial and Long differ by more than ~2 SE, in opposite directions). Long was run at batch 5 (OOM at batch 10, see above); other suites at batch 10.

The paper's own ablation (Table 13) reports ~80–83% average at 1–10 action steps, below the 87.3 headline, so the headline is likely not reachable from the public checkpoint.

Per-task success (%), seed 1000:

| Suite | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Object | 70 | 100 | 90 | 100 | 90 | 70 | 100 | 90 | 100 | 90 |
| Spatial | 60 | 80 | 100 | 70 | 70 | 80 | 70 | 90 | 60 | 70 |
| Goal | 90 | 90 | 90 | 40 | 90 | 80 | 80 | 100 | 80 | 40 |
| Long | 10 | 40 | 50 | 100 | 0 | 100 | 70 | 30 | 30 | 30 |

Reproduce: `bash eval/run_protocol.sh HuggingFaceVLA/smolvla_libero 1000` (set `BATCH_SIZE=5` unless WSL has >16 GB RAM).

## Diagnosis log (LIBERO-Object, 100 episodes, seed 1000)

| Change from default | Success % | Notes |
|---|---|---|
| none (`n_action_steps=1`) | 81 | ~55 min |
| `n_action_steps=10` | 79 | not the cause; GPU idle, rendering-bound |
| MuJoCo 3.3.2 | 84 | within noise of 81 (binomial SE ≈ 4 pp at n=100) |
| render 256×256 | **88** | largest single effect; matches training-data resolution |
| MuJoCo 3.3.2 + render 256×256 | 88 | MuJoCo version has no effect |

**Conclusion:** render resolution was the main cause. At 256×256 we get 88% on Object vs the community's 93% (difference ≈ 1.2 SE, not significant). `eval/eval_libero.sh` now renders at 256×256 by default. The paper's 96% remains ~2.4 SE above us; consistent with the paper's own Table 13 showing lower numbers than its headline table.

Ruled out by inspection: images not flipped (rollout videos look correct, policy targets the right objects); batching.

Background on suspects:
- **Render resolution**: LeRobot's LIBERO env rendered at 256×256 when this checkpoint was made (Sept 2025); since [0699b46d](https://github.com/huggingface/lerobot/commit/0699b46d) (Oct 2025) it renders at 360×360 (also in 0.5.1). Training data is 256×256.
- **MuJoCo version** (3.8.1 vs community 3.3.2): contact physics changes across versions.
- LeRobot env changes after 0.5.1: fps plumbing (#4124, default 30 → 20), reset-on-termination (#4273), env lifecycle (#4194). A LeRobot 0.5.1 + MuJoCo 3.3.2 environment is built for an exact community-config comparison.

## Eval speed (RTX 4070, llvmpipe rendering)

| Setting | Full protocol (400 eps) |
|---|---|
| batch 1, `n_action_steps=1` | ~7.7 h |
| batch 10 sync, `n_action_steps=1` | ~4 h (inference-bound) |
| batch 10 sync, `n_action_steps=10` | ~2.8 h (render-bound) |

## Training throughput

`scripts/measure_throughput.sh`: SmolVLA from `SmolVLM2-500M-Instruct` weights (`--policy.type=smolvla --policy.load_vlm_weights=true`), default `train_expert_only=true` (100M of 450M params trainable), on `lerobot/libero` @ `a1aaacb`, 8 dataloader workers, pyav decoding.

| Setting | Samples/s | Torch mem | Notes |
|---|---|---|---|
| batch 32, fp32 | **54** | 6.7 GB | data_s ≈ 0.005 s/step: GPU-bound, not data-bound |
| batch 64, fp32 | ~12 | >12 GB | spills past VRAM into shared system memory; unusable |
| `--policy.use_amp=true` | n/a | | crashes in LeRobot 0.6.1: `_amp_foreach_non_finite_check_and_unscale_cuda not implemented for 'BFloat16'` |

So the local ceiling is **~54 samples/s ≈ 4.7M samples/day**. Fixing AMP (bf16) is the obvious speedup to try in Phase 2.

## Compute estimate

| Job | Samples | Local (4070) |
|---|---|---|
| SmolVLA paper pretraining (200k steps × 256) | 51.2M | ~11 days per run: **not feasible locally** |
| SmolVLA paper LIBERO post-training (100k × 64) | 6.4M | ~33 h |
| Reduced post-training (30k × 32) | ~1M | ~5 h |
| Proposed pretraining budget per condition | 5M (~0.5 epoch of the 10.6M-frame community pool) | ~26 h |
| LIBERO eval, full protocol, one seed | 400 episodes | ~4 h |

Per curation condition (1 pretrain + 3 post-train seeds + 3 evals): ~26 + 15 + 12 ≈ **53 GPU-hours**. Conditions A and B ≈ 4.5 days; adding C ≈ 6.5 days of uninterrupted GPU time; Phase 4 ablations add ~2 days each.

**Recommendation:** keep everything local at the 5M-sample pretraining budget; that fits the Weeks 4–8 schedule. Rent cloud GPUs only if we want a larger pretraining budget (paper scale on a single H100 would be roughly 1.5–2 days per run, ≈ $100–150 per run at current on-demand prices; verify before booking). Before committing, try to (a) fix bf16 AMP and (b) move rendering to the GPU (`GALLIUM_DRIVER=d3d12`) to cut eval time.
