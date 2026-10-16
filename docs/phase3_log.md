# Phase 3 run log

Operational record for the A-vs-B runs (design in [PLAN.md](PLAN.md); runner `scripts/run_condition.sh`,
queue `scripts/run_phase3.sh`, watchdog `scripts/watchdog.ps1`).

## 2026-10-09 → 10-15: storage failure and move to C

- **10-09 12:34** `A_random_seed1000` pretraining starts (WSL distro on the external D drive).
- **10-09 ~16:00** WSL shuts down at step ~8,000, before the first checkpoint (then every 20k steps).
  Lost: ~4 h of training. Fix: checkpoint every 5,000 steps, then every 1,000.
- **10-10 → 10-15** The D drive (WD Elements USB) logs bad blocks every ~30 min (149 in Oct 2–9, 382 in
  Oct 10–14). Each burst flips the distro's ext4 read-only and wedges WSL; training died every ~30 min and
  advanced 3,500 steps in 10 h despite a watchdog (relaunch every 5 min, WSL restart on I/O errors).
- **10-15** New distro `cvla` on C (`C:\WSL\cvla`, Ubuntu 24.04). Identical environment (Python 3.12.3,
  PyTorch 2.11.0+cu130, LeRobot 0.6.1, `scripts/setup_wsl.sh`). Copied from D while it still read:
  checkpoints (through step 8,000), pool index, conditions, labels, eval results, base models, LIBERO cache.
  `pool_video` could not be read off D (the bad sectors sit under it), so it was **rebuilt** with
  `scripts/rebuild_pool.py`: the same 25,803 video files, fetched at the Hub revisions pinned in the index
  (identical to `main` on 10-15), transcoded with the unchanged `pretrain/transcode.py`; 0 missing, 0 failed;
  `pretrain/check_pool.py` passes (per-source normalization, padding, decoding, 985 samples/s).
- **10-15 18:25** Training resumes on C from the step-8,000 checkpoint (weights, optimizer, scheduler, sampler
  position, RNG). First steps match the D run's logged losses at the same steps. Throughput 46–47 samples/s
  (34–44 on D).

**Effect on the experiment:** none on the training recipe or data. The resumed run consumes exactly the
sampler sequence it would have on D; only wall-clock was lost (~6 days). All conditions are trained on C.
