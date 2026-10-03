"""Episode quality detectors from actions/states alone (no video).

Each detector returns a raw score per episode; flags are set by thresholds
calibrated *within a dataset* (robots differ in action units, scale and rate),
so scores are compared only against episodes of the same dataset.

Detectors (mapped to failure modes from the demonstration-quality paper):
  idle_start_s / idle_end_s : seconds spent motionless at the start / end
                              (operator not yet moving, or left recording after
                              finishing). Seconds, not fractions: fractions are
                              length-confounded (a fixed pause is a larger share
                              of a short episode).
  truncation                : the episode ends while still moving fast, i.e. cut
                              off mid-motion rather than settling. Median over the
                              tail, excluding the final frame, so a one-frame
                              glitch does not count.
  spike                     : largest single-step jump relative to its neighboring
                              steps: a recording glitch (e.g. state reset on the
                              last frame, seen in jaco_play). Relative to neighbors,
                              not the episode median, because multi-phase tasks
                              (fmb) have legitimately fast transport segments.
  jerk                      : mean normalized jerk over moving frames, i.e.
                              erratic, flailing control.

Motion is measured on a per-dataset trajectory (see `fit_spec`): the robot state
when it is informative, otherwise the integral of the (delta) action. Discrete
dimensions such as binary grippers are excluded: an open/close step is not motion
and otherwise reads as a teleport.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from curation.load import Episode

EPS = 1e-8
STILL = 0.1  # motion below this (in typical-step units) counts as stationary


@dataclass
class Spec:
    """How to turn one dataset's episodes into a normalized trajectory."""

    use_action: bool  # True: integrate actions because the state is unusable
    live: np.ndarray  # mask of continuous dimensions that actually move
    scale: np.ndarray  # typical per-step change of each live dimension
    angles: np.ndarray | None = None  # mask of radian angle dims that wrap at ±pi


def _raw(ep: Episode, use_action: bool, angles: np.ndarray | None = None) -> np.ndarray:
    x = ep.action if use_action else ep.state
    if angles is not None and angles.any():
        x = x.copy()
        x[:, angles] = np.unwrap(x[:, angles], axis=0)
    return x


def _wrapping_angles(episodes: list[Episode]) -> np.ndarray:
    """State dims that look like radian angles wrapping at ±pi (e.g. OXE EEF Euler angles).

    Bounded within ±(pi + 0.05) and showing single-step jumps larger than pi.
    Degree-valued joints (SO-100) exceed the bound and are left alone.
    """
    S = np.concatenate([e.state for e in episodes])
    jumps = np.concatenate([np.abs(np.diff(e.state, axis=0)) for e in episodes if len(e.state) > 1])
    return (np.abs(S).max(axis=0) <= np.pi + 0.05) & ((jumps > np.pi).mean(axis=0) > 0)


def fit_spec(episodes: list[Episode], sample: int = 500) -> Spec:
    sub = episodes[:: max(1, len(episodes) // sample)]
    use_action, angles = True, None
    if sub[0].state is not None and sub[0].state.shape[1] > 0:
        S = np.concatenate([e.state for e in sub])
        use_action = S.std(axis=0).max() < EPS  # all-constant state (e.g. zeros in roboturk)
        if not use_action:
            angles = _wrapping_angles(sub)
    X = np.concatenate([_raw(e, use_action, angles) for e in sub])
    if use_action:
        steps = np.abs(X)
    else:
        steps = np.abs(np.concatenate([np.diff(_raw(e, False, angles), axis=0) for e in sub]))
    live = np.zeros(X.shape[1], bool)
    scale = np.ones(X.shape[1])
    for j in range(X.shape[1]):
        if len(np.unique(np.round(X[:, j], 6))) <= 3:  # binary gripper, flags, constant
            continue
        nz = steps[:, j][steps[:, j] > EPS]
        if len(nz) > 0.05 * len(steps):  # changes in at least 5% of steps
            live[j], scale[j] = True, np.median(nz)
    return Spec(use_action, live, scale, angles)


def trajectory(ep: Episode, spec: Spec) -> np.ndarray:
    """(T, D_live) trajectory in units of the dataset's typical step."""
    x = _raw(ep, spec.use_action, spec.angles)[:, spec.live]
    if spec.use_action:
        x = np.cumsum(x, axis=0)
    return x / spec.scale[spec.live]


def motion_signal(traj: np.ndarray) -> np.ndarray:
    """Per-step motion magnitude (~1 = a typical step for this robot)."""
    if len(traj) < 2 or traj.shape[1] == 0:
        return np.zeros(len(traj))
    d = np.abs(np.diff(traj, axis=0))
    return np.concatenate([[0.0], np.median(d, axis=1) if d.shape[1] > 2 else d.mean(axis=1)])


def idle_fractions(motion: np.ndarray) -> tuple[float, float]:
    """Fraction of frames at the start / end before the first / after the last moving frame."""
    moving = np.flatnonzero(motion > STILL)
    n = len(motion)
    if n == 0 or len(moving) == 0:
        return 1.0, 1.0
    return moving[0] / n, (n - 1 - moving[-1]) / n


def _moving_ref(motion: np.ndarray) -> float:
    moving = motion[motion > STILL]
    return float(np.median(moving)) if len(moving) else EPS


def truncation_score(motion: np.ndarray, tail_frac: float = 0.05, min_tail: int = 3) -> float:
    """Median motion over the last few frames (excluding the final one) relative to
    the episode's median moving speed. Settled endings ~0; cut mid-motion >~1."""
    tail = max(min_tail, int(round(len(motion) * tail_frac)))
    return float(np.median(motion[-tail - 1 : -1]) / (_moving_ref(motion) + EPS))


def spike_score(motion: np.ndarray, k: int = 3) -> float:
    """Largest single-step motion relative to its neighbors (k steps either side).

    A recording glitch is a jump the adjacent steps do not share; a fast but smooth
    move (e.g. fmb's transport phase between slow insertions) has fast neighbors
    too. The floor (half the episode's median moving speed) stops tiny jitter
    during idle periods from producing huge ratios.
    """
    n = len(motion)
    if n < 2 * k + 1:
        return 0.0
    floor = 0.5 * _moving_ref(motion)
    best = 0.0
    for t in range(1, n):
        nb = np.concatenate([motion[max(1, t - k) : t], motion[t + 1 : t + 1 + k]])
        if len(nb):
            best = max(best, motion[t] / (np.median(nb) + floor + EPS))
    return float(best)


def jerk_score(traj: np.ndarray) -> float:
    """Mean |third difference| of the normalized trajectory over moving frames.

    Averaged (not summed) over moving frames only, so idle time and episode
    length do not inflate it directly; length confounding is still checked.
    """
    if len(traj) < 5 or traj.shape[1] == 0:
        return 0.0
    j = np.abs(np.diff(traj, n=3, axis=0)).mean(axis=1)
    v = np.abs(np.diff(traj, axis=0)).mean(axis=1)[2:]
    moving = v > STILL
    return float(j[moving].mean()) if moving.any() else 0.0


def score_dataset(episodes: list[Episode]) -> list[dict]:
    """Raw detector scores for every episode of one dataset."""
    spec = fit_spec(episodes)
    rows = []
    for ep in episodes:
        traj = trajectory(ep, spec)
        m = motion_signal(traj)
        idle_start, idle_end = idle_fractions(m)
        rows.append(
            dict(
                key=ep.key,
                source=ep.source,
                subset=ep.subset,
                episode_index=ep.episode_index,
                task=ep.task,
                length=ep.length,
                duration_s=ep.duration_s,
                fps=ep.fps,
                success_label=ep.success,
                motion_from="action" if spec.use_action else "state",
                live_dims=int(spec.live.sum()),
                idle_start_frac=idle_start,
                idle_end_frac=idle_end,
                idle_start_s=idle_start * ep.duration_s,
                idle_end_s=idle_end * ep.duration_s,
                truncation=truncation_score(m),
                spike=spike_score(m),
                jerk=jerk_score(traj),
            )
        )
    return rows
