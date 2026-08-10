"""Percussive progressive refine: soft env gate + attack/release."""
from __future__ import annotations

import numpy as np

from io_util import HOP, N_FFT, SR

PERC_REFINE_PARAMS = {
    "rms_win": N_FFT,
    "rms_hop": HOP,
    "norm_block_s": 2.0,
    "attack_ms": 5.0,
    "release_ms": 40.0,
    "eps": 1e-12,
}


def _otsu(x: np.ndarray, bins: int = 256) -> float:
    """Otsu threshold on 1D samples (same spirit as src/peak_pick.otsu)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise RuntimeError("Otsu: empty input")
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return lo
    hist, edges = np.histogram(x, bins=bins, range=(lo, hi))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return lo
    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * (edges[:-1] + edges[1:]) * 0.5)
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    idx = int(np.nanargmax(sigma_b))
    return float(0.5 * (edges[idx] + edges[idx + 1]))


def _frame_rms_mono(mono: np.ndarray, win: int, hop: int) -> np.ndarray:
    x = np.asarray(mono, dtype=np.float64)
    if len(x) < win:
        return np.array([np.sqrt(np.mean(x * x))], dtype=np.float64)
    n_frames = 1 + (len(x) - win) // hop
    out = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        s = i * hop
        seg = x[s : s + win]
        out[i] = np.sqrt(np.mean(seg * seg))
    return out


def _stereo_mean_rms(stereo: np.ndarray, win: int, hop: int) -> np.ndarray:
    y = np.asarray(stereo, dtype=np.float32)
    mid = y.mean(axis=1)
    return _frame_rms_mono(mid, win, hop)


def _block_p99_norm(env: np.ndarray, sr_frames: float, block_s: float) -> np.ndarray:
    """Per-block 99-pct normalize, then clip to [0, 1]."""
    env = np.asarray(env, dtype=np.float64)
    block = max(1, int(round(block_s * sr_frames)))
    out = np.zeros_like(env)
    for start in range(0, len(env), block):
        end = min(start + block, len(env))
        seg = env[start:end]
        positive = seg[seg > 0]
        if positive.size == 0:
            scale = 1.0
        else:
            scale = float(np.percentile(positive, 99))
            if scale < PERC_REFINE_PARAMS["eps"]:
                scale = 1.0
        out[start:end] = np.clip(seg / scale, 0.0, 1.0)
    return out


def _interp_to_samples(frame_values: np.ndarray, n_samples: int, hop: int) -> np.ndarray:
    """Linear upsample frame series to sample rate (hop spacing)."""
    if frame_values.size == 0:
        return np.zeros(n_samples, dtype=np.float64)
    # Frame centers at hop/2, hop + hop/2, ...
    frame_idx = np.arange(frame_values.size, dtype=np.float64)
    sample_pos = frame_idx * hop + (PERC_REFINE_PARAMS["rms_win"] * 0.5)
    target = np.arange(n_samples, dtype=np.float64)
    # Extrapolate with edge values
    return np.interp(target, sample_pos, frame_values).astype(np.float64)


def _attack_release_follower(
    env_samples: np.ndarray, attack_ms: float, release_ms: float, sr: int
) -> np.ndarray:
    att = np.exp(-1.0 / max(sr * attack_ms / 1000.0, 1.0))
    rel = np.exp(-1.0 / max(sr * release_ms / 1000.0, 1.0))
    out = np.empty_like(env_samples)
    level = 0.0
    for i, v in enumerate(env_samples):
        coef = att if v > level else rel
        level = coef * level + (1.0 - coef) * v
        out[i] = level
    return out


def soft_env_gate(stereo: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """A1: Otsu soft mask on 2s-p99-normalized RMS envelope."""
    y = np.asarray(stereo, dtype=np.float32)
    win = PERC_REFINE_PARAMS["rms_win"]
    hop = PERC_REFINE_PARAMS["rms_hop"]
    block_s = PERC_REFINE_PARAMS["norm_block_s"]

    env = _stereo_mean_rms(y, win, hop)
    sr_frames = SR / hop
    env_norm = _block_p99_norm(env, sr_frames, block_s)
    positive = env_norm[env_norm > 0]
    if positive.size < 8:
        raise RuntimeError("soft_env_gate: insufficient positive env for Otsu")
    thr = _otsu(positive)
    if thr <= 0:
        thr = float(np.median(positive)) if positive.size else 1.0
    mask_frames = np.clip(env_norm / thr, 0.0, 1.0)
    mask = _interp_to_samples(mask_frames, y.shape[0], hop).astype(np.float32)
    mask_st = np.column_stack([mask, mask])
    gated = (y * mask_st).astype(np.float32)
    removed = (y * (1.0 - mask_st)).astype(np.float32)
    meta = {
        "otsu_thr": float(thr),
        "mask_mean": float(mask.mean()),
        "mask_p50": float(np.percentile(mask, 50)),
        "mask_p95": float(np.percentile(mask, 95)),
    }
    return gated, removed, meta


def attack_release(stereo: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """A2: attack/release follower gain vs local 2s p99.

    Follower runs on the hop-grid RMS envelope (then linearly upsampled),
    so attack/release times map to the same physical ms on the hop clock.
    """
    y = np.asarray(stereo, dtype=np.float32)
    win = PERC_REFINE_PARAMS["rms_win"]
    hop = PERC_REFINE_PARAMS["rms_hop"]
    block_s = PERC_REFINE_PARAMS["norm_block_s"]
    eps = PERC_REFINE_PARAMS["eps"]

    env = _stereo_mean_rms(y, win, hop)
    # Effective sample rate of the envelope series
    env_sr = SR / hop
    follower = _attack_release_follower(
        env,
        PERC_REFINE_PARAMS["attack_ms"],
        PERC_REFINE_PARAMS["release_ms"],
        int(round(env_sr)),
    )

    block = max(1, int(round(block_s * env_sr)))
    local_p99 = np.empty_like(follower)
    for start in range(0, len(follower), block):
        end = min(start + block, len(follower))
        seg = follower[start:end]
        positive = seg[seg > 0]
        scale = float(np.percentile(positive, 99)) if positive.size else 1.0
        if scale < eps:
            scale = 1.0
        local_p99[start:end] = scale

    gain_frames = np.clip(follower / (local_p99 + eps), 0.0, 1.0)
    gain = _interp_to_samples(gain_frames, y.shape[0], hop).astype(np.float32)
    gain_st = np.column_stack([gain, gain])
    shaped = (y * gain_st).astype(np.float32)
    removed = (y * (1.0 - gain_st)).astype(np.float32)
    meta = {
        "attack_ms": PERC_REFINE_PARAMS["attack_ms"],
        "release_ms": PERC_REFINE_PARAMS["release_ms"],
        "follower_grid": "rms_hop",
        "gain_mean": float(gain.mean()),
        "gain_p50": float(np.percentile(gain, 50)),
        "gain_p95": float(np.percentile(gain, 95)),
    }
    return shaped, removed, meta
