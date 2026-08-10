"""Upward leveler for weak residual frames (strong frames untouched).

Fixed machinery: RMS 2048/256, target T = 2s-block p99.
g = clip(T / max(e, eps), 1, g_max) with optional floor / Otsu-split.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from io_util import HOP, N_FFT, SR
from peak_pick import otsu as peak_otsu

UPWARD_PARAMS = {
    "rms_win": N_FFT,
    "rms_hop": HOP,
    "norm_block_s": 2.0,
    "eps": 1e-12,
}

# (variant_key, g_max or None, floor_pct or None, otsu_split)
# floor_pct: if set, frames with e < block percentile of positive → g=1
UPWARD_VARIANTS: tuple[tuple[str, float | None, float | None, bool], ...] = (
    ("up_open", None, None, False),       # no cap, no floor — include tiny e
    ("up_g4", 4.0, None, False),
    ("up_g10", 10.0, None, False),
    ("up_open_p05", None, 5.0, False),    # careful floor: block p5
    ("up_open_p01", None, 1.0, False),    # milder floor: block p1
    ("up_otsu_open", None, None, True),   # boost only e <= Otsu; no floor
    ("up_otsu_g10", 10.0, None, True),
)


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


def _interp_to_samples(
    frame_values: np.ndarray, n_samples: int, hop: int, win: int = N_FFT
) -> np.ndarray:
    frame_values = np.asarray(frame_values, dtype=np.float64)
    if frame_values.size == 0:
        return np.ones(n_samples, dtype=np.float64)
    sample_pos = np.arange(len(frame_values), dtype=np.float64) * hop + 0.5 * win
    target = np.arange(n_samples, dtype=np.float64)
    return np.interp(target, sample_pos, frame_values).astype(np.float64)


def frame_upward_gains(
    env: np.ndarray,
    *,
    hop: int = HOP,
    block_s: float = UPWARD_PARAMS["norm_block_s"],
    g_max: float | None = None,
    floor_pct: float | None = None,
    floor_rel: float | None = None,
    otsu_split: bool = False,
    eps: float = UPWARD_PARAMS["eps"],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Per-frame gain >= 1 for weak frames; strong frames stay at 1.

    floor_pct: skip boost if e < block percentile of positive.
    floor_rel: skip boost if e < floor_rel * T (T = block p99).
    """
    env = np.asarray(env, dtype=np.float64)
    block = max(1, int(round(block_s * (SR / hop))))
    gains = np.ones_like(env, dtype=np.float64)

    otsu_thr = float(peak_otsu(env[np.isfinite(env)])) if otsu_split and env.size else None

    n_boost = 0
    n_floor_skip = 0
    n_strong_skip = 0
    g_applied: list[float] = []

    for start in range(0, len(env), block):
        end = min(start + block, len(env))
        seg = env[start:end]
        positive = seg[seg > 0]
        if positive.size == 0:
            T = 1.0
            floor_pct_val = 0.0
        else:
            T = float(np.percentile(positive, 99))
            if T < eps:
                T = float(np.max(positive))
            floor_pct_val = (
                float(np.percentile(positive, floor_pct))
                if floor_pct is not None
                else 0.0
            )
        floor_rel_val = (float(floor_rel) * T) if floor_rel is not None else 0.0
        floor = max(floor_pct_val, floor_rel_val)

        for i in range(start, end):
            e = float(env[i])
            if otsu_thr is not None and e > otsu_thr:
                n_strong_skip += 1
                continue
            if (floor_pct is not None or floor_rel is not None) and e < floor:
                n_floor_skip += 1
                continue
            denom = e if e > eps else eps
            g = T / denom
            if g < 1.0:
                n_strong_skip += 1
                continue
            if g_max is not None:
                g = min(g, float(g_max))
            gains[i] = g
            n_boost += 1
            g_applied.append(g)

    meta = {
        "n_frames": int(len(env)),
        "n_boost": int(n_boost),
        "n_floor_skip": int(n_floor_skip),
        "n_strong_skip": int(n_strong_skip),
        "boost_frac": float(n_boost / max(len(env), 1)),
        "g_max_param": g_max,
        "floor_pct": floor_pct,
        "floor_rel": floor_rel,
        "otsu_split": otsu_split,
        "otsu_thr": otsu_thr,
        "g_mean_boosted": float(np.mean(g_applied)) if g_applied else 1.0,
        "g_p95_boosted": float(np.percentile(g_applied, 95)) if g_applied else 1.0,
        "g_max_applied": float(np.max(g_applied)) if g_applied else 1.0,
    }
    return gains, meta


def upward_level_stereo(
    stereo: np.ndarray,
    *,
    g_max: float | None = None,
    floor_pct: float | None = None,
    floor_rel: float | None = None,
    otsu_split: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply upward gains derived from mono-mean RMS env to both channels."""
    y = np.asarray(stereo, dtype=np.float32)
    win = UPWARD_PARAMS["rms_win"]
    hop = UPWARD_PARAMS["rms_hop"]
    mono = y.mean(axis=1)
    env = _frame_rms_mono(mono, win, hop)
    gains_f, meta = frame_upward_gains(
        env,
        hop=hop,
        block_s=UPWARD_PARAMS["norm_block_s"],
        g_max=g_max,
        floor_pct=floor_pct,
        floor_rel=floor_rel,
        otsu_split=otsu_split,
        eps=UPWARD_PARAMS["eps"],
    )
    g_samp = _interp_to_samples(gains_f, y.shape[0], hop, win).astype(np.float32)
    out = (y * g_samp[:, None]).astype(np.float32)
    meta = {
        **meta,
        "params": dict(UPWARD_PARAMS),
        "gain_sample_mean": float(np.mean(g_samp)),
        "gain_sample_max": float(np.max(g_samp)),
    }
    return out, meta
