"""무작위 배치 귀무 모형. 바닥 정의 [D-18]."""
from __future__ import annotations

import numpy as np

from config import WINDOW_S, MIN_EVENT_GAP_S

SEED = 20260807


def random_events(
    n_events: int,
    duration_s: float,
    min_gap_s: float = MIN_EVENT_GAP_S,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """n_events개를 [0, duration_s)에 균등 무작위 배치, min_gap 제약."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    if n_events == 0:
        return np.array([], dtype=np.float64)
    for _ in range(200):
        pts = np.sort(rng.uniform(0, duration_s, n_events))
        if len(pts) < 2 or np.min(np.diff(pts)) >= min_gap_s:
            return pts
    return np.sort(rng.uniform(0, duration_s, n_events))


def null_window_counts(
    n_events: int,
    duration_s: float,
    window_s: float = WINDOW_S,
    n_real: int = 1000,
    seed: int = SEED,
) -> np.ndarray:
    """귀무 분포: 윈도우별 계수의 (n_real, n_windows) 배열."""
    n_windows = int(np.ceil(duration_s / window_s))
    rng = np.random.default_rng(seed)
    out = np.zeros((n_real, n_windows), dtype=int)
    for r in range(n_real):
        pts = random_events(n_events, duration_s, rng=rng)
        for t in pts:
            w = min(int(t / window_s), n_windows - 1)
            out[r, w] += 1
    return out


def null_correlation(
    system_counts: np.ndarray,
    human_counts: np.ndarray,
    n_perm: int = 5000,
    seed: int = SEED,
) -> dict:
    """순열 검정: 시스템-인간 상관이 우연 이상인가.

    Returns:
        observed_rho, null_mean, null_std, p_value
    """
    from scipy.stats import spearmanr

    if len(system_counts) < 3:
        return {"error": "윈도우 3개 미만"}

    rho_obs, _ = spearmanr(system_counts, human_counts)
    rng = np.random.default_rng(seed)
    rho_null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(human_counts)
        rho_null[i], _ = spearmanr(system_counts, perm)
    p = float(np.mean(rho_null >= rho_obs))
    return {
        "observed_rho": round(float(rho_obs), 4),
        "null_mean": round(float(rho_null.mean()), 4),
        "null_std": round(float(rho_null.std()), 4),
        "p_value": round(p, 6),
    }
