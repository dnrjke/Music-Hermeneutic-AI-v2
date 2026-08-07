"""피크 검출. Otsu 임계(자유 파라미터 0) + 극대점 + 최소간격 필터.

v1 predictor.py peaks()와 threshold_survey.py otsu()에서 이식.
[D-21] 준수: 출력을 보고 임계를 고르지 않는다.

세션 3에서 확정된 adaptive 파이프라인:
  block-gate(Otsu/norm) → Q1 rescue → 탐욕적 최소간격 선택
"""
from __future__ import annotations

import numpy as np

from config import HOP, SR, MIN_EVENT_GAP_S, WINDOW_S

BAND_NAMES = ("low", "mid", "high")


def otsu(x: np.ndarray, bins: int = 256) -> float:
    """Otsu 임계 — 자유 파라미터 0개."""
    h, e = np.histogram(x, bins=bins)
    c = 0.5 * (e[:-1] + e[1:])
    w = h.astype(np.float64) / max(h.sum(), 1)
    w0 = np.cumsum(w)
    m0 = np.cumsum(w * c)
    mt = m0[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        var_b = (mt * w0 - m0) ** 2 / (w0 * (1 - w0))
    var_b = np.nan_to_num(var_b, nan=-np.inf)
    return float(c[int(np.argmax(var_b))])


def peaks(envelope: np.ndarray, times: np.ndarray,
          min_gap_s: float = MIN_EVENT_GAP_S) -> np.ndarray:
    """Otsu 임계로 온셋 피크를 검출한다."""
    fin = np.isfinite(envelope)
    if fin.sum() < 3:
        return np.array([], dtype=np.float64)

    v = np.where(fin, envelope, -np.inf)
    thr = otsu(envelope[fin])

    loc = np.flatnonzero(
        (v[1:-1] >= v[:-2]) & (v[1:-1] > v[2:]) & (v[1:-1] > thr)
    ) + 1

    if len(loc) == 0:
        return np.array([], dtype=np.float64)

    out: list[int] = []
    for i in sorted(loc.tolist(), key=lambda i: -v[i]):
        if all(abs(times[i] - times[j]) >= min_gap_s for j in out):
            out.append(i)

    if not out:
        return np.array([], dtype=np.float64)
    return times[np.sort(np.asarray(out, dtype=int))]


# ── block-gated adaptive detection ──

def _local_norm(env: np.ndarray, times: np.ndarray,
                block_s: float = 16.0, pct: float = 99.0) -> np.ndarray:
    """블록별 자기 정규화."""
    frame_dt = HOP / SR
    out = np.zeros_like(env)
    dur = times[-1] + frame_dt
    n_blocks = int(np.ceil(dur / block_s))
    for bi in range(n_blocks):
        t0, t1 = bi * block_s, (bi + 1) * block_s
        mask = (times >= t0) & (times < t1)
        seg = env[mask]
        pos = seg[seg > 0]
        if len(pos) < 5:
            continue
        p = np.percentile(pos, pct)
        if p < 1e-12:
            continue
        out[mask] = np.clip(seg / p, 0, None)
    return out


def _greedy_select(candidates: list[tuple[float, float]],
                   min_gap_s: float) -> list[float]:
    """진폭 내림차순 탐욕적 최소간격 선택."""
    candidates.sort(key=lambda x: -x[1])
    selected: list[float] = []
    for t, _v in candidates:
        if all(abs(t - s) >= min_gap_s for s in selected):
            selected.append(t)
    return sorted(selected)


def peaks_adaptive(env: np.ndarray, times: np.ndarray,
                   bands: dict[str, np.ndarray],
                   duration_s: float,
                   block_s: float = 16.0,
                   min_gap_s: float = MIN_EVENT_GAP_S) -> np.ndarray:
    """block-gated adaptive detection + Q1 rescue.

    1. block-gate: visible 블록 → Otsu, invisible 블록 → local norm
    2. norm 잔여 합산: norm에만 있는 피크도 포함 (gate+norm잔여)
    3. Q1 prototype rescue: 윈도우 내 프로토타입 유사 피크 구조
    4. 탐욕적 최소간격 선택
    """
    fin = np.isfinite(env)
    if fin.sum() < 3:
        return np.array([], dtype=np.float64)

    global_thr = otsu(env[fin])
    time_to_idx = {t: i for i, t in enumerate(times)}

    # 1. base Otsu 피크
    pk_base = peaks(env, times, min_gap_s=min_gap_s)

    # 2. local norm 피크
    env_normed = _local_norm(env, times, block_s=block_s)
    pk_norm = peaks(env_normed, times, min_gap_s=min_gap_s)

    # 3. block-gate + norm 잔여 합산
    frame_dt = HOP / SR
    dur = times[-1] + frame_dt
    n_blocks = int(np.ceil(dur / block_s))

    gate_times: set[float] = set()
    for bi in range(n_blocks):
        t0, t1 = bi * block_s, (bi + 1) * block_s
        mask_t = (times >= t0) & (times < t1)
        seg = env[mask_t]
        pos = seg[seg > 0]
        block_pct = float(np.percentile(pos, 99.0)) if len(pos) >= 5 else 0.0

        if block_pct < global_thr:
            pks = pk_norm[(pk_norm >= t0) & (pk_norm < t1)]
        else:
            pks = pk_base[(pk_base >= t0) & (pk_base < t1)]
        for t in pks:
            gate_times.add(t)

    # norm 잔여: norm에만 있는 피크도 합산
    for t in pk_norm:
        if t not in gate_times:
            if all(abs(t - g) >= min_gap_s for g in gate_times):
                gate_times.add(t)

    # 4. Q1 prototype rescue
    all_lm_idx: list[int] = []
    v = np.where(fin, env, -np.inf)
    for i in range(1, len(v) - 1):
        if v[i] >= v[i - 1] and v[i] > v[i + 1]:
            all_lm_idx.append(i)

    rescued: set[float] = set()
    n_windows = int(np.ceil(duration_s / WINDOW_S))
    for wi in range(n_windows):
        wt0 = wi * WINDOW_S
        wt1 = wt0 + WINDOW_S
        win_base = [t for t in gate_times if wt0 <= t < wt1]
        if len(win_base) < 2:
            continue
        profiles = []
        for t in win_base:
            idx = time_to_idx.get(t)
            if idx is None:
                continue
            profiles.append(np.array([bands[b][idx] for b in BAND_NAMES]))
        if len(profiles) < 2:
            continue
        prototype = np.median(profiles, axis=0)
        sims = [_cosine(pf, prototype) for pf in profiles]
        sim_floor = float(np.percentile(sims, 25))
        wb = set(win_base)
        for i in all_lm_idx:
            t = times[i]
            if not (wt0 <= t < wt1):
                continue
            if t in wb:
                continue
            pf = np.array([bands[b][i] for b in BAND_NAMES])
            if _cosine(pf, prototype) >= sim_floor:
                rescued.add(t)

    # 5. 합산 + 탐욕적 선택
    all_times = gate_times | rescued
    candidates = []
    for t in all_times:
        idx = time_to_idx.get(t)
        if idx is not None:
            candidates.append((t, float(env[idx])))

    result = _greedy_select(candidates, min_gap_s)
    if not result:
        return np.array([], dtype=np.float64)
    return np.array(result)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
