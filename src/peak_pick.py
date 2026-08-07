"""피크 검출. Otsu 임계(자유 파라미터 0) + 극대점 + 최소간격 필터.

v1 predictor.py peaks()와 threshold_survey.py otsu()에서 이식.
[D-21] 준수: 출력을 보고 임계를 고르지 않는다.
"""
from __future__ import annotations

import numpy as np

from config import MIN_EVENT_GAP_S


def otsu(x: np.ndarray, bins: int = 256) -> float:
    """Otsu 임계 — 자유 파라미터 0개.

    분포를 두 무리로 가장 잘 가르는 값. 무리 간 분산 최대화.
    v1 threshold_survey.py에서 그대로 이식.
    """
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
    """Otsu 임계로 온셋 피크를 검출한다.

    1. Otsu 임계 계산 (출력 무관, [D-21])
    2. 극대점 필터: env[i] >= env[i-1] AND env[i] > env[i+1] AND env[i] > threshold
    3. 진폭 내림차순 탐욕적 선택 + 최소간격 제약

    Returns:
        피크 시각 배열 (초), 시간순 정렬
    """
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


def peaks_with_mid(env_full: np.ndarray, env_mid: np.ndarray,
                   times: np.ndarray,
                   min_gap_s: float = MIN_EVENT_GAP_S) -> np.ndarray:
    """full-band + mid-band(120-2000Hz) 병합 탐지.

    full-band Otsu가 퍼커시브 구간에 의해 높아지면 좁은 대역의
    토널 온셋(피아노 등)을 놓친다. mid-band에서 독립적으로 Otsu를
    적용하여 토널 온셋을 보완한다.

    두 피크 집합을 합친 뒤 진폭 내림차순 탐욕 선택 + 최소간격.
    """
    pk_full = peaks(env_full, times, min_gap_s=min_gap_s)
    pk_mid = peaks(env_mid, times, min_gap_s=min_gap_s)

    time_to_idx: dict[float, int] = {}
    for i, t in enumerate(times):
        time_to_idx[t] = i

    candidates: list[tuple[float, float]] = []
    seen: set[float] = set()
    for t in pk_full:
        idx = time_to_idx.get(t)
        if idx is not None:
            candidates.append((t, env_full[idx]))
            seen.add(t)
    for t in pk_mid:
        if t not in seen:
            idx = time_to_idx.get(t)
            if idx is not None:
                candidates.append((t, env_mid[idx]))

    candidates.sort(key=lambda x: -x[1])
    selected: list[float] = []
    for t, _v in candidates:
        if all(abs(t - s) >= min_gap_s for s in selected):
            selected.append(t)

    if not selected:
        return np.array([], dtype=np.float64)
    return np.array(sorted(selected))
