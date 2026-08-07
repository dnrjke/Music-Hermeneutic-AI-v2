"""Morphological opening + Q1 prototype rescue 조합.

1단: opening 잔차 → 충격 성분 분리 (점진적 변화 제거)
2단: 잔차에서 Otsu 피크 (1차 탐지)
3단: 1차 피크의 대역 프로파일 → Q1 rescue (약한 충격 구조)
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from scipy.ndimage import grey_opening
from config import audio_paths, MIN_EVENT_GAP_S, HOP, SR, WINDOW_S
from audio_io import load_mono, duration_s
from onset import superflux_envelope, band_envelopes
from peak_pick import peaks, otsu


frame_dt = HOP / SR


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def impulse_q1_peaks(env_full, bands, times, dur,
                     kernel=5, window_s=WINDOW_S,
                     min_gap_s=MIN_EVENT_GAP_S):
    """opening 잔차 + Q1 prototype rescue."""
    # 1단: 충격 성분 분리
    opened = grey_opening(env_full, size=kernel)
    residual = np.maximum(env_full - opened, 0)

    # 2단: 잔차에서 1차 탐지
    pk_base = peaks(residual, times, min_gap_s=min_gap_s)
    if len(pk_base) < 1:
        return pk_base, 0

    time_to_idx = {t: i for i, t in enumerate(times)}
    band_names = ["low", "mid", "high"]
    thr = otsu(residual[np.isfinite(residual)])

    # 잔차의 극대점 전체
    all_lm = []
    for i in range(1, len(residual) - 1):
        if residual[i] >= residual[i-1] and residual[i] > residual[i+1]:
            all_lm.append(i)

    # 3단: 국소 프로토타입 Q1 rescue (대역 프로파일 기준)
    rescued_set = set()
    n_windows = int(np.ceil(dur / window_s))
    for wi in range(n_windows):
        t0, t1 = wi * window_s, (wi + 1) * window_s
        win_base = [t for t in pk_base if t0 <= t < t1]
        if len(win_base) < 2:
            continue

        profiles = []
        for t in win_base:
            i = time_to_idx[t]
            profiles.append(np.array([bands[b][i] for b in band_names]))
        prototype = np.median(profiles, axis=0)
        sims = [cosine_sim(p, prototype) for p in profiles]
        sim_floor = float(np.percentile(sims, 25))

        for i in all_lm:
            t = times[i]
            if not (t0 <= t < t1):
                continue
            if residual[i] > thr or t in set(win_base):
                continue
            p = np.array([bands[b][i] for b in band_names])
            sim = cosine_sim(p, prototype)
            if sim >= sim_floor:
                rescued_set.add(t)

    # 병합
    all_cands = [(t, residual[time_to_idx[t]]) for t in pk_base]
    for t in rescued_set:
        all_cands.append((t, residual[time_to_idx[t]]))
    all_cands.sort(key=lambda x: -x[1])

    selected = []
    for t, v in all_cands:
        if all(abs(t - s) >= min_gap_s for s in selected):
            selected.append(t)

    n_rescued = sum(1 for t in selected if t in rescued_set)
    return np.array(sorted(selected)), n_rescued


# 비교: 기존 / opening만 / Q1만 / opening+Q1
for p in audio_paths():
    print(f"\n{'='*60}")
    print(f"▸ {p.name}")
    dur = duration_s(p)
    mono = load_mono(p)
    env, times = superflux_envelope(mono)
    bands = band_envelopes(mono)

    pk_orig = peaks(env, times)

    # opening만 (k=5)
    opened = grey_opening(env, size=5)
    residual = np.maximum(env - opened, 0)
    pk_imp = peaks(residual, times)

    # Q1만 (이전 _diag_proto_q1.py 재현)
    from _diag_proto_q1 import local_proto_rescue
    pk_q1, _ = local_proto_rescue(env, bands, times, dur, sim_percentile=25)

    # opening + Q1
    pk_combo, n_rescued = impulse_q1_peaks(env, bands, times, dur, kernel=5)

    methods = [
        ("A-기존(Otsu)", pk_orig),
        ("B-opening(k=5)", pk_imp),
        ("C-Q1 rescue", pk_q1),
        ("D-opening+Q1", pk_combo),
    ]

    header = f"  {'방법':22s}: {'전체':>5s}"
    segs = [(0,4,"0-4"), (16,20,"16-20"), (56,60,"56-60"),
            (200,204,"200-204"), (268,272,"268-272")]
    for t0, t1, label in segs:
        if t1 <= dur:
            header += f"  {label:>7s}"
    print(header)

    for label, pk in methods:
        n = len(pk)
        line = f"  {label:22s}: {n:5d}"
        for t0, t1, _ in segs:
            if t1 <= dur:
                c = int(((pk >= t0) & (pk < t1)).sum())
                line += f"  {c:7d}"
        extra = ""
        if label == "D-opening+Q1":
            extra = f"  (rescued={n_rescued})"
        print(line + extra)
