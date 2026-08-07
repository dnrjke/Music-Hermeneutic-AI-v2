"""Q1 rescue → opening 필터.

1단: 기존 Otsu 1차 탐지
2단: Q1 prototype rescue (약한 충격 구조)
3단: opening 잔차에서 극대점이 아닌 것 제거 (점진적 변화 필터)
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


def q1_rescue_then_filter(env_full, bands, times, dur,
                          kernel=5, window_s=WINDOW_S,
                          min_gap_s=MIN_EVENT_GAP_S):
    """Q1 rescue → opening 필터."""
    # 1단: 기존 Otsu 1차 탐지
    pk_base = peaks(env_full, times, min_gap_s=min_gap_s)
    if len(pk_base) < 1:
        return pk_base, 0, 0

    time_to_idx = {t: i for i, t in enumerate(times)}
    band_names = ["low", "mid", "high"]
    thr = otsu(env_full[np.isfinite(env_full)])

    # 극대점 전체
    all_lm_idx = set()
    for i in range(1, len(env_full) - 1):
        if env_full[i] >= env_full[i-1] and env_full[i] > env_full[i+1]:
            all_lm_idx.add(i)

    # 2단: Q1 prototype rescue
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

        for i in all_lm_idx:
            t = times[i]
            if not (t0 <= t < t1):
                continue
            if env_full[i] > thr or t in set(win_base):
                continue
            p = np.array([bands[b][i] for b in band_names])
            sim = cosine_sim(p, prototype)
            if sim >= sim_floor:
                rescued_set.add(t)

    # 병합 (Q1 rescue 결과)
    all_cands = [(t, env_full[time_to_idx[t]]) for t in pk_base]
    for t in rescued_set:
        all_cands.append((t, env_full[time_to_idx[t]]))
    all_cands.sort(key=lambda x: -x[1])

    pre_filter = []
    for t, v in all_cands:
        if all(abs(t - s) >= min_gap_s for s in pre_filter):
            pre_filter.append(t)

    n_pre = len(pre_filter)

    # 3단: opening 필터 — 잔차에서 극대점인 것만 유지
    opened = grey_opening(env_full, size=kernel)
    residual = np.maximum(env_full - opened, 0)

    # 잔차 극대점 집합
    res_lm = set()
    for i in range(1, len(residual) - 1):
        if residual[i] >= residual[i-1] and residual[i] > residual[i+1]:
            if residual[i] > 0:
                res_lm.add(times[i])

    # 1차 피크(Otsu 초과)는 무조건 유지, 구조된 것만 필터 적용
    base_set = set(pk_base.tolist())
    filtered = []
    n_filtered_out = 0
    for t in sorted(pre_filter):
        if t in base_set:
            filtered.append(t)
        elif t in res_lm:
            filtered.append(t)
        else:
            n_filtered_out += 1

    n_rescued = sum(1 for t in filtered if t in rescued_set)
    return np.array(filtered), n_rescued, n_filtered_out


for p in audio_paths():
    print(f"\n{'='*60}")
    print(f"▸ {p.name}")
    dur = duration_s(p)
    mono = load_mono(p)
    env, times = superflux_envelope(mono)
    bands = band_envelopes(mono)

    pk_orig = peaks(env, times)

    # 다양한 커널로 필터 효과 비교
    segs = [(0,4,"0-4"), (16,20,"16-20"), (56,60,"56-60"),
            (200,204,"200-204"), (268,272,"268-272")]

    header = f"  {'방법':28s}: {'전체':>5s}"
    for t0, t1, label in segs:
        if t1 <= dur:
            header += f"  {label:>7s}"
    print(header)

    # A: 기존
    n = len(pk_orig)
    line = f"  {'A-기존(Otsu)':28s}: {n:5d}"
    for t0, t1, _ in segs:
        if t1 <= dur:
            c = int(((pk_orig >= t0) & (pk_orig < t1)).sum())
            line += f"  {c:7d}"
    print(line)

    # B: Q1만
    # (inline)
    from _diag_proto_q1 import local_proto_rescue
    pk_q1, _ = local_proto_rescue(env, bands, times, dur, sim_percentile=25)
    n = len(pk_q1)
    line = f"  {'B-Q1 rescue':28s}: {n:5d}"
    for t0, t1, _ in segs:
        if t1 <= dur:
            c = int(((pk_q1 >= t0) & (pk_q1 < t1)).sum())
            line += f"  {c:7d}"
    print(line)

    # C, D: Q1 → opening 필터
    for k in [5, 7]:
        pk, n_r, n_fo = q1_rescue_then_filter(env, bands, times, dur, kernel=k)
        n = len(pk)
        label = f"C-Q1→filter(k={k})"
        line = f"  {label:28s}: {n:5d}"
        for t0, t1, _ in segs:
            if t1 <= dur:
                c = int(((pk >= t0) & (pk < t1)).sum())
                line += f"  {c:7d}"
        print(f"{line}  (rescued={n_r}, filtered={n_fo})")
