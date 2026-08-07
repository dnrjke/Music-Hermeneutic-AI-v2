"""VN 3:00 전후 밀집 구간 진단.

구간별 탐지 수 + Otsu 임계 + 극대점 수 + 구조 현황 상세.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from config import audio_paths, MIN_EVENT_GAP_S, HOP, SR, WINDOW_S
from audio_io import load_mono, duration_s
from onset import superflux_envelope, band_envelopes
from peak_pick import peaks, otsu


BAND_NAMES = ["low", "mid", "high"]


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


for p in audio_paths("Viyella"):
    print(f"▸ {p.name}")
    mono = load_mono(p)
    dur = duration_s(p)
    env, times = superflux_envelope(mono)
    bands = band_envelopes(mono)

    pk_base = peaks(env, times)
    thr = otsu(env[np.isfinite(env)])
    print(f"  전역 Otsu: {thr:.4f}")
    print(f"  전역 피크: {len(pk_base)}")
    print()

    # 4초 윈도우별 상세 (2:30 ~ 4:30)
    print(f"  {'구간':>10s}  {'극대점':>5s}  {'Otsu초과':>7s}  {'Q1구조':>5s}  "
          f"{'SIR구조':>6s}  {'env_max':>8s}  {'env_med':>8s}  "
          f"{'low_med':>8s}  {'mid_med':>8s}  {'high_med':>8s}")

    for t0 in range(150, 270, 4):
        t1 = t0 + 4
        if t1 > dur:
            break
        mask = (times >= t0) & (times < t1)
        seg_env = env[mask]
        seg_times = times[mask]

        # 극대점
        n_lm = 0
        n_above = 0
        lm_indices = []
        for i in range(1, len(seg_env) - 1):
            if seg_env[i] >= seg_env[i-1] and seg_env[i] > seg_env[i+1]:
                n_lm += 1
                if seg_env[i] > thr:
                    n_above += 1

        base_in = pk_base[(pk_base >= t0) & (pk_base < t1)]

        # Q1 rescue count (이 윈도우)
        time_to_idx = {t: i for i, t in enumerate(times)}
        win_base = [t for t in pk_base if t0 <= t < t1]
        q1_rescued = 0
        if len(win_base) >= 2:
            profiles = []
            for t in win_base:
                i = time_to_idx[t]
                profiles.append(np.array([bands[b][i] for b in BAND_NAMES]))
            prototype = np.median(profiles, axis=0)
            sims = [cosine_sim(pf, prototype) for pf in profiles]
            sim_floor = float(np.percentile(sims, 25))

            full_mask_idx = np.where(mask)[0]
            for gi in full_mask_idx:
                if env[gi] > thr:
                    continue
                if gi > 0 and gi < len(env) - 1:
                    if env[gi] >= env[gi-1] and env[gi] > env[gi+1]:
                        pf = np.array([bands[b][gi] for b in BAND_NAMES])
                        if cosine_sim(pf, prototype) >= sim_floor:
                            q1_rescued += 1

        # SIR(u3) rescue count
        from _diag_spectral_identity import band_uniformity, band_coincidence, local_maxima_mask
        u3 = band_uniformity(bands, mode="u3")
        lm_all = local_maxima_mask(np.where(np.isfinite(env), env, -np.inf))
        u3_thr = otsu(u3[lm_all])
        coin = band_coincidence(bands)
        sir_rescued = 0
        for gi in np.where(mask)[0]:
            if env[gi] > thr:
                continue
            if lm_all[gi]:
                if u3[gi] >= u3_thr and coin[gi] >= 2:
                    sir_rescued += 1

        # 대역별 중앙값
        seg_low = bands["low"][mask]
        seg_mid = bands["mid"][mask]
        seg_high = bands["high"][mask]

        label = f"{t0//60}:{t0%60:02d}-{t1//60}:{t1%60:02d}"
        print(f"  {label:>10s}  {n_lm:5d}  {n_above:7d}  {q1_rescued:5d}  "
              f"{sir_rescued:6d}  {seg_env.max():8.3f}  {np.median(seg_env):8.3f}  "
              f"{np.median(seg_low):8.3f}  {np.median(seg_mid):8.3f}  {np.median(seg_high):8.3f}")

    # 밀집 구간(176-240s) vs 비밀집 구간 비교
    print(f"\n  --- 밀집 구간 vs 비밀집 구간 비교 ---")
    for label, s, e in [("밀집 176-240", 176, 240),
                        ("인트로 0-16", 0, 16),
                        ("킥 264-276", 264, 276)]:
        m = (times >= s) & (times < e)
        se = env[m]
        n_lm = 0
        n_above = 0
        for i in range(1, len(se) - 1):
            if se[i] >= se[i-1] and se[i] > se[i+1]:
                n_lm += 1
                if se[i] > thr:
                    n_above += 1
        base_c = int(((pk_base >= s) & (pk_base < e)).sum())
        print(f"  {label:20s}: 극대점={n_lm:5d}  Otsu초과={n_above:4d}  "
              f"base피크={base_c:4d}  env범위=[{se.min():.3f}, {se.max():.3f}]  "
              f"env중앙={np.median(se):.3f}")
