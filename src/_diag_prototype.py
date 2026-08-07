"""프로토타입 기반 rescue 진단.

1차 탐지(full-band Otsu)로 잡힌 피크의 대역 프로파일 원형을 구한다.
임계 미달 극대점 중 원형과 코사인 유사도가 높은 것을 구조한다.
구조 기준: 탐지된 피크들의 유사도 최솟값 이상 (자유 파라미터 0).
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from config import audio_paths, MIN_EVENT_GAP_S, HOP, SR
from audio_io import load_mono, duration_s
from onset import superflux_envelope, band_envelopes
from peak_pick import peaks, otsu


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def prototype_rescue(env_full, bands, times, min_gap_s=MIN_EVENT_GAP_S):
    """1차 탐지 프로파일 기반 2차 rescue."""
    pk_base = peaks(env_full, times, min_gap_s=min_gap_s)
    if len(pk_base) < 3:
        return pk_base, 0

    time_to_idx = {}
    for i, t in enumerate(times):
        time_to_idx[t] = i

    # 1차 피크의 대역 프로파일
    band_names = ["low", "mid", "high"]
    base_profiles = []
    base_sims = []
    for t in pk_base:
        i = time_to_idx[t]
        p = np.array([bands[b][i] for b in band_names])
        base_profiles.append(p)

    profiles_arr = np.array(base_profiles)
    prototype = np.median(profiles_arr, axis=0)

    # 탐지된 피크들의 prototype 유사도 → 최솟값 = 구조 기준
    for p in base_profiles:
        base_sims.append(cosine_sim(p, prototype))
    sim_floor = min(base_sims)

    # Otsu 임계
    thr = otsu(env_full[np.isfinite(env_full)])

    # 임계 미달 극대점 탐색
    candidates = []
    for i in range(1, len(env_full) - 1):
        if env_full[i] >= env_full[i-1] and env_full[i] > env_full[i+1]:
            if env_full[i] <= thr:
                p = np.array([bands[b][i] for b in band_names])
                sim = cosine_sim(p, prototype)
                if sim >= sim_floor:
                    candidates.append((times[i], env_full[i], sim))

    # 진폭 내림차순 탐욕 선택 (기존 피크와 합산)
    all_cands = [(t, env_full[time_to_idx[t]]) for t in pk_base]
    rescued_times = set()
    for t, v, s in candidates:
        all_cands.append((t, v))
        rescued_times.add(t)

    all_cands.sort(key=lambda x: -x[1])
    selected = []
    for t, v in all_cands:
        if all(abs(t - s) >= min_gap_s for s in selected):
            selected.append(t)

    n_rescued = sum(1 for t in selected if t in rescued_times)
    return np.array(sorted(selected)), n_rescued


# 진단
for p in audio_paths():
    print(f"\n{'='*60}")
    print(f"▸ {p.name}")
    dur = duration_s(p)
    mono = load_mono(p)
    env, times = superflux_envelope(mono)
    bands = band_envelopes(mono)

    pk_old = peaks(env, times)
    pk_new, n_rescued = prototype_rescue(env, bands, times)
    n_old, n_new = len(pk_old), len(pk_new)

    # prototype 정보
    time_to_idx = {t: i for i, t in enumerate(times)}
    band_names = ["low", "mid", "high"]
    profiles = []
    for t in pk_old:
        i = time_to_idx[t]
        profiles.append([bands[b][i] for b in band_names])
    proto = np.median(profiles, axis=0)
    sims = [cosine_sim(np.array(p), proto) for p in profiles]

    print(f"  prototype: low={proto[0]:.3f}  mid={proto[1]:.3f}  high={proto[2]:.3f}")
    print(f"  1차 피크 유사도: min={min(sims):.4f}  median={np.median(sims):.4f}  max={max(sims):.4f}")
    print(f"  기존: {n_old:5d}개 ({n_old/dur:.1f}/s)")
    print(f"  신규: {n_new:5d}개 ({n_new/dur:.1f}/s)  [구조: {n_rescued}]")
    print(f"  증가: {n_new-n_old:+d} ({(n_new/max(n_old,1)-1)*100:+.1f}%)")

    windows = [("0-4s", 0, 4), ("4-8s", 4, 8), ("16-20s", 16, 20),
               ("56-60s", 56, 60), ("100-104s", 100, 104),
               ("200-204s", 200, 204), ("268-272s", 268, 272)]
    for label, t0, t1 in windows:
        if t1 > dur:
            continue
        o = int(((pk_old >= t0) & (pk_old < t1)).sum())
        n = int(((pk_new >= t0) & (pk_new < t1)).sum())
        if o == n == 0:
            continue
        delta = f"Δ{n-o:+d}" if n != o else "  ="
        print(f"    {label:12s}: {o:3d} → {n:3d}  {delta}")
