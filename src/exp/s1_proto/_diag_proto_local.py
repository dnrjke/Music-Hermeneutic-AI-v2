"""국소 프로토타입 rescue 진단.

각 윈도우(4초)에서:
1. 탐지된 피크의 대역 프로파일 → 국소 원형
2. 임계 미달 극대점 중 원형과 닮은 것 구조
3. 유사도 기준 = 해당 윈도우 탐지 피크의 유사도 최솟값
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from config import audio_paths, MIN_EVENT_GAP_S, WINDOW_S
from audio_io import load_mono, duration_s
from onset import superflux_envelope, band_envelopes
from peak_pick import peaks, otsu


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def local_prototype_rescue(env_full, bands, times, dur,
                           window_s=WINDOW_S,
                           min_gap_s=MIN_EVENT_GAP_S):
    pk_base = peaks(env_full, times, min_gap_s=min_gap_s)
    if len(pk_base) < 1:
        return pk_base, 0, {}

    time_to_idx = {t: i for i, t in enumerate(times)}
    band_names = ["low", "mid", "high"]
    thr = otsu(env_full[np.isfinite(env_full)])

    # 전체 극대점 (임계 무관)
    all_lm = []
    for i in range(1, len(env_full) - 1):
        if env_full[i] >= env_full[i-1] and env_full[i] > env_full[i+1]:
            all_lm.append(i)

    rescued_set = set()
    debug_info = {}

    n_windows = int(np.ceil(dur / window_s))
    for wi in range(n_windows):
        t0, t1 = wi * window_s, (wi + 1) * window_s

        # 이 윈도우의 1차 피크
        win_base = [t for t in pk_base if t0 <= t < t1]
        if len(win_base) < 2:
            continue

        # 국소 프로파일 → 원형
        profiles = []
        for t in win_base:
            i = time_to_idx[t]
            profiles.append(np.array([bands[b][i] for b in band_names]))
        prototype = np.median(profiles, axis=0)

        # 윈도우 내 탐지 피크의 유사도 최솟값
        sims = [cosine_sim(p, prototype) for p in profiles]
        sim_floor = min(sims)

        # 임계 미달 극대점 구조
        win_rescued = []
        for i in all_lm:
            t = times[i]
            if not (t0 <= t < t1):
                continue
            if env_full[i] > thr:
                continue  # 이미 1차에서 탐지됨
            if t in set(win_base):
                continue
            p = np.array([bands[b][i] for b in band_names])
            sim = cosine_sim(p, prototype)
            if sim >= sim_floor:
                win_rescued.append((t, env_full[i], sim))
                rescued_set.add(t)

        if win_rescued:
            debug_info[wi] = {
                "t0": t0, "t1": t1,
                "n_base": len(win_base),
                "sim_floor": sim_floor,
                "n_rescued": len(win_rescued),
                "proto": prototype.tolist(),
            }

    # 병합: 기존 + 구조된 피크, 진폭 내림차순 탐욕
    all_cands = [(t, env_full[time_to_idx[t]]) for t in pk_base]
    for t in rescued_set:
        all_cands.append((t, env_full[time_to_idx[t]]))
    all_cands.sort(key=lambda x: -x[1])

    selected = []
    for t, v in all_cands:
        if all(abs(t - s) >= min_gap_s for s in selected):
            selected.append(t)

    n_rescued_final = sum(1 for t in selected if t in rescued_set)
    return np.array(sorted(selected)), n_rescued_final, debug_info


for p in audio_paths():
    print(f"\n{'='*60}")
    print(f"▸ {p.name}")
    dur = duration_s(p)
    mono = load_mono(p)
    env, times = superflux_envelope(mono)
    bands = band_envelopes(mono)

    pk_old = peaks(env, times)
    pk_new, n_rescued, dbg = local_prototype_rescue(env, bands, times, dur)
    n_old, n_new = len(pk_old), len(pk_new)

    print(f"  기존: {n_old:5d}개 ({n_old/dur:.1f}/s)")
    print(f"  신규: {n_new:5d}개 ({n_new/dur:.1f}/s)  [구조: {n_rescued}]")
    print(f"  증가: {n_new-n_old:+d} ({(n_new/max(n_old,1)-1)*100:+.1f}%)")

    # 주요 윈도우 상세
    windows = [("0-4s", 0, 4), ("4-8s", 4, 8), ("16-20s", 16, 20),
               ("56-60s", 56, 60), ("100-104s", 100, 104),
               ("200-204s", 200, 204), ("268-272s", 268, 272)]
    for label, t0, t1 in windows:
        if t1 > dur:
            continue
        o = int(((pk_old >= t0) & (pk_old < t1)).sum())
        n = int(((pk_new >= t0) & (pk_new < t1)).sum())
        wi = int(t0 / WINDOW_S)
        d = dbg.get(wi, {})
        extra = ""
        if d:
            extra = f"  sim≥{d['sim_floor']:.3f}  proto=[{','.join(f'{x:.2f}' for x in d['proto'])}]"
        delta = f"Δ{n-o:+d}" if n != o else "  ="
        print(f"    {label:12s}: {o:3d} → {n:3d}  {delta}{extra}")
