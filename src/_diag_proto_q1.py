"""국소 프로토타입 rescue — 유사도 기준 비교.

min, Q1, median, Q3 각각을 floor로 사용했을 때의 결과 비교.
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


def local_proto_rescue(env_full, bands, times, dur, sim_percentile=0,
                       window_s=WINDOW_S, min_gap_s=MIN_EVENT_GAP_S):
    pk_base = peaks(env_full, times, min_gap_s=min_gap_s)
    if len(pk_base) < 1:
        return pk_base, 0

    time_to_idx = {t: i for i, t in enumerate(times)}
    band_names = ["low", "mid", "high"]
    thr = otsu(env_full[np.isfinite(env_full)])

    all_lm = []
    for i in range(1, len(env_full) - 1):
        if env_full[i] >= env_full[i-1] and env_full[i] > env_full[i+1]:
            all_lm.append(i)

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
        sim_floor = float(np.percentile(sims, sim_percentile))

        for i in all_lm:
            t = times[i]
            if not (t0 <= t < t1):
                continue
            if env_full[i] > thr or t in set(win_base):
                continue
            p = np.array([bands[b][i] for b in band_names])
            sim = cosine_sim(p, prototype)
            if sim >= sim_floor:
                rescued_set.add(t)

    all_cands = [(t, env_full[time_to_idx[t]]) for t in pk_base]
    for t in rescued_set:
        all_cands.append((t, env_full[time_to_idx[t]]))
    all_cands.sort(key=lambda x: -x[1])

    selected = []
    for t, v in all_cands:
        if all(abs(t - s) >= min_gap_s for s in selected):
            selected.append(t)

    n_rescued = sum(1 for t in selected if t in rescued_set)
    return np.array(sorted(selected)), n_rescued


# 주요 구간만 표시
human_ref = {
    "03.Grievous Lady.wav": {0: 18, 4: 14},  # 0-4s, 16-20s (win4)
    "07.Viyella's Nightmare.wav": {50: 6, 67: 14},
}

for p in audio_paths():
    print(f"\n{'='*60}")
    print(f"▸ {p.name}")
    dur = duration_s(p)
    mono = load_mono(p)
    env, times = superflux_envelope(mono)
    bands = band_envelopes(mono)

    pk_base = peaks(env, times)
    n_base = len(pk_base)

    header = f"  {'방법':22s}: {'전체':>5s}  {'Δ':>4s}"
    for label in ["0-4", "16-20", "56-60", "200-204", "268-272"]:
        t1 = int(label.split("-")[1])
        if t1 <= dur:
            header += f"  {label:>7s}"
    print(header)

    for pct_label, pct in [("min(P0)", 0), ("Q1(P25)", 25),
                           ("med(P50)", 50), ("Q3(P75)", 75)]:
        pk, nr = local_proto_rescue(env, bands, times, dur, sim_percentile=pct)
        n = len(pk)
        line = f"  {pct_label:22s}: {n:5d}  {n-n_base:+4d}"
        for label, t0, t1 in [(None, 0, 4), (None, 16, 20), (None, 56, 60),
                               (None, 200, 204), (None, 268, 272)]:
            if t1 > dur:
                continue
            c = int(((pk >= t0) & (pk < t1)).sum())
            line += f"  {c:7d}"
        print(line)

    # 기존 기준선
    line = f"  {'기존(Otsu)':22s}: {n_base:5d}     "
    for label, t0, t1 in [(None, 0, 4), (None, 16, 20), (None, 56, 60),
                           (None, 200, 204), (None, 268, 272)]:
        if t1 > dur:
            continue
        c = int(((pk_base >= t0) & (pk_base < t1)).sum())
        line += f"  {c:7d}"
    print(line)
