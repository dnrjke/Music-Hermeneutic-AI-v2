import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from config import audio_paths, MIN_EVENT_GAP_S
from audio_io import load_mono, duration_s
from onset import superflux_envelope, band_envelopes
from peak_pick import peaks, otsu


def mid_merge_peaks(env_full, bands, times, min_gap_s=MIN_EVENT_GAP_S):
    """full-band + mid-band 단순 병합 (극대점 제약 없음).

    양쪽 피크를 합치고 진폭 내림차순 탐욕 선택.
    """
    pk_full = peaks(env_full, times, min_gap_s=min_gap_s)
    pk_mid = peaks(bands["mid"], times, min_gap_s=min_gap_s)

    # 시각→진폭 매핑 (full-band 기준, mid-band 피크는 mid 진폭 사용)
    time_to_idx = {}
    for i, t in enumerate(times):
        time_to_idx[t] = i

    candidates = []
    seen = set()
    for t in pk_full:
        idx = time_to_idx.get(t)
        if idx is not None:
            candidates.append((t, env_full[idx]))
            seen.add(t)
    for t in pk_mid:
        if t not in seen:
            idx = time_to_idx.get(t)
            if idx is not None:
                candidates.append((t, bands["mid"][idx]))

    # 진폭 내림차순 탐욕
    candidates.sort(key=lambda x: -x[1])
    selected = []
    for t, v in candidates:
        if all(abs(t - s) >= min_gap_s for s in selected):
            selected.append(t)

    return np.array(sorted(selected))


for p in audio_paths():
    print(f"\n{'='*60}")
    print(f"▸ {p.name}")
    dur = duration_s(p)
    mono = load_mono(p)
    env, times = superflux_envelope(mono)
    bands = band_envelopes(mono)

    pk_old = peaks(env, times)
    pk_new = mid_merge_peaks(env, bands, times)

    n_old, n_new = len(pk_old), len(pk_new)
    thr_full = otsu(env[np.isfinite(env)])
    thr_mid = otsu(bands["mid"][np.isfinite(bands["mid"])])
    print(f"  기존: {n_old:5d}개 ({n_old/dur:.1f}/s)  Otsu(full)={thr_full:.3f}")
    print(f"  신규: {n_new:5d}개 ({n_new/dur:.1f}/s)  Otsu(mid)={thr_mid:.3f}")
    print(f"  증가: {n_new-n_old:+d} ({(n_new/max(n_old,1)-1)*100:+.1f}%)")

    windows = [("0-4s", 0, 4), ("4-8s", 4, 8), ("8-12s", 8, 12),
               ("28-32s", 28, 32), ("56-60s", 56, 60),
               ("100-104s", 100, 104), ("120-124s", 120, 124)]
    for label, t0, t1 in windows:
        if t1 > dur:
            continue
        o = int(((pk_old >= t0) & (pk_old < t1)).sum())
        n = int(((pk_new >= t0) & (pk_new < t1)).sum())
        delta = f"Δ{n-o:+d}" if n != o else "  ="
        print(f"    {label:12s}: {o:3d} → {n:3d}  {delta}")
