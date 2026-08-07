import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from config import audio_paths, MIN_EVENT_GAP_S, HOP, SR
from audio_io import load_mono, duration_s
from onset import superflux_envelope, band_envelopes
from peak_pick import peaks, otsu


def mid_rescue_peaks(env_full, bands, times, min_gap_s=MIN_EVENT_GAP_S):
    """full-band 탐지 + mid-band 전용 구조.

    mid-band(120-2000Hz)에서 Otsu 초과 피크 중,
    full-band 극대점이면서 기존 탐지와 min_gap 이상 떨어진 것만 추가.
    """
    pk_base = peaks(env_full, times, min_gap_s=min_gap_s)
    base_set = set(pk_base.tolist())

    # full-band 극대점 시각
    is_lm = np.zeros(len(env_full), dtype=bool)
    for i in range(1, len(env_full) - 1):
        if env_full[i] >= env_full[i-1] and env_full[i] > env_full[i+1]:
            is_lm[i] = True
    lm_times = set(times[is_lm].tolist())

    # mid-band 피크 구조
    mid_peaks = peaks(bands["mid"], times, min_gap_s=min_gap_s)
    rescued = []
    for t in mid_peaks:
        if t in base_set:
            continue
        if t not in lm_times:
            continue
        if len(pk_base) > 0:
            nearest = np.min(np.abs(pk_base - t))
            if nearest < min_gap_s:
                continue
        rescued.append(t)

    all_peaks = sorted(base_set | set(rescued))
    if len(all_peaks) < 2:
        return np.array(all_peaks), len(rescued)
    keep = [all_peaks[0]]
    for t in all_peaks[1:]:
        if t - keep[-1] >= min_gap_s:
            keep.append(t)
    return np.array(keep), len(rescued)


for p in audio_paths():
    print(f"\n{'='*60}")
    print(f"▸ {p.name}")
    dur = duration_s(p)
    mono = load_mono(p)
    env, times = superflux_envelope(mono)
    bands = band_envelopes(mono)

    pk_old = peaks(env, times)
    pk_new, n_rescued = mid_rescue_peaks(env, bands, times)

    n_old, n_new = len(pk_old), len(pk_new)
    print(f"  기존: {n_old:5d}개 ({n_old/dur:.1f}/s)")
    print(f"  신규: {n_new:5d}개 ({n_new/dur:.1f}/s)  "
          f"[구조된 mid 피크: {n_rescued}]")
    print(f"  증가: {n_new-n_old:+d} ({(n_new/max(n_old,1)-1)*100:+.1f}%)")

    # 구간별 비교
    windows = [("0-4s", 0, 4), ("4-8s", 4, 8), ("8-12s", 8, 12),
               ("52-56s", 52, 56), ("56-60s", 56, 60),
               ("96-100s", 96, 100), ("100-104s", 100, 104)]
    for label, t0, t1 in windows:
        if t1 > dur:
            continue
        o = int(((pk_old >= t0) & (pk_old < t1)).sum())
        n = int(((pk_new >= t0) & (pk_new < t1)).sum())
        delta = f"Δ{n-o:+d}" if n != o else "  ="
        print(f"    {label:10s}: {o:3d} → {n:3d}  {delta}")
