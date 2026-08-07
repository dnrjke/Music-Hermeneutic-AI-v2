import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from config import audio_paths, MIN_EVENT_GAP_S
from audio_io import load_mono, duration_s
from onset import superflux_envelope, band_envelopes
from peak_pick import peaks, otsu


def rescue_peaks(env_full, bands, times, min_gap_s=MIN_EVENT_GAP_S):
    """기존 full-band 탐지 + 대역별 피크 구조.

    각 대역에서 Otsu 초과 피크 중 기존 탐지에서 min_gap 이내에
    없는 것만 추가. 단, full-band에서 극대점이어야 한다(노이즈 방지).
    """
    pk_base = peaks(env_full, times, min_gap_s=min_gap_s)
    base_set = set(pk_base.tolist())

    # full-band 극대점 시각 (임계 무관)
    is_localmax = np.zeros(len(env_full), dtype=bool)
    for i in range(1, len(env_full) - 1):
        if env_full[i] >= env_full[i-1] and env_full[i] > env_full[i+1]:
            is_localmax[i] = True
    localmax_times = set(times[is_localmax].tolist())

    rescued = []
    for name, band_env in bands.items():
        band_peaks = peaks(band_env, times, min_gap_s=min_gap_s)
        for t in band_peaks:
            if t in base_set:
                continue
            if t not in localmax_times:
                continue
            # 기존 피크와 min_gap 이상 거리
            if len(pk_base) > 0:
                nearest = np.min(np.abs(pk_base - t))
                if nearest < min_gap_s:
                    continue
            rescued.append(t)

    all_peaks = sorted(base_set | set(rescued))
    # 최종 min_gap 병합
    if len(all_peaks) < 2:
        return np.array(all_peaks)
    keep = [all_peaks[0]]
    for t in all_peaks[1:]:
        if t - keep[-1] >= min_gap_s:
            keep.append(t)
    return np.array(keep)


def bandmax_norm_peaks(bands, times, min_gap_s=MIN_EVENT_GAP_S):
    """각 대역을 자체 Otsu로 나눠 정규화 → max → Otsu → 피크."""
    normed = []
    for name, env in bands.items():
        thr = otsu(env[np.isfinite(env)])
        if thr > 0:
            normed.append(env / thr)
        else:
            normed.append(env)
    bandmax = np.maximum.reduce(normed)
    return peaks(bandmax, times, min_gap_s=min_gap_s)


for p in audio_paths():
    print(f"\n{'='*60}")
    print(f"▸ {p.name}")
    dur = duration_s(p)
    mono = load_mono(p)
    env, times = superflux_envelope(mono)
    bands = band_envelopes(mono)

    pk_a = peaks(env, times)
    pk_rescue = rescue_peaks(env, bands, times)
    pk_bmax = bandmax_norm_peaks(bands, times)

    methods = [
        ("A-기존(full Otsu)", pk_a),
        ("B-rescue(full+대역구조)", pk_rescue),
        ("C-bandmax-norm", pk_bmax),
    ]

    for label, pk in methods:
        n = len(pk)
        segments = []
        for tag, t0, t1 in [("0-4s", 0, 4), ("56-60s", 56, 60), ("100-104s", 100, 104)]:
            seg_n = int(((pk >= t0) & (pk < t1)).sum()) if n > 0 else 0
            segments.append(f"{tag}:{seg_n:2d}")
        print(f"  {label:28s}: {n:5d}개 ({n/dur:.1f}/s)  [{', '.join(segments)}]")
