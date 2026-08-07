import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from config import audio_paths, MIN_EVENT_GAP_S
from audio_io import load_mono, duration_s
from onset import superflux_envelope, band_envelopes
from peak_pick import peaks, otsu


def merge_peaks(peak_arrays, min_gap_s=MIN_EVENT_GAP_S):
    """여러 대역의 피크를 합치고 최소간격 병합."""
    all_t = set()
    for pk in peak_arrays:
        all_t.update(pk.tolist())
    merged = np.array(sorted(all_t))
    if len(merged) < 2:
        return merged
    keep = [merged[0]]
    for t in merged[1:]:
        if t - keep[-1] >= min_gap_s:
            keep.append(t)
    return np.array(keep)


for p in audio_paths():
    print(f"\n{'='*60}")
    print(f"▸ {p.name}")
    dur = duration_s(p)
    mono = load_mono(p)
    env, times = superflux_envelope(mono)
    bands = band_envelopes(mono)

    pk_full = peaks(env, times)
    pk_low = peaks(bands["low"], times)
    pk_mid = peaks(bands["mid"], times)
    pk_high = peaks(bands["high"], times)

    # A: 기존 (전대역 Otsu)
    # B: 3대역 합집합 (full 제외)
    pk_3band = merge_peaks([pk_low, pk_mid, pk_high])
    # C: full + mid 합집합만
    pk_full_mid = merge_peaks([pk_full, pk_mid])

    for label, pk in [("A-기존(full)", pk_full),
                      ("B-3대역합(L+M+H)", pk_3band),
                      ("C-full+mid", pk_full_mid),
                      ("  low단독", pk_low),
                      ("  mid단독", pk_mid),
                      ("  high단독", pk_high)]:
        n = len(pk)
        seg_04 = int(((pk >= 0) & (pk < 4)).sum()) if n > 0 else 0
        seg_56 = int(((pk >= 56) & (pk < 60)).sum()) if n > 0 else 0
        thr_info = ""
        if "단독" in label:
            band_name = label.strip().replace("단독", "")
            thr_info = f"  otsu={otsu(bands[band_name]):.3f}"
        elif label.startswith("A"):
            thr_info = f"  otsu={otsu(env):.3f}"
        print(f"  {label:22s}: {n:5d}개 ({n/dur:.1f}/s)  "
              f"[0-4s: {seg_04:2d}] [56-60s: {seg_56:2d}]{thr_info}")
