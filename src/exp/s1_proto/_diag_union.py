import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from config import audio_paths, MIN_EVENT_GAP_S
from audio_io import load_mono, duration_s
from onset import superflux_envelope, band_envelopes
from peak_pick import peaks, otsu


def peaks_band_union(env_full, bands, times, min_gap_s=MIN_EVENT_GAP_S):
    """대역별 독립 Otsu + 합집합 + 최소간격 병합."""
    all_times = set()
    for name, env in [("full", env_full)] + list(bands.items()):
        pk = peaks(env, times, min_gap_s=min_gap_s)
        all_times.update(pk.tolist())
    merged = np.array(sorted(all_times))
    if len(merged) < 2:
        return merged
    # 최소간격 병합 (강한 것 우선은 불가 — 대역이 섞여 있으므로 시간순 탐욕)
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

    pk_old = peaks(env, times)
    pk_new = peaks_band_union(env, bands, times)

    print(f"  기존(전대역 Otsu): {len(pk_old)}개  ({len(pk_old)/dur:.1f}/s)")
    print(f"  신규(대역합집합):  {len(pk_new)}개  ({len(pk_new)/dur:.1f}/s)")
    print(f"  증가: +{len(pk_new)-len(pk_old)} ({(len(pk_new)/max(len(pk_old),1)-1)*100:+.0f}%)")

    # 구간별 비교: 0-4s (피아노), 기존 최대 밀도 구간
    for label, t0, t1 in [("0-4s", 0, 4), ("56-60s", 56, 60), ("100-104s", 100, 104)]:
        old_n = int(((pk_old >= t0) & (pk_old < t1)).sum())
        new_n = int(((pk_new >= t0) & (pk_new < t1)).sum())
        print(f"    {label:10s}: 기존 {old_n:3d} → 신규 {new_n:3d}  (Δ {new_n-old_n:+d})")
