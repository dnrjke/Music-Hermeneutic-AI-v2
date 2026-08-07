import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from config import audio_paths, MIN_EVENT_GAP_S, HOP, SR
from audio_io import load_mono, duration_s
from onset import superflux_envelope
from peak_pick import peaks, otsu


def local_otsu_peaks(env, times, window_s=16.0, hop_s=4.0,
                     min_gap_s=MIN_EVENT_GAP_S):
    """슬라이딩 윈도우 Otsu: 각 윈도우의 임계를 따로 계산.

    각 시각에 대해 해당 위치를 포함하는 모든 윈도우의 Otsu 중
    최솟값을 사용 (가장 관대한 윈도우 기준).
    """
    dt = HOP / SR  # 프레임 간격
    n = len(env)
    thresholds = np.full(n, np.inf)

    # 슬라이딩 윈도우
    win_frames = int(window_s / dt)
    hop_frames = int(hop_s / dt)

    starts = list(range(0, max(1, n - win_frames + 1), hop_frames))
    if starts[-1] + win_frames < n:
        starts.append(max(0, n - win_frames))

    for s in starts:
        e = min(s + win_frames, n)
        seg = env[s:e]
        seg_valid = seg[np.isfinite(seg)]
        if len(seg_valid) < 10:
            continue
        thr = otsu(seg_valid)
        # 이 윈도우 내 각 프레임의 임계를 min으로 갱신
        thresholds[s:e] = np.minimum(thresholds[s:e], thr)

    # inf 남은 곳 → 전역 Otsu
    global_thr = otsu(env[np.isfinite(env)])
    thresholds[thresholds == np.inf] = global_thr

    # 극대점 중 국소 임계 초과인 것
    candidates = []
    for i in range(1, len(env) - 1):
        if env[i] >= env[i-1] and env[i] > env[i+1]:
            if env[i] > thresholds[i]:
                candidates.append((times[i], env[i]))

    if not candidates:
        return np.array([])

    # 진폭 내림차순 탐욕적 선택
    candidates.sort(key=lambda x: -x[1])
    selected = []
    for t, v in candidates:
        if all(abs(t - s) >= min_gap_s for s in selected):
            selected.append(t)

    return np.array(sorted(selected))


# 테스트: 다양한 윈도우 크기
for p in audio_paths():
    print(f"\n{'='*60}")
    print(f"▸ {p.name}")
    dur = duration_s(p)
    mono = load_mono(p)
    env, times = superflux_envelope(mono)

    pk_a = peaks(env, times)

    print(f"  {'방법':28s}: {'전체':>6s} ({'rate':>5s})  "
          f"{'0-4s':>5s}  {'56-60s':>6s}  {'100-104s':>8s}")

    for label, pk in [("A-기존(global Otsu)", pk_a)]:
        n = len(pk)
        s1 = int(((pk >= 0) & (pk < 4)).sum()) if n else 0
        s2 = int(((pk >= 56) & (pk < 60)).sum()) if n else 0
        s3 = int(((pk >= 100) & (pk < 104)).sum()) if n else 0
        print(f"  {label:28s}: {n:5d}개 ({n/dur:4.1f}/s)  "
              f"{s1:5d}  {s2:6d}  {s3:8d}")

    for ws in [8, 16, 32]:
        pk_loc = local_otsu_peaks(env, times, window_s=ws, hop_s=ws/4)
        n = len(pk_loc)
        s1 = int(((pk_loc >= 0) & (pk_loc < 4)).sum()) if n else 0
        s2 = int(((pk_loc >= 56) & (pk_loc < 60)).sum()) if n else 0
        s3 = int(((pk_loc >= 100) & (pk_loc < 104)).sum()) if n else 0
        label = f"B-local Otsu (win={ws}s)"
        print(f"  {label:28s}: {n:5d}개 ({n/dur:4.1f}/s)  "
              f"{s1:5d}  {s2:6d}  {s3:8d}")
