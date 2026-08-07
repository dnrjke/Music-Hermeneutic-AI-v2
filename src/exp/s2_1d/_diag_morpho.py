"""Morphological opening으로 충격/점진 분리 진단.

성운 파이프라인의 별-성운 분리를 음향에 적용:
  opening(envelope, kernel) → 점진 성분
  envelope - opening        → 충격 성분 (별 = 건반/킥)

커널 크기: MIN_EVENT_GAP_S / frame_dt ≈ 5 frames (30ms / 5.8ms)
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from scipy.ndimage import grey_opening
from config import audio_paths, MIN_EVENT_GAP_S, HOP, SR, WINDOW_S
from audio_io import load_mono, duration_s
from onset import superflux_envelope, band_envelopes
from peak_pick import peaks, otsu


frame_dt = HOP / SR
kernel_frames = max(3, round(MIN_EVENT_GAP_S / frame_dt))

print(f"프레임 간격: {frame_dt*1000:.1f}ms")
print(f"커널 크기: {kernel_frames} frames ({kernel_frames * frame_dt * 1000:.0f}ms)")


def impulse_peaks(env, times, kernel=kernel_frames, min_gap_s=MIN_EVENT_GAP_S):
    """충격 성분에서 피크 탐지."""
    opened = grey_opening(env, size=kernel)
    residual = np.maximum(env - opened, 0)
    return peaks(residual, times, min_gap_s=min_gap_s), residual


for p in audio_paths():
    print(f"\n{'='*60}")
    print(f"▸ {p.name}")
    dur = duration_s(p)
    mono = load_mono(p)
    env, times = superflux_envelope(mono)

    pk_orig = peaks(env, times)
    n_orig = len(pk_orig)

    print(f"  {'방법':28s}: {'전체':>5s} ({'/s':>5s})"
          f"  {'0-4':>5s}  {'16-20':>5s}  {'56-60':>5s}"
          f"  {'200-204':>7s}  {'268-272':>7s}")

    # 기존
    def seg_counts(pk):
        result = []
        for t0, t1 in [(0,4),(16,20),(56,60),(200,204),(268,272)]:
            if t1 > dur:
                result.append("      ")
            else:
                n = int(((pk >= t0) & (pk < t1)).sum())
                result.append(f"{n:5d}  ")
        return "".join(result)

    print(f"  {'A-기존(full Otsu)':28s}: {n_orig:5d} ({n_orig/dur:4.1f}/s)  {seg_counts(pk_orig)}")

    # 다양한 커널 크기
    for k in [3, 5, 7, 9, 13]:
        pk_imp, residual = impulse_peaks(env, times, kernel=k)
        n = len(pk_imp)
        thr_imp = otsu(residual[np.isfinite(residual)])
        thr_orig = otsu(env[np.isfinite(env)])
        label = f"B-impulse k={k} ({k*frame_dt*1000:.0f}ms)"
        print(f"  {label:28s}: {n:5d} ({n/dur:4.1f}/s)  {seg_counts(pk_imp)}"
              f"  otsu {thr_orig:.2f}→{thr_imp:.2f}")

    # GL 0-4s 상세 분석
    if "Grievous" in p.name:
        print(f"\n  --- GL 0-4s 상세 ---")
        mask = (times >= 0) & (times < 4)
        seg = env[mask]

        for k in [3, 5, 7]:
            opened = grey_opening(seg, size=k)
            residual = np.maximum(seg - opened, 0)
            # 극대점 수 (임계 무관)
            n_lm = 0
            for i in range(1, len(residual) - 1):
                if residual[i] >= residual[i-1] and residual[i] > residual[i+1]:
                    n_lm += 1
            thr = otsu(residual[residual > 0]) if (residual > 0).sum() > 10 else 0
            above = 0
            for i in range(1, len(residual) - 1):
                if residual[i] >= residual[i-1] and residual[i] > residual[i+1]:
                    if residual[i] > thr:
                        above += 1
            print(f"    k={k}: 극대점={n_lm}  otsu={thr:.3f}  "
                  f"otsu초과={above}  "
                  f"잔차 max={residual.max():.3f}  mean={residual[residual>0].mean():.3f}")
