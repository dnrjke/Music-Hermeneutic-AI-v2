import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from config import audio_paths
from audio_io import load_mono
from onset import superflux_envelope, band_envelopes
from peak_pick import otsu, peaks

p = [x for x in audio_paths() if "Grievous" in x.name][0]
mono = load_mono(p)
env, times = superflux_envelope(mono)
bands = band_envelopes(mono)

thr_global = otsu(env[np.isfinite(env)])
print(f"전역 Otsu 임계: {thr_global:.4f}")
print(f"포락선 전곡: min {env.min():.4f}  max {env.max():.4f}  "
      f"median {np.median(env):.4f}  mean {env.mean():.4f}")

# 0-4s 구간
mask = (times >= 0) & (times < 4)
seg = env[mask]
seg_t = times[mask]
print(f"\n=== 0-4s 구간 ===")
print(f"포락선: min {seg.min():.4f}  max {seg.max():.4f}  "
      f"median {np.median(seg):.4f}  mean {seg.mean():.4f}")

# 이 구간의 극대점 (임계 없이)
all_peaks = []
for i in range(1, len(seg) - 1):
    if seg[i] >= seg[i-1] and seg[i] > seg[i+1]:
        all_peaks.append((seg_t[i], seg[i]))

print(f"극대점(임계 없이): {len(all_peaks)}개")
print(f"극대점 값 분포: {sorted([v for _, v in all_peaks], reverse=True)[:25]}")

# 전역 Otsu로 잘리는 개수
above = [(t, v) for t, v in all_peaks if v > thr_global]
below = [(t, v) for t, v in all_peaks if v <= thr_global]
print(f"\n전역 Otsu({thr_global:.4f}) 초과: {len(above)}개")
print(f"전역 Otsu 이하: {len(below)}개")
if below:
    print(f"  놓친 것 최대값: {max(v for _, v in below):.4f}")

# 이 구간만의 Otsu
thr_local = otsu(seg)
above_local = [(t, v) for t, v in all_peaks if v > thr_local]
print(f"\n구간 Otsu({thr_local:.4f}) 초과: {len(above_local)}개")

# 대역별 확인
for band_name in ["low", "mid", "high"]:
    b = bands[band_name][mask]
    print(f"\n  {band_name:4s} 대역: mean {b.mean():.4f}  max {b.max():.4f}")

# 실제 peaks() 결과
pk = peaks(env, times)
pk_in_seg = pk[(pk >= 0) & (pk < 4)]
print(f"\n실제 peaks() 결과 (0-4s): {len(pk_in_seg)}개")
print(f"  시각: {[f'{t:.3f}' for t in pk_in_seg]}")

# mid 대역만으로 피크 잡기
mid_env = bands["mid"]
pk_mid = peaks(mid_env, times)
pk_mid_seg = pk_mid[(pk_mid >= 0) & (pk_mid < 4)]
print(f"\nmid 대역 peaks() (0-4s): {len(pk_mid_seg)}개")
thr_mid = otsu(mid_env[np.isfinite(mid_env)])
print(f"  mid Otsu: {thr_mid:.4f}  mid 구간 max: {bands['mid'][mask].max():.4f}")
