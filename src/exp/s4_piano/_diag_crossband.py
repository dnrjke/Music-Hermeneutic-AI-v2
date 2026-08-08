"""s4_piano B-1: 교차대역 동시성(coincidence) 진단.

가설: 피아노 타건 = 광대역 임펄스 (low+mid+high 동시 flux).
      등간격 저역 비트 = 저역 전용 (mid/high flux 부재).

band_envelopes()의 low/mid/high에서 baseline 피크 각각에 대해
±10ms 내 mid/high 동시 존재 여부를 진단.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

import numpy as np
import librosa
from pathlib import Path
from config import (SR, N_FFT, HOP, N_MELS, FMIN, WINDOW_S,
                    MIN_EVENT_GAP_S, OUT_DIR,
                    SUPERFLUX_LAG, SUPERFLUX_MAX)
from audio_io import load_mono, duration_s
from onset import superflux_envelope, band_envelopes
from peak_pick import otsu

frame_dt = HOP / SR
COINCIDENCE_WINDOW_S = 0.010  # ±10ms


def count_seg(pk, t0, t1):
    return int(((pk >= t0) & (pk < t1)).sum())


def greedy_select(indices, values, times, min_gap_s):
    order = sorted(indices, key=lambda i: -values[i])
    selected = []
    for i in order:
        if all(abs(times[i] - times[j]) >= min_gap_s for j in selected):
            selected.append(i)
    if not selected:
        return np.array([], dtype=np.float64)
    return times[np.sort(np.asarray(selected, dtype=int))]


def peaks_otsu(env, times, min_gap_s=MIN_EVENT_GAP_S):
    fin = np.isfinite(env)
    if fin.sum() < 3:
        return np.array([], dtype=np.float64), 0.0
    v = np.where(fin, env, -np.inf)
    thr = otsu(env[fin])
    loc = np.flatnonzero(
        (v[1:-1] >= v[:-2]) & (v[1:-1] > v[2:]) & (v[1:-1] > thr)
    ) + 1
    if len(loc) == 0:
        return np.array([], dtype=np.float64), thr
    return greedy_select(loc, v, times, min_gap_s), thr


def band_activity(band_env, times, peak_time, window_s=COINCIDENCE_WINDOW_S):
    """peak_time ±window_s 내 band_env의 최대값과 평균값."""
    mask = (times >= peak_time - window_s) & (times <= peak_time + window_s)
    if mask.sum() == 0:
        return 0.0, 0.0
    seg = band_env[mask]
    return float(np.max(seg)), float(np.mean(seg))


# ── 메인 ──

audio_path = Path(r"E:\game\Music Hermeneutic AI v2\audio\102 - Dir.wav")
print(f"▸ {audio_path.name}")
mono = load_mono(audio_path)
dur = duration_s(audio_path)

env_base, _ = superflux_envelope(mono)
times = librosa.frames_to_time(np.arange(len(env_base)), sr=SR, hop_length=HOP)
bands = band_envelopes(mono)
print(f"  길이: {dur:.1f}s")

# 각 대역 env fix length
for bname in bands:
    bands[bname] = librosa.util.fix_length(bands[bname], size=len(times))

# baseline 피크
pk_base, thr_base = peaks_otsu(env_base, times)
print(f"  baseline: {len(pk_base)} 피크 (Otsu={thr_base:.4f})")

# ── 각 대역별 Otsu 임계 (참고) ──
print(f"\n  대역별 Otsu:")
band_thrs = {}
for bname, benv in bands.items():
    fin = np.isfinite(benv)
    pos = benv[fin & (benv > 0)]
    if len(pos) > 10:
        t = otsu(pos)
    else:
        t = 0.0
    band_thrs[bname] = t
    print(f"    {bname:>5s}: Otsu={t:.6f}  max={benv.max():.4f}  "
          f"median={np.median(benv[benv > 0]):.6f}")

# ── 각 baseline 피크에서 교차대역 동시성 진단 ──
print(f"\n{'='*80}")
print(f"  교차대역 동시성 진단 (±{COINCIDENCE_WINDOW_S*1000:.0f}ms)")
print(f"{'='*80}")

results = []
for pk_t in pk_base:
    row = {"time": pk_t}
    for bname, benv in bands.items():
        mx, mn = band_activity(benv, times, pk_t)
        row[f"{bname}_max"] = mx
        row[f"{bname}_mean"] = mn
        row[f"{bname}_active"] = mx > band_thrs[bname]
    row["all_active"] = all(row[f"{b}_active"] for b in bands)
    row["low_only"] = row["low_active"] and not row["mid_active"] and not row["high_active"]
    row["mid_high"] = row["mid_active"] and row["high_active"]
    results.append(row)

# ── 통계 ──
n_total = len(results)
n_all = sum(1 for r in results if r["all_active"])
n_low_only = sum(1 for r in results if r["low_only"])
n_mid_high = sum(1 for r in results if r["mid_high"])
n_no_low = sum(1 for r in results if not r["low_active"] and r["mid_high"])

print(f"\n  전체 피크: {n_total}")
print(f"  all_active (low∧mid∧high):  {n_all:4d}  ({n_all/n_total*100:.1f}%)")
print(f"  low_only (low만, mid·high 부재): {n_low_only:4d}  ({n_low_only/n_total*100:.1f}%)")
print(f"  mid∧high (mid+high 활성):  {n_mid_high:4d}  ({n_mid_high/n_total*100:.1f}%)")
print(f"  mid∧high but not low:     {n_no_low:4d}  ({n_no_low/n_total*100:.1f}%)")

# ── 시간 구간별 ──
print(f"\n{'='*80}")
print(f"  4초 윈도우별 분포")
print(f"{'='*80}")
print(f"  {'시각':>5s}  {'총':>4s}  {'all':>4s}  {'l만':>4s}  {'m∧h':>4s}  {'기타':>4s}")

n_win = int(np.ceil(dur / WINDOW_S))
for wi in range(n_win):
    t0, t1 = wi * WINDOW_S, (wi + 1) * WINDOW_S
    seg = [r for r in results if t0 <= r["time"] < t1]
    if not seg:
        ts = f"{int(t0)//60}:{int(t0)%60:02d}"
        print(f"  {ts:>5s}     0     0     0     0     0")
        continue
    ts = f"{int(t0)//60}:{int(t0)%60:02d}"
    sa = sum(1 for r in seg if r["all_active"])
    sl = sum(1 for r in seg if r["low_only"])
    sm = sum(1 for r in seg if r["mid_high"])
    so = len(seg) - sa - sl - (sm - sa)
    print(f"  {ts:>5s}  {len(seg):4d}  {sa:4d}  {sl:4d}  {sm:4d}  {so:4d}")

# ── 도입부 이후 등간격 구간 집중 분석 (9~30초) ──
print(f"\n{'='*80}")
print(f"  9-30초 구간 상세 (등간격 비트 의심 구간)")
print(f"{'='*80}")
print(f"  {'시각(s)':>8s}  {'low_max':>8s}  {'mid_max':>8s}  {'high_max':>8s}  "
      f"{'l':>2s} {'m':>2s} {'h':>2s}  판정")

focus = [r for r in results if 9.0 <= r["time"] < 30.0]
for r in focus:
    lo = "●" if r["low_active"] else "○"
    mi = "●" if r["mid_active"] else "○"
    hi = "●" if r["high_active"] else "○"
    if r["all_active"]:
        judge = "광대역(타건?)"
    elif r["low_only"]:
        judge = "★저역전용(비트?)"
    elif r["mid_high"]:
        judge = "중고역"
    else:
        judge = "미활성"
    print(f"  {r['time']:8.3f}  {r['low_max']:8.5f}  {r['mid_max']:8.5f}  "
          f"{r['high_max']:8.5f}  {lo:>2s} {mi:>2s} {hi:>2s}  {judge}")

# ── 등간격 분석 ──
print(f"\n{'='*80}")
print(f"  등간격 분석")
print(f"{'='*80}")

# 9초 이후 low_only 피크들의 간격 분석
low_only_times = np.array([r["time"] for r in results
                            if r["low_only"] and r["time"] >= 9.0])
all_active_times = np.array([r["time"] for r in results
                              if r["all_active"] and r["time"] >= 9.0])

print(f"\n  9초 이후:")
print(f"  low_only 피크: {len(low_only_times)}")
if len(low_only_times) >= 2:
    gaps = np.diff(low_only_times) * 1000
    print(f"    간격: 중앙값={np.median(gaps):.1f}ms  "
          f"평균={gaps.mean():.1f}ms  std={gaps.std():.1f}ms  "
          f"min={gaps.min():.1f}ms  max={gaps.max():.1f}ms")

print(f"  all_active 피크: {len(all_active_times)}")
if len(all_active_times) >= 2:
    gaps = np.diff(all_active_times) * 1000
    print(f"    간격: 중앙값={np.median(gaps):.1f}ms  "
          f"평균={gaps.mean():.1f}ms  std={gaps.std():.1f}ms")

# ── 게이트 적용 시 결과 예측 ──
print(f"\n{'='*80}")
print(f"  교차대역 게이트 적용 시 예상")
print(f"{'='*80}")

# 게이트: all_active OR mid_high만 통과 (low_only 제거)
pk_gated = np.array([r["time"] for r in results if not r["low_only"]])
pk_removed = np.array([r["time"] for r in results if r["low_only"]])

print(f"  원래: {n_total}")
print(f"  게이트 통과 (low_only 제거): {len(pk_gated)}")
print(f"  제거: {len(pk_removed)}")
print(f"\n  4초 윈도우별 (게이트 전/후):")
print(f"  {'시각':>5s}  {'전':>4s}  {'후':>4s}  {'제거':>4s}")

for wi in range(n_win):
    t0, t1 = wi * WINDOW_S, (wi + 1) * WINDOW_S
    cb = count_seg(pk_base, t0, t1)
    cg = count_seg(pk_gated, t0, t1)
    cr = count_seg(pk_removed, t0, t1)
    if cb == 0 and cg == 0:
        continue
    ts = f"{int(t0)//60}:{int(t0)%60:02d}"
    marker = f"  ◀ -{cr}" if cr > 0 else ""
    print(f"  {ts:>5s}  {cb:4d}  {cg:4d}  {cr:4d}{marker}")

print(f"  {'합계':>5s}  {n_total:4d}  {len(pk_gated):4d}  {len(pk_removed):4d}")
