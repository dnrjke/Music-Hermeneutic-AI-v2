"""s4_piano: 등간격 비트 artifact의 출처 진단 (대역 무관).

질문: baseline의 등간격 클릭이 저역 비트가 아니라 배경음 유래일 수 있는가?
접근:
  1. baseline 피크 IOI(inter-onset interval) 자기상관 → 지배 주기 추정
  2. 주기 격자에 위상 고정된 피크 식별
  3. 격자 피크 vs 비격자 피크의 스펙트럼 특성 비교
  4. novelty_norm 피크와의 대조
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

import numpy as np
import librosa
from pathlib import Path
from config import (SR, N_FFT, HOP, N_MELS, FMIN, WINDOW_S,
                    MIN_EVENT_GAP_S, SUPERFLUX_LAG)
from audio_io import load_mono, duration_s
from onset import superflux_envelope, band_envelopes
from peak_pick import otsu
from scipy.ndimage import uniform_filter1d

frame_dt = HOP / SR


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


def _get_logmel(mono):
    S = np.abs(librosa.stft(mono, n_fft=N_FFT, hop_length=HOP))
    mel = librosa.feature.melspectrogram(S=S**2, sr=SR, n_mels=N_MELS, fmin=FMIN)
    logmel = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    return logmel, S.shape[1]


def env_spectral_novelty(mono):
    logmel, n = _get_logmel(mono)
    norms = np.linalg.norm(logmel, axis=0, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normed = logmel / norms
    cos_sim = np.sum(normed[:, 1:] * normed[:, :-1], axis=0)
    cos_sim = np.clip(cos_sim, -1, 1)
    distance = 1 - cos_sim
    env = np.zeros(n, dtype=np.float64)
    env[1:1 + len(distance)] = np.maximum(distance, 0)
    return env


def bandpass_norm_peaks(env, times, smooth_s=2.0, norm_block_s=2.0,
                        min_gap_s=MIN_EVENT_GAP_S):
    env_pos = np.maximum(env, 0.0)
    smooth_frames = max(3, int(smooth_s / frame_dt) | 1)
    env_smooth = uniform_filter1d(env_pos, size=smooth_frames, mode='reflect')
    residual = np.maximum(env_pos - env_smooth, 0.0)
    dur_t = times[-1] + frame_dt
    n_blocks = int(np.ceil(dur_t / norm_block_s))
    normed = np.zeros_like(residual)
    for bi in range(n_blocks):
        t0, t1 = bi * norm_block_s, (bi + 1) * norm_block_s
        mask = (times >= t0) & (times < t1)
        seg = residual[mask]
        pos = seg[seg > 0]
        if len(pos) < 3:
            continue
        p99 = np.percentile(pos, 99.0)
        if p99 < 1e-12:
            continue
        normed[mask] = np.clip(seg / p99, 0, None)
    pos_vals = normed[normed > 0]
    if len(pos_vals) < 10:
        return np.array([], dtype=np.float64)
    thr = otsu(pos_vals)
    loc = np.flatnonzero(
        (normed[1:-1] >= normed[:-2]) &
        (normed[1:-1] > normed[2:]) &
        (normed[1:-1] > thr)
    ) + 1
    if len(loc) == 0:
        return np.array([], dtype=np.float64)
    return greedy_select(loc, normed, times, min_gap_s)


# ── 메인 ──

audio_path = Path(r"E:\game\Music Hermeneutic AI v2\audio\102 - Dir.wav")
print(f"▸ {audio_path.name}")
mono = load_mono(audio_path)
dur = duration_s(audio_path)

env_base, _ = superflux_envelope(mono)
times = librosa.frames_to_time(np.arange(len(env_base)), sr=SR, hop_length=HOP)
logmel, _ = _get_logmel(mono)

pk_base, thr_base = peaks_otsu(env_base, times)
print(f"  baseline: {len(pk_base)} 피크")

# novelty norm (현재 최선)
e_nov = env_spectral_novelty(mono)
e_nov = librosa.util.fix_length(e_nov, size=len(times))
pk_nov_norm = bandpass_norm_peaks(e_nov, times)
print(f"  novelty_norm: {len(pk_nov_norm)} 피크")

# ── 1. IOI 자기상관으로 주기 추정 ──
print(f"\n{'='*72}")
print(f"  1. IOI 분석 — 지배 주기 추정")
print(f"{'='*72}")

# 9초 이후만 (도입부 제외)
pk_post9 = pk_base[pk_base >= 9.0]
ioi = np.diff(pk_post9) * 1000  # ms

print(f"  9초 이후 피크: {len(pk_post9)}")
print(f"  IOI: 중앙값={np.median(ioi):.1f}ms  평균={np.mean(ioi):.1f}ms  "
      f"std={np.std(ioi):.1f}ms")

# IOI 히스토그램 (10ms 빈)
bins = np.arange(0, 500, 10)
hist, edges = np.histogram(ioi, bins=bins)
top_bins = np.argsort(hist)[::-1][:5]
print(f"\n  IOI 히스토그램 상위 5 빈:")
for bi in top_bins:
    if hist[bi] > 0:
        print(f"    {edges[bi]:.0f}-{edges[bi+1]:.0f}ms: {hist[bi]}개")

# 피크 시각을 이산 임펄스열로 → 자기상관
resolution_ms = 5.0
max_lag_ms = 2000.0
n_bins_ac = int(dur * 1000 / resolution_ms)
impulse = np.zeros(n_bins_ac)
for t in pk_post9:
    idx = int(t * 1000 / resolution_ms)
    if 0 <= idx < n_bins_ac:
        impulse[idx] = 1.0

max_lag = int(max_lag_ms / resolution_ms)
ac = np.correlate(impulse, impulse, mode='full')
ac = ac[len(impulse)-1:]  # 양의 래그만
ac = ac[:max_lag+1]
ac[0] = 0  # 자기 자신 제거

# 상위 피크
from scipy.signal import find_peaks as sp_find_peaks
ac_peaks, _ = sp_find_peaks(ac, height=ac.max() * 0.3)
if len(ac_peaks) > 0:
    ac_vals = ac[ac_peaks]
    top_ac = ac_peaks[np.argsort(ac_vals)[::-1][:5]]
    print(f"\n  자기상관 상위 주기:")
    for lag in top_ac:
        period_ms = lag * resolution_ms
        bpm = 60000 / period_ms if period_ms > 0 else 0
        print(f"    {period_ms:.0f}ms ({bpm:.1f} BPM)  상관값={ac[lag]:.1f}")
    dominant_period_ms = top_ac[0] * resolution_ms
else:
    print(f"  자기상관 피크 없음")
    dominant_period_ms = np.median(ioi)

print(f"\n  지배 주기: {dominant_period_ms:.0f}ms ({60000/dominant_period_ms:.1f} BPM)")

# ── 2. 주기 격자에 위상 고정된 피크 식별 ──
print(f"\n{'='*72}")
print(f"  2. 격자 위상 고정 분석")
print(f"{'='*72}")

period_s = dominant_period_ms / 1000.0
GRID_TOL_MS = 20.0  # ±20ms 이내면 격자에 고정

# 격자 시작점 = 첫 피크
grid_start = pk_post9[0]
on_grid = []
off_grid = []

for t in pk_post9:
    phase = ((t - grid_start) % period_s) * 1000  # ms
    if phase > period_s * 1000 / 2:
        phase = period_s * 1000 - phase
    if phase <= GRID_TOL_MS:
        on_grid.append(t)
    else:
        off_grid.append(t)

on_grid = np.array(on_grid)
off_grid = np.array(off_grid)

print(f"  주기: {dominant_period_ms:.0f}ms  허용 오차: ±{GRID_TOL_MS:.0f}ms")
print(f"  격자 위 피크: {len(on_grid)} ({len(on_grid)/len(pk_post9)*100:.1f}%)")
print(f"  격자 밖 피크: {len(off_grid)} ({len(off_grid)/len(pk_post9)*100:.1f}%)")

# ── 3. 격자 피크 vs 비격자 피크 스펙트럼 비교 ──
print(f"\n{'='*72}")
print(f"  3. 격자 피크 vs 비격자 피크 — 스펙트럼 특성")
print(f"{'='*72}")

mel_freqs = librosa.mel_frequencies(n_mels=N_MELS, fmin=FMIN, fmax=SR/2)

def spectral_profile(peak_times, logmel, times):
    """피크 시각들의 평균 mel 스펙트럼."""
    profiles = []
    for t in peak_times:
        idx = np.argmin(np.abs(times - t))
        if 0 <= idx < logmel.shape[1]:
            profiles.append(logmel[:, idx])
    if not profiles:
        return np.zeros(logmel.shape[0])
    return np.mean(profiles, axis=0)

def spectral_centroid_from_mel(peak_times, logmel, times, mel_freqs):
    """피크 시각들의 스펙트럼 중심 주파수 분포."""
    centroids = []
    for t in peak_times:
        idx = np.argmin(np.abs(times - t))
        if 0 <= idx < logmel.shape[1]:
            spec = 10 ** (logmel[:, idx] / 10)  # dB → linear
            spec = np.maximum(spec, 0)
            total = spec.sum()
            if total > 0:
                centroid = np.sum(mel_freqs * spec) / total
                centroids.append(centroid)
    return np.array(centroids)

prof_on = spectral_profile(on_grid, logmel, times)
prof_off = spectral_profile(off_grid, logmel, times)
diff = prof_on - prof_off

# 대역별 에너지 비교
band_edges = [20, 120, 500, 2000, 5000, 20000]
band_names = ["20-120", "120-500", "500-2k", "2k-5k", "5k-20k"]

print(f"\n  대역별 평균 에너지 (dB):")
print(f"  {'대역(Hz)':>10s}  {'격자':>8s}  {'비격자':>8s}  {'차이':>8s}")
for i in range(len(band_names)):
    f_lo, f_hi = band_edges[i], band_edges[i+1]
    mask = (mel_freqs >= f_lo) & (mel_freqs < f_hi)
    if mask.sum() == 0:
        continue
    e_on = prof_on[mask].mean()
    e_off = prof_off[mask].mean()
    d = e_on - e_off
    marker = " ◀" if abs(d) > 1.0 else ""
    print(f"  {band_names[i]:>10s}  {e_on:8.2f}  {e_off:8.2f}  {d:+8.2f}{marker}")

# 스펙트럼 중심 주파수
cent_on = spectral_centroid_from_mel(on_grid, logmel, times, mel_freqs)
cent_off = spectral_centroid_from_mel(off_grid, logmel, times, mel_freqs)

print(f"\n  스펙트럼 중심 주파수:")
print(f"  격자 피크:   중앙값={np.median(cent_on):.0f}Hz  평균={np.mean(cent_on):.0f}Hz")
print(f"  비격자 피크: 중앙값={np.median(cent_off):.0f}Hz  평균={np.mean(cent_off):.0f}Hz")

# ── 4. SuperFlux envelope 값 비교 ──
print(f"\n  SuperFlux envelope 값:")

def env_at_peaks(peak_times, env, times):
    vals = []
    for t in peak_times:
        idx = np.argmin(np.abs(times - t))
        if 0 <= idx < len(env):
            vals.append(env[idx])
    return np.array(vals)

env_on = env_at_peaks(on_grid, env_base, times)
env_off = env_at_peaks(off_grid, env_base, times)

print(f"  격자 피크:   중앙값={np.median(env_on):.3f}  평균={np.mean(env_on):.3f}")
print(f"  비격자 피크: 중앙값={np.median(env_off):.3f}  평균={np.mean(env_off):.3f}")

# ── 5. novelty_norm과의 대조 ──
print(f"\n{'='*72}")
print(f"  4. novelty_norm과의 대조")
print(f"{'='*72}")

set_nov = set(pk_nov_norm.tolist())

def match_count(peaks, reference, tol=MIN_EVENT_GAP_S):
    matched = 0
    for t in peaks:
        if any(abs(t - r) < tol for r in reference):
            matched += 1
    return matched

on_in_nov = match_count(on_grid, pk_nov_norm)
off_in_nov = match_count(off_grid, pk_nov_norm)

print(f"  격자 피크 {len(on_grid)}개 중 novelty_norm에도 존재: "
      f"{on_in_nov} ({on_in_nov/max(len(on_grid),1)*100:.1f}%)")
print(f"  비격자 피크 {len(off_grid)}개 중 novelty_norm에도 존재: "
      f"{off_in_nov} ({off_in_nov/max(len(off_grid),1)*100:.1f}%)")

# ── 6. 4초 윈도우별 격자/비격자 분포 ──
print(f"\n{'='*72}")
print(f"  5. 4초 윈도우별 격자/비격자 분포")
print(f"{'='*72}")

def count_seg(pk, t0, t1):
    return int(((pk >= t0) & (pk < t1)).sum())

print(f"  {'시각':>5s}  {'총':>4s}  {'격자':>4s}  {'비격자':>4s}  {'격자%':>6s}")
n_win = int(np.ceil(dur / WINDOW_S))
for wi in range(n_win):
    t0, t1 = wi * WINDOW_S, (wi + 1) * WINDOW_S
    total = count_seg(pk_post9, t0, t1)
    n_on = count_seg(on_grid, t0, t1)
    n_off = count_seg(off_grid, t0, t1)
    if total == 0:
        continue
    ts = f"{int(t0)//60}:{int(t0)%60:02d}"
    pct = n_on / total * 100 if total > 0 else 0
    marker = " ◀" if pct > 40 else ""
    print(f"  {ts:>5s}  {total:4d}  {n_on:4d}  {n_off:4d}  {pct:5.1f}%{marker}")

# ── 7. 격자 피크의 시간 분포 ──
print(f"\n{'='*72}")
print(f"  6. 격자 피크 시간 나열 (처음 30개)")
print(f"{'='*72}")
for i, t in enumerate(on_grid[:30]):
    phase = ((t - grid_start) % period_s) * 1000
    if phase > period_s * 1000 / 2:
        phase = period_s * 1000 - phase
    idx = np.argmin(np.abs(times - t))
    sf_val = env_base[idx] if idx < len(env_base) else 0
    print(f"  {i+1:3d}. {t:7.3f}s  위상오차={phase:+6.1f}ms  "
          f"SuperFlux={sf_val:6.3f}")
