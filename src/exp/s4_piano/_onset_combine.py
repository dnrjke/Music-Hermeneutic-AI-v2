"""s4_piano: onset function 조합 실험.

1. no-maxfilter ∩ novelty — 선율 흐름 + 음악적 의미 교집합
2. novelty + 국소 정규화 — 링잉 구간 과다탐지 억제
3. plain flux ∩ novelty — 사건 충실 + 음악적 의미 교집합
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

import numpy as np
import soundfile as sf
import librosa
from scipy.ndimage import uniform_filter1d
from pathlib import Path
from config import (SR, N_FFT, HOP, N_MELS, FMIN, WINDOW_S,
                    MIN_EVENT_GAP_S, OUT_DIR,
                    SUPERFLUX_LAG)
from audio_io import load_mono, duration_s
from onset import superflux_envelope
from peak_pick import otsu

frame_dt = HOP / SR


def count_seg(pk, t0, t1):
    return int(((pk >= t0) & (pk < t1)).sum())


def _click(sr, freq, dur_ms=12.0, amp=0.7):
    n = int(sr * dur_ms / 1000)
    t = np.arange(n, dtype=np.float32) / sr
    e = np.exp(-t * 1000 / dur_ms)
    return (amp * e * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def overlay(mono, pk_times, click):
    out = mono.copy()
    for t in pk_times:
        idx = int(t * SR)
        end = min(idx + len(click), len(out))
        n = end - idx
        if n > 0:
            out[idx:end] += click[:n]
    return np.clip(out, -1, 1)


def overlay_compare(mono, common, new_only, click_c, click_n):
    out = mono.copy()
    for t in common:
        idx = int(t * SR)
        end = min(idx + len(click_c), len(out))
        n = end - idx
        if n > 0:
            out[idx:end] += click_c[:n]
    for t in new_only:
        idx = int(t * SR)
        end = min(idx + len(click_n), len(out))
        n = end - idx
        if n > 0:
            out[idx:end] += click_n[:n]
    return np.clip(out, -1, 1)


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
    order = sorted(loc.tolist(), key=lambda i: -v[i])
    selected = []
    for i in order:
        if all(abs(times[i] - times[j]) >= min_gap_s for j in selected):
            selected.append(i)
    if not selected:
        return np.array([], dtype=np.float64), thr
    return times[np.sort(np.asarray(selected, dtype=int))], thr


def _get_logmel(mono):
    S = np.abs(librosa.stft(mono, n_fft=N_FFT, hop_length=HOP))
    mel = librosa.feature.melspectrogram(S=S**2, sr=SR, n_mels=N_MELS, fmin=FMIN)
    logmel = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    return logmel, S.shape[1]


def onset_no_maxfilter(mono):
    logmel, n = _get_logmel(mono)
    env = librosa.onset.onset_strength(
        S=logmel, sr=SR, hop_length=HOP,
        lag=SUPERFLUX_LAG, max_size=1, detrend=True)
    return librosa.util.fix_length(env, size=n)


def onset_plain_flux(mono):
    logmel, n = _get_logmel(mono)
    env = librosa.onset.onset_strength(
        S=logmel, sr=SR, hop_length=HOP,
        lag=1, max_size=1, detrend=False)
    return librosa.util.fix_length(env, size=n)


def onset_spectral_novelty(mono):
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


def intersect_peaks(pk_a, pk_b, tolerance_s=MIN_EVENT_GAP_S):
    matched = []
    for t in pk_a:
        if any(abs(t - tb) < tolerance_s for tb in pk_b):
            matched.append(t)
    return np.array(sorted(matched))


def novelty_local_norm(env_nov, times, smooth_s=2.0, norm_block_s=2.0):
    env_pos = np.maximum(env_nov, 0.0)
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
        return np.array([], dtype=np.float64), 0.0
    thr = otsu(pos_vals)
    v = normed
    loc = np.flatnonzero(
        (v[1:-1] >= v[:-2]) & (v[1:-1] > v[2:]) & (v[1:-1] > thr)
    ) + 1
    if len(loc) == 0:
        return np.array([], dtype=np.float64), thr
    order = sorted(loc.tolist(), key=lambda i: -v[i])
    selected = []
    for i in order:
        if all(abs(times[i] - times[j]) >= MIN_EVENT_GAP_S for j in selected):
            selected.append(i)
    if not selected:
        return np.array([], dtype=np.float64), thr
    return times[np.sort(np.asarray(selected, dtype=int))], thr


# ── 메인 ──

audio_path = Path(r"E:\game\Music Hermeneutic AI v2\audio\102 - Dir.wav")
print(f"▸ {audio_path.name}")
mono = load_mono(audio_path)
dur = duration_s(audio_path)

env_base, _ = superflux_envelope(mono)
times = librosa.frames_to_time(np.arange(len(env_base)), sr=SR, hop_length=HOP)
print(f"  길이: {dur:.1f}s")

# 기존 결과
pk_base, _ = peaks_otsu(env_base, times)
pk_nomaxf, _ = peaks_otsu(onset_no_maxfilter(mono), times)
pk_plain, _ = peaks_otsu(onset_plain_flux(mono), times)
env_novelty = onset_spectral_novelty(mono)
pk_novelty, _ = peaks_otsu(env_novelty, times)

print(f"  baseline:     {len(pk_base)}")
print(f"  no-maxfilter: {len(pk_nomaxf)}")
print(f"  plain flux:   {len(pk_plain)}")
print(f"  novelty:      {len(pk_novelty)}")

set_base = set(pk_base.tolist())

# ── 3종 조합 ──

# 1. no-maxf ∩ novelty
pk_nmxf_nov = intersect_peaks(pk_nomaxf, pk_novelty)
s1 = set(pk_nmxf_nov.tolist())
print(f"\n  [1] no-maxf ∩ novelty:    {len(pk_nmxf_nov):4d}  "
      f"공통={len(s1 & set_base)}  신규={len(s1 - set_base)}  "
      f"소실={len(set_base - s1)}")

# 2. novelty + 국소정규화
pk_nov_norm, thr_nn = novelty_local_norm(env_novelty, times)
s2 = set(pk_nov_norm.tolist())
print(f"  [2] novelty + norm:       {len(pk_nov_norm):4d}  "
      f"공통={len(s2 & set_base)}  신규={len(s2 - set_base)}  "
      f"소실={len(set_base - s2)}")

# 3. plain flux ∩ novelty
pk_plain_nov = intersect_peaks(pk_plain, pk_novelty)
s3 = set(pk_plain_nov.tolist())
print(f"  [3] plain ∩ novelty:      {len(pk_plain_nov):4d}  "
      f"공통={len(s3 & set_base)}  신규={len(s3 - set_base)}  "
      f"소실={len(set_base - s3)}")

# ── 4초 윈도우별 ──
print(f"\n{'='*72}")
print(f"  4초 윈도우별")
print(f"{'='*72}")
print(f"  {'시각':>5s}  {'base':>5s}  {'∩nmxf':>6s}  {'n+nrm':>6s}  {'∩pln':>6s}")

n_win = int(np.ceil(dur / WINDOW_S))
for wi in range(n_win):
    t0, t1 = wi * WINDOW_S, (wi + 1) * WINDOW_S
    ts = f"{int(t0)//60}:{int(t0)%60:02d}"
    cb = count_seg(pk_base, t0, t1)
    c1 = count_seg(pk_nmxf_nov, t0, t1)
    c2 = count_seg(pk_nov_norm, t0, t1)
    c3 = count_seg(pk_plain_nov, t0, t1)
    print(f"  {ts:>5s}  {cb:5d}  {c1:6d}  {c2:6d}  {c3:6d}")

print(f"  {'합계':>5s}  {len(pk_base):5d}  {len(pk_nmxf_nov):6d}  "
      f"{len(pk_nov_norm):6d}  {len(pk_plain_nov):6d}")

# ── 소니파이 ──
print(f"\n{'='*72}")
print(f"  소니파이 생성")
print(f"{'='*72}")

dest = OUT_DIR / "sonify" / "Dir"
dest.mkdir(parents=True, exist_ok=True)

click3k = _click(SR, 3000.0)
click5k = _click(SR, 5000.0, 15.0, 0.8)

combos = [
    ("intersect",      pk_nmxf_nov, s1,  "no-maxf ∩ novelty"),
    ("novelty_norm",   pk_nov_norm, s2,  "novelty + 국소정규화"),
    ("plain_novelty",  pk_plain_nov, s3, "plain ∩ novelty"),
]

for tag, pk, s, desc in combos:
    common_s = sorted(s & set_base)
    new_s = sorted(s - set_base)

    # 단독
    fname = f"전체_{tag}_클릭.wav"
    sf.write(str(dest / fname), overlay(mono, pk, click3k), SR)
    print(f"  {fname:34s}  {len(pk):4d}  ({desc})")

    # 비교
    fname = f"전체_{tag}_비교_클릭.wav"
    sf.write(str(dest / fname),
             overlay_compare(mono, common_s, new_s, click3k, click5k), SR)
    print(f"  {fname:34s}  공통={len(common_s)} 신규={len(new_s)}")

print(f"\n  비교 소니파이: 3kHz=baseline 공통, 5kHz=신규")
print(f"\n출력: {dest}")
