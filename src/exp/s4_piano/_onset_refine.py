"""s4_piano: log1p / novelty 개선 — 피아노 멜로디 추적.

방향: SuperFlux baseline의 등간격 비트 artifact를 회피하고
      피아노 메인 멜로디에 반응하는 onset 도출.

소스 2종:
  - log1p: np.log1p(SuperFlux) → 동적 범위 압축, 강한 비트 억제
  - novelty: 코사인 거리 → 진폭 불변, 스펙트럼 형태 변화

처리: bandpass(smooth 제거) + 국소 99-pct 정규화 (블록 1s/2s/4s)
deburst: 30ms(기본) vs 100ms(burst 억제) — 각각 소니파이
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
                    MIN_EVENT_GAP_S, OUT_DIR, SUPERFLUX_LAG)
from audio_io import load_mono, duration_s
from onset import superflux_envelope
from peak_pick import otsu

frame_dt = HOP / SR


# ── 유틸 ──

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


# ── envelope 소스 ──

def _get_logmel(mono):
    S = np.abs(librosa.stft(mono, n_fft=N_FFT, hop_length=HOP))
    mel = librosa.feature.melspectrogram(S=S**2, sr=SR, n_mels=N_MELS, fmin=FMIN)
    logmel = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    return logmel, S.shape[1]


def env_log1p(mono):
    env_base, _ = superflux_envelope(mono)
    env_pos = np.maximum(env_base, 0.0)
    return np.log1p(env_pos)


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


# ── 공통 처리: bandpass + local norm + peak pick ──

def bandpass_norm_peaks(env, times, smooth_s=2.0, norm_block_s=2.0,
                        min_gap_s=MIN_EVENT_GAP_S):
    """bandpass(smooth 제거) + 국소 99-pct 정규화 + Otsu."""
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
        return np.array([], dtype=np.float64), 0.0
    thr = otsu(pos_vals)
    loc = np.flatnonzero(
        (normed[1:-1] >= normed[:-2]) &
        (normed[1:-1] > normed[2:]) &
        (normed[1:-1] > thr)
    ) + 1
    if len(loc) == 0:
        return np.array([], dtype=np.float64), thr
    return greedy_select(loc, normed, times, min_gap_s), thr


def raw_peaks(env, times, min_gap_s=MIN_EVENT_GAP_S):
    """bandpass/norm 없이 직접 Otsu."""
    env_pos = np.maximum(env, 0.0)
    fin = np.isfinite(env_pos)
    if fin.sum() < 3:
        return np.array([], dtype=np.float64), 0.0
    thr = otsu(env_pos[fin])
    v = np.where(fin, env_pos, -np.inf)
    loc = np.flatnonzero(
        (v[1:-1] >= v[:-2]) & (v[1:-1] > v[2:]) & (v[1:-1] > thr)
    ) + 1
    if len(loc) == 0:
        return np.array([], dtype=np.float64), thr
    return greedy_select(loc, v, times, min_gap_s), thr


# ── 메인 ──

audio_path = Path(r"E:\game\Music Hermeneutic AI v2\audio\102 - Dir.wav")
print(f"▸ {audio_path.name}")
mono = load_mono(audio_path)
dur = duration_s(audio_path)

env_base, _ = superflux_envelope(mono)
times = librosa.frames_to_time(np.arange(len(env_base)), sr=SR, hop_length=HOP)
print(f"  길이: {dur:.1f}s")

# baseline
pk_base, _ = raw_peaks(env_base, times)
set_base = set(pk_base.tolist())
print(f"  baseline: {len(pk_base)}")

# ── envelope 소스 계산 ──
print(f"\n  envelope 소스 계산...")
e_log1p = env_log1p(mono)
e_novelty = env_spectral_novelty(mono)
e_novelty = librosa.util.fix_length(e_novelty, size=len(times))

# raw (bandpass/norm 없이)
pk_log1p_raw, _ = raw_peaks(e_log1p, times)
pk_nov_raw, _ = raw_peaks(e_novelty, times)
print(f"  log1p raw:    {len(pk_log1p_raw)}")
print(f"  novelty raw:  {len(pk_nov_raw)}")

# ── 전 조합 계산 ──
BLOCK_SIZES = [1.0, 2.0, 4.0]
GAP_CONFIGS = [
    ("",    MIN_EVENT_GAP_S),   # 30ms 기본
    ("_db", 0.100),             # 100ms deburst
]
SOURCES = [
    ("log1p",   e_log1p),
    ("nov",     e_novelty),
]

results = {}

print(f"\n{'='*72}")
print(f"  조합별 피크 수")
print(f"{'='*72}")
print(f"  {'variant':>25s}  {'총':>5s}  {'b공통':>5s}  {'신규':>5s}  {'소실':>5s}")

for src_name, env in SOURCES:
    for block_s in BLOCK_SIZES:
        for gap_tag, gap_s in GAP_CONFIGS:
            tag = f"{src_name}_n{int(block_s)}s{gap_tag}"
            pk, thr = bandpass_norm_peaks(
                env, times, smooth_s=2.0,
                norm_block_s=block_s, min_gap_s=gap_s)
            results[tag] = pk
            s = set(pk.tolist())
            print(f"  {tag:>25s}  {len(pk):5d}  {len(s & set_base):5d}  "
                  f"{len(s - set_base):5d}  {len(set_base - s):5d}")

# raw도 추가
results["log1p_raw"] = pk_log1p_raw
results["nov_raw"] = pk_nov_raw

# ── 연속 발화 분석 ──
print(f"\n{'='*72}")
print(f"  연속 발화 분석")
print(f"{'='*72}")
for tag, pk in sorted(results.items()):
    if len(pk) < 2:
        continue
    gaps = np.diff(pk) * 1000
    n50 = (gaps < 50).sum()
    n100 = (gaps < 100).sum()
    print(f"  {tag:>25s}: 중앙값={np.median(gaps):6.1f}ms  "
          f"<50ms:{n50:3d}  <100ms:{n100:3d}")

# ── 4초 윈도우별 (주요 variant만) ──
print(f"\n{'='*72}")
print(f"  4초 윈도우별")
print(f"{'='*72}")

show_tags = ["log1p_n2s", "log1p_n2s_db", "nov_n2s", "nov_n2s_db"]
header = f"  {'시각':>5s}  {'base':>5s}"
for t in show_tags:
    header += f"  {t:>14s}"
print(header)

n_win = int(np.ceil(dur / WINDOW_S))
for wi in range(n_win):
    t0, t1 = wi * WINDOW_S, (wi + 1) * WINDOW_S
    ts = f"{int(t0)//60}:{int(t0)%60:02d}"
    line = f"  {ts:>5s}  {count_seg(pk_base, t0, t1):5d}"
    for t in show_tags:
        line += f"  {count_seg(results[t], t0, t1):14d}"
    print(line)

line = f"  {'합계':>5s}  {len(pk_base):5d}"
for t in show_tags:
    line += f"  {len(results[t]):14d}"
print(line)

# ── 소니파이 ──
print(f"\n{'='*72}")
print(f"  소니파이 생성")
print(f"{'='*72}")

dest = OUT_DIR / "sonify" / "Dir"
dest.mkdir(parents=True, exist_ok=True)

click3k = _click(SR, 3000.0)
click5k = _click(SR, 5000.0, 15.0, 0.8)

# raw 소니파이
for tag in ["log1p_raw", "nov_raw"]:
    pk = results[tag]
    fname = f"전체_{tag}_클릭.wav"
    sf.write(str(dest / fname), overlay(mono, pk, click3k), SR)
    print(f"  {fname:40s}  {len(pk):4d}")

# 전 조합 소니파이
for tag, pk in sorted(results.items()):
    if tag.endswith("_raw"):
        continue
    s = set(pk.tolist())
    common_s = sorted(s & set_base)
    new_s = sorted(s - set_base)

    fname = f"전체_{tag}_클릭.wav"
    sf.write(str(dest / fname), overlay(mono, pk, click3k), SR)
    print(f"  {fname:40s}  {len(pk):4d}")

    fname = f"전체_{tag}_비교_클릭.wav"
    sf.write(str(dest / fname),
             overlay_compare(mono, common_s, new_s, click3k, click5k), SR)
    print(f"  {fname:40s}  공통={len(common_s)} 신규={len(new_s)}")

print(f"\n  비교 소니파이: 3kHz=baseline 공통, 5kHz=신규")
print(f"\n출력: {dest}")
