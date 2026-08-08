"""s4_piano: novelty 연속 발화(burst) 억제 실험.

문제: novelty가 링잉/폴리포닉 구간에서 매 프레임 높은 값 → 30ms 간격으로
빼곡히 찍혀 "두두두두" 연속 클릭 발생.

대응:
  1. wide gap — novelty에 100ms 최소 간격 (피아노 최속 타건 ~8/s에 대응)
  2. prominence — scipy peak_prominences로 burst 내 중복 제거
각각 no-maxf 게이트 적용.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

import numpy as np
import soundfile as sf
import librosa
from scipy.signal import find_peaks, peak_prominences
from pathlib import Path
from config import (SR, N_FFT, HOP, N_MELS, FMIN, WINDOW_S,
                    MIN_EVENT_GAP_S, OUT_DIR, SUPERFLUX_LAG)
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


def onset_no_maxfilter(mono):
    logmel, n = _get_logmel(mono)
    env = librosa.onset.onset_strength(
        S=logmel, sr=SR, hop_length=HOP,
        lag=SUPERFLUX_LAG, max_size=1, detrend=True)
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


# ── novelty deburst 방법들 ──

def novelty_wide_gap(env_nov, times, min_gap_s=0.100):
    """1. 넓은 최소 간격 — burst에서 최대값만 생존."""
    fin = np.isfinite(env_nov)
    v = np.where(fin, env_nov, -np.inf)
    thr = otsu(env_nov[fin])
    loc = np.flatnonzero(
        (v[1:-1] >= v[:-2]) & (v[1:-1] > v[2:]) & (v[1:-1] > thr)
    ) + 1
    if len(loc) == 0:
        return np.array([], dtype=np.float64), thr
    return greedy_select(loc, v, times, min_gap_s), thr


def novelty_prominence(env_nov, times, min_gap_s=MIN_EVENT_GAP_S):
    """2. peak prominence 필터 — 돌출도 낮은 burst 내 중복 제거.

    prominence = 피크 높이 - 양옆 최저 골짜기 중 높은 쪽.
    burst 내부의 작은 요철은 prominence가 낮아 자연 제거.
    prominence 임계는 Otsu로 결정 (파라미터 0개).
    """
    env_pos = np.maximum(env_nov, 0.0)
    # scipy find_peaks로 모든 극대점
    peak_idx, _ = find_peaks(env_pos)
    if len(peak_idx) < 3:
        return np.array([], dtype=np.float64), 0.0

    proms, _, _ = peak_prominences(env_pos, peak_idx)

    # prominence에 Otsu → 유의미한 돌출만 선택
    thr_prom = otsu(proms)
    significant = peak_idx[proms > thr_prom]

    if len(significant) == 0:
        return np.array([], dtype=np.float64), thr_prom

    return greedy_select(significant, env_pos, times, min_gap_s), thr_prom


# ── 메인 ──

audio_path = Path(r"E:\game\Music Hermeneutic AI v2\audio\102 - Dir.wav")
print(f"▸ {audio_path.name}")
mono = load_mono(audio_path)
dur = duration_s(audio_path)

env_base, _ = superflux_envelope(mono)
times = librosa.frames_to_time(np.arange(len(env_base)), sr=SR, hop_length=HOP)
print(f"  길이: {dur:.1f}s")

# 기존
pk_base, _ = peaks_otsu(env_base, times)
pk_nomaxf, _ = peaks_otsu(onset_no_maxfilter(mono), times)
env_novelty = onset_spectral_novelty(mono)
pk_novelty, _ = peaks_otsu(env_novelty, times)

print(f"  baseline:       {len(pk_base)}")
print(f"  no-maxfilter:   {len(pk_nomaxf)}")
print(f"  novelty (raw):  {len(pk_novelty)}")

set_base = set(pk_base.tolist())

# 기존 교집합 (이전 실험)
pk_nmxf_nov = intersect_peaks(pk_nomaxf, pk_novelty)
s_prev = set(pk_nmxf_nov.tolist())

# ── deburst novelty ──

# 1. wide gap (100ms)
pk_nov_wide, thr_w = novelty_wide_gap(env_novelty, times, min_gap_s=0.100)
print(f"\n  novelty wide gap (100ms): {len(pk_nov_wide)}")

# 2. prominence filter
pk_nov_prom, thr_p = novelty_prominence(env_novelty, times)
print(f"  novelty prominence:       {len(pk_nov_prom)}  "
      f"(prom Otsu={thr_p:.6f})")

# ── 각각 no-maxf 게이트 ──

pk_gate_wide = intersect_peaks(pk_nomaxf, pk_nov_wide)
pk_gate_prom = intersect_peaks(pk_nomaxf, pk_nov_prom)

s_wide = set(pk_gate_wide.tolist())
s_prom = set(pk_gate_prom.tolist())

print(f"\n  게이트 결과:")
print(f"  {'variant':>20s}  {'총':>5s}  {'b공통':>5s}  {'신규':>5s}  {'소실':>5s}")

for label, s in [("이전(∩ raw)", s_prev),
                 ("∩ wide100ms", s_wide),
                 ("∩ prominence", s_prom)]:
    print(f"  {label:>20s}  {len(s):5d}  {len(s & set_base):5d}  "
          f"{len(s - set_base):5d}  {len(set_base - s):5d}")

# ── burst 분석: 연속 발화 구간 확인 ──
print(f"\n{'='*72}")
print(f"  연속 발화 분석 (novelty 피크 간격 분포)")
print(f"{'='*72}")

for label, pk in [("raw novelty", pk_novelty),
                  ("wide 100ms", pk_nov_wide),
                  ("prominence", pk_nov_prom)]:
    if len(pk) < 2:
        continue
    gaps = np.diff(pk) * 1000  # ms
    print(f"  {label:>15s}: 간격 중앙값={np.median(gaps):.1f}ms  "
          f"최소={gaps.min():.1f}ms  "
          f"<50ms: {(gaps < 50).sum()}개  "
          f"<100ms: {(gaps < 100).sum()}개")

# ── 4초 윈도우별 ──
print(f"\n{'='*72}")
print(f"  4초 윈도우별")
print(f"{'='*72}")
print(f"  {'시각':>5s}  {'base':>5s}  {'∩raw':>5s}  {'∩wide':>6s}  {'∩prom':>6s}")

n_win = int(np.ceil(dur / WINDOW_S))
for wi in range(n_win):
    t0, t1 = wi * WINDOW_S, (wi + 1) * WINDOW_S
    ts = f"{int(t0)//60}:{int(t0)%60:02d}"
    cb = count_seg(pk_base, t0, t1)
    cr = count_seg(pk_nmxf_nov, t0, t1)
    cw = count_seg(pk_gate_wide, t0, t1)
    cp = count_seg(pk_gate_prom, t0, t1)
    print(f"  {ts:>5s}  {cb:5d}  {cr:5d}  {cw:6d}  {cp:6d}")

print(f"  {'합계':>5s}  {len(pk_base):5d}  {len(pk_nmxf_nov):5d}  "
      f"{len(pk_gate_wide):6d}  {len(pk_gate_prom):6d}")

# ── 소니파이 ──
print(f"\n{'='*72}")
print(f"  소니파이 생성")
print(f"{'='*72}")

dest = OUT_DIR / "sonify" / "Dir"
dest.mkdir(parents=True, exist_ok=True)

click3k = _click(SR, 3000.0)
click5k = _click(SR, 5000.0, 15.0, 0.8)

combos = [
    ("intersect_wide",  pk_gate_wide, s_wide, "no-maxf ∩ novelty(100ms)"),
    ("intersect_prom",  pk_gate_prom, s_prom, "no-maxf ∩ novelty(prominence)"),
]

for tag, pk, s, desc in combos:
    common_s = sorted(s & set_base)
    new_s = sorted(s - set_base)

    fname = f"전체_{tag}_클릭.wav"
    sf.write(str(dest / fname), overlay(mono, pk, click3k), SR)
    print(f"  {fname:36s}  {len(pk):4d}  ({desc})")

    fname = f"전체_{tag}_비교_클릭.wav"
    sf.write(str(dest / fname),
             overlay_compare(mono, common_s, new_s, click3k, click5k), SR)
    print(f"  {fname:36s}  공통={len(common_s)} 신규={len(new_s)}")

print(f"\n  비교 소니파이: 3kHz=baseline 공통, 5kHz=신규")
print(f"\n출력: {dest}")
