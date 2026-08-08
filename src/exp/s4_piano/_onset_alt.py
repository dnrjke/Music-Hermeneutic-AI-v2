"""s4_piano: onset function 교체 실험.

SuperFlux가 피아노 타건을 놓치는 구조적 원인:
  - max_size=3: 인접 mel 빈의 높은 에너지가 현재 빈의 증가를 할인
  - lag=2 (~12ms): 직전 잔향 대비 → 새 타건의 차이가 작음
  - detrend: 평균 제거 → 약한 사건이 음수로 밀려남

4종 대안:
  1. plain flux — lag=1, max_size=1, detrend 없음. 순수 양의 스펙트럼 변화량
  2. no-maxfilter — lag=2, max_size=1. SuperFlux에서 주파수 스미어만 제거
  3. spectral novelty — 인접 프레임 간 코사인 거리. 진폭 불변, 스펙트럼 형태 변화 감지
  4. per-band Otsu — low/mid/high 각각 독립 Otsu, 합산. 대역별 적정 임계

각각 + 기존 baseline의 소니파이 생성.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from config import (SR, N_FFT, HOP, N_MELS, FMIN, WINDOW_S,
                    MIN_EVENT_GAP_S, OUT_DIR,
                    SUPERFLUX_LAG, SUPERFLUX_MAX)
from audio_io import load_mono, duration_s
from onset import superflux_envelope, band_envelopes
from peak_pick import otsu


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


# ── onset function 구현 ──

def _get_logmel(mono):
    """공통 logmel 계산."""
    S = np.abs(librosa.stft(mono, n_fft=N_FFT, hop_length=HOP))
    mel = librosa.feature.melspectrogram(S=S**2, sr=SR, n_mels=N_MELS, fmin=FMIN)
    logmel = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    return logmel, S.shape[1]


def onset_plain_flux(mono):
    """1. 순수 스펙트럼 플럭스. lag=1, max_size=1, detrend=False."""
    logmel, n = _get_logmel(mono)
    env = librosa.onset.onset_strength(
        S=logmel, sr=SR, hop_length=HOP,
        lag=1, max_size=1, detrend=False)
    return librosa.util.fix_length(env, size=n)


def onset_no_maxfilter(mono):
    """2. SuperFlux에서 주파수 최대 필터만 제거. lag=2, max_size=1."""
    logmel, n = _get_logmel(mono)
    env = librosa.onset.onset_strength(
        S=logmel, sr=SR, hop_length=HOP,
        lag=SUPERFLUX_LAG, max_size=1, detrend=True)
    return librosa.util.fix_length(env, size=n)


def onset_spectral_novelty(mono):
    """3. 스펙트럼 코사인 거리. 인접 프레임 간 스펙트럼 형태 변화.

    코사인 거리 = 1 - cos_sim. 진폭 스케일 불변.
    새 타건: 새 고조파 출현 → 스펙트럼 형태 변화 → 높은 거리.
    잔향 지속: 동일 고조파 유지 → 형태 불변 → 낮은 거리.
    """
    logmel, n = _get_logmel(mono)
    # 프레임별 정규화 (unit vector)
    norms = np.linalg.norm(logmel, axis=0, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normed = logmel / norms
    # 인접 프레임 코사인 유사도
    cos_sim = np.sum(normed[:, 1:] * normed[:, :-1], axis=0)
    cos_sim = np.clip(cos_sim, -1, 1)
    distance = 1 - cos_sim
    env = np.zeros(n, dtype=np.float64)
    env[1:1 + len(distance)] = np.maximum(distance, 0)
    return env


def peaks_perband(mono, times, min_gap_s=MIN_EVENT_GAP_S):
    """4. 대역별 독립 Otsu + 합산.

    low/mid/high 각각 Otsu → 각 대역 피크 → 전체 탐욕 선택.
    mid 대역 Otsu는 비트가 아닌 피아노 역학에 맞춰짐.
    """
    bands = band_envelopes(mono)
    n = len(times)

    all_candidates = []
    band_stats = {}
    for bname, benv in bands.items():
        benv = librosa.util.fix_length(benv, size=n)
        fin = np.isfinite(benv)
        if fin.sum() < 3:
            band_stats[bname] = (0, 0.0)
            continue
        v = np.where(fin, benv, -np.inf)
        thr = otsu(benv[fin])
        loc = np.flatnonzero(
            (v[1:-1] >= v[:-2]) & (v[1:-1] > v[2:]) & (v[1:-1] > thr)
        ) + 1
        band_stats[bname] = (len(loc), thr)
        for i in loc:
            all_candidates.append((i, float(v[i]), bname))

    if not all_candidates:
        return np.array([], dtype=np.float64), band_stats

    # 전체에서 탐욕 선택 (진폭 내림차순)
    all_candidates.sort(key=lambda x: -x[1])
    selected = []
    for i, v, bn in all_candidates:
        if all(abs(times[i] - times[j]) >= min_gap_s for j in selected):
            selected.append(i)

    if not selected:
        return np.array([], dtype=np.float64), band_stats
    return times[np.sort(np.asarray(selected, dtype=int))], band_stats


# ── 메인 ──

audio_path = Path(r"E:\game\Music Hermeneutic AI v2\audio\102 - Dir.wav")
print(f"▸ {audio_path.name}")
mono = load_mono(audio_path)
dur = duration_s(audio_path)
env_base, times_base = superflux_envelope(mono)
times = librosa.frames_to_time(np.arange(len(env_base)), sr=SR, hop_length=HOP)
print(f"  길이: {dur:.1f}s  프레임: {len(env_base)}")

# ── 기존 baseline ──
pk_base, thr_base = peaks_otsu(env_base, times)
print(f"\n  [baseline] SuperFlux (lag={SUPERFLUX_LAG}, max={SUPERFLUX_MAX}, detrend)")
print(f"    Otsu: {thr_base:.4f}  피크: {len(pk_base)}")

# ── 4종 대안 ──
env1 = onset_plain_flux(mono)
pk1, thr1 = peaks_otsu(env1, times)
print(f"\n  [plain] lag=1, max=1, no detrend")
print(f"    Otsu: {thr1:.4f}  피크: {len(pk1)}")

env2 = onset_no_maxfilter(mono)
pk2, thr2 = peaks_otsu(env2, times)
print(f"\n  [no-maxf] lag=2, max=1, detrend")
print(f"    Otsu: {thr2:.4f}  피크: {len(pk2)}")

env3 = onset_spectral_novelty(mono)
pk3, thr3 = peaks_otsu(env3, times)
print(f"\n  [novelty] 코사인 거리")
print(f"    Otsu: {thr3:.4f}  피크: {len(pk3)}")

pk4, bstats = peaks_perband(mono, times)
print(f"\n  [perband] 대역별 독립 Otsu")
for bn, (cnt, thr) in bstats.items():
    print(f"    {bn:>5s}: Otsu={thr:.4f}  극대점={cnt}")
print(f"    합산 피크: {len(pk4)}")

# ── baseline 대비 차이 ──
print(f"\n{'='*72}")
print(f"  baseline 대비 차이")
print(f"{'='*72}")

set_base = set(pk_base.tolist())
variants = [
    ("plain",   pk1),
    ("no-maxf", pk2),
    ("novelty", pk3),
    ("perband", pk4),
]

for name, pk in variants:
    s = set(pk.tolist())
    common = s & set_base
    new = s - set_base
    lost = set_base - s
    print(f"  {name:>8s}: 총={len(pk):4d}  공통={len(common):4d}  "
          f"신규={len(new):4d}  소실={len(lost):4d}")

# ── 4초 윈도우별 ──
print(f"\n{'='*72}")
print(f"  4초 윈도우별")
print(f"{'='*72}")
header = f"  {'시각':>5s}  {'base':>5s}"
for name, _ in variants:
    header += f"  {name:>8s}"
print(header)

n_win = int(np.ceil(dur / WINDOW_S))
for wi in range(n_win):
    t0, t1 = wi * WINDOW_S, (wi + 1) * WINDOW_S
    ts = f"{int(t0)//60}:{int(t0)%60:02d}"
    line = f"  {ts:>5s}  {count_seg(pk_base, t0, t1):5d}"
    for name, pk in variants:
        line += f"  {count_seg(pk, t0, t1):8d}"
    print(line)

totals = f"  {'합계':>5s}  {len(pk_base):5d}"
for _, pk in variants:
    totals += f"  {len(pk):8d}"
print(totals)

# ── 소니파이 ──
print(f"\n{'='*72}")
print(f"  소니파이 생성")
print(f"{'='*72}")

dest = OUT_DIR / "sonify" / "Dir"
dest.mkdir(parents=True, exist_ok=True)

click3k = _click(SR, 3000.0)
click5k = _click(SR, 5000.0, 15.0, 0.8)

files = [
    ("전체_baseline_클릭.wav",  pk_base,  "baseline"),
    ("전체_plain_클릭.wav",     pk1,      "plain flux"),
    ("전체_nomaxf_클릭.wav",    pk2,      "no-maxfilter"),
    ("전체_novelty_클릭.wav",   pk3,      "spectral novelty"),
    ("전체_perband_클릭.wav",   pk4,      "per-band Otsu"),
]

for fname, pk, desc in files:
    sf.write(str(dest / fname), overlay(mono, pk, click3k), SR)
    print(f"  {fname:30s}  {len(pk):4d}  ({desc})")

# 각 variant의 신규 피크만 별도 소니파이 (5kHz)
for name, pk in variants:
    new_only = sorted(set(pk.tolist()) - set_base)
    if len(new_only) < 3:
        continue
    fname = f"전체_{name}_신규_클릭.wav"
    sf.write(str(dest / fname),
             overlay(mono, np.array(new_only), click5k), SR)
    print(f"  {fname:30s}  {len(new_only):4d}  (5kHz, 신규만)")

# 각 variant의 비교 소니파이 (공통=3kHz, 신규=5kHz)
for name, pk in variants:
    s = set(pk.tolist())
    common = sorted(s & set_base)
    new_only = sorted(s - set_base)
    if len(new_only) < 3:
        continue
    out = mono.copy()
    for t in common:
        idx = int(t * SR)
        end = min(idx + len(click3k), len(out))
        n = end - idx
        if n > 0:
            out[idx:end] += click3k[:n]
    for t in new_only:
        idx = int(t * SR)
        end = min(idx + len(click5k), len(out))
        n = end - idx
        if n > 0:
            out[idx:end] += click5k[:n]
    out = np.clip(out, -1, 1)
    fname = f"전체_{name}_비교_클릭.wav"
    sf.write(str(dest / fname), out, SR)
    print(f"  {fname:30s}  공통={len(common)} 신규={len(new_only)}")

print(f"\n  비교 소니파이: 3kHz=baseline과 공통, 5kHz=해당 variant에만 있는 신규")
print(f"\n출력: {dest}")
