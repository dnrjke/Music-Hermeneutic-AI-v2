"""s4_piano: Dir 피아노 온셋 탐지 — 2-pass "별 억제" 파이프라인 v2.

v1 실패 교훈: 강한 사건 근방 마스킹 → 트랙 전체 차폐. 음악에선 강한/여린
사건이 시간적으로 겹치므로 공간 분리(Veil 별 억제)와 달리 마스킹 불가.

v2 접근: 마스킹 없이, 밴드패스 + 국소 정규화만으로 여린 사건을 현상.
  Pass 1: SuperFlux → Otsu (기존)
  Pass 2: onset - smooth(onset, 2s) → 국소 정규화(블록 2s/4s) → Otsu
  합산: Pass 1 ∪ Pass 2
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from pathlib import Path
from config import (SR, HOP, WINDOW_S, MIN_EVENT_GAP_S, OUT_DIR)
from audio_io import load_mono, duration_s
from onset import superflux_envelope, band_envelopes
from peak_pick import otsu


# ── 유틸 ──

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


def overlay_diff(mono, common, p1_only, p2_only,
                 click_common, click_p1, click_p2):
    out = mono.copy()
    for t in common:
        idx = int(t * SR)
        end = min(idx + len(click_common), len(out))
        n = end - idx
        if n > 0:
            out[idx:end] += click_common[:n]
    for t in pass1_only:
        idx = int(t * SR)
        end = min(idx + len(click_p1), len(out))
        n = end - idx
        if n > 0:
            out[idx:end] += click_p1[:n]
    for t in p2_only:
        idx = int(t * SR)
        end = min(idx + len(click_p2), len(out))
        n = end - idx
        if n > 0:
            out[idx:end] += click_p2[:n]
    return np.clip(out, -1, 1)


def greedy_select(indices, values, times, min_gap_s=MIN_EVENT_GAP_S):
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
    pk = greedy_select(loc, v, times, min_gap_s)
    return pk, thr


# ── Pass 2: 밴드패스 잔차 + 국소 정규화 ──

def pass2_bandpass_norm(env, times, smooth_s=2.0, norm_block_s=2.0):
    """밴드패스(onset - smooth) + 국소 99-pct 정규화 + Otsu.

    Veil 대응: blur(σ=2)-blur(σ=40) 밴드패스로 저주파 글로우 제거 후
    per-channel 99-pct 정규화.

    마스킹 없음 — 국소 정규화가 강한 사건 근방의 큰 잔차를 자체 블록 내에서
    정규화하므로, Otsu가 전역적으로 낮아질 수 있다.
    """
    env_pos = np.maximum(env, 0.0)
    frame_dt = HOP / SR

    # 밴드패스: onset - smooth(onset)
    smooth_frames = max(3, int(smooth_s / frame_dt) | 1)
    env_smooth = uniform_filter1d(env_pos, size=smooth_frames, mode='reflect')
    residual = np.maximum(env_pos - env_smooth, 0.0)

    # 국소 99-pct 정규화
    dur = times[-1] + frame_dt
    n_blocks = int(np.ceil(dur / norm_block_s))
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

    # Otsu 피크 검출 (양수값에 대해)
    pos_vals = normed[normed > 0]
    if len(pos_vals) < 10:
        return np.array([], dtype=np.float64), 0.0, residual, normed
    thr = otsu(pos_vals)
    loc = np.flatnonzero(
        (normed[1:-1] >= normed[:-2]) &
        (normed[1:-1] > normed[2:]) &
        (normed[1:-1] > thr)
    ) + 1
    if len(loc) == 0:
        return np.array([], dtype=np.float64), thr, residual, normed
    pk = greedy_select(loc, normed, times)
    return pk, thr, residual, normed


# ── 국소 정규화 직접 적용 (비교군) ──

def local_norm_detect(env, times, block_s=2.0):
    """밴드패스 없이 국소 정규화만. 블록 크기 변화의 효과 확인용."""
    env_pos = np.maximum(env, 0.0)
    frame_dt = HOP / SR
    dur = times[-1] + frame_dt
    n_blocks = int(np.ceil(dur / block_s))
    normed = np.zeros_like(env_pos)
    for bi in range(n_blocks):
        t0, t1 = bi * block_s, (bi + 1) * block_s
        mask = (times >= t0) & (times < t1)
        seg = env_pos[mask]
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
    return greedy_select(loc, normed, times), thr


# ── 메인 ──

audio_path = Path(r"E:\game\Music Hermeneutic AI v2\audio\102 - Dir.wav")
print(f"▸ {audio_path.name}")
mono = load_mono(audio_path)
dur = duration_s(audio_path)
env, times = superflux_envelope(mono)
print(f"  길이: {dur:.1f}s  프레임: {len(env)}")

# ── Pass 1 ──
pk1, thr1 = peaks_otsu(env, times)
print(f"\n  Pass 1 (기존 Otsu): {len(pk1)}  임계={thr1:.4f}")

# ── Pass 2 변형들 ──
configs = [
    ("bp2s+norm2s",  2.0, 2.0),  # 밴드패스 2s + norm 2s
    ("bp2s+norm4s",  2.0, 4.0),  # 밴드패스 2s + norm 4s
    ("bp1s+norm2s",  1.0, 2.0),  # 밴드패스 1s + norm 2s
]

p2_results = {}
for label, smooth_s, norm_s in configs:
    pk2, thr2, res, nrm = pass2_bandpass_norm(
        env, times, smooth_s=smooth_s, norm_block_s=norm_s)
    p2_results[label] = pk2
    print(f"  Pass 2 [{label}]: {len(pk2):4d}  Otsu={thr2:.4f}  "
          f"잔차max={res.max():.3f}")

# 국소 정규화 직접 비교
ln_configs = [
    ("norm2s", 2.0),
    ("norm4s", 4.0),
    ("norm16s", 16.0),
]

ln_results = {}
for label, block_s in ln_configs:
    pk, thr = local_norm_detect(env, times, block_s=block_s)
    ln_results[label] = pk
    print(f"  Local norm [{label}]: {len(pk):4d}  Otsu={thr:.4f}")

# ── 합산 결과 비교 ──
print(f"\n{'='*72}")
print(f"  합산 결과 비교 (Pass 1 ∪ Pass 2)")
print(f"{'='*72}")

set1 = set(pk1.tolist())
all_variants = {}

for label, pk2 in p2_results.items():
    s2 = set(pk2.tolist())
    union = sorted(set1 | s2)
    p2_only = sorted(s2 - set1)
    all_variants[f"2p_{label}"] = (union, p2_only)
    print(f"  2pass [{label}]: 합산={len(union):4d}  신규={len(p2_only):4d}")

for label, pk in ln_results.items():
    s = set(pk.tolist())
    union = sorted(set1 | s)
    new_only = sorted(s - set1)
    all_variants[f"ln_{label}"] = (union, new_only)
    print(f"  norm  [{label:>5s}]: 합산={len(union):4d}  신규={len(new_only):4d}")

# ── 4초 윈도우별 비교 (최유망 후보) ──
# 가장 많은 신규를 찾은 variant 선택
best_name = max(all_variants, key=lambda k: len(all_variants[k][1]))
best_union, best_new = all_variants[best_name]
print(f"\n  최유망: {best_name} (신규 {len(best_new)})")

print(f"\n{'='*72}")
print(f"  4초 윈도우별 — Pass 1 vs {best_name}")
print(f"{'='*72}")
print(f"  {'시각':>5s}  {'P1':>5s}  {'best':>5s}  {'diff':>5s}")

n_win = int(np.ceil(dur / WINDOW_S))
pk_best = np.array(best_union)
for wi in range(n_win):
    t0, t1 = wi * WINDOW_S, (wi + 1) * WINDOW_S
    ts = f"{int(t0)//60}:{int(t0)%60:02d}"
    c1 = count_seg(pk1, t0, t1)
    cb = count_seg(pk_best, t0, t1)
    d = cb - c1
    marker = f" ◀ +{d}" if d > 0 else ""
    print(f"  {ts:>5s}  {c1:5d}  {cb:5d}  {d:+5d}{marker}")

print(f"  {'합계':>5s}  {len(pk1):5d}  {len(best_union):5d}  "
      f"{len(best_union)-len(pk1):+5d}")

# ── 소니파이 생성 ──
print(f"\n{'='*72}")
print(f"  소니파이 생성")
print(f"{'='*72}")

dest = OUT_DIR / "sonify" / "Dir"
dest.mkdir(parents=True, exist_ok=True)

click3k = _click(SR, 3000.0)
click5k = _click(SR, 5000.0, 15.0, 0.8)
click1k5 = _click(SR, 1500.0, 15.0, 0.8)

# 1. baseline
sf.write(str(dest / "전체_baseline_클릭.wav"),
         overlay(mono, pk1, click3k), SR)
print(f"  전체_baseline_클릭.wav          {len(pk1):4d}")

# 최유망 variant의 소니파이
common = sorted(set(pk1.tolist()) & set(best_union))
pass1_only = sorted(set(pk1.tolist()) - set(best_union))
pass2_only = sorted(set(best_union) - set(pk1.tolist()))

# 2. 합산 단독
sf.write(str(dest / "전체_2pass_클릭.wav"),
         overlay(mono, pk_best, click3k), SR)
print(f"  전체_2pass_클릭.wav             {len(best_union):4d}")

# 3. Pass 2 신규만
if len(pass2_only) > 0:
    sf.write(str(dest / "전체_pass2_신규_클릭.wav"),
             overlay(mono, np.array(pass2_only), click5k), SR)
    print(f"  전체_pass2_신규_클릭.wav        {len(pass2_only):4d} (5kHz)")

# 4. 비교 (공통=3kHz, P1전용=1.5kHz, P2신규=5kHz)
sf.write(str(dest / "전체_2pass_비교_클릭.wav"),
         overlay_diff(mono, common, pass1_only, pass2_only,
                       click3k, click1k5, click5k), SR)
print(f"  전체_2pass_비교_클릭.wav        공통={len(common)} "
      f"P1전용={len(pass1_only)} 신규={len(pass2_only)}")

# 5. 상위 2-3개 variant도 추가 소니파이
for vname, (vunion, vnew) in sorted(all_variants.items(),
                                      key=lambda x: -len(x[1][1]))[:3]:
    if vname == best_name:
        continue
    pk_v = np.array(vunion)
    fname = f"전체_{vname}_클릭.wav"
    sf.write(str(dest / fname), overlay(mono, pk_v, click3k), SR)
    print(f"  {fname:34s}  {len(vunion):4d} (신규={len(vnew)})")

print(f"\n  비교 소니파이 범례:")
print(f"    3kHz = 공통 / 합산")
print(f"    1.5kHz = Pass 1 전용")
print(f"    5kHz = Pass 2 신규 (여린 사건)")

print(f"\n출력: {dest}")
