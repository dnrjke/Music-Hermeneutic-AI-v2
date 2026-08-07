"""asinh 다이내믹 레인지 압축 + 배경 감산 진단.

성운 파이프라인(process_nebula_v2.py)에서:
  - asinh stretch: 밝은 피크 압축, 어두운 디테일 보존
  - background subtraction: 산만한 배경을 빼서 국소 변동 노출

VN 문제: env_max 22(인트로) vs 1-3(3분대 밀집 구간)
  → 전역 Otsu 2.82가 밀집 구간을 전부 묻음
  = 성운 사진에서 밝은 별이 전역 정규화를 지배해
    희미한 필라멘트가 임계 아래로 밀린 것과 동일

해법: SuperFlux 포락선에 asinh 압축 적용 후 Otsu
  → 다이내믹 레인지 축소 → 밀집 구간의 사건이 임계에 근접
  → Q1/SIR rescue가 더 많은 base를 확보 → 프로토타입 안정
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from scipy.ndimage import gaussian_filter1d
from config import audio_paths, MIN_EVENT_GAP_S, HOP, SR, WINDOW_S
from audio_io import load_mono, duration_s
from onset import superflux_envelope, band_envelopes
from peak_pick import peaks, otsu


frame_dt = HOP / SR


def asinh_compress(env, beta=None):
    """asinh 다이내믹 레인지 압축.

    beta=None이면 데이터에서 도출 (자유 파라미터 0):
      beta = median(positive values) / std(positive values)
    천문학적 관례: beta가 클수록 강한 압축.
    """
    pos = env[env > 0]
    if len(pos) < 10:
        return env.copy()
    if beta is None:
        beta = float(np.median(pos) / max(np.std(pos), 1e-12))
        beta = max(beta, 0.1)

    env_max = env.max()
    if env_max <= 0:
        return env.copy()
    normed = np.clip(env / env_max, 0, None)
    compressed = np.arcsinh(normed * beta) / np.arcsinh(beta)
    return compressed * env_max


def bg_subtract(env, sigma_frames=None):
    """가우시안 배경 감산.

    sigma_frames=None이면 4초 윈도우에 해당하는 프레임 수.
    """
    if sigma_frames is None:
        sigma_frames = int(WINDOW_S / frame_dt)
    bg = gaussian_filter1d(np.clip(env, 0, None), sigma=sigma_frames)
    return np.maximum(env - bg, 0)


# 비교 구간
SEGMENTS = {
    "Grievous":  [(0, 4, 18), (16, 20, 14)],
    "Nightmare": [(176, 180, None), (180, 184, None), (184, 188, None),
                  (188, 192, None), (192, 196, None), (196, 200, None),
                  (200, 204, 6), (268, 272, 14)],
    "cry":       [(0, 4, None)],
    "Swift":     [(0, 4, None)],
}


def track_key(name):
    for k in SEGMENTS:
        if k in name:
            return k
    return None


for p in audio_paths():
    key = track_key(p.name)
    if key is None:
        continue
    segs = SEGMENTS[key]

    print(f"\n{'='*72}")
    print(f"▸ {p.name}")
    mono = load_mono(p)
    dur = duration_s(p)
    env, times = superflux_envelope(mono)
    bands = band_envelopes(mono)

    # 원본 Otsu
    thr_orig = otsu(env[np.isfinite(env)])
    pk_orig = peaks(env, times)

    # A: asinh 압축 후 Otsu
    env_asinh = asinh_compress(env)
    thr_asinh = otsu(env_asinh[np.isfinite(env_asinh)])
    pk_asinh = peaks(env_asinh, times)

    # B: 배경 감산 후 Otsu
    env_bgsub = bg_subtract(env)
    thr_bgsub = otsu(env_bgsub[env_bgsub > 0]) if (env_bgsub > 0).sum() > 10 else 0
    pk_bgsub = peaks(env_bgsub, times)

    # C: asinh + 배경 감산
    env_both = bg_subtract(asinh_compress(env))
    thr_both = otsu(env_both[env_both > 0]) if (env_both > 0).sum() > 10 else 0
    pk_both = peaks(env_both, times)

    # beta 확인
    pos = env[env > 0]
    beta = float(np.median(pos) / max(np.std(pos), 1e-12))
    print(f"  asinh beta(자동): {beta:.3f}")
    print(f"  Otsu: 원본={thr_orig:.3f}  asinh={thr_asinh:.3f}  "
          f"bgsub={thr_bgsub:.3f}  both={thr_both:.3f}")
    print(f"  전체 피크: 원본={len(pk_orig)}  asinh={len(pk_asinh)}  "
          f"bgsub={len(pk_bgsub)}  both={len(pk_both)}")

    header = f"  {'구간':>11s}  {'원본':>5s}  {'asinh':>5s}  {'bgsub':>5s}  {'both':>5s}"
    if any(h is not None for _, _, h in segs):
        header += f"  {'인간':>4s}"
    print(header)

    for t0, t1, human in segs:
        if t1 > dur:
            continue
        c_orig = int(((pk_orig >= t0) & (pk_orig < t1)).sum())
        c_asinh = int(((pk_asinh >= t0) & (pk_asinh < t1)).sum())
        c_bgsub = int(((pk_bgsub >= t0) & (pk_bgsub < t1)).sum())
        c_both = int(((pk_both >= t0) & (pk_both < t1)).sum())

        label = f"{t0//60}:{t0%60:02d}-{t1//60}:{t1%60:02d}"
        line = f"  {label:>11s}  {c_orig:5d}  {c_asinh:5d}  {c_bgsub:5d}  {c_both:5d}"
        if human is not None:
            line += f"  {human:4d}"
        elif any(h is not None for _, _, h in segs):
            line += f"  {'—':>4s}"
        print(line)

    # VN 밀집 구간 상세: 극대점 중 Otsu 초과 비율
    if key == "Nightmare":
        print(f"\n  --- VN 밀집 구간 (2:56-4:00) Otsu 초과 극대점 ---")
        for label, e, thr in [("원본", env, thr_orig),
                               ("asinh", env_asinh, thr_asinh),
                               ("bgsub", env_bgsub, thr_bgsub),
                               ("both", env_both, thr_both)]:
            mask = (times >= 176) & (times < 240)
            seg = e[mask]
            n_lm = 0
            n_above = 0
            for i in range(1, len(seg) - 1):
                if seg[i] >= seg[i-1] and seg[i] > seg[i+1]:
                    n_lm += 1
                    if seg[i] > thr:
                        n_above += 1
            print(f"    {label:6s}: 극대점={n_lm:5d}  Otsu초과={n_above:4d}  "
                  f"비율={n_above/max(n_lm,1)*100:5.1f}%")
