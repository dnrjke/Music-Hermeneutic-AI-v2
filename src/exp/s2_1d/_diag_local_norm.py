"""구간별 자기 정규화 후 Otsu — 성운 STEP 2 "채널별 99-pct 정규화" 응용.

성운 파이프라인에서 Ha 절대 밝기가 OIII를 임계 아래로 미는 문제를
"각 채널 자기 99-pct 정규화 후 union"으로 해결. 이를 시간 축에 적용:

1. 포락선을 N-초 블록으로 분할
2. 각 블록의 양수값을 자기 99-pct로 정규화 (0~1 스케일)
3. 정규화된 포락선에 전역 Otsu
4. 결과: 조용한 구간의 사건이 시끄러운 구간과 같은 스케일에서 경쟁

블록 크기는 자유 파라미터가 아님:
  - 성운에서 "채널" = 독립 관측 축 → 음향에서 "블록" = 독립 맥락 단위
  - WINDOW_S(4초)의 배수, 곡의 구조적 단위에 해당
  - 16초 = 4윈도우 = ~170BPM에서 ~16마디 ≈ 1프레이즈
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from config import audio_paths, MIN_EVENT_GAP_S, HOP, SR, WINDOW_S
from audio_io import load_mono, duration_s
from onset import superflux_envelope, band_envelopes
from peak_pick import peaks, otsu


frame_dt = HOP / SR


def local_norm(env, times, block_s=16.0, pct=99.0):
    """블록별 자기 정규화.

    각 블록의 양수값 pct-percentile로 나눔.
    전역 스케일은 사라지고 국소 스케일만 남음.
    """
    out = np.zeros_like(env)
    dur = times[-1] + frame_dt
    n_blocks = int(np.ceil(dur / block_s))

    for bi in range(n_blocks):
        t0, t1 = bi * block_s, (bi + 1) * block_s
        mask = (times >= t0) & (times < t1)
        seg = env[mask]
        pos = seg[seg > 0]
        if len(pos) < 5:
            out[mask] = 0
            continue
        p = np.percentile(pos, pct)
        if p < 1e-12:
            out[mask] = 0
            continue
        out[mask] = np.clip(seg / p, 0, None)

    return out


SEGMENTS = {
    "Grievous":  [(0, 4, 18), (16, 20, 14)],
    "Nightmare": [(0, 4, None), (176, 180, None), (180, 184, None),
                  (184, 188, None), (188, 192, None), (192, 196, None),
                  (196, 200, None), (200, 204, 6), (264, 268, None),
                  (268, 272, 14)],
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

    pk_orig = peaks(env, times)
    thr_orig = otsu(env[np.isfinite(env)])

    # 다양한 블록 크기
    for block_s in [8, 16, 32]:
        env_normed = local_norm(env, times, block_s=block_s)
        fin = env_normed[env_normed > 0]
        if len(fin) < 10:
            continue
        thr_n = otsu(fin)
        pk_n = peaks(env_normed, times)

        label_h = f"norm({block_s}s)"
        if block_s == 16:
            # 상세 출력
            print(f"  Otsu: 원본={thr_orig:.3f}  {label_h}={thr_n:.3f}")
            print(f"  전체 피크: 원본={len(pk_orig)}  {label_h}={len(pk_n)}")

            header = f"  {'구간':>11s}  {'원본':>5s}  {label_h:>8s}"
            if any(h is not None for _, _, h in segs):
                header += f"  {'인간':>4s}"
            print(header)

            for t0, t1, human in segs:
                if t1 > dur:
                    continue
                c_orig = int(((pk_orig >= t0) & (pk_orig < t1)).sum())
                c_n = int(((pk_n >= t0) & (pk_n < t1)).sum())
                lab = f"{t0//60}:{t0%60:02d}-{t1//60}:{t1%60:02d}"
                line = f"  {lab:>11s}  {c_orig:5d}  {c_n:8d}"
                if human is not None:
                    line += f"  {human:4d}"
                elif any(h is not None for _, _, h in segs):
                    line += f"  {'—':>4s}"
                print(line)

    # VN 밀집 구간 상세
    if key == "Nightmare":
        print(f"\n  --- VN 밀집 구간 Otsu 초과 극대점 ---")
        for block_s in [8, 16, 32]:
            env_n = local_norm(env, times, block_s=block_s)
            fin = env_n[env_n > 0]
            thr_n = otsu(fin) if len(fin) > 10 else 0
            mask = (times >= 176) & (times < 240)
            seg = env_n[mask]
            n_lm = 0
            n_above = 0
            for i in range(1, len(seg) - 1):
                if seg[i] >= seg[i-1] and seg[i] > seg[i+1]:
                    n_lm += 1
                    if seg[i] > thr_n:
                        n_above += 1
            print(f"    norm({block_s:2d}s): 극대점={n_lm:5d}  Otsu초과={n_above:4d}  "
                  f"비율={n_above/max(n_lm,1)*100:5.1f}%  Otsu={thr_n:.3f}")
