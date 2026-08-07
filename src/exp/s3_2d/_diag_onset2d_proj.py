"""2D Onset Map → 1D 투영 진단.

2D per-bin 정규화 + Otsu 임계 후 시간 축으로 투영,
1D peak picking. 2D의 정규화 이점 + 1D의 안정적 계수.

투영 방식 비교:
  A) column_max  — 각 프레임에서 임계 초과 빈이 하나라도 있으면 1
  B) column_frac — 각 프레임에서 임계 초과 빈의 비율 (0~1)
  C) column_sum  — 각 프레임에서 정규화 flux의 합 (임계 초과분만)
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

import numpy as np
import librosa
from scipy import ndimage
from config import (
    SR, N_FFT, HOP, N_MELS, FMIN,
    SUPERFLUX_LAG, SUPERFLUX_MAX, WINDOW_S,
    MIN_EVENT_GAP_S, audio_paths,
)
from audio_io import load_mono, duration_s
from peak_pick import peaks, otsu


def superflux_2d(mono):
    S = np.abs(librosa.stft(mono, n_fft=N_FFT, hop_length=HOP))
    mel = librosa.feature.melspectrogram(S=S ** 2, sr=SR, n_mels=N_MELS, fmin=FMIN)
    logmel = librosa.power_to_db(mel, ref=np.max, top_db=80.0)

    ref = ndimage.maximum_filter1d(logmel, size=SUPERFLUX_MAX, axis=0)
    diff = logmel[:, SUPERFLUX_LAG:] - ref[:, :-SUPERFLUX_LAG]
    flux_2d = np.maximum(diff, 0.0)

    n_frames = S.shape[1]
    pad_width = n_frames - flux_2d.shape[1]
    if pad_width > 0:
        flux_2d = np.pad(flux_2d, ((0, 0), (pad_width, 0)), mode="constant")

    times = librosa.frames_to_time(np.arange(n_frames), sr=SR, hop_length=HOP)
    return flux_2d, times


def norm_perbin(flux_2d, pct=99.0):
    out = np.zeros_like(flux_2d)
    for b in range(flux_2d.shape[0]):
        row = flux_2d[b]
        pos = row[row > 0]
        if len(pos) < 5:
            continue
        p = np.percentile(pos, pct)
        if p < 1e-12:
            continue
        out[b] = np.clip(row / p, 0, None)
    return out


def project_and_pick(flux_normed, times, mode="frac",
                     se_height=3, min_gap_s=MIN_EVENT_GAP_S):
    """2D 정규화 map을 1D로 투영 후 peak pick.

    mode:
      "frac" — 빈별 Otsu 초과 비율 (0~1). 많은 빈이 동시에 반응할수록 높음.
      "sum"  — 임계 초과 flux의 빈별 합.
      "max"  — 프레임별 최대 flux.
    """
    pos = flux_normed[flux_normed > 0]
    if len(pos) < 10:
        return np.array([]), np.zeros(len(times))

    thr = otsu(pos)

    binary = flux_normed > thr
    if se_height > 1:
        se = np.ones((se_height, 1), dtype=bool)
        binary = ndimage.binary_opening(binary, structure=se)

    if mode == "frac":
        proj = binary.mean(axis=0)
    elif mode == "sum":
        masked = np.where(binary, flux_normed, 0.0)
        proj = masked.sum(axis=0)
    elif mode == "max":
        proj = flux_normed.max(axis=0)
    else:
        raise ValueError(f"unknown mode: {mode}")

    pk = peaks(proj, times, min_gap_s=min_gap_s)
    return pk, proj


def local_norm_1d(env, times, block_s=16.0, pct=99.0):
    frame_dt = HOP / SR
    out = np.zeros_like(env)
    dur = times[-1] + frame_dt
    n_blocks = int(np.ceil(dur / block_s))
    for bi in range(n_blocks):
        t0, t1 = bi * block_s, (bi + 1) * block_s
        mask = (times >= t0) & (times < t1)
        seg = env[mask]
        pos_seg = seg[seg > 0]
        if len(pos_seg) < 5:
            continue
        p = np.percentile(pos_seg, pct)
        if p < 1e-12:
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

    print(f"\n{'='*72}")
    print(f"▸ {p.name}")
    mono = load_mono(p)
    dur = duration_s(p)

    # 1D baselines
    from onset import superflux_envelope
    env_1d, times = superflux_envelope(mono)
    pk_1d = peaks(env_1d, times)
    env_1d_norm = local_norm_1d(env_1d, times)
    pk_1d_norm = peaks(env_1d_norm, times)

    # 2D
    flux_2d, times_2d = superflux_2d(mono)
    flux_normed = norm_perbin(flux_2d)

    segs = SEGMENTS[key]

    # 헤더
    print(f"\n  {'방법':>22s}  {'전체':>5s}", end="")
    for t0, t1, h in segs:
        label = f"{t0//60}:{t0%60:02d}"
        print(f"  {label:>6s}", end="")
    print()

    # 1D Otsu
    line = f"  {'1D Otsu':>22s}  {len(pk_1d):5d}"
    for t0, t1, h in segs:
        if t1 > dur:
            line += f"  {'—':>6s}"
            continue
        c = int(((pk_1d >= t0) & (pk_1d < t1)).sum())
        s = f"{c}"
        if h is not None:
            s += f"({h})"
        line += f"  {s:>6s}"
    print(line)

    # 1D norm(16s)
    line = f"  {'1D norm(16s)':>22s}  {len(pk_1d_norm):5d}"
    for t0, t1, h in segs:
        if t1 > dur:
            line += f"  {'—':>6s}"
            continue
        c = int(((pk_1d_norm >= t0) & (pk_1d_norm < t1)).sum())
        line += f"  {c:>6d}"
    print(line)

    # 2D → 1D 투영
    for mode in ["frac", "sum"]:
        for se_h in [1, 3, 5]:
            pk_2d, proj = project_and_pick(flux_normed, times_2d,
                                           mode=mode, se_height=se_h)
            label = f"2D→{mode} se={se_h}"
            line = f"  {label:>22s}  {len(pk_2d):5d}"
            for t0, t1, h in segs:
                if t1 > dur:
                    line += f"  {'—':>6s}"
                    continue
                c = int(((pk_2d >= t0) & (pk_2d < t1)).sum())
                line += f"  {c:>6d}"
            print(line)

    # VN 밀집 구간 윈도우별 상세
    if key == "Nightmare":
        print(f"\n  --- VN 밀집 구간 (176-240s) 4초 윈도우별 ---")
        best_methods = [
            ("1D Otsu", pk_1d),
            ("1D norm(16s)", pk_1d_norm),
        ]
        for mode in ["frac", "sum"]:
            for se_h in [1, 3]:
                pk, _ = project_and_pick(flux_normed, times_2d,
                                         mode=mode, se_height=se_h)
                best_methods.append((f"2D→{mode} se={se_h}", pk))

        for name, pk in best_methods:
            detail = []
            for wi_t0 in range(176, 240, 4):
                wi_t1 = wi_t0 + 4
                c = int(((pk >= wi_t0) & (pk < wi_t1)).sum())
                detail.append(f"{c:2d}")
            total = sum(int(x) for x in detail)
            print(f"    {name:>22s}: [{', '.join(detail)}] ={total:4d}")
