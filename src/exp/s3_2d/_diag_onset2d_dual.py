"""2D 양축 정규화 진단 — per-bin(주파수) + per-block(시간) 정규화.

성운 파이프라인 완전 재현:
  STEP 2a: 각 채널(=mel bin)을 자기 99-pct로 정규화 → 주파수 축 균등화
  STEP 2b: 각 시간 블록을 자기 99-pct로 정규화 → 시간 축 균등화

조합:
  A) bin→block: per-bin 먼저, 그 위에 per-block
  B) block→bin: per-block 먼저, 그 위에 per-bin
  C) 2D→1D→norm: per-bin → 1D 투영 → 1D local norm
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
    """주파수 축 정규화: 각 mel bin을 자기 99-pct로."""
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


def norm_perblock_2d(flux_2d, times, block_s=16.0, pct=99.0):
    """시간 축 정규화 (2D): 각 블록의 전 빈을 자기 99-pct로."""
    frame_dt = HOP / SR
    out = np.zeros_like(flux_2d)
    dur = times[-1] + frame_dt
    n_blocks = int(np.ceil(dur / block_s))
    for bi in range(n_blocks):
        t0, t1 = bi * block_s, (bi + 1) * block_s
        mask = (times >= t0) & (times < t1)
        seg = flux_2d[:, mask]
        pos = seg[seg > 0]
        if len(pos) < 5:
            continue
        p = np.percentile(pos, pct)
        if p < 1e-12:
            continue
        out[:, mask] = np.clip(seg / p, 0, None)
    return out


def norm_1d_local(env, times, block_s=16.0, pct=99.0):
    """1D 시간 축 정규화."""
    frame_dt = HOP / SR
    out = np.zeros_like(env)
    dur = times[-1] + frame_dt
    n_blocks = int(np.ceil(dur / block_s))
    for bi in range(n_blocks):
        t0, t1 = bi * block_s, (bi + 1) * block_s
        mask = (times >= t0) & (times < t1)
        seg = env[mask]
        pos = seg[seg > 0]
        if len(pos) < 5:
            continue
        p = np.percentile(pos, pct)
        if p < 1e-12:
            continue
        out[mask] = np.clip(seg / p, 0, None)
    return out


def project_frac(flux_normed, times, se_height=1, min_gap_s=MIN_EVENT_GAP_S):
    """2D → 1D frac 투영 → peak pick."""
    pos = flux_normed[flux_normed > 0]
    if len(pos) < 10:
        return np.array([]), np.zeros(len(times))

    thr = otsu(pos)
    binary = flux_normed > thr

    if se_height > 1:
        se = np.ones((se_height, 1), dtype=bool)
        binary = ndimage.binary_opening(binary, structure=se)

    proj = binary.mean(axis=0).astype(np.float64)
    pk = peaks(proj, times, min_gap_s=min_gap_s)
    return pk, proj


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


def count_seg(pk, t0, t1):
    return int(((pk >= t0) & (pk < t1)).sum())


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
    env_1d_norm = norm_1d_local(env_1d, times)
    pk_1d_norm = peaks(env_1d_norm, times)

    # 2D
    flux_2d, times_2d = superflux_2d(mono)

    # 정규화 조합
    methods = {}

    # A: bin→block (주파수 먼저 → 시간)
    ab = norm_perbin(flux_2d)
    ab = norm_perblock_2d(ab, times_2d)
    pk_ab, _ = project_frac(ab, times_2d, se_height=1)
    methods["2D bin→blk→frac"] = pk_ab

    pk_ab3, _ = project_frac(ab, times_2d, se_height=3)
    methods["2D bin→blk→frac se3"] = pk_ab3

    # B: block→bin (시간 먼저 → 주파수)
    ba = norm_perblock_2d(flux_2d, times_2d)
    ba = norm_perbin(ba)
    pk_ba, _ = project_frac(ba, times_2d, se_height=1)
    methods["2D blk→bin→frac"] = pk_ba

    # C: bin→1D→norm (per-bin → 합산 → 1D local norm)
    bn = norm_perbin(flux_2d)
    proj_sum = bn.sum(axis=0)
    proj_sum_norm = norm_1d_local(proj_sum, times_2d)
    pk_c = peaks(proj_sum_norm, times_2d)
    methods["2D bin→sum→norm"] = pk_c

    # D: bin→frac→norm (per-bin → frac → 1D local norm)
    pos_bn = bn[bn > 0]
    thr_bn = otsu(pos_bn) if len(pos_bn) > 10 else 0
    frac_proj = (bn > thr_bn).mean(axis=0).astype(np.float64)
    frac_norm = norm_1d_local(frac_proj, times_2d)
    pk_d = peaks(frac_norm, times_2d)
    methods["2D bin→frac→norm"] = pk_d

    # E: raw 2D block norm만 (per-bin 없이)
    bk_only = norm_perblock_2d(flux_2d, times_2d)
    proj_bk = bk_only.sum(axis=0)
    pk_e = peaks(proj_bk, times_2d)
    methods["2D blk→sum"] = pk_e

    segs = SEGMENTS[key]

    # 헤더
    print(f"\n  {'방법':>24s}  {'전체':>5s}", end="")
    for t0, t1, h in segs:
        label = f"{t0//60}:{t0%60:02d}"
        print(f"  {label:>6s}", end="")
    print()

    # 1D Otsu
    line = f"  {'1D Otsu':>24s}  {len(pk_1d):5d}"
    for t0, t1, h in segs:
        if t1 > dur:
            line += f"  {'—':>6s}"
            continue
        c = count_seg(pk_1d, t0, t1)
        s = f"{c}"
        if h is not None:
            s += f"({h})"
        line += f"  {s:>6s}"
    print(line)

    # 1D norm
    line = f"  {'1D norm(16s)':>24s}  {len(pk_1d_norm):5d}"
    for t0, t1, h in segs:
        if t1 > dur:
            line += f"  {'—':>6s}"
            continue
        line += f"  {count_seg(pk_1d_norm, t0, t1):>6d}"
    print(line)

    # 2D methods
    for name, pk in methods.items():
        line = f"  {name:>24s}  {len(pk):5d}"
        for t0, t1, h in segs:
            if t1 > dur:
                line += f"  {'—':>6s}"
                continue
            line += f"  {count_seg(pk, t0, t1):>6d}"
        print(line)

    # VN 상세
    if key == "Nightmare":
        print(f"\n  --- VN 밀집 구간 (176-240s) 4초 윈도우별 ---")
        all_methods = [
            ("1D Otsu", pk_1d),
            ("1D norm(16s)", pk_1d_norm),
        ]
        all_methods.extend(methods.items())

        for name, pk in all_methods:
            detail = []
            for wi_t0 in range(176, 240, 4):
                wi_t1 = wi_t0 + 4
                c = count_seg(pk, wi_t0, wi_t1)
                detail.append(f"{c:2d}")
            total = sum(int(x) for x in detail)
            print(f"    {name:>24s}: [{', '.join(detail)}] ={total:4d}")
