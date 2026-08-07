"""2D Onset Map 진단 — mel spectrogram에 천체 영상처리 적용.

SuperFlux를 1D로 합산하기 전의 per-bin spectral flux를 2D 행렬로 유지,
성운 파이프라인의 기법을 2D에서 적용한다.

파이프라인:
  mel spectrogram (N_MELS × T)
    → per-bin positive spectral flux (SuperFlux 방식, 합산 안 함)
    → per-bin 99-pct 정규화 (= BASELINE STEP 2 "채널별 자기 99-pct")
    → 2D Otsu threshold
    → morphological opening (수직 SE)
    → connected component labeling
    → 성분 시간 중심 → 사건 시각
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


def superflux_2d(mono: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """per-bin SuperFlux 2D onset map.

    librosa onset_strength 내부 로직을 풀어서 합산 전 단계를 반환.

    Returns:
        (flux_2d, times) — flux_2d shape (N_MELS, T), times shape (T,)
    """
    S = np.abs(librosa.stft(mono, n_fft=N_FFT, hop_length=HOP))
    mel = librosa.feature.melspectrogram(S=S ** 2, sr=SR, n_mels=N_MELS, fmin=FMIN)
    logmel = librosa.power_to_db(mel, ref=np.max, top_db=80.0)

    lag = SUPERFLUX_LAG
    mx = SUPERFLUX_MAX

    # max-filtered reference (SuperFlux: max along frequency axis over window)
    ref = ndimage.maximum_filter1d(logmel, size=mx, axis=0)

    # positive spectral flux per bin
    diff = logmel[:, lag:] - ref[:, :-lag]
    flux_2d = np.maximum(diff, 0.0)

    n_frames = S.shape[1]
    # pad to match original frame count
    pad_width = n_frames - flux_2d.shape[1]
    if pad_width > 0:
        flux_2d = np.pad(flux_2d, ((0, 0), (pad_width, 0)), mode="constant")

    times = librosa.frames_to_time(np.arange(n_frames), sr=SR, hop_length=HOP)
    return flux_2d, times


def norm_2d_perbin(flux_2d: np.ndarray, pct: float = 99.0) -> np.ndarray:
    """per-bin (=per-channel) 99-pct 정규화.

    성운 BASELINE STEP 2: "각 채널 자기 99-pct 정규화".
    각 mel bin의 양수값을 자기 pct-percentile로 나눔.
    """
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


def detect_2d(flux_normed: np.ndarray, times: np.ndarray,
              se_height: int = 3, min_gap_s: float = MIN_EVENT_GAP_S,
              connectivity: int = 2
              ) -> tuple[np.ndarray, int, np.ndarray]:
    """2D onset map에서 connected component로 사건 탐지.

    1. 2D Otsu threshold
    2. morphological opening (수직 SE, height bins)
    3. connected component labeling
    4. 성분 시간 중심 → 사건 시각

    Returns:
        (event_times, n_components, binary_mask)
    """
    pos = flux_normed[flux_normed > 0]
    if len(pos) < 10:
        return np.array([]), 0, np.zeros_like(flux_normed, dtype=bool)

    thr = otsu(pos)
    binary = flux_normed > thr

    # morphological opening with vertical SE (removes narrow horizontal noise)
    if se_height > 1:
        se = np.ones((se_height, 1), dtype=bool)
        binary = ndimage.binary_opening(binary, structure=se)

    # connected component labeling
    struct = ndimage.generate_binary_structure(2, connectivity)
    labels, n_comp = ndimage.label(binary, structure=struct)

    if n_comp == 0:
        return np.array([]), 0, binary

    # 각 성분의 시간 중심
    centers = ndimage.center_of_mass(binary.astype(float), labels, range(1, n_comp + 1))
    t_centers = []
    for cy, cx in centers:
        ti = int(round(cx))
        ti = min(ti, len(times) - 1)
        t_centers.append(times[ti])

    # 진폭(성분 크기) 내림차순 탐욕 선택 + 최소간격
    comp_sizes = ndimage.sum(binary.astype(float), labels, range(1, n_comp + 1))
    order = np.argsort(-np.array(comp_sizes))

    selected: list[float] = []
    for idx in order:
        t = t_centers[idx]
        if all(abs(t - s) >= min_gap_s for s in selected):
            selected.append(t)

    return np.array(sorted(selected)), n_comp, binary


# ── 비교용 1D 기법들

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


# ── 메인

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

    # 1D baseline
    from onset import superflux_envelope
    env_1d, times = superflux_envelope(mono)
    pk_1d = peaks(env_1d, times)

    env_1d_norm = local_norm_1d(env_1d, times)
    pk_1d_norm = peaks(env_1d_norm, times)

    # 2D
    flux_2d, times_2d = superflux_2d(mono)
    flux_normed = norm_2d_perbin(flux_2d)

    # se_height 비교
    print(f"\n  {'방법':>18s}  {'전체':>5s}", end="")
    segs = SEGMENTS[key]
    for t0, t1, h in segs:
        label = f"{t0//60}:{t0%60:02d}"
        print(f"  {label:>6s}", end="")
    print()

    # 1D Otsu
    line = f"  {'1D Otsu':>18s}  {len(pk_1d):5d}"
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
    line = f"  {'1D norm(16s)':>18s}  {len(pk_1d_norm):5d}"
    for t0, t1, h in segs:
        if t1 > dur:
            line += f"  {'—':>6s}"
            continue
        c = int(((pk_1d_norm >= t0) & (pk_1d_norm < t1)).sum())
        line += f"  {c:>6d}"
    print(line)

    # 2D variants
    for se_h in [1, 3, 5]:
        pk_2d, n_comp, mask_2d = detect_2d(flux_normed, times_2d, se_height=se_h)
        label = f"2D se={se_h}"
        line = f"  {label:>18s}  {len(pk_2d):5d}"
        for t0, t1, h in segs:
            if t1 > dur:
                line += f"  {'—':>6s}"
                continue
            c = int(((pk_2d >= t0) & (pk_2d < t1)).sum())
            line += f"  {c:>6d}"
        print(line)

    # 2D 통계
    pos = flux_normed[flux_normed > 0]
    thr_2d = otsu(pos) if len(pos) > 10 else 0
    print(f"\n  2D 통계: shape={flux_2d.shape}, Otsu(2D)={thr_2d:.4f}")
    print(f"  비영 비율: {(flux_2d > 0).mean()*100:.1f}%")
    print(f"  정규화 후 비영: {(flux_normed > 0).mean()*100:.1f}%")

    # VN 밀집 구간 상세
    if key == "Nightmare":
        print(f"\n  --- VN 밀집 구간 (176-240s) 2D connected components ---")
        for se_h in [1, 3, 5]:
            mask_t = (times_2d >= 176) & (times_2d < 240)
            seg_normed = flux_normed[:, mask_t]
            seg_times = times_2d[mask_t]
            pk_seg, nc, _ = detect_2d(seg_normed, seg_times, se_height=se_h)
            # 4초 윈도우별
            w_counts = []
            for wi_t0 in range(176, 240, 4):
                wi_t1 = wi_t0 + 4
                c = int(((pk_seg >= (wi_t0-176)*HOP/SR) & (pk_seg < (wi_t1-176)*HOP/SR)).sum())
                # 시간 보정: seg_times는 0부터 시작이 아니라 176s부터
                c2 = 0
                for t in pk_seg:
                    # seg_times는 원본 times의 부분집합이므로 절대 시간
                    pass
                w_counts.append(c)

            # 다시 전체에서 구간 추출
            pk_full, _, _ = detect_2d(flux_normed, times_2d, se_height=se_h)
            detail = []
            for wi_t0 in range(176, 240, 4):
                wi_t1 = wi_t0 + 4
                c = int(((pk_full >= wi_t0) & (pk_full < wi_t1)).sum())
                detail.append(f"{c:2d}")
            print(f"    se={se_h}: 4s윈도우별 [{', '.join(detail)}]  "
                  f"합계={sum(int(x) for x in detail)}")
