"""s4_piano B-2: CQT 기반 spectral novelty.

가설: 피아노의 반음 격자에 맞는 CQT 표현은 mel 표현보다 배음을
분리하여, 울리는 음에 가려진 새 타건의 스펙트럼 형태 변화를 보존한다.

[D-21] CQT와 후처리 파라미터는 실행 전에 물리적으로 고정한다.
mel/CQT 양쪽에 같은 cosine novelty와 2초 local norm을 적용해
주파수 표현만 비교한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import librosa
import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from scipy.signal import correlate, find_peaks

from audio_io import duration_s, load_mono
from config import (
    FMIN,
    HOP,
    MIN_EVENT_GAP_S,
    N_FFT,
    N_MELS,
    OUT_DIR,
    SR,
    WINDOW_S,
)
from peak_pick import otsu


FRAME_DT = HOP / SR
SMOOTH_S = 2.0
NORM_BLOCK_S = 2.0
NORM_PERCENTILE = 99.0
MATCH_TOL_S = MIN_EVENT_GAP_S

# A0부터 9옥타브. 88개 건반의 기본음과 고차 배음을 함께 포함한다.
CQT_FMIN = float(librosa.note_to_hz("A0"))
CQT_BINS_PER_OCTAVE = 12
CQT_N_BINS = 9 * CQT_BINS_PER_OCTAVE
CQT_TUNING = 0.0


def logmel(mono: np.ndarray) -> np.ndarray:
    """기존 s4 대조군과 동일한 log-mel 표현."""
    stft_mag = np.abs(librosa.stft(mono, n_fft=N_FFT, hop_length=HOP))
    mel = librosa.feature.melspectrogram(
        S=stft_mag**2,
        sr=SR,
        n_mels=N_MELS,
        fmin=FMIN,
    )
    return librosa.power_to_db(mel, ref=np.max, top_db=80.0)


def logcqt(mono: np.ndarray, n_frames: int) -> np.ndarray:
    """선고정한 반음 단위 CQT를 dB 표현으로 변환한다."""
    cqt_mag = np.abs(
        librosa.cqt(
            mono,
            sr=SR,
            hop_length=HOP,
            fmin=CQT_FMIN,
            n_bins=CQT_N_BINS,
            bins_per_octave=CQT_BINS_PER_OCTAVE,
            tuning=CQT_TUNING,
        )
    )
    cqt_db = librosa.amplitude_to_db(cqt_mag, ref=np.max, top_db=80.0)
    return librosa.util.fix_length(cqt_db, size=n_frames, axis=1)


def cosine_novelty(spec_db: np.ndarray) -> np.ndarray:
    """인접 프레임 스펙트럼의 cosine distance."""
    if spec_db.ndim != 2 or spec_db.shape[1] < 2:
        raise ValueError(f"잘못된 스펙트럼 shape: {spec_db.shape}")
    norms = np.linalg.norm(spec_db, axis=0, keepdims=True)
    normed = spec_db / np.maximum(norms, 1e-12)
    similarity = np.sum(normed[:, 1:] * normed[:, :-1], axis=0)
    distance = 1.0 - np.clip(similarity, -1.0, 1.0)
    env = np.zeros(spec_db.shape[1], dtype=np.float64)
    env[1:] = np.maximum(distance, 0.0)
    return env


def greedy_select(
    indices: np.ndarray,
    values: np.ndarray,
    times: np.ndarray,
    min_gap_s: float,
) -> np.ndarray:
    selected: list[int] = []
    for i in sorted(indices.tolist(), key=lambda j: -values[j]):
        if all(abs(times[i] - times[j]) >= min_gap_s for j in selected):
            selected.append(i)
    if not selected:
        return np.array([], dtype=np.float64)
    return times[np.sort(np.asarray(selected, dtype=int))]


def bandpass_norm_peaks(
    env: np.ndarray,
    times: np.ndarray,
) -> tuple[np.ndarray, float]:
    """양 군 공통: 2초 smooth 제거 + 2초 99-pct norm + Otsu."""
    if env.shape != times.shape:
        raise ValueError(f"envelope/time 길이 불일치: {env.shape} != {times.shape}")
    if not np.isfinite(env).all():
        raise ValueError("envelope에 NaN/Inf가 있습니다")

    env_pos = np.maximum(env, 0.0)
    smooth_frames = max(3, int(SMOOTH_S / FRAME_DT) | 1)
    smooth = uniform_filter1d(env_pos, size=smooth_frames, mode="reflect")
    residual = np.maximum(env_pos - smooth, 0.0)

    duration = times[-1] + FRAME_DT
    n_blocks = int(np.ceil(duration / NORM_BLOCK_S))
    normed = np.zeros_like(residual)
    for block_i in range(n_blocks):
        t0 = block_i * NORM_BLOCK_S
        t1 = (block_i + 1) * NORM_BLOCK_S
        mask = (times >= t0) & (times < t1)
        segment = residual[mask]
        positive = segment[segment > 0]
        if len(positive) < 3:
            continue
        scale = np.percentile(positive, NORM_PERCENTILE)
        if scale >= 1e-12:
            normed[mask] = np.clip(segment / scale, 0.0, None)

    positive = normed[normed > 0]
    if len(positive) < 10:
        raise RuntimeError("Otsu를 계산할 양수 local-norm 표본이 부족합니다")
    threshold = otsu(positive)
    maxima = np.flatnonzero(
        (normed[1:-1] >= normed[:-2])
        & (normed[1:-1] > normed[2:])
        & (normed[1:-1] > threshold)
    ) + 1
    peaks = greedy_select(maxima, normed, times, MIN_EVENT_GAP_S)
    if len(peaks) == 0:
        raise RuntimeError("검출된 피크가 없습니다")
    return peaks, threshold


def one_to_one_match(
    mel_peaks: np.ndarray,
    cqt_peaks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """±30ms 후보를 거리순으로 배정해 중복 없는 공통 피크를 만든다."""
    candidates: list[tuple[float, int, int]] = []
    for mel_i, mel_t in enumerate(mel_peaks):
        lo = int(np.searchsorted(cqt_peaks, mel_t - MATCH_TOL_S, side="left"))
        hi = int(np.searchsorted(cqt_peaks, mel_t + MATCH_TOL_S, side="right"))
        candidates.extend(
            (abs(mel_t - cqt_peaks[cqt_i]), mel_i, cqt_i)
            for cqt_i in range(lo, hi)
        )

    used_mel: set[int] = set()
    used_cqt: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, mel_i, cqt_i in sorted(candidates):
        if mel_i not in used_mel and cqt_i not in used_cqt:
            used_mel.add(mel_i)
            used_cqt.add(cqt_i)
            pairs.append((mel_i, cqt_i))

    common = np.array(
        sorted((mel_peaks[i] + cqt_peaks[j]) / 2.0 for i, j in pairs),
        dtype=np.float64,
    )
    mel_only = np.delete(mel_peaks, sorted(used_mel))
    cqt_only = np.delete(cqt_peaks, sorted(used_cqt))
    return common, mel_only, cqt_only


def periodicity_metrics(peaks: np.ndarray, duration: float) -> dict[str, float | int]:
    """기존 artifact 진단과 같은 9초 이후 IOI/자기상관 요약."""
    post9 = peaks[peaks >= 9.0]
    if len(post9) < 3:
        raise RuntimeError("주기성 진단에 필요한 피크가 부족합니다")
    ioi_ms = np.diff(post9) * 1000.0

    resolution_ms = 5.0
    impulse = np.zeros(int(np.ceil(duration * 1000.0 / resolution_ms)))
    indices = (post9 * 1000.0 / resolution_ms).astype(int)
    indices = indices[(indices >= 0) & (indices < len(impulse))]
    impulse[indices] = 1.0

    max_lag = int(2000.0 / resolution_ms)
    ac = correlate(impulse, impulse, mode="full", method="fft")
    ac = ac[len(impulse) - 1:len(impulse) + max_lag]
    ac[0] = 0.0
    ac_peaks, _ = find_peaks(ac, height=np.max(ac) * 0.3)
    dominant_ms = (
        float(ac_peaks[np.argmax(ac[ac_peaks])] * resolution_ms)
        if len(ac_peaks)
        else float(np.median(ioi_ms))
    )
    return {
        "post9_peaks": int(len(post9)),
        "ioi_median_ms": float(np.median(ioi_ms)),
        "ioi_lt50": int(np.sum(ioi_ms < 50.0)),
        "ioi_lt100": int(np.sum(ioi_ms < 100.0)),
        "ioi_120_130": int(np.sum((ioi_ms >= 120.0) & (ioi_ms < 130.0))),
        "ioi_120_130_pct": float(
            100.0 * np.mean((ioi_ms >= 120.0) & (ioi_ms < 130.0))
        ),
        "dominant_period_ms": dominant_ms,
    }


def count_windows(peaks: np.ndarray, duration: float) -> list[int]:
    return [
        int(np.sum((peaks >= t0) & (peaks < t0 + WINDOW_S)))
        for t0 in np.arange(0.0, duration, WINDOW_S)
    ]


def click(freq: float, duration_ms: float = 12.0, amp: float = 0.7) -> np.ndarray:
    n = int(SR * duration_ms / 1000.0)
    t = np.arange(n, dtype=np.float32) / SR
    decay = np.exp(-t * 1000.0 / duration_ms)
    return (amp * decay * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def overlay_groups(
    mono: np.ndarray,
    groups: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    out = mono.copy()
    for peak_times, click_sound in groups:
        for peak_t in peak_times:
            start = int(peak_t * SR)
            stop = min(start + len(click_sound), len(out))
            if stop > start:
                out[start:stop] += click_sound[:stop - start]
    return np.clip(out, -1.0, 1.0)


def print_summary(
    label: str,
    peaks: np.ndarray,
    threshold: float,
    metrics: dict[str, float | int],
) -> None:
    print(
        f"  {label:4s}: {len(peaks):4d} peaks  Otsu={threshold:.6f}  "
        f"IOI={metrics['ioi_median_ms']:.1f}ms  "
        f"<50={metrics['ioi_lt50']}  <100={metrics['ioi_lt100']}  "
        f"120-130={metrics['ioi_120_130_pct']:.1f}%  "
        f"AC={metrics['dominant_period_ms']:.0f}ms"
    )


def main() -> None:
    audio_path = Path(r"E:\game\Music Hermeneutic AI v2\audio\102 - Dir.wav")
    print(f"▸ B-2 CQT novelty: {audio_path.name}")
    print(
        f"  CQT: fmin={CQT_FMIN:.1f}Hz, bins={CQT_N_BINS}, "
        f"bpo={CQT_BINS_PER_OCTAVE}, tuning={CQT_TUNING:.1f}"
    )
    mono = load_mono(audio_path)
    duration = duration_s(audio_path)

    mel_db = logmel(mono)
    cqt_db = logcqt(mono, mel_db.shape[1])
    times = librosa.frames_to_time(
        np.arange(mel_db.shape[1]),
        sr=SR,
        hop_length=HOP,
    )
    if cqt_db.shape[1] != len(times):
        raise RuntimeError("CQT 프레임 정렬에 실패했습니다")

    mel_env = cosine_novelty(mel_db)
    cqt_env = cosine_novelty(cqt_db)
    mel_peaks, mel_threshold = bandpass_norm_peaks(mel_env, times)
    cqt_peaks, cqt_threshold = bandpass_norm_peaks(cqt_env, times)
    common, mel_only, cqt_only = one_to_one_match(mel_peaks, cqt_peaks)

    mel_metrics = periodicity_metrics(mel_peaks, duration)
    cqt_metrics = periodicity_metrics(cqt_peaks, duration)
    print("\n  요약")
    print_summary("mel", mel_peaks, mel_threshold, mel_metrics)
    print_summary("CQT", cqt_peaks, cqt_threshold, cqt_metrics)
    print(
        f"  ±{MATCH_TOL_S * 1000:.0f}ms 일대일 매칭: "
        f"공통={len(common)}, CQT전용={len(cqt_only)}, mel전용={len(mel_only)}"
    )

    mel_windows = count_windows(mel_peaks, duration)
    cqt_windows = count_windows(cqt_peaks, duration)
    print(f"\n  {'시각':>5s}  {'mel':>4s}  {'CQT':>4s}  {'차':>4s}")
    for i, (mel_count, cqt_count) in enumerate(zip(mel_windows, cqt_windows)):
        t0 = i * WINDOW_S
        timestamp = f"{int(t0) // 60}:{int(t0) % 60:02d}"
        print(f"  {timestamp:>5s}  {mel_count:4d}  {cqt_count:4d}  {cqt_count-mel_count:+4d}")

    destination = OUT_DIR / "sonify" / "Dir"
    destination.mkdir(parents=True, exist_ok=True)
    cqt_path = destination / "전체_cqt_n2s_클릭.wav"
    compare_path = destination / "전체_cqt_n2s_vs_mel_n2s_비교_클릭.wav"
    metrics_path = destination / "cqt_b2_metrics.json"

    sf.write(
        str(cqt_path),
        overlay_groups(mono, [(cqt_peaks, click(3000.0))]),
        SR,
    )
    sf.write(
        str(compare_path),
        overlay_groups(
            mono,
            [
                (common, click(3000.0)),
                (cqt_only, click(5000.0, 15.0, 0.8)),
                (mel_only, click(1500.0, 15.0, 0.8)),
            ],
        ),
        SR,
    )

    report = {
        "experiment": "s4 B-2 CQT novelty",
        "audio": audio_path.name,
        "duration_s": duration,
        "fixed_parameters": {
            "sr": SR,
            "hop": HOP,
            "cqt_fmin_hz": CQT_FMIN,
            "cqt_bins_per_octave": CQT_BINS_PER_OCTAVE,
            "cqt_n_bins": CQT_N_BINS,
            "cqt_tuning": CQT_TUNING,
            "smooth_s": SMOOTH_S,
            "norm_block_s": NORM_BLOCK_S,
            "norm_percentile": NORM_PERCENTILE,
            "min_event_gap_s": MIN_EVENT_GAP_S,
            "match_tolerance_s": MATCH_TOL_S,
        },
        "mel": {
            "peaks": int(len(mel_peaks)),
            "otsu": mel_threshold,
            "periodicity": mel_metrics,
            "window_counts": mel_windows,
        },
        "cqt": {
            "peaks": int(len(cqt_peaks)),
            "otsu": cqt_threshold,
            "periodicity": cqt_metrics,
            "window_counts": cqt_windows,
        },
        "matching": {
            "common": int(len(common)),
            "cqt_only": int(len(cqt_only)),
            "mel_only": int(len(mel_only)),
        },
        "listening_legend_hz": {
            "common": 3000,
            "cqt_only": 5000,
            "mel_only": 1500,
        },
    }
    metrics_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\n  CQT 전체(3kHz): {cqt_path}")
    print(
        "  비교 범례: 공통=3kHz, CQT전용=5kHz, mel전용=1.5kHz\n"
        f"  비교 파일: {compare_path}\n"
        f"  수치 기록: {metrics_path}"
    )


if __name__ == "__main__":
    main()
