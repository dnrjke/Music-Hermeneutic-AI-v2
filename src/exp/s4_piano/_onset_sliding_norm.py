"""s4_piano A-2: A-3 fusion의 block p99를 sliding p99로 교체.

채택된 fusion_n2s와 표현, 2초 smooth, 99-percentile, 기하평균,
Otsu, 30ms 최소간격을 모두 공유한다. 유일한 변수는 정규화 scale이
고정 2초 타일인지 centered 2초 sliding window인지다.
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
from scipy.ndimage import percentile_filter, uniform_filter1d

from audio_io import duration_s, load_mono
from config import HOP, MIN_EVENT_GAP_S, OUT_DIR, SR, WINDOW_S
from _onset_wtmm_fusion import (
    CONTROL_SCALE_S,
    FRAME_DT,
    NORM_PERCENTILE,
    click,
    cosine_novelty,
    get_logmel,
    no_max_flux,
    one_to_one_time_match,
    overlay_groups,
    periodicity_metrics,
    response_peaks,
    scale_normalize,
    window_counts,
)


EXPECTED_BLOCK_PEAKS = 351
SLIDING_WINDOW_S = CONTROL_SCALE_S
BOUNDARY_TOL_S = MIN_EVENT_GAP_S
EDGE_S = SLIDING_WINDOW_S / 2.0


def highpass_residual(env: np.ndarray) -> np.ndarray:
    positive = np.maximum(env, 0.0)
    window_frames = max(3, int(SLIDING_WINDOW_S / FRAME_DT) | 1)
    smooth = uniform_filter1d(positive, size=window_frames, mode="reflect")
    return np.maximum(positive - smooth, 0.0)


def sliding_normalize(env: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centered 2초 p99로 residual을 연속 정규화한다."""
    if env.shape != times.shape:
        raise ValueError(f"envelope/time 길이 불일치: {env.shape} != {times.shape}")
    if not np.isfinite(env).all():
        raise ValueError("envelope에 NaN/Inf가 있습니다")

    residual = highpass_residual(env)
    window_frames = max(3, int(SLIDING_WINDOW_S / FRAME_DT) | 1)
    local_scale = percentile_filter(
        residual,
        percentile=NORM_PERCENTILE,
        size=window_frames,
        mode="reflect",
    )
    normalized = np.zeros_like(residual)
    valid = local_scale >= 1e-12
    normalized[valid] = residual[valid] / local_scale[valid]
    if not np.isfinite(normalized).all():
        raise RuntimeError("sliding 정규화 결과에 NaN/Inf가 있습니다")
    return normalized, local_scale


def boundary_jump_metrics(response: np.ndarray, times: np.ndarray) -> dict[str, object]:
    boundaries = np.arange(
        SLIDING_WINDOW_S,
        times[-1],
        SLIDING_WINDOW_S,
    )
    indices = np.searchsorted(times, boundaries, side="left")
    indices = indices[(indices > 0) & (indices < len(response))]
    jumps = np.abs(response[indices] - response[indices - 1])
    all_jumps = np.abs(np.diff(response))
    return {
        "boundary_count": int(len(indices)),
        "boundary_jump_median": float(np.median(jumps)),
        "boundary_jump_p95": float(np.percentile(jumps, 95.0)),
        "boundary_jump_max": float(np.max(jumps)),
        "all_frame_jump_p95": float(np.percentile(all_jumps, 95.0)),
    }


def peaks_near_boundaries(peaks: np.ndarray, duration: float) -> int:
    boundaries = np.arange(SLIDING_WINDOW_S, duration, SLIDING_WINDOW_S)
    if len(boundaries) == 0 or len(peaks) == 0:
        return 0
    distances = np.min(np.abs(peaks[:, None] - boundaries[None, :]), axis=1)
    return int(np.sum(distances <= BOUNDARY_TOL_S))


def edge_peak_metrics(peaks: np.ndarray, duration: float) -> dict[str, int]:
    return {
        "first_1s": int(np.sum(peaks < EDGE_S)),
        "last_1s": int(np.sum(peaks >= max(0.0, duration - EDGE_S))),
        "middle": int(
            np.sum((peaks >= EDGE_S) & (peaks < max(EDGE_S, duration - EDGE_S)))
        ),
    }


def main() -> None:
    audio_path = Path(r"E:\game\Music Hermeneutic AI v2\audio\102 - Dir.wav")
    print(f"▸ A-2 sliding percentile: {audio_path.name}")
    window_frames = max(3, int(SLIDING_WINDOW_S / FRAME_DT) | 1)
    print(
        f"  window={SLIDING_WINDOW_S:g}s ({window_frames} frames), "
        f"p={NORM_PERCENTILE:g}, mode=reflect"
    )

    mono = load_mono(audio_path)
    duration = duration_s(audio_path)
    logmel = get_logmel(mono)
    times = librosa.frames_to_time(
        np.arange(logmel.shape[1]),
        sr=SR,
        hop_length=HOP,
    )
    novelty = cosine_novelty(logmel)
    flux = no_max_flux(logmel)

    novelty_block = scale_normalize(novelty, times, CONTROL_SCALE_S)
    flux_block = scale_normalize(flux, times, CONTROL_SCALE_S)
    fusion_block = np.sqrt(novelty_block * flux_block)

    novelty_sliding, novelty_scale = sliding_normalize(novelty, times)
    flux_sliding, flux_scale = sliding_normalize(flux, times)
    fusion_sliding = np.sqrt(novelty_sliding * flux_sliding)

    for label, response in (
        ("fusion_block", fusion_block),
        ("fusion_sliding", fusion_sliding),
    ):
        if response.shape != times.shape or not np.isfinite(response).all():
            raise RuntimeError(f"{label} 응답 검증 실패")

    block_peaks, block_threshold = response_peaks(fusion_block, times)
    if len(block_peaks) != EXPECTED_BLOCK_PEAKS:
        raise RuntimeError(
            f"fusion_n2s 대조군 재현 실패: {len(block_peaks)} != "
            f"{EXPECTED_BLOCK_PEAKS}"
        )
    sliding_peaks, sliding_threshold = response_peaks(fusion_sliding, times)
    common, block_only, sliding_only = one_to_one_time_match(
        block_peaks,
        sliding_peaks,
    )

    block_periodicity = periodicity_metrics(block_peaks, duration)
    sliding_periodicity = periodicity_metrics(sliding_peaks, duration)
    block_boundary = boundary_jump_metrics(fusion_block, times)
    sliding_boundary = boundary_jump_metrics(fusion_sliding, times)

    print("\n  요약")
    for label, peaks, threshold, periodicity in (
        ("block", block_peaks, block_threshold, block_periodicity),
        ("slide", sliding_peaks, sliding_threshold, sliding_periodicity),
    ):
        print(
            f"  {label:5s}: {len(peaks):4d} peaks  Otsu={threshold:.6f}  "
            f"IOI={periodicity['ioi_median_ms']:.1f}ms  "
            f"<50={periodicity['ioi_lt50']}  <100={periodicity['ioi_lt100']}  "
            f"120-130={periodicity['ioi_120_130_pct']:.1f}%  "
            f"AC={periodicity['dominant_period_ms']:.0f}ms"
        )
    print(
        f"  ±{BOUNDARY_TOL_S * 1000:.0f}ms 일대일 매칭: 공통={len(common)}, "
        f"sliding전용={len(sliding_only)}, block전용={len(block_only)}"
    )
    print(
        "  2초 경계 jump p95: "
        f"block={block_boundary['boundary_jump_p95']:.6f}, "
        f"sliding={sliding_boundary['boundary_jump_p95']:.6f}"
    )
    print(
        "  경계 ±30ms 피크: "
        f"block={peaks_near_boundaries(block_peaks, duration)}, "
        f"sliding={peaks_near_boundaries(sliding_peaks, duration)}"
    )

    block_windows = window_counts(block_peaks, duration)
    sliding_windows = window_counts(sliding_peaks, duration)
    print(f"\n  {'시각':>5s}  {'block':>5s}  {'slide':>5s}  {'차':>4s}")
    for window_i, (block_count, sliding_count) in enumerate(
        zip(block_windows, sliding_windows)
    ):
        t0 = window_i * WINDOW_S
        timestamp = f"{int(t0) // 60}:{int(t0) % 60:02d}"
        print(
            f"  {timestamp:>5s}  {block_count:5d}  {sliding_count:5d}  "
            f"{sliding_count - block_count:+4d}"
        )

    destination = OUT_DIR / "sonify" / "Dir"
    destination.mkdir(parents=True, exist_ok=True)
    full_path = destination / "전체_fusion_slide_n2s_클릭.wav"
    compare_path = destination / "전체_fusion_slide_n2s_vs_fusion_n2s_비교_클릭.wav"
    metrics_path = destination / "sliding_norm_metrics.json"

    sf.write(
        str(full_path),
        overlay_groups(mono, [(sliding_peaks, click(3000.0))]),
        SR,
    )
    sf.write(
        str(compare_path),
        overlay_groups(
            mono,
            [
                (common, click(3000.0)),
                (sliding_only, click(5000.0, 15.0, 0.8)),
                (block_only, click(1500.0, 15.0, 0.8)),
            ],
        ),
        SR,
    )

    report = {
        "experiment": "s4 A-2 sliding percentile on A-3 fusion",
        "audio": audio_path.name,
        "duration_s": duration,
        "fixed_parameters": {
            "sr": SR,
            "hop": HOP,
            "smooth_s": CONTROL_SCALE_S,
            "norm_window_s": SLIDING_WINDOW_S,
            "norm_window_frames": window_frames,
            "norm_percentile": NORM_PERCENTILE,
            "sliding_mode": "reflect",
            "min_event_gap_s": MIN_EVENT_GAP_S,
            "matching_tolerance_s": BOUNDARY_TOL_S,
            "fusion": "sqrt(novelty_norm * no_max_flux_norm)",
        },
        "block": {
            "peaks": int(len(block_peaks)),
            "otsu": block_threshold,
            "periodicity": block_periodicity,
            "window_counts": block_windows,
            "boundary": block_boundary,
            "peaks_near_2s_boundaries": peaks_near_boundaries(
                block_peaks,
                duration,
            ),
            "edge_peaks": edge_peak_metrics(block_peaks, duration),
        },
        "sliding": {
            "peaks": int(len(sliding_peaks)),
            "otsu": sliding_threshold,
            "periodicity": sliding_periodicity,
            "window_counts": sliding_windows,
            "boundary": sliding_boundary,
            "peaks_near_2s_boundaries": peaks_near_boundaries(
                sliding_peaks,
                duration,
            ),
            "edge_peaks": edge_peak_metrics(sliding_peaks, duration),
            "novelty_scale_zero_frames": int(np.sum(novelty_scale < 1e-12)),
            "flux_scale_zero_frames": int(np.sum(flux_scale < 1e-12)),
        },
        "matching": {
            "common": int(len(common)),
            "sliding_only": int(len(sliding_only)),
            "block_only": int(len(block_only)),
        },
        "listening_legend_hz": {
            "common": 3000,
            "sliding_only": 5000,
            "block_only": 1500,
        },
    }
    metrics_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\n  비교 범례: 공통=3kHz, sliding전용=5kHz, block전용=1.5kHz")
    print(f"  {full_path.name}")
    print(f"  {compare_path.name}")
    print(f"  {metrics_path.name}")


if __name__ == "__main__":
    main()
