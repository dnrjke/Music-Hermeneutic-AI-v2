"""s4_piano B-3: complex-domain flux + hysteresis attribution.

Bello complex-domain ODF로 steady-state 위상/크기 예측에서 벗어나는 사건을
측정한다. A-2 sliding A-3의 100~300ms meso-burst가 실제 재타건의 위상
리셋을 동반하는지 분해하고, novelty/flux/complex 연속값 융합을 비교한다.
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

from audio_io import duration_s, load_mono
from config import HOP, MIN_EVENT_GAP_S, N_FFT, OUT_DIR, SR, WINDOW_S
from _onset_sliding_norm import sliding_normalize
from _onset_wtmm_fusion import (
    click,
    cosine_novelty,
    get_logmel,
    no_max_flux,
    one_to_one_time_match,
    overlay_groups,
    periodicity_metrics,
    response_peaks,
    window_counts,
)


EXPECTED_CONTROL_PEAKS = 355
MATCH_TOL_S = MIN_EVENT_GAP_S
MESO_BINS_S = ((0.100, 0.130), (0.130, 0.300))


def complex_domain_odf(mono: np.ndarray, n_frames: int) -> np.ndarray:
    """Bello: 이전 크기와 일정 위상속도로 현재 complex STFT를 예측."""
    stft = librosa.stft(mono, n_fft=N_FFT, hop_length=HOP)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    predicted_phase = 2.0 * phase[:, 1:-1] - phase[:, :-2]
    target = magnitude[:, 1:-1] * np.exp(1j * predicted_phase)
    deviation = np.abs(stft[:, 2:] - target)

    envelope = np.zeros(stft.shape[1], dtype=np.float64)
    envelope[2:] = np.sum(deviation, axis=0)
    envelope = librosa.util.fix_length(envelope, size=n_frames)
    if not np.isfinite(envelope).all():
        raise RuntimeError("complex-domain ODF에 NaN/Inf가 있습니다")
    return envelope


def support_mask(
    candidates: np.ndarray,
    evidence: np.ndarray,
    tolerance_s: float = MATCH_TOL_S,
) -> np.ndarray:
    """각 candidate에 ±tolerance complex 피크가 있는지 표시."""
    supported = np.zeros(len(candidates), dtype=bool)
    for candidate_i, candidate_time in enumerate(candidates):
        left = int(np.searchsorted(evidence, candidate_time - tolerance_s, side="left"))
        right = int(np.searchsorted(evidence, candidate_time + tolerance_s, side="right"))
        supported[candidate_i] = right > left
    return supported


def meso_attribution(
    peaks: np.ndarray,
    phase_supported: np.ndarray,
) -> dict[str, dict[str, int]]:
    """100~300ms 연속쌍의 두 번째 피크가 complex 지지를 받는지 집계."""
    intervals = np.diff(peaks)
    report: dict[str, dict[str, int]] = {}
    for lower, upper in MESO_BINS_S:
        pair_mask = (intervals >= lower) & (intervals < upper)
        follower_support = phase_supported[1:][pair_mask]
        label = f"{int(lower * 1000)}-{int(upper * 1000)}ms"
        report[label] = {
            "pairs": int(np.sum(pair_mask)),
            "follower_phase_supported": int(np.sum(follower_support)),
            "follower_phase_unsupported": int(np.sum(~follower_support)),
        }
    return report


def variant_report(
    peaks: np.ndarray,
    threshold: float,
    duration: float,
) -> dict[str, object]:
    return {
        "peaks": int(len(peaks)),
        "otsu": threshold,
        "periodicity": periodicity_metrics(peaks, duration),
        "window_counts": window_counts(peaks, duration),
        "intro_0_4s": int(np.sum((peaks >= 0.0) & (peaks < 4.0))),
    }


def print_variant(label: str, report: dict[str, object]) -> None:
    periodicity = report["periodicity"]
    print(
        f"  {label:13s}: {report['peaks']:4d} peaks  "
        f"Otsu={report['otsu']:.6f}  "
        f"IOI={periodicity['ioi_median_ms']:.1f}ms  "
        f"<50={periodicity['ioi_lt50']}  <100={periodicity['ioi_lt100']}  "
        f"120-130={periodicity['ioi_120_130_pct']:.1f}%  "
        f"AC={periodicity['dominant_period_ms']:.0f}ms  "
        f"0-4s={report['intro_0_4s']}"
    )


def main() -> None:
    audio_path = Path(r"E:\game\Music Hermeneutic AI v2\audio\102 - Dir.wav")
    print(f"▸ B-3 complex flux + hysteresis: {audio_path.name}")
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
    complex_odf = complex_domain_odf(mono, logmel.shape[1])
    if not (
        novelty.shape == flux.shape == complex_odf.shape == times.shape
    ):
        raise RuntimeError("envelope 프레임 정렬에 실패했습니다")

    novelty_norm, _ = sliding_normalize(novelty, times)
    flux_norm, _ = sliding_normalize(flux, times)
    complex_norm, _ = sliding_normalize(complex_odf, times)

    fusion_slide = np.sqrt(novelty_norm * flux_norm)
    nov_complex_slide = np.sqrt(novelty_norm * complex_norm)
    tri_complex_slide = np.cbrt(novelty_norm * flux_norm * complex_norm)

    responses = {
        "fusion_slide": fusion_slide,
        "complex_slide": complex_norm,
        "nov_complex_slide": nov_complex_slide,
        "tri_complex_slide": tri_complex_slide,
    }
    for label, response in responses.items():
        if response.shape != times.shape or not np.isfinite(response).all():
            raise RuntimeError(f"{label} 응답 검증 실패")

    peaks: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    reports: dict[str, dict[str, object]] = {}
    for label, response in responses.items():
        variant_peaks, threshold = response_peaks(response, times)
        peaks[label] = variant_peaks
        thresholds[label] = threshold
        reports[label] = variant_report(variant_peaks, threshold, duration)

    if len(peaks["fusion_slide"]) != EXPECTED_CONTROL_PEAKS:
        raise RuntimeError(
            f"A-2 대조군 재현 실패: {len(peaks['fusion_slide'])} != "
            f"{EXPECTED_CONTROL_PEAKS}"
        )

    print("\n  요약")
    for label in responses:
        print_variant(label, reports[label])

    matching: dict[str, dict[str, int]] = {}
    for label in ("complex_slide", "nov_complex_slide", "tri_complex_slide"):
        common, control_only, variant_only = one_to_one_time_match(
            peaks["fusion_slide"],
            peaks[label],
        )
        matching[label] = {
            "common": int(len(common)),
            "variant_only": int(len(variant_only)),
            "control_only": int(len(control_only)),
        }
        print(
            f"  {label:13s} vs A-2: 공통={len(common)}, "
            f"변형전용={len(variant_only)}, A-2전용={len(control_only)}"
        )

    phase_supported_mask = support_mask(
        peaks["fusion_slide"],
        peaks["complex_slide"],
    )
    phase_supported = peaks["fusion_slide"][phase_supported_mask]
    phase_unsupported = peaks["fusion_slide"][~phase_supported_mask]
    meso = meso_attribution(peaks["fusion_slide"], phase_supported_mask)
    print(
        f"\n  A-2 phase 귀속: supported={len(phase_supported)}, "
        f"unsupported={len(phase_unsupported)}"
    )
    for interval, values in meso.items():
        print(
            f"  {interval}: pairs={values['pairs']}, "
            f"follower supported={values['follower_phase_supported']}, "
            f"unsupported={values['follower_phase_unsupported']}"
        )

    print(
        f"\n  {'시각':>5s}  {'A-2':>5s}  {'complex':>7s}  "
        f"{'nov+cx':>6s}  {'tri':>5s}"
    )
    all_windows = {
        label: reports[label]["window_counts"]
        for label in responses
    }
    for window_i in range(len(all_windows["fusion_slide"])):
        t0 = window_i * WINDOW_S
        timestamp = f"{int(t0) // 60}:{int(t0) % 60:02d}"
        print(
            f"  {timestamp:>5s}  "
            f"{all_windows['fusion_slide'][window_i]:5d}  "
            f"{all_windows['complex_slide'][window_i]:7d}  "
            f"{all_windows['nov_complex_slide'][window_i]:6d}  "
            f"{all_windows['tri_complex_slide'][window_i]:5d}"
        )

    destination = OUT_DIR / "sonify" / "Dir"
    destination.mkdir(parents=True, exist_ok=True)
    click_common = click(3000.0)
    click_variant = click(5000.0, 15.0, 0.8)
    click_control = click(1500.0, 15.0, 0.8)
    output_paths: list[Path] = []

    for label in ("complex_slide", "nov_complex_slide", "tri_complex_slide"):
        full_path = destination / f"전체_{label}_클릭.wav"
        sf.write(
            str(full_path),
            overlay_groups(mono, [(peaks[label], click_common)]),
            SR,
        )
        output_paths.append(full_path)

        common, control_only, variant_only = one_to_one_time_match(
            peaks["fusion_slide"],
            peaks[label],
        )
        compare_path = destination / f"전체_{label}_vs_fusion_slide_비교_클릭.wav"
        sf.write(
            str(compare_path),
            overlay_groups(
                mono,
                [
                    (common, click_common),
                    (variant_only, click_variant),
                    (control_only, click_control),
                ],
            ),
            SR,
        )
        output_paths.append(compare_path)

    attribution_path = (
        destination / "전체_fusion_slide_complex_attribution_클릭.wav"
    )
    sf.write(
        str(attribution_path),
        overlay_groups(
            mono,
            [
                (phase_supported, click_common),
                (phase_unsupported, click_variant),
            ],
        ),
        SR,
    )
    output_paths.append(attribution_path)

    metrics_path = destination / "complex_hysteresis_metrics.json"
    report = {
        "experiment": "s4 B-3 complex flux + hysteresis attribution",
        "audio": audio_path.name,
        "duration_s": duration,
        "fixed_parameters": {
            "sr": SR,
            "n_fft": N_FFT,
            "hop": HOP,
            "complex_prediction": "|X[n-1]| * exp(j*(2phi[n-1]-phi[n-2]))",
            "normalization": "2s smooth residual + 2s sliding p99",
            "min_event_gap_s": MIN_EVENT_GAP_S,
            "matching_tolerance_s": MATCH_TOL_S,
            "meso_bins_s": [list(interval) for interval in MESO_BINS_S],
        },
        "variants": reports,
        "matching_vs_fusion_slide": matching,
        "hysteresis_attribution": {
            "phase_supported": int(len(phase_supported)),
            "phase_unsupported": int(len(phase_unsupported)),
            "meso_follower_support": meso,
        },
        "listening_legend_hz": {
            "comparison_common": 3000,
            "comparison_variant_only": 5000,
            "comparison_a2_only": 1500,
            "attribution_phase_supported": 3000,
            "attribution_phase_unsupported": 5000,
        },
    }
    metrics_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n  비교 범례: 공통=3kHz, B-3전용=5kHz, A-2전용=1.5kHz")
    print("  귀속 범례: phase-supported=3kHz, phase-unsupported=5kHz")
    for path in output_paths:
        print(f"  {path.name}")
    print(f"  {metrics_path.name}")


if __name__ == "__main__":
    main()
