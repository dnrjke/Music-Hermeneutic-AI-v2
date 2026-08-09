"""s4_piano: positive spectral-distribution novelty.

단순 positive log-mel flux는 기존 plain flux와 중복되므로 반복하지 않는다.
log-mel을 [0, 1] activation으로 옮긴 뒤 프레임별 L1 분포로 정규화하고,
새 프레임에서 점유율이 증가한 주파수 질량만 합산한다. 진폭 스케일에는
불변이고 cosine과 다른 L1/total-variation 기하를 사용한다.
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
from config import FMIN, HOP, N_MELS, OUT_DIR, SR, WINDOW_S
from _onset_complex_hysteresis import complex_domain_odf
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


EXPECTED_A2_PEAKS = 355
EXPECTED_TRI_COMPLEX_PEAKS = 306
TOP_DB = 80.0


def positive_distribution_novelty(
    logmel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """L1-normalized spectral distribution의 양의 질량 이동."""
    activation = np.clip((logmel + TOP_DB) / TOP_DB, 0.0, 1.0)
    totals = activation.sum(axis=0, keepdims=True)
    distribution = np.divide(
        activation,
        totals,
        out=np.zeros_like(activation),
        where=totals >= 1e-12,
    )
    positive_delta = np.zeros_like(distribution)
    positive_delta[:, 1:] = np.maximum(
        distribution[:, 1:] - distribution[:, :-1],
        0.0,
    )
    envelope = positive_delta.sum(axis=0, dtype=np.float64)
    if not np.isfinite(envelope).all():
        raise RuntimeError("positive distribution novelty에 NaN/Inf가 있습니다")
    return envelope, positive_delta


def variant_report(
    peaks: np.ndarray,
    threshold: float | None,
    duration: float,
) -> dict[str, object]:
    return {
        "peaks": int(len(peaks)),
        "otsu": threshold,
        "periodicity": periodicity_metrics(peaks, duration),
        "window_counts": window_counts(peaks, duration),
        "intro_0_4s": int(np.sum((peaks >= 0.0) & (peaks < 4.0))),
    }


def print_variant(name: str, report: dict[str, object]) -> None:
    periodicity = report["periodicity"]
    print(
        f"  {name:20s}: {report['peaks']:4d}  "
        f"IOI={periodicity['ioi_median_ms']:.1f}ms  "
        f"<50={periodicity['ioi_lt50']}  <100={periodicity['ioi_lt100']}  "
        f"120-130={periodicity['ioi_120_130_pct']:.1f}%  "
        f"AC={periodicity['dominant_period_ms']:.0f}ms  "
        f"0-4s={report['intro_0_4s']}"
    )


def frame_indices(peak_times: np.ndarray, times: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(times, peak_times, side="left")
    indices = np.clip(indices, 0, len(times) - 1)
    previous = np.maximum(indices - 1, 0)
    use_previous = (
        np.abs(times[previous] - peak_times)
        < np.abs(times[indices] - peak_times)
    )
    indices[use_previous] = previous[use_previous]
    return indices


def contribution_summary(
    peak_times: np.ndarray,
    positive_delta: np.ndarray,
    times: np.ndarray,
    mel_frequencies: np.ndarray,
) -> dict[str, object]:
    if len(peak_times) == 0:
        return {
            "events": 0,
            "weighted_frequency_hz": 0.0,
            "top_bins": [],
        }
    indices = frame_indices(peak_times, times)
    contribution = positive_delta[:, indices].sum(axis=1)
    total = float(contribution.sum())
    if total < 1e-12:
        return {
            "events": int(len(peak_times)),
            "weighted_frequency_hz": 0.0,
            "top_bins": [],
        }
    shares = contribution / total
    top = np.argsort(shares)[::-1][:10]
    return {
        "events": int(len(peak_times)),
        "weighted_frequency_hz": float(np.sum(mel_frequencies * shares)),
        "top_bins": [
            {
                "bin": int(index),
                "frequency_hz": float(mel_frequencies[index]),
                "share": float(shares[index]),
            }
            for index in top
        ],
    }


def main() -> None:
    audio_path = Path(r"E:\game\Music Hermeneutic AI v2\audio\102 - Dir.wav")
    print(f"▸ Positive distribution novelty: {audio_path.name}")
    mono = load_mono(audio_path)
    duration = duration_s(audio_path)
    logmel = get_logmel(mono)
    times = librosa.frames_to_time(
        np.arange(logmel.shape[1]),
        sr=SR,
        hop_length=HOP,
    )

    cosine = cosine_novelty(logmel)
    posdist, positive_delta = positive_distribution_novelty(logmel)
    flux = no_max_flux(logmel)
    complex_odf = complex_domain_odf(mono, logmel.shape[1])
    if not (
        cosine.shape == posdist.shape == flux.shape == complex_odf.shape == times.shape
    ):
        raise RuntimeError("onset envelope 프레임 정렬 실패")

    cosine_norm, _ = sliding_normalize(cosine, times)
    posdist_norm, _ = sliding_normalize(posdist, times)
    flux_norm, _ = sliding_normalize(flux, times)
    complex_norm, _ = sliding_normalize(complex_odf, times)

    responses = {
        "cosine_slide": cosine_norm,
        "fusion_slide": np.sqrt(cosine_norm * flux_norm),
        "tri_complex_slide": np.cbrt(cosine_norm * flux_norm * complex_norm),
        "posdist_slide": posdist_norm,
        "posdist_flux_slide": np.sqrt(posdist_norm * flux_norm),
        "posdist_tri_slide": np.cbrt(posdist_norm * flux_norm * complex_norm),
    }
    peaks: dict[str, np.ndarray] = {}
    reports: dict[str, dict[str, object]] = {}
    for name, response in responses.items():
        if not np.isfinite(response).all():
            raise RuntimeError(f"{name} 응답에 NaN/Inf가 있습니다")
        variant_peaks, threshold = response_peaks(response, times)
        peaks[name] = variant_peaks
        reports[name] = variant_report(variant_peaks, threshold, duration)

    if len(peaks["fusion_slide"]) != EXPECTED_A2_PEAKS:
        raise RuntimeError(
            f"A-2 재현 실패: {len(peaks['fusion_slide'])} != {EXPECTED_A2_PEAKS}"
        )
    if len(peaks["tri_complex_slide"]) != EXPECTED_TRI_COMPLEX_PEAKS:
        raise RuntimeError(
            "tri-complex 재현 실패: "
            f"{len(peaks['tri_complex_slide'])} != {EXPECTED_TRI_COMPLEX_PEAKS}"
        )

    print("\n  요약")
    for name in responses:
        print_variant(name, reports[name])

    comparisons = {
        "posdist_slide_vs_cosine_slide": ("cosine_slide", "posdist_slide"),
        "posdist_flux_vs_fusion_slide": ("fusion_slide", "posdist_flux_slide"),
        "posdist_tri_vs_tri_complex": (
            "tri_complex_slide",
            "posdist_tri_slide",
        ),
    }
    matching: dict[str, dict[str, int]] = {}
    event_roles: dict[str, dict[str, object]] = {}
    matched_times: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for label, (control_name, variant_name) in comparisons.items():
        common, control_only, variant_only = one_to_one_time_match(
            peaks[control_name],
            peaks[variant_name],
        )
        matched_times[label] = (common, control_only, variant_only)
        matching[label] = {
            "common": int(len(common)),
            "variant_only": int(len(variant_only)),
            "control_only": int(len(control_only)),
        }
        event_roles[label] = {
            "common": periodicity_metrics(common, duration),
            "variant_only": periodicity_metrics(variant_only, duration),
            "control_only": periodicity_metrics(control_only, duration),
        }
        print(
            f"  {label}: 공통={len(common)}, positive전용={len(variant_only)}, "
            f"기존전용={len(control_only)}"
        )

    _, _, positive_only = matched_times["posdist_flux_vs_fusion_slide"]
    rescue_peaks = np.sort(
        np.concatenate([peaks["fusion_slide"], positive_only])
    )
    if len(rescue_peaks) > 1 and np.min(np.diff(rescue_peaks)) < 0.03:
        raise RuntimeError("A-2와 positive rescue 사이 30ms 미만 충돌")
    peaks["a2_posdist_rescue"] = rescue_peaks
    reports["a2_posdist_rescue"] = variant_report(rescue_peaks, None, duration)
    print_variant("a2_posdist_rescue", reports["a2_posdist_rescue"])

    mel_frequencies = librosa.mel_frequencies(
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=SR / 2,
    )
    frequency_attribution: dict[str, dict[str, object]] = {
        "posdist_all": contribution_summary(
            peaks["posdist_slide"],
            positive_delta,
            times,
            mel_frequencies,
        ),
    }
    for label, (_, _, variant_only) in matched_times.items():
        frequency_attribution[f"{label}_positive_only"] = contribution_summary(
            variant_only,
            positive_delta,
            times,
            mel_frequencies,
        )

    print(
        f"\n  {'시각':>5s}  {'cos':>4s}  {'A-2':>4s}  "
        f"{'pos':>4s}  {'pos×f':>5s}  {'pos×tri':>7s}"
    )
    windows = {
        name: reports[name]["window_counts"]
        for name in reports
    }
    for window_i in range(len(windows["fusion_slide"])):
        t0 = window_i * WINDOW_S
        timestamp = f"{int(t0) // 60}:{int(t0) % 60:02d}"
        print(
            f"  {timestamp:>5s}  "
            f"{windows['cosine_slide'][window_i]:4d}  "
            f"{windows['fusion_slide'][window_i]:4d}  "
            f"{windows['posdist_slide'][window_i]:4d}  "
            f"{windows['posdist_flux_slide'][window_i]:5d}  "
            f"{windows['posdist_tri_slide'][window_i]:7d}"
        )

    destination = OUT_DIR / "sonify" / "Dir"
    destination.mkdir(parents=True, exist_ok=True)
    click_common = click(3000.0)
    click_variant = click(5000.0, 15.0, 0.8)
    click_control = click(1500.0, 15.0, 0.8)
    output_paths: list[Path] = []

    for name in ("posdist_slide", "posdist_flux_slide", "posdist_tri_slide"):
        path = destination / f"전체_{name}_클릭.wav"
        sf.write(
            str(path),
            overlay_groups(mono, [(peaks[name], click_common)]),
            SR,
        )
        output_paths.append(path)

    rescue_path = destination / "전체_a2_posdist_rescue_클릭.wav"
    sf.write(
        str(rescue_path),
        overlay_groups(
            mono,
            [
                (peaks["fusion_slide"], click_common),
                (positive_only, click_variant),
            ],
        ),
        SR,
    )
    output_paths.append(rescue_path)

    compare_filenames = {
        "posdist_slide_vs_cosine_slide": (
            "전체_posdist_slide_vs_cosine_slide_비교_클릭.wav"
        ),
        "posdist_flux_vs_fusion_slide": (
            "전체_posdist_flux_slide_vs_fusion_slide_비교_클릭.wav"
        ),
        "posdist_tri_vs_tri_complex": (
            "전체_posdist_tri_slide_vs_tri_complex_slide_비교_클릭.wav"
        ),
    }
    for label, filename in compare_filenames.items():
        common, control_only, variant_only = matched_times[label]
        path = destination / filename
        sf.write(
            str(path),
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
        output_paths.append(path)

    _, a2_only, positive_only = matched_times["posdist_flux_vs_fusion_slide"]
    role_outputs = {
        "전체_posdist_flux_A2공통_클릭.wav": matched_times[
            "posdist_flux_vs_fusion_slide"
        ][0],
        "전체_posdist_flux_positive전용_클릭.wav": positive_only,
        "전체_posdist_flux_A2전용_클릭.wav": a2_only,
    }
    for filename, role_peaks in role_outputs.items():
        path = destination / filename
        sf.write(
            str(path),
            overlay_groups(mono, [(role_peaks, click_common)]),
            SR,
        )
        output_paths.append(path)

    metrics_path = destination / "posdist_metrics.json"
    report = {
        "experiment": "s4 positive spectral-distribution novelty",
        "audio": audio_path.name,
        "duration_s": duration,
        "definition": {
            "activation": "clip((logmel + 80) / 80, 0, 1)",
            "distribution": "activation / sum_frequency(activation)",
            "novelty": "sum_frequency(max(p[t] - p[t-1], 0))",
            "note": "amplitude-invariant total-variation geometry; not raw plain flux",
        },
        "variants": reports,
        "peak_times_s": {
            name: [float(time) for time in variant_peaks]
            for name, variant_peaks in peaks.items()
        },
        "matching": matching,
        "event_roles": event_roles,
        "frequency_attribution": frequency_attribution,
        "listening_legend_hz": {
            "common": 3000,
            "positive_variant_only": 5000,
            "existing_control_only": 1500,
        },
    }
    metrics_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\n  비교 범례: 공통=3kHz, positive전용=5kHz, 기존전용=1.5kHz")
    for path in output_paths:
        print(f"  {path.name}")
    print(f"  {metrics_path.name}")


if __name__ == "__main__":
    main()
