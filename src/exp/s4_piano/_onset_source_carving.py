"""s4_piano: 성운 baseline식 source-derived event carving.

엄격한 tri-complex 결과를 최종 출력으로 쓰지 않고 soft confidence mask로
사용한다. A-2 원형 피크 위치를 보존하면서 tri 증거가 약한 사건만 연속
감쇠하고, A-2/complex의 공통·잔여 역할을 분리 소니파이한다.
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
from config import HOP, MIN_EVENT_GAP_S, OUT_DIR, SR, WINDOW_S
from peak_pick import otsu
from _onset_complex_hysteresis import complex_domain_odf
from _onset_sliding_norm import sliding_normalize
from _onset_wtmm_fusion import (
    click,
    cosine_novelty,
    get_logmel,
    no_max_flux,
    overlay_groups,
    response_peaks,
    window_counts,
)


EXPECTED_A2_PEAKS = 355
EXPECTED_COMPLEX_PEAKS = 450
EXPECTED_TRI_PEAKS = 306
MATCH_TOL_S = MIN_EVENT_GAP_S
MESO_BINS_S = ((0.100, 0.130), (0.130, 0.300))


def values_at_peaks(
    response: np.ndarray,
    peak_times: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    indices = np.searchsorted(times, peak_times, side="left")
    indices = np.clip(indices, 0, len(response) - 1)
    previous = np.maximum(indices - 1, 0)
    use_previous = (
        np.abs(times[previous] - peak_times)
        < np.abs(times[indices] - peak_times)
    )
    indices[use_previous] = previous[use_previous]
    return response[indices]


def one_to_one_pair_masks(
    left: np.ndarray,
    right: np.ndarray,
    tolerance_s: float = MATCH_TOL_S,
) -> tuple[np.ndarray, np.ndarray]:
    """거리순 일대일 배정 결과를 양쪽 boolean mask로 반환."""
    candidates: list[tuple[float, int, int]] = []
    for left_i, left_time in enumerate(left):
        lo = int(np.searchsorted(right, left_time - tolerance_s, side="left"))
        hi = int(np.searchsorted(right, left_time + tolerance_s, side="right"))
        candidates.extend(
            (abs(left_time - right[right_i]), left_i, right_i)
            for right_i in range(lo, hi)
        )
    left_mask = np.zeros(len(left), dtype=bool)
    right_mask = np.zeros(len(right), dtype=bool)
    for _, left_i, right_i in sorted(candidates):
        if not left_mask[left_i] and not right_mask[right_i]:
            left_mask[left_i] = True
            right_mask[right_i] = True
    return left_mask, right_mask


def confidence_summary(
    peak_times: np.ndarray,
    times: np.ndarray,
    responses: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for name, response in responses.items():
        values = values_at_peaks(response, peak_times, times)
        if len(values) == 0:
            summary[name] = {"median": 0.0, "p90": 0.0}
        else:
            summary[name] = {
                "median": float(np.median(values)),
                "p90": float(np.percentile(values, 90.0)),
            }
    return summary


def interval_counts(peaks: np.ndarray) -> dict[str, int]:
    if len(peaks) < 2:
        return {
            f"{int(lo * 1000)}-{int(hi * 1000)}ms": 0
            for lo, hi in MESO_BINS_S
        }
    intervals = np.diff(peaks)
    return {
        f"{int(lo * 1000)}-{int(hi * 1000)}ms": int(
            np.sum((intervals >= lo) & (intervals < hi))
        )
        for lo, hi in MESO_BINS_S
    }


def follower_counts(
    source_peaks: np.ndarray,
    category_mask: np.ndarray,
) -> dict[str, int]:
    """source의 meso 쌍에서 follower가 category에 속하는 횟수."""
    intervals = np.diff(source_peaks)
    return {
        f"{int(lo * 1000)}-{int(hi * 1000)}ms": int(
            np.sum(
                (intervals >= lo)
                & (intervals < hi)
                & category_mask[1:]
            )
        )
        for lo, hi in MESO_BINS_S
    }


def category_report(
    peaks: np.ndarray,
    duration: float,
    times: np.ndarray,
    responses: dict[str, np.ndarray],
    *,
    source_peaks: np.ndarray | None = None,
    source_mask: np.ndarray | None = None,
) -> dict[str, object]:
    report: dict[str, object] = {
        "count": int(len(peaks)),
        "intro_0_4s": int(np.sum((peaks >= 0.0) & (peaks < 4.0))),
        "window_counts": window_counts(peaks, duration),
        "own_interval_counts": interval_counts(peaks),
        "confidence": confidence_summary(peaks, times, responses),
    }
    if source_peaks is not None and source_mask is not None:
        report["source_meso_follower_counts"] = follower_counts(
            source_peaks,
            source_mask,
        )
    return report


def overlay_stereo_ab(
    mono: np.ndarray,
    left_peaks: np.ndarray,
    right_peaks: np.ndarray,
) -> np.ndarray:
    left = overlay_groups(mono, [(left_peaks, click(3000.0))])
    right = overlay_groups(mono, [(right_peaks, click(3000.0))])
    return np.column_stack([left, right])


def main() -> None:
    audio_path = Path(r"E:\game\Music Hermeneutic AI v2\audio\102 - Dir.wav")
    print(f"▸ Source-derived event carving: {audio_path.name}")
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

    novelty_norm, _ = sliding_normalize(novelty, times)
    flux_norm, _ = sliding_normalize(flux, times)
    complex_norm, _ = sliding_normalize(complex_odf, times)
    fusion_slide = np.sqrt(novelty_norm * flux_norm)
    tri_response = np.cbrt(novelty_norm * flux_norm * complex_norm)

    a2_peaks, a2_threshold = response_peaks(fusion_slide, times)
    complex_peaks, complex_threshold = response_peaks(complex_norm, times)
    tri_peaks, tri_threshold = response_peaks(tri_response, times)
    expected = (
        (len(a2_peaks), EXPECTED_A2_PEAKS, "A-2"),
        (len(complex_peaks), EXPECTED_COMPLEX_PEAKS, "complex"),
        (len(tri_peaks), EXPECTED_TRI_PEAKS, "tri"),
    )
    for actual, wanted, label in expected:
        if actual != wanted:
            raise RuntimeError(f"{label} 재현 실패: {actual} != {wanted}")

    tri_mask = np.clip(tri_response / max(tri_threshold, 1e-12), 0.0, 1.0)
    carved_response = fusion_slide * np.sqrt(tri_mask)
    carved_positive = carved_response[carved_response > 0]
    if len(carved_positive) < 10 or not np.isfinite(carved_response).all():
        raise RuntimeError("carved response 검증 실패")
    carved_threshold = otsu(carved_positive)

    carved_scores = values_at_peaks(carved_response, a2_peaks, times)
    carved_keep_mask = carved_scores > carved_threshold
    carved_peaks = a2_peaks[carved_keep_mask]
    carved_removed = a2_peaks[~carved_keep_mask]
    if not np.all(np.isin(carved_peaks, a2_peaks)):
        raise RuntimeError("carved가 A-2 부분집합이 아닙니다")

    a2_complex_mask, complex_a2_mask = one_to_one_pair_masks(
        a2_peaks,
        complex_peaks,
    )
    a2_tri_mask, _ = one_to_one_pair_masks(a2_peaks, tri_peaks)
    core_a2_complex = a2_peaks[a2_complex_mask]
    residual_a2_only = a2_peaks[~a2_complex_mask]
    rescue_complex_only = complex_peaks[~complex_a2_mask]
    core_a2_tri = a2_peaks[a2_tri_mask]

    response_map = {
        "fusion_slide": fusion_slide,
        "complex_norm": complex_norm,
        "tri_response": tri_response,
        "tri_mask": tri_mask,
        "carved_response": carved_response,
    }
    categories = {
        "a2_all": category_report(
            a2_peaks,
            duration,
            times,
            response_map,
            source_peaks=a2_peaks,
            source_mask=np.ones(len(a2_peaks), dtype=bool),
        ),
        "a2_complex_core": category_report(
            core_a2_complex,
            duration,
            times,
            response_map,
            source_peaks=a2_peaks,
            source_mask=a2_complex_mask,
        ),
        "a2_only_residual": category_report(
            residual_a2_only,
            duration,
            times,
            response_map,
            source_peaks=a2_peaks,
            source_mask=~a2_complex_mask,
        ),
        "complex_only_rescue": category_report(
            rescue_complex_only,
            duration,
            times,
            response_map,
        ),
        "a2_tri_core": category_report(
            core_a2_tri,
            duration,
            times,
            response_map,
            source_peaks=a2_peaks,
            source_mask=a2_tri_mask,
        ),
        "carved_survived": category_report(
            carved_peaks,
            duration,
            times,
            response_map,
            source_peaks=a2_peaks,
            source_mask=carved_keep_mask,
        ),
        "carved_removed": category_report(
            carved_removed,
            duration,
            times,
            response_map,
            source_peaks=a2_peaks,
            source_mask=~carved_keep_mask,
        ),
    }

    print("\n  재현")
    print(
        f"  A-2={len(a2_peaks)}, complex={len(complex_peaks)}, "
        f"tri={len(tri_peaks)}"
    )
    print(
        f"  A2∩complex={len(core_a2_complex)}, A2-only={len(residual_a2_only)}, "
        f"complex-only={len(rescue_complex_only)}"
    )
    print(
        f"  A2∩tri={len(core_a2_tri)}, carved={len(carved_peaks)}, "
        f"carved-removed={len(carved_removed)}, carved Otsu={carved_threshold:.6f}"
    )
    print("\n  meso follower (A-2 원 순서)")
    for name in (
        "a2_complex_core",
        "a2_only_residual",
        "a2_tri_core",
        "carved_survived",
        "carved_removed",
    ):
        print(
            f"  {name:20s}: "
            f"{categories[name]['source_meso_follower_counts']}"
        )

    destination = OUT_DIR / "sonify" / "Dir"
    destination.mkdir(parents=True, exist_ok=True)
    click_3k = click(3000.0)
    click_removed = click(1500.0, 15.0, 0.8)
    output_paths: list[Path] = []

    isolated = {
        "전체_core_a2_complex_only_클릭.wav": core_a2_complex,
        "전체_residual_a2_only_클릭.wav": residual_a2_only,
        "전체_rescue_complex_only_클릭.wav": rescue_complex_only,
        "전체_carved_클릭.wav": carved_peaks,
        "전체_carved_removed_only_클릭.wav": carved_removed,
    }
    for filename, peaks in isolated.items():
        path = destination / filename
        sf.write(str(path), overlay_groups(mono, [(peaks, click_3k)]), SR)
        output_paths.append(path)

    stereo_path = destination / "전체_a2_vs_complex_stereo.wav"
    sf.write(
        str(stereo_path),
        overlay_stereo_ab(mono, a2_peaks, complex_peaks),
        SR,
    )
    output_paths.append(stereo_path)

    compare_path = destination / "전체_carved_vs_a2_비교_클릭.wav"
    sf.write(
        str(compare_path),
        overlay_groups(
            mono,
            [
                (carved_peaks, click_3k),
                (carved_removed, click_removed),
            ],
        ),
        SR,
    )
    output_paths.append(compare_path)

    metrics_path = destination / "source_carving_metrics.json"
    report = {
        "experiment": "s4 source-derived event carving",
        "audio": audio_path.name,
        "duration_s": duration,
        "fixed_parameters": {
            "sr": SR,
            "hop": HOP,
            "matching_tolerance_s": MATCH_TOL_S,
            "tri_mask": "clip(tri_response / tri_otsu, 0, 1)",
            "carved_response": "fusion_slide * sqrt(tri_mask)",
            "carved_selection": "existing A-2 peaks with carved_score > carved_otsu",
        },
        "reproduced": {
            "a2_peaks": int(len(a2_peaks)),
            "a2_otsu": a2_threshold,
            "complex_peaks": int(len(complex_peaks)),
            "complex_otsu": complex_threshold,
            "tri_peaks": int(len(tri_peaks)),
            "tri_otsu": tri_threshold,
        },
        "carved": {
            "peaks": int(len(carved_peaks)),
            "removed": int(len(carved_removed)),
            "otsu": carved_threshold,
            "is_a2_subset": True,
        },
        "categories": categories,
        "listening": {
            "isolated_click_hz": 3000,
            "stereo_left": "A-2 clicks",
            "stereo_right": "complex clicks",
            "carved_compare_common_hz": 3000,
            "carved_compare_removed_hz": 1500,
        },
    }
    metrics_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n  소니파이")
    for path in output_paths:
        print(f"  {path.name}")
    print(f"  {metrics_path.name}")
    print("  stereo: L=A-2, R=complex")
    print("  carved 비교: 생존=3kHz, 감쇠=1.5kHz")


if __name__ == "__main__":
    main()
