"""s4_piano A-1/A-3: WTMM-inspired 극대 정합 + 연속값 기하평균.

A-1: novelty의 디아딕 스케일 사다리에서 인접 스케일 극대선을 정합한다.
A-3: novelty와 no-maxfilter flux의 정규화 연속값을 기하평균한다.

[D-21] 스케일, 정합 허용오차, 정합 기준과 융합식은 실행 전에 고정한다.
현재 최선 nov_n2s와 A-1, A-3, A-1+A-3을 분해 비교한다.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
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
    SUPERFLUX_LAG,
    WINDOW_S,
)
from peak_pick import otsu


FRAME_DT = HOP / SR
SCALES_S = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
NORM_PERCENTILE = 99.0
CHAIN_TOL_S = MIN_EVENT_GAP_S
CONTROL_SCALE_S = 2.0


def get_logmel(mono: np.ndarray) -> np.ndarray:
    stft_mag = np.abs(librosa.stft(mono, n_fft=N_FFT, hop_length=HOP))
    mel = librosa.feature.melspectrogram(
        S=stft_mag**2,
        sr=SR,
        n_mels=N_MELS,
        fmin=FMIN,
    )
    return librosa.power_to_db(mel, ref=np.max, top_db=80.0)


def cosine_novelty(logmel: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(logmel, axis=0, keepdims=True)
    normalized = logmel / np.maximum(norms, 1e-12)
    similarity = np.sum(normalized[:, 1:] * normalized[:, :-1], axis=0)
    env = np.zeros(logmel.shape[1], dtype=np.float64)
    env[1:] = np.maximum(1.0 - np.clip(similarity, -1.0, 1.0), 0.0)
    return env


def no_max_flux(logmel: np.ndarray) -> np.ndarray:
    env = librosa.onset.onset_strength(
        S=logmel,
        sr=SR,
        hop_length=HOP,
        lag=SUPERFLUX_LAG,
        max_size=1,
        detrend=True,
    )
    return librosa.util.fix_length(env, size=logmel.shape[1])


def scale_normalize(env: np.ndarray, times: np.ndarray, scale_s: float) -> np.ndarray:
    """scale 길이 smooth를 제거하고 같은 길이 블록에서 99-pct 정규화."""
    if env.shape != times.shape:
        raise ValueError(f"envelope/time 길이 불일치: {env.shape} != {times.shape}")
    if not np.isfinite(env).all():
        raise ValueError("envelope에 NaN/Inf가 있습니다")

    positive = np.maximum(env, 0.0)
    smooth_frames = max(3, int(scale_s / FRAME_DT) | 1)
    smooth = uniform_filter1d(positive, size=smooth_frames, mode="reflect")
    residual = np.maximum(positive - smooth, 0.0)

    normalized = np.zeros_like(residual)
    duration = times[-1] + FRAME_DT
    for block_i in range(int(np.ceil(duration / scale_s))):
        t0 = block_i * scale_s
        t1 = (block_i + 1) * scale_s
        mask = (times >= t0) & (times < t1)
        segment = residual[mask]
        segment_positive = segment[segment > 0]
        if len(segment_positive) < 3:
            continue
        scale = np.percentile(segment_positive, NORM_PERCENTILE)
        if scale >= 1e-12:
            normalized[mask] = np.clip(segment / scale, 0.0, None)
    return normalized


def local_maxima_above_otsu(response: np.ndarray) -> tuple[np.ndarray, float]:
    positive = response[response > 0]
    if len(positive) < 10:
        raise RuntimeError("Otsu를 계산할 양수 응답이 부족합니다")
    threshold = otsu(positive)
    maxima = np.flatnonzero(
        (response[1:-1] >= response[:-2])
        & (response[1:-1] > response[2:])
        & (response[1:-1] > threshold)
    ) + 1
    if len(maxima) == 0:
        raise RuntimeError("Otsu 위 국소극대가 없습니다")
    return maxima, threshold


def greedy_times(
    indices: np.ndarray,
    values: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    selected: list[int] = []
    for index in sorted(indices.tolist(), key=lambda i: -values[i]):
        if all(abs(times[index] - times[other]) >= MIN_EVENT_GAP_S for other in selected):
            selected.append(index)
    if not selected:
        raise RuntimeError("30ms 선택 후 피크가 없습니다")
    return times[np.sort(np.asarray(selected, dtype=int))]


def response_peaks(response: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, float]:
    maxima, threshold = local_maxima_above_otsu(response)
    return greedy_times(maxima, response, times), threshold


def one_to_one_index_pairs(
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    times: np.ndarray,
    tolerance_s: float,
) -> list[tuple[int, int]]:
    candidates: list[tuple[float, int, int]] = []
    right_times = times[right_indices]
    for left_pos, left_index in enumerate(left_indices):
        left_time = times[left_index]
        lo = int(np.searchsorted(right_times, left_time - tolerance_s, side="left"))
        hi = int(np.searchsorted(right_times, left_time + tolerance_s, side="right"))
        candidates.extend(
            (abs(left_time - right_times[right_pos]), left_pos, right_pos)
            for right_pos in range(lo, hi)
        )

    used_left: set[int] = set()
    used_right: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, left_pos, right_pos in sorted(candidates):
        if left_pos not in used_left and right_pos not in used_right:
            used_left.add(left_pos)
            used_right.add(right_pos)
            pairs.append((left_pos, right_pos))
    return pairs


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def chain_scale_maxima(
    responses: list[np.ndarray],
    times: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """인접 스케일 일대일 정합 후 2스케일 이상 지속하는 chain만 채택."""
    maxima_by_scale: list[np.ndarray] = []
    thresholds: list[float] = []
    node_ids: list[np.ndarray] = []
    node_scale: list[int] = []
    node_frame: list[int] = []
    node_strength: list[float] = []

    for scale_i, response in enumerate(responses):
        maxima, threshold = local_maxima_above_otsu(response)
        maxima_by_scale.append(maxima)
        thresholds.append(threshold)
        ids = np.arange(len(node_frame), len(node_frame) + len(maxima), dtype=int)
        node_ids.append(ids)
        node_scale.extend([scale_i] * len(maxima))
        node_frame.extend(maxima.tolist())
        node_strength.extend(response[maxima].tolist())

    union_find = UnionFind(len(node_frame))
    adjacent_match_counts: list[int] = []
    for scale_i in range(len(responses) - 1):
        pairs = one_to_one_index_pairs(
            maxima_by_scale[scale_i],
            maxima_by_scale[scale_i + 1],
            times,
            CHAIN_TOL_S,
        )
        adjacent_match_counts.append(len(pairs))
        for left_pos, right_pos in pairs:
            union_find.union(
                int(node_ids[scale_i][left_pos]),
                int(node_ids[scale_i + 1][right_pos]),
            )

    components: dict[int, list[int]] = defaultdict(list)
    for node_id in range(len(node_frame)):
        components[union_find.find(node_id)].append(node_id)

    candidate_times: list[float] = []
    candidate_scores: list[float] = []
    candidate_supports: list[int] = []
    for component in components.values():
        scales = {node_scale[node_id] for node_id in component}
        support = len(scales)
        if support < 2:
            continue
        component_times = [times[node_frame[node_id]] for node_id in component]
        mean_strength = float(np.mean([node_strength[node_id] for node_id in component]))
        candidate_times.append(float(np.median(component_times)))
        # support가 응답 강도보다 항상 우선하도록 [0,1) tie-break를 더한다.
        candidate_scores.append(support + mean_strength / (1.0 + mean_strength))
        candidate_supports.append(support)

    if not candidate_times:
        raise RuntimeError("2-of-adjacent-scale 조건을 만족한 chain이 없습니다")

    candidate_times_array = np.asarray(candidate_times)
    order = np.argsort(candidate_times_array)
    candidate_times_array = candidate_times_array[order]
    candidate_scores_array = np.asarray(candidate_scores)[order]
    candidate_supports_array = np.asarray(candidate_supports, dtype=int)[order]

    selected: list[int] = []
    for index in sorted(
        range(len(candidate_times_array)),
        key=lambda i: -candidate_scores_array[i],
    ):
        if all(
            abs(candidate_times_array[index] - candidate_times_array[other])
            >= MIN_EVENT_GAP_S
            for other in selected
        ):
            selected.append(index)
    selected_array = np.sort(np.asarray(selected, dtype=int))
    peaks = candidate_times_array[selected_array]
    selected_supports = candidate_supports_array[selected_array]

    diagnostics: dict[str, object] = {
        "scale_candidate_counts": {
            f"{scale:g}s": int(len(maxima))
            for scale, maxima in zip(SCALES_S, maxima_by_scale)
        },
        "scale_otsu": {
            f"{scale:g}s": threshold
            for scale, threshold in zip(SCALES_S, thresholds)
        },
        "adjacent_match_counts": {
            f"{SCALES_S[i]:g}s-{SCALES_S[i + 1]:g}s": count
            for i, count in enumerate(adjacent_match_counts)
        },
        "qualified_chains_before_gap": int(len(candidate_times_array)),
        "selected_chains": int(len(peaks)),
        "support_distribution": {
            str(support): int(count)
            for support, count in sorted(Counter(selected_supports.tolist()).items())
        },
    }
    return peaks, diagnostics


def one_to_one_time_match(
    control: np.ndarray,
    variant: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidates: list[tuple[float, int, int]] = []
    for control_i, control_time in enumerate(control):
        lo = int(np.searchsorted(variant, control_time - CHAIN_TOL_S, side="left"))
        hi = int(np.searchsorted(variant, control_time + CHAIN_TOL_S, side="right"))
        candidates.extend(
            (abs(control_time - variant[variant_i]), control_i, variant_i)
            for variant_i in range(lo, hi)
        )
    used_control: set[int] = set()
    used_variant: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, control_i, variant_i in sorted(candidates):
        if control_i not in used_control and variant_i not in used_variant:
            used_control.add(control_i)
            used_variant.add(variant_i)
            pairs.append((control_i, variant_i))
    common = np.asarray(
        sorted((control[i] + variant[j]) / 2.0 for i, j in pairs),
        dtype=np.float64,
    )
    return (
        common,
        np.delete(control, sorted(used_control)),
        np.delete(variant, sorted(used_variant)),
    )


def periodicity_metrics(peaks: np.ndarray, duration: float) -> dict[str, float | int]:
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
    autocorrelation = correlate(impulse, impulse, mode="full", method="fft")
    autocorrelation = autocorrelation[
        len(impulse) - 1:len(impulse) + max_lag
    ]
    autocorrelation[0] = 0.0
    ac_peaks, _ = find_peaks(
        autocorrelation,
        height=np.max(autocorrelation) * 0.3,
    )
    dominant_ms = (
        float(ac_peaks[np.argmax(autocorrelation[ac_peaks])] * resolution_ms)
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


def window_counts(peaks: np.ndarray, duration: float) -> list[int]:
    return [
        int(np.sum((peaks >= t0) & (peaks < t0 + WINDOW_S)))
        for t0 in np.arange(0.0, duration, WINDOW_S)
    ]


def click(freq: float, duration_ms: float = 12.0, amp: float = 0.7) -> np.ndarray:
    n = int(SR * duration_ms / 1000.0)
    time = np.arange(n, dtype=np.float32) / SR
    decay = np.exp(-time * 1000.0 / duration_ms)
    return (amp * decay * np.sin(2.0 * np.pi * freq * time)).astype(np.float32)


def overlay_groups(
    mono: np.ndarray,
    groups: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    output = mono.copy()
    for peak_times, click_sound in groups:
        for peak_time in peak_times:
            start = int(peak_time * SR)
            stop = min(start + len(click_sound), len(output))
            if stop > start:
                output[start:stop] += click_sound[:stop - start]
    return np.clip(output, -1.0, 1.0)


def main() -> None:
    audio_path = Path(r"E:\game\Music Hermeneutic AI v2\audio\102 - Dir.wav")
    print(f"▸ A-1/A-3 WTMM + fusion: {audio_path.name}")
    print(f"  scales={SCALES_S}, chain=adjacent 2+, tol={CHAIN_TOL_S * 1000:.0f}ms")
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
    if novelty.shape != times.shape or flux.shape != times.shape:
        raise RuntimeError("onset envelope 프레임 정렬에 실패했습니다")

    novelty_responses = [
        scale_normalize(novelty, times, scale)
        for scale in SCALES_S
    ]
    flux_responses = [
        scale_normalize(flux, times, scale)
        for scale in SCALES_S
    ]
    fused_responses = [
        np.sqrt(novelty_response * flux_response)
        for novelty_response, flux_response in zip(novelty_responses, flux_responses)
    ]

    control_index = SCALES_S.index(CONTROL_SCALE_S)
    control_peaks, control_threshold = response_peaks(
        novelty_responses[control_index],
        times,
    )
    wtmm_nov_peaks, wtmm_nov_chain = chain_scale_maxima(
        novelty_responses,
        times,
    )
    fusion_n2s_peaks, fusion_n2s_threshold = response_peaks(
        fused_responses[control_index],
        times,
    )
    wtmm_fusion_peaks, wtmm_fusion_chain = chain_scale_maxima(
        fused_responses,
        times,
    )

    variants = {
        "nov_n2s": control_peaks,
        "wtmm_nov": wtmm_nov_peaks,
        "fusion_n2s": fusion_n2s_peaks,
        "wtmm_fusion": wtmm_fusion_peaks,
    }
    metrics: dict[str, dict[str, object]] = {}
    matches: dict[str, dict[str, int]] = {}

    print("\n  요약")
    for name, peaks in variants.items():
        periodicity = periodicity_metrics(peaks, duration)
        metrics[name] = {
            "peaks": int(len(peaks)),
            "periodicity": periodicity,
            "window_counts": window_counts(peaks, duration),
        }
        print(
            f"  {name:12s}: {len(peaks):4d} peaks  "
            f"IOI={periodicity['ioi_median_ms']:.1f}ms  "
            f"<50={periodicity['ioi_lt50']}  <100={periodicity['ioi_lt100']}  "
            f"120-130={periodicity['ioi_120_130_pct']:.1f}%  "
            f"AC={periodicity['dominant_period_ms']:.0f}ms"
        )

    for name in ("wtmm_nov", "fusion_n2s", "wtmm_fusion"):
        common, control_only, variant_only = one_to_one_time_match(
            control_peaks,
            variants[name],
        )
        matches[name] = {
            "common": int(len(common)),
            "variant_only": int(len(variant_only)),
            "control_only": int(len(control_only)),
        }
        print(
            f"  {name:12s} vs nov_n2s: 공통={len(common)}, "
            f"변형전용={len(variant_only)}, 기존전용={len(control_only)}"
        )

    print(f"\n  {'시각':>5s}  {'control':>7s}  {'A-1':>5s}  {'A-3':>5s}  {'A1+A3':>6s}")
    all_windows = {
        name: metrics[name]["window_counts"]
        for name in variants
    }
    for window_i in range(len(all_windows["nov_n2s"])):
        t0 = window_i * WINDOW_S
        timestamp = f"{int(t0) // 60}:{int(t0) % 60:02d}"
        print(
            f"  {timestamp:>5s}  "
            f"{all_windows['nov_n2s'][window_i]:7d}  "
            f"{all_windows['wtmm_nov'][window_i]:5d}  "
            f"{all_windows['fusion_n2s'][window_i]:5d}  "
            f"{all_windows['wtmm_fusion'][window_i]:6d}"
        )

    destination = OUT_DIR / "sonify" / "Dir"
    destination.mkdir(parents=True, exist_ok=True)
    click_common = click(3000.0)
    click_variant = click(5000.0, 15.0, 0.8)
    click_control = click(1500.0, 15.0, 0.8)

    output_paths: list[Path] = []
    for name in ("wtmm_nov", "fusion_n2s", "wtmm_fusion"):
        peaks = variants[name]
        full_path = destination / f"전체_{name}_클릭.wav"
        sf.write(
            str(full_path),
            overlay_groups(mono, [(peaks, click_common)]),
            SR,
        )
        output_paths.append(full_path)

        common, control_only, variant_only = one_to_one_time_match(
            control_peaks,
            peaks,
        )
        compare_path = destination / f"전체_{name}_vs_nov_n2s_비교_클릭.wav"
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

    report = {
        "experiment": "s4 A-1/A-3 WTMM + geometric-mean fusion",
        "audio": audio_path.name,
        "duration_s": duration,
        "fixed_parameters": {
            "sr": SR,
            "hop": HOP,
            "scales_s": list(SCALES_S),
            "norm_percentile": NORM_PERCENTILE,
            "chain_tolerance_s": CHAIN_TOL_S,
            "chain_rule": "one-to-one adjacent-scale chain, support >= 2",
            "control_scale_s": CONTROL_SCALE_S,
            "min_event_gap_s": MIN_EVENT_GAP_S,
            "flux": {
                "lag": SUPERFLUX_LAG,
                "max_size": 1,
                "detrend": True,
            },
            "fusion": "sqrt(novelty_norm * no_max_flux_norm)",
        },
        "thresholds": {
            "nov_n2s": control_threshold,
            "fusion_n2s": fusion_n2s_threshold,
        },
        "variants": metrics,
        "matching_vs_nov_n2s": matches,
        "chains": {
            "wtmm_nov": wtmm_nov_chain,
            "wtmm_fusion": wtmm_fusion_chain,
        },
        "listening_legend_hz": {
            "common": 3000,
            "variant_only": 5000,
            "nov_n2s_only": 1500,
        },
    }
    metrics_path = destination / "wtmm_fusion_metrics.json"
    metrics_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n  비교 범례: 공통=3kHz, 변형전용=5kHz, nov_n2s전용=1.5kHz")
    for path in output_paths:
        print(f"  {path.name}")
    print(f"  {metrics_path.name}")


if __name__ == "__main__":
    main()
