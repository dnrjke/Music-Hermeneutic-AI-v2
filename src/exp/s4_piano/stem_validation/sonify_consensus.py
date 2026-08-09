"""Sonify three-model piano-stem support without treating stems as truth."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa
import numpy as np
import soundfile as sf

from audio_io import duration_s, load_mono
from config import HOP, OUT_DIR, SR
from _onset_complex_hysteresis import complex_domain_odf
from _onset_posdist import positive_distribution_novelty
from _onset_sliding_norm import sliding_normalize
from _onset_source_carving import one_to_one_pair_masks
from _onset_wtmm_fusion import (
    click,
    cosine_novelty,
    get_logmel,
    no_max_flux,
    one_to_one_time_match,
    periodicity_metrics,
    response_peaks,
    scale_normalize,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "audio" / "102 - Dir.wav"
STEM_ROOT = ROOT / "out" / "stems" / "Dir"
SONIFY_DIR = OUT_DIR / "sonify" / "Dir"
POSDIST_METRICS = SONIFY_DIR / "posdist_metrics.json"
MODELS = ("bs_roformer", "spleeter", "demucs")
MATCH_TOLERANCE_S = 0.03


def a2_positive_rescue(audio_path: Path) -> tuple[np.ndarray, dict[str, int]]:
    mono = load_mono(audio_path)
    logmel = get_logmel(mono)
    times = librosa.frames_to_time(
        np.arange(logmel.shape[1]),
        sr=SR,
        hop_length=HOP,
    )
    cosine = cosine_novelty(logmel)
    positive, _ = positive_distribution_novelty(logmel)
    flux = no_max_flux(logmel)
    cosine_norm, _ = sliding_normalize(cosine, times)
    positive_norm, _ = sliding_normalize(positive, times)
    flux_norm, _ = sliding_normalize(flux, times)

    a2, _ = response_peaks(np.sqrt(cosine_norm * flux_norm), times)
    positive_flux, _ = response_peaks(
        np.sqrt(positive_norm * flux_norm),
        times,
    )
    _, _, positive_only = one_to_one_time_match(a2, positive_flux)
    positive_only = np.asarray(
        [
            time
            for time in positive_only
            if len(a2) == 0
            or float(np.min(np.abs(a2 - time))) > MATCH_TOLERANCE_S
        ],
        dtype=np.float64,
    )
    rescue = np.sort(np.concatenate([a2, positive_only]))
    if len(rescue) > 1 and np.min(np.diff(rescue)) < MATCH_TOLERANCE_S:
        raise RuntimeError(f"{audio_path.name}: rescue 피크 30ms 충돌")
    return rescue, {
        "a2": int(len(a2)),
        "positive_flux": int(len(positive_flux)),
        "positive_only": int(len(positive_only)),
        "rescue": int(len(rescue)),
    }


def support_mask(reference: np.ndarray, evidence: np.ndarray) -> np.ndarray:
    left_mask, _ = one_to_one_pair_masks(
        reference,
        evidence,
        tolerance_s=MATCH_TOLERANCE_S,
    )
    return left_mask


def consensus_clusters(
    model_peaks: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    events = sorted(
        (float(time), model, index)
        for model, peaks in model_peaks.items()
        for index, time in enumerate(peaks)
    )
    consumed: set[tuple[str, int]] = set()
    consensus_times: list[float] = []
    consensus_votes: list[int] = []
    for seed_time, seed_model, seed_index in events:
        seed_key = (seed_model, seed_index)
        if seed_key in consumed:
            continue
        group = [(seed_time, seed_model, seed_index)]
        for model in MODELS:
            if model == seed_model:
                continue
            candidates = [
                (abs(time - seed_time), time, index)
                for time, event_model, index in events
                if event_model == model
                and (model, index) not in consumed
                and abs(time - seed_time) <= MATCH_TOLERANCE_S
            ]
            if candidates:
                _, time, index = min(candidates)
                group.append((time, model, index))
        for _, model, index in group:
            consumed.add((model, index))
        if len(group) >= 2:
            consensus_times.append(float(np.mean([item[0] for item in group])))
            consensus_votes.append(len(group))
    order = np.argsort(consensus_times)
    return (
        np.asarray(consensus_times, dtype=np.float64)[order],
        np.asarray(consensus_votes, dtype=np.int64)[order],
    )


def original_candidates() -> dict[str, np.ndarray]:
    mono = load_mono(SOURCE)
    logmel = get_logmel(mono)
    times = librosa.frames_to_time(
        np.arange(logmel.shape[1]),
        sr=SR,
        hop_length=HOP,
    )
    cosine = cosine_novelty(logmel)
    flux = no_max_flux(logmel)
    complex_odf = complex_domain_odf(mono, len(times))

    cosine_block = scale_normalize(cosine, times, 2.0)
    flux_block = scale_normalize(flux, times, 2.0)
    cosine_slide, _ = sliding_normalize(cosine, times)
    flux_slide, _ = sliding_normalize(flux, times)
    complex_slide, _ = sliding_normalize(complex_odf, times)

    responses = {
        "nov_n2s": cosine_block,
        "fusion_n2s": np.sqrt(cosine_block * flux_block),
        "a2_fusion_slide": np.sqrt(cosine_slide * flux_slide),
        "tri_complex_slide": np.cbrt(
            cosine_slide * flux_slide * complex_slide
        ),
    }
    candidates = {
        name: response_peaks(response, times)[0]
        for name, response in responses.items()
    }
    posdist = json.loads(POSDIST_METRICS.read_text(encoding="utf-8"))
    candidates["a2_posdist_rescue"] = np.asarray(
        posdist["peak_times_s"]["a2_posdist_rescue"],
        dtype=np.float64,
    )
    expected = {
        "nov_n2s": 613,
        "fusion_n2s": 351,
        "a2_fusion_slide": 355,
        "tri_complex_slide": 306,
        "a2_posdist_rescue": 395,
    }
    for name, count in expected.items():
        if len(candidates[name]) != count:
            raise RuntimeError(
                f"기존 후보 재현 실패 {name}: {len(candidates[name])} != {count}"
            )
    return candidates


def mono_without_normalization(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != SR:
        raise RuntimeError(f"{path}: sample rate {sample_rate}")
    return audio.mean(axis=1, dtype=np.float32)


def save_overlay(
    path: Path,
    base: np.ndarray,
    groups: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    output = base.copy()
    for peak_times, click_sound in groups:
        for peak_time in peak_times:
            start = int(peak_time * SR)
            stop = min(start + len(click_sound), len(output))
            if stop > start:
                output[start:stop] += click_sound[: stop - start]
    peak = float(np.max(np.abs(output)))
    if peak > 0.98:
        output *= 0.98 / peak
    sf.write(path, output, SR)


def main() -> None:
    if not POSDIST_METRICS.exists():
        raise FileNotFoundError(
            "peak_times_s가 포함된 posdist_metrics.json을 먼저 생성해야 합니다"
        )
    SONIFY_DIR.mkdir(parents=True, exist_ok=True)
    duration = duration_s(SOURCE)
    original = load_mono(SOURCE)
    candidates = original_candidates()
    candidate395 = candidates["a2_posdist_rescue"]

    model_peaks: dict[str, np.ndarray] = {}
    model_reports: dict[str, object] = {}
    output_paths: list[Path] = []
    standard_click = click(3000.0)
    for model in MODELS:
        piano_path = STEM_ROOT / model / "piano.wav"
        residual_path = STEM_ROOT / model / "residual.wav"
        if not piano_path.exists() or not residual_path.exists():
            raise FileNotFoundError(f"{model} canonical stem 누락")
        peaks, components = a2_positive_rescue(piano_path)
        model_peaks[model] = peaks
        model_reports[model] = {
            **components,
            "periodicity": periodicity_metrics(peaks, duration),
            "peak_times_s": [float(time) for time in peaks],
        }

        piano = mono_without_normalization(piano_path)
        piano_scale = max(1.0, float(np.max(np.abs(piano))) / 0.8)
        piano = piano / piano_scale
        path = SONIFY_DIR / f"전체_stem_{model}_piano_candidate395_클릭.wav"
        save_overlay(path, piano, [(candidate395, standard_click)])
        output_paths.append(path)

    support_matrix = np.vstack(
        [support_mask(candidate395, model_peaks[model]) for model in MODELS]
    )
    support_count = support_matrix.sum(axis=0)
    support_categories = {
        "3of3": candidate395[support_count == 3],
        "2of3": candidate395[support_count == 2],
        "0to1of3": candidate395[support_count <= 1],
    }
    support_clicks = {
        "3of3": click(3000.0),
        "2of3": click(5000.0, 15.0, 0.8),
        "0to1of3": click(1500.0, 15.0, 0.8),
    }
    support_path = SONIFY_DIR / "전체_stem_support395_비교_클릭.wav"
    save_overlay(
        support_path,
        original,
        [
            (support_categories["3of3"], support_clicks["3of3"]),
            (support_categories["2of3"], support_clicks["2of3"]),
            (support_categories["0to1of3"], support_clicks["0to1of3"]),
        ],
    )
    output_paths.append(support_path)
    for category, times in support_categories.items():
        path = SONIFY_DIR / f"전체_stem_support395_{category}_클릭.wav"
        save_overlay(path, original, [(times, standard_click)])
        output_paths.append(path)

    bs_common, bs_only, candidate_only = one_to_one_time_match(
        model_peaks["bs_roformer"],
        candidate395,
    )
    bs_role_times = {
        "common": bs_common,
        "bs_only": bs_only,
        "candidate_only": candidate_only,
    }
    bs_compare_path = SONIFY_DIR / "전체_bs_reference_vs_candidate395_비교_클릭.wav"
    save_overlay(
        bs_compare_path,
        original,
        [
            (bs_common, click(3000.0)),
            (bs_only, click(5000.0, 15.0, 0.8)),
            (candidate_only, click(1500.0, 15.0, 0.8)),
        ],
    )
    output_paths.append(bs_compare_path)
    bs_piano = mono_without_normalization(
        STEM_ROOT / "bs_roformer" / "piano.wav"
    )
    bs_piano /= max(1.0, float(np.max(np.abs(bs_piano))) / 0.8)
    bs_piano_compare_path = (
        SONIFY_DIR / "전체_bs_piano_reference_vs_candidate395_비교_클릭.wav"
    )
    save_overlay(
        bs_piano_compare_path,
        bs_piano,
        [
            (bs_common, click(3000.0)),
            (bs_only, click(5000.0, 15.0, 0.8)),
            (candidate_only, click(1500.0, 15.0, 0.8)),
        ],
    )
    output_paths.append(bs_piano_compare_path)
    for category, times in bs_role_times.items():
        path = SONIFY_DIR / f"전체_bs_reference_{category}_클릭.wav"
        save_overlay(path, original, [(times, standard_click)])
        output_paths.append(path)

    consensus, consensus_votes = consensus_clusters(model_peaks)
    consensus_supported = support_mask(consensus, candidate395)
    consensus_missed = consensus[~consensus_supported]
    consensus_path = SONIFY_DIR / "전체_stem_consensus_all_클릭.wav"
    save_overlay(consensus_path, original, [(consensus, standard_click)])
    output_paths.append(consensus_path)
    missed_path = SONIFY_DIR / "전체_stem_consensus_missed_클릭.wav"
    save_overlay(missed_path, original, [(consensus_missed, standard_click)])
    output_paths.append(missed_path)

    candidate_comparison: dict[str, object] = {}
    for name, candidate_times in candidates.items():
        common, consensus_only, candidate_only = one_to_one_time_match(
            consensus,
            candidate_times,
        )
        per_model = {
            model: int(np.sum(support_mask(candidate_times, model_peaks[model])))
            for model in MODELS
        }
        candidate_comparison[name] = {
            "peaks": int(len(candidate_times)),
            "consensus_common": int(len(common)),
            "consensus_missed": int(len(consensus_only)),
            "candidate_only": int(len(candidate_only)),
            "consensus_coverage_pct": (
                100.0 * len(common) / len(consensus) if len(consensus) else 0.0
            ),
            "candidate_support_pct": (
                100.0 * len(common) / len(candidate_times)
                if len(candidate_times)
                else 0.0
            ),
            "per_model_supported": per_model,
        }

    metrics = {
        "experiment": "s4 three-model piano stem consensus",
        "diagnostic_only": True,
        "support_definition": (
            "same fixed A-2 + positive rescue detector on each piano stem; "
            "one-to-one ±30ms matching"
        ),
        "models": model_reports,
        "candidate395_support": {
            category: int(len(times))
            for category, times in support_categories.items()
        },
        "candidate395_exact_support": {
            str(votes): int(np.sum(support_count == votes))
            for votes in range(4)
        },
        "bs_primary_reference": {
            "reference_events": int(len(model_peaks["bs_roformer"])),
            "common": int(len(bs_common)),
            "bs_only": int(len(bs_only)),
            "candidate395_only": int(len(candidate_only)),
            "times_s": {
                category: [float(time) for time in times]
                for category, times in bs_role_times.items()
            },
            "warning": (
                "primary attribution reference after listening validation; "
                "not absolute onset ground truth"
            ),
        },
        "candidate395_support_times_s": {
            category: [float(time) for time in times]
            for category, times in support_categories.items()
        },
        "stem_consensus": {
            "events_2plus": int(len(consensus)),
            "events_3of3": int(np.sum(consensus_votes == 3)),
            "events_2of3": int(np.sum(consensus_votes == 2)),
            "matched_by_candidate395": int(np.sum(consensus_supported)),
            "missed_by_candidate395": int(len(consensus_missed)),
            "times_s": [float(time) for time in consensus],
            "votes": [int(vote) for vote in consensus_votes],
            "missed_times_s": [float(time) for time in consensus_missed],
        },
        "candidate_comparison": candidate_comparison,
        "listening_legend_hz": {
            "candidate_supported_3of3": 3000,
            "candidate_supported_2of3": 5000,
            "candidate_supported_0to1of3": 1500,
        },
        "warning": (
            "stem consensus is model-derived attribution evidence, not onset "
            "ground truth; listen for music-box leakage and separation artifacts"
        ),
    }
    metrics_path = SONIFY_DIR / "stem_consensus_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("  model rescue peaks")
    for model in MODELS:
        print(f"    {model}: {len(model_peaks[model])}")
    print(
        "  candidate395 support: "
        + ", ".join(
            f"{category}={len(times)}"
            for category, times in support_categories.items()
        )
    )
    print(
        f"  stem consensus={len(consensus)}, "
        f"candidate395 missed={len(consensus_missed)}"
    )
    for path in output_paths:
        print(f"  {path.name}")
    print(f"  {metrics_path.name}")


if __name__ == "__main__":
    main()
