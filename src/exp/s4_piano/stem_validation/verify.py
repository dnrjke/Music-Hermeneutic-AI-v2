"""Verify canonical stems and create piano/residual listening pairs."""
from __future__ import annotations

import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import soundfile as sf


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "audio" / "102 - Dir.wav"
STEM_ROOT = ROOT / "out" / "stems" / "Dir"
SONIFY_ROOT = ROOT / "out" / "sonify" / "Dir"
CONSENSUS_METRICS = SONIFY_ROOT / "stem_consensus_metrics.json"
MODELS = ("bs_roformer", "spleeter", "demucs")
SR = 44_100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float64)
    return audio.mean(axis=1, dtype=np.float64)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def rms_envelope(audio: np.ndarray, frame: int = 2048, hop: int = 512) -> np.ndarray:
    signal = mono(audio)
    if len(signal) < frame:
        return np.asarray([np.sqrt(np.mean(signal**2))])
    count = 1 + (len(signal) - frame) // hop
    shape = (count, frame)
    strides = (signal.strides[0] * hop, signal.strides[0])
    windows = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)
    return np.sqrt(np.mean(windows**2, axis=1))


def load(path: Path) -> tuple[np.ndarray, int]:
    return sf.read(path, dtype="float32", always_2d=True)


def one_to_one_count(left: np.ndarray, right: np.ndarray, tol: float = 0.03) -> int:
    candidates: list[tuple[float, int, int]] = []
    for left_i, left_time in enumerate(left):
        lo = int(np.searchsorted(right, left_time - tol, side="left"))
        hi = int(np.searchsorted(right, left_time + tol, side="right"))
        candidates.extend(
            (abs(left_time - right[right_i]), left_i, right_i)
            for right_i in range(lo, hi)
        )
    used_left: set[int] = set()
    used_right: set[int] = set()
    for _, left_i, right_i in sorted(candidates):
        if left_i not in used_left and right_i not in used_right:
            used_left.add(left_i)
            used_right.add(right_i)
    return len(used_left)


def audio_report(path: Path) -> dict[str, object]:
    info = sf.info(path)
    audio, _ = load(path)
    peak = float(np.max(np.abs(audio)))
    return {
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "duration_s": info.duration,
        "subtype": info.subtype,
        "peak": peak,
        "samples_at_full_scale": int(np.sum(np.abs(audio) >= 0.999969)),
        "samples_outside_unit_range": int(np.sum(np.abs(audio) > 1.0)),
        "sha256": sha256(path),
    }


def main() -> None:
    original, original_rate = load(SOURCE)
    if original_rate != SR:
        raise RuntimeError(f"원본 sample rate: {original_rate}")
    consensus = json.loads(CONSENSUS_METRICS.read_text(encoding="utf-8"))

    stem_reports: dict[str, object] = {}
    piano_audio: dict[str, np.ndarray] = {}
    model_peaks: dict[str, np.ndarray] = {}
    listening_paths: list[Path] = []
    for model in MODELS:
        model_dir = STEM_ROOT / model
        piano, piano_rate = load(model_dir / "piano.wav")
        residual, residual_rate = load(model_dir / "residual.wav")
        if piano_rate != SR or residual_rate != SR:
            raise RuntimeError(f"{model}: canonical sample rate 불일치")
        if piano.shape != original.shape or residual.shape != original.shape:
            raise RuntimeError(f"{model}: canonical shape 불일치")
        piano_audio[model] = piano
        model_peaks[model] = np.asarray(
            consensus["models"][model]["peak_times_s"],
            dtype=np.float64,
        )

        stem_paths = sorted(
            path
            for path in model_dir.glob("*.wav")
            if path.name != "residual.wav"
        )
        stem_sum = np.zeros_like(original, dtype=np.float64)
        for path in stem_paths:
            stem, _ = load(path)
            stem_sum += stem
        reconstruction_error = stem_sum - original
        original_rms = float(np.sqrt(np.mean(original.astype(np.float64) ** 2)))

        piano_mono = mono(piano)
        residual_mono = mono(residual)
        piano_scale = max(1.0, float(np.max(np.abs(piano_mono))) / 0.8)
        residual_scale = max(1.0, float(np.max(np.abs(residual_mono))) / 0.8)
        stereo = np.column_stack(
            [piano_mono / piano_scale, residual_mono / residual_scale]
        ).astype(np.float32)
        listening_path = (
            SONIFY_ROOT / f"전체_stem_{model}_piano_L_residual_R.wav"
        )
        sf.write(listening_path, stereo, SR)
        listening_paths.append(listening_path)

        stem_reports[model] = {
            "piano": audio_report(model_dir / "piano.wav"),
            "residual": audio_report(model_dir / "residual.wav"),
            "stem_sum_reconstruction": {
                "stems": [path.stem for path in stem_paths],
                "error_rms": float(np.sqrt(np.mean(reconstruction_error**2))),
                "error_to_original_rms_pct": (
                    100.0
                    * float(np.sqrt(np.mean(reconstruction_error**2)))
                    / original_rms
                ),
                "waveform_correlation": correlation(stem_sum.ravel(), original.ravel()),
            },
            "piano_to_original": {
                "waveform_correlation": correlation(piano_mono, mono(original)),
                "rms_envelope_correlation": correlation(
                    rms_envelope(piano),
                    rms_envelope(original),
                ),
            },
        }

    pairwise: dict[str, object] = {}
    for left, right in combinations(MODELS, 2):
        left_audio = piano_audio[left]
        right_audio = piano_audio[right]
        pairwise[f"{left}_vs_{right}"] = {
            "piano_waveform_correlation": correlation(
                mono(left_audio),
                mono(right_audio),
            ),
            "piano_rms_envelope_correlation": correlation(
                rms_envelope(left_audio),
                rms_envelope(right_audio),
            ),
            "event_common_30ms": one_to_one_count(
                model_peaks[left],
                model_peaks[right],
            ),
            "left_events": int(len(model_peaks[left])),
            "right_events": int(len(model_peaks[right])),
        }

    sonify_paths = sorted(SONIFY_ROOT.glob("전체_stem_*.wav"))
    sonify_reports = {
        path.name: audio_report(path)
        for path in sonify_paths
    }
    invalid = [
        name
        for name, report in sonify_reports.items()
        if report["sample_rate"] != SR
        or abs(report["duration_s"] - len(original) / SR) > 1e-6
        or report["samples_at_full_scale"] > 0
    ]
    metrics = {
        "experiment": "s4 piano stem validation checks",
        "original": audio_report(SOURCE),
        "models": stem_reports,
        "pairwise_piano": pairwise,
        "sonifications": sonify_reports,
        "invalid_sonifications": invalid,
        "interpretation_guard": (
            "correlation and model agreement measure consistency, not ground-truth "
            "piano isolation; listening must judge music-box leakage and artifacts"
        ),
    }
    metrics_path = STEM_ROOT / "stem_validation_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("  pairwise piano")
    for label, report in pairwise.items():
        print(
            f"    {label}: waveform={report['piano_waveform_correlation']:.3f}, "
            f"rms-env={report['piano_rms_envelope_correlation']:.3f}, "
            f"events={report['event_common_30ms']}"
        )
    print(f"  invalid sonifications: {invalid}")
    for path in listening_paths:
        print(f"  {path.name}")
    print(f"  {metrics_path}")


if __name__ == "__main__":
    main()
