"""Sonify transcription agreement and candidate-395 event roles."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "exp" / "s4_piano"))

from config import SR  # noqa: E402
from exp.s4_piano._onset_wtmm_fusion import click  # noqa: E402


TRANSCRIPTION_DIR = ROOT / "out" / "transcription" / "Dir"
SONIFY_DIR = ROOT / "out" / "sonify" / "Dir" / "transcription"
EVALUATION = TRANSCRIPTION_DIR / "transcription_evaluation.json"
ORIGINAL = ROOT / "audio" / "102 - Dir.wav"
BS_PIANO = ROOT / "out" / "stems" / "Dir" / "bs_roformer" / "piano.wav"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mono(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != SR:
        raise RuntimeError(f"{path}: sample rate {sample_rate} != {SR}")
    return audio.mean(axis=1, dtype=np.float32)


def save_overlay(
    path: Path,
    base: np.ndarray,
    groups: list[tuple[np.ndarray, float]],
) -> dict[str, float | int | str]:
    output = base.copy()
    for times, frequency in groups:
        click_sound = click(frequency)
        for event_time in times:
            start = int(float(event_time) * SR)
            stop = min(start + len(click_sound), len(output))
            if 0 <= start < stop:
                output[start:stop] += click_sound[: stop - start]
    pre_scale_peak = float(np.max(np.abs(output)))
    if pre_scale_peak > 0.98:
        output *= 0.98 / pre_scale_peak
    sf.write(path, output, SR, subtype="PCM_16")
    return {
        "sample_rate": SR,
        "frames": len(output),
        "channels": 1,
        "pre_scale_peak": pre_scale_peak,
        "written_peak": float(np.max(np.abs(output))),
        "sha256": sha256(path),
    }


def as_times(roles: dict[str, list[float]], name: str) -> np.ndarray:
    return np.asarray(roles[name], dtype=np.float64)


def main() -> None:
    evaluation = json.loads(EVALUATION.read_text(encoding="utf-8"))
    roles = evaluation["event_roles"]
    SONIFY_DIR.mkdir(parents=True, exist_ok=True)
    bases = {
        "전체": mono(ORIGINAL),
        "BS피아노": mono(BS_PIANO),
    }
    recipes = {
        "transkun전사": {
            "description": "Transkun 전체 cluster: 3kHz",
            "groups": [(as_times(roles, "transkun_all"), 3000.0)],
        },
        "transkun_vs_basicpitch_역할": {
            "description": (
                "공통 3kHz, Transkun-only 5kHz, Basic-Pitch-only 1.5kHz"
            ),
            "groups": [
                (as_times(roles, "transkun_basic_common"), 3000.0),
                (as_times(roles, "transkun_only"), 5000.0),
                (as_times(roles, "basic_pitch_only"), 1500.0),
            ],
        },
        "transkun_vs_395_역할": {
            "description": (
                "공통 3kHz, Transkun-only 5kHz, 395-only 1.5kHz"
            ),
            "groups": [
                (as_times(roles, "transkun_395_common"), 3000.0),
                (as_times(roles, "transkun_only_vs_395"), 5000.0),
                (as_times(roles, "candidate395_only"), 1500.0),
            ],
        },
        "두전사공통_395누락": {
            "description": (
                "Transkun과 Basic Pitch 공통이지만 395가 놓친 사건: 5kHz"
            ),
            "groups": [
                (
                    as_times(
                        roles,
                        "both_transcriptions_common_missed_by_395",
                    ),
                    5000.0,
                )
            ],
        },
    }

    outputs: dict[str, dict[str, object]] = {}
    for base_name, base_audio in bases.items():
        for recipe_name, recipe in recipes.items():
            filename = f"{base_name}_{recipe_name}_클릭.wav"
            path = SONIFY_DIR / filename
            audio_report = save_overlay(
                path,
                base_audio,
                recipe["groups"],
            )
            outputs[filename] = {
                "base": base_name,
                "description": recipe["description"],
                "event_counts": [
                    len(group[0]) for group in recipe["groups"]
                ],
                **audio_report,
            }
            print(f"✓ {filename}")

    manifest = {
        "experiment": "BS stem transcription role sonification",
        "legend_hz": {
            "common": 3000,
            "new_reference_only": 5000,
            "existing_or_audit_only": 1500,
        },
        "outputs": outputs,
    }
    (SONIFY_DIR / "sonification_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
