"""Run fixed HPSS / LPC / sinusoidal passes on BS piano stem."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# Local imports (script directory on path)
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from io_util import (  # noqa: E402
    HOP,
    N_FFT,
    OUTPUT_DIR,
    SOURCE_PIANO,
    SR,
    audio_stats,
    read_stereo,
    sha256_file,
    write_json,
    write_listening_wav,
)
from passes_hpss import HPSS_PARAMS, hpss_components  # noqa: E402
from passes_lpc import LPC_PARAMS, lpc_components  # noqa: E402
from passes_sinusoidal import SINE_PARAMS, sinusoidal_components  # noqa: E402


def _file_entry(path: Path, role: str) -> dict[str, Any]:
    audio, sr = read_stereo(path)
    stats = audio_stats(audio)
    return {
        "role": role,
        "path": str(path.relative_to(HERE.parents[3])).replace("\\", "/"),
        "sha256": sha256_file(path),
        "sample_rate": sr,
        "frames": int(audio.shape[0]),
        "channels": int(audio.shape[1]),
        "duration_s": float(audio.shape[0] / sr),
        **stats,
    }


def run_once() -> dict[str, Any]:
    if not SOURCE_PIANO.exists():
        raise FileNotFoundError(SOURCE_PIANO)

    source_sha = sha256_file(SOURCE_PIANO)
    stereo, sr = read_stereo(SOURCE_PIANO)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    harmonic, percussive = hpss_components(stereo)
    t_hpss = time.perf_counter() - t0

    t0 = time.perf_counter()
    lpc_residual, lpc_synthesis = lpc_components(stereo)
    t_lpc = time.perf_counter() - t0

    t0 = time.perf_counter()
    sine_tonal, sine_residual = sinusoidal_components(stereo)
    t_sine = time.perf_counter() - t0

    outputs = {
        "hpss_percussive.wav": (percussive, "hpss_event_candidate"),
        "hpss_harmonic.wav": (harmonic, "hpss_complement"),
        "lpc_residual.wav": (lpc_residual, "lpc_event_candidate"),
        "lpc_synthesis.wav": (lpc_synthesis, "lpc_complement"),
        "sine_residual.wav": (sine_residual, "sine_event_candidate"),
        "sine_tonal.wav": (sine_tonal, "sine_complement"),
    }

    files: dict[str, Any] = {
        "source_piano": {
            "role": "input_reference",
            "path": str(SOURCE_PIANO.relative_to(HERE.parents[3])).replace("\\", "/"),
            "sha256": source_sha,
            "sample_rate": sr,
            "frames": int(stereo.shape[0]),
            "channels": int(stereo.shape[1]),
            "duration_s": float(stereo.shape[0] / sr),
            **audio_stats(stereo),
        }
    }

    for name, (audio, role) in outputs.items():
        if audio.shape != stereo.shape:
            raise RuntimeError(f"{name}: shape {audio.shape} != source {stereo.shape}")
        path = OUTPUT_DIR / name
        files[name] = {
            **write_listening_wav(path, audio, sr),
            "role": role,
        }

    manifest = {
        "experiment": "stem_event_sculpt_3pass",
        "fixed_rules": {
            "input": "out/stems/Dir/bs_roformer/piano.wav only",
            "no_odf_mask": True,
            "no_395_compare_sonify": True,
            "no_lr_stem_stereo_sonify": True,
            "full_length": True,
            "soft_limit_listen_peak": 0.98,
            "stft": {"n_fft": N_FFT, "hop_length": HOP, "sr": SR},
        },
        "parameters": {
            "hpss": HPSS_PARAMS,
            "lpc": LPC_PARAMS,
            "sinusoidal": SINE_PARAMS,
        },
        "runtime_s": {
            "hpss": t_hpss,
            "lpc": t_lpc,
            "sinusoidal": t_sine,
        },
        "files": files,
    }
    write_json(OUTPUT_DIR / "sculpt_manifest.json", manifest)
    return manifest


def determinism_check(manifest: dict[str, Any]) -> dict[str, Any]:
    first_hashes = {
        name: entry["sha256"]
        for name, entry in manifest["files"].items()
        if name.endswith(".wav")
    }
    second = run_once()
    second_hashes = {
        name: entry["sha256"]
        for name, entry in second["files"].items()
        if name.endswith(".wav")
    }
    mismatches = [
        name
        for name in first_hashes
        if first_hashes[name] != second_hashes.get(name)
    ]
    report = {
        "matched": len(mismatches) == 0,
        "compared_files": sorted(first_hashes),
        "mismatches": mismatches,
    }
    write_json(OUTPUT_DIR / "sculpt_determinism.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--determinism-check",
        action="store_true",
        help="Run twice and compare output WAV SHA-256",
    )
    args = parser.parse_args()

    print(f"input: {SOURCE_PIANO}")
    print(f"output: {OUTPUT_DIR}")
    manifest = run_once()
    for name, entry in manifest["files"].items():
        if name == "source_piano":
            print(
                f"  source peak={entry['peak']:.4f} rms={entry['rms']:.4f} "
                f"dur={entry['duration_s']:.3f}s"
            )
        else:
            print(
                f"  {name}: role={entry['role']} peak={entry['peak']:.4f} "
                f"rms={entry['rms']:.4f} sha={entry['sha256'][:12]}"
            )
    print(
        "runtime_s:",
        {k: round(v, 2) for k, v in manifest["runtime_s"].items()},
    )

    if args.determinism_check:
        print("determinism-check: second run…")
        report = determinism_check(manifest)
        if not report["matched"]:
            raise SystemExit(f"determinism failed: {report['mismatches']}")
        print("determinism-check: OK")


if __name__ == "__main__":
    main()
