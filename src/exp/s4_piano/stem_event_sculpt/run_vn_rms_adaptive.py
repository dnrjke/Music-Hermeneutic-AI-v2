"""Viyella's Nightmare: RMS/SF adaptive click variants (Dir experiment port).

Does NOT overwrite existing out/sonify/VN/전체_*.wav (gate/norm/Q1/adaptive…).
Writes new filenames + manifest alongside.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[2]
ROOT = HERE.parents[3]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_io import duration_s, load_mono  # noqa: E402
from config import MIN_EVENT_GAP_S, OUT_DIR, SR, audio_paths  # noqa: E402
from onset import band_envelopes, superflux_envelope  # noqa: E402
from peak_pick import peaks, peaks_adaptive  # noqa: E402

from run_lpc_o12_rms_adaptive import (  # noqa: E402
    CLICK_PARAMS,
    _click,
    _overlay,
    _one_to_one_counts,
    peaks_blockgate_norm_only,
    rms_envelope_and_bands,
)

DEST = OUT_DIR / "sonify" / "VN"
OUT_NAMES = {
    "rms_plain": "전체_rms_plain_클릭.wav",
    "rms_adaptive_noq1": "전체_rms_adaptive_noq1_클릭.wav",
    "rms_adaptive": "전체_rms_adaptive_클릭.wav",
    "sf_adaptive": "전체_sf_adaptive_재현_클릭.wav",
}
PROTECTED = {
    "전체_1차탐지_클릭.wav",
    "전체_adaptive_클릭.wav",
    "전체_gate+norm잔여_클릭.wav",
    "전체_gate+Q1_클릭.wav",
    "전체_gate_클릭.wav",
    "전체_norm+Q1_클릭.wav",
    "전체_norm+SIR_클릭.wav",
    "전체_norm_클릭.wav",
    "전체_Q1_클릭.wav",
    "전체_SIR_클릭.wav",
}


def _find_vn() -> Path:
    for path in audio_paths():
        if "Viyella" in path.name and "Nightmare" in path.name:
            return path
    raise FileNotFoundError("07.Viyella's Nightmare not found in audio_paths()")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_click_wav(
    path: Path, mono: np.ndarray, peaks_t: np.ndarray
) -> dict[str, Any]:
    if path.name in PROTECTED:
        raise RuntimeError(f"refusing to overwrite protected file: {path.name}")
    out = _overlay(mono, peaks_t, _click())
    out = np.clip(out, -1.0, 1.0).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), out, SR)
    rel = str(path.relative_to(ROOT)).replace("\\", "/")
    return {
        "path": rel,
        "sha256": _sha256(path),
        "frames": int(out.shape[0]),
        "duration_s": float(out.shape[0] / SR),
        "peak": float(np.max(np.abs(out))),
        "rms": float(np.sqrt(np.mean(out.astype(np.float64) ** 2))),
        "n_peaks": int(len(peaks_t)),
    }


def run_once() -> dict[str, Any]:
    src = _find_vn()
    for name in OUT_NAMES.values():
        if name in PROTECTED:
            raise RuntimeError(f"output name collides with protected: {name}")

    # Snapshot protected SHAs before write (prove untouched after)
    protected_before = {
        name: _sha256(DEST / name) if (DEST / name).exists() else None
        for name in sorted(PROTECTED)
    }

    mono = load_mono(src)
    dur = duration_s(src)
    DEST.mkdir(parents=True, exist_ok=True)

    env_rms, times_rms, bands_rms = rms_envelope_and_bands(mono)
    pk_plain = peaks(env_rms, times_rms, min_gap_s=CLICK_PARAMS["min_gap_s"])
    pk_noq1 = peaks_blockgate_norm_only(
        env_rms,
        times_rms,
        block_s=CLICK_PARAMS["block_s"],
        min_gap_s=CLICK_PARAMS["min_gap_s"],
    )
    pk_rms_ad = peaks_adaptive(
        env_rms,
        times_rms,
        bands_rms,
        dur,
        block_s=CLICK_PARAMS["block_s"],
        min_gap_s=CLICK_PARAMS["min_gap_s"],
    )

    env_sf, times_sf = superflux_envelope(mono)
    bands_sf = band_envelopes(mono)
    pk_sf = peaks_adaptive(
        env_sf,
        times_sf,
        bands_sf,
        dur,
        block_s=CLICK_PARAMS["block_s"],
        min_gap_s=CLICK_PARAMS["min_gap_s"],
    )

    variants = {
        "rms_plain": pk_plain,
        "rms_adaptive_noq1": pk_noq1,
        "rms_adaptive": pk_rms_ad,
        "sf_adaptive": pk_sf,
    }

    files: dict[str, Any] = {}
    peak_sets: dict[str, list[float]] = {}
    for key, pk in variants.items():
        out_name = OUT_NAMES[key]
        entry = _write_click_wav(DEST / out_name, mono, pk)
        entry["variant"] = key
        files[out_name] = entry
        peak_sets[key] = [float(t) for t in pk]
        print(f"  {out_name}: peaks={len(pk)}")

    protected_after = {
        name: _sha256(DEST / name) if (DEST / name).exists() else None
        for name in sorted(PROTECTED)
    }
    if protected_before != protected_after:
        raise RuntimeError(
            f"protected file SHA changed: {protected_before} vs {protected_after}"
        )

    tol = CLICK_PARAMS["match_tol_s"]
    vs = {
        "rms_adaptive_vs_sf_adaptive": _one_to_one_counts(pk_sf, pk_rms_ad, tol),
        "rms_adaptive_vs_rms_plain": _one_to_one_counts(pk_plain, pk_rms_ad, tol),
        "rms_adaptive_vs_rms_noq1": _one_to_one_counts(pk_noq1, pk_rms_ad, tol),
        "rms_noq1_vs_rms_plain": _one_to_one_counts(pk_plain, pk_noq1, tol),
    }

    legacy = DEST / "전체_adaptive_클릭.wav"
    manifest = {
        "experiment": "vn_rms_adaptive_sonify",
        "track": src.name,
        "source": str(src).replace("\\", "/"),
        "note": (
            "Port of Dir RMS-vs-SuperFlux adaptive experiment to VN full mix. "
            "Existing VN 전체_* clicks preserved; new filenames only."
        ),
        "protected_untouched": sorted(PROTECTED),
        "protected_sha256": protected_after,
        "legacy_adaptive": {
            "path": "out/sonify/VN/전체_adaptive_클릭.wav",
            "exists": legacy.exists(),
            "sha256": _sha256(legacy) if legacy.exists() else None,
        },
        "fixed_rules": {
            "input": "load_mono (LUFS) full track",
            "variants": list(OUT_NAMES.keys()),
            "out_names": OUT_NAMES,
            "click": CLICK_PARAMS,
            "min_gap_s": MIN_EVENT_GAP_S,
        },
        "duration_s": float(dur),
        "peak_counts": {k: int(len(v)) for k, v in variants.items()},
        "vs_30ms": vs,
        "peak_times_s": peak_sets,
        "files": files,
    }
    (DEST / "vn_rms_adaptive_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def determinism_check(manifest: dict[str, Any]) -> dict[str, Any]:
    first = {n: e["sha256"] for n, e in manifest["files"].items()}
    first_peaks = {k: list(v) for k, v in manifest["peak_times_s"].items()}
    second = run_once()
    mismatches = [
        n for n in first if first[n] != second["files"].get(n, {}).get("sha256")
    ]
    peak_mismatch = [
        k for k in first_peaks if first_peaks[k] != second["peak_times_s"].get(k)
    ]
    report = {
        "matched": len(mismatches) == 0 and len(peak_mismatch) == 0,
        "wav_mismatches": mismatches,
        "peak_mismatches": peak_mismatch,
    }
    (DEST / "vn_rms_adaptive_determinism.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--determinism-check", action="store_true")
    args = parser.parse_args()

    src = _find_vn()
    print(f"source: {src}")
    print(f"output: {DEST}")
    print(f"new files only: {list(OUT_NAMES.values())}")

    manifest = run_once()
    print("vs_30ms:")
    for k, v in manifest["vs_30ms"].items():
        print(f"  {k}: {v}")
    print("protected SHA unchanged: OK")

    if args.determinism_check:
        print("determinism-check: second run…")
        report = determinism_check(manifest)
        if not report["matched"]:
            raise SystemExit(f"determinism failed: {report}")
        print("determinism-check: OK")


if __name__ == "__main__":
    main()
