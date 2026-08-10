"""LPC o24/o36 residual × SuperFlux peaks_adaptive clicks.

Same detector as lpc_o12_residual adaptive / 전체_adaptive_클릭.
Filenames per request: lpc_o{24,36}_residual_sf_adaptive_클릭.wav
Does not modify o12 refine outputs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import MIN_EVENT_GAP_S, SR  # noqa: E402
from onset import band_envelopes, superflux_envelope  # noqa: E402
from peak_pick import peaks_adaptive  # noqa: E402

from io_util import (  # noqa: E402
    N_FFT,
    HOP,
    OUTPUT_DIR,
    audio_stats,
    click_wav_name,
    read_stereo,
    sha256_file,
    write_json,
    write_listening_wav,
)

PASS2_DIR = OUTPUT_DIR / "pass2"
OUT_DIR = PASS2_DIR / "lpc_sf_adaptive"

SOURCES = (
    (PASS2_DIR / "lpc_o24_residual.wav", "o24", "lpc_o24_residual_sf_adaptive"),
    (PASS2_DIR / "lpc_o36_residual.wav", "o36", "lpc_o36_residual_sf_adaptive"),
)

CLICK_PARAMS = {
    "rms_win": N_FFT,
    "rms_hop": HOP,
    "click_freq_hz": 3000.0,
    "click_dur_ms": 12.0,
    "click_amp": 0.7,
    "min_gap_s": MIN_EVENT_GAP_S,
}


def _click(sr: int = SR) -> np.ndarray:
    n = int(sr * CLICK_PARAMS["click_dur_ms"] / 1000.0)
    t = np.arange(n, dtype=np.float32) / sr
    env = np.exp(-t * 1000.0 / CLICK_PARAMS["click_dur_ms"])
    return (
        CLICK_PARAMS["click_amp"]
        * env
        * np.sin(2 * np.pi * CLICK_PARAMS["click_freq_hz"] * t)
    ).astype(np.float32)


def _overlay(mono: np.ndarray, peak_times: np.ndarray, click: np.ndarray) -> np.ndarray:
    out = mono.astype(np.float32, copy=True)
    for t in peak_times:
        idx = int(t * SR)
        end = min(idx + len(click), len(out))
        n = end - idx
        if n > 0:
            out[idx:end] += click[:n]
    return out


def run_once() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    click = _click()
    files: dict[str, Any] = {}
    peak_sets: dict[str, list[float]] = {}
    peak_meta: dict[str, Any] = {}

    for src, key, stem in SOURCES:
        if not src.exists():
            raise FileNotFoundError(src)
        stereo, sr = read_stereo(src)
        if sr != SR:
            raise RuntimeError(f"{src.name}: sr={sr}")
        mono = stereo.mean(axis=1).astype(np.float32)
        dur = float(len(mono) / SR)
        env, times = superflux_envelope(mono)
        bands = band_envelopes(mono)
        pk = peaks_adaptive(
            env, times, bands, dur, min_gap_s=CLICK_PARAMS["min_gap_s"]
        )
        out_name = click_wav_name(stem, len(pk))
        peak_sets[key] = [float(t) for t in pk]
        peak_meta[key] = {
            "n_peaks": int(len(pk)),
            "method": "superflux_peaks_adaptive",
            "env_max": float(np.nanmax(env)) if env.size else 0.0,
            "source": src.name,
            "source_sha256": sha256_file(src),
            "source_stats": audio_stats(stereo),
            "filename_stem": stem,
        }
        overlaid = _overlay(mono, pk, click)
        entry = write_listening_wav(
            OUT_DIR / out_name, overlaid, SR, limit_mode="clip"
        )
        files[out_name] = {
            **entry,
            "role": "lpc_residual_sf_adaptive_click",
            "order_key": key,
            "n_peaks": int(len(pk)),
            "peak_meta": peak_meta[key],
        }
        print(f"  {out_name}: peaks={len(pk)}")

    # Optional cross-order ±30ms vs o12 adaptive if present
    vs_o12: dict[str, Any] = {}
    o12_man = PASS2_DIR / "lpc_o12_refine" / "lpc_o12_refine_manifest.json"
    if o12_man.exists():
        import json

        data = json.loads(o12_man.read_text(encoding="utf-8"))
        o12_times = np.asarray(
            data.get("peak_times_s", {}).get("adaptive", []),
            dtype=np.float64,
        )
        if o12_times.size:

            def _counts(ref: np.ndarray, cand: np.ndarray) -> dict[str, int]:
                pairs: list[tuple[float, int, int]] = []
                for i, t in enumerate(ref):
                    lo = int(np.searchsorted(cand, t - MIN_EVENT_GAP_S, side="left"))
                    hi = int(np.searchsorted(cand, t + MIN_EVENT_GAP_S, side="right"))
                    for j in range(lo, hi):
                        pairs.append((abs(float(t - cand[j])), i, j))
                used_r: set[int] = set()
                used_c: set[int] = set()
                for _, i, j in sorted(pairs):
                    if i not in used_r and j not in used_c:
                        used_r.add(i)
                        used_c.add(j)
                common = len(used_r)
                return {
                    "common": common,
                    "reference_only": int(len(ref) - common),
                    "candidate_only": int(len(cand) - common),
                    "reference_n": int(len(ref)),
                    "candidate_n": int(len(cand)),
                }

            for key in ("o24", "o36"):
                vs_o12[f"{key}_vs_o12_adaptive"] = _counts(
                    o12_times, np.asarray(peak_sets[key], dtype=np.float64)
                )

    manifest = {
        "experiment": "lpc_o24_o36_sf_adaptive_clicks",
        "note": (
            "SuperFlux+peaks_adaptive on lpc_o24/o36 residual "
            "(same as o12 adaptive / 전체_adaptive_클릭). "
            "o12 refine outputs untouched."
        ),
        "fixed_rules": {
            "detector": "superflux_envelope + band_envelopes + peaks_adaptive",
            "click": CLICK_PARAMS,
            "listen_limit_mode": "clip",
            "outputs": [s[2] + "_클릭_p{N}.wav" for s in SOURCES],
            "filename_convention": "click_wav_name(stem, n_peaks) → *_클릭_p{N}.wav",
        },
        "peak_meta": peak_meta,
        "vs_o12_adaptive_30ms": vs_o12,
        "peak_times_s": peak_sets,
        "files": files,
        "o12_reference": {
            "path": "out/stems/Dir/event_sculpt/pass2/lpc_o12_refine/lpc_o12_residual_adaptive_클릭.wav",
            "note": "prior adopted; not rewritten",
        },
    }
    write_json(OUT_DIR / "lpc_sf_adaptive_manifest.json", manifest)
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
    write_json(OUT_DIR / "lpc_sf_adaptive_determinism.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--determinism-check", action="store_true")
    args = parser.parse_args()

    print(f"output: {OUT_DIR}")
    manifest = run_once()
    if manifest.get("vs_o12_adaptive_30ms"):
        print(f"vs_o12: {manifest['vs_o12_adaptive_30ms']}")

    if args.determinism_check:
        print("determinism-check: second run…")
        report = determinism_check(manifest)
        if not report["matched"]:
            raise SystemExit(f"determinism failed: {report}")
        print("determinism-check: OK")


if __name__ == "__main__":
    main()
