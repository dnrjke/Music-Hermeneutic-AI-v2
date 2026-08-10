"""Click sonify all current LPC sculpt WAVs (pass1 + pass2 order sweep).

Same fixed rules as pass2_clicks: mono-mean → RMS(2048/256) → Otsu →
greedy 30ms → 3kHz overlay. Not perc-based; no 395 compare / L/R.
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

from config import HOP, MIN_EVENT_GAP_S, SR  # noqa: E402
from peak_pick import peaks  # noqa: E402

from io_util import (  # noqa: E402
    N_FFT,
    OUTPUT_DIR,
    read_stereo,
    sha256_file,
    write_json,
    write_listening_wav,
)

PASS2_DIR = OUTPUT_DIR / "pass2"

CLICK_PARAMS = {
    "rms_win": N_FFT,
    "rms_hop": HOP,
    "click_freq_hz": 3000.0,
    "click_dur_ms": 12.0,
    "click_amp": 0.7,
    "min_gap_s": MIN_EVENT_GAP_S,
}

# (source_dir, wav_name, key)
CANDIDATES: tuple[tuple[Path, str, str], ...] = (
    (OUTPUT_DIR, "lpc_residual.wav", "pass1_residual_o24"),
    (OUTPUT_DIR, "lpc_synthesis.wav", "pass1_synthesis_o24"),
    (PASS2_DIR, "lpc_o12_residual.wav", "pass2_residual_o12"),
    (PASS2_DIR, "lpc_o12_synthesis.wav", "pass2_synthesis_o12"),
    (PASS2_DIR, "lpc_o24_residual.wav", "pass2_residual_o24"),
    (PASS2_DIR, "lpc_o24_synthesis.wav", "pass2_synthesis_o24"),
    (PASS2_DIR, "lpc_o36_residual.wav", "pass2_residual_o36"),
    (PASS2_DIR, "lpc_o36_synthesis.wav", "pass2_synthesis_o36"),
)


def _click(sr: int = SR) -> np.ndarray:
    n = int(sr * CLICK_PARAMS["click_dur_ms"] / 1000.0)
    t = np.arange(n, dtype=np.float32) / sr
    env = np.exp(-t * 1000.0 / CLICK_PARAMS["click_dur_ms"])
    return (
        CLICK_PARAMS["click_amp"]
        * env
        * np.sin(2 * np.pi * CLICK_PARAMS["click_freq_hz"] * t)
    ).astype(np.float32)


def _rms_envelope(mono: np.ndarray, win: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(mono, dtype=np.float64)
    if len(x) < win:
        env = np.array([np.sqrt(np.mean(x * x))], dtype=np.float64)
        times = np.array([0.5 * len(x) / SR], dtype=np.float64)
        return env, times
    n_frames = 1 + (len(x) - win) // hop
    env = np.empty(n_frames, dtype=np.float64)
    times = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        s = i * hop
        seg = x[s : s + win]
        env[i] = np.sqrt(np.mean(seg * seg))
        times[i] = (s + 0.5 * win) / SR
    return env, times


def _overlay(mono: np.ndarray, peak_times: np.ndarray, click: np.ndarray) -> np.ndarray:
    out = mono.astype(np.float32, copy=True)
    for t in peak_times:
        idx = int(t * SR)
        end = min(idx + len(click), len(out))
        n = end - idx
        if n > 0:
            out[idx:end] += click[:n]
    return out


def peaks_from_residue(stereo: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    mono = stereo.mean(axis=1).astype(np.float32)
    env, times = _rms_envelope(
        mono, CLICK_PARAMS["rms_win"], CLICK_PARAMS["rms_hop"]
    )
    pk = peaks(env, times, min_gap_s=CLICK_PARAMS["min_gap_s"])
    meta = {
        "n_peaks": int(len(pk)),
        "env_frames": int(len(env)),
        "env_max": float(env.max()) if env.size else 0.0,
        "env_rms": float(np.sqrt(np.mean(env * env))) if env.size else 0.0,
    }
    return pk, meta


def run_once() -> dict[str, Any]:
    click = _click()
    files: dict[str, Any] = {}
    peak_sets: dict[str, list[float]] = {}

    for src_dir, wav_name, key in CANDIDATES:
        src = src_dir / wav_name
        if not src.exists():
            raise FileNotFoundError(src)
        stereo, sr = read_stereo(src)
        if sr != SR:
            raise RuntimeError(f"{wav_name}: sr={sr}")
        mono = stereo.mean(axis=1).astype(np.float32)
        pk, meta = peaks_from_residue(stereo)
        peak_sets[key] = [float(t) for t in pk]
        overlaid = _overlay(mono, pk, click)
        out_name = wav_name.replace(".wav", "_클릭.wav")
        out_path = src_dir / out_name
        entry = write_listening_wav(out_path, overlaid, SR)
        files[out_name] = {
            **entry,
            "role": "lpc_click_sonify",
            "source_wav": wav_name,
            "candidate": key,
            "n_peaks": meta["n_peaks"],
            "peak_meta": meta,
            "source_sha256": sha256_file(src),
        }
        print(
            f"  {out_name}: peaks={meta['n_peaks']} "
            f"peak={entry['peak']:.4f} rms={entry['rms']:.4f}"
        )

    manifest = {
        "experiment": "stem_event_sculpt_lpc_clicks",
        "note": (
            "All current LPC residual/synthesis WAVs (pass1 o24 + pass2 o12/24/36). "
            "Same peak/click rules as pass2_clicks; not perc-based."
        ),
        "fixed_rules": {
            "style": "Dir mono click overlay on residue mono-mean",
            "no_395_compare": True,
            "no_lr_stereo": True,
            "peak_pick": "RMS envelope + Otsu local-max + greedy min_gap",
            "click_hz": CLICK_PARAMS["click_freq_hz"],
            "min_gap_s": CLICK_PARAMS["min_gap_s"],
            "rms_win": CLICK_PARAMS["rms_win"],
            "rms_hop": CLICK_PARAMS["rms_hop"],
            "candidates": [c[2] for c in CANDIDATES],
        },
        "parameters": CLICK_PARAMS,
        "peak_times_s": peak_sets,
        "files": files,
    }
    write_json(PASS2_DIR / "lpc_clicks_manifest.json", manifest)
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
    write_json(PASS2_DIR / "lpc_clicks_determinism.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--determinism-check", action="store_true")
    args = parser.parse_args()

    print(f"pass1 dir: {OUTPUT_DIR}")
    print(f"pass2 dir: {PASS2_DIR}")
    print(f"candidates: {len(CANDIDATES)}")
    manifest = run_once()
    print("peaks summary:")
    for key, times in manifest["peak_times_s"].items():
        print(f"  {key}: {len(times)}")

    if args.determinism_check:
        print("determinism-check: second run…")
        report = determinism_check(manifest)
        if not report["matched"]:
            raise SystemExit(f"determinism failed: {report}")
        print("determinism-check: OK")


if __name__ == "__main__":
    main()
