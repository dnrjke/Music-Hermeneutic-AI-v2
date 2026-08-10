"""LPC residual upward-level variants + Dir-style click sonify.

Weak frames pulled toward 2s-block p99; already-strong frames stay gain=1.
Includes no-floor open boost (tiny e not excluded) and careful floor / Otsu-split
variants. Softgate not used.
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
    audio_stats,
    read_stereo,
    sha256_file,
    write_json,
    write_listening_wav,
)
from passes_upward import (  # noqa: E402
    UPWARD_PARAMS,
    UPWARD_VARIANTS,
    upward_level_stereo,
)

PASS2_DIR = OUTPUT_DIR / "pass2"
OUT_DIR = PASS2_DIR / "lpc_upward"

CLICK_PARAMS = {
    "rms_win": N_FFT,
    "rms_hop": HOP,
    "click_freq_hz": 3000.0,
    "click_dur_ms": 12.0,
    "click_amp": 0.7,
    "min_gap_s": MIN_EVENT_GAP_S,
}

# Residuals only (weak-stem focus). pass1 o24 == pass2 o24 content, both kept.
RESIDUALS: tuple[tuple[Path, str, str], ...] = (
    (OUTPUT_DIR, "lpc_residual.wav", "pass1_residual_o24"),
    (PASS2_DIR, "lpc_o12_residual.wav", "pass2_residual_o12"),
    (PASS2_DIR, "lpc_o24_residual.wav", "pass2_residual_o24"),
    (PASS2_DIR, "lpc_o36_residual.wav", "pass2_residual_o36"),
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


def _peaks_from_mono(mono: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    env, times = _rms_envelope(
        mono, CLICK_PARAMS["rms_win"], CLICK_PARAMS["rms_hop"]
    )
    pk = peaks(env, times, min_gap_s=CLICK_PARAMS["min_gap_s"])
    return pk, {
        "n_peaks": int(len(pk)),
        "env_max": float(env.max()) if env.size else 0.0,
        "env_rms": float(np.sqrt(np.mean(env * env))) if env.size else 0.0,
    }


def run_once() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    click = _click()
    files: dict[str, Any] = {}
    peak_sets: dict[str, list[float]] = {}
    peak_meta: dict[str, Any] = {}
    upward_meta: dict[str, Any] = {}

    for src_dir, wav_name, src_key in RESIDUALS:
        src = src_dir / wav_name
        if not src.exists():
            raise FileNotFoundError(src)
        stereo, sr = read_stereo(src)
        if sr != SR:
            raise RuntimeError(f"{wav_name}: sr={sr}")
        stem = wav_name.replace(".wav", "")

        for variant, g_max, floor_pct, otsu_split in UPWARD_VARIANTS:
            leveled, meta = upward_level_stereo(
                stereo,
                g_max=g_max,
                floor_pct=floor_pct,
                otsu_split=otsu_split,
            )
            key = f"{src_key}__{variant}"
            upward_meta[key] = {
                **meta,
                "source_wav": wav_name,
                "source_sha256": sha256_file(src),
                "source_stats": audio_stats(stereo),
                "leveled_stats": audio_stats(leveled),
            }

            mono = leveled.mean(axis=1).astype(np.float32)
            pk, pmeta = _peaks_from_mono(mono)
            peak_sets[key] = [float(t) for t in pk]
            peak_meta[key] = pmeta

            stem_name = f"{stem}_{variant}.wav"
            click_name = f"{stem}_{variant}_클릭.wav"
            stem_entry = write_listening_wav(
                OUT_DIR / stem_name, leveled, SR, limit_mode="clip"
            )
            overlaid = _overlay(mono, pk, click)
            click_entry = write_listening_wav(
                OUT_DIR / click_name, overlaid, SR, limit_mode="clip"
            )
            files[stem_name] = {
                **stem_entry,
                "role": "lpc_upward_stem",
                "variant": variant,
                "source_key": src_key,
            }
            files[click_name] = {
                **click_entry,
                "role": "lpc_upward_click",
                "variant": variant,
                "source_key": src_key,
                "n_peaks": pmeta["n_peaks"],
                "peak_meta": pmeta,
            }
            print(
                f"  {click_name}: peaks={pmeta['n_peaks']} "
                f"boost_frac={meta['boost_frac']:.3f} "
                f"g_max_applied={meta['g_max_applied']:.1f}"
            )

    manifest = {
        "experiment": "stem_event_sculpt_lpc_upward_clicks",
        "note": (
            "Upward level residual: weak frames → block p99; strong stay g=1. "
            "up_open/up_g* have NO tiny-e floor (numerical eps only). "
            "up_open_p01/p05 carefully skip below block percentile. "
            "up_otsu_* boost only e<=Otsu. Softgate not used."
        ),
        "fixed_rules": {
            "inputs": [c[1] for c in RESIDUALS],
            "variants": [
                {
                    "key": k,
                    "g_max": g,
                    "floor_pct": f,
                    "otsu_split": o,
                }
                for k, g, f, o in UPWARD_VARIANTS
            ],
            "upward": UPWARD_PARAMS,
            "click": CLICK_PARAMS,
            "listen_limit_mode": "clip",
            "no_softgate": True,
            "no_395_compare": True,
        },
        "upward_meta": upward_meta,
        "peak_meta": peak_meta,
        "peak_times_s": peak_sets,
        "files": files,
    }
    write_json(OUT_DIR / "lpc_upward_manifest.json", manifest)
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
    write_json(OUT_DIR / "lpc_upward_determinism.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--determinism-check", action="store_true")
    args = parser.parse_args()

    print(f"output: {OUT_DIR}")
    print(f"residuals: {len(RESIDUALS)} × variants: {len(UPWARD_VARIANTS)}")
    for k, g, f, o in UPWARD_VARIANTS:
        print(f"  {k}: g_max={g} floor_pct={f} otsu_split={o}")

    manifest = run_once()
    print("peaks by variant (o24 residual):")
    for k, g, f, o in UPWARD_VARIANTS:
        key = f"pass2_residual_o24__{k}"
        n = manifest["peak_meta"].get(key, {}).get("n_peaks")
        print(f"  {k}: {n}")

    if args.determinism_check:
        print("determinism-check: second run…")
        report = determinism_check(manifest)
        if not report["matched"]:
            raise SystemExit(f"determinism failed: {report}")
        print("determinism-check: OK")


if __name__ == "__main__":
    main()
