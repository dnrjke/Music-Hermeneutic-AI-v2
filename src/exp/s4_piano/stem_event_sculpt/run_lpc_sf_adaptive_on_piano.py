"""Mix LPC o12/o24/o36 sf_adaptive click peaks onto BS piano stem (listen pack).

For each order: overlay 3kHz clicks (same times as residual sf_adaptive) on
piano mono-mean. Also write stereo L=piano / R=residual+clicks for context.

Uses peak counts in filenames (D-v2-04). Does not modify source click WAVs.
"""
from __future__ import annotations

import argparse
import json
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

import soundfile as sf

from config import MIN_EVENT_GAP_S, SR  # noqa: E402

from io_util import (  # noqa: E402
    OUTPUT_DIR,
    SOURCE_PIANO,
    audio_stats,
    click_wav_name,
    read_stereo,
    sha256_file,
    write_json,
    write_listening_wav,
)

PASS2_DIR = OUTPUT_DIR / "pass2"
OUT_DIR = PASS2_DIR / "lpc_sf_adaptive_on_piano"

# (order_key, peak_times loader, residual_click_wav for R channel)
SERIES: tuple[tuple[str, Path, Path, str], ...] = (
    (
        "o12",
        PASS2_DIR / "lpc_o12_refine" / "lpc_o12_refine_manifest.json",
        PASS2_DIR / "lpc_o12_refine" / "lpc_o12_residual_adaptive_클릭.wav",
        "adaptive",  # key in peak_times_s
    ),
    (
        "o24",
        PASS2_DIR / "lpc_sf_adaptive" / "lpc_sf_adaptive_manifest.json",
        PASS2_DIR / "lpc_sf_adaptive" / "lpc_o24_residual_sf_adaptive_클릭_p406.wav",
        "o24",
    ),
    (
        "o36",
        PASS2_DIR / "lpc_sf_adaptive" / "lpc_sf_adaptive_manifest.json",
        PASS2_DIR / "lpc_sf_adaptive" / "lpc_o36_residual_sf_adaptive_클릭_p418.wav",
        "o36",
    ),
)

CLICK_PARAMS = {
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


def _load_mono_any(path: Path) -> np.ndarray:
    """Load wav as mono float32 (1ch or mean of stereo)."""
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if sample_rate != SR:
        raise RuntimeError(f"{path.name}: sr={sample_rate}")
    if audio.shape[1] == 1:
        return audio[:, 0].astype(np.float32, copy=False)
    return audio.mean(axis=1).astype(np.float32)


def _load_peaks(manifest_path: Path, key: str) -> np.ndarray:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    times = data.get("peak_times_s", {}).get(key)
    if times is None:
        raise KeyError(f"{manifest_path.name}: missing peak_times_s[{key}]")
    return np.asarray(times, dtype=np.float64)


def run_once() -> dict[str, Any]:
    if not SOURCE_PIANO.exists():
        raise FileNotFoundError(SOURCE_PIANO)
    piano, sr = read_stereo(SOURCE_PIANO)
    if sr != SR:
        raise RuntimeError(f"piano sr={sr}")
    piano_mono = piano.mean(axis=1).astype(np.float32)
    click = _click()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files: dict[str, Any] = {}
    peak_meta: dict[str, Any] = {}

    for order, man_path, resid_click_path, peak_key in SERIES:
        if not man_path.exists():
            raise FileNotFoundError(man_path)
        if not resid_click_path.exists():
            raise FileNotFoundError(resid_click_path)

        pk = _load_peaks(man_path, peak_key)
        n = int(len(pk))
        stem = f"lpc_{order}_residual_sf_adaptive_on_piano"
        out_name = click_wav_name(stem, n)

        on_piano = _overlay(piano_mono, pk, click)
        entry = write_listening_wav(
            OUT_DIR / out_name, on_piano, SR, limit_mode="clip"
        )

        # Stereo: L = piano+clicks, R = residual sf_adaptive click sonify (mono)
        r_mono = _load_mono_any(resid_click_path)
        n_p = len(piano_mono)
        if len(r_mono) < n_p:
            pad = np.zeros(n_p - len(r_mono), dtype=np.float32)
            r_mono = np.concatenate([r_mono, pad])
        elif len(r_mono) > n_p:
            r_mono = r_mono[:n_p]
        stereo = np.column_stack([on_piano, r_mono]).astype(np.float32)
        stereo_name = (
            f"lpc_{order}_residual_sf_adaptive_pianoL_residClickR_p{n}.wav"
        )
        stereo_entry = write_listening_wav(
            OUT_DIR / stereo_name, stereo, SR, limit_mode="clip"
        )

        peak_meta[order] = {
            "n_peaks": n,
            "manifest": str(man_path).replace("\\", "/"),
            "peak_key": peak_key,
            "source_residual_click": resid_click_path.name,
            "source_residual_click_sha256": sha256_file(resid_click_path),
        }
        files[out_name] = {
            **entry,
            "role": "sf_adaptive_clicks_on_piano",
            "order": order,
            "n_peaks": n,
        }
        files[stereo_name] = {
            **stereo_entry,
            "role": "stereo_pianoL_residualClickR",
            "order": order,
            "n_peaks": n,
            "L": "piano_mono + sf_adaptive clicks",
            "R": "residual sf_adaptive click sonify",
        }
        print(f"  {out_name}: peaks={n}")
        print(f"  {stereo_name}")

    manifest = {
        "experiment": "lpc_sf_adaptive_on_piano_listen_pack",
        "note": (
            "o12/o24/o36 sf_adaptive peak times overlaid on BS piano stem. "
            "Also stereo L=piano+clicks / R=residual click sonify. "
            "Source residual click WAVs unchanged."
        ),
        "fixed_rules": {
            "piano": "out/stems/Dir/bs_roformer/piano.wav",
            "piano_sha256": sha256_file(SOURCE_PIANO),
            "click": CLICK_PARAMS,
            "filename_convention": "click_wav_name → *_클릭_p{N}.wav",
            "listen_limit_mode": "clip",
            "series": [
                {
                    "order": o,
                    "manifest": str(m).replace("\\", "/"),
                    "residual_click": str(r).replace("\\", "/"),
                    "peak_key": k,
                }
                for o, m, r, k in SERIES
            ],
        },
        "piano_stats": audio_stats(piano),
        "peak_meta": peak_meta,
        "files": files,
    }
    write_json(OUT_DIR / "lpc_sf_adaptive_on_piano_manifest.json", manifest)
    return manifest


def determinism_check(manifest: dict[str, Any]) -> dict[str, Any]:
    first = {n: e["sha256"] for n, e in manifest["files"].items()}
    second = run_once()
    mismatches = [
        n for n in first if first[n] != second["files"].get(n, {}).get("sha256")
    ]
    report = {
        "matched": len(mismatches) == 0,
        "wav_mismatches": mismatches,
    }
    write_json(OUT_DIR / "lpc_sf_adaptive_on_piano_determinism.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--determinism-check", action="store_true")
    args = parser.parse_args()

    print(f"piano: {SOURCE_PIANO}")
    print(f"output: {OUT_DIR}")
    manifest = run_once()
    for name, entry in manifest["files"].items():
        if entry.get("role") == "sf_adaptive_clicks_on_piano":
            print(f"packed: {name} peaks={entry['n_peaks']}")

    if args.determinism_check:
        print("determinism-check: second run…")
        report = determinism_check(manifest)
        if not report["matched"]:
            raise SystemExit(f"determinism failed: {report}")
        print("determinism-check: OK")


if __name__ == "__main__":
    main()
