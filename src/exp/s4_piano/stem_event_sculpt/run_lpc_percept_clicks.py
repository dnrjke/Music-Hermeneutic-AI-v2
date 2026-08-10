"""LPC click sonify with loudness/perceptual compensation (no softgate).

Variants per LPC WAV (pass1 + pass2 o12/24/36 residual & synthesis):
  - lufs:     stereo LUFS −23 → mono → RMS → Otsu → click
  - k_env:    stereo LUFS −23 → mono → BS.1770-4 K-weight → RMS → Otsu → click

Softgate deferred (attenuation always-on issue). ISO 226 not used (SPL free param).
Peak pick matches run_lpc_clicks (no 2s-p99). Overlay on the detection mono.
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

from audio_io import lufs_normalize  # noqa: E402
from config import HOP, MIN_EVENT_GAP_S, SR, TARGET_LUFS  # noqa: E402
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
from passes_percept import K_WEIGHT_PARAMS, k_weight_mono  # noqa: E402

PASS2_DIR = OUTPUT_DIR / "pass2"

CLICK_PARAMS = {
    "rms_win": N_FFT,
    "rms_hop": HOP,
    "click_freq_hz": 3000.0,
    "click_dur_ms": 12.0,
    "click_amp": 0.7,
    "min_gap_s": MIN_EVENT_GAP_S,
    "lufs_target": TARGET_LUFS,
    "match_tol_s": MIN_EVENT_GAP_S,
}

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

VARIANTS = ("lufs", "k_env")


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


def _stereo_lufs(stereo: np.ndarray) -> tuple[np.ndarray, float | None]:
    L, R = lufs_normalize(stereo[:, 0], stereo[:, 1])
    out = np.column_stack([L, R]).astype(np.float32)
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(SR)
        loud = float(meter.integrated_loudness(out))
    except Exception:
        loud = None
    return out, loud


def _one_to_one_counts(
    reference: np.ndarray, candidate: np.ndarray, tol_s: float
) -> dict[str, int]:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    pairs: list[tuple[float, int, int]] = []
    for i, t in enumerate(ref):
        lo = int(np.searchsorted(cand, t - tol_s, side="left"))
        hi = int(np.searchsorted(cand, t + tol_s, side="right"))
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


def _peaks_from_mono(mono: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
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
    peak_meta: dict[str, Any] = {}
    vs_plain: dict[str, Any] = {}
    lufs_meta: dict[str, Any] = {}

    # Load plain LPC click peak times for diagnostic compare (if present)
    plain_manifest_path = PASS2_DIR / "lpc_clicks_manifest.json"
    plain_peaks: dict[str, list[float]] = {}
    if plain_manifest_path.exists():
        import json

        plain = json.loads(plain_manifest_path.read_text(encoding="utf-8"))
        plain_peaks = {
            k: list(v) for k, v in plain.get("peak_times_s", {}).items()
        }

    for src_dir, wav_name, key in CANDIDATES:
        src = src_dir / wav_name
        if not src.exists():
            raise FileNotFoundError(src)
        stereo, sr = read_stereo(src)
        if sr != SR:
            raise RuntimeError(f"{wav_name}: sr={sr}")

        stereo_lufs, loud = _stereo_lufs(stereo)
        lufs_meta[key] = {
            "integrated_lufs": loud,
            "post_lufs_stats": audio_stats(stereo_lufs),
            "source_sha256": sha256_file(src),
        }
        mono_lufs = stereo_lufs.mean(axis=1).astype(np.float32)

        for variant in VARIANTS:
            if variant == "lufs":
                det_mono = mono_lufs
            elif variant == "k_env":
                det_mono = k_weight_mono(mono_lufs)
            else:
                raise ValueError(variant)

            pk, meta = _peaks_from_mono(det_mono)
            cand_key = f"{key}__{variant}"
            peak_sets[cand_key] = [float(t) for t in pk]
            peak_meta[cand_key] = meta

            if key in plain_peaks:
                vs_plain[cand_key] = _one_to_one_counts(
                    np.asarray(plain_peaks[key], dtype=np.float64),
                    pk,
                    CLICK_PARAMS["match_tol_s"],
                )

            stem = wav_name.replace(".wav", "")
            out_name = f"{stem}_{variant}_클릭.wav"
            overlaid = _overlay(det_mono, pk, click)
            entry = write_listening_wav(
                src_dir / out_name, overlaid, SR, limit_mode="clip"
            )
            files[out_name] = {
                **entry,
                "role": f"lpc_{variant}_click_sonify",
                "source_wav": wav_name,
                "candidate": key,
                "variant": variant,
                "n_peaks": meta["n_peaks"],
                "peak_meta": meta,
                "source_sha256": sha256_file(src),
            }
            print(
                f"  {out_name}: peaks={meta['n_peaks']} "
                f"peak={entry['peak']:.4f} rms={entry['rms']:.4f}"
            )

    manifest = {
        "experiment": "stem_event_sculpt_lpc_percept_clicks",
        "note": (
            "LPC loudness/perceptual click variants. "
            "lufs = LUFS−23 then RMS+Otsu. "
            "k_env = LUFS−23 then K-weight then RMS+Otsu. "
            "Softgate deferred. ISO 226 not used."
        ),
        "fixed_rules": {
            "style": "Dir mono click overlay on detection mono",
            "no_395_compare": True,
            "no_lr_stereo": True,
            "no_softgate": True,
            "peak_pick": "RMS envelope + Otsu local-max + greedy min_gap",
            "no_2s_p99": True,
            "variants": list(VARIANTS),
            "k_weight": K_WEIGHT_PARAMS,
            "click": CLICK_PARAMS,
            "listen_limit_mode": "clip",
            "candidates": [c[2] for c in CANDIDATES],
        },
        "lufs_meta": lufs_meta,
        "peak_meta": peak_meta,
        "vs_plain_lpc_clicks_30ms": vs_plain,
        "peak_times_s": peak_sets,
        "files": files,
    }
    write_json(PASS2_DIR / "lpc_percept_clicks_manifest.json", manifest)
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
    write_json(PASS2_DIR / "lpc_percept_clicks_determinism.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--determinism-check", action="store_true")
    args = parser.parse_args()

    print(f"pass1: {OUTPUT_DIR}")
    print(f"pass2: {PASS2_DIR}")
    print(f"variants: {VARIANTS} (softgate deferred)")
    print(f"LUFS target: {TARGET_LUFS}")
    print(f"K-weight: {K_WEIGHT_PARAMS}")

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
