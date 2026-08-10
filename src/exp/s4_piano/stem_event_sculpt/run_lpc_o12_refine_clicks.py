"""lpc_o12 residual only: stronger floor / softgate / adaptive click variants.

Problem addressed: prior upward clicks fired on non-event spots.
Sources: pass2/lpc_o12_residual.wav only.

Variants:
  rms_plain              — RMS+Otsu baseline (re-emit)
  up_g4_floor_p25/p50    — upward g≤4 + strong percentile floor
  up_g4_floor_rel10/25   — upward g≤4 + floor = rel×block_p99
  softgate               — pass2 soft_env_gate then RMS+Otsu
  up_g4_softgate         — up_g4 (no floor) then softgate then RMS+Otsu
  adaptive               — SuperFlux + peaks_adaptive (=전체_adaptive_클릭)
  up_g4_floor_p25_adaptive / softgate_adaptive
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

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
from onset import band_envelopes, superflux_envelope  # noqa: E402
from peak_pick import peaks, peaks_adaptive  # noqa: E402

from io_util import (  # noqa: E402
    N_FFT,
    OUTPUT_DIR,
    audio_stats,
    read_stereo,
    sha256_file,
    write_json,
    write_listening_wav,
)
from passes_perc_refine import soft_env_gate  # noqa: E402
from passes_upward import upward_level_stereo  # noqa: E402

PASS2_DIR = OUTPUT_DIR / "pass2"
SOURCE = PASS2_DIR / "lpc_o12_residual.wav"
OUT_DIR = PASS2_DIR / "lpc_o12_refine"

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


def _peaks_rms(mono: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    env, times = _rms_envelope(
        mono, CLICK_PARAMS["rms_win"], CLICK_PARAMS["rms_hop"]
    )
    pk = peaks(env, times, min_gap_s=CLICK_PARAMS["min_gap_s"])
    return pk, {
        "method": "rms_otsu",
        "n_peaks": int(len(pk)),
        "env_max": float(env.max()) if env.size else 0.0,
    }


def _peaks_adaptive(mono: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    env, times = superflux_envelope(mono)
    bands = band_envelopes(mono)
    dur = float(len(mono) / SR)
    pk = peaks_adaptive(env, times, bands, dur, min_gap_s=CLICK_PARAMS["min_gap_s"])
    return pk, {
        "method": "superflux_peaks_adaptive",
        "n_peaks": int(len(pk)),
        "env_max": float(np.nanmax(env)) if env.size else 0.0,
        "note": "same detector family as 전체_adaptive_클릭",
    }


def _prep_plain(stereo: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    return stereo.astype(np.float32, copy=True), {"prep": "plain"}


def _prep_up(
    g_max: float,
    floor_pct: float | None = None,
    floor_rel: float | None = None,
) -> Callable[[np.ndarray], tuple[np.ndarray, dict[str, Any]]]:
    def _fn(stereo: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        out, meta = upward_level_stereo(
            stereo,
            g_max=g_max,
            floor_pct=floor_pct,
            floor_rel=floor_rel,
            otsu_split=False,
        )
        return out, {"prep": "upward", **meta}

    return _fn


def _prep_softgate(stereo: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    gated, _removed, meta = soft_env_gate(stereo)
    return gated, {"prep": "softgate", **meta}


def _prep_up_then_softgate(stereo: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    up, meta_up = upward_level_stereo(stereo, g_max=4.0)
    gated, _removed, meta_sg = soft_env_gate(up)
    return gated, {
        "prep": "up_g4_then_softgate",
        "upward": meta_up,
        "softgate": meta_sg,
    }


# (key, prep_fn, peak_fn, write_stem)
VARIANTS: tuple[
    tuple[
        str,
        Callable[[np.ndarray], tuple[np.ndarray, dict[str, Any]]],
        Callable[[np.ndarray], tuple[np.ndarray, dict[str, Any]]],
        bool,
    ],
    ...,
] = (
    ("rms_plain", _prep_plain, _peaks_rms, False),
    ("up_g4_floor_p25", _prep_up(4.0, floor_pct=25.0), _peaks_rms, True),
    ("up_g4_floor_p50", _prep_up(4.0, floor_pct=50.0), _peaks_rms, True),
    ("up_g4_floor_rel10", _prep_up(4.0, floor_rel=0.10), _peaks_rms, True),
    ("up_g4_floor_rel25", _prep_up(4.0, floor_rel=0.25), _peaks_rms, True),
    ("softgate", _prep_softgate, _peaks_rms, True),
    ("up_g4_softgate", _prep_up_then_softgate, _peaks_rms, True),
    ("adaptive", _prep_plain, _peaks_adaptive, False),
    ("up_g4_floor_p25_adaptive", _prep_up(4.0, floor_pct=25.0), _peaks_adaptive, True),
    ("softgate_adaptive", _prep_softgate, _peaks_adaptive, True),
)


def run_once() -> dict[str, Any]:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    stereo, sr = read_stereo(SOURCE)
    if sr != SR:
        raise RuntimeError(f"sr={sr}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    click = _click()
    files: dict[str, Any] = {}
    peak_sets: dict[str, list[float]] = {}
    peak_meta: dict[str, Any] = {}
    prep_meta: dict[str, Any] = {}

    for key, prep_fn, peak_fn, write_stem in VARIANTS:
        leveled, pmeta = prep_fn(stereo)
        prep_meta[key] = {
            **pmeta,
            "leveled_stats": audio_stats(leveled),
        }
        mono = leveled.mean(axis=1).astype(np.float32)
        pk, pkmeta = peak_fn(mono)
        peak_sets[key] = [float(t) for t in pk]
        peak_meta[key] = pkmeta

        if write_stem:
            stem_name = f"lpc_o12_residual_{key}.wav"
            stem_entry = write_listening_wav(
                OUT_DIR / stem_name, leveled, SR, limit_mode="clip"
            )
            files[stem_name] = {
                **stem_entry,
                "role": "o12_refine_stem",
                "variant": key,
            }

        click_name = f"lpc_o12_residual_{key}_클릭.wav"
        overlaid = _overlay(mono, pk, click)
        click_entry = write_listening_wav(
            OUT_DIR / click_name, overlaid, SR, limit_mode="clip"
        )
        files[click_name] = {
            **click_entry,
            "role": "o12_refine_click",
            "variant": key,
            "n_peaks": pkmeta["n_peaks"],
            "peak_meta": pkmeta,
        }
        print(
            f"  {click_name}: peaks={pkmeta['n_peaks']} "
            f"method={pkmeta.get('method')} "
            f"boost_frac={pmeta.get('boost_frac', pmeta.get('upward', {}).get('boost_frac', '-'))}"
        )

    manifest = {
        "experiment": "stem_event_sculpt_lpc_o12_refine_clicks",
        "note": (
            "lpc_o12 only. Stronger tiny-e floors, softgate, and "
            "SuperFlux+peaks_adaptive (전체_adaptive_클릭 family). "
            "Addresses non-event click dominance from open upward."
        ),
        "fixed_rules": {
            "source": "out/stems/Dir/event_sculpt/pass2/lpc_o12_residual.wav",
            "source_sha256": sha256_file(SOURCE),
            "variants": [v[0] for v in VARIANTS],
            "click": CLICK_PARAMS,
            "listen_limit_mode": "clip",
            "adaptive_ref": "src/exp/s3_2d/_sonify_adaptive.py → 전체_adaptive_클릭",
        },
        "source_stats": audio_stats(stereo),
        "prep_meta": prep_meta,
        "peak_meta": peak_meta,
        "peak_times_s": peak_sets,
        "files": files,
    }
    write_json(OUT_DIR / "lpc_o12_refine_manifest.json", manifest)
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
    write_json(OUT_DIR / "lpc_o12_refine_determinism.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--determinism-check", action="store_true")
    args = parser.parse_args()

    print(f"source: {SOURCE}")
    print(f"output: {OUT_DIR}")
    print(f"variants: {[v[0] for v in VARIANTS]}")

    manifest = run_once()
    print("peaks summary:")
    for k, meta in manifest["peak_meta"].items():
        print(f"  {k}: {meta['n_peaks']} ({meta.get('method')})")

    if args.determinism_check:
        print("determinism-check: second run…")
        report = determinism_check(manifest)
        if not report["matched"]:
            raise SystemExit(f"determinism failed: {report}")
        print("determinism-check: OK")


if __name__ == "__main__":
    main()
