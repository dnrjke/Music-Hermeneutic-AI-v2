"""LPC orders below 12 + k_env_adaptive → sf_adaptive peaks on piano (listen).

Fixed sub-12 grid: {4, 6, 8} (frame/hop/pre-emphasis = LPC_PARAMS; no retune).
For each order: LPC residual → SuperFlux+peaks_adaptive → clicks on BS piano.
Also: perc_tilt_k_env_adaptive peak times → clicks on piano.

Filenames: *_클릭_p{N}.wav (D-v2-04). Does not modify o12/24/36 pass2 residuals.
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

from config import MIN_EVENT_GAP_S, SR  # noqa: E402
from onset import band_envelopes, superflux_envelope  # noqa: E402
from peak_pick import peaks_adaptive  # noqa: E402

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
from passes_lpc import LPC_PARAMS, lpc_components  # noqa: E402

PASS2_DIR = OUTPUT_DIR / "pass2"
OUT_DIR = PASS2_DIR / "lpc_sf_adaptive_on_piano"
RES_DIR = PASS2_DIR / "lpc_low_order"

# Explicit sub-12 grid (user request); not an extension of {12,24,36} retuning.
LOW_ORDERS = (4, 6, 8)

TILT_DIR = OUTPUT_DIR / "tilt"
K_ENV_ADAPTIVE_MANIFEST = TILT_DIR / "tilt_k_env_adaptive_manifest.json"

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


def _sf_adaptive_peaks(mono: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    dur = float(len(mono) / SR)
    env, times = superflux_envelope(mono)
    bands = band_envelopes(mono)
    pk = peaks_adaptive(
        env, times, bands, dur, min_gap_s=CLICK_PARAMS["min_gap_s"]
    )
    meta = {
        "n_peaks": int(len(pk)),
        "method": "superflux_peaks_adaptive",
        "env_max": float(np.nanmax(env)) if env.size else 0.0,
    }
    return pk, meta


def _write_on_piano_pack(
    *,
    tag: str,
    piano_mono: np.ndarray,
    residual_stereo: np.ndarray | None,
    peak_times: np.ndarray,
    click: np.ndarray,
    files: dict[str, Any],
) -> str:
    n = int(len(peak_times))
    on_piano = _overlay(piano_mono, peak_times, click)
    out_name = click_wav_name(f"{tag}_on_piano", n)
    entry = write_listening_wav(OUT_DIR / out_name, on_piano, SR, limit_mode="clip")
    files[out_name] = {
        **entry,
        "role": "sf_adaptive_clicks_on_piano",
        "tag": tag,
        "n_peaks": n,
    }

    if residual_stereo is not None:
        # residual + clicks on R for stereo context
        r_mono = residual_stereo.mean(axis=1).astype(np.float32)
        r_click = _overlay(r_mono, peak_times, click)
        n_p = len(piano_mono)
        if len(r_click) < n_p:
            r_click = np.concatenate(
                [r_click, np.zeros(n_p - len(r_click), dtype=np.float32)]
            )
        elif len(r_click) > n_p:
            r_click = r_click[:n_p]
        stereo = np.column_stack([on_piano, r_click]).astype(np.float32)
        stereo_name = f"{tag}_pianoL_residClickR_p{n}.wav"
        st_entry = write_listening_wav(
            OUT_DIR / stereo_name, stereo, SR, limit_mode="clip"
        )
        files[stereo_name] = {
            **st_entry,
            "role": "stereo_pianoL_residualClickR",
            "tag": tag,
            "n_peaks": n,
        }
    else:
        # k_env: no LPC residual — stereo L=piano+clicks, R=piano dry
        stereo = np.column_stack([on_piano, piano_mono]).astype(np.float32)
        stereo_name = f"{tag}_pianoL_click_pianoR_dry_p{n}.wav"
        st_entry = write_listening_wav(
            OUT_DIR / stereo_name, stereo, SR, limit_mode="clip"
        )
        files[stereo_name] = {
            **st_entry,
            "role": "stereo_pianoL_click_pianoR_dry",
            "tag": tag,
            "n_peaks": n,
        }

    print(f"  {out_name}: peaks={n}")
    return out_name


def run_once() -> dict[str, Any]:
    if not SOURCE_PIANO.exists():
        raise FileNotFoundError(SOURCE_PIANO)
    piano, sr = read_stereo(SOURCE_PIANO)
    if sr != SR:
        raise RuntimeError(f"sr={sr}")
    piano_mono = piano.mean(axis=1).astype(np.float32)
    click = _click()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    files: dict[str, Any] = {}
    peak_meta: dict[str, Any] = {}
    peak_times_s: dict[str, list[float]] = {}

    # --- LPC orders < 12 ---
    for order in LOW_ORDERS:
        residual, synthesis = lpc_components(piano, order=order)
        res_name = f"lpc_o{order}_residual.wav"
        syn_name = f"lpc_o{order}_synthesis.wav"
        res_entry = write_listening_wav(
            RES_DIR / res_name, residual, SR, limit_mode="clip"
        )
        syn_entry = write_listening_wav(
            RES_DIR / syn_name, synthesis, SR, limit_mode="clip"
        )
        files[f"low/{res_name}"] = {**res_entry, "role": "lpc_low_residual", "order": order}
        files[f"low/{syn_name}"] = {
            **syn_entry,
            "role": "lpc_low_synthesis",
            "order": order,
        }

        res_mono = residual.mean(axis=1).astype(np.float32)
        pk, pmeta = _sf_adaptive_peaks(res_mono)
        # also save residual-only click (series naming)
        resid_click_name = click_wav_name(
            f"lpc_o{order}_residual_sf_adaptive", len(pk)
        )
        resid_click = write_listening_wav(
            RES_DIR / resid_click_name,
            _overlay(res_mono, pk, click),
            SR,
            limit_mode="clip",
        )
        files[f"low/{resid_click_name}"] = {
            **resid_click,
            "role": "lpc_low_residual_sf_adaptive_click",
            "order": order,
            "n_peaks": int(len(pk)),
        }

        tag = f"lpc_o{order}_residual_sf_adaptive"
        _write_on_piano_pack(
            tag=tag,
            piano_mono=piano_mono,
            residual_stereo=residual,
            peak_times=pk,
            click=click,
            files=files,
        )
        peak_meta[f"o{order}"] = {
            **pmeta,
            "order": order,
            "lpc_fixed": {
                "frame": LPC_PARAMS["frame"],
                "hop": LPC_PARAMS["hop"],
                "pre_emphasis": LPC_PARAMS["pre_emphasis"],
            },
            "residual_stats": audio_stats(residual),
        }
        peak_times_s[f"o{order}"] = [float(t) for t in pk]

    # --- perc_tilt_k_env_adaptive on piano ---
    if not K_ENV_ADAPTIVE_MANIFEST.exists():
        raise FileNotFoundError(K_ENV_ADAPTIVE_MANIFEST)
    k_man = json.loads(K_ENV_ADAPTIVE_MANIFEST.read_text(encoding="utf-8"))
    k_times = np.asarray(
        k_man.get("peak_times_s", {}).get("perc_tilt_k_env_adaptive", []),
        dtype=np.float64,
    )
    if k_times.size == 0:
        raise RuntimeError("no perc_tilt_k_env_adaptive peak times in manifest")
    _write_on_piano_pack(
        tag="perc_tilt_k_env_adaptive",
        piano_mono=piano_mono,
        residual_stereo=None,
        peak_times=k_times,
        click=click,
        files=files,
    )
    peak_meta["k_env_adaptive"] = {
        "n_peaks": int(len(k_times)),
        "source_manifest": str(K_ENV_ADAPTIVE_MANIFEST).replace("\\", "/"),
        "method": "peaks_from_tilt_k_env_adaptive_manifest",
    }
    peak_times_s["k_env_adaptive"] = [float(t) for t in k_times]

    manifest = {
        "experiment": "lpc_low_order_sf_adaptive_on_piano_plus_k_env",
        "note": (
            "LPC orders {4,6,8} residual → SuperFlux adaptive → on piano. "
            "Also perc_tilt_k_env_adaptive peaks on piano. "
            "pass2 o12/24/36 residuals untouched."
        ),
        "fixed_rules": {
            "low_orders": list(LOW_ORDERS),
            "lpc_frame_hop_pre": {
                "frame": LPC_PARAMS["frame"],
                "hop": LPC_PARAMS["hop"],
                "pre_emphasis": LPC_PARAMS["pre_emphasis"],
            },
            "detector": "superflux_envelope + peaks_adaptive",
            "piano": "out/stems/Dir/bs_roformer/piano.wav",
            "piano_sha256": sha256_file(SOURCE_PIANO),
            "click": CLICK_PARAMS,
            "filename_convention": "click_wav_name → *_클릭_p{N}.wav",
            "listen_limit_mode": "clip",
        },
        "piano_stats": audio_stats(piano),
        "peak_meta": peak_meta,
        "peak_times_s": peak_times_s,
        "files": files,
    }
    write_json(OUT_DIR / "lpc_low_and_k_env_on_piano_manifest.json", manifest)
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
    write_json(OUT_DIR / "lpc_low_and_k_env_on_piano_determinism.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--determinism-check", action="store_true")
    args = parser.parse_args()

    print(f"piano: {SOURCE_PIANO}")
    print(f"low orders: {LOW_ORDERS}")
    print(f"output: {OUT_DIR}")
    print(f"residuals: {RES_DIR}")

    manifest = run_once()
    print("peak counts:")
    for k, meta in manifest["peak_meta"].items():
        print(f"  {k}: {meta['n_peaks']}")

    if args.determinism_check:
        print("determinism-check: second run…")
        report = determinism_check(manifest)
        if not report["matched"]:
            raise SystemExit(f"determinism failed: {report}")
        print("determinism-check: OK")


if __name__ == "__main__":
    main()
