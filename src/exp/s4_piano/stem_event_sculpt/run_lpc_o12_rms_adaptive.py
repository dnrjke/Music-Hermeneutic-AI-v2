"""lpc_o12: adaptive pipeline with RMS (plain) instead of SuperFlux.

Same peaks_adaptive structure as 전체_adaptive_클릭:
  block-gate (16s) + norm residual + Q1 rescue + greedy 30ms.
Envelope & Q1 band profiles are frame-RMS (2048/256), not SuperFlux.

Variants:
  rms_adaptive     — RMS env + RMS bandpass bands → peaks_adaptive
  rms_adaptive_noq1 — RMS env → block-gate+norm residual only (no Q1)
  sf_adaptive      — SuperFlux adaptive re-emit (control; same as prior best)

Compares to existing rms_plain (RMS+Otsu only) conceptually.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import (  # noqa: E402
    HOP,
    MIN_EVENT_GAP_S,
    N_FFT,
    ONSET_BANDS,
    SR,
    WINDOW_S,
)
from onset import band_envelopes, superflux_envelope  # noqa: E402
from peak_pick import (  # noqa: E402
    BAND_NAMES,
    _greedy_select,
    _local_norm,
    otsu,
    peaks,
    peaks_adaptive,
)

from io_util import (  # noqa: E402
    OUTPUT_DIR,
    audio_stats,
    read_stereo,
    sha256_file,
    write_json,
    write_listening_wav,
)

PASS2_DIR = OUTPUT_DIR / "pass2"
SOURCE = PASS2_DIR / "lpc_o12_residual.wav"
OUT_DIR = PASS2_DIR / "lpc_o12_rms_adaptive"

CLICK_PARAMS = {
    "rms_win": N_FFT,
    "rms_hop": HOP,
    "click_freq_hz": 3000.0,
    "click_dur_ms": 12.0,
    "click_amp": 0.7,
    "min_gap_s": MIN_EVENT_GAP_S,
    "block_s": 16.0,
    "match_tol_s": MIN_EVENT_GAP_S,
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


def _bandpass(mono: np.ndarray, lo_hz: float, hi_hz: float) -> np.ndarray:
    """Zero-phase Butterworth bandpass; clamps to (1Hz, Nyquist-1)."""
    nyq = 0.5 * SR
    lo = max(lo_hz, 1.0) / nyq
    hi = min(hi_hz, nyq - 1.0) / nyq
    if not (0.0 < lo < hi < 1.0):
        return np.asarray(mono, dtype=np.float64)
    sos = signal.butter(2, [lo, hi], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, np.asarray(mono, dtype=np.float64))


def rms_envelope_and_bands(
    mono: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Fullband RMS env + per-ONSET_BANDS RMS envs on the same frame grid."""
    win = CLICK_PARAMS["rms_win"]
    hop = CLICK_PARAMS["rms_hop"]
    env, times = _rms_envelope(mono, win, hop)
    bands: dict[str, np.ndarray] = {}
    for lab, lo, hi in ONSET_BANDS:
        filtered = _bandpass(mono, lo, hi)
        b_env, b_times = _rms_envelope(filtered, win, hop)
        if len(b_env) != len(env) or not np.allclose(b_times, times):
            raise RuntimeError(f"band grid mismatch: {lab}")
        bands[lab] = b_env
    # peaks_adaptive expects BAND_NAMES keys
    for name in BAND_NAMES:
        if name not in bands:
            raise RuntimeError(f"missing band {name}")
    return env, times, bands


def peaks_blockgate_norm_only(
    env: np.ndarray,
    times: np.ndarray,
    *,
    block_s: float = 16.0,
    min_gap_s: float = MIN_EVENT_GAP_S,
) -> np.ndarray:
    """peaks_adaptive steps 1–3 only (no Q1)."""
    fin = np.isfinite(env)
    if fin.sum() < 3:
        return np.array([], dtype=np.float64)

    pk_base = peaks(env, times, min_gap_s=min_gap_s)
    env_normed = _local_norm(env, times, block_s=block_s)
    pk_norm = peaks(env_normed, times, min_gap_s=min_gap_s)

    frame_dt = HOP / SR
    dur = times[-1] + frame_dt
    n_blocks = int(np.ceil(dur / block_s))
    global_thr = otsu(env[fin])

    gate_times: set[float] = set()
    for bi in range(n_blocks):
        t0, t1 = bi * block_s, (bi + 1) * block_s
        mask_t = (times >= t0) & (times < t1)
        seg = env[mask_t]
        pos = seg[seg > 0]
        block_pct = float(np.percentile(pos, 99.0)) if len(pos) >= 5 else 0.0
        if block_pct < global_thr:
            pks = pk_norm[(pk_norm >= t0) & (pk_norm < t1)]
        else:
            pks = pk_base[(pk_base >= t0) & (pk_base < t1)]
        for t in pks:
            gate_times.add(t)

    for t in pk_norm:
        if t not in gate_times:
            if all(abs(t - g) >= min_gap_s for g in gate_times):
                gate_times.add(t)

    time_to_idx = {float(t): i for i, t in enumerate(times)}
    candidates = []
    for t in gate_times:
        idx = time_to_idx.get(float(t))
        if idx is not None:
            candidates.append((float(t), float(env[idx])))
    result = _greedy_select(candidates, min_gap_s)
    if not result:
        return np.array([], dtype=np.float64)
    return np.array(result, dtype=np.float64)


def _overlay(mono: np.ndarray, peak_times: np.ndarray, click: np.ndarray) -> np.ndarray:
    out = mono.astype(np.float32, copy=True)
    for t in peak_times:
        idx = int(t * SR)
        end = min(idx + len(click), len(out))
        n = end - idx
        if n > 0:
            out[idx:end] += click[:n]
    return out


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


def run_once() -> dict[str, Any]:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    stereo, sr = read_stereo(SOURCE)
    if sr != SR:
        raise RuntimeError(f"sr={sr}")
    mono = stereo.mean(axis=1).astype(np.float32)
    dur = float(len(mono) / SR)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    click = _click()

    # --- RMS adaptive family ---
    env_rms, times_rms, bands_rms = rms_envelope_and_bands(mono)
    pk_rms_ad = peaks_adaptive(
        env_rms,
        times_rms,
        bands_rms,
        dur,
        block_s=CLICK_PARAMS["block_s"],
        min_gap_s=CLICK_PARAMS["min_gap_s"],
    )
    pk_rms_noq1 = peaks_blockgate_norm_only(
        env_rms,
        times_rms,
        block_s=CLICK_PARAMS["block_s"],
        min_gap_s=CLICK_PARAMS["min_gap_s"],
    )
    pk_rms_plain = peaks(
        env_rms, times_rms, min_gap_s=CLICK_PARAMS["min_gap_s"]
    )

    # --- SuperFlux adaptive control ---
    env_sf, times_sf = superflux_envelope(mono)
    bands_sf = band_envelopes(mono)
    pk_sf_ad = peaks_adaptive(
        env_sf,
        times_sf,
        bands_sf,
        dur,
        block_s=CLICK_PARAMS["block_s"],
        min_gap_s=CLICK_PARAMS["min_gap_s"],
    )

    variants: dict[str, tuple[np.ndarray, dict[str, Any]]] = {
        "rms_plain": (
            pk_rms_plain,
            {
                "method": "rms_otsu",
                "n_peaks": int(len(pk_rms_plain)),
                "env_max": float(env_rms.max()),
            },
        ),
        "rms_adaptive_noq1": (
            pk_rms_noq1,
            {
                "method": "rms_blockgate_norm_no_q1",
                "n_peaks": int(len(pk_rms_noq1)),
                "env_max": float(env_rms.max()),
            },
        ),
        "rms_adaptive": (
            pk_rms_ad,
            {
                "method": "rms_peaks_adaptive",
                "n_peaks": int(len(pk_rms_ad)),
                "env_max": float(env_rms.max()),
                "note": "SuperFlux replaced by RMS env + RMS bandpass bands for Q1",
            },
        ),
        "sf_adaptive": (
            pk_sf_ad,
            {
                "method": "superflux_peaks_adaptive",
                "n_peaks": int(len(pk_sf_ad)),
                "env_max": float(np.nanmax(env_sf)),
                "note": "control = prior best / 전체_adaptive family",
            },
        ),
    }

    files: dict[str, Any] = {}
    peak_sets: dict[str, list[float]] = {}
    peak_meta: dict[str, Any] = {}
    for key, (pk, meta) in variants.items():
        peak_sets[key] = [float(t) for t in pk]
        peak_meta[key] = meta
        name = f"lpc_o12_residual_{key}_클릭.wav"
        overlaid = _overlay(mono, pk, click)
        entry = write_listening_wav(OUT_DIR / name, overlaid, SR, limit_mode="clip")
        files[name] = {
            **entry,
            "role": "o12_rms_adaptive_click",
            "variant": key,
            "n_peaks": meta["n_peaks"],
            "peak_meta": meta,
        }
        print(f"  {name}: peaks={meta['n_peaks']} ({meta['method']})")

    tol = CLICK_PARAMS["match_tol_s"]
    vs = {
        "rms_adaptive_vs_sf_adaptive": _one_to_one_counts(
            pk_sf_ad, pk_rms_ad, tol
        ),
        "rms_adaptive_vs_rms_plain": _one_to_one_counts(
            pk_rms_plain, pk_rms_ad, tol
        ),
        "rms_adaptive_vs_rms_noq1": _one_to_one_counts(
            pk_rms_noq1, pk_rms_ad, tol
        ),
        "rms_noq1_vs_rms_plain": _one_to_one_counts(
            pk_rms_plain, pk_rms_noq1, tol
        ),
    }

    manifest = {
        "experiment": "stem_event_sculpt_lpc_o12_rms_adaptive",
        "note": (
            "Replace SuperFlux with plain RMS inside peaks_adaptive. "
            "Q1 bands = Butterworth bandpass RMS on ONSET_BANDS (not SuperFlux). "
            "rms_adaptive_noq1 isolates block-gate+norm. sf_adaptive is control."
        ),
        "fixed_rules": {
            "source": "out/stems/Dir/event_sculpt/pass2/lpc_o12_residual.wav",
            "source_sha256": sha256_file(SOURCE),
            "rms_win": CLICK_PARAMS["rms_win"],
            "rms_hop": CLICK_PARAMS["rms_hop"],
            "block_s": CLICK_PARAMS["block_s"],
            "window_s_q1": WINDOW_S,
            "onset_bands": ONSET_BANDS,
            "bandpass": "butterworth order=2 sosfiltfilt",
            "click": CLICK_PARAMS,
            "listen_limit_mode": "clip",
        },
        "source_stats": audio_stats(stereo),
        "peak_meta": peak_meta,
        "vs_30ms": vs,
        "peak_times_s": peak_sets,
        "files": files,
    }
    write_json(OUT_DIR / "lpc_o12_rms_adaptive_manifest.json", manifest)
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
    write_json(OUT_DIR / "lpc_o12_rms_adaptive_determinism.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--determinism-check", action="store_true")
    args = parser.parse_args()

    print(f"source: {SOURCE}")
    print(f"output: {OUT_DIR}")
    manifest = run_once()
    print("vs_30ms:")
    for k, v in manifest["vs_30ms"].items():
        print(f"  {k}: {v}")

    if args.determinism_check:
        print("determinism-check: second run…")
        report = determinism_check(manifest)
        if not report["matched"]:
            raise SystemExit(f"determinism failed: {report}")
        print("determinism-check: OK")


if __name__ == "__main__":
    main()
