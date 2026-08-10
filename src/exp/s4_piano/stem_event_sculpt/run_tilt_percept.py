"""Tilt follow-up: K-weight env (A) + sine_residual lowgate (B).

Baseline locked: v2 tilt→LUFS (perc_tilt_high). v3 softmakeup rejected.
A/B are 1st-pass separated; A∘B not in this run.
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
from passes_percept import (  # noqa: E402
    K_WEIGHT_PARAMS,
    k_weight_mono,
    sine_lowband_mono,
)
from passes_tilt import TILT_PARAMS, spectral_tilt  # noqa: E402

SOURCE_PERC = OUTPUT_DIR / "hpss_percussive.wav"
SOURCE_SINE_RES = OUTPUT_DIR / "sine_residual.wav"
TILT_DIR = OUTPUT_DIR / "tilt"
V2_CLICK = TILT_DIR / "perc_tilt_high_클릭.wav"
V2_STEM = TILT_DIR / "perc_tilt_high.wav"

CLICK_PARAMS = {
    "rms_win": N_FFT,
    "rms_hop": HOP,
    "env_norm_block_s": 2.0,
    "click_freq_hz": 3000.0,
    "click_dur_ms": 12.0,
    "click_amp": 0.7,
    "min_gap_s": MIN_EVENT_GAP_S,
    "lufs_target": TARGET_LUFS,
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


def _block_p99_norm(env: np.ndarray, hop: int, block_s: float) -> np.ndarray:
    sr_frames = SR / hop
    block = max(1, int(round(block_s * sr_frames)))
    out = np.zeros_like(env, dtype=np.float64)
    for start in range(0, len(env), block):
        end = min(start + block, len(env))
        seg = env[start:end]
        positive = seg[seg > 0]
        if positive.size == 0:
            scale = 1.0
        else:
            scale = float(np.percentile(positive, 99))
            if scale < 1e-12:
                scale = 1.0
        out[start:end] = np.clip(seg / scale, 0.0, 1.0)
    return out


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


def _peaks_from_env(
    env_n: np.ndarray, times: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    pk = peaks(env_n, times, min_gap_s=CLICK_PARAMS["min_gap_s"])
    meta = {
        "n_peaks": int(len(pk)),
        "env_norm_max": float(env_n.max()) if env_n.size else 0.0,
    }
    return pk, meta


def _one_to_one_counts(
    reference: np.ndarray, candidate: np.ndarray, tol_s: float
) -> dict[str, int]:
    """Greedy ascending-|delta| one-to-one match counts (diagnostic only)."""
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


def _write_click(
    mono: np.ndarray,
    peak_times: np.ndarray,
    *,
    name: str,
    role: str,
) -> dict[str, Any]:
    overlaid = _overlay(mono, peak_times, _click())
    entry = write_listening_wav(
        TILT_DIR / name, overlaid, SR, limit_mode="clip"
    )
    return {**entry, "role": role, "n_peaks": int(len(peak_times))}


def run_once() -> dict[str, Any]:
    if not SOURCE_PERC.exists():
        raise FileNotFoundError(SOURCE_PERC)
    if not SOURCE_SINE_RES.exists():
        raise FileNotFoundError(SOURCE_SINE_RES)

    perc, sr = read_stereo(SOURCE_PERC)
    if sr != SR:
        raise RuntimeError(f"sr={sr}")
    sine_res, sr_s = read_stereo(SOURCE_SINE_RES)
    if sr_s != SR:
        raise RuntimeError(f"sine sr={sr_s}")
    if sine_res.shape[0] != perc.shape[0]:
        raise RuntimeError(
            f"length mismatch perc={perc.shape[0]} sine={sine_res.shape[0]}"
        )

    tilted_raw, tilt_meta = spectral_tilt(perc)
    tilted_v2, loud_v2 = _stereo_lufs(tilted_raw)
    mono_v2 = tilted_v2.mean(axis=1).astype(np.float32)

    win = CLICK_PARAMS["rms_win"]
    hop = CLICK_PARAMS["rms_hop"]
    block_s = CLICK_PARAMS["env_norm_block_s"]

    # --- v2 baseline (reproduce) ---
    env_v2, times = _rms_envelope(mono_v2, win, hop)
    env_v2_n = _block_p99_norm(env_v2, hop, block_s)
    pk_v2, meta_v2 = _peaks_from_env(env_v2_n, times)
    meta_v2["env_max"] = float(env_v2.max()) if env_v2.size else 0.0

    # --- A: K-weight envelope ---
    mono_k = k_weight_mono(mono_v2)
    env_k, times_k = _rms_envelope(mono_k, win, hop)
    if len(times_k) != len(times):
        raise RuntimeError("K-env frame grid mismatch vs v2")
    env_k_n = _block_p99_norm(env_k, hop, block_s)
    pk_k, meta_k = _peaks_from_env(env_k_n, times_k)
    meta_k["env_max"] = float(env_k.max()) if env_k.size else 0.0

    # --- B: sine_residual lowband softgate ---
    sine_low, sine_low_meta = sine_lowband_mono(sine_res)
    env_sine, times_s = _rms_envelope(sine_low, win, hop)
    if len(times_s) != len(times):
        raise RuntimeError("sine-low frame grid mismatch vs v2")
    soft = _block_p99_norm(env_sine, hop, block_s)
    env_gated_n = np.clip(env_v2_n * (1.0 - soft), 0.0, 1.0)
    pk_g, meta_g = _peaks_from_env(env_gated_n, times)
    gate_on = soft > 0.0
    meta_g.update(
        {
            "env_max": float(env_v2.max()) if env_v2.size else 0.0,
            "gate_frames_on_frac": float(np.mean(gate_on)) if soft.size else 0.0,
            "gate_mean": float(np.mean(soft)) if soft.size else 0.0,
            "gate_p50": float(np.percentile(soft, 50)) if soft.size else 0.0,
            "gate_p99": float(np.percentile(soft, 99)) if soft.size else 0.0,
        }
    )

    TILT_DIR.mkdir(parents=True, exist_ok=True)
    # Do not rewrite v2 stem/click; compare against reproduced peaks + existing files.
    click_k = _write_click(
        mono_k,
        pk_k,
        name="perc_tilt_k_env_클릭.wav",
        role="tilt_v2_k_weight_env_click",
    )
    click_g = _write_click(
        mono_v2,
        pk_g,
        name="perc_tilt_sine_lowgate_클릭.wav",
        role="tilt_v2_sine_lowband_softgate_click",
    )

    tol = CLICK_PARAMS["match_tol_s"]
    vs_v2_k = _one_to_one_counts(pk_v2, pk_k, tol)
    vs_v2_g = _one_to_one_counts(pk_v2, pk_g, tol)

    manifest = {
        "experiment": "stem_event_sculpt_tilt_percept",
        "version": "v2_baseline_plus_k_env_plus_sine_lowgate",
        "note": (
            "v2 locked baseline (tilt→LUFS). v3 softmakeup rejected (same peaks). "
            "A: K-weight mono before RMS. "
            "B: sine_residual f<f_ref softgate × v2 env_n. "
            "A∘B deferred until after 1st listen."
        ),
        "fixed_rules": {
            "input_percussive": "out/stems/Dir/event_sculpt/hpss_percussive.wav",
            "input_sine_residual": "out/stems/Dir/event_sculpt/sine_residual.wav",
            "baseline": "v2 tilt→LUFS→RMS→2s-p99→Otsu",
            "tilt": TILT_PARAMS,
            "k_weight": K_WEIGHT_PARAMS,
            "sine_lowband": {
                "cutoff": "TILT_PARAMS.f_ref_hz",
                "rule": "STFT bins with f < f_ref, istft, mono-mean",
            },
            "gate_combine": "env_det = env_tilt_n * (1 - soft); soft = 2s-p99(sine_low_rms)",
            "click": CLICK_PARAMS,
            "listen_limit_mode": "clip",
            "no_395_compare": True,
            "no_A_compose_B": True,
        },
        "v3_status": {
            "verdict": "rejected",
            "reason": "peaks and peak times identical to v2; soft→LUFS ≈ global gain invert; v2 preferred",
        },
        "tilt_meta": {
            **tilt_meta,
            "v2_lufs": loud_v2,
            "v2_post_lufs_stats": audio_stats(tilted_v2),
            "sine_lowband": sine_low_meta,
            "sine_low_stats": audio_stats(sine_low),
            "k_filtered_mono_stats": audio_stats(mono_k),
        },
        "source_percussive_sha256": sha256_file(SOURCE_PERC),
        "source_sine_residual_sha256": sha256_file(SOURCE_SINE_RES),
        "existing_v2_files": {
            "perc_tilt_high.wav": {
                "exists": V2_STEM.exists(),
                "sha256": sha256_file(V2_STEM) if V2_STEM.exists() else None,
            },
            "perc_tilt_high_클릭.wav": {
                "exists": V2_CLICK.exists(),
                "sha256": sha256_file(V2_CLICK) if V2_CLICK.exists() else None,
            },
        },
        "peak_meta": {
            "v2_lufs_direct": meta_v2,
            "k_env": meta_k,
            "sine_lowgate": meta_g,
        },
        "vs_v2_30ms": {
            "k_env": vs_v2_k,
            "sine_lowgate": vs_v2_g,
        },
        "peak_times_s": {
            "perc_tilt_high": [float(t) for t in pk_v2],
            "perc_tilt_k_env": [float(t) for t in pk_k],
            "perc_tilt_sine_lowgate": [float(t) for t in pk_g],
        },
        "files": {
            "perc_tilt_k_env_클릭.wav": click_k,
            "perc_tilt_sine_lowgate_클릭.wav": click_g,
        },
    }
    write_json(TILT_DIR / "tilt_percept_manifest.json", manifest)
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
    write_json(TILT_DIR / "tilt_percept_determinism.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--determinism-check", action="store_true")
    args = parser.parse_args()

    print(f"input perc: {SOURCE_PERC}")
    print(f"input sine: {SOURCE_SINE_RES}")
    print(f"output: {TILT_DIR}")
    print(f"tilt: {TILT_PARAMS}")
    print(f"K-weight: {K_WEIGHT_PARAMS}")
    print(f"LUFS target: {TARGET_LUFS}")

    manifest = run_once()
    pm = manifest["peak_meta"]
    print(
        "peaks: "
        f"v2={pm['v2_lufs_direct']['n_peaks']} "
        f"k_env={pm['k_env']['n_peaks']} "
        f"sine_lowgate={pm['sine_lowgate']['n_peaks']}"
    )
    print(f"vs_v2_30ms: {manifest['vs_v2_30ms']}")
    print(
        "gate: "
        f"on_frac={pm['sine_lowgate']['gate_frames_on_frac']:.4f} "
        f"mean={pm['sine_lowgate']['gate_mean']:.4f}"
    )
    for name, entry in manifest["files"].items():
        print(
            f"  {name}: peaks={entry['n_peaks']} peak={entry['peak']:.4f} "
            f"rms={entry['rms']:.4f}"
        )

    if args.determinism_check:
        print("determinism-check: second run…")
        report = determinism_check(manifest)
        if not report["matched"]:
            raise SystemExit(f"determinism failed: {report}")
        print("determinism-check: OK")


if __name__ == "__main__":
    main()
