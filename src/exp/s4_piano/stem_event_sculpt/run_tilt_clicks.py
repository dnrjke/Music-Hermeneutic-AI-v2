"""Tilt hpss_percussive (low↓ high↑), then Dir-style mono clicks.

v2: tilt → LUFS −23 → peak/sonify (clip write). Soft-scale-only crushed level.
v3: tilt → soft_scale(0.98) → LUFS makeup → peak/sonify (clip write).
Both use 2s p99 RMS env before Otsu. Untilted control: perc_raw_클릭.
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
    soft_limit_for_listen,
    write_json,
    write_listening_wav,
)
from passes_tilt import TILT_PARAMS, spectral_tilt  # noqa: E402

SOURCE_PERC = OUTPUT_DIR / "hpss_percussive.wav"
TILT_DIR = OUTPUT_DIR / "tilt"

CLICK_PARAMS = {
    "rms_win": N_FFT,
    "rms_hop": HOP,
    "env_norm_block_s": 2.0,
    "click_freq_hz": 3000.0,
    "click_dur_ms": 12.0,
    "click_amp": 0.7,
    "min_gap_s": MIN_EVENT_GAP_S,
    "lufs_target": TARGET_LUFS,
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


def _peaks_from_mono(mono: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    env, times = _rms_envelope(
        mono, CLICK_PARAMS["rms_win"], CLICK_PARAMS["rms_hop"]
    )
    env_n = _block_p99_norm(
        env, CLICK_PARAMS["rms_hop"], CLICK_PARAMS["env_norm_block_s"]
    )
    pk = peaks(env_n, times, min_gap_s=CLICK_PARAMS["min_gap_s"])
    meta = {
        "n_peaks": int(len(pk)),
        "env_max": float(env.max()) if env.size else 0.0,
        "env_norm_max": float(env_n.max()) if env_n.size else 0.0,
    }
    return pk, meta


def _stem_and_click(
    stereo: np.ndarray,
    *,
    stem_name: str,
    click_name: str,
    role_stem: str,
    role_click: str,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, dict[str, Any]]:
    mono = stereo.mean(axis=1).astype(np.float32)
    pk, peak_meta = _peaks_from_mono(mono)
    overlaid = _overlay(mono, pk, _click())
    stem_entry = write_listening_wav(
        TILT_DIR / stem_name, stereo, SR, limit_mode="clip"
    )
    click_entry = write_listening_wav(
        TILT_DIR / click_name, overlaid, SR, limit_mode="clip"
    )
    stem_entry = {**stem_entry, "role": role_stem, "n_peaks_for_click": int(len(pk))}
    click_entry = {**click_entry, "role": role_click, "n_peaks": int(len(pk))}
    return stem_entry, click_entry, pk, peak_meta


def run_once() -> dict[str, Any]:
    if not SOURCE_PERC.exists():
        raise FileNotFoundError(SOURCE_PERC)

    perc, sr = read_stereo(SOURCE_PERC)
    if sr != SR:
        raise RuntimeError(f"sr={sr}")

    tilted_raw, tilt_meta = spectral_tilt(perc)
    TILT_DIR.mkdir(parents=True, exist_ok=True)

    # v2: tilt → LUFS → peak/listen (no soft-scale before LUFS)
    tilted_v2, loud_v2 = _stereo_lufs(tilted_raw)
    stem_v2, click_v2, pk_v2, meta_v2 = _stem_and_click(
        tilted_v2,
        stem_name="perc_tilt_high.wav",
        click_name="perc_tilt_high_클릭.wav",
        role_stem="tilted_lufs_direct",
        role_click="tilted_lufs_direct_click",
    )

    # v3: tilt → soft-scale → LUFS makeup → peak/listen
    soft = soft_limit_for_listen(tilted_raw)
    soft_stats = audio_stats(soft)
    tilted_v3, loud_v3 = _stereo_lufs(soft)
    stem_v3, click_v3, pk_v3, meta_v3 = _stem_and_click(
        tilted_v3,
        stem_name="perc_tilt_softmakeup.wav",
        click_name="perc_tilt_softmakeup_클릭.wav",
        role_stem="tilted_softlimit_then_lufs_makeup",
        role_click="tilted_softmakeup_click",
    )

    # Untilted control
    perc_lufs, loud_raw = _stereo_lufs(perc)
    stem_raw, click_raw, pk_raw, meta_raw = _stem_and_click(
        perc_lufs,
        stem_name="perc_raw_lufs_ref.wav",
        click_name="perc_raw_클릭.wav",
        role_stem="untilted_lufs_ref_stem",
        role_click="untilted_percussive_click_control",
    )

    manifest = {
        "experiment": "stem_event_sculpt_tilt_high_clicks",
        "version": "v2_lufs_direct_plus_v3_softmakeup",
        "note": (
            "v2: tilt→LUFS→clip write. "
            "v3: tilt→soft_scale(0.98)→LUFS makeup→clip write. "
            "Both use 2s-p99 RMS env before Otsu."
        ),
        "fixed_rules": {
            "input": "out/stems/Dir/event_sculpt/hpss_percussive.wav",
            "tilt": TILT_PARAMS,
            "click": CLICK_PARAMS,
            "env_block_p99_before_otsu": True,
            "listen_limit_mode": "clip",
            "no_395_compare": True,
        },
        "tilt_meta": {
            **tilt_meta,
            "pre_lufs_tilt_stats": audio_stats(tilted_raw),
            "v2_lufs": loud_v2,
            "v2_post_lufs_stats": audio_stats(tilted_v2),
            "v3_after_soft_stats": soft_stats,
            "v3_lufs_after_makeup": loud_v3,
            "v3_post_lufs_stats": audio_stats(tilted_v3),
            "lufs_raw_control": loud_raw,
        },
        "source_percussive_sha256": sha256_file(SOURCE_PERC),
        "source_percussive_stats": audio_stats(perc),
        "peak_meta": {
            "v2_lufs_direct": meta_v2,
            "v3_softmakeup": meta_v3,
            "raw": meta_raw,
        },
        "peak_times_s": {
            "perc_tilt_high": [float(t) for t in pk_v2],
            "perc_tilt_softmakeup": [float(t) for t in pk_v3],
            "perc_raw": [float(t) for t in pk_raw],
        },
        "files": {
            "perc_tilt_high.wav": stem_v2,
            "perc_tilt_high_클릭.wav": click_v2,
            "perc_tilt_softmakeup.wav": stem_v3,
            "perc_tilt_softmakeup_클릭.wav": click_v3,
            "perc_raw_클릭.wav": click_raw,
            "perc_raw_lufs_ref.wav": stem_raw,
        },
        "supersedes": {
            "v1_tilt_peaks": 140,
            "v1_raw_peaks": 1177,
            "v1_issue": "soft-scale alone crushed loudness",
        },
    }
    write_json(TILT_DIR / "tilt_manifest.json", manifest)
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
    write_json(TILT_DIR / "tilt_determinism.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--determinism-check", action="store_true")
    args = parser.parse_args()

    print(f"input: {SOURCE_PERC}")
    print(f"output: {TILT_DIR}")
    print(f"tilt: {TILT_PARAMS}")
    print(f"LUFS target: {TARGET_LUFS}")
    manifest = run_once()
    for name, entry in manifest["files"].items():
        peaks_n = entry.get("n_peaks", entry.get("n_peaks_for_click", "?"))
        print(
            f"  {name}: peaks={peaks_n} peak={entry['peak']:.4f} "
            f"rms={entry['rms']:.4f}"
        )
    tm = manifest["tilt_meta"]
    print(
        "lufs: "
        f"v2={tm.get('v2_lufs')} v3_makeup={tm.get('v3_lufs_after_makeup')} "
        f"raw={tm.get('lufs_raw_control')}"
    )
    print(
        "peaks: "
        f"v2={manifest['peak_meta']['v2_lufs_direct']['n_peaks']} "
        f"v3={manifest['peak_meta']['v3_softmakeup']['n_peaks']} "
        f"raw={manifest['peak_meta']['raw']['n_peaks']}"
    )

    if args.determinism_check:
        print("determinism-check: second run…")
        report = determinism_check(manifest)
        if not report["matched"]:
            raise SystemExit(f"determinism failed: {report}")
        print("determinism-check: OK")


if __name__ == "__main__":
    main()
