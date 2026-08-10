"""Apply 전체_adaptive (SuperFlux+peaks_adaptive) to perc_tilt_k_env material.

Material (same chain as perc_tilt_k_env_클릭):
  hpss_percussive → spectral_tilt → LUFS−23 → mono-mean → BS.1770-4 K-weight

Detection: SuperFlux + peaks_adaptive (block-gate + norm residual + Q1).
Does not overwrite existing tilt clicks (k_env / v2 / sine_lowgate).
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

from audio_io import lufs_normalize  # noqa: E402
from config import MIN_EVENT_GAP_S, SR, TARGET_LUFS  # noqa: E402
from onset import band_envelopes, superflux_envelope  # noqa: E402
from peak_pick import peaks, peaks_adaptive  # noqa: E402

from io_util import (  # noqa: E402
    N_FFT,
    OUTPUT_DIR,
    HOP,
    audio_stats,
    click_wav_name,
    read_stereo,
    sha256_file,
    write_json,
    write_listening_wav,
)
from passes_percept import K_WEIGHT_PARAMS, k_weight_mono  # noqa: E402
from passes_tilt import TILT_PARAMS, spectral_tilt  # noqa: E402

SOURCE_PERC = OUTPUT_DIR / "hpss_percussive.wav"
TILT_DIR = OUTPUT_DIR / "tilt"
V2_STEM = TILT_DIR / "perc_tilt_high.wav"
K_ENV_CLICK = TILT_DIR / "perc_tilt_k_env_클릭.wav"
OUT_STEM_REF = TILT_DIR / "perc_tilt_k_env_material_mono.wav"
OUT_CLICK_STEM = "perc_tilt_k_env_adaptive"

# Existing tilt products — never overwrite
PROTECTED = {
    "perc_tilt_high.wav",
    "perc_tilt_high_클릭.wav",
    "perc_tilt_k_env_클릭.wav",
    "perc_tilt_sine_lowgate_클릭.wav",
    "perc_tilt_softmakeup.wav",
    "perc_tilt_softmakeup_클릭.wav",
    "perc_raw_클릭.wav",
    "perc_raw_lufs_ref.wav",
}

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


def _build_k_env_material() -> tuple[np.ndarray, dict[str, Any]]:
    """Return mono K-weighted material identical in chain to k_env clicks."""
    if V2_STEM.exists():
        tilted_v2, sr = read_stereo(V2_STEM)
        if sr != SR:
            raise RuntimeError(f"sr={sr}")
        source_note = "loaded perc_tilt_high.wav (locked v2 tilt→LUFS)"
        tilt_meta: dict[str, Any] = {"from_file": str(V2_STEM)}
        loud = None
    else:
        if not SOURCE_PERC.exists():
            raise FileNotFoundError(SOURCE_PERC)
        perc, sr = read_stereo(SOURCE_PERC)
        if sr != SR:
            raise RuntimeError(f"sr={sr}")
        tilted_raw, tilt_meta = spectral_tilt(perc)
        tilted_v2, loud = _stereo_lufs(tilted_raw)
        source_note = "regenerated tilt→LUFS from hpss_percussive"

    mono_v2 = tilted_v2.mean(axis=1).astype(np.float32)
    mono_k = k_weight_mono(mono_v2)
    meta = {
        "source_note": source_note,
        "tilt_meta": tilt_meta,
        "v2_lufs": loud,
        "v2_stem_sha256": sha256_file(V2_STEM) if V2_STEM.exists() else None,
        "k_weight": K_WEIGHT_PARAMS,
        "tilt_params": TILT_PARAMS,
        "mono_v2_stats": audio_stats(mono_v2),
        "mono_k_stats": audio_stats(mono_k),
    }
    return mono_k, meta


def run_once() -> dict[str, Any]:
    for name in PROTECTED:
        if OUT_STEM_REF.name == name:
            raise RuntimeError(f"output collides with protected: {name}")

    protected_before = {
        name: sha256_file(TILT_DIR / name) if (TILT_DIR / name).exists() else None
        for name in sorted(PROTECTED)
    }

    mono_k, mat_meta = _build_k_env_material()
    dur = float(len(mono_k) / SR)

    # Adaptive (본선 전체_adaptive_클릭 방식)
    env_sf, times_sf = superflux_envelope(mono_k)
    bands = band_envelopes(mono_k)
    pk_ad = peaks_adaptive(
        env_sf, times_sf, bands, dur, min_gap_s=CLICK_PARAMS["min_gap_s"]
    )
    out_click_name = click_wav_name(OUT_CLICK_STEM, len(pk_ad))
    out_click_path = TILT_DIR / out_click_name
    if out_click_name in PROTECTED:
        raise RuntimeError(f"output collides with protected: {out_click_name}")

    # Control: same peak path as original k_env (RMS → 2s-p99 → Otsu)
    env_rms, times_rms = _rms_envelope(
        mono_k, CLICK_PARAMS["rms_win"], CLICK_PARAMS["rms_hop"]
    )
    env_n = _block_p99_norm(env_rms, CLICK_PARAMS["rms_hop"], 2.0)
    pk_rms = peaks(env_n, times_rms, min_gap_s=CLICK_PARAMS["min_gap_s"])

    TILT_DIR.mkdir(parents=True, exist_ok=True)
    stem_entry = write_listening_wav(
        OUT_STEM_REF, mono_k, SR, limit_mode="clip"
    )
    click_ad = write_listening_wav(
        out_click_path,
        _overlay(mono_k, pk_ad, _click()),
        SR,
        limit_mode="clip",
    )

    protected_after = {
        name: sha256_file(TILT_DIR / name) if (TILT_DIR / name).exists() else None
        for name in sorted(PROTECTED)
    }
    if protected_before != protected_after:
        raise RuntimeError("protected tilt files changed unexpectedly")

    # Compare to stored k_env peaks if manifest exists
    vs_stored: dict[str, Any] | None = None
    percept_man = TILT_DIR / "tilt_percept_manifest.json"
    if percept_man.exists():
        data = json.loads(percept_man.read_text(encoding="utf-8"))
        stored = np.asarray(
            data.get("peak_times_s", {}).get("perc_tilt_k_env", []),
            dtype=np.float64,
        )
        if stored.size:
            vs_stored = _one_to_one_counts(
                stored, pk_ad, CLICK_PARAMS["match_tol_s"]
            )

    manifest = {
        "experiment": "perc_tilt_k_env_superflux_adaptive",
        "note": (
            "Same audio material as perc_tilt_k_env_클릭 "
            "(tilt→LUFS→K-weight mono), but peaks via "
            "SuperFlux+peaks_adaptive like 전체_adaptive_클릭. "
            "Prior work applied adaptive only on LPC; this is perc/tilt path."
        ),
        "vn_session_status": {
            "verdict": "closed_keep_existing",
            "keep": "SuperFlux adaptive (not RMS-adaptive on VN)",
        },
        "fixed_rules": {
            "material": "perc_tilt_high (tilt→LUFS) → mono-mean → K-weight",
            "detector": "superflux_envelope + band_envelopes + peaks_adaptive",
            "click": CLICK_PARAMS,
            "listen_limit_mode": "clip",
            "protected_untouched": sorted(PROTECTED),
            "filename_convention": "click_wav_name → *_클릭_p{N}.wav",
        },
        "material_meta": mat_meta,
        "protected_sha256": protected_after,
        "peak_meta": {
            "k_env_adaptive": {
                "n_peaks": int(len(pk_ad)),
                "method": "superflux_peaks_adaptive",
                "env_max": float(np.nanmax(env_sf)) if env_sf.size else 0.0,
                "filename": out_click_name,
            },
            "k_env_rms_recompute": {
                "n_peaks": int(len(pk_rms)),
                "method": "rms_2s_p99_otsu_on_same_material",
                "env_max": float(env_rms.max()) if env_rms.size else 0.0,
            },
        },
        "vs_30ms": {
            "adaptive_vs_rms_recompute": _one_to_one_counts(
                pk_rms, pk_ad, CLICK_PARAMS["match_tol_s"]
            ),
            "adaptive_vs_stored_k_env": vs_stored,
        },
        "peak_times_s": {
            "perc_tilt_k_env_adaptive": [float(t) for t in pk_ad],
            "perc_tilt_k_env_rms_recompute": [float(t) for t in pk_rms],
        },
        "files": {
            OUT_STEM_REF.name: {
                **stem_entry,
                "role": "k_env_material_mono_ref",
            },
            out_click_name: {
                **click_ad,
                "role": "k_env_material_superflux_adaptive_click",
                "n_peaks": int(len(pk_ad)),
            },
        },
        "existing_k_env_click": {
            "path": str(K_ENV_CLICK).replace("\\", "/"),
            "exists": K_ENV_CLICK.exists(),
            "sha256": sha256_file(K_ENV_CLICK) if K_ENV_CLICK.exists() else None,
        },
    }
    write_json(TILT_DIR / "tilt_k_env_adaptive_manifest.json", manifest)
    return manifest


def determinism_check(manifest: dict[str, Any]) -> dict[str, Any]:
    first = {n: e["sha256"] for n, e in manifest["files"].items()}
    first_peaks = {
        k: list(v) for k, v in manifest["peak_times_s"].items()
    }
    second = run_once()
    mismatches = [
        n for n in first if first[n] != second["files"].get(n, {}).get("sha256")
    ]
    peak_mismatch = [
        k
        for k in first_peaks
        if first_peaks[k] != second["peak_times_s"].get(k)
    ]
    report = {
        "matched": len(mismatches) == 0 and len(peak_mismatch) == 0,
        "wav_mismatches": mismatches,
        "peak_mismatches": peak_mismatch,
    }
    write_json(TILT_DIR / "tilt_k_env_adaptive_determinism.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--determinism-check", action="store_true")
    args = parser.parse_args()

    print(f"v2 stem: {V2_STEM} exists={V2_STEM.exists()}")
    print(f"output stem: {OUT_CLICK_STEM}_클릭_p{{N}}.wav")
    print(f"protected (untouched): {sorted(PROTECTED)}")

    manifest = run_once()
    pm = manifest["peak_meta"]
    print(
        "peaks: "
        f"adaptive={pm['k_env_adaptive']['n_peaks']} "
        f"file={pm['k_env_adaptive'].get('filename')} "
        f"rms_recompute={pm['k_env_rms_recompute']['n_peaks']}"
    )
    print(f"vs_30ms: {manifest['vs_30ms']}")
    print("protected SHA unchanged: OK")

    if args.determinism_check:
        print("determinism-check: second run…")
        report = determinism_check(manifest)
        if not report["matched"]:
            raise SystemExit(f"determinism failed: {report}")
        print("determinism-check: OK")


if __name__ == "__main__":
    main()