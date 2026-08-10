"""Shared tilt/perc ±30ms presence clusters → 3kHz clicks on BS piano.

Used by agreement / disagreement on-piano runners for locked tilt materials.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

from config import MIN_EVENT_GAP_S, SR
from gen_tilt_material_peak_diff_doc import (
    TOL,
    cluster_presence,
    load_series,
)
from io_util import (
    OUTPUT_DIR,
    SOURCE_PIANO,
    audio_stats,
    click_wav_name,
    read_stereo,
    sha256_file,
    write_json,
    write_listening_wav,
)

PresenceKind = Literal["agreement", "disagreement"]

OUT_DIR = OUTPUT_DIR / "tilt"

CLICK_PARAMS = {
    "click_freq_hz": 3000.0,
    "click_dur_ms": 12.0,
    "click_amp": 0.7,
    "min_gap_s": MIN_EVENT_GAP_S,
}

_KIND_META: dict[PresenceKind, dict[str, str]] = {
    "disagreement": {
        "tag": "tilt_material_disagreement",
        "experiment": "tilt_material_disagreement_on_piano",
        "role": "tilt_material_disagreement_clicks_on_piano",
        "definition": (
            "±30ms presence cluster over included locked tilt/perc series; "
            "include iff not all series present"
        ),
        "note": (
            "Presence clusters (±30ms) across locked perc/tilt peak_times "
            "(raw/tilt_high/k_env/k_env_adaptive when available); "
            "clicks only where not all included series agree. "
            "Does not modify protected tilt WAVs."
        ),
    },
    "agreement": {
        "tag": "tilt_material_agreement",
        "experiment": "tilt_material_agreement_on_piano",
        "role": "tilt_material_agreement_clicks_on_piano",
        "definition": (
            "±30ms presence cluster over included locked tilt/perc series; "
            "include iff all series present"
        ),
        "note": (
            "Presence clusters (±30ms) across locked perc/tilt peak_times "
            "(raw/tilt_high/k_env/k_env_adaptive when available); "
            "clicks only where all included series agree. "
            "Complement of tilt_material_disagreement_on_piano. "
            "Does not modify protected tilt WAVs."
        ),
    },
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


def presence_peaks(kind: PresenceKind) -> tuple[list[float], dict[str, Any]]:
    series, keys, skipped = load_series()
    key_tuple = tuple(keys)
    clusters = cluster_presence(series, key_tuple)
    all_mask = frozenset(key_tuple)
    keep_all = kind == "agreement"
    kept: list[dict[str, Any]] = []
    n_all = 0
    for cl in clusters:
        present = frozenset(cl["orders"])
        is_all = present == all_mask
        if is_all:
            n_all += 1
        if is_all != keep_all:
            continue
        kept.append(
            {
                "rep": float(cl["rep"]),
                "series": sorted(present, key=key_tuple.index),
                "n_series": len(present),
            }
        )
    times = [d["rep"] for d in kept]
    meta = {
        "kind": kind,
        "tol_s": TOL,
        "series_keys": list(key_tuple),
        "skipped": skipped,
        "series_counts": {k: len(series[k]) for k in key_tuple},
        "n_clusters_total": len(clusters),
        "n_all_present": n_all,
        "n_disagreement": len(clusters) - n_all,
        "n_kept": len(times),
        "definition": _KIND_META[kind]["definition"],
        "clusters": kept,
    }
    return times, meta


def run_once(kind: PresenceKind) -> dict[str, Any]:
    meta_cfg = _KIND_META[kind]
    tag = meta_cfg["tag"]
    if not SOURCE_PIANO.exists():
        raise FileNotFoundError(SOURCE_PIANO)
    piano, sr = read_stereo(SOURCE_PIANO)
    if sr != SR:
        raise RuntimeError(f"sr={sr}")
    piano_mono = piano.mean(axis=1).astype(np.float32)
    click = _click()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    times, peak_meta = presence_peaks(kind)
    peak_arr = np.asarray(times, dtype=np.float64)
    n = int(len(peak_arr))

    on_piano = _overlay(piano_mono, peak_arr, click)
    out_name = click_wav_name(f"{tag}_on_piano", n)
    mono_entry = write_listening_wav(
        OUT_DIR / out_name, on_piano, SR, limit_mode="clip"
    )

    stereo = np.column_stack([on_piano, piano_mono]).astype(np.float32)
    stereo_name = f"{tag}_pianoL_click_pianoR_dry_p{n}.wav"
    stereo_entry = write_listening_wav(
        OUT_DIR / stereo_name, stereo, SR, limit_mode="clip"
    )

    files = {
        out_name: {
            **mono_entry,
            "role": meta_cfg["role"],
            "n_peaks": n,
        },
        stereo_name: {
            **stereo_entry,
            "role": "stereo_pianoL_click_pianoR_dry",
            "n_peaks": n,
        },
    }

    peak_meta_out: dict[str, Any] = {
        "n_peaks": n,
        "n_clusters_total": peak_meta["n_clusters_total"],
        "n_all_present": peak_meta["n_all_present"],
        "n_disagreement": peak_meta["n_disagreement"],
        "series_keys": peak_meta["series_keys"],
        "series_counts": peak_meta["series_counts"],
        "skipped": peak_meta["skipped"],
        "definition": peak_meta["definition"],
    }
    if kind == "disagreement":
        peak_meta_out["n_all_present_excluded"] = peak_meta["n_all_present"]
    else:
        peak_meta_out["n_disagreement_excluded"] = peak_meta["n_disagreement"]

    sources = {
        "perc_raw": "tilt/tilt_manifest.json → peak_times_s.perc_raw",
        "perc_tilt_high": "tilt/tilt_manifest.json → peak_times_s.perc_tilt_high",
        "perc_tilt_k_env": (
            "tilt/tilt_percept_manifest.json → peak_times_s.perc_tilt_k_env"
        ),
        "perc_tilt_k_env_adaptive": (
            "tilt/tilt_k_env_adaptive_manifest.json "
            "→ peak_times_s.perc_tilt_k_env_adaptive"
        ),
    }

    manifest = {
        "experiment": meta_cfg["experiment"],
        "note": meta_cfg["note"],
        "fixed_rules": {
            "tol_s": TOL,
            "series": peak_meta["series_keys"],
            "detector": "locked peak_times from existing tilt manifests (no retune)",
            "sources": {
                k: sources[k]
                for k in peak_meta["series_keys"]
                if k in sources
            },
            "piano": "out/stems/Dir/bs_roformer/piano.wav",
            "piano_sha256": sha256_file(SOURCE_PIANO),
            "click": CLICK_PARAMS,
            "filename_convention": "click_wav_name → *_클릭_p{N}.wav",
            "listen_limit_mode": "clip",
            "protected_untouched": True,
        },
        "piano_stats": audio_stats(piano),
        "peak_meta": peak_meta_out,
        "peak_times_s": times,
        "cluster_series": [
            {
                "t": c["rep"],
                "series": c["series"],
                "n_series": c["n_series"],
            }
            for c in peak_meta["clusters"]
        ],
        "files": files,
    }
    write_json(OUT_DIR / f"{meta_cfg['experiment']}_manifest.json", manifest)
    complement = (
        peak_meta["n_all_present"]
        if kind == "disagreement"
        else peak_meta["n_disagreement"]
    )
    print(f"  {out_name}: peaks={n} (complement={complement})")
    print(f"  {stereo_name}")
    return manifest


def determinism_check(kind: PresenceKind, manifest: dict[str, Any]) -> dict[str, Any]:
    first = {n: e["sha256"] for n, e in manifest["files"].items()}
    first_peaks = list(manifest["peak_times_s"])
    second = run_once(kind)
    mismatches = [
        n for n in first if first[n] != second["files"].get(n, {}).get("sha256")
    ]
    peak_mismatch = first_peaks != list(second["peak_times_s"])
    report = {
        "matched": len(mismatches) == 0 and not peak_mismatch,
        "wav_mismatches": mismatches,
        "peak_mismatch": peak_mismatch,
        "n_peaks": len(first_peaks),
    }
    exp = _KIND_META[kind]["experiment"]
    write_json(OUT_DIR / f"{exp}_determinism.json", report)
    return report


def main_for(kind: PresenceKind, *, determinism: bool) -> None:
    print(f"piano: {SOURCE_PIANO}")
    print(f"output: {OUT_DIR}")
    series, keys, skipped = load_series()
    print(f"kind: {kind}; tol: ±{int(TOL * 1000)}ms; series: {keys}")
    if skipped:
        print(f"skipped: {skipped}")
    print(f"series_counts: {{{', '.join(f'{k}:{len(series[k])}' for k in keys)}}}")

    manifest = run_once(kind)
    n = manifest["peak_meta"]["n_peaks"]
    print(f"{kind} peaks: {n}")

    if determinism:
        print("determinism-check: second run…")
        report = determinism_check(kind, manifest)
        if not report["matched"]:
            raise SystemExit(f"determinism failed: {report}")
        print("determinism-check: OK")
