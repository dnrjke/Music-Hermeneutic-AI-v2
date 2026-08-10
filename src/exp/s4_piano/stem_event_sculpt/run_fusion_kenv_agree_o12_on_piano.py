"""Fuse k_env_adaptive ∪ LPC-order-agreement + o12-only (100ms deburst) on piano.

Merge order (fixed):
  1) perc_tilt_k_env_adaptive peaks (base)
  2) add lpc_order_agreement peaks not within ±30ms of base
  3) o12 adaptive peaks not in (1∪2) → chronological 100ms wide-gap deburst → add

Outputs (D-v2-04):
  - unified 3kHz clicks on piano
  - frequency-separated: k_env=3kHz, agreement-only=5kHz, o12-deburst=2kHz
  - stereo L=clicks / R=piano dry for each

Does not modify source manifests or protected WAVs.
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

PASS2 = OUTPUT_DIR / "pass2"
OUT_DIR = PASS2 / "lpc_sf_adaptive_on_piano"
TILT_DIR = OUTPUT_DIR / "tilt"

MATCH_TOL_S = 0.03
DEBURST_GAP_S = 0.100  # novelty deburst convention (~8 attacks/s)

KENV_MANIFEST = TILT_DIR / "tilt_k_env_adaptive_manifest.json"
AGREE_MANIFEST = OUT_DIR / "lpc_order_agreement_on_piano_manifest.json"
O12_MANIFEST = PASS2 / "lpc_o12_refine" / "lpc_o12_refine_manifest.json"

TAG_UNIFIED = "fusion_kenv_agree_o12db_on_piano"
TAG_FREQSEP = "fusion_kenv_agree_o12db_on_piano_freqsep"


def freqsep_click_wav_name(*, n_total: int, n_5k: int, n_2k: int) -> str:
    """Freqsep filename with 5k/2k counts visible (plus total pN)."""
    return (
        f"{TAG_FREQSEP}_클릭_p{int(n_total)}_5k{int(n_5k)}_2k{int(n_2k)}.wav"
    )


# Layer click freqs (structure path historically used 3k + 5k)
FREQ_KENV_HZ = 3000.0
FREQ_AGREE_ONLY_HZ = 5000.0
FREQ_O12_DB_HZ = 2000.0

CLICK_DUR_MS = 12.0
CLICK_AMP = 0.7
# Quiet piano bed for click-audibility (fixed; ~-14 dB)
PIANO_GAIN_LOW = 0.20


def _click(freq_hz: float, *, sr: int = SR) -> np.ndarray:
    n = int(sr * CLICK_DUR_MS / 1000.0)
    t = np.arange(n, dtype=np.float32) / sr
    env = np.exp(-t * 1000.0 / CLICK_DUR_MS)
    return (CLICK_AMP * env * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _overlay(
    mono: np.ndarray, peak_times: list[float] | np.ndarray, click: np.ndarray
) -> np.ndarray:
    out = mono.astype(np.float32, copy=True)
    for t in peak_times:
        idx = int(float(t) * SR)
        end = min(idx + len(click), len(out))
        n = end - idx
        if n > 0:
            out[idx:end] += click[:n]
    return out


def _covered(t: float, accepted: list[float], tol: float = MATCH_TOL_S) -> bool:
    return any(abs(t - a) <= tol for a in accepted)


def _add_uncovered(
    base: list[float], candidates: list[float], *, tol: float = MATCH_TOL_S
) -> tuple[list[float], list[float]]:
    out = list(base)
    added: list[float] = []
    for t in candidates:
        if not _covered(t, out, tol):
            out.append(float(t))
            added.append(float(t))
    out.sort()
    added.sort()
    return out, added


def _deburst_chrono(times: list[float], gap_s: float = DEBURST_GAP_S) -> list[float]:
    """Wide-gap keep: chronological first survivor in each burst window."""
    kept: list[float] = []
    for t in sorted(times):
        if not kept or (t - kept[-1]) >= gap_s:
            kept.append(float(t))
    return kept


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_layers() -> dict[str, list[float]]:
    if not KENV_MANIFEST.exists():
        raise FileNotFoundError(KENV_MANIFEST)
    if not AGREE_MANIFEST.exists():
        raise FileNotFoundError(AGREE_MANIFEST)
    if not O12_MANIFEST.exists():
        raise FileNotFoundError(O12_MANIFEST)

    kenv = [
        float(t)
        for t in _load_json(KENV_MANIFEST)["peak_times_s"]["perc_tilt_k_env_adaptive"]
    ]
    agree = [float(t) for t in _load_json(AGREE_MANIFEST)["peak_times_s"]]
    o12 = [float(t) for t in _load_json(O12_MANIFEST)["peak_times_s"]["adaptive"]]
    return {"kenv": sorted(kenv), "agree": sorted(agree), "o12": sorted(o12)}


def fuse(layers: dict[str, list[float]]) -> dict[str, Any]:
    kenv = layers["kenv"]
    agree = layers["agree"]
    o12 = layers["o12"]

    base1 = list(kenv)
    after_agree, agree_only = _add_uncovered(base1, agree)
    o12_raw_extra = [t for t in o12 if not _covered(t, after_agree)]
    o12_db = _deburst_chrono(o12_raw_extra, DEBURST_GAP_S)
    # o12_db may still be within 30ms of each other only if gap>tol; still check vs base
    final, o12_added = _add_uncovered(after_agree, o12_db)

    return {
        "kenv": kenv,
        "agree_only": agree_only,
        "o12_raw_extra": o12_raw_extra,
        "o12_deburst_extra": o12_added,
        "o12_deburst_dropped": [
            t for t in o12_raw_extra if not _covered(t, o12_added, tol=1e-9)
        ],
        "final": final,
        "counts": {
            "kenv": len(kenv),
            "agree": len(agree),
            "o12": len(o12),
            "agree_only_added": len(agree_only),
            "o12_raw_extra": len(o12_raw_extra),
            "o12_deburst_extra": len(o12_added),
            "o12_deburst_dropped": len(o12_raw_extra) - len(o12_added),
            "final_unified": len(final),
            "freqsep_events": len(kenv) + len(agree_only) + len(o12_added),
        },
    }


def run_once() -> dict[str, Any]:
    if not SOURCE_PIANO.exists():
        raise FileNotFoundError(SOURCE_PIANO)
    piano, sr = read_stereo(SOURCE_PIANO)
    if sr != SR:
        raise RuntimeError(f"sr={sr}")
    piano_mono = piano.mean(axis=1).astype(np.float32)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    layers = load_layers()
    fused = fuse(layers)
    counts = fused["counts"]
    n_uni = int(counts["final_unified"])
    n_fs = int(counts["freqsep_events"])

    click3 = _click(FREQ_KENV_HZ)
    click5 = _click(FREQ_AGREE_ONLY_HZ)
    click2 = _click(FREQ_O12_DB_HZ)

    # Unified 3kHz
    uni = _overlay(piano_mono, fused["final"], click3)
    uni_name = click_wav_name(TAG_UNIFIED, n_uni)
    uni_entry = write_listening_wav(OUT_DIR / uni_name, uni, SR, limit_mode="clip")
    uni_stereo = np.column_stack([uni, piano_mono]).astype(np.float32)
    uni_st_name = f"fusion_kenv_agree_o12db_pianoL_click_pianoR_dry_p{n_uni}.wav"
    uni_st_entry = write_listening_wav(
        OUT_DIR / uni_st_name, uni_stereo, SR, limit_mode="clip"
    )

    n_kenv = int(counts["kenv"])
    n_5k = int(counts["agree_only_added"])
    n_2k = int(counts["o12_deburst_extra"])

    # Frequency-separated layers (disjoint by construction)
    freq = piano_mono.astype(np.float32, copy=True)
    freq = _overlay(freq, fused["kenv"], click3)
    freq = _overlay(freq, fused["agree_only"], click5)
    freq = _overlay(freq, fused["o12_deburst_extra"], click2)
    fs_name = freqsep_click_wav_name(n_total=n_fs, n_5k=n_5k, n_2k=n_2k)
    fs_entry = write_listening_wav(OUT_DIR / fs_name, freq, SR, limit_mode="clip")
    fs_stereo = np.column_stack([freq, piano_mono]).astype(np.float32)
    fs_st_name = (
        f"fusion_kenv_agree_o12db_freqsep_pianoL_click_pianoR_dry"
        f"_p{n_fs}_5k{n_5k}_2k{n_2k}.wav"
    )
    fs_st_entry = write_listening_wav(
        OUT_DIR / fs_st_name, fs_stereo, SR, limit_mode="clip"
    )

    # A/B stereo: L=unified 3k, R=freqsep (same piano bed under both)
    ab = np.column_stack([uni, freq]).astype(np.float32)
    ab_name = (
        f"fusion_kenv_agree_o12db_unifiedL_freqsepR"
        f"_p{n_uni}_5k{n_5k}_2k{n_2k}.wav"
    )
    ab_entry = write_listening_wav(OUT_DIR / ab_name, ab, SR, limit_mode="clip")

    # Click-only (no piano bed) — same length as piano for scrub sync
    silence = np.zeros_like(piano_mono)
    kenv_only = _overlay(silence, fused["kenv"], click3)
    o12_only = _overlay(silence, layers["o12"], click3)
    freq_only = _overlay(silence, fused["kenv"], click3)
    freq_only = _overlay(freq_only, fused["agree_only"], click5)
    freq_only = _overlay(freq_only, fused["o12_deburst_extra"], click2)
    uni_only = _overlay(silence, fused["final"], click3)

    n_o12 = int(counts["o12"])
    kenv_np_name = click_wav_name("perc_tilt_k_env_adaptive_nopiano", n_kenv)
    o12_np_name = click_wav_name("lpc_o12_residual_sf_adaptive_nopiano", n_o12)
    fs_np_name = (
        f"fusion_kenv_agree_o12db_freqsep_nopiano"
        f"_클릭_p{n_fs}_5k{n_5k}_2k{n_2k}.wav"
    )
    uni_np_name = click_wav_name("fusion_kenv_agree_o12db_nopiano", n_uni)
    # L=kenv-only R=o12-only for A/B without piano
    ab_np = np.column_stack([kenv_only, o12_only]).astype(np.float32)
    ab_np_name = (
        f"kenvAd_nopianoL_o12_nopianoR_클릭_p{n_kenv}x{n_o12}.wav"
    )

    kenv_np_entry = write_listening_wav(
        OUT_DIR / kenv_np_name, kenv_only, SR, limit_mode="clip"
    )
    o12_np_entry = write_listening_wav(
        OUT_DIR / o12_np_name, o12_only, SR, limit_mode="clip"
    )
    fs_np_entry = write_listening_wav(
        OUT_DIR / fs_np_name, freq_only, SR, limit_mode="clip"
    )
    uni_np_entry = write_listening_wav(
        OUT_DIR / uni_np_name, uni_only, SR, limit_mode="clip"
    )
    ab_np_entry = write_listening_wav(
        OUT_DIR / ab_np_name, ab_np, SR, limit_mode="clip"
    )

    # Low-volume piano bed (clicks full amp)
    piano_low = (piano_mono * np.float32(PIANO_GAIN_LOW)).astype(np.float32)
    g_tag = f"g{PIANO_GAIN_LOW:.2f}".replace(".", "p")

    kenv_lp = _overlay(piano_low, fused["kenv"], click3)
    o12_lp = _overlay(piano_low, layers["o12"], click3)
    uni_lp = _overlay(piano_low, fused["final"], click3)
    freq_lp = _overlay(piano_low, fused["kenv"], click3)
    freq_lp = _overlay(freq_lp, fused["agree_only"], click5)
    freq_lp = _overlay(freq_lp, fused["o12_deburst_extra"], click2)
    ab_lp = np.column_stack([kenv_lp, o12_lp]).astype(np.float32)

    kenv_lp_name = click_wav_name(
        f"perc_tilt_k_env_adaptive_on_piano_low_{g_tag}", n_kenv
    )
    o12_lp_name = click_wav_name(
        f"lpc_o12_residual_sf_adaptive_on_piano_low_{g_tag}", n_o12
    )
    uni_lp_name = click_wav_name(
        f"fusion_kenv_agree_o12db_on_piano_low_{g_tag}", n_uni
    )
    fs_lp_name = (
        f"fusion_kenv_agree_o12db_on_piano_freqsep_low_{g_tag}"
        f"_클릭_p{n_fs}_5k{n_5k}_2k{n_2k}.wav"
    )
    ab_lp_name = (
        f"kenvAd_on_piano_lowL_o12_on_piano_lowR_{g_tag}_클릭_p{n_kenv}x{n_o12}.wav"
    )

    kenv_lp_entry = write_listening_wav(
        OUT_DIR / kenv_lp_name, kenv_lp, SR, limit_mode="clip"
    )
    o12_lp_entry = write_listening_wav(
        OUT_DIR / o12_lp_name, o12_lp, SR, limit_mode="clip"
    )
    uni_lp_entry = write_listening_wav(
        OUT_DIR / uni_lp_name, uni_lp, SR, limit_mode="clip"
    )
    fs_lp_entry = write_listening_wav(
        OUT_DIR / fs_lp_name, freq_lp, SR, limit_mode="clip"
    )
    ab_lp_entry = write_listening_wav(
        OUT_DIR / ab_lp_name, ab_lp, SR, limit_mode="clip"
    )

    # Conservative: kenv ∪ agree-only (5k×4); exclude o12 deburst (2k×21)
    cons_times = sorted(fused["kenv"] + fused["agree_only"])
    n_cons = len(cons_times)
    cons_uni = _overlay(piano_mono, cons_times, click3)
    cons_freq = _overlay(piano_mono, fused["kenv"], click3)
    cons_freq = _overlay(cons_freq, fused["agree_only"], click5)
    cons_uni_lp = _overlay(piano_low, cons_times, click3)
    cons_freq_lp = _overlay(piano_low, fused["kenv"], click3)
    cons_freq_lp = _overlay(cons_freq_lp, fused["agree_only"], click5)

    cons_uni_name = click_wav_name("fusion_kenv_agree_only_on_piano", n_cons)
    cons_fs_name = (
        f"fusion_kenv_agree_only_on_piano_freqsep_클릭_p{n_cons}_5k{n_5k}.wav"
    )
    cons_uni_lp_name = click_wav_name(
        f"fusion_kenv_agree_only_on_piano_low_{g_tag}", n_cons
    )
    cons_fs_lp_name = (
        f"fusion_kenv_agree_only_on_piano_freqsep_low_{g_tag}"
        f"_클릭_p{n_cons}_5k{n_5k}.wav"
    )

    cons_uni_entry = write_listening_wav(
        OUT_DIR / cons_uni_name, cons_uni, SR, limit_mode="clip"
    )
    cons_fs_entry = write_listening_wav(
        OUT_DIR / cons_fs_name, cons_freq, SR, limit_mode="clip"
    )
    cons_uni_lp_entry = write_listening_wav(
        OUT_DIR / cons_uni_lp_name, cons_uni_lp, SR, limit_mode="clip"
    )
    cons_fs_lp_entry = write_listening_wav(
        OUT_DIR / cons_fs_lp_name, cons_freq_lp, SR, limit_mode="clip"
    )
    print(
        f"  conservative (no 2k): {cons_uni_name} / {cons_fs_name} / "
        f"{cons_uni_lp_name}"
    )

    files = {
        uni_name: {**uni_entry, "role": "unified_3khz_on_piano", "n_peaks": n_uni},
        uni_st_name: {
            **uni_st_entry,
            "role": "stereo_unified_pianoL_dryR",
            "n_peaks": n_uni,
        },
        fs_name: {
            **fs_entry,
            "role": "freqsep_on_piano",
            "n_events": n_fs,
            "n_3k_kenv": n_kenv,
            "n_5k_agree_only": n_5k,
            "n_2k_o12_deburst": n_2k,
            "freqs_hz": {
                "kenv": FREQ_KENV_HZ,
                "agree_only": FREQ_AGREE_ONLY_HZ,
                "o12_deburst": FREQ_O12_DB_HZ,
            },
        },
        fs_st_name: {
            **fs_st_entry,
            "role": "stereo_freqsep_pianoL_dryR",
            "n_events": n_fs,
            "n_5k_agree_only": n_5k,
            "n_2k_o12_deburst": n_2k,
        },
        ab_name: {
            **ab_entry,
            "role": "stereo_unifiedL_freqsepR",
            "n_unified": n_uni,
            "n_freqsep_events": n_fs,
            "n_5k_agree_only": n_5k,
            "n_2k_o12_deburst": n_2k,
        },
        kenv_np_name: {
            **kenv_np_entry,
            "role": "kenv_adaptive_clicks_nopiano",
            "n_peaks": n_kenv,
        },
        o12_np_name: {
            **o12_np_entry,
            "role": "o12_adaptive_clicks_nopiano",
            "n_peaks": n_o12,
        },
        fs_np_name: {
            **fs_np_entry,
            "role": "freqsep_clicks_nopiano",
            "n_events": n_fs,
            "n_5k_agree_only": n_5k,
            "n_2k_o12_deburst": n_2k,
        },
        uni_np_name: {
            **uni_np_entry,
            "role": "unified_clicks_nopiano",
            "n_peaks": n_uni,
        },
        ab_np_name: {
            **ab_np_entry,
            "role": "stereo_kenv_nopianoL_o12_nopianoR",
            "n_kenv": n_kenv,
            "n_o12": n_o12,
            "note": "L=kenv_adaptive 3kHz only; R=o12 3kHz only; no piano",
        },
        kenv_lp_name: {
            **kenv_lp_entry,
            "role": "kenv_adaptive_clicks_lowpiano",
            "n_peaks": n_kenv,
            "piano_gain": PIANO_GAIN_LOW,
        },
        o12_lp_name: {
            **o12_lp_entry,
            "role": "o12_adaptive_clicks_lowpiano",
            "n_peaks": n_o12,
            "piano_gain": PIANO_GAIN_LOW,
        },
        uni_lp_name: {
            **uni_lp_entry,
            "role": "unified_clicks_lowpiano",
            "n_peaks": n_uni,
            "piano_gain": PIANO_GAIN_LOW,
        },
        fs_lp_name: {
            **fs_lp_entry,
            "role": "freqsep_clicks_lowpiano",
            "n_events": n_fs,
            "n_5k_agree_only": n_5k,
            "n_2k_o12_deburst": n_2k,
            "piano_gain": PIANO_GAIN_LOW,
        },
        ab_lp_name: {
            **ab_lp_entry,
            "role": "stereo_kenv_lowpianoL_o12_lowpianoR",
            "n_kenv": n_kenv,
            "n_o12": n_o12,
            "piano_gain": PIANO_GAIN_LOW,
            "note": "L=kenv+low piano; R=o12+low piano; clicks full amp",
        },
        cons_uni_name: {
            **cons_uni_entry,
            "role": "conservative_kenv_agree_only_unified_on_piano",
            "n_peaks": n_cons,
            "n_3k_kenv": n_kenv,
            "n_5k_agree_only": n_5k,
            "excluded": "o12_deburst_2k",
        },
        cons_fs_name: {
            **cons_fs_entry,
            "role": "conservative_kenv_agree_only_freqsep_on_piano",
            "n_events": n_cons,
            "n_3k_kenv": n_kenv,
            "n_5k_agree_only": n_5k,
            "excluded": "o12_deburst_2k",
            "freqs_hz": {"kenv": FREQ_KENV_HZ, "agree_only": FREQ_AGREE_ONLY_HZ},
        },
        cons_uni_lp_name: {
            **cons_uni_lp_entry,
            "role": "conservative_kenv_agree_only_unified_lowpiano",
            "n_peaks": n_cons,
            "piano_gain": PIANO_GAIN_LOW,
            "excluded": "o12_deburst_2k",
        },
        cons_fs_lp_name: {
            **cons_fs_lp_entry,
            "role": "conservative_kenv_agree_only_freqsep_lowpiano",
            "n_events": n_cons,
            "n_5k_agree_only": n_5k,
            "piano_gain": PIANO_GAIN_LOW,
            "excluded": "o12_deburst_2k",
        },
    }

    manifest = {
        "experiment": "fusion_kenv_agree_o12db_on_piano",
        "note": (
            "k_env_adaptive base ∪ LPC-order agreement extras ∪ "
            "o12-only after 100ms chronological deburst; "
            "also frequency-separated listen (3k/5k/2k); "
            "conservative variant = kenv∪agree-only (exclude 2k o12db)."
        ),
        "fixed_rules": {
            "merge_order": [
                "perc_tilt_k_env_adaptive",
                "lpc_order_agreement not within ±30ms",
                "o12 adaptive not in union → chrono 100ms deburst → add ±30ms",
            ],
            "conservative_variant": "kenv ∪ agree_only; exclude o12_deburst_2k",
            "match_tol_s": MATCH_TOL_S,
            "deburst_gap_s": DEBURST_GAP_S,
            "deburst_rule": "chronological_first_survivor",
            "click_unified_hz": FREQ_KENV_HZ,
            "click_freqsep_hz": {
                "kenv": FREQ_KENV_HZ,
                "agree_only": FREQ_AGREE_ONLY_HZ,
                "o12_deburst_extra": FREQ_O12_DB_HZ,
            },
            "click_dur_ms": CLICK_DUR_MS,
            "click_amp": CLICK_AMP,
            "piano_gain_low": PIANO_GAIN_LOW,
            "min_gap_s_sources": MIN_EVENT_GAP_S,
            "piano": "out/stems/Dir/bs_roformer/piano.wav",
            "piano_sha256": sha256_file(SOURCE_PIANO),
            "sources": {
                "kenv": str(KENV_MANIFEST).replace("\\", "/"),
                "agreement": str(AGREE_MANIFEST).replace("\\", "/"),
                "o12": str(O12_MANIFEST).replace("\\", "/"),
            },
            "filename_convention": (
                "unified: click_wav_name → *_클릭_p{N}.wav; "
                "freqsep: *_클릭_p{N}_5k{n5}_2k{n2}.wav; "
                "conservative freqsep: *_클릭_p{N}_5k{n5}.wav"
            ),
            "listen_limit_mode": "clip",
        },
        "piano_stats": audio_stats(piano),
        "counts": {
            **counts,
            "conservative_kenv_agree_only": n_cons,
        },
        "peak_times_s": {
            "kenv": fused["kenv"],
            "agree_only": fused["agree_only"],
            "o12_raw_extra": fused["o12_raw_extra"],
            "o12_deburst_extra": fused["o12_deburst_extra"],
            "final_unified": fused["final"],
            "conservative_kenv_agree_only": cons_times,
        },
        "files": files,
    }
    write_json(OUT_DIR / "fusion_kenv_agree_o12db_on_piano_manifest.json", manifest)

    print(f"  {uni_name}: unified={n_uni}")
    print(f"  {fs_name}: freqsep total={n_fs} (3k={n_kenv} 5k={n_5k} 2k={n_2k})")
    print(f"  nopiano: {kenv_np_name} | {o12_np_name} | {fs_np_name}")
    print(f"  nopiano A/B: {ab_np_name}")
    print(f"  lowpiano g={PIANO_GAIN_LOW}: {ab_lp_name}")
    print(f"  lowpiano freqsep: {fs_lp_name}")
    print(
        "  counts:",
        f"kenv={counts['kenv']}",
        f"agree_only+={counts['agree_only_added']}",
        f"o12_raw_extra={counts['o12_raw_extra']}",
        f"o12_db+={counts['o12_deburst_extra']}",
        f"o12_db_drop={counts['o12_deburst_dropped']}",
    )
    return manifest


def determinism_check(manifest: dict[str, Any]) -> dict[str, Any]:
    first = {n: e["sha256"] for n, e in manifest["files"].items()}
    first_final = list(manifest["peak_times_s"]["final_unified"])
    second = run_once()
    mismatches = [
        n for n in first if first[n] != second["files"].get(n, {}).get("sha256")
    ]
    peak_mismatch = first_final != list(second["peak_times_s"]["final_unified"])
    report = {
        "matched": len(mismatches) == 0 and not peak_mismatch,
        "wav_mismatches": mismatches,
        "peak_mismatch": peak_mismatch,
        "n_unified": len(first_final),
    }
    write_json(OUT_DIR / "fusion_kenv_agree_o12db_on_piano_determinism.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--determinism-check", action="store_true")
    args = parser.parse_args()

    print(f"piano: {SOURCE_PIANO}")
    print(f"output: {OUT_DIR}")
    print(
        f"merge: kenv → agree ±{int(MATCH_TOL_S*1000)}ms → "
        f"o12-extra deburst {int(DEBURST_GAP_S*1000)}ms"
    )
    print(
        f"freqsep: kenv={FREQ_KENV_HZ:.0f} agree_only={FREQ_AGREE_ONLY_HZ:.0f} "
        f"o12db={FREQ_O12_DB_HZ:.0f}"
    )

    manifest = run_once()
    if args.determinism_check:
        print("determinism-check: second run…")
        report = determinism_check(manifest)
        if not report["matched"]:
            raise SystemExit(f"determinism failed: {report}")
        print("determinism-check: OK")


if __name__ == "__main__":
    main()
