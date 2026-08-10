"""506 vs 395 role sonify on low piano (D-v2-04).

Roles (±30ms one-to-one):
  common     → 3 kHz
  506-only   → 5 kHz
  395-only   → 1.5 kHz

Outputs (low piano only, gain fixed):
  - combined freqsep (all three roles)
  - each role solo
  - stereo L=combined / R=low piano dry

506 = fusion_kenv_agree_only (kenv∪agree-only, no o12db).
395 = a2_posdist_rescue from posdist_metrics.json.
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
S4 = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(S4) not in sys.path:
    sys.path.insert(0, str(S4))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from _onset_wtmm_fusion import one_to_one_time_match  # noqa: E402
from config import SR  # noqa: E402
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

ROOT = HERE.parents[3]
PASS2 = OUTPUT_DIR / "pass2"
OUT_DIR = PASS2 / "lpc_sf_adaptive_on_piano"
FUSION_MANIFEST = OUT_DIR / "fusion_kenv_agree_o12db_on_piano_manifest.json"
POSDIST_METRICS = ROOT / "out" / "sonify" / "Dir" / "posdist_metrics.json"

MATCH_TOL_S = 0.03
PIANO_GAIN_LOW = 0.20
CLICK_DUR_MS = 12.0
CLICK_AMP = 0.7

FREQ_COMMON_HZ = 3000.0
FREQ_506_ONLY_HZ = 5000.0
FREQ_395_ONLY_HZ = 1500.0


def _click(freq_hz: float) -> np.ndarray:
    n = int(SR * CLICK_DUR_MS / 1000.0)
    t = np.arange(n, dtype=np.float32) / SR
    env = np.exp(-t * 1000.0 / CLICK_DUR_MS)
    return (CLICK_AMP * env * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _overlay(mono: np.ndarray, times: list[float] | np.ndarray, click: np.ndarray) -> np.ndarray:
    out = mono.astype(np.float32, copy=True)
    for t in times:
        idx = int(float(t) * SR)
        end = min(idx + len(click), len(out))
        n = end - idx
        if n > 0:
            out[idx:end] += click[:n]
    return out


def _load_peaks() -> tuple[np.ndarray, np.ndarray]:
    if not FUSION_MANIFEST.exists():
        raise FileNotFoundError(FUSION_MANIFEST)
    if not POSDIST_METRICS.exists():
        raise FileNotFoundError(POSDIST_METRICS)
    fusion = json.loads(FUSION_MANIFEST.read_text(encoding="utf-8"))
    key = "conservative_kenv_agree_only"
    if key not in fusion.get("peak_times_s", {}):
        raise KeyError(f"{FUSION_MANIFEST.name}: missing peak_times_s[{key}]")
    p506 = np.asarray(fusion["peak_times_s"][key], dtype=np.float64)
    if len(p506) != 506:
        raise RuntimeError(f"expected 506 peaks, got {len(p506)}")

    posdist = json.loads(POSDIST_METRICS.read_text(encoding="utf-8"))
    p395 = np.asarray(
        posdist["peak_times_s"]["a2_posdist_rescue"], dtype=np.float64
    )
    if len(p395) != 395:
        raise RuntimeError(f"expected 395 peaks, got {len(p395)}")
    return p506, p395


def run_once() -> dict[str, Any]:
    if not SOURCE_PIANO.exists():
        raise FileNotFoundError(SOURCE_PIANO)
    piano, sr = read_stereo(SOURCE_PIANO)
    if sr != SR:
        raise RuntimeError(f"sr={sr}")
    piano_mono = piano.mean(axis=1).astype(np.float32)
    piano_low = (piano_mono * np.float32(PIANO_GAIN_LOW)).astype(np.float32)
    g_tag = f"g{PIANO_GAIN_LOW:.2f}".replace(".", "p")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    p506, p395 = _load_peaks()
    common, only_506_from_ref, only_395 = one_to_one_time_match(p506, p395)
    # one_to_one_time_match(a,b) → common(from a), a_only, b_only
    common_t = [float(t) for t in common]
    only_506 = [float(t) for t in only_506_from_ref]
    only_395_t = [float(t) for t in only_395]

    n_c = len(common_t)
    n_6 = len(only_506)
    n_3 = len(only_395_t)
    n_events = n_c + n_6 + n_3

    c3 = _click(FREQ_COMMON_HZ)
    c5 = _click(FREQ_506_ONLY_HZ)
    c15 = _click(FREQ_395_ONLY_HZ)

    combined = _overlay(piano_low, common_t, c3)
    combined = _overlay(combined, only_506, c5)
    combined = _overlay(combined, only_395_t, c15)

    solo_common = _overlay(piano_low, common_t, c3)
    solo_506 = _overlay(piano_low, only_506, c5)
    solo_395 = _overlay(piano_low, only_395_t, c15)

    # Unified 3kHz on union (common + 506-only + 395-only)
    union_t = sorted(common_t + only_506 + only_395_t)
    unified = _overlay(piano_low, union_t, c3)

    tag = f"cmp506_vs_395_low_{g_tag}"
    combined_name = (
        f"{tag}_freqsep_클릭_p{n_events}_c{n_c}_6o{n_6}_3o{n_3}.wav"
    )
    unified_name = (
        f"{tag}_unified3k_클릭_p{n_events}_c{n_c}_6o{n_6}_3o{n_3}.wav"
    )
    common_name = click_wav_name(f"{tag}_common3k", n_c)
    only506_name = click_wav_name(f"{tag}_506only5k", n_6)
    only395_name = click_wav_name(f"{tag}_395only1p5k", n_3)
    stereo_name = (
        f"{tag}_freqsepL_pianodryR_클릭_p{n_events}_c{n_c}_6o{n_6}_3o{n_3}.wav"
    )
    stereo_uni_name = (
        f"{tag}_unifiedL_freqsepR_클릭_p{n_events}_c{n_c}_6o{n_6}_3o{n_3}.wav"
    )

    files: dict[str, Any] = {}
    for name, audio, role, n in (
        (combined_name, combined, "freqsep_all_roles_lowpiano", n_events),
        (unified_name, unified, "unified_3khz_union_lowpiano", n_events),
        (common_name, solo_common, "common_3khz_solo_lowpiano", n_c),
        (only506_name, solo_506, "only506_5khz_solo_lowpiano", n_6),
        (only395_name, solo_395, "only395_1p5khz_solo_lowpiano", n_3),
    ):
        entry = write_listening_wav(OUT_DIR / name, audio, SR, limit_mode="clip")
        files[name] = {**entry, "role": role, "n": n}

    stereo = np.column_stack([combined, piano_low]).astype(np.float32)
    st_entry = write_listening_wav(
        OUT_DIR / stereo_name, stereo, SR, limit_mode="clip"
    )
    files[stereo_name] = {
        **st_entry,
        "role": "stereo_freqsepL_lowpiano_dryR",
        "n_events": n_events,
    }
    stereo_uf = np.column_stack([unified, combined]).astype(np.float32)
    st_uf_entry = write_listening_wav(
        OUT_DIR / stereo_uni_name, stereo_uf, SR, limit_mode="clip"
    )
    files[stereo_uni_name] = {
        **st_uf_entry,
        "role": "stereo_unifiedL_freqsepR_lowpiano",
        "n_events": n_events,
    }

    manifest = {
        "experiment": "cmp506_vs_395_lowpiano_roles",
        "note": (
            "506 (fusion_kenv_agree_only) vs 395 (a2_posdist_rescue); "
            "roles ±30ms; low piano only."
        ),
        "fixed_rules": {
            "match_tol_s": MATCH_TOL_S,
            "piano_gain_low": PIANO_GAIN_LOW,
            "freqs_hz": {
                "common": FREQ_COMMON_HZ,
                "only_506": FREQ_506_ONLY_HZ,
                "only_395": FREQ_395_ONLY_HZ,
            },
            "click_dur_ms": CLICK_DUR_MS,
            "click_amp": CLICK_AMP,
            "listen_limit_mode": "clip",
            "piano": "out/stems/Dir/bs_roformer/piano.wav",
            "piano_sha256": sha256_file(SOURCE_PIANO),
            "sources": {
                "506": str(FUSION_MANIFEST).replace("\\", "/")
                + " → peak_times_s.conservative_kenv_agree_only",
                "395": str(POSDIST_METRICS).replace("\\", "/")
                + " → peak_times_s.a2_posdist_rescue",
            },
        },
        "piano_stats": audio_stats(piano),
        "counts": {
            "n_506": int(len(p506)),
            "n_395": int(len(p395)),
            "common": n_c,
            "only_506": n_6,
            "only_395": n_3,
            "freqsep_events": n_events,
        },
        "peak_times_s": {
            "common": common_t,
            "only_506": only_506,
            "only_395": only_395_t,
            "union_unified": union_t,
        },
        "files": files,
    }
    write_json(OUT_DIR / "cmp506_vs_395_lowpiano_manifest.json", manifest)

    print(f"  common={n_c}  only506={n_6}  only395={n_3}")
    print(f"  {combined_name}")
    print(f"  {unified_name}")
    print(f"  {common_name}")
    print(f"  {only506_name}")
    print(f"  {only395_name}")
    return manifest


def determinism_check(manifest: dict[str, Any]) -> dict[str, Any]:
    first = {n: e["sha256"] for n, e in manifest["files"].items()}
    first_counts = dict(manifest["counts"])
    second = run_once()
    mismatches = [
        n for n in first if first[n] != second["files"].get(n, {}).get("sha256")
    ]
    report = {
        "matched": len(mismatches) == 0 and first_counts == second["counts"],
        "wav_mismatches": mismatches,
        "counts_mismatch": first_counts != second["counts"],
    }
    write_json(OUT_DIR / "cmp506_vs_395_lowpiano_determinism.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--determinism-check", action="store_true")
    args = parser.parse_args()

    print(f"piano: {SOURCE_PIANO}")
    print(f"output: {OUT_DIR}")
    print(
        f"roles: common={FREQ_COMMON_HZ:.0f} only506={FREQ_506_ONLY_HZ:.0f} "
        f"only395={FREQ_395_ONLY_HZ:.0f}; low g={PIANO_GAIN_LOW}"
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
