"""Union_506∪395 + consensus-missed-3 on low piano.

Full-track and concatenated excerpts around ~21s and ~1:32.
- unified: all events 3 kHz
- freqsep: union 3 kHz / missed-3 5 kHz
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
for p in (HERE, S4, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

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
ONP = OUTPUT_DIR / "pass2" / "lpc_sf_adaptive_on_piano"
COV = OUTPUT_DIR / "pass2" / "consensus_coverage"
CMP_MANIFEST = ONP / "cmp506_vs_395_lowpiano_manifest.json"
MISS_MANIFEST = COV / "union506_395_consensus_missed_manifest.json"

PIANO_GAIN_LOW = 0.20
CLICK_DUR_MS = 12.0
CLICK_AMP = 0.7
FREQ_UNION_HZ = 3000.0
FREQ_MISS_HZ = 5000.0

# Excerpt windows (inclusive of both ~21s events; 1:32 event)
EXCERPT_PAD_S = 2.5
EXCERPT_REGIONS = (
    (21.141769 - EXCERPT_PAD_S, 21.524898 + EXCERPT_PAD_S),  # ~18.64–24.02
    (92.815964 - EXCERPT_PAD_S, 92.815964 + EXCERPT_PAD_S),  # ~90.32–95.32
)
GAP_S = 0.5  # silence between concatenated excerpts


def _click(freq_hz: float) -> np.ndarray:
    n = int(SR * CLICK_DUR_MS / 1000.0)
    t = np.arange(n, dtype=np.float32) / SR
    env = np.exp(-t * 1000.0 / CLICK_DUR_MS)
    return (CLICK_AMP * env * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _overlay(mono: np.ndarray, times: list[float], click: np.ndarray) -> np.ndarray:
    out = mono.astype(np.float32, copy=True)
    n_audio = len(out)
    for t in times:
        idx = int(float(t) * SR)
        if idx >= n_audio or idx < 0:
            continue
        end = min(idx + len(click), n_audio)
        n = end - idx
        if n > 0:
            out[idx:end] += click[:n]
    return out


def _slice_region(
    audio: np.ndarray, t0: float, t1: float
) -> tuple[np.ndarray, float]:
    i0 = max(0, int(t0 * SR))
    i1 = min(len(audio), int(t1 * SR))
    return audio[i0:i1].astype(np.float32, copy=True), float(i0 / SR)


def _concat_excerpts(full: np.ndarray) -> np.ndarray:
    parts: list[np.ndarray] = []
    gap = np.zeros(int(GAP_S * SR), dtype=np.float32)
    for i, (t0, t1) in enumerate(EXCERPT_REGIONS):
        seg, _ = _slice_region(full, t0, t1)
        if i > 0:
            parts.append(gap)
        parts.append(seg)
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def run_once() -> dict[str, Any]:
    if not SOURCE_PIANO.exists():
        raise FileNotFoundError(SOURCE_PIANO)
    if not CMP_MANIFEST.exists():
        raise FileNotFoundError(CMP_MANIFEST)
    if not MISS_MANIFEST.exists():
        raise FileNotFoundError(MISS_MANIFEST)

    piano, sr = read_stereo(SOURCE_PIANO)
    if sr != SR:
        raise RuntimeError(f"sr={sr}")
    piano_mono = piano.mean(axis=1).astype(np.float32)
    piano_low = (piano_mono * np.float32(PIANO_GAIN_LOW)).astype(np.float32)
    g_tag = f"g{PIANO_GAIN_LOW:.2f}".replace(".", "p")

    cmp_data = json.loads(CMP_MANIFEST.read_text(encoding="utf-8"))
    miss_data = json.loads(MISS_MANIFEST.read_text(encoding="utf-8"))
    union = [float(t) for t in cmp_data["peak_times_s"]["union_unified"]]
    missed = [float(t) for t in miss_data["peak_times_s"]]
    if len(missed) != 3:
        raise RuntimeError(f"expected 3 missed, got {len(missed)}")

    # Apply missed-3 onto union (no dedupe needed: they are unmatched)
    combined_times = sorted(union + missed)
    n_u = len(union)
    n_m = len(missed)
    n_all = len(combined_times)

    c3 = _click(FREQ_UNION_HZ)
    c5 = _click(FREQ_MISS_HZ)

    unified = _overlay(piano_low, combined_times, c3)
    freqsep = _overlay(piano_low, union, c3)
    freqsep = _overlay(freqsep, missed, c5)

    ONP.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}

    uni_name = click_wav_name(
        f"union506_395_plus_miss3_low_{g_tag}_unified3k", n_all
    )
    fs_name = (
        f"union506_395_plus_miss3_low_{g_tag}_freqsep"
        f"_클릭_p{n_all}_u{n_u}_m{n_m}.wav"
    )
    uni_ex_name = (
        f"union506_395_plus_miss3_low_{g_tag}_unified3k"
        f"_excerpt21_132_클릭_p{n_all}.wav"
    )
    fs_ex_name = (
        f"union506_395_plus_miss3_low_{g_tag}_freqsep"
        f"_excerpt21_132_클릭_p{n_all}_u{n_u}_m{n_m}.wav"
    )

    uni_ex = _concat_excerpts(unified)
    fs_ex = _concat_excerpts(freqsep)

    for name, audio, role in (
        (uni_name, unified, "full_unified_lowpiano"),
        (fs_name, freqsep, "full_freqsep_lowpiano"),
        (uni_ex_name, uni_ex, "excerpt21_132_unified_lowpiano"),
        (fs_ex_name, fs_ex, "excerpt21_132_freqsep_lowpiano"),
    ):
        entry = write_listening_wav(ONP / name, audio, SR, limit_mode="clip")
        files[name] = {
            **entry,
            "role": role,
            "n_peaks_source": n_all,
            "n_union": n_u,
            "n_missed_added": n_m,
        }
        print(f"  {name}")

    manifest = {
        "experiment": "union506_395_plus_consensus_miss3_lowpiano",
        "note": (
            "union_506∪395 + 3 stem-consensus misses; "
            "full track + concatenated excerpts (~21s, ~1:32); "
            "unified 3kHz and freqsep (union 3k / miss 5k); low piano only."
        ),
        "fixed_rules": {
            "piano_gain_low": PIANO_GAIN_LOW,
            "freqs_hz": {"union": FREQ_UNION_HZ, "missed": FREQ_MISS_HZ},
            "excerpt_pad_s": EXCERPT_PAD_S,
            "excerpt_regions_s": [
                {"t0": a, "t1": b} for a, b in EXCERPT_REGIONS
            ],
            "excerpt_gap_s": GAP_S,
            "click_dur_ms": CLICK_DUR_MS,
            "click_amp": CLICK_AMP,
            "listen_limit_mode": "clip",
            "piano_sha256": sha256_file(SOURCE_PIANO),
            "sources": {
                "union": str(CMP_MANIFEST).replace("\\", "/"),
                "missed": str(MISS_MANIFEST).replace("\\", "/"),
            },
        },
        "counts": {
            "union": n_u,
            "missed_added": n_m,
            "union_plus_miss": n_all,
        },
        "peak_times_s": {
            "union": union,
            "missed": missed,
            "union_plus_miss": combined_times,
        },
        "missed_mmss": [
            f"{int(t // 60)}:{t - 60 * int(t // 60):06.3f}" for t in missed
        ],
        "piano_stats": audio_stats(piano),
        "files": files,
    }
    write_json(ONP / "union506_395_plus_miss3_lowpiano_manifest.json", manifest)
    return manifest


def determinism_check(manifest: dict[str, Any]) -> dict[str, Any]:
    first = {n: e["sha256"] for n, e in manifest["files"].items()}
    second = run_once()
    mismatches = [
        n for n in first if first[n] != second["files"].get(n, {}).get("sha256")
    ]
    report = {
        "matched": len(mismatches) == 0,
        "wav_mismatches": mismatches,
        "n": manifest["counts"]["union_plus_miss"],
    }
    write_json(ONP / "union506_395_plus_miss3_lowpiano_determinism.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--determinism-check", action="store_true")
    args = parser.parse_args()
    print(f"piano: {SOURCE_PIANO}")
    print(f"output: {ONP}")
    print(
        f"union+miss3; excerpts pad={EXCERPT_PAD_S}s "
        f"regions={EXCERPT_REGIONS}; low g={PIANO_GAIN_LOW}"
    )
    manifest = run_once()
    print("counts:", manifest["counts"])
    if args.determinism_check:
        print("determinism-check: second run…")
        report = determinism_check(manifest)
        if not report["matched"]:
            raise SystemExit(f"determinism failed: {report}")
        print("determinism-check: OK")


if __name__ == "__main__":
    main()
