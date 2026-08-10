"""506 vs Dir 전체_adaptive on original mix — raw + LUFS beds.

Peaks from cmp506_vs_dirAdaptive_lowpiano_manifest.json.
- unified: union @ 3 kHz
- freqsep: adaptive @ 3 kHz / 506 @ 5 kHz

Beds (mono mean of 102-Dir):
- raw: no LUFS (playback level ≈ original file)
- lufs: audio_io.load_mono TARGET_LUFS normalize
Each × gain 1.0 and 0.20.
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

from audio_io import load_mono, read_raw  # noqa: E402
from config import SR, TARGET_LUFS  # noqa: E402
from io_util import (  # noqa: E402
    OUTPUT_DIR,
    audio_stats,
    sha256_file,
    write_json,
    write_listening_wav,
)

ROOT = HERE.parents[3]
DIR_AUDIO = ROOT / "audio" / "102 - Dir.wav"
CMP_MANIFEST = (
    OUTPUT_DIR
    / "pass2"
    / "lpc_sf_adaptive_on_piano"
    / "cmp506_vs_dirAdaptive_lowpiano_manifest.json"
)
OUT_DIR = OUTPUT_DIR / "pass2" / "lpc_sf_adaptive_on_piano"

GAIN_FULL = 1.0
GAIN_LOW = 0.20
CLICK_DUR_MS = 12.0
CLICK_AMP = 0.7
FREQ_ADAPTIVE_HZ = 3000.0
FREQ_506_HZ = 5000.0
FREQ_UNIFIED_HZ = 3000.0


def _click(freq_hz: float) -> np.ndarray:
    n = int(SR * CLICK_DUR_MS / 1000.0)
    t = np.arange(n, dtype=np.float32) / SR
    env = np.exp(-t * 1000.0 / CLICK_DUR_MS)
    return (CLICK_AMP * env * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _overlay(mono: np.ndarray, times: list[float], click: np.ndarray) -> np.ndarray:
    out = mono.astype(np.float32, copy=True)
    for t in times:
        idx = int(float(t) * SR)
        end = min(idx + len(click), len(out))
        n = end - idx
        if n > 0:
            out[idx:end] += click[:n]
    return out


def _g_tag(g: float) -> str:
    return f"g{g:.2f}".replace(".", "p")


def run_once() -> dict[str, Any]:
    if not DIR_AUDIO.exists():
        raise FileNotFoundError(DIR_AUDIO)
    if not CMP_MANIFEST.exists():
        raise FileNotFoundError(CMP_MANIFEST)

    L, R = read_raw(DIR_AUDIO)
    mix_raw = (0.5 * (L + R)).astype(np.float32)
    mix_lufs = load_mono(DIR_AUDIO).astype(np.float32)

    cmp_data = json.loads(CMP_MANIFEST.read_text(encoding="utf-8"))
    pts = cmp_data["peak_times_s"]
    counts = cmp_data["counts"]
    adaptive = [float(t) for t in pts["adaptive"]]
    p506 = [float(t) for t in pts["p506"]]
    union = [float(t) for t in pts["union"]]
    n_ad = int(counts["n_adaptive"])
    n_506 = int(counts["n_506"])
    n_u = int(counts["union"])
    n_c = int(counts["common"])
    n_6o = int(counts["only_506"])
    n_ao = int(counts["only_adaptive"])

    c3 = _click(FREQ_UNIFIED_HZ)
    c5 = _click(FREQ_506_HZ)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}

    beds = (
        ("raw", mix_raw, "raw_mono_mean_no_lufs"),
        ("lufs", mix_lufs, f"load_mono_TARGET_LUFS_{TARGET_LUFS}"),
    )
    gains = (GAIN_FULL, GAIN_LOW)

    for level_tag, mix, level_note in beds:
        for gain in gains:
            bed = (mix * np.float32(gain)).astype(np.float32)
            gtag = _g_tag(gain)
            unified = _overlay(bed, union, c3)
            freqsep = _overlay(bed, adaptive, c3)
            freqsep = _overlay(freqsep, p506, c5)

            uni_name = (
                f"cmp506_vs_dirAdaptive_origmix_{level_tag}_{gtag}_unified3k"
                f"_클릭_p{n_u}_c{n_c}_6o{n_6o}_ao{n_ao}.wav"
            )
            fs_name = (
                f"cmp506_vs_dirAdaptive_origmix_{level_tag}_{gtag}_freqsep"
                f"_클릭_p{n_u}_ad{n_ad}_506_{n_506}.wav"
            )
            gain_label = "full" if gain >= 0.99 else "low"
            for name, audio, role in (
                (
                    uni_name,
                    unified,
                    f"unified_origmix_{level_tag}_{gain_label}",
                ),
                (
                    fs_name,
                    freqsep,
                    f"freqsep_origmix_{level_tag}_{gain_label}",
                ),
            ):
                entry = write_listening_wav(
                    OUT_DIR / name, audio, SR, limit_mode="clip"
                )
                files[name] = {
                    **entry,
                    "role": role,
                    "bed": "102-Dir",
                    "bed_level": level_note,
                    "bed_gain": gain,
                    "n_union": n_u,
                }
                print(f"  {name}")

    # Compatibility aliases: untagged origmix_g* == raw (previous fix)
    for gain in gains:
        gtag = _g_tag(gain)
        for kind in ("unified3k", "freqsep"):
            if kind == "unified3k":
                src = (
                    f"cmp506_vs_dirAdaptive_origmix_raw_{gtag}_unified3k"
                    f"_클릭_p{n_u}_c{n_c}_6o{n_6o}_ao{n_ao}.wav"
                )
                alias = (
                    f"cmp506_vs_dirAdaptive_origmix_{gtag}_unified3k"
                    f"_클릭_p{n_u}_c{n_c}_6o{n_6o}_ao{n_ao}.wav"
                )
            else:
                src = (
                    f"cmp506_vs_dirAdaptive_origmix_raw_{gtag}_freqsep"
                    f"_클릭_p{n_u}_ad{n_ad}_506_{n_506}.wav"
                )
                alias = (
                    f"cmp506_vs_dirAdaptive_origmix_{gtag}_freqsep"
                    f"_클릭_p{n_u}_ad{n_ad}_506_{n_506}.wav"
                )
            src_path = OUT_DIR / src
            alias_path = OUT_DIR / alias
            if src_path.exists():
                alias_path.write_bytes(src_path.read_bytes())
                files[alias] = {
                    **files[src],
                    "role": files[src]["role"] + "_alias_of_raw",
                    "alias_of": src,
                }
                print(f"  alias {alias} -> {src}")

    manifest = {
        "experiment": "cmp506_vs_dirAdaptive_on_original_mix",
        "note": (
            "Same peaks as cmp506_vs_dirAdaptive_lowpiano; "
            "origmix beds: raw (no LUFS) and lufs (load_mono); "
            "each at gain 1.0 and 0.20; unified 3k / freqsep ad3k+506 5k. "
            "Untagged origmix_g* filenames are aliases of raw."
        ),
        "fixed_rules": {
            "bed": str(DIR_AUDIO).replace("\\", "/"),
            "bed_sha256": sha256_file(DIR_AUDIO),
            "bed_levels": ["raw", "lufs"],
            "target_lufs": TARGET_LUFS,
            "gains": [GAIN_FULL, GAIN_LOW],
            "freqs_hz": {
                "unified": FREQ_UNIFIED_HZ,
                "adaptive": FREQ_ADAPTIVE_HZ,
                "506": FREQ_506_HZ,
            },
            "peaks_from": str(CMP_MANIFEST).replace("\\", "/"),
            "listen_limit_mode": "clip",
        },
        "counts": counts,
        "mix_stats": {
            "raw": audio_stats(mix_raw),
            "lufs": audio_stats(mix_lufs),
        },
        "files": files,
    }
    write_json(OUT_DIR / "cmp506_vs_dirAdaptive_original_manifest.json", manifest)
    return manifest


def determinism_check(manifest: dict[str, Any]) -> dict[str, Any]:
    first = {n: e["sha256"] for n, e in manifest["files"].items()}
    second = run_once()
    mismatches = [
        n for n in first if first[n] != second["files"].get(n, {}).get("sha256")
    ]
    report = {"matched": len(mismatches) == 0, "wav_mismatches": mismatches}
    write_json(
        OUT_DIR / "cmp506_vs_dirAdaptive_original_determinism.json", report
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--determinism-check", action="store_true")
    args = parser.parse_args()
    print(f"bed: {DIR_AUDIO}")
    print(f"TARGET_LUFS={TARGET_LUFS}")
    print(f"peaks: {CMP_MANIFEST}")
    manifest = run_once()
    if args.determinism_check:
        print("determinism-check: second run…")
        report = determinism_check(manifest)
        if not report["matched"]:
            raise SystemExit(f"determinism failed: {report}")
        print("determinism-check: OK")


if __name__ == "__main__":
    main()
