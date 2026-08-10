"""506 vs Dir 전체_adaptive — unified + freqsep (506 = higher Hz), low piano.

전체_adaptive peaks: recompute SuperFlux+peaks_adaptive on 102-Dir.wav
(same as src/exp/s3_2d/_sonify_adaptive.py).
506: fusion_kenv_agree_only (conservative).

- unified: all events 3 kHz (union ±30ms merged)
- freqsep: adaptive 3 kHz / 506 5 kHz (506 higher)
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

from _onset_wtmm_fusion import one_to_one_time_match  # noqa: E402
from audio_io import duration_s, load_mono  # noqa: E402
from config import SR  # noqa: E402
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

ROOT = HERE.parents[3]
DIR_AUDIO = ROOT / "audio" / "102 - Dir.wav"
REF_ADAPTIVE_WAV = ROOT / "out" / "sonify" / "Dir" / "전체_adaptive_클릭.wav"
FUSION_MANIFEST = (
    OUTPUT_DIR
    / "pass2"
    / "lpc_sf_adaptive_on_piano"
    / "fusion_kenv_agree_o12db_on_piano_manifest.json"
)
OUT_DIR = OUTPUT_DIR / "pass2" / "lpc_sf_adaptive_on_piano"

MATCH_TOL_S = 0.03
PIANO_GAIN_LOW = 0.20
CLICK_DUR_MS = 12.0
CLICK_AMP = 0.7
FREQ_COMMON_HZ = 3000.0
FREQ_ADAPTIVE_HZ = 3000.0
FREQ_506_HZ = 5000.0  # higher


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


def load_adaptive_peaks() -> np.ndarray:
    if not DIR_AUDIO.exists():
        raise FileNotFoundError(DIR_AUDIO)
    mono = load_mono(DIR_AUDIO)
    dur = duration_s(DIR_AUDIO)
    env, times = superflux_envelope(mono)
    bands = band_envelopes(mono)
    pk = peaks_adaptive(env, times, bands, dur)
    return np.asarray(pk, dtype=np.float64)


def load_506() -> np.ndarray:
    if not FUSION_MANIFEST.exists():
        raise FileNotFoundError(FUSION_MANIFEST)
    data = json.loads(FUSION_MANIFEST.read_text(encoding="utf-8"))
    key = "conservative_kenv_agree_only"
    times = data["peak_times_s"][key]
    arr = np.asarray(times, dtype=np.float64)
    if len(arr) != 506:
        raise RuntimeError(f"expected 506, got {len(arr)}")
    return arr


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

    p506 = load_506()
    p_ad = load_adaptive_peaks()
    n_506 = int(len(p506))
    n_ad = int(len(p_ad))

    common, only_506, only_ad = one_to_one_time_match(p506, p_ad)
    common_t = [float(t) for t in common]
    only_506_t = [float(t) for t in only_506]
    only_ad_t = [float(t) for t in only_ad]
    union_t = sorted(common_t + only_506_t + only_ad_t)
    n_c, n_6o, n_ao = len(common_t), len(only_506_t), len(only_ad_t)
    n_u = len(union_t)

    c3 = _click(FREQ_COMMON_HZ)
    c5 = _click(FREQ_506_HZ)

    unified = _overlay(piano_low, union_t, c3)
    # freqsep: adaptive layer 3k (all adaptive times), 506 layer 5k (all 506)
    freqsep = _overlay(piano_low, [float(t) for t in p_ad], c3)
    freqsep = _overlay(freqsep, [float(t) for t in p506], c5)

    tag = f"cmp506_vs_dirAdaptive_low_{g_tag}"
    uni_name = (
        f"{tag}_unified3k_클릭_p{n_u}_c{n_c}_6o{n_6o}_ao{n_ao}.wav"
    )
    fs_name = (
        f"{tag}_freqsep_클릭_p{n_u}_ad{n_ad}_506_{n_506}.wav"
    )
    # also role solos for scrub (optional but useful)
    common_name = click_wav_name(f"{tag}_common3k", n_c)
    only506_name = click_wav_name(f"{tag}_506only5k", n_6o)
    only_ad_name = click_wav_name(f"{tag}_adaptiveOnly3k", n_ao)

    files: dict[str, Any] = {}
    for name, audio, role, n in (
        (uni_name, unified, "unified_3khz_union_lowpiano", n_u),
        (fs_name, freqsep, "freqsep_adaptive3k_506_5k_lowpiano", n_u),
        (
            common_name,
            _overlay(piano_low, common_t, c3),
            "common_solo_lowpiano",
            n_c,
        ),
        (
            only506_name,
            _overlay(piano_low, only_506_t, c5),
            "506only_solo_lowpiano",
            n_6o,
        ),
        (
            only_ad_name,
            _overlay(piano_low, only_ad_t, c3),
            "adaptive_only_solo_lowpiano",
            n_ao,
        ),
    ):
        entry = write_listening_wav(OUT_DIR / name, audio, SR, limit_mode="clip")
        files[name] = {**entry, "role": role, "n": n}
        print(f"  {name}")

    stereo = np.column_stack([unified, freqsep]).astype(np.float32)
    st_name = (
        f"{tag}_unifiedL_freqsepR_클릭_p{n_u}_c{n_c}_6o{n_6o}_ao{n_ao}.wav"
    )
    st_entry = write_listening_wav(OUT_DIR / st_name, stereo, SR, limit_mode="clip")
    files[st_name] = {**st_entry, "role": "stereo_unifiedL_freqsepR", "n": n_u}

    manifest = {
        "experiment": "cmp506_vs_dir_전체_adaptive_lowpiano",
        "note": (
            "506 (fusion_kenv_agree_only) vs Dir 전체_adaptive "
            "(SuperFlux+peaks_adaptive on 102-Dir); "
            "unified 3kHz union; freqsep adaptive=3kHz / 506=5kHz; low piano."
        ),
        "fixed_rules": {
            "match_tol_s": MATCH_TOL_S,
            "piano_gain_low": PIANO_GAIN_LOW,
            "freqs_hz": {
                "unified": FREQ_COMMON_HZ,
                "adaptive_layer": FREQ_ADAPTIVE_HZ,
                "506_layer": FREQ_506_HZ,
            },
            "adaptive_detector": "superflux_envelope + band_envelopes + peaks_adaptive",
            "adaptive_source_audio": str(DIR_AUDIO).replace("\\", "/"),
            "adaptive_reference_wav": str(REF_ADAPTIVE_WAV).replace("\\", "/"),
            "506_source": str(FUSION_MANIFEST).replace("\\", "/")
            + " → conservative_kenv_agree_only",
            "piano_sha256": sha256_file(SOURCE_PIANO),
            "listen_limit_mode": "clip",
        },
        "counts": {
            "n_506": n_506,
            "n_adaptive": n_ad,
            "common": n_c,
            "only_506": n_6o,
            "only_adaptive": n_ao,
            "union": n_u,
        },
        "peak_times_s": {
            "p506": [float(t) for t in p506],
            "adaptive": [float(t) for t in p_ad],
            "common": common_t,
            "only_506": only_506_t,
            "only_adaptive": only_ad_t,
            "union": union_t,
        },
        "piano_stats": audio_stats(piano),
        "files": files,
    }
    write_json(OUT_DIR / "cmp506_vs_dirAdaptive_lowpiano_manifest.json", manifest)
    print(
        f"  counts: ad={n_ad} 506={n_506} common={n_c} "
        f"6only={n_6o} ad_only={n_ao} union={n_u}"
    )
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
    write_json(OUT_DIR / "cmp506_vs_dirAdaptive_lowpiano_determinism.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--determinism-check", action="store_true")
    args = parser.parse_args()
    print(f"adaptive audio: {DIR_AUDIO}")
    print(f"ref wav: {REF_ADAPTIVE_WAV.exists()} {REF_ADAPTIVE_WAV}")
    print(f"output: {OUT_DIR}")
    print(f"freqsep: adaptive={FREQ_ADAPTIVE_HZ:.0f} 506={FREQ_506_HZ:.0f}")
    manifest = run_once()
    if args.determinism_check:
        print("determinism-check: second run…")
        report = determinism_check(manifest)
        if not report["matched"]:
            raise SystemExit(f"determinism failed: {report}")
        print("determinism-check: OK")


if __name__ == "__main__":
    main()
