"""Sonify locked stem-consensus 234 (session-10 metrics).

Meaningful listening set (not just one unified overlay):
  - unified 3 kHz on low piano / origmix LUFS
  - vote freqsep: 3-of-3 → 3 kHz, 2-of-3 → 5 kHz
  - vs764 coverage: covered → 3 kHz, missed-by-764 → 5 kHz
  - missed-by-764 alone (n=2)
  - markdown timestamp table

Reference: out/sonify/Dir/stem_consensus_metrics.json (locked).
Does not recompute stem detectors.
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
from audio_io import load_mono, read_raw  # noqa: E402
from config import SR, TARGET_LUFS  # noqa: E402
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
CONSENSUS_METRICS = ROOT / "out" / "sonify" / "Dir" / "stem_consensus_metrics.json"
UNION764_MANIFEST = (
    OUTPUT_DIR
    / "pass2"
    / "lpc_sf_adaptive_on_piano"
    / "cmp506_vs_dirAdaptive_lowpiano_manifest.json"
)
OUT_DIR = OUTPUT_DIR / "pass2" / "consensus_coverage"
ONP_DIR = OUTPUT_DIR / "pass2" / "lpc_sf_adaptive_on_piano"

MATCH_TOL_S = 0.03
PIANO_GAIN_LOW = 0.20
CLICK_DUR_MS = 12.0
CLICK_AMP = 0.7
FREQ_UNIFIED_HZ = 3000.0
FREQ_VOTE3_HZ = 3000.0
FREQ_VOTE2_HZ = 5000.0
FREQ_COVERED_HZ = 3000.0
FREQ_MISSED_HZ = 5000.0
MODELS = ("bs_roformer", "spleeter", "demucs")


def _fmt(t: float) -> str:
    m = int(t // 60)
    s = t - 60 * m
    return f"{m}:{s:06.3f}"


def _g_tag(g: float) -> str:
    return f"g{g:.2f}".replace(".", "p")


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


def _write(
    files: dict[str, Any],
    dest: Path,
    name: str,
    audio: np.ndarray,
    role: str,
    extra: dict[str, Any] | None = None,
) -> None:
    entry = write_listening_wav(dest / name, audio, SR, limit_mode="clip")
    files[name] = {
        **entry,
        "role": role,
        "dir": str(dest).replace("\\", "/"),
        **(extra or {}),
    }
    print(f"  {dest.name}/{name}")


def run_once() -> dict[str, Any]:
    if not CONSENSUS_METRICS.exists():
        raise FileNotFoundError(CONSENSUS_METRICS)
    if not UNION764_MANIFEST.exists():
        raise FileNotFoundError(UNION764_MANIFEST)
    if not SOURCE_PIANO.exists():
        raise FileNotFoundError(SOURCE_PIANO)
    if not DIR_AUDIO.exists():
        raise FileNotFoundError(DIR_AUDIO)

    cons = json.loads(CONSENSUS_METRICS.read_text(encoding="utf-8"))
    times = np.asarray(cons["stem_consensus"]["times_s"], dtype=np.float64)
    votes = np.asarray(cons["stem_consensus"]["votes"], dtype=np.int64)
    n = int(len(times))
    expected = int(cons["stem_consensus"]["events_2plus"])
    if n != expected or n != 234:
        raise RuntimeError(f"expected locked consensus 234, got {n} (events_2plus={expected})")
    if len(votes) != n:
        raise RuntimeError("votes length mismatch")

    n_v3 = int(np.sum(votes == 3))
    n_v2 = int(np.sum(votes == 2))
    if n_v3 + n_v2 != n:
        raise RuntimeError(f"unexpected votes outside {{2,3}}: v3={n_v3} v2={n_v2}")

    union764 = np.asarray(
        json.loads(UNION764_MANIFEST.read_text(encoding="utf-8"))["peak_times_s"][
            "union"
        ],
        dtype=np.float64,
    )
    covered, missed, _cand_only = one_to_one_time_match(times, union764)
    covered_t = np.asarray([float(t) for t in covered], dtype=np.float64)
    missed_t = np.asarray([float(t) for t in missed], dtype=np.float64)
    if len(covered_t) != 232 or len(missed_t) != 2:
        raise RuntimeError(
            f"expected 764 coverage 232/2, got {len(covered_t)}/{len(missed_t)}"
        )

    vote3_t = times[votes == 3]
    vote2_t = times[votes == 2]

    # beds
    piano, sr = read_stereo(SOURCE_PIANO)
    if sr != SR:
        raise RuntimeError(f"piano sr={sr}")
    piano_mono = piano.mean(axis=1).astype(np.float32)
    piano_low = (piano_mono * np.float32(PIANO_GAIN_LOW)).astype(np.float32)
    mix_lufs = load_mono(DIR_AUDIO).astype(np.float32)
    L, R = read_raw(DIR_AUDIO)
    mix_raw = (0.5 * (L + R)).astype(np.float32)

    c3 = _click(FREQ_UNIFIED_HZ)
    c5 = _click(FREQ_VOTE2_HZ)
    c_miss = _click(FREQ_MISSED_HZ)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ONP_DIR.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    g_low = _g_tag(PIANO_GAIN_LOW)
    g_full = _g_tag(1.0)

    # --- low piano ---
    uni_low = click_wav_name(f"stem_consensus_234_low_{g_low}_unified3k", n)
    _write(
        files,
        ONP_DIR,
        uni_low,
        _overlay(piano_low, times, c3),
        "consensus234_unified_lowpiano",
        {"n_peaks": n, "bed": "bs_piano", "bed_gain": PIANO_GAIN_LOW},
    )

    fs_low = (
        f"stem_consensus_234_low_{g_low}_freqsep"
        f"_클릭_p{n}_v3_{n_v3}_v2_{n_v2}.wav"
    )
    vote_audio = _overlay(piano_low, vote3_t, c3)
    vote_audio = _overlay(vote_audio, vote2_t, c5)
    _write(
        files,
        ONP_DIR,
        fs_low,
        vote_audio,
        "consensus234_vote_freqsep_lowpiano",
        {
            "n_peaks": n,
            "n_vote3": n_v3,
            "n_vote2": n_v2,
            "freqs_hz": {"vote3": FREQ_VOTE3_HZ, "vote2": FREQ_VOTE2_HZ},
            "bed": "bs_piano",
            "bed_gain": PIANO_GAIN_LOW,
        },
    )

    vs764_low = (
        f"stem_consensus_234_vs764_low_{g_low}_freqsep"
        f"_클릭_p{n}_c{len(covered_t)}_m{len(missed_t)}.wav"
    )
    vs_audio = _overlay(piano_low, covered_t, c3)
    vs_audio = _overlay(vs_audio, missed_t, c_miss)
    _write(
        files,
        ONP_DIR,
        vs764_low,
        vs_audio,
        "consensus234_vs764_coverage_lowpiano",
        {
            "n_peaks": n,
            "covered_by_764": int(len(covered_t)),
            "missed_by_764": int(len(missed_t)),
            "freqs_hz": {"covered": FREQ_COVERED_HZ, "missed": FREQ_MISSED_HZ},
            "bed": "bs_piano",
            "bed_gain": PIANO_GAIN_LOW,
        },
    )

    miss_low = click_wav_name(f"stem_consensus_234_missed_by_764_low_{g_low}", len(missed_t))
    _write(
        files,
        ONP_DIR,
        miss_low,
        _overlay(piano_low, missed_t, c_miss),
        "consensus234_missed_by_764_solo_lowpiano",
        {"n_peaks": int(len(missed_t)), "bed": "bs_piano", "bed_gain": PIANO_GAIN_LOW},
    )

    # --- origmix LUFS (same level family as 전체_adaptive) ---
    uni_lufs = click_wav_name(
        f"stem_consensus_234_origmix_lufs_{g_full}_unified3k", n
    )
    _write(
        files,
        ONP_DIR,
        uni_lufs,
        _overlay(mix_lufs, times, c3),
        "consensus234_unified_origmix_lufs",
        {
            "n_peaks": n,
            "bed": "102-Dir",
            "bed_level": f"load_mono_TARGET_LUFS_{TARGET_LUFS}",
            "bed_gain": 1.0,
        },
    )

    fs_lufs = (
        f"stem_consensus_234_origmix_lufs_{g_full}_freqsep"
        f"_클릭_p{n}_v3_{n_v3}_v2_{n_v2}.wav"
    )
    vote_lufs = _overlay(mix_lufs, vote3_t, c3)
    vote_lufs = _overlay(vote_lufs, vote2_t, c5)
    _write(
        files,
        ONP_DIR,
        fs_lufs,
        vote_lufs,
        "consensus234_vote_freqsep_origmix_lufs",
        {
            "n_peaks": n,
            "n_vote3": n_v3,
            "n_vote2": n_v2,
            "freqs_hz": {"vote3": FREQ_VOTE3_HZ, "vote2": FREQ_VOTE2_HZ},
            "bed": "102-Dir",
            "bed_level": f"load_mono_TARGET_LUFS_{TARGET_LUFS}",
            "bed_gain": 1.0,
        },
    )

    # event table + missed detail
    events: list[dict[str, Any]] = []
    missed_set = {float(t) for t in missed_t}
    for i, (t, v) in enumerate(zip(times, votes)):
        tf = float(t)
        supporting = []
        for model in MODELS:
            peaks = np.asarray(
                cons["models"][model]["peak_times_s"], dtype=np.float64
            )
            d = np.abs(peaks - tf)
            j = int(np.argmin(d))
            if d[j] <= MATCH_TOL_S:
                supporting.append(model)
        covered_flag = tf not in missed_set and any(
            abs(tf - float(c)) <= MATCH_TOL_S for c in covered_t
        )
        # more reliable: match by index via nearest covered/missed
        d_c = float(np.min(np.abs(covered_t - tf))) if len(covered_t) else 1e9
        d_m = float(np.min(np.abs(missed_t - tf))) if len(missed_t) else 1e9
        status = "missed_by_764" if d_m <= d_c and d_m <= MATCH_TOL_S else "covered_by_764"
        events.append(
            {
                "i": i,
                "t_s": tf,
                "t_mmss": _fmt(tf),
                "votes": int(v),
                "supporting_models": supporting,
                "vs764": status,
            }
        )

    missed_events = [e for e in events if e["vs764"] == "missed_by_764"]
    if len(missed_events) != 2:
        # fallback: build from missed_t directly
        missed_events = []
        for t in missed_t:
            idx = int(np.argmin(np.abs(times - t)))
            missed_events.append(events[idx])

    md_lines = [
        "# stem 합의 234 소니파이",
        "",
        "## 정의",
        "",
        "- 잠긴 분모: `out/sonify/Dir/stem_consensus_metrics.json`",
        "- BS / Spleeter / Demucs piano에 **동일** A-2+positive rescue → ±30ms **2+** 합의",
        f"- n=**{n}** (3-of-3 **{n_v3}**, 2-of-3 **{n_v2}**)",
        "- 진단용 attribution 증거 (onset GT 아님)",
        "",
        "## 청취 범례",
        "",
        "| 파일 역할 | 클릭 |",
        "|-----------|------|",
        "| unified | 전부 3 kHz |",
        "| vote freqsep | 3-of-3 → **3 kHz**, 2-of-3 → **5 kHz** |",
        "| vs764 freqsep | 764가 덮음 → **3 kHz**, 764 miss → **5 kHz** |",
        "| missed_by_764 solo | miss 2점만 5 kHz |",
        "",
        "## 764 대비 miss 2",
        "",
    ]
    for e in missed_events:
        md_lines.append(
            f"- `{e['t_mmss']}` ({e['t_s']:.6f}s) votes={e['votes']} "
            f"models={','.join(e['supporting_models'])}"
        )
    md_lines += [
        "",
        "## 전체 타임스탬프",
        "",
        "| # | t | votes | models | vs764 |",
        "|--:|---|------:|--------|-------|",
    ]
    for e in events:
        md_lines.append(
            f"| {e['i']} | {e['t_mmss']} | {e['votes']} | "
            f"{'+'.join(e['supporting_models'])} | {e['vs764']} |"
        )
    md_lines.append("")
    md_path = OUT_DIR / "stem_consensus_234_sonify.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"  wrote {md_path}")

    manifest = {
        "experiment": "stem_consensus_234_sonify",
        "note": (
            "Locked session-10 stem consensus (234) sonify with vote/vs764 "
            "frequency separation; beds = low BS piano and origmix LUFS."
        ),
        "warning": (
            "stem consensus is model-derived attribution evidence, not onset GT"
        ),
        "fixed_rules": {
            "consensus_source": str(CONSENSUS_METRICS).replace("\\", "/"),
            "consensus_sha256": sha256_file(CONSENSUS_METRICS),
            "union764_source": str(UNION764_MANIFEST).replace("\\", "/"),
            "match_tol_s": MATCH_TOL_S,
            "target_lufs": TARGET_LUFS,
            "freqs_hz": {
                "unified": FREQ_UNIFIED_HZ,
                "vote3": FREQ_VOTE3_HZ,
                "vote2": FREQ_VOTE2_HZ,
                "covered_by_764": FREQ_COVERED_HZ,
                "missed_by_764": FREQ_MISSED_HZ,
            },
            "listen_limit_mode": "clip",
        },
        "counts": {
            "consensus_n": n,
            "vote3": n_v3,
            "vote2": n_v2,
            "covered_by_764": int(len(covered_t)),
            "missed_by_764": int(len(missed_t)),
            "union764_n": int(len(union764)),
        },
        "missed_by_764": missed_events,
        "events": events,
        "bed_stats": {
            "piano_low": audio_stats(piano_low),
            "origmix_lufs": audio_stats(mix_lufs),
            "origmix_raw_ref": audio_stats(mix_raw),
        },
        "files": files,
        "doc": str(md_path).replace("\\", "/"),
        "legacy_all_click": str(
            ROOT / "out" / "sonify" / "Dir" / "전체_stem_consensus_all_클릭.wav"
        ).replace("\\", "/"),
    }
    write_json(OUT_DIR / "stem_consensus_234_sonify_manifest.json", manifest)
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
        "counts_ok": first_counts_ok(manifest, second),
    }
    write_json(OUT_DIR / "stem_consensus_234_sonify_determinism.json", report)
    return report


def first_counts_ok(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return a["counts"] == b["counts"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--determinism-check", action="store_true")
    args = parser.parse_args()
    print(f"consensus: {CONSENSUS_METRICS}")
    print(f"union764: {UNION764_MANIFEST}")
    manifest = run_once()
    print(
        "counts:",
        manifest["counts"]["consensus_n"],
        "v3",
        manifest["counts"]["vote3"],
        "v2",
        manifest["counts"]["vote2"],
        "764miss",
        manifest["counts"]["missed_by_764"],
    )
    if args.determinism_check:
        print("determinism-check: second run…")
        report = determinism_check(manifest)
        if not report["matched"] or not report["counts_ok"]:
            raise SystemExit(f"determinism failed: {report}")
        print("determinism-check: OK")


if __name__ == "__main__":
    main()
