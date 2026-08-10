"""Sonify + document the 3 stem-consensus events missed by union_506∪395.

Reference: locked stem_consensus (234). Union from cmp506_vs_395 manifest.
Match: one-to-one ±30ms. Low-piano click sonify (gain 0.20) + markdown timestamps.
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
CONSENSUS_METRICS = ROOT / "out" / "sonify" / "Dir" / "stem_consensus_metrics.json"
CMP_MANIFEST = (
    OUTPUT_DIR
    / "pass2"
    / "lpc_sf_adaptive_on_piano"
    / "cmp506_vs_395_lowpiano_manifest.json"
)
OUT_DIR = OUTPUT_DIR / "pass2" / "consensus_coverage"
ONP_DIR = OUTPUT_DIR / "pass2" / "lpc_sf_adaptive_on_piano"

MATCH_TOL_S = 0.03
PIANO_GAIN_LOW = 0.20
CLICK_FREQ_HZ = 3000.0
CLICK_DUR_MS = 12.0
CLICK_AMP = 0.7
MODELS = ("bs_roformer", "spleeter", "demucs")


def _fmt(t: float) -> str:
    m = int(t // 60)
    s = t - 60 * m
    return f"{m}:{s:06.3f}"


def _click() -> np.ndarray:
    n = int(SR * CLICK_DUR_MS / 1000.0)
    t = np.arange(n, dtype=np.float32) / SR
    env = np.exp(-t * 1000.0 / CLICK_DUR_MS)
    return (
        CLICK_AMP * env * np.sin(2 * np.pi * CLICK_FREQ_HZ * t)
    ).astype(np.float32)


def _overlay(mono: np.ndarray, times: np.ndarray, click: np.ndarray) -> np.ndarray:
    out = mono.astype(np.float32, copy=True)
    for t in times:
        idx = int(float(t) * SR)
        end = min(idx + len(click), len(out))
        n = end - idx
        if n > 0:
            out[idx:end] += click[:n]
    return out


def find_missed() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cons_data = json.loads(CONSENSUS_METRICS.read_text(encoding="utf-8"))
    consensus = np.asarray(
        cons_data["stem_consensus"]["times_s"], dtype=np.float64
    )
    votes = np.asarray(cons_data["stem_consensus"]["votes"], dtype=np.int64)
    cmp_data = json.loads(CMP_MANIFEST.read_text(encoding="utf-8"))
    union = np.asarray(
        cmp_data["peak_times_s"]["union_unified"], dtype=np.float64
    )
    _common, cons_only, _union_only = one_to_one_time_match(consensus, union)
    if len(cons_only) != 3:
        raise RuntimeError(f"expected 3 missed, got {len(cons_only)}")

    details: list[dict[str, Any]] = []
    for t in cons_only:
        idx = int(np.argmin(np.abs(consensus - t)))
        supporting: list[dict[str, Any]] = []
        for model in MODELS:
            peaks = np.asarray(
                cons_data["models"][model]["peak_times_s"], dtype=np.float64
            )
            d = np.abs(peaks - float(t))
            j = int(np.argmin(d))
            if d[j] <= MATCH_TOL_S:
                supporting.append(
                    {
                        "model": model,
                        "peak_s": float(peaks[j]),
                        "skew_ms": float((peaks[j] - t) * 1000.0),
                    }
                )
        details.append(
            {
                "t_s": float(t),
                "t_mmss": _fmt(float(t)),
                "votes": int(votes[idx]),
                "consensus_index": idx,
                "supporting_models": supporting,
            }
        )
    meta = {
        "consensus_n": int(len(consensus)),
        "union_n": int(len(union)),
        "matched": int(len(_common)),
        "missed_n": int(len(cons_only)),
    }
    return cons_only, votes, {"meta": meta, "events": details, "cons_data_votes": votes}


def run_once() -> dict[str, Any]:
    if not SOURCE_PIANO.exists():
        raise FileNotFoundError(SOURCE_PIANO)
    piano, sr = read_stereo(SOURCE_PIANO)
    if sr != SR:
        raise RuntimeError(f"sr={sr}")
    piano_mono = piano.mean(axis=1).astype(np.float32)
    piano_low = (piano_mono * np.float32(PIANO_GAIN_LOW)).astype(np.float32)
    g_tag = f"g{PIANO_GAIN_LOW:.2f}".replace(".", "p")

    missed, _votes, info = find_missed()
    click = _click()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ONP_DIR.mkdir(parents=True, exist_ok=True)

    n = int(len(missed))
    low = _overlay(piano_low, missed, click)
    silence = np.zeros_like(piano_mono)
    nopiano = _overlay(silence, missed, click)

    low_name = click_wav_name(
        f"union506_395_consensus_missed_low_{g_tag}", n
    )
    np_name = click_wav_name("union506_395_consensus_missed_nopiano", n)
    low_entry = write_listening_wav(ONP_DIR / low_name, low, SR, limit_mode="clip")
    np_entry = write_listening_wav(ONP_DIR / np_name, nopiano, SR, limit_mode="clip")

    files = {
        low_name: {
            **low_entry,
            "role": "consensus_missed_by_union_lowpiano",
            "n_peaks": n,
            "dir": str(ONP_DIR).replace("\\", "/"),
        },
        np_name: {
            **np_entry,
            "role": "consensus_missed_by_union_nopiano",
            "n_peaks": n,
        },
    }

    # Markdown doc
    lines = [
        "# union_506∪395가 놓친 stem합의 사건 (3)",
        "",
        "## 정의",
        "",
        "- 분모: 세션10 잠긴 stem 합의 **234** (BS/Spleeter/Demucs piano에",
        "  동일 A-2+positive rescue, ±30ms로 **2+** 모델 합의).",
        "- 후보: `union_506_or_395` (506∪395, n=612).",
        "- 매칭: one-to-one ±30ms → 공통 231, **missed 3**.",
        "- 진단용 (합의 ≠ GT).",
        "",
        "## 출현 시각",
        "",
        "| # | mm:ss.mmm | seconds | votes | 지지 모델 |",
        "|--:|-----------|--------:|------:|-----------|",
    ]
    for i, ev in enumerate(info["events"], start=1):
        models = ", ".join(
            f"{s['model']} ({s['skew_ms']:+.1f}ms)" for s in ev["supporting_models"]
        )
        lines.append(
            f"| {i} | `{ev['t_mmss']}` | {ev['t_s']:.6f} | {ev['votes']} | {models} |"
        )
    lines.extend(
        [
            "",
            "## 소니파이",
            "",
            f"- low piano (×{PIANO_GAIN_LOW}): "
            f"`pass2/lpc_sf_adaptive_on_piano/{low_name}`",
            f"- nopiano: `pass2/lpc_sf_adaptive_on_piano/{np_name}`",
            "",
            "클릭: 3 kHz / 12 ms / amp 0.7.",
            "",
            "---",
            "",
            "생성: `src/exp/s4_piano/stem_event_sculpt/run_union_consensus_missed.py`",
            "",
        ]
    )
    md_path = OUT_DIR / "union506_395_consensus_missed.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "experiment": "union506_395_consensus_missed",
        "diagnostic_only": True,
        "fixed_rules": {
            "match_tol_s": MATCH_TOL_S,
            "piano_gain_low": PIANO_GAIN_LOW,
            "click_hz": CLICK_FREQ_HZ,
            "consensus_source": str(CONSENSUS_METRICS).replace("\\", "/"),
            "union_source": str(CMP_MANIFEST).replace("\\", "/"),
            "piano_sha256": sha256_file(SOURCE_PIANO),
        },
        "meta": info["meta"],
        "events": info["events"],
        "peak_times_s": [float(t) for t in missed],
        "piano_stats": audio_stats(piano),
        "files": files,
        "doc": str(md_path).replace("\\", "/"),
    }
    write_json(OUT_DIR / "union506_395_consensus_missed_manifest.json", manifest)

    print(f"missed={n}")
    for ev in info["events"]:
        print(f"  {ev['t_mmss']}  votes={ev['votes']}")
    print(f"  {low_name}")
    print(f"  {np_name}")
    print(f"  doc: {md_path}")
    return manifest


def determinism_check(manifest: dict[str, Any]) -> dict[str, Any]:
    first = {n: e["sha256"] for n, e in manifest["files"].items()}
    first_times = list(manifest["peak_times_s"])
    second = run_once()
    mismatches = [
        n for n in first if first[n] != second["files"].get(n, {}).get("sha256")
    ]
    report = {
        "matched": len(mismatches) == 0
        and first_times == list(second["peak_times_s"]),
        "wav_mismatches": mismatches,
        "peak_mismatch": first_times != list(second["peak_times_s"]),
    }
    write_json(OUT_DIR / "union506_395_consensus_missed_determinism.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--determinism-check", action="store_true")
    args = parser.parse_args()
    manifest = run_once()
    if args.determinism_check:
        print("determinism-check: second run…")
        report = determinism_check(manifest)
        if not report["matched"]:
            raise SystemExit(f"determinism failed: {report}")
        print("determinism-check: OK")


if __name__ == "__main__":
    main()
