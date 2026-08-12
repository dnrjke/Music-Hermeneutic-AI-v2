#!/usr/bin/env python3
"""Sonify fuse_v1 rescue-only clicks on low piano (gain 0.20).

Matches Dir sculpt style: 3kHz click, 12ms, amp 0.7; bed = BS piano × 0.20.
No s4_piano imports.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[5]
FUSE_OUT = Path(__file__).resolve().parents[1] / "out"
PIANO = REPO_ROOT / "out" / "stems" / "Dir" / "bs_roformer" / "piano.wav"

PIANO_GAIN_LOW = 0.20
CLICK_FREQ_HZ = 3000.0
CLICK_DUR_MS = 12.0
CLICK_AMP = 0.7
SR_EXPECT = 44100


def _click(sr: int) -> np.ndarray:
    n = int(sr * CLICK_DUR_MS / 1000.0)
    t = np.arange(n, dtype=np.float32) / sr
    env = np.exp(-t * 1000.0 / CLICK_DUR_MS)
    return (CLICK_AMP * env * np.sin(2 * np.pi * CLICK_FREQ_HZ * t)).astype(
        np.float32
    )


def _overlay(mono: np.ndarray, times: list[float], click: np.ndarray, sr: int) -> np.ndarray:
    out = mono.astype(np.float32, copy=True)
    for t in times:
        idx = int(round(t * sr))
        if idx < 0 or idx >= len(out):
            continue
        end = min(idx + len(click), len(out))
        n = end - idx
        if n > 0:
            out[idx:end] += click[:n]
    peak = float(np.max(np.abs(out))) + 1e-12
    if peak > 1.0:
        out = (out / peak * 0.98).astype(np.float32)
    return out


def find_fuse_run(name: str) -> Path:
    matches = sorted(FUSE_OUT.glob(f"*_{name}"))
    if not matches:
        raise FileNotFoundError(f"no fuse out *_{name} under {FUSE_OUT}")
    return matches[-1]


def rescue_onsets(notes: list[dict], source: str) -> list[float]:
    times = []
    for n in notes:
        if n.get("source") == source:
            times.append(float(n["onset_s"]))
    return sorted(times)


def sonify_one(
    fuse_name: str,
    source_tag: str,
    piano_mono: np.ndarray,
    sr: int,
    click: np.ndarray,
) -> dict:
    run_dir = find_fuse_run(fuse_name)
    notes = json.loads((run_dir / "notes.json").read_text(encoding="utf-8"))
    times = rescue_onsets(notes, source_tag)
    n = len(times)
    bed = (piano_mono * np.float32(PIANO_GAIN_LOW)).astype(np.float32)
    audio = _overlay(bed, times, click, sr)

    g_tag = f"g{PIANO_GAIN_LOW:.2f}".replace(".", "p")
    wav_name = f"{fuse_name}_rescueOnly3k_low_{g_tag}_클릭_p{n}.wav"
    wav_path = run_dir / wav_name
    sf.write(str(wav_path), audio, sr, subtype="FLOAT")

    # Also window crop abs 30–60 for scrub (same as fuse window)
    i0 = int(round(30.0 * sr))
    i1 = int(round(60.0 * sr))
    crop = audio[i0:i1]
    crop_name = f"{fuse_name}_rescueOnly3k_low_{g_tag}_t30_60_클릭_p{n}.wav"
    crop_path = run_dir / crop_name
    sf.write(str(crop_path), crop, sr, subtype="FLOAT")

    meta = {
        "fuse_run": str(run_dir),
        "source_tag": source_tag,
        "n_clicks": n,
        "onset_s": times,
        "wav": wav_name,
        "wav_window_30_60": crop_name,
        "piano_gain_low": PIANO_GAIN_LOW,
        "click_freq_hz": CLICK_FREQ_HZ,
    }
    (run_dir / "sonify_rescue_only_manifest.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "track": "midi_fuse",
                "sonify": "rescue_only_lowpiano_3k",
                **meta,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"{fuse_name}: n={n} → {wav_path.name} + {crop_name}")
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rescue-only low-piano click sonify")
    ap.parse_args(argv)

    if not PIANO.is_file():
        raise SystemExit(f"missing {PIANO}")
    piano, sr = sf.read(str(PIANO), always_2d=True, dtype="float32")
    if sr != SR_EXPECT:
        print(f"warn: sr={sr} (expected {SR_EXPECT})", file=sys.stderr)
    mono = piano.mean(axis=1).astype(np.float32)
    click = _click(sr)

    jobs = [
        ("clip_harmonic", "hpss_harmonic"),
        ("clip_synthesis", "lpc_synthesis"),
    ]
    for fuse_name, tag in jobs:
        sonify_one(fuse_name, tag, mono, sr, click)
    return 0


if __name__ == "__main__":
    sys.exit(main())
