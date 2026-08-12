#!/usr/bin/env python3
"""D0 sonify: 506 MIDI onsets as 5kHz clicks on low piano (gain 0.20).

No s4 / clean_amt / midi_fuse imports.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

REPO_ROOT = Path(__file__).resolve().parents[5]
VIA_OUT = Path(__file__).resolve().parents[1] / "out"
DEFAULT_CFG = Path(__file__).resolve().parents[1] / "configs" / "d0_dir.yaml"


def _click(sr: int, freq_hz: float, dur_ms: float, amp: float) -> np.ndarray:
    n = int(sr * dur_ms / 1000.0)
    t = np.arange(n, dtype=np.float32) / sr
    env = np.exp(-t * 1000.0 / dur_ms)
    return (amp * env * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _overlay(mono: np.ndarray, times: list[float], click: np.ndarray, sr: int) -> np.ndarray:
    out = mono.astype(np.float32, copy=True)
    for t in times:
        idx = int(round(float(t) * sr))
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


def resolve_run_dir(arg: Path | None) -> Path:
    if arg is not None:
        p = arg if arg.is_absolute() else (Path.cwd() / arg)
        return p.resolve()
    pointer = VIA_OUT / "latest_d0.txt"
    if pointer.is_file():
        return Path(pointer.read_text(encoding="utf-8").strip())
    matches = sorted(VIA_OUT.glob("*_via764_D0_*"))
    if not matches:
        raise SystemExit(f"no D0 run under {VIA_OUT}; run d0_onset_midi.py first")
    return matches[-1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="via_764 D0 lowpiano sonify")
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=DEFAULT_CFG)
    ap.add_argument("--repo-root", type=Path, default=None)
    args = ap.parse_args(argv)

    root = (args.repo_root or REPO_ROOT).resolve()
    cfg_path = args.config if args.config.is_absolute() else (root / args.config)
    if not cfg_path.is_file():
        cfg_path = args.config.resolve()
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_dir = resolve_run_dir(args.run_dir)
    notes = json.loads((run_dir / "notes.json").read_text(encoding="utf-8"))
    times = [float(n["onset_s"]) for n in notes]
    n = len(times)

    son = cfg.get("sonify") or {}
    gain = float(son.get("piano_gain_low", 0.20))
    freq = float(son.get("click_freq_hz", 5000.0))
    dur_ms = float(son.get("click_dur_ms", 12.0))
    amp = float(son.get("click_amp", 0.7))

    audio_rel = (cfg.get("audio") or {})["path"]
    piano_path = root / audio_rel
    piano, sr = sf.read(str(piano_path), always_2d=True, dtype="float32")
    mono = piano.mean(axis=1).astype(np.float32)
    bed = (mono * np.float32(gain)).astype(np.float32)
    click = _click(sr, freq, dur_ms, amp)
    audio = _overlay(bed, times, click, sr)

    g_tag = f"g{gain:.2f}".replace(".", "p")
    f_tag = f"{int(freq/1000)}k" if freq >= 1000 else f"{int(freq)}"
    wav_name = f"sonify_midi_on_lowpiano_{f_tag}_{g_tag}_클릭_p{n}.wav"
    wav_path = run_dir / wav_name
    sf.write(str(wav_path), audio, sr, subtype="FLOAT")

    crop_name = None
    if son.get("window_t30_60", True):
        i0 = int(round(30.0 * sr))
        i1 = int(round(60.0 * sr))
        crop = audio[i0:i1]
        # count clicks in window
        n_win = sum(1 for t in times if 30.0 <= t < 60.0)
        crop_name = f"sonify_midi_on_lowpiano_{f_tag}_{g_tag}_t30_60_클릭_p{n_win}.wav"
        sf.write(str(run_dir / crop_name), crop, sr, subtype="FLOAT")

    man_path = run_dir / "manifest.json"
    if man_path.is_file():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        man.setdefault("outputs", {})["sonify_lowpiano"] = wav_name
        if crop_name:
            man["outputs"]["sonify_lowpiano_t30_60"] = crop_name
        man_path.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote {wav_path}")
    if crop_name:
        print(f"wrote {run_dir / crop_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
