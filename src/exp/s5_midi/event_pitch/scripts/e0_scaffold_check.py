#!/usr/bin/env python3
"""E0: verify RO inputs for event_pitch (no pitch inference)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
import yaml

REPO_ROOT = Path(__file__).resolve().parents[5]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="event_pitch E0 scaffold check")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, default=None)
    args = ap.parse_args(argv)

    root = (args.repo_root or REPO_ROOT).resolve()
    cfg_path = (Path.cwd() / args.config).resolve() if not args.config.is_absolute() else args.config
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    man_path = root / cfg["peaks"]["manifest"]
    audio_path = root / cfg["audio"]["path"]
    key = cfg["peaks"]["key"]
    expect_n = int(cfg["peaks"].get("expect_n") or 0)

    ok = True
    if not man_path.is_file():
        print(f"MISSING peaks: {man_path}")
        ok = False
    if not audio_path.is_file():
        print(f"MISSING audio: {audio_path}")
        ok = False
    if not ok:
        return 1

    man = json.loads(man_path.read_text(encoding="utf-8"))
    times = [float(t) for t in man["peak_times_s"][key]]
    w0 = float(cfg["pilot"]["start_s"])
    w1 = float(cfg["pilot"]["end_s"])
    in_win = [t for t in times if w0 <= t < w1]
    info = sf.info(str(audio_path))

    print(f"peaks: {man_path.name} key={key} n={len(times)} (expect {expect_n})")
    print(f"pilot [{w0},{w1}): n={len(in_win)}")
    print(f"audio: {audio_path.name} sr={info.samplerate} dur={info.duration:.2f}s")
    if expect_n and len(times) != expect_n:
        print(f"WARN: peak count {len(times)} != expect_n {expect_n}")
    print("E0 OK — frequency-preserving pitch is E1+")
    return 0


if __name__ == "__main__":
    sys.exit(main())
