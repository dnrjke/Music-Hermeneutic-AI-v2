#!/usr/bin/env python3
"""Optional: render piano.mid to preview.wav for A/B listen.

Stub: requires a local SoundFont + fluidsynth, or replace with another renderer.
Does not use s4_piano.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render MIDI preview (optional)")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--soundfont", type=Path, default=None, help="SF2 path")
    ap.add_argument("--fluidsynth", type=str, default="fluidsynth")
    args = ap.parse_args(argv)

    run_dir = args.run_dir.resolve()
    mid = run_dir / "piano.mid"
    if not mid.is_file():
        raise SystemExit(f"missing {mid}")
    out_wav = run_dir / "preview.wav"

    fs = shutil.which(args.fluidsynth)
    if not fs or args.soundfont is None or not args.soundfont.is_file():
        raise SystemExit(
            "preview render needs fluidsynth on PATH and --soundfont path/to.sf2. "
            "Skip this step for M1 if unavailable; use DAW to open piano.mid instead."
        )

    cmd = [
        fs,
        "-ni",
        "-g",
        "1.0",
        "-F",
        str(out_wav),
        "-r",
        "44100",
        str(args.soundfont),
        str(mid),
    ]
    subprocess.check_call(cmd)
    print(f"wrote {out_wav}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
