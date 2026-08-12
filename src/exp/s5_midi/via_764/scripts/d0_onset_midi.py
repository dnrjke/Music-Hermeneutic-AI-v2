#!/usr/bin/env python3
"""D0: load Dir 506 peaks → placeholder MIDI (no pitch fill).

No imports from clean_amt / midi_fuse / stem_norm / s4_piano.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise SystemExit("PyYAML required (use clean_amt env)") from e

REPO_ROOT = Path(__file__).resolve().parents[5]
OUT_ROOT = Path(__file__).resolve().parents[1] / "out"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_midi(notes: list[dict], path: Path, tempo_bpm: float = 120.0) -> None:
    import mido

    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm)))
    track.append(mido.Message("program_change", program=0, time=0))

    events: list[tuple[float, str, dict]] = []
    for n in notes:
        events.append((float(n["onset_s"]), "on", n))
        events.append((float(n["offset_s"]), "off", n))
    events.sort(key=lambda x: (x[0], 0 if x[1] == "off" else 1))

    tpb = mid.ticks_per_beat
    last_tick = 0
    for t_sec, kind, n in events:
        tick = int(round(t_sec * tpb * (tempo_bpm / 60.0)))
        delta = max(0, tick - last_tick)
        last_tick = tick
        pitch = int(n["pitch"])
        vel = int(n.get("velocity", 64))
        if kind == "on":
            track.append(
                mido.Message("note_on", note=pitch, velocity=max(1, vel), time=delta)
            )
        else:
            track.append(mido.Message("note_off", note=pitch, velocity=0, time=delta))
    mid.save(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="via_764 D0 onset MIDI")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, default=None)
    args = ap.parse_args(argv)

    root = (args.repo_root or REPO_ROOT).resolve()
    cfg_path = args.config if args.config.is_absolute() else (Path.cwd() / args.config)
    cfg_path = cfg_path.resolve()
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    peaks_cfg = cfg["peaks"]
    man_path = root / peaks_cfg["manifest"]
    if not man_path.is_file():
        raise SystemExit(f"missing peaks manifest: {man_path}")
    man = json.loads(man_path.read_text(encoding="utf-8"))
    key = peaks_cfg["key"]
    times = man.get("peak_times_s", {}).get(key)
    if times is None:
        raise SystemExit(f"key {key!r} not in peak_times_s")
    times = [float(t) for t in times]
    expect_n = int(peaks_cfg.get("expect_n") or 0)
    if expect_n and len(times) != expect_n:
        raise SystemExit(f"expected {expect_n} peaks, got {len(times)}")

    midi_cfg = cfg.get("midi") or {}
    pitch = int(midi_cfg.get("placeholder_pitch", 60))
    vel = int(midi_cfg.get("placeholder_vel", 80))
    dur = float(midi_cfg.get("note_dur_s", 0.05))

    notes = []
    for i, t in enumerate(times):
        notes.append(
            {
                "onset_s": t,
                "offset_s": t + dur,
                "pitch": pitch,
                "velocity": vel,
                "source_peak_id": i,
            }
        )

    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    run_id = f"{day}_via764_D0_dir_506"
    out_dir = OUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    mid_path = out_dir / "piano_from_506.mid"
    notes_path = out_dir / "notes.json"
    notes_path.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
    write_midi(notes, mid_path)

    # Alignment: note onset vs peak (should be ~0)
    deltas = [abs(float(n["onset_s"]) - times[i]) for i, n in enumerate(notes)]
    alignment = {
        "n_peaks": len(times),
        "n_notes": len(notes),
        "max_abs_dt_s": float(max(deltas) if deltas else 0.0),
        "mean_abs_dt_s": float(sum(deltas) / len(deltas) if deltas else 0.0),
        "key": key,
        "ok": len(notes) == len(times) and (max(deltas) if deltas else 0.0) < 1e-9,
    }
    (out_dir / "alignment_vs_506.json").write_text(
        json.dumps(alignment, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    audio_rel = (cfg.get("audio") or {}).get("path")
    audio_path = (root / audio_rel) if audio_rel else None

    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "track": "via_764",
        "stage": "D0",
        "config_path": str(cfg_path),
        "peaks": {
            "manifest": str(man_path),
            "manifest_sha256": sha256_file(man_path),
            "key": key,
            "n": len(times),
        },
        "audio": {
            "path": str(audio_path) if audio_path else None,
            "sha256": sha256_file(audio_path) if audio_path and audio_path.is_file() else None,
        },
        "midi": midi_cfg,
        "outputs": {
            "piano_from_506_mid": "piano_from_506.mid",
            "notes_json": "notes.json",
            "alignment_vs_506_json": "alignment_vs_506.json",
        },
        "alignment": alignment,
        "notes": "D0 placeholder pitch only; no clean_amt/s4 import",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Stable pointer for sonify
    (OUT_ROOT / "latest_d0.txt").write_text(str(out_dir.resolve()), encoding="utf-8")
    print(f"wrote {out_dir}  n={len(notes)} max|dt|={alignment['max_abs_dt_s']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
