#!/usr/bin/env python3
"""fuse_v1: clip ⊕ harmonic | clip ⊕ synthesis (onset+pitch gap rescue).

No s4_piano imports. Reads clean_amt out/ notes.json only.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
CLEAN_OUT = Path(__file__).resolve().parents[2] / "clean_amt" / "out"
OUT_ROOT = Path(__file__).resolve().parents[1] / "out"

# Absolute wall-clock window (Dir piano)
WIN_START = 30.0
WIN_END = 60.0
TOL_S = 0.03
CLIP_ABS_OFFSET = 30.0  # clip local 0 == abs 30


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


def find_run(role_suffix: str) -> Path:
    matches = sorted(CLEAN_OUT.glob(f"*_{role_suffix}"))
    if not matches:
        raise FileNotFoundError(f"no clean_amt out for *_{role_suffix} under {CLEAN_OUT}")
    return matches[-1]


def load_notes(run_dir: Path) -> list[dict]:
    path = run_dir / "notes.json"
    return json.loads(path.read_text(encoding="utf-8"))


def to_abs_clip(notes: list[dict]) -> list[dict]:
    """Clip file local time → absolute ( +30 ). Keep window [30,60)."""
    out = []
    for n in notes:
        onset = float(n["onset_s"]) + CLIP_ABS_OFFSET
        offset = float(n["offset_s"]) + CLIP_ABS_OFFSET
        if onset < WIN_START or onset >= WIN_END:
            continue
        out.append(
            {
                "onset_s": onset,
                "offset_s": min(offset, WIN_END + 5.0),
                "pitch": int(n["pitch"]),
                "velocity": int(n.get("velocity", 64)),
                "source": "clip",
            }
        )
    return out


def filter_abs_window(notes: list[dict], source: str) -> list[dict]:
    out = []
    for n in notes:
        onset = float(n["onset_s"])
        if onset < WIN_START or onset >= WIN_END:
            continue
        out.append(
            {
                "onset_s": onset,
                "offset_s": float(n["offset_s"]),
                "pitch": int(n["pitch"]),
                "velocity": int(n.get("velocity", 64)),
                "source": source,
            }
        )
    return out


def has_same_pitch_near(
    base: list[dict], onset: float, pitch: int, tol_s: float
) -> bool:
    for n in base:
        if int(n["pitch"]) != pitch:
            continue
        if abs(float(n["onset_s"]) - onset) <= tol_s:
            return True
    return False


def fuse_rescue(
    base: list[dict], rescue: list[dict], tol_s: float
) -> tuple[list[dict], dict]:
    fused = [dict(n) for n in base]
    added = 0
    skipped = 0
    for n in rescue:
        if has_same_pitch_near(fused, float(n["onset_s"]), int(n["pitch"]), tol_s):
            skipped += 1
            continue
        fused.append(dict(n))
        added += 1
    fused.sort(key=lambda x: (float(x["onset_s"]), int(x["pitch"])))
    stats = {
        "n_base": len(base),
        "n_rescue_in": len(rescue),
        "n_added": added,
        "n_skipped_overlap": skipped,
        "n_fused": len(fused),
        "tol_s": tol_s,
    }
    return fused, stats


def run_one(
    name: str,
    base: list[dict],
    rescue: list[dict],
    rescue_role: str,
    clip_run: Path,
    rescue_run: Path,
) -> Path:
    fused, stats = fuse_rescue(base, rescue, TOL_S)
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    run_id = f"{day}_midi_fuse_{name}"
    out_dir = OUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    notes_path = out_dir / "notes.json"
    mid_path = out_dir / "piano.mid"
    notes_path.write_text(json.dumps(fused, ensure_ascii=False, indent=2), encoding="utf-8")
    write_midi(fused, mid_path)

    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "track": "midi_fuse",
        "recipe": "fuse_v1_clip_rescue",
        "name": name,
        "window_abs_s": [WIN_START, WIN_END],
        "clip_abs_offset_s": CLIP_ABS_OFFSET,
        "tol_s": TOL_S,
        "inputs": {
            "base_role": "stem_dir_clip",
            "base_run": str(clip_run),
            "rescue_role": rescue_role,
            "rescue_run": str(rescue_run),
        },
        "stats": stats,
        "outputs": {"piano_mid": "piano.mid", "notes_json": "notes.json"},
        "notes": "Keep all clip notes in window; add rescue if no same-pitch within tol",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"{name}: base={stats['n_base']} +added={stats['n_added']} "
        f"skip={stats['n_skipped_overlap']} → fused={stats['n_fused']}  {out_dir}"
    )
    return out_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="midi_fuse v1")
    ap.parse_args(argv)

    clip_run = find_run("stem_dir_clip")
    harm_run = find_run("stem_dir_hpss_harmonic")
    synth_run = find_run("stem_dir_lpc_synthesis")

    base = to_abs_clip(load_notes(clip_run))
    harm = filter_abs_window(load_notes(harm_run), "hpss_harmonic")
    synth = filter_abs_window(load_notes(synth_run), "lpc_synthesis")

    run_one("clip_harmonic", base, harm, "stem_dir_hpss_harmonic", clip_run, harm_run)
    run_one("clip_synthesis", base, synth, "stem_dir_lpc_synthesis", clip_run, synth_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
