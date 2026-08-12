#!/usr/bin/env python3
"""fuse_v1 full-length: piano Transkun ⊕ hpss_harmonic Transkun.

Same recipe as clip⊕harmonic (keep base; add rescue if no same-pitch ±tol),
but on full-stem absolute-time AMT runs (no 30–60 window crop).
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

TOL_S = 0.03


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
                mido.Message("note_on", note=pitch, velocity=max(1, min(127, vel)), time=delta)
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
    return json.loads((run_dir / "notes.json").read_text(encoding="utf-8"))


def tag_source(notes: list[dict], source: str) -> list[dict]:
    out = []
    for n in notes:
        out.append(
            {
                "onset_s": float(n["onset_s"]),
                "offset_s": float(n["offset_s"]),
                "pitch": int(n["pitch"]),
                "velocity": int(n.get("velocity", 64)),
                "source": source,
            }
        )
    return out


def has_same_pitch_near(base: list[dict], onset: float, pitch: int, tol_s: float) -> bool:
    for n in base:
        if int(n["pitch"]) != pitch:
            continue
        if abs(float(n["onset_s"]) - onset) <= tol_s:
            return True
    return False


def fuse_rescue(base: list[dict], rescue: list[dict], tol_s: float) -> tuple[list[dict], dict]:
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


def shift0(notes: list[dict], origin: float) -> list[dict]:
    return [
        {
            **n,
            "onset_s": max(0.0, float(n["onset_s"]) - origin),
            "offset_s": max(0.01, float(n["offset_s"]) - origin),
        }
        for n in notes
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="midi_fuse v1 full-length piano⊕harmonic")
    ap.add_argument("--repo-root", type=Path, default=None)
    args = ap.parse_args(argv)
    _ = args.repo_root  # reserved; paths are relative to CLEAN_OUT/OUT_ROOT

    piano_run = find_run("stem_dir_piano_full")
    harm_run = find_run("stem_dir_hpss_harmonic_full")
    base = tag_source(load_notes(piano_run), "piano")
    harm = tag_source(load_notes(harm_run), "hpss_harmonic")
    fused, stats = fuse_rescue(base, harm, TOL_S)

    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    name = "piano_harmonic_full"
    run_id = f"{day}_midi_fuse_{name}"
    out_dir = OUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "notes.json").write_text(
        json.dumps(fused, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_midi(fused, out_dir / "piano.mid")
    # convenience crops for prior listen windows
    for w0, w1, tag in [(30.0, 60.0, "t30_60"), (60.0, 90.0, "t60_90")]:
        crop = [n for n in fused if w0 <= float(n["onset_s"]) < w1]
        write_midi(shift0(crop, w0), out_dir / f"piano_{tag}_listen_t0.mid")
        (out_dir / f"notes_{tag}.json").write_text(
            json.dumps(crop, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # also piano-only and harmonic-only for A/B
    write_midi(base, out_dir / "piano_base_only.mid")
    write_midi(harm, out_dir / "harmonic_rescue_in.mid")
    write_midi(shift0([n for n in base if 60 <= float(n["onset_s"]) < 90], 60.0), out_dir / "piano_base_t60_90_listen_t0.mid")
    added_only = [n for n in fused if n.get("source") == "hpss_harmonic"]
    write_midi(shift0([n for n in added_only if 60 <= float(n["onset_s"]) < 90], 60.0), out_dir / "harmonic_added_t60_90_listen_t0.mid")

    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "track": "midi_fuse",
        "recipe": "fuse_v1_piano_harmonic_full",
        "name": name,
        "window_abs_s": "full",
        "tol_s": TOL_S,
        "inputs": {
            "base_role": "stem_dir_piano_full",
            "base_run": str(piano_run),
            "rescue_role": "stem_dir_hpss_harmonic_full",
            "rescue_run": str(harm_run),
        },
        "stats": stats,
        "outputs": {
            "piano_mid": "piano.mid",
            "notes_json": "notes.json",
            "listen_t30_60": "piano_t30_60_listen_t0.mid",
            "listen_t60_90": "piano_t60_90_listen_t0.mid",
            "piano_base_only": "piano_base_only.mid",
            "harmonic_rescue_in": "harmonic_rescue_in.mid",
        },
        "notes": (
            "Full-length counterpart of locked clip⊕harmonic: keep piano Transkun; "
            "add hpss_harmonic Transkun if no same-pitch within tol. "
            "Pilot clip⊕harmonic remains 30-60 only."
        ),
        "pilot_clip_harmonic": "out/20260812_midi_fuse_clip_harmonic (30-60 only; not replaced)",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_ROOT / "latest_piano_harmonic_full.txt").write_text(
        str(out_dir.resolve()), encoding="utf-8"
    )
    print(
        f"{name}: base={stats['n_base']} +added={stats['n_added']} "
        f"skip={stats['n_skipped_overlap']} → fused={stats['n_fused']}  {out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
