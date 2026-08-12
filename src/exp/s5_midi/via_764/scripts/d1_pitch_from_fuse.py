#!/usr/bin/env python3
"""D1: pitch transplant from midi_fuse clip⊕harmonic (RO notes.json).

Onsets stay 506. Pitch = best fuse note within ±tol (max_vel or max_pitch).
No package imports from midi_fuse — path read only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

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
                mido.Message("note_on", note=pitch, velocity=max(1, min(127, vel)), time=delta)
            )
        else:
            track.append(mido.Message("note_off", note=pitch, velocity=0, time=delta))
    mid.save(path)


def pick_pitch(cands: list[dict], mode: str) -> dict | None:
    if not cands:
        return None
    if mode == "max_pitch":
        return max(cands, key=lambda n: (int(n["pitch"]), int(n.get("velocity", 0))))
    return max(cands, key=lambda n: (int(n.get("velocity", 0)), int(n["pitch"])))


def rms_velocity(mono: np.ndarray, sr: int, t0: float, t1: float, vmin: int, vmax: int) -> int:
    i0 = max(0, int(round(t0 * sr)))
    i1 = min(len(mono), int(round(t1 * sr)))
    if i1 <= i0:
        return (vmin + vmax) // 2
    seg = mono[i0:i1]
    rms = float(np.sqrt(np.mean(seg**2) + 1e-12))
    lo, hi = 1e-4, 0.3
    x = (np.log10(rms + 1e-12) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
    x = float(np.clip(x, 0.0, 1.0))
    return int(round(vmin + x * (vmax - vmin)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="via_764 D1 fuse pitch transplant")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, default=None)
    args = ap.parse_args(argv)

    root = (args.repo_root or REPO_ROOT).resolve()
    cfg_path = (Path.cwd() / args.config).resolve() if not args.config.is_absolute() else args.config
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    peaks_cfg = cfg["peaks"]
    man = json.loads((root / peaks_cfg["manifest"]).read_text(encoding="utf-8"))
    key = peaks_cfg["key"]
    all_times = [float(t) for t in man["peak_times_s"][key]]
    w0 = float(cfg["pilot"]["start_s"])
    w1 = float(cfg["pilot"]["end_s"])
    indexed = [(i, t) for i, t in enumerate(all_times) if w0 <= t < w1]

    fuse_notes_path = root / cfg["fuse"]["notes_json"]
    if not fuse_notes_path.is_file():
        raise SystemExit(f"missing fuse notes: {fuse_notes_path}")
    fuse_notes = json.loads(fuse_notes_path.read_text(encoding="utf-8"))
    tol = float(cfg["fuse"]["tol_s"])
    pick_mode = str(cfg["fuse"].get("pick") or "max_vel")
    fallback = int(cfg["fuse"].get("fallback_pitch") or 60)

    audio_path = root / cfg["audio"]["path"]
    stereo, sr = sf.read(str(audio_path), always_2d=True, dtype="float32")
    mono = stereo.mean(axis=1).astype(np.float32)

    off_cfg = cfg["note_off"]
    gap = float(off_cfg["gap_s"])
    max_dur = float(off_cfg["max_dur_s"])
    min_dur = float(off_cfg["min_dur_s"])
    vcfg = cfg["velocity"]
    pre = float(cfg.get("vel_window", {}).get("pre_s", 0.02))
    post = float(cfg.get("vel_window", {}).get("post_s", 0.10))

    notes: list[dict] = []
    pitch_meta: list[dict] = []
    n_miss = 0
    for peak_id, onset in indexed:
        cands = [
            n
            for n in fuse_notes
            if abs(float(n["onset_s"]) - onset) <= tol
        ]
        chosen = pick_pitch(cands, pick_mode)
        if chosen is None:
            pitch = fallback
            n_miss += 1
            meta = {"ok": False, "reason": "no_fuse_note", "method": "fuse_transplant"}
        else:
            pitch = int(chosen["pitch"])
            meta = {
                "ok": True,
                "method": "fuse_transplant",
                "fuse_onset": float(chosen["onset_s"]),
                "fuse_source": chosen.get("source"),
                "fuse_vel": int(chosen.get("velocity", 0)),
                "n_cands": len(cands),
            }
        next_t = next((t2 for t2 in all_times if t2 > onset + 1e-9), None)
        if next_t is not None:
            offset = min(onset + max_dur, next_t - gap)
        else:
            offset = onset + max_dur
        if offset - onset < min_dur:
            offset = onset + min_dur
        # Prefer fuse velocity when available
        if chosen is not None:
            vel = int(chosen.get("velocity", 64))
        else:
            vel = rms_velocity(
                mono,
                sr,
                onset - pre,
                onset + post,
                int(vcfg["rms_to_vel_min"]),
                int(vcfg["rms_to_vel_max"]),
            )
        notes.append(
            {
                "onset_s": onset,
                "offset_s": float(offset),
                "pitch": pitch,
                "velocity": vel,
                "source_peak_id": int(peak_id),
            }
        )
        pitch_meta.append({"source_peak_id": peak_id, "onset_s": onset, **meta, "pitch": pitch})

    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    run_id = f"{day}_via764_D1_dir_506_fuse_pitch_t{int(w0)}_{int(w1)}"
    out_dir = OUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
    write_midi(notes, out_dir / "piano_from_506.mid")
    (out_dir / "pitch_meta.json").write_text(json.dumps(pitch_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    pitches = [n["pitch"] for n in notes]
    summary = {
        "method": "fuse_clip_harmonic_transplant",
        "n_notes": len(notes),
        "n_miss": n_miss,
        "pitch_median": float(np.median(pitches)),
        "pitch_min": int(min(pitches)),
        "pitch_max": int(max(pitches)),
        "tol_s": tol,
        "pick": pick_mode,
        "fuse_notes": str(fuse_notes_path),
    }
    (out_dir / "pitch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "track": "via_764",
        "stage": "D1",
        "config_path": str(cfg_path),
        "method": "fuse_transplant",
        "fuse": {
            "notes_json": str(fuse_notes_path),
            "sha256": sha256_file(fuse_notes_path),
            "tol_s": tol,
            "pick": pick_mode,
        },
        "peaks": {"key": key, "n_pilot": len(indexed)},
        "summary": summary,
        "outputs": {"piano_from_506_mid": "piano_from_506.mid", "notes_json": "notes.json"},
        "notes": "506 onsets; pitch from midi_fuse clip⊕harmonic RO",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_ROOT / "latest_d1_fuse.txt").write_text(str(out_dir.resolve()), encoding="utf-8")
    print(f"wrote {out_dir}  n={len(notes)} miss={n_miss} pitch med={summary['pitch_median']:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
