#!/usr/bin/env python3
"""D1: pitch from local Transkun windows (concatenated, one infer).

Each 506 onset → short stem crop; crops concatenated with silence;
one `python -m transkun.transcribe`; map notes back per segment.
Onsets remain 506. No clean_amt/s4 package imports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
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


def midi_to_notes(path: Path) -> list[dict]:
    import mido

    mid = mido.MidiFile(path)
    abs_t = 0.0
    active: dict[int, tuple[float, int]] = {}
    notes: list[dict] = []
    for msg in mid:
        abs_t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            active[msg.note] = (abs_t, msg.velocity)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            if msg.note in active:
                onset, vel = active.pop(msg.note)
                notes.append(
                    {
                        "onset_s": onset,
                        "offset_s": abs_t,
                        "pitch": int(msg.note),
                        "velocity": int(vel),
                    }
                )
    return notes


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


def pick_pitch(
    cands: list[dict], mode: str
) -> dict | None:
    if not cands:
        return None
    if mode == "max_pitch":
        return max(cands, key=lambda n: (int(n["pitch"]), int(n.get("velocity", 0))))
    return max(cands, key=lambda n: (int(n.get("velocity", 0)), int(n["pitch"])))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="via_764 D1 local AMT pitch")
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

    audio_path = root / cfg["audio"]["path"]
    stereo, sr = sf.read(str(audio_path), always_2d=True, dtype="float32")
    mono = stereo.mean(axis=1).astype(np.float32)

    acfg = cfg["amt"]
    pre = float(acfg["pre_s"])
    post = float(acfg["post_s"])
    gap_s = float(acfg["gap_s"])
    pick_tol = float(acfg["pick_tol_s"])
    pick_mode = str(acfg.get("pick") or "max_vel")
    fallback = int(acfg.get("fallback_pitch") or 60)
    device = str(acfg.get("device") or "cuda")

    seg_n = int(round((pre + post) * sr))
    gap_n = int(round(gap_s * sr))
    pieces: list[np.ndarray] = []
    seg_starts: list[float] = []  # onset of segment in concat time
    t_cursor = 0.0
    for _pid, onset in indexed:
        i0 = max(0, int(round((onset - pre) * sr)))
        i1 = min(len(mono), i0 + seg_n)
        clip = mono[i0:i1]
        if len(clip) < seg_n:
            clip = np.pad(clip, (0, seg_n - len(clip)))
        # peak normalize each clip lightly
        peak = float(np.max(np.abs(clip))) + 1e-12
        clip = (clip / peak * 0.95).astype(np.float32)
        seg_starts.append(t_cursor)
        pieces.append(clip)
        pieces.append(np.zeros(gap_n, dtype=np.float32))
        t_cursor += (seg_n + gap_n) / sr

    concat = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)

    off_cfg = cfg["note_off"]
    gap = float(off_cfg["gap_s"])
    max_dur = float(off_cfg["max_dur_s"])
    min_dur = float(off_cfg["min_dur_s"])
    vcfg = cfg["velocity"]

    with tempfile.TemporaryDirectory(prefix="via764_amt_") as td:
        td_path = Path(td)
        wav_in = td_path / "concat.wav"
        mid_out = td_path / "concat.mid"
        sf.write(str(wav_in), concat, sr, subtype="PCM_16")
        cmd = [
            sys.executable,
            "-m",
            "transkun.transcribe",
            str(wav_in),
            str(mid_out),
            "--device",
            device,
        ]
        try:
            import torch

            if device == "cuda" and not torch.cuda.is_available():
                cmd[-1] = "cpu"
        except ImportError:
            cmd[-1] = "cpu"
        subprocess.check_call(cmd)
        amt_notes = midi_to_notes(mid_out)

    notes: list[dict] = []
    pitch_meta: list[dict] = []
    n_miss = 0
    for k, (peak_id, onset) in enumerate(indexed):
        seg_t0 = seg_starts[k]
        target = seg_t0 + pre
        near = [
            n
            for n in amt_notes
            if abs(float(n["onset_s"]) - target) <= pick_tol
        ]
        if not near:
            # any note inside segment body
            seg_t1 = seg_t0 + (seg_n / sr)
            near = [
                n
                for n in amt_notes
                if seg_t0 <= float(n["onset_s"]) < seg_t1
            ]
        chosen = pick_pitch(near, pick_mode)
        if chosen is None:
            pitch = fallback
            n_miss += 1
            meta = {"ok": False, "reason": "no_amt_note", "method": "local_amt"}
        else:
            pitch = int(chosen["pitch"])
            meta = {
                "ok": True,
                "method": "local_amt",
                "amt_onset_concat": float(chosen["onset_s"]),
                "amt_vel": int(chosen.get("velocity", 0)),
                "n_cands": len(near),
            }

        next_t = next((t2 for t2 in all_times if t2 > onset + 1e-9), None)
        if next_t is not None:
            offset = min(onset + max_dur, next_t - gap)
        else:
            offset = onset + max_dur
        if offset - onset < min_dur:
            offset = onset + min_dur
        vel = rms_velocity(mono, sr, onset - pre, onset + post, int(vcfg["rms_to_vel_min"]), int(vcfg["rms_to_vel_max"]))
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
    run_id = f"{day}_via764_D1_dir_506_local_amt_t{int(w0)}_{int(w1)}"
    out_dir = OUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
    write_midi(notes, out_dir / "piano_from_506.mid")
    (out_dir / "pitch_meta.json").write_text(json.dumps(pitch_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    pitches = [n["pitch"] for n in notes]
    summary = {
        "method": "local_amt_concat_transkun",
        "n_notes": len(notes),
        "n_miss": n_miss,
        "pitch_median": float(np.median(pitches)),
        "pitch_min": int(min(pitches)),
        "pitch_max": int(max(pitches)),
        "n_amt_raw": len(amt_notes),
    }
    (out_dir / "pitch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "track": "via_764",
        "stage": "D1",
        "config_path": str(cfg_path),
        "method": "local_amt",
        "peaks": {"manifest": str(root / peaks_cfg["manifest"]), "key": key, "n_pilot": len(indexed)},
        "audio": {"path": str(audio_path), "sha256": sha256_file(audio_path)},
        "amt": acfg,
        "summary": summary,
        "outputs": {"piano_from_506_mid": "piano_from_506.mid", "notes_json": "notes.json"},
        "notes": "506 onsets locked; pitch from concatenated local Transkun windows",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_ROOT / "latest_d1_amt.txt").write_text(str(out_dir.resolve()), encoding="utf-8")
    print(f"wrote {out_dir}  n={len(notes)} miss={n_miss} pitch med={summary['pitch_median']:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
