#!/usr/bin/env python3
"""E2: Basic Pitch frame activations at fixed 506 onsets.

Uses model_output['note'] (and optional 'onset') at t_i — NOT decoded note_events.
Frequency-preserving neural frame MPE; SuperFlux / fuse transplant unused.
No imports from via_764 / clean_amt packages / midi_fuse / s4_piano.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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


def run_basic_pitch_frames(wav_path: Path) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Return note[T,88], onset[T,88], seconds_per_frame, midi0."""
    from basic_pitch.constants import (
        ANNOTATIONS_BASE_FREQUENCY,
        AUDIO_SAMPLE_RATE,
        FFT_HOP,
    )
    from basic_pitch.inference import predict

    model_output, _midi, _events = predict(str(wav_path))
    note = np.asarray(model_output["note"], dtype=np.float64)
    onset = np.asarray(model_output["onset"], dtype=np.float64)
    spf = float(FFT_HOP) / float(AUDIO_SAMPLE_RATE)
    # 27.5 Hz = MIDI 21
    midi0 = int(round(69.0 + 12.0 * np.log2(ANNOTATIONS_BASE_FREQUENCY / 440.0)))
    return note, onset, spf, midi0


def pitch_at_time(
    note: np.ndarray,
    onset: np.ndarray,
    spf: float,
    midi0: int,
    t_rel: float,
    delay_s: float,
    post_s: float,
    use_onset_gate: bool,
    fallback: int,
    midi_min: int,
    midi_max: int,
) -> tuple[int, dict]:
    t0 = t_rel + delay_s
    t1 = t_rel + post_s
    f0 = int(np.floor(t0 / spf))
    f1 = int(np.ceil(t1 / spf))
    f0 = max(0, min(f0, note.shape[0] - 1))
    f1 = max(f0 + 1, min(f1, note.shape[0]))
    slab = note[f0:f1]  # (F, 88)
    if use_onset_gate:
        # Gate by max onset in same frames (broadcast)
        o = onset[f0:f1]
        slab = slab * (0.25 + 0.75 * o)
    act = np.mean(slab, axis=0)
    # Restrict MIDI range
    lo = max(0, midi_min - midi0)
    hi = min(act.shape[0], midi_max - midi0 + 1)
    if hi <= lo:
        return fallback, {"ok": False, "reason": "range_empty", "method": "basic_pitch_frame"}
    band = act[lo:hi]
    if not np.any(np.isfinite(band)) or float(np.max(band)) <= 0:
        return fallback, {"ok": False, "reason": "zero_act", "method": "basic_pitch_frame"}
    j = int(np.argmax(band))
    pitch = midi0 + lo + j
    top_idx = np.argsort(band)[::-1][:3]
    top3 = [
        {"midi": int(midi0 + lo + int(i)), "act": float(band[i])} for i in top_idx
    ]
    return pitch, {
        "ok": True,
        "method": "basic_pitch_frame",
        "act": float(band[j]),
        "top3": top3,
        "frames": [f0, f1],
        "t0_rel": t0,
        "t1_rel": t1,
        "use_onset_gate": use_onset_gate,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="event_pitch E2 Basic Pitch frame @ 506")
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
    pad = float(cfg["pilot"].get("clip_pad_s") or 2.0)
    indexed = [(i, t) for i, t in enumerate(all_times) if w0 <= t < w1]

    audio_path = root / cfg["audio"]["path"]
    stereo, sr = sf.read(str(audio_path), always_2d=True, dtype="float32")
    mono = stereo.mean(axis=1).astype(np.float32)
    dur = len(mono) / float(sr)

    clip0 = max(0.0, w0 - pad)
    clip1 = min(dur, w1 + pad)
    i0 = int(round(clip0 * sr))
    i1 = int(round(clip1 * sr))
    clip_mono = mono[i0:i1]

    pcfg = cfg["pitch"]
    off_cfg = cfg["note_off"]
    vcfg = cfg["velocity"]
    gap = float(off_cfg["gap_s"])
    max_dur = float(off_cfg["max_dur_s"])
    min_dur = float(off_cfg["min_dur_s"])
    fallback = int(pcfg.get("fallback_pitch") or 60)

    with tempfile.TemporaryDirectory(prefix="event_pitch_e2_") as td:
        clip_path = Path(td) / "pilot_clip.wav"
        sf.write(str(clip_path), clip_mono, sr)
        print(f"BP infer clip [{clip0:.2f},{clip1:.2f}] -> {clip_path.name}", flush=True)
        note, onset, spf, midi0 = run_basic_pitch_frames(clip_path)

    notes: list[dict] = []
    pitch_meta: list[dict] = []
    n_miss = 0
    for peak_id, onset_abs in indexed:
        t_rel = onset_abs - clip0
        pitch, meta = pitch_at_time(
            note,
            onset,
            spf,
            midi0,
            t_rel,
            delay_s=float(pcfg["delay_s"]),
            post_s=float(pcfg["post_s"]),
            use_onset_gate=bool(pcfg.get("use_onset_gate", True)),
            fallback=fallback,
            midi_min=int(pcfg.get("midi_min") or 21),
            midi_max=int(pcfg.get("midi_max") or 108),
        )
        if not meta.get("ok"):
            n_miss += 1
        next_t = next((t2 for t2 in all_times if t2 > onset_abs + 1e-9), None)
        if next_t is not None:
            offset = min(onset_abs + max_dur, next_t - gap)
        else:
            offset = onset_abs + max_dur
        if offset - onset_abs < min_dur:
            offset = onset_abs + min_dur
        vel = rms_velocity(
            mono,
            sr,
            onset_abs - float(vcfg.get("pre_s", 0.02)),
            onset_abs + float(vcfg.get("post_s", 0.10)),
            int(vcfg["rms_to_vel_min"]),
            int(vcfg["rms_to_vel_max"]),
        )
        notes.append(
            {
                "onset_s": onset_abs,
                "offset_s": float(offset),
                "pitch": int(pitch),
                "velocity": vel,
                "source_peak_id": int(peak_id),
            }
        )
        pitch_meta.append(
            {"source_peak_id": peak_id, "onset_s": onset_abs, **meta, "pitch": int(pitch)}
        )

    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    run_id = f"{day}_event_pitch_E2_dir_506_bp_frame_t{int(w0)}_{int(w1)}"
    out_dir = OUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
    write_midi(notes, out_dir / "piano_from_506.mid")
    (out_dir / "pitch_meta.json").write_text(json.dumps(pitch_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    pitches = [n["pitch"] for n in notes]
    summary = {
        "method": "basic_pitch_frame",
        "n_notes": len(notes),
        "n_miss": n_miss,
        "pitch_median": float(np.median(pitches)) if pitches else None,
        "pitch_min": int(min(pitches)) if pitches else None,
        "pitch_max": int(max(pitches)) if pitches else None,
        "use_onset_gate": bool(pcfg.get("use_onset_gate", True)),
        "clip": {"start_s": clip0, "end_s": clip1},
        "spf": spf,
        "midi0": midi0,
        "n_frames": int(note.shape[0]),
    }
    (out_dir / "pitch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "track": "event_pitch",
        "stage": "E2",
        "axis": "frequency_preserving_reestimation",
        "config_path": str(cfg_path),
        "method": "basic_pitch_frame",
        "note": (
            "E1 CQT no-go; E2 samples Basic Pitch note(+onset) frames at 506 times; "
            "decoded note_events unused."
        ),
        "audio": {"path": str(audio_path), "sha256": sha256_file(audio_path)},
        "peaks": {
            "key": key,
            "n_pilot": len(indexed),
            "sha256": sha256_file(root / peaks_cfg["manifest"]),
        },
        "summary": summary,
        "outputs": {"piano_from_506_mid": "piano_from_506.mid", "notes_json": "notes.json"},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_ROOT / "latest_e2.txt").write_text(str(out_dir.resolve()), encoding="utf-8")
    print(f"wrote {out_dir}  n={len(notes)} miss={n_miss} pitch med={summary['pitch_median']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
