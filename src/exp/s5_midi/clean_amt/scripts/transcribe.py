#!/usr/bin/env python3
"""clean_amt: wav/flac -> piano.mid + notes.json + manifest.json

Pilot: Transkun 2.0.1 (independent env/.venv). Do not import s4_piano.
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

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise SystemExit("PyYAML required: pip install -r env/requirements.txt") from e

# scripts -> clean_amt -> s5_midi -> exp -> src -> repo
REPO_ROOT = Path(__file__).resolve().parents[5]
OUT_ROOT = Path(__file__).resolve().parents[1] / "out"
CLEAN_AMT_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_id_for(cfg: dict, model_id: str) -> str:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    role = cfg.get("role", "run")
    mid = model_id.replace(" ", "_")
    return f"{day}_clean_amt_{mid}_{role}"


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


def prepare_audio_window(
    audio_path: Path,
    start_s: float,
    end_s: float | None,
    preprocess: dict,
    dest_wav: Path,
) -> None:
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(str(audio_path), always_2d=True)
    i0 = int(round(start_s * sr))
    i1 = int(round(float(end_s) * sr)) if end_s is not None else data.shape[0]
    i0 = max(0, min(i0, data.shape[0]))
    i1 = max(i0, min(i1, data.shape[0]))
    clip = data[i0:i1]
    if preprocess.get("mono", True):
        clip = clip.mean(axis=1, keepdims=True)
    norm = preprocess.get("normalize") or "none"
    if norm == "peak":
        peak = float(np.max(np.abs(clip))) + 1e-12
        clip = clip / peak * 0.95
    elif norm == "rms":
        rms = float(np.sqrt(np.mean(clip**2))) + 1e-12
        clip = clip / rms * 0.1
    sf.write(str(dest_wav), clip, sr, subtype="PCM_16")


def postprocess_notes(notes: list[dict], post: dict) -> list[dict]:
    min_dur = float(post.get("min_dur_s") or 0.0)
    min_vel = int(post.get("min_vel") or 0)
    out = []
    for n in notes:
        dur = float(n["offset_s"]) - float(n["onset_s"])
        if dur < min_dur:
            continue
        if int(n.get("velocity", 0)) < min_vel:
            continue
        out.append(n)
    return out


def infer_transkun(
    window_wav: Path,
    work_dir: Path,
    device: str = "cuda",
) -> list[dict]:
    out_mid = work_dir / "raw_transkun.mid"
    cmd = [
        sys.executable,
        "-m",
        "transkun.transcribe",
        str(window_wav),
        str(out_mid),
        "--device",
        device,
    ]
    subprocess.check_call(cmd)
    if not out_mid.is_file():
        raise RuntimeError(f"transkun produced no MIDI: {out_mid}")
    return midi_to_notes(out_mid)


def infer_notes(
    audio_path: Path,
    cfg: dict,
    work_dir: Path,
    start_s: float,
    end_s: float | None,
) -> list[dict]:
    model_id = (cfg.get("model_id") or "TBD").lower()
    preprocess = cfg.get("preprocess") or {}
    post = cfg.get("postprocess") or {}
    window_wav = work_dir / "window.wav"
    prepare_audio_window(audio_path, start_s, end_s, preprocess, window_wav)

    if model_id == "transkun":
        device = "cuda"
        try:
            import torch

            if not torch.cuda.is_available():
                device = "cpu"
        except ImportError:
            device = "cpu"
        notes = infer_transkun(window_wav, work_dir, device=device)
    else:
        raise SystemExit(
            f"No backend registered for model_id={model_id!r}. "
            "Pilot is transkun; basic_pitch is M2 audit."
        )

    # Times are relative to window start → absolute in source file
    for n in notes:
        n["onset_s"] = float(n["onset_s"]) + start_s
        n["offset_s"] = float(n["offset_s"]) + start_s
    return postprocess_notes(notes, post)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="clean_amt transcribe")
    ap.add_argument("--config", required=True, type=Path, help="YAML under configs/")
    ap.add_argument("--repo-root", type=Path, default=None)
    args = ap.parse_args(argv)

    repo_root = (args.repo_root or REPO_ROOT).resolve()

    cfg_path = args.config
    if not cfg_path.is_absolute():
        cfg_path = (Path.cwd() / cfg_path).resolve()
    cfg = load_config(cfg_path)

    inp = cfg["input"]
    audio_path = Path(inp["path"])
    audio = audio_path if audio_path.is_absolute() else (repo_root / audio_path).resolve()
    if not audio.is_file():
        raise SystemExit(f"input not found: {audio}")

    model_id = str(cfg.get("model_id", "TBD"))
    rid = run_id_for(cfg, model_id if model_id != "TBD" else "nomodel")
    out_dir = OUT_ROOT / rid
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "_work"
    work_dir.mkdir(exist_ok=True)

    digest = sha256_file(audio)
    start_s = float(inp.get("start_s") or 0.0)
    end_s = inp.get("end_s")

    notes = infer_notes(audio, cfg, work_dir, start_s, end_s)

    notes_path = out_dir / "notes.json"
    mid_path = out_dir / "piano.mid"
    notes_path.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
    write_midi(notes, mid_path)

    manifest = {
        "run_id": rid,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "track": "clean_amt",
        "model_id": model_id,
        "model_version": cfg.get("model_version"),
        "config_path": str(cfg_path),
        "input": {
            "path": str(audio),
            "sha256": digest,
            "start_s": start_s,
            "end_s": end_s,
            "role": cfg.get("role"),
            "gt_midi": inp.get("gt_midi"),
        },
        "preprocess": cfg.get("preprocess") or {},
        "postprocess": cfg.get("postprocess") or {},
        "outputs": {
            "piano_mid": "piano.mid",
            "notes_json": "notes.json",
            "preview_wav": None,
            "metrics_json": None,
        },
        "notes": "clean_amt; pilot Transkun 2.0.1; no s4_piano import",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {out_dir}  notes={len(notes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
