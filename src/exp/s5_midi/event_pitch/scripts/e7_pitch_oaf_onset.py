#!/usr/bin/env python3
"""E7: OaF-family 88 onset map @ 506 (ByteDance high-res CRNN)."""
from __future__ import annotations

import argparse
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from _common import (
    OUT_ROOT,
    REPO_ROOT,
    argmax_pitch_vector,
    build_notes_from_pitches,
    load_config,
    load_mono,
    load_peaks_pilot,
    resolve_cfg_path,
    write_run,
)


def ensure_checkpoint(weights_dir: Path) -> Path:
    weights_dir.mkdir(parents=True, exist_ok=True)
    dest = weights_dir / "note_F1=0.9677_pedal_F1=0.9186.pth"
    if dest.is_file() and dest.stat().st_size > 1.6e8:
        return dest
    url = (
        "https://zenodo.org/record/4034264/files/"
        "CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
    )
    print(f"downloading checkpoint -> {dest} (~165MB)", flush=True)
    urllib.request.urlretrieve(url, dest)
    return dest


def run_bytedance_onset(wav_path: Path, ckpt: Path, device_str: str) -> tuple[np.ndarray, float, str]:
    import librosa
    from piano_transcription_inference import PianoTranscription, config as pt_config
    from piano_transcription_inference import sample_rate

    audio, _ = librosa.core.load(str(wav_path), sr=sample_rate, mono=True)
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    transcriptor = PianoTranscription(
        checkpoint_path=str(ckpt), segment_samples=sample_rate * 10, device=device
    )
    result = transcriptor.transcribe(audio, midi_path=None)
    onset = np.asarray(result["output_dict"]["reg_onset_output"], dtype=np.float64)
    # (frames, 88)
    fps = float(pt_config.frames_per_second)
    spf = 1.0 / fps
    return onset, spf, f"bytedance_hr/{device_str}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="event_pitch E7 OaF/ByteDance onset @ 506")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, default=None)
    args = ap.parse_args(argv)

    root = (args.repo_root or REPO_ROOT).resolve()
    cfg_path = resolve_cfg_path(args.config)
    cfg = load_config(cfg_path)
    all_times, indexed, peaks_path = load_peaks_pilot(root, cfg)
    mono, sr, audio_path = load_mono(root, cfg)
    w0 = float(cfg["pilot"]["start_s"])
    w1 = float(cfg["pilot"]["end_s"])
    pad = float(cfg["pilot"].get("clip_pad_s") or 1.0)
    dur = len(mono) / float(sr)
    clip0 = max(0.0, w0 - pad)
    clip1 = min(dur, w1 + pad)
    i0 = int(round(clip0 * sr))
    i1 = int(round(clip1 * sr))

    pcfg = cfg["pitch"]
    half = int(pcfg.get("half_win_frames") or 1)
    fallback = int(pcfg.get("fallback_pitch") or 60)
    midi_min = int(pcfg.get("midi_min") or 21)
    midi_max = int(pcfg.get("midi_max") or 108)
    midi0 = 21

    weights_dir = OUT_ROOT.parent / "weights"
    ckpt = ensure_checkpoint(weights_dir)

    with tempfile.TemporaryDirectory(prefix="event_pitch_e7_") as td:
        clip_path = Path(td) / "pilot_clip.wav"
        sf.write(str(clip_path), mono[i0:i1], sr)
        print("E7 ByteDance onset infer...", flush=True)
        onset, spf, backend = run_bytedance_onset(clip_path, ckpt, "cuda")

    pitches: list[int] = []
    metas: list[dict] = []
    for _pid, t_abs in indexed:
        t_rel = t_abs - clip0
        f = int(round(t_rel / spf))
        f0 = max(0, f - half)
        f1 = min(onset.shape[0], f + half + 1)
        vec = np.mean(onset[f0:f1], axis=0)
        pitch, meta = argmax_pitch_vector(vec, midi0, midi_min, midi_max, fallback, "oaf_onset")
        meta["frames"] = [f0, f1]
        meta["backend"] = backend
        pitches.append(pitch)
        metas.append(meta)

    notes, pitch_meta, n_miss = build_notes_from_pitches(
        indexed, pitches, metas, all_times, mono, sr, cfg
    )
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    run_id = f"{day}_event_pitch_E7_dir_506_oaf_onset_t{int(w0)}_{int(w1)}"
    out = write_run(
        run_id=run_id,
        stage="E7",
        method="oaf_onset",
        cfg_path=cfg_path,
        audio_path=audio_path,
        peaks_path=peaks_path,
        peaks_key=cfg["peaks"]["key"],
        n_pilot=len(indexed),
        notes=notes,
        pitch_meta=pitch_meta,
        n_miss=n_miss,
        extra_summary={"backend": backend, "spf": spf, "clip": [clip0, clip1]},
        note="E7: ByteDance high-res onset regression (OaF-family); MIDI decode unused.",
    )
    print(f"wrote {out}  n={len(notes)} miss={n_miss} backend={backend}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
