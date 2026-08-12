#!/usr/bin/env python3
"""E4: Basic Pitch onset[T,88] at fixed 506 times (not note frame)."""
from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from _common import (
    REPO_ROOT,
    argmax_pitch_vector,
    build_notes_from_pitches,
    load_config,
    load_mono,
    load_peaks_pilot,
    resolve_cfg_path,
    write_run,
)


def run_bp_onset(wav_path: Path) -> tuple[np.ndarray, float, int]:
    from basic_pitch.constants import (
        ANNOTATIONS_BASE_FREQUENCY,
        AUDIO_SAMPLE_RATE,
        FFT_HOP,
    )
    from basic_pitch.inference import predict

    model_output, _m, _e = predict(str(wav_path))
    onset = np.asarray(model_output["onset"], dtype=np.float64)
    spf = float(FFT_HOP) / float(AUDIO_SAMPLE_RATE)
    midi0 = int(round(69.0 + 12.0 * np.log2(ANNOTATIONS_BASE_FREQUENCY / 440.0)))
    return onset, spf, midi0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="event_pitch E4 BP onset @ 506")
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
    pad = float(cfg["pilot"].get("clip_pad_s") or 2.0)
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

    with tempfile.TemporaryDirectory(prefix="event_pitch_e4_") as td:
        clip_path = Path(td) / "pilot_clip.wav"
        sf.write(str(clip_path), mono[i0:i1], sr)
        print(f"BP onset infer [{clip0:.2f},{clip1:.2f}]", flush=True)
        onset, spf, midi0 = run_bp_onset(clip_path)

    pitches: list[int] = []
    metas: list[dict] = []
    for _pid, t_abs in indexed:
        t_rel = t_abs - clip0
        f = int(round(t_rel / spf))
        f0 = max(0, f - half)
        f1 = min(onset.shape[0], f + half + 1)
        vec = np.mean(onset[f0:f1], axis=0)
        pitch, meta = argmax_pitch_vector(vec, midi0, midi_min, midi_max, fallback, "basic_pitch_onset")
        meta["frames"] = [f0, f1]
        pitches.append(pitch)
        metas.append(meta)

    notes, pitch_meta, n_miss = build_notes_from_pitches(
        indexed, pitches, metas, all_times, mono, sr, cfg
    )
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    run_id = f"{day}_event_pitch_E4_dir_506_bp_onset_t{int(w0)}_{int(w1)}"
    out = write_run(
        run_id=run_id,
        stage="E4",
        method="basic_pitch_onset",
        cfg_path=cfg_path,
        audio_path=audio_path,
        peaks_path=peaks_path,
        peaks_key=cfg["peaks"]["key"],
        n_pilot=len(indexed),
        notes=notes,
        pitch_meta=pitch_meta,
        n_miss=n_miss,
        extra_summary={"spf": spf, "midi0": midi0, "clip": [clip0, clip1]},
        note="E4: Basic Pitch onset head only (E2 used note frame — wrong quantity).",
    )
    print(f"wrote {out}  n={len(notes)} miss={n_miss}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
