#!/usr/bin/env python3
"""E6: 88-key triangular filterbank SuperFlux @ 506."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from _common import (
    REPO_ROOT,
    build_notes_from_pitches,
    load_config,
    load_mono,
    load_peaks_pilot,
    resolve_cfg_path,
    write_run,
)
from _filterbank_sf import piano88_centers, pick_at_time, superflux_2d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="event_pitch E6 88-key SuperFlux")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, default=None)
    args = ap.parse_args(argv)

    root = (args.repo_root or REPO_ROOT).resolve()
    cfg_path = resolve_cfg_path(args.config)
    cfg = load_config(cfg_path)
    all_times, indexed, peaks_path = load_peaks_pilot(root, cfg)
    mono, sr, audio_path = load_mono(root, cfg)
    pcfg = cfg["pitch"]
    w0 = float(cfg["pilot"]["start_s"])
    w1 = float(cfg["pilot"]["end_s"])
    midi_lo = int(pcfg.get("midi_lo") or 21)
    midi_hi = int(pcfg.get("midi_hi") or 108)

    centers, midis = piano88_centers(midi_lo, midi_hi)
    print(f"E6 88-key SF n_keys={len(midis)}...", flush=True)
    sf_b, times = superflux_2d(
        mono,
        sr,
        centers,
        n_fft=int(pcfg["n_fft"]),
        hop_length=int(pcfg["hop_length"]),
        lag=int(pcfg["lag"]),
        max_size=int(pcfg["max_size"]),
    )
    # Map band index -> MIDI axis 128
    sf_p = np.zeros((sf_b.shape[0], 128), dtype=np.float64)
    for bi, m in enumerate(midis):
        sf_p[:, int(m)] = sf_b[:, bi]

    half = int(pcfg.get("half_win_frames") or 1)
    fallback = int(pcfg.get("fallback_pitch") or 60)

    pitches: list[int] = []
    metas: list[dict] = []
    for _pid, t in indexed:
        pitch, meta = pick_at_time(
            sf_p, times, t, half, midi_lo, midi_hi, fallback, "piano88_superflux"
        )
        pitches.append(pitch)
        metas.append(meta)

    notes, pitch_meta, n_miss = build_notes_from_pitches(
        indexed, pitches, metas, all_times, mono, sr, cfg
    )
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    run_id = f"{day}_event_pitch_E6_dir_506_88key_sf_t{int(w0)}_{int(w1)}"
    out = write_run(
        run_id=run_id,
        stage="E6",
        method="piano88_superflux",
        cfg_path=cfg_path,
        audio_path=audio_path,
        peaks_path=peaks_path,
        peaks_key=cfg["peaks"]["key"],
        n_pilot=len(indexed),
        notes=notes,
        pitch_meta=pitch_meta,
        n_miss=n_miss,
        extra_summary={"n_keys": int(len(midis)), "midi_lo": midi_lo, "midi_hi": midi_hi},
        note="E6: one triangular band per piano key, SuperFlux without collapsing keys.",
    )
    print(f"wrote {out}  n={len(notes)} miss={n_miss}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
