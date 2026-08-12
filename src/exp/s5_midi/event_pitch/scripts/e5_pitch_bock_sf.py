#!/usr/bin/env python3
"""E5: Böck quarter-tone filterbank SuperFlux (2D) @ 506."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from _common import (
    REPO_ROOT,
    build_notes_from_pitches,
    load_config,
    load_mono,
    load_peaks_pilot,
    resolve_cfg_path,
    write_run,
)
from _filterbank_sf import (
    aggregate_to_midi,
    pick_at_time,
    quartertone_centers,
    superflux_2d,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="event_pitch E5 Böck QT SuperFlux")
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

    centers = quartertone_centers(
        float(pcfg["fmin_hz"]), float(pcfg["fmax_hz"]), int(pcfg["bands_per_octave"])
    )
    print(f"E5 Böck QT SF bands={len(centers)}...", flush=True)
    sf_b, times = superflux_2d(
        mono,
        sr,
        centers,
        n_fft=int(pcfg["n_fft"]),
        hop_length=int(pcfg["hop_length"]),
        lag=int(pcfg["lag"]),
        max_size=int(pcfg["max_size"]),
    )
    sf_p = aggregate_to_midi(sf_b, centers)

    half = int(pcfg.get("half_win_frames") or 1)
    fallback = int(pcfg.get("fallback_pitch") or 60)
    midi_min = int(pcfg.get("midi_min") or 21)
    midi_max = int(pcfg.get("midi_max") or 108)

    pitches: list[int] = []
    metas: list[dict] = []
    for _pid, t in indexed:
        pitch, meta = pick_at_time(
            sf_p, times, t, half, midi_min, midi_max, fallback, "bock_quartertone_superflux"
        )
        pitches.append(pitch)
        metas.append(meta)

    notes, pitch_meta, n_miss = build_notes_from_pitches(
        indexed, pitches, metas, all_times, mono, sr, cfg
    )
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    run_id = f"{day}_event_pitch_E5_dir_506_bock_sf_t{int(w0)}_{int(w1)}"
    out = write_run(
        run_id=run_id,
        stage="E5",
        method="bock_quartertone_superflux",
        cfg_path=cfg_path,
        audio_path=audio_path,
        peaks_path=peaks_path,
        peaks_key=cfg["peaks"]["key"],
        n_pilot=len(indexed),
        notes=notes,
        pitch_meta=pitch_meta,
        n_miss=n_miss,
        extra_summary={"n_bands": len(centers), "bands_per_octave": int(pcfg["bands_per_octave"])},
        note="E5: Böck-style quarter-tone filterbank SuperFlux before 1D sum.",
    )
    print(f"wrote {out}  n={len(notes)} miss={n_miss}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
