#!/usr/bin/env python3
"""E1: CQT harmonic salience top-1 pitch at fixed 506 onsets.

Frequency-preserving re-estimation at t_i (not SuperFlux 1D, not fuse transplant).
No imports from via_764 / clean_amt / midi_fuse / stem_norm / s4_piano.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import librosa
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


def hz_to_midi(hz: float) -> int:
    return int(np.clip(np.round(69.0 + 12.0 * np.log2(hz / 440.0)), 0, 127))


def midi_to_hz(midi: int) -> float:
    return float(440.0 * (2.0 ** ((midi - 69) / 12.0)))


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


def estimate_cqt_harmonic_salience(
    mono: np.ndarray,
    sr: int,
    t_onset: float,
    delay_s: float,
    post_s: float,
    fmin: float,
    fmax: float,
    fallback: int,
    bins_per_octave: int,
    n_harmonics: int,
    harmonic_decay: float,
) -> tuple[int, dict]:
    """Mean CQT over [t+delay, t+post]; score F0 candidates by harmonic sum."""
    t0 = t_onset + delay_s
    t1 = t_onset + post_s
    i0 = max(0, int(round(t0 * sr)))
    i1 = min(len(mono), int(round(t1 * sr)))
    if i1 - i0 < int(0.02 * sr):
        return fallback, {"ok": False, "reason": "short_window", "method": "cqt_harmonic_salience"}

    y = mono[i0:i1].astype(np.float32)
    # Enough samples for stable CQT; pad if needed
    n_bins = int(np.ceil(bins_per_octave * np.log2(fmax / fmin)))
    C = np.abs(
        librosa.cqt(
            y,
            sr=sr,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            hop_length=max(64, len(y) // 8),
        )
    )
    if C.size == 0:
        return fallback, {"ok": False, "reason": "empty_cqt", "method": "cqt_harmonic_salience"}

    mag = np.mean(C, axis=1)  # (n_bins,)
    freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave)

    # Candidate F0 = each CQT bin in [fmin, fmax/2] so harmonics fit
    scores: list[tuple[float, float, int]] = []  # score, hz, midi
    for bi, f0 in enumerate(freqs):
        if f0 < fmin or f0 > fmax:
            continue
        score = 0.0
        for h in range(1, n_harmonics + 1):
            target = f0 * h
            if target > freqs[-1] * 1.01:
                break
            j = int(np.argmin(np.abs(freqs - target)))
            score += float(mag[j]) * (harmonic_decay ** (h - 1))
        scores.append((score, float(f0), hz_to_midi(float(f0))))

    if not scores:
        return fallback, {"ok": False, "reason": "no_candidates", "method": "cqt_harmonic_salience"}

    scores.sort(key=lambda x: -x[0])
    best_score, best_hz, best_midi = scores[0]
    top3 = [{"midi": m, "hz": hz, "score": sc} for sc, hz, m in scores[:3]]
    return best_midi, {
        "ok": True,
        "method": "cqt_harmonic_salience",
        "hz": best_hz,
        "score": best_score,
        "top3": top3,
        "t0": t0,
        "t1": t1,
        "n_bins": n_bins,
        "bins_per_octave": bins_per_octave,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="event_pitch E1 CQT salience at 506 onsets")
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

    pcfg = cfg["pitch"]
    off_cfg = cfg["note_off"]
    vcfg = cfg["velocity"]
    gap = float(off_cfg["gap_s"])
    max_dur = float(off_cfg["max_dur_s"])
    min_dur = float(off_cfg["min_dur_s"])
    fallback = int(pcfg.get("fallback_pitch") or 60)

    notes: list[dict] = []
    pitch_meta: list[dict] = []
    n_miss = 0
    for peak_id, onset in indexed:
        pitch, meta = estimate_cqt_harmonic_salience(
            mono,
            sr,
            onset,
            delay_s=float(pcfg["delay_s"]),
            post_s=float(pcfg["post_s"]),
            fmin=float(pcfg["fmin_hz"]),
            fmax=float(pcfg["fmax_hz"]),
            fallback=fallback,
            bins_per_octave=int(pcfg.get("bins_per_octave") or 36),
            n_harmonics=int(pcfg.get("n_harmonics") or 5),
            harmonic_decay=float(pcfg.get("harmonic_decay") or 0.6),
        )
        if not meta.get("ok"):
            n_miss += 1
        next_t = next((t2 for t2 in all_times if t2 > onset + 1e-9), None)
        if next_t is not None:
            offset = min(onset + max_dur, next_t - gap)
        else:
            offset = onset + max_dur
        if offset - onset < min_dur:
            offset = onset + min_dur
        vel = rms_velocity(
            mono,
            sr,
            onset - float(vcfg.get("pre_s", 0.02)),
            onset + float(vcfg.get("post_s", 0.10)),
            int(vcfg["rms_to_vel_min"]),
            int(vcfg["rms_to_vel_max"]),
        )
        notes.append(
            {
                "onset_s": onset,
                "offset_s": float(offset),
                "pitch": int(pitch),
                "velocity": vel,
                "source_peak_id": int(peak_id),
            }
        )
        pitch_meta.append({"source_peak_id": peak_id, "onset_s": onset, **meta, "pitch": int(pitch)})

    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    run_id = f"{day}_event_pitch_E1_dir_506_cqt_t{int(w0)}_{int(w1)}"
    out_dir = OUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
    write_midi(notes, out_dir / "piano_from_506.mid")
    (out_dir / "pitch_meta.json").write_text(json.dumps(pitch_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    pitches = [n["pitch"] for n in notes]
    summary = {
        "method": "cqt_harmonic_salience",
        "n_notes": len(notes),
        "n_miss": n_miss,
        "pitch_median": float(np.median(pitches)) if pitches else None,
        "pitch_min": int(min(pitches)) if pitches else None,
        "pitch_max": int(max(pitches)) if pitches else None,
        "window": {"delay_s": float(pcfg["delay_s"]), "post_s": float(pcfg["post_s"])},
    }
    (out_dir / "pitch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "track": "event_pitch",
        "stage": "E1",
        "axis": "frequency_preserving_reestimation",
        "config_path": str(cfg_path),
        "method": "cqt_harmonic_salience",
        "audio": {"path": str(audio_path), "sha256": sha256_file(audio_path)},
        "peaks": {
            "manifest": str(root / peaks_cfg["manifest"]),
            "key": key,
            "n_pilot": len(indexed),
            "sha256": sha256_file(root / peaks_cfg["manifest"]),
        },
        "summary": summary,
        "outputs": {"piano_from_506_mid": "piano_from_506.mid", "notes_json": "notes.json"},
        "notes": (
            "NEW axis vs via_764: pitch from CQT harmonic salience at fixed 506 times; "
            "SuperFlux not used for pitch; no fuse transplant."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_ROOT / "latest_e1.txt").write_text(str(out_dir.resolve()), encoding="utf-8")
    print(f"wrote {out_dir}  n={len(notes)} miss={n_miss} pitch med={summary['pitch_median']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
