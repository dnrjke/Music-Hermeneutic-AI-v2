#!/usr/bin/env python3
"""E3: pitch-axis SuperFlux at fixed 506 onsets.

Goal (reframed): not 'any F0 in a window' (E1/E2 no-go), but a
frequency-resolved SuperFlux — onset strength kept as a function of pitch
at t_i, then argmax. Same family as 1D SuperFlux used for 506, without
folding frequency.

No imports from via_764 / clean_amt packages / midi_fuse / s4_piano / src.onset.
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
from scipy.ndimage import maximum_filter1d

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


def pitch_superflux(
    mono: np.ndarray,
    sr: int,
    *,
    fmin: float,
    fmax: float,
    bins_per_octave: int,
    lag: int,
    max_size: int,
    hop_length: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (sf_pitch[T, 128], times[T], midi_axis[128]).

    sf_pitch[t, midi] = SuperFlux energy at that MIDI (bins aggregated),
    frequency axis NOT collapsed to a scalar.
    """
    n_bins = int(np.ceil(bins_per_octave * np.log2(fmax / fmin)))
    C = np.abs(
        librosa.cqt(
            mono,
            sr=sr,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            hop_length=hop_length,
        )
    )
    # power → dB-like log (SuperFlux family)
    logC = librosa.amplitude_to_db(C, ref=np.max, top_db=80.0)
    # max filter along frequency (SuperFlux local max in previous spectrum)
    size = int(2 * max_size + 1)
    ref = maximum_filter1d(logC, size=size, axis=0, mode="nearest")
    # positive temporal difference with lag
    diff = logC[:, lag:] - ref[:, :-lag]
    sf_bin = np.maximum(0.0, diff)  # (n_bins, T-lag)
    # left-pad lag frames with zeros to align with CQT frames
    sf_bin = np.pad(sf_bin, ((0, 0), (lag, 0)), mode="constant")

    freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave)
    midis = np.array([hz_to_midi(float(f)) for f in freqs], dtype=np.int32)

    # Aggregate to MIDI axis 0..127
    sf_pitch = np.zeros((128, sf_bin.shape[1]), dtype=np.float64)
    for bi, m in enumerate(midis):
        if 0 <= m <= 127:
            sf_pitch[m] += sf_bin[bi]

    times = librosa.frames_to_time(np.arange(sf_pitch.shape[1]), sr=sr, hop_length=hop_length)
    midi_axis = np.arange(128, dtype=np.int32)
    return sf_pitch.T, times, midi_axis  # (T, 128), (T,), (128,)


def pick_at_onset(
    sf_pitch: np.ndarray,
    times: np.ndarray,
    t_onset: float,
    half_win: int,
    midi_min: int,
    midi_max: int,
    fallback: int,
) -> tuple[int, dict]:
    idx = int(np.argmin(np.abs(times - t_onset)))
    i0 = max(0, idx - half_win)
    i1 = min(sf_pitch.shape[0], idx + half_win + 1)
    vec = np.mean(sf_pitch[i0:i1], axis=0)
    lo, hi = midi_min, midi_max + 1
    band = vec[lo:hi]
    if not np.any(band > 0):
        return fallback, {
            "ok": False,
            "reason": "zero_flux",
            "method": "pitch_superflux",
            "frame": idx,
        }
    j = int(np.argmax(band))
    pitch = lo + j
    top = np.argsort(band)[::-1][:3]
    top3 = [{"midi": int(lo + int(i)), "flux": float(band[i])} for i in top]
    return pitch, {
        "ok": True,
        "method": "pitch_superflux",
        "flux": float(band[j]),
        "top3": top3,
        "frame": idx,
        "frames": [i0, i1],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="event_pitch E3 pitch-axis SuperFlux @ 506")
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
    print("computing pitch-axis SuperFlux (CQT, no frequency fold)...", flush=True)
    sf_pitch, times, _midi_axis = pitch_superflux(
        mono,
        sr,
        fmin=float(pcfg["fmin_hz"]),
        fmax=float(pcfg["fmax_hz"]),
        bins_per_octave=int(pcfg.get("bins_per_octave") or 36),
        lag=int(pcfg.get("lag") or 2),
        max_size=int(pcfg.get("max_size") or 3),
    )

    off_cfg = cfg["note_off"]
    vcfg = cfg["velocity"]
    gap = float(off_cfg["gap_s"])
    max_dur = float(off_cfg["max_dur_s"])
    min_dur = float(off_cfg["min_dur_s"])
    fallback = int(pcfg.get("fallback_pitch") or 60)
    half_win = int(pcfg.get("half_win_frames") or 1)
    midi_min = hz_to_midi(float(pcfg["fmin_hz"]))
    midi_max = hz_to_midi(float(pcfg["fmax_hz"]))

    notes: list[dict] = []
    pitch_meta: list[dict] = []
    n_miss = 0
    for peak_id, onset in indexed:
        pitch, meta = pick_at_onset(
            sf_pitch, times, onset, half_win, midi_min, midi_max, fallback
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
    run_id = f"{day}_event_pitch_E3_dir_506_pitch_sf_t{int(w0)}_{int(w1)}"
    out_dir = OUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
    write_midi(notes, out_dir / "piano_from_506.mid")
    (out_dir / "pitch_meta.json").write_text(json.dumps(pitch_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    pitches = [n["pitch"] for n in notes]
    summary = {
        "method": "pitch_superflux",
        "n_notes": len(notes),
        "n_miss": n_miss,
        "pitch_median": float(np.median(pitches)) if pitches else None,
        "pitch_min": int(min(pitches)) if pitches else None,
        "pitch_max": int(max(pitches)) if pitches else None,
        "lag": int(pcfg.get("lag") or 2),
        "max_size": int(pcfg.get("max_size") or 3),
        "half_win_frames": half_win,
    }
    (out_dir / "pitch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "track": "event_pitch",
        "stage": "E3",
        "axis": "pitch_superflux",
        "config_path": str(cfg_path),
        "method": "pitch_superflux",
        "note": (
            "E1/E2 no-go: window F0/frame MPE ≠ goal. "
            "E3 = frequency-resolved SuperFlux (pitch axis) at 506 times; "
            "1D SuperFlux fold forbidden."
        ),
        "audio": {"path": str(audio_path), "sha256": sha256_file(audio_path)},
        "peaks": {"key": key, "n_pilot": len(indexed)},
        "summary": summary,
        "outputs": {"piano_from_506_mid": "piano_from_506.mid", "notes_json": "notes.json"},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_ROOT / "latest_e3.txt").write_text(str(out_dir.resolve()), encoding="utf-8")
    print(f"wrote {out_dir}  n={len(notes)} miss={n_miss} pitch med={summary['pitch_median']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
