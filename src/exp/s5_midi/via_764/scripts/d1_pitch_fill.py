#!/usr/bin/env python3
"""D1: fill top-1 pitch at fixed 506 onsets (pilot window).

Onsets stay locked to peaks.
Pitch methods: harmonic_peak (v2) | spectral_peak | pyin_top1 (legacy no-go).
Note-off: next onset − gap, capped by max_dur.
No imports from clean_amt / midi_fuse / stem_norm / s4_piano.
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


def hz_to_midi(hz: float) -> int:
    return int(np.clip(np.round(69.0 + 12.0 * np.log2(hz / 440.0)), 0, 127))


def midi_to_hz(midi: int) -> float:
    return float(440.0 * (2.0 ** ((midi - 69) / 12.0)))


def _window_slice(
    mono: np.ndarray, sr: int, t0: float, t1: float
) -> tuple[np.ndarray | None, dict]:
    i0 = max(0, int(round(t0 * sr)))
    i1 = min(len(mono), int(round(t1 * sr)))
    if i1 - i0 < int(0.02 * sr):
        return None, {"ok": False, "reason": "short_window"}
    return mono[i0:i1].astype(np.float32), {"ok": True, "i0": i0, "i1": i1}


def _mag_spectrum(y: np.ndarray, sr: int, n_fft: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(y)
    if n < n_fft:
        y = np.pad(y, (0, n_fft - n))
    else:
        y = y[:n_fft]
    w = np.hanning(len(y)).astype(np.float32)
    spec = np.fft.rfft(y * w, n=n_fft)
    mag = np.abs(spec).astype(np.float64)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    return freqs, mag


def estimate_pitch_spectral_peak(
    mono: np.ndarray,
    sr: int,
    t0: float,
    t1: float,
    fmin: float,
    fmax: float,
    fallback: int,
    n_fft: int = 4096,
) -> tuple[int, dict]:
    y, meta0 = _window_slice(mono, sr, t0, t1)
    if y is None:
        return fallback, meta0
    freqs, mag = _mag_spectrum(y, sr, n_fft)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return fallback, {"ok": False, "reason": "band_empty", "method": "spectral_peak"}
    band_idx = np.where(mask)[0]
    idx = int(band_idx[np.argmax(mag[band_idx])])
    hz = float(freqs[idx])
    pitch = hz_to_midi(hz)
    return pitch, {
        "ok": True,
        "method": "spectral_peak",
        "hz": hz,
        "mag": float(mag[idx]),
    }


def estimate_pitch_harmonic(
    mono: np.ndarray,
    sr: int,
    t0: float,
    t1: float,
    fmin: float,
    fmax: float,
    fallback: int,
    n_fft: int = 4096,
    n_harmonics: int = 5,
    midi_lo: int = 28,
    midi_hi: int = 96,
) -> tuple[int, dict]:
    y, meta0 = _window_slice(mono, sr, t0, t1)
    if y is None:
        return fallback, meta0
    freqs, mag = _mag_spectrum(y, sr, n_fft)

    def mag_at(hz: float) -> float:
        if hz <= 0 or hz >= freqs[-1]:
            return 0.0
        return float(np.interp(hz, freqs, mag))

    best_midi = fallback
    best_score = -1.0
    for midi in range(midi_lo, midi_hi + 1):
        f0 = midi_to_hz(midi)
        if f0 < fmin or f0 > fmax:
            continue
        score = 0.0
        for h in range(1, n_harmonics + 1):
            score += mag_at(f0 * h) / h
        if score > best_score:
            best_score = score
            best_midi = midi
    if best_score <= 0:
        return fallback, {"ok": False, "reason": "no_score", "method": "harmonic_peak"}
    return best_midi, {
        "ok": True,
        "method": "harmonic_peak",
        "hz": midi_to_hz(best_midi),
        "score": float(best_score),
        "n_harmonics": n_harmonics,
    }


def estimate_pitch_pyin(
    mono: np.ndarray,
    sr: int,
    t0: float,
    t1: float,
    fmin: float,
    fmax: float,
    fallback: int,
    frame_length: int = 2048,
) -> tuple[int, dict]:
    import librosa

    y, meta0 = _window_slice(mono, sr, t0, t1)
    if y is None:
        return fallback, meta0
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=int(frame_length),
    )
    if f0 is None:
        return fallback, {"ok": False, "reason": "pyin_none", "method": "pyin_top1"}
    f0 = np.asarray(f0, dtype=np.float64)
    if voiced_flag is not None:
        voiced = np.asarray(voiced_flag, dtype=bool)
        vals = f0[voiced & np.isfinite(f0) & (f0 > 0)]
    else:
        vals = f0[np.isfinite(f0) & (f0 > 0)]
    if vals.size == 0:
        return fallback, {"ok": False, "reason": "unvoiced", "method": "pyin_top1"}
    hz = float(np.median(vals))
    pitch = hz_to_midi(hz)
    return pitch, {"ok": True, "method": "pyin_top1", "hz": hz, "n_voiced": int(vals.size)}


def estimate_pitch(
    method: str,
    mono: np.ndarray,
    sr: int,
    t0: float,
    t1: float,
    pcfg: dict,
) -> tuple[int, dict]:
    fmin = float(pcfg["fmin_hz"])
    fmax = float(pcfg["fmax_hz"])
    fallback = int(pcfg["fallback_pitch"])
    n_fft = int(pcfg.get("n_fft") or 4096)
    if method == "pyin_top1":
        return estimate_pitch_pyin(
            mono,
            sr,
            t0,
            t1,
            fmin,
            fmax,
            fallback,
            frame_length=int(pcfg.get("frame_length") or 2048),
        )
    if method == "spectral_peak":
        return estimate_pitch_spectral_peak(
            mono, sr, t0, t1, fmin, fmax, fallback, n_fft=n_fft
        )
    if method == "harmonic_peak":
        return estimate_pitch_harmonic(
            mono,
            sr,
            t0,
            t1,
            fmin,
            fmax,
            fallback,
            n_fft=n_fft,
            n_harmonics=int(pcfg.get("n_harmonics") or 5),
            midi_lo=int(pcfg.get("midi_lo") or 28),
            midi_hi=int(pcfg.get("midi_hi") or 96),
        )
    raise SystemExit(f"unknown pitch.method={method!r}")


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
    ap = argparse.ArgumentParser(description="via_764 D1 pitch fill")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, default=None)
    args = ap.parse_args(argv)

    root = (args.repo_root or REPO_ROOT).resolve()
    cfg_path = args.config if args.config.is_absolute() else (Path.cwd() / args.config)
    cfg_path = cfg_path.resolve()
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    peaks_cfg = cfg["peaks"]
    man_path = root / peaks_cfg["manifest"]
    man = json.loads(man_path.read_text(encoding="utf-8"))
    key = peaks_cfg["key"]
    all_times = [float(t) for t in man["peak_times_s"][key]]
    expect_n = int(peaks_cfg.get("expect_n") or 0)
    if expect_n and len(all_times) != expect_n:
        raise SystemExit(f"expected {expect_n} peaks, got {len(all_times)}")

    pilot = cfg["pilot"]
    w0 = float(pilot["start_s"])
    w1 = float(pilot["end_s"])
    indexed = [(i, t) for i, t in enumerate(all_times) if w0 <= t < w1]
    if not indexed:
        raise SystemExit(f"no peaks in [{w0}, {w1})")

    audio_path = root / cfg["audio"]["path"]
    stereo, sr = sf.read(str(audio_path), always_2d=True, dtype="float32")
    mono = stereo.mean(axis=1).astype(np.float32)

    pcfg = cfg["pitch"]
    method = str(pcfg.get("method") or "harmonic_peak")
    pre = float(pcfg["pre_s"])
    post = float(pcfg["post_s"])

    off_cfg = cfg["note_off"]
    gap = float(off_cfg["gap_s"])
    max_dur = float(off_cfg["max_dur_s"])
    min_dur = float(off_cfg["min_dur_s"])

    vcfg = cfg["velocity"]
    vmin = int(vcfg["rms_to_vel_min"])
    vmax = int(vcfg["rms_to_vel_max"])

    times_only = [t for _, t in indexed]
    notes: list[dict] = []
    pitch_meta: list[dict] = []
    n_fallback = 0

    for _k, (peak_id, onset) in enumerate(indexed):
        t_a0 = onset - pre
        t_a1 = onset + post
        pitch, meta = estimate_pitch(method, mono, sr, t_a0, t_a1, pcfg)
        if not meta.get("ok"):
            n_fallback += 1
        next_t = None
        for t2 in all_times:
            if t2 > onset + 1e-9:
                next_t = t2
                break
        if next_t is not None:
            offset = min(onset + max_dur, next_t - gap)
        else:
            offset = onset + max_dur
        if offset - onset < min_dur:
            offset = onset + min_dur

        vel = rms_velocity(mono, sr, t_a0, t_a1, vmin, vmax)
        notes.append(
            {
                "onset_s": onset,
                "offset_s": float(offset),
                "pitch": int(pitch),
                "velocity": int(vel),
                "source_peak_id": int(peak_id),
            }
        )
        pitch_meta.append(
            {"source_peak_id": peak_id, "onset_s": onset, **meta, "pitch": pitch}
        )

    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    run_id = f"{day}_via764_D1_dir_506_{method}_t{int(w0)}_{int(w1)}"
    out_dir = OUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "notes.json").write_text(
        json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_midi(notes, out_dir / "piano_from_506.mid")
    (out_dir / "pitch_meta.json").write_text(
        json.dumps(pitch_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    pitches = [n["pitch"] for n in notes]
    summary = {
        "method": method,
        "n_notes": len(notes),
        "n_fallback": n_fallback,
        "pitch_mean": float(np.mean(pitches)),
        "pitch_median": float(np.median(pitches)),
        "pitch_min": int(min(pitches)),
        "pitch_max": int(max(pitches)),
        "window_s": [w0, w1],
    }
    (out_dir / "pitch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    deltas = [abs(notes[i]["onset_s"] - times_only[i]) for i in range(len(notes))]
    alignment = {
        "n_peaks_in_window": len(indexed),
        "n_notes": len(notes),
        "max_abs_dt_s": float(max(deltas) if deltas else 0.0),
        "ok": (max(deltas) if deltas else 0.0) < 1e-9,
    }
    (out_dir / "alignment_vs_506.json").write_text(
        json.dumps(alignment, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "track": "via_764",
        "stage": "D1",
        "config_path": str(cfg_path),
        "peaks": {
            "manifest": str(man_path),
            "manifest_sha256": sha256_file(man_path),
            "key": key,
            "n_full": len(all_times),
            "n_pilot": len(indexed),
        },
        "audio": {"path": str(audio_path), "sha256": sha256_file(audio_path)},
        "pilot": pilot,
        "pitch": pcfg,
        "note_off": off_cfg,
        "velocity": vcfg,
        "outputs": {
            "piano_from_506_mid": "piano_from_506.mid",
            "notes_json": "notes.json",
            "pitch_meta_json": "pitch_meta.json",
            "pitch_summary_json": "pitch_summary.json",
            "alignment_vs_506_json": "alignment_vs_506.json",
        },
        "summary": summary,
        "alignment": alignment,
        "notes": f"D1 method={method}; onset locked to 506; pyin legacy no-go",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_ROOT / "latest_d1.txt").write_text(str(out_dir.resolve()), encoding="utf-8")
    print(
        f"wrote {out_dir}  method={method} n={len(notes)} fallback={n_fallback} "
        f"pitch med={summary['pitch_median']:.0f} "
        f"range={summary['pitch_min']}-{summary['pitch_max']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
