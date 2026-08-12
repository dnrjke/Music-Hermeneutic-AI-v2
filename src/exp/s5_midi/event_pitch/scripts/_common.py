"""Shared helpers for event_pitch E4+ (no sibling-track imports)."""
from __future__ import annotations

import hashlib
import json
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


def load_config(cfg_path: Path) -> dict:
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_cfg_path(config: Path) -> Path:
    return (Path.cwd() / config).resolve() if not config.is_absolute() else config


def load_peaks_pilot(root: Path, cfg: dict) -> tuple[list[float], list[tuple[int, float]], Path]:
    peaks_cfg = cfg["peaks"]
    man_path = root / peaks_cfg["manifest"]
    man = json.loads(man_path.read_text(encoding="utf-8"))
    key = peaks_cfg["key"]
    all_times = [float(t) for t in man["peak_times_s"][key]]
    w0 = float(cfg["pilot"]["start_s"])
    w1 = float(cfg["pilot"]["end_s"])
    indexed = [(i, t) for i, t in enumerate(all_times) if w0 <= t < w1]
    return all_times, indexed, man_path


def load_mono(root: Path, cfg: dict) -> tuple[np.ndarray, int, Path]:
    audio_path = root / cfg["audio"]["path"]
    stereo, sr = sf.read(str(audio_path), always_2d=True, dtype="float32")
    mono = stereo.mean(axis=1).astype(np.float32)
    return mono, int(sr), audio_path


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


def note_off_time(
    onset: float, all_times: list[float], gap: float, max_dur: float, min_dur: float
) -> float:
    next_t = next((t2 for t2 in all_times if t2 > onset + 1e-9), None)
    if next_t is not None:
        offset = min(onset + max_dur, next_t - gap)
    else:
        offset = onset + max_dur
    if offset - onset < min_dur:
        offset = onset + min_dur
    return float(offset)


def build_notes_from_pitches(
    indexed: list[tuple[int, float]],
    pitches: list[int],
    metas: list[dict],
    all_times: list[float],
    mono: np.ndarray,
    sr: int,
    cfg: dict,
) -> tuple[list[dict], list[dict], int]:
    off = cfg["note_off"]
    vcfg = cfg["velocity"]
    gap = float(off["gap_s"])
    max_dur = float(off["max_dur_s"])
    min_dur = float(off["min_dur_s"])
    notes: list[dict] = []
    pitch_meta: list[dict] = []
    n_miss = 0
    for (peak_id, onset), pitch, meta in zip(indexed, pitches, metas):
        if not meta.get("ok", True):
            n_miss += 1
        offset = note_off_time(onset, all_times, gap, max_dur, min_dur)
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
                "offset_s": offset,
                "pitch": int(pitch),
                "velocity": vel,
                "source_peak_id": int(peak_id),
            }
        )
        pitch_meta.append({"source_peak_id": peak_id, "onset_s": onset, **meta, "pitch": int(pitch)})
    return notes, pitch_meta, n_miss


def write_run(
    *,
    run_id: str,
    stage: str,
    method: str,
    cfg_path: Path,
    audio_path: Path,
    peaks_path: Path,
    peaks_key: str,
    n_pilot: int,
    notes: list[dict],
    pitch_meta: list[dict],
    n_miss: int,
    extra_summary: dict | None = None,
    note: str = "",
) -> Path:
    out_dir = OUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
    write_midi(notes, out_dir / "piano_from_506.mid")
    (out_dir / "pitch_meta.json").write_text(json.dumps(pitch_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    pitches = [n["pitch"] for n in notes]
    summary = {
        "method": method,
        "n_notes": len(notes),
        "n_miss": n_miss,
        "pitch_median": float(np.median(pitches)) if pitches else None,
        "pitch_min": int(min(pitches)) if pitches else None,
        "pitch_max": int(max(pitches)) if pitches else None,
    }
    if extra_summary:
        summary.update(extra_summary)
    (out_dir / "pitch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "track": "event_pitch",
        "stage": stage,
        "axis": "pitch_wise_onset",
        "config_path": str(cfg_path),
        "method": method,
        "note": note,
        "audio": {"path": str(audio_path), "sha256": sha256_file(audio_path)},
        "peaks": {"path": str(peaks_path), "key": peaks_key, "n_pilot": n_pilot},
        "summary": summary,
        "outputs": {"piano_from_506_mid": "piano_from_506.mid", "notes_json": "notes.json"},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_ROOT / f"latest_{stage.lower()}.txt").write_text(str(out_dir.resolve()), encoding="utf-8")
    return out_dir


def argmax_pitch_vector(
    vec: np.ndarray,
    midi0: int,
    midi_min: int,
    midi_max: int,
    fallback: int,
    method: str,
) -> tuple[int, dict]:
    lo = max(0, midi_min - midi0)
    hi = min(len(vec), midi_max - midi0 + 1)
    if hi <= lo:
        return fallback, {"ok": False, "reason": "range_empty", "method": method}
    band = np.asarray(vec[lo:hi], dtype=np.float64)
    if not np.any(np.isfinite(band)) or float(np.nanmax(band)) <= 0:
        return fallback, {"ok": False, "reason": "zero_act", "method": method}
    j = int(np.nanargmax(band))
    pitch = midi0 + lo + j
    top = np.argsort(band)[::-1][:3]
    top3 = [{"midi": int(midi0 + lo + int(i)), "val": float(band[i])} for i in top]
    return pitch, {
        "ok": True,
        "method": method,
        "val": float(band[j]),
        "top3": top3,
    }


def hz_to_midi(hz: float) -> int:
    return int(np.clip(np.round(69.0 + 12.0 * np.log2(hz / 440.0)), 0, 127))
