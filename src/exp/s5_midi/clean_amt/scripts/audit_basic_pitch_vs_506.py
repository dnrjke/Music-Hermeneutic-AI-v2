#!/usr/bin/env python3
"""Audit B: Basic Pitch on Dir piano + 506 hit/miss vs fuse/Transkun.

No sibling-track package imports. Reads baseline notes.json by path (RO).
Does not invent pitch — only BP notes + snap hits to 506.
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


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def prepare_window(audio: Path, start_s: float, end_s: float, out_wav: Path, normalize: str) -> None:
    stereo, sr = sf.read(str(audio), always_2d=True, dtype="float32")
    mono = stereo.mean(axis=1).astype(np.float32)
    i0 = max(0, int(round(start_s * sr)))
    i1 = min(len(mono), int(round(end_s * sr)))
    clip = mono[i0:i1]
    if normalize == "peak":
        peak = float(np.max(np.abs(clip))) + 1e-12
        clip = clip / peak * 0.95
    sf.write(str(out_wav), clip, sr)


def amp_to_velocity(amp: float) -> int:
    """Basic Pitch note_events[3] is amplitude in ~[0,1], not MIDI velocity."""
    a = float(amp)
    if a <= 0:
        return 64
    if a <= 1.0:
        return int(max(1, min(127, round(a * 127.0))))
    return int(max(1, min(127, round(a))))


def run_basic_pitch(window_wav: Path) -> list[dict]:
    from basic_pitch.inference import predict

    _model_output, midi_data, note_events = predict(str(window_wav))
    notes: list[dict] = []
    # note_events: (start_s, end_s, pitch_midi, amplitude_0_1, ...)
    for ev in note_events:
        if isinstance(ev, dict):
            onset = float(ev.get("start_time") or ev.get("onset_s") or ev["start"])
            offset = float(ev.get("end_time") or ev.get("offset_s") or ev["end"])
            pitch = int(ev.get("pitch_midi") or ev.get("pitch") or ev["midi_note"])
            amp = float(ev.get("amplitude") or ev.get("velocity") or 0.5)
        else:
            onset = float(ev[0])
            offset = float(ev[1])
            pitch = int(ev[2])
            amp = float(ev[3]) if len(ev) > 3 else 0.5
        notes.append(
            {
                "onset_s": onset,
                "offset_s": offset,
                "pitch": pitch,
                "velocity": amp_to_velocity(amp),
            }
        )
    notes.sort(key=lambda n: (n["onset_s"], n["pitch"]))
    _ = midi_data
    return notes


def shift_to_zero(notes: list[dict], origin_s: float) -> list[dict]:
    """Relocate so window start plays at t=0 (DAW listen convenience)."""
    out = []
    for n in notes:
        out.append(
            {
                **n,
                "onset_s": max(0.0, float(n["onset_s"]) - origin_s),
                "offset_s": max(0.01, float(n["offset_s"]) - origin_s),
            }
        )
    return out


def has_hit(t: float, notes: list[dict], tol: float) -> bool:
    return any(abs(float(n["onset_s"]) - t) <= tol for n in notes)


def pick_note(t: float, notes: list[dict], tol: float) -> dict | None:
    cands = [n for n in notes if abs(float(n["onset_s"]) - t) <= tol]
    if not cands:
        return None
    return max(cands, key=lambda n: int(n.get("velocity", 0)))


def filter_win(notes: list[dict], w0: float, w1: float) -> list[dict]:
    return [n for n in notes if w0 <= float(n["onset_s"]) < w1]


def shift_notes(notes: list[dict], offset_s: float) -> list[dict]:
    out = []
    for n in notes:
        out.append(
            {
                **n,
                "onset_s": float(n["onset_s"]) + offset_s,
                "offset_s": float(n["offset_s"]) + offset_s,
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Basic Pitch audit vs 506")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, default=None)
    args = ap.parse_args(argv)

    root = (args.repo_root or REPO_ROOT).resolve()
    cfg_path = args.config if args.config.is_absolute() else (Path.cwd() / args.config).resolve()
    cfg = load_config(cfg_path)

    audio = root / cfg["input"]["path"]
    if not audio.is_file():
        raise SystemExit(f"missing audio: {audio}")

    man = json.loads((root / cfg["peaks"]["manifest"]).read_text(encoding="utf-8"))
    key = cfg["peaks"]["key"]
    all_peaks = [float(t) for t in man["peak_times_s"][key]]
    tol = float(cfg["tol_s"])

    fuse_all = json.loads((root / cfg["baselines"]["fuse_notes"]).read_text(encoding="utf-8"))
    tk_all = json.loads((root / cfg["baselines"]["transkun_notes"]).read_text(encoding="utf-8"))

    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    run_id = f"{day}_clean_amt_basic_pitch_dir_piano_506"
    out_dir = OUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    window_reports: list[dict] = []
    normalize = str((cfg.get("preprocess") or {}).get("normalize") or "peak")

    for win in cfg["windows"]:
        w0 = float(win["start_s"])
        w1 = float(win["end_s"])
        tag = f"t{int(w0)}_{int(w1)}"
        print(f"Basic Pitch window {w0}-{w1} ...", flush=True)
        with tempfile.TemporaryDirectory(prefix="bp_audit_") as td:
            tw = Path(td) / "window.wav"
            prepare_window(audio, w0, w1, tw, normalize)
            rel_notes = run_basic_pitch(tw)
        notes = shift_notes(rel_notes, w0)
        peaks = [t for t in all_peaks if w0 <= t < w1]
        fuse_w = filter_win(fuse_all, w0, w1)
        tk_w = filter_win(tk_all, w0, w1)

        snap: list[dict] = []
        misses: list[dict] = []
        n_hit = 0
        n_bp_new_vs_fuse = 0
        n_bp_new_vs_tk = 0
        n_bp_new_vs_both = 0
        for i, t in enumerate(peaks):
            hit_bp = has_hit(t, notes, tol)
            hit_fuse = has_hit(t, fuse_w, tol)
            hit_tk = has_hit(t, tk_w, tol)
            if hit_bp:
                n_hit += 1
                pn = pick_note(t, notes, tol)
                assert pn is not None
                dur = max(0.04, float(pn["offset_s"]) - float(pn["onset_s"]))
                snap.append(
                    {
                        "onset_s": t,
                        "offset_s": t + dur,
                        "pitch": int(pn["pitch"]),
                        "velocity": int(pn.get("velocity", 80)),
                        "source_peak_id": i,
                    }
                )
                if not hit_fuse:
                    n_bp_new_vs_fuse += 1
                if not hit_tk:
                    n_bp_new_vs_tk += 1
                if not hit_fuse and not hit_tk:
                    n_bp_new_vs_both += 1
            else:
                misses.append({"onset_s": t, "hit_fuse": hit_fuse, "hit_transkun": hit_tk})

        wdir = out_dir / tag
        wdir.mkdir(parents=True, exist_ok=True)
        (wdir / "notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
        write_midi(notes, wdir / "piano.mid")
        write_midi(shift_to_zero(notes, w0), wdir / "piano_listen_t0.mid")
        write_midi(snap, wdir / "piano_506_snap_hits_only.mid")
        write_midi(shift_to_zero(snap, w0), wdir / "piano_506_snap_hits_only_listen_t0.mid")
        (wdir / "misses_506.json").write_text(
            json.dumps(misses, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report = {
            "window": [w0, w1],
            "n_peaks_506": len(peaks),
            "n_bp_notes": len(notes),
            "n_hit_506": n_hit,
            "n_miss_506": len(peaks) - n_hit,
            "hit_rate": n_hit / len(peaks) if peaks else 0.0,
            "n_fuse_notes_in_window": len(fuse_w),
            "n_transkun_notes_in_window": len(tk_w),
            "n_506_hit_bp_not_fuse": n_bp_new_vs_fuse,
            "n_506_hit_bp_not_transkun": n_bp_new_vs_tk,
            "n_506_hit_bp_not_fuse_nor_transkun": n_bp_new_vs_both,
            "outputs": {
                "piano_mid": f"{tag}/piano.mid",
                "piano_listen_t0_mid": f"{tag}/piano_listen_t0.mid",
                "snap_hits_mid": f"{tag}/piano_506_snap_hits_only.mid",
                "snap_hits_listen_t0_mid": f"{tag}/piano_506_snap_hits_only_listen_t0.mid",
                "notes_json": f"{tag}/notes.json",
                "misses_json": f"{tag}/misses_506.json",
            },
            "listen_note": (
                "piano.mid uses absolute stem time (silence until window start). "
                "Use *_listen_t0.mid to hear from t=0. Velocity = BP amplitude×127."
            ),
        }
        (wdir / "vs_506.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        window_reports.append(report)
        print(
            f"  {tag}: bp_notes={len(notes)} hit={n_hit}/{len(peaks)} "
            f"new_vs_both={n_bp_new_vs_both}",
            flush=True,
        )

    summary = {
        "model_id": cfg["model_id"],
        "model_version": str(cfg.get("model_version")),
        "tol_s": tol,
        "windows": window_reports,
        "listen": {
            "primary": "t60_90/piano_listen_t0.mid (raw BP, starts at 0)",
            "snap_506_hits": "t60_90/piano_506_snap_hits_only_listen_t0.mid",
            "absolute_time": "t60_90/piano.mid (silent until 60s if playhead at 0)",
            "also_t30_60": "t30_60/piano_listen_t0.mid",
        },
        "bugfix": "velocity was int(amplitude)→1; now amplitude×127. Also emit *_listen_t0.mid.",
    }
    (out_dir / "pitch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # markdown table for README paste
    lines = [
        "# Basic Pitch vs 506",
        "",
        f"tol=±{tol}s · audio=`{cfg['input']['path']}`",
        "",
        "| window | 506 | BP notes | hit | miss | BP hit ∧ ¬fuse | BP hit ∧ ¬TK | BP hit ∧ ¬fuse∧¬TK |",
        "|--------|----:|---------:|----:|-----:|---------------:|-------------:|-------------------:|",
    ]
    for r in window_reports:
        w0, w1 = r["window"]
        lines.append(
            f"| {w0:.0f}–{w1:.0f}s | {r['n_peaks_506']} | {r['n_bp_notes']} | "
            f"{r['n_hit_506']} | {r['n_miss_506']} | {r['n_506_hit_bp_not_fuse']} | "
            f"{r['n_506_hit_bp_not_transkun']} | {r['n_506_hit_bp_not_fuse_nor_transkun']} |"
        )
    lines.append("")
    lines.append(
        "청취: `*_listen_t0.mid` (창 시작=0). `piano.mid`는 절대시각이라 재생헤드 0이면 "
        "창 시작(30/60s)까지 무음. velocity = BP amplitude×127."
    )
    (out_dir / "vs_506.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "track": "clean_amt",
        "stage": "audit_B_basic_pitch_506",
        "config_path": str(cfg_path),
        "model_id": cfg["model_id"],
        "model_version": str(cfg.get("model_version")),
        "audio": {"path": str(audio), "sha256": sha256_file(audio)},
        "peaks": {"path": str(root / cfg["peaks"]["manifest"]), "key": key},
        "summary": summary,
        "note": "AMT explore: Basic Pitch raw notes; 506 snap hits only; no pitch invent.",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_ROOT / "latest_basic_pitch_506.txt").write_text(str(out_dir.resolve()), encoding="utf-8")
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
