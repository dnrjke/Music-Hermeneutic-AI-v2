#!/usr/bin/env python3
"""E8: 506-gated piano(+harmonic) stem → Transkun → clip⊕harmonic-style fuse.

Hypothesis: constrain AMT input energy to 506 attack neighborhoods so detected
notes align with the attack skeleton (without transplanting pitches).

Modes:
  hard_window — silence outside [t-pre, t+post] (B)
  duck        — outside peaks attenuated to duck_gain (D; less harmonic damage)

No imports from via_764 / midi_fuse / clean_amt packages.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from _common import (
    OUT_ROOT,
    REPO_ROOT,
    load_config,
    load_peaks_pilot,
    resolve_cfg_path,
    sha256_file,
    write_midi,
)


def load_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    stereo, sr = sf.read(str(path), always_2d=True, dtype="float32")
    return stereo.mean(axis=1).astype(np.float32), int(sr)


def build_gain(
    n: int,
    sr: int,
    peaks: list[float],
    *,
    mode: str,
    pre_s: float,
    post_s: float,
    duck_gain: float,
    duck_fade_s: float,
) -> np.ndarray:
    if mode == "hard_window":
        g = np.zeros(n, dtype=np.float32)
        for t in peaks:
            i0 = max(0, int(round((t - pre_s) * sr)))
            i1 = min(n, int(round((t + post_s) * sr)))
            if i1 > i0:
                g[i0:i1] = 1.0
        return g

    if mode == "duck":
        g = np.full(n, float(duck_gain), dtype=np.float32)
        fade = max(1, int(round(duck_fade_s * sr)))
        for t in peaks:
            i0 = max(0, int(round((t - pre_s) * sr)))
            i1 = min(n, int(round((t + post_s) * sr)))
            if i1 <= i0:
                continue
            g[i0:i1] = 1.0
            # short linear ramps
            a0 = max(0, i0 - fade)
            a1 = i0
            if a1 > a0:
                ramp = np.linspace(duck_gain, 1.0, a1 - a0, dtype=np.float32)
                g[a0:a1] = np.maximum(g[a0:a1], ramp)
            b0 = i1
            b1 = min(n, i1 + fade)
            if b1 > b0:
                ramp = np.linspace(1.0, duck_gain, b1 - b0, dtype=np.float32)
                g[b0:b1] = np.maximum(g[b0:b1], ramp)
        return g

    raise ValueError(f"unknown gate mode: {mode}")


def midi_to_notes(mid_path: Path) -> list[dict]:
    import mido

    mid = mido.MidiFile(str(mid_path))
    tempo = 500000
    tpb = mid.ticks_per_beat
    abs_tick = 0
    active: dict[int, tuple[float, int]] = {}
    notes: list[dict] = []
    for tr in mid.tracks:
        abs_tick = 0
        for msg in tr:
            abs_tick += msg.time
            if msg.type == "set_tempo":
                tempo = msg.tempo
            if msg.type == "note_on" and msg.velocity > 0:
                t = mido.tick2second(abs_tick, tpb, tempo)
                active[msg.note] = (t, msg.velocity)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                if msg.note in active:
                    t0, vel = active.pop(msg.note)
                    t1 = mido.tick2second(abs_tick, tpb, tempo)
                    notes.append(
                        {
                            "onset_s": float(t0),
                            "offset_s": float(max(t1, t0 + 0.03)),
                            "pitch": int(msg.note),
                            "velocity": int(vel),
                        }
                    )
    notes.sort(key=lambda n: n["onset_s"])
    return notes


def run_transkun(wav: Path, out_mid: Path, device: str) -> list[dict]:
    cmd = [
        sys.executable,
        "-m",
        "transkun.transcribe",
        str(wav),
        str(out_mid),
        "--device",
        device,
    ]
    subprocess.check_call(cmd)
    return midi_to_notes(out_mid)


def shift_notes(notes: list[dict], offset_s: float, source: str) -> list[dict]:
    out = []
    for n in notes:
        out.append(
            {
                "onset_s": float(n["onset_s"]) + offset_s,
                "offset_s": float(n["offset_s"]) + offset_s,
                "pitch": int(n["pitch"]),
                "velocity": int(n.get("velocity", 64)),
                "source": source,
            }
        )
    return out


def fuse_rescue(base: list[dict], rescue: list[dict], tol_s: float) -> tuple[list[dict], dict]:
    fused = [dict(n) for n in base]
    n_add = 0
    for r in rescue:
        hit = False
        for b in base:
            if int(b["pitch"]) != int(r["pitch"]):
                continue
            if abs(float(b["onset_s"]) - float(r["onset_s"])) <= tol_s:
                hit = True
                break
        if not hit:
            fused.append(dict(r))
            n_add += 1
    fused.sort(key=lambda n: (float(n["onset_s"]), int(n["pitch"])))
    return fused, {"n_base": len(base), "n_rescue": len(rescue), "n_added": n_add, "n_fused": len(fused)}


def run_mode(
    *,
    root: Path,
    cfg: dict,
    cfg_path: Path,
    mode: str,
    all_times: list[float],
    peaks_path: Path,
) -> Path:
    w0 = float(cfg["pilot"]["start_s"])
    w1 = float(cfg["pilot"]["end_s"])
    gcfg = cfg["gate"]
    pre = float(gcfg["pre_s"])
    post = float(gcfg["post_s"])
    duck_gain = float(gcfg.get("duck_gain") or 0.01)
    duck_fade = float(gcfg.get("duck_fade_s") or 0.015)
    device = str(cfg.get("transkun", {}).get("device") or "cuda")
    tol = float(cfg.get("fuse", {}).get("tol_s") or 0.03)

    piano_path = root / cfg["audio"]["piano"]
    harm_path = root / cfg["audio"]["harmonic"]
    piano, sr = load_audio_mono(piano_path)
    harm, sr_h = load_audio_mono(harm_path)
    if sr_h != sr:
        raise SystemExit(f"sr mismatch piano={sr} harmonic={sr_h}")

    # Use all 506 peaks (full track) so gate is consistent; AMT on pilot slice
    gain = build_gain(
        len(piano),
        sr,
        all_times,
        mode=mode,
        pre_s=pre,
        post_s=post,
        duck_gain=duck_gain,
        duck_fade_s=duck_fade,
    )
    piano_g = piano * gain
    harm_g = harm * gain

    i0 = int(round(w0 * sr))
    i1 = int(round(w1 * sr))
    piano_win = piano_g[i0:i1]
    harm_win = harm_g[i0:i1]

    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    run_id = f"{day}_event_pitch_E8_dir_506_gate_{mode}_t{int(w0)}_{int(w1)}"
    out_dir = OUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    sf.write(str(out_dir / "piano_gated_full.wav"), piano_g, sr)
    sf.write(str(out_dir / "piano_gated_pilot.wav"), piano_win, sr)
    sf.write(str(out_dir / "harmonic_gated_pilot.wav"), harm_win, sr)
    # gain preview (mono click-like energy)
    sf.write(str(out_dir / "gate_gain_pilot.wav"), gain[i0:i1], sr)

    with tempfile.TemporaryDirectory(prefix=f"e8_{mode}_") as td:
        td_path = Path(td)
        p_wav = td_path / "piano.wav"
        h_wav = td_path / "harm.wav"
        sf.write(str(p_wav), piano_win, sr)
        sf.write(str(h_wav), harm_win, sr)
        print(f"[{mode}] Transkun piano gated...", flush=True)
        p_notes = run_transkun(p_wav, td_path / "piano.mid", device)
        print(f"[{mode}] Transkun harmonic gated...", flush=True)
        h_notes = run_transkun(h_wav, td_path / "harm.mid", device)

    p_abs = shift_notes(p_notes, w0, "gated_piano")
    h_abs = shift_notes(h_notes, w0, "gated_harmonic")
    # keep onsets in pilot window
    p_abs = [n for n in p_abs if w0 <= float(n["onset_s"]) < w1]
    h_abs = [n for n in h_abs if w0 <= float(n["onset_s"]) < w1]
    fused, fuse_stats = fuse_rescue(p_abs, h_abs, tol)

    (out_dir / "notes_piano.json").write_text(
        json.dumps(p_abs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "notes_harmonic.json").write_text(
        json.dumps(h_abs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "notes.json").write_text(
        json.dumps(fused, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_midi(p_abs, out_dir / "piano_only.mid")
    write_midi(fused, out_dir / "piano_from_506.mid")

    # onset proximity to 506
    peak_win = [t for t in all_times if w0 <= t < w1]
    def hit_rate(notes: list[dict], tol: float = 0.05) -> float:
        if not notes:
            return 0.0
        hits = 0
        for n in notes:
            t = float(n["onset_s"])
            if any(abs(t - p) <= tol for p in peak_win):
                hits += 1
        return hits / len(notes)

    summary = {
        "mode": mode,
        "gate": {"pre_s": pre, "post_s": post, "duck_gain": duck_gain if mode == "duck" else None},
        "n_piano": len(p_abs),
        "n_harmonic": len(h_abs),
        "fuse": fuse_stats,
        "frac_notes_near_506_50ms": {
            "piano": hit_rate(p_abs),
            "fused": hit_rate(fused),
        },
    }
    (out_dir / "pitch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "track": "event_pitch",
        "stage": "E8",
        "method": f"gate506_{mode}_transkun_fuse",
        "note": (
            "506-gated stem → Transkun on piano+harmonic pilots → "
            "fuse rescue (same rule as midi_fuse clip⊕harmonic)."
        ),
        "config_path": str(cfg_path),
        "peaks": {"path": str(peaks_path), "key": cfg["peaks"]["key"]},
        "audio": {
            "piano": str(piano_path),
            "harmonic": str(harm_path),
            "piano_sha256": sha256_file(piano_path),
        },
        "summary": summary,
        "outputs": {
            "fused_mid": "piano_from_506.mid",
            "piano_only_mid": "piano_only.mid",
            "piano_gated_pilot": "piano_gated_pilot.wav",
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[{mode}] wrote {out_dir}  piano={len(p_abs)} harm={len(h_abs)} "
        f"fused={len(fused)} near506={summary['frac_notes_near_506_50ms']}"
    )
    return out_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="event_pitch E8 506-gated AMT")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, default=None)
    ap.add_argument(
        "--modes",
        default="hard_window,duck",
        help="comma list: hard_window,duck",
    )
    args = ap.parse_args(argv)

    root = (args.repo_root or REPO_ROOT).resolve()
    cfg_path = resolve_cfg_path(args.config)
    cfg = load_config(cfg_path)
    all_times, _indexed, peaks_path = load_peaks_pilot(root, cfg)
    # load_peaks_pilot filters indexed to pilot; we need ALL times for gate
    man = json.loads(peaks_path.read_text(encoding="utf-8"))
    all_times = [float(t) for t in man["peak_times_s"][cfg["peaks"]["key"]]]

    modes = [m.strip() for m in str(args.modes).split(",") if m.strip()]
    paths = []
    for mode in modes:
        paths.append(run_mode(root=root, cfg=cfg, cfg_path=cfg_path, mode=mode, all_times=all_times, peaks_path=peaks_path))
    (OUT_ROOT / "latest_e8.txt").write_text(
        "\n".join(str(p) for p in paths), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
