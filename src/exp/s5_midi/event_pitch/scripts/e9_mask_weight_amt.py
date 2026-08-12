#!/usr/bin/env python3
"""E9: Use attack/pitch masks as weights on FULL-stem AMT (veil-nebula style).

E8 ran AMT on gated audio ≈ cutting existing transcription — weak.
Here the full stem keeps harmonics; masks only weight/select:

  masks (time series):
    gated_stem_env — envelope of soft-506-gated piano (pitch-local audio as weight)
    soft506        — peak kernels
    sf_piano       — SuperFlux(piano)
    sf_kenv        — SuperFlux(k_env material) — often better attack mask
    sf_kenv_x_gated — product

  apply:
    1) Transkun on full piano pilot → per-506 pick note by vel*mask
    2) optional ByteDance onset × mask → argmax @ 506

Analogy (veil baseline): nebula/detail map weights compositing; never replace
the source photo with the mask alone.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from scipy.ndimage import maximum_filter1d

from _common import (
    OUT_ROOT,
    REPO_ROOT,
    argmax_pitch_vector,
    load_config,
    resolve_cfg_path,
    sha256_file,
    write_midi,
)


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), always_2d=True, dtype="float32")
    return x.mean(axis=1).astype(np.float32), int(sr)


def normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.maximum(x, 0.0)
    hi = float(np.percentile(x[x > 0], 99.5)) if np.any(x > 0) else 1.0
    if hi < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip(x / hi, 0.0, 1.0).astype(np.float32)


def soft506_mask(n: int, sr: int, peaks: list[float], pre_s: float, post_s: float) -> np.ndarray:
    g = np.zeros(n, dtype=np.float32)
    for t in peaks:
        i0 = max(0, int(round((t - pre_s) * sr)))
        i1 = min(n, int(round((t + post_s) * sr)))
        if i1 <= i0:
            continue
        w = np.hanning(i1 - i0).astype(np.float32)
        g[i0:i1] = np.maximum(g[i0:i1], w)
    return normalize01(g)


def gated_stem_envelope(
    mono: np.ndarray, sr: int, peaks: list[float], pre_s: float, post_s: float
) -> np.ndarray:
    """Pitch-bearing local audio → RMS envelope (veil 'detail map' role)."""
    gate = soft506_mask(len(mono), sr, peaks, pre_s, post_s)
    # keep a floor so envelope tracks gated energy, not absolute silence pops
    y = mono * (0.05 + 0.95 * gate)
    hop = 256
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    env = np.interp(
        np.arange(len(mono)),
        librosa.frames_to_samples(np.arange(len(rms)), hop_length=hop),
        rms,
    ).astype(np.float32)
    return normalize01(env * gate)  # emphasize attack neighborhoods


def superflux_envelope(mono: np.ndarray, sr: int) -> np.ndarray:
    hop = 256
    S = np.abs(librosa.stft(mono, n_fft=2048, hop_length=hop)) ** 2
    mel = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=128, fmin=27.5)
    logmel = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    # SuperFlux-ish: max-filter along freq then positive diff
    ref = maximum_filter1d(logmel, size=3, axis=0, mode="nearest")
    lag = 2
    diff = logmel[:, lag:] - ref[:, :-lag]
    sf = np.maximum(0.0, diff).sum(axis=0)
    sf = np.pad(sf, (lag, 0))
    env = np.interp(
        np.arange(len(mono)),
        librosa.frames_to_samples(np.arange(len(sf)), hop_length=hop),
        sf,
    ).astype(np.float32)
    return normalize01(env)


def mask_at_time(mask: np.ndarray, sr: int, t: float) -> float:
    i = int(np.clip(round(t * sr), 0, len(mask) - 1))
    return float(mask[i])


def midi_to_notes(mid_path: Path) -> list[dict]:
    import mido

    mid = mido.MidiFile(str(mid_path))
    tempo = 500000
    tpb = mid.ticks_per_beat
    notes: list[dict] = []
    for tr in mid.tracks:
        abs_tick = 0
        active: dict[int, tuple[float, int]] = {}
        for msg in tr:
            abs_tick += msg.time
            if msg.type == "set_tempo":
                tempo = msg.tempo
            if msg.type == "note_on" and msg.velocity > 0:
                active[msg.note] = (mido.tick2second(abs_tick, tpb, tempo), msg.velocity)
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
    subprocess.check_call(
        [sys.executable, "-m", "transkun.transcribe", str(wav), str(out_mid), "--device", device]
    )
    return midi_to_notes(out_mid)


def select_by_mask(
    amt_notes: list[dict],
    peaks: list[float],
    mask: np.ndarray,
    sr: int,
    tol_s: float,
    rescue_q: float,
) -> tuple[list[dict], dict]:
    """Per-506: pick AMT note in ±tol maximizing velocity * mask(t)."""
    chosen: list[dict] = []
    used = set()
    n_empty = 0
    for t in peaks:
        cands = []
        for i, n in enumerate(amt_notes):
            if abs(float(n["onset_s"]) - t) <= tol_s:
                m = mask_at_time(mask, sr, float(n["onset_s"]))
                score = m * (float(n.get("velocity", 64)) / 127.0)
                cands.append((score, m, i, n))
        if not cands:
            n_empty += 1
            continue
        cands.sort(key=lambda x: -x[0])
        _sc, _m, idx, n = cands[0]
        if idx in used:
            # allow same note for one peak only
            alt = next((c for c in cands if c[2] not in used), None)
            if alt is None:
                n_empty += 1
                continue
            _sc, _m, idx, n = alt
        used.add(idx)
        out = dict(n)
        out["source"] = "amt_mask_pick"
        out["mask"] = _m
        out["score"] = _sc
        out["anchor_506"] = t
        chosen.append(out)

    # rescue: high-mask AMT notes not used
    mask_at_peaks = [mask_at_time(mask, sr, t) for t in peaks]
    thr = float(np.quantile(mask_at_peaks, rescue_q)) if mask_at_peaks else 0.5
    n_rescue = 0
    for i, n in enumerate(amt_notes):
        if i in used:
            continue
        m = mask_at_time(mask, sr, float(n["onset_s"]))
        if m >= thr and any(abs(float(n["onset_s"]) - t) <= tol_s for t in peaks):
            out = dict(n)
            out["source"] = "amt_mask_rescue"
            out["mask"] = m
            chosen.append(out)
            used.add(i)
            n_rescue += 1
    chosen.sort(key=lambda n: float(n["onset_s"]))
    stats = {
        "n_peaks": len(peaks),
        "n_chosen": len(chosen),
        "n_empty_peaks": n_empty,
        "n_rescue": n_rescue,
        "rescue_thr": thr,
    }
    return chosen, stats


def ensure_bd_ckpt(weights_dir: Path) -> Path:
    weights_dir.mkdir(parents=True, exist_ok=True)
    dest = weights_dir / "note_F1=0.9677_pedal_F1=0.9186.pth"
    if dest.is_file() and dest.stat().st_size > 1.6e8:
        return dest
    url = (
        "https://zenodo.org/record/4034264/files/"
        "CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
    )
    print(f"downloading {dest.name}...", flush=True)
    urllib.request.urlretrieve(url, dest)
    return dest


def bytedance_onset(wav_path: Path, ckpt: Path, device_str: str) -> tuple[np.ndarray, float]:
    from piano_transcription_inference import PianoTranscription, config as pt_config
    from piano_transcription_inference import sample_rate

    audio, _ = librosa.core.load(str(wav_path), sr=sample_rate, mono=True)
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    tr = PianoTranscription(checkpoint_path=str(ckpt), device=device)
    out = tr.transcribe(audio, midi_path=None)
    onset = np.asarray(out["output_dict"]["reg_onset_output"], dtype=np.float64)
    spf = 1.0 / float(pt_config.frames_per_second)
    return onset, spf


def onset_argmax_at_peaks(
    onset: np.ndarray,
    spf: float,
    peaks: list[float],
    mask: np.ndarray,
    sr: int,
    clip0: float,
    half: int,
) -> list[dict]:
    midi0 = 21
    notes = []
    for t_abs in peaks:
        t_rel = t_abs - clip0
        # weight frames by mask
        f = int(round(t_rel / spf))
        f0 = max(0, f - half)
        f1 = min(onset.shape[0], f + half + 1)
        # temporal weight from mask at corresponding times
        vec = np.zeros(onset.shape[1], dtype=np.float64)
        for fi in range(f0, f1):
            t = clip0 + fi * spf
            w = mask_at_time(mask, sr, t)
            vec += onset[fi] * (0.15 + 0.85 * w)
        pitch, meta = argmax_pitch_vector(vec, midi0, 21, 108, 60, "onset_x_mask")
        if not meta.get("ok"):
            continue
        notes.append(
            {
                "onset_s": t_abs,
                "offset_s": t_abs + 0.2,
                "pitch": int(pitch),
                "velocity": int(np.clip(40 + 80 * meta.get("val", 0), 1, 127)),
                "source": "onset_x_mask",
                "mask": mask_at_time(mask, sr, t_abs),
                **{k: meta[k] for k in ("val", "top3") if k in meta},
            }
        )
    return notes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="event_pitch E9 mask×full-stem")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, default=None)
    args = ap.parse_args(argv)

    root = (args.repo_root or REPO_ROOT).resolve()
    cfg_path = resolve_cfg_path(args.config)
    cfg = load_config(cfg_path)

    man = json.loads((root / cfg["peaks"]["manifest"]).read_text(encoding="utf-8"))
    key = cfg["peaks"]["key"]
    all_times = [float(t) for t in man["peak_times_s"][key]]
    w0 = float(cfg["pilot"]["start_s"])
    w1 = float(cfg["pilot"]["end_s"])
    peaks = [t for t in all_times if w0 <= t < w1]

    piano, sr = load_mono(root / cfg["audio"]["piano"])
    k_env, sr_k = load_mono(root / cfg["audio"]["k_env"])
    if sr_k != sr:
        k_env = librosa.resample(k_env, orig_sr=sr_k, target_sr=sr)

    gcfg = cfg["gate_for_stem_env"]
    pre, post = float(gcfg["pre_s"]), float(gcfg["post_s"])

    print("building masks...", flush=True)
    masks: dict[str, np.ndarray] = {
        "soft506": soft506_mask(len(piano), sr, all_times, pre, post),
        "gated_stem_env": gated_stem_envelope(piano, sr, all_times, pre, post),
        "sf_piano": superflux_envelope(piano, sr),
        "sf_kenv": superflux_envelope(k_env, sr),
    }
    masks["sf_kenv_x_gated"] = normalize01(masks["sf_kenv"] * masks["gated_stem_env"])

    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    run_id = f"{day}_event_pitch_E9_dir_506_mask_weight_t{int(w0)}_{int(w1)}"
    out_dir = OUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = out_dir / "masks"
    mask_dir.mkdir(exist_ok=True)
    i0, i1 = int(round(w0 * sr)), int(round(w1 * sr))
    for name, m in masks.items():
        sf.write(str(mask_dir / f"{name}_pilot.wav"), m[i0:i1], sr)

    # Full-stem Transkun on pilot (no gate on audio)
    piano_win = piano[i0:i1]
    device = str(cfg.get("transkun", {}).get("device") or "cuda")
    with tempfile.TemporaryDirectory(prefix="e9_amt_") as td:
        tw = Path(td) / "piano_full.wav"
        sf.write(str(tw), piano_win, sr)
        print("Transkun FULL piano pilot...", flush=True)
        amt_local = run_transkun(tw, Path(td) / "raw.mid", device)
    amt = [
        {
            "onset_s": float(n["onset_s"]) + w0,
            "offset_s": float(n["offset_s"]) + w0,
            "pitch": int(n["pitch"]),
            "velocity": int(n.get("velocity", 64)),
            "source": "transkun_full",
        }
        for n in amt_local
        if w0 <= float(n["onset_s"]) + w0 < w1
    ]
    (out_dir / "notes_amt_full.json").write_text(
        json.dumps(amt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_midi(amt, out_dir / "amt_full.mid")

    tol = float(cfg["select"]["tol_s"])
    rescue_q = float(cfg["select"]["rescue_quantile"])
    summary: dict = {"n_amt_full": len(amt), "n_peaks": len(peaks), "masks": {}}

    want = list(cfg.get("masks", {}).get("names") or masks.keys())
    for name in want:
        if name not in masks:
            continue
        m = masks[name]
        notes, stats = select_by_mask(amt, peaks, m, sr, tol, rescue_q)
        write_midi(notes, out_dir / f"piano_from_506_amt_{name}.mid")
        (out_dir / f"notes_amt_{name}.json").write_text(
            json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary["masks"][name] = {"path": "amt_select", **stats}
        print(f"  amt×{name}: n={len(notes)} empty_peaks={stats['n_empty_peaks']}", flush=True)

    # Onset × mask @ 506
    if bool(cfg.get("onset_reweight", {}).get("enable", True)):
        ckpt = ensure_bd_ckpt(OUT_ROOT.parent / "weights")
        with tempfile.TemporaryDirectory(prefix="e9_on_") as td:
            # slightly padded clip for BD
            pad = 1.0
            c0, c1 = max(0.0, w0 - pad), min(len(piano) / sr, w1 + pad)
            j0, j1 = int(round(c0 * sr)), int(round(c1 * sr))
            cw = Path(td) / "clip.wav"
            sf.write(str(cw), piano[j0:j1], sr)
            print("ByteDance onset on FULL clip...", flush=True)
            onset, spf = bytedance_onset(cw, ckpt, device)
        half = int(cfg.get("onset_reweight", {}).get("half_win_frames") or 1)
        for name in ("sf_kenv_x_gated", "gated_stem_env", "sf_kenv"):
            if name not in masks:
                continue
            notes = onset_argmax_at_peaks(
                onset, spf, peaks, masks[name], sr, c0, half
            )
            write_midi(notes, out_dir / f"piano_from_506_onset_{name}.mid")
            (out_dir / f"notes_onset_{name}.json").write_text(
                json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            summary["masks"].setdefault(name, {})["onset_n"] = len(notes)
            print(f"  onset×{name}: n={len(notes)}", flush=True)

    # Default listen target: best structural mask product
    default = out_dir / "piano_from_506_amt_sf_kenv_x_gated.mid"
    if default.is_file():
        import shutil

        shutil.copy(default, out_dir / "piano_from_506.mid")

    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "track": "event_pitch",
        "stage": "E9",
        "method": "mask_weight_full_stem",
        "note": (
            "Veil-style: full stem AMT/onset; masks (gated-stem env, SF k_env, …) "
            "only weight/select. Not AMT-on-gated (E8)."
        ),
        "config_path": str(cfg_path),
        "audio": {
            "piano": str(root / cfg["audio"]["piano"]),
            "k_env": str(root / cfg["audio"]["k_env"]),
            "piano_sha256": sha256_file(root / cfg["audio"]["piano"]),
        },
        "summary": summary,
        "listen_primary": "piano_from_506.mid (= amt×sf_kenv_x_gated)",
        "listen_alts": [
            "piano_from_506_amt_gated_stem_env.mid",
            "piano_from_506_onset_sf_kenv_x_gated.mid",
            "amt_full.mid",
        ],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "pitch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_ROOT / "latest_e9.txt").write_text(str(out_dir.resolve()), encoding="utf-8")
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
