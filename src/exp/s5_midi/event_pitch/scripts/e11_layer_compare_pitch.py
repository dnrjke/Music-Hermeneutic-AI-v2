#!/usr/bin/env python3
"""E11: pitch at 506 via separated layers + agreement (no tilt/K warp).

Hypothesis: 506-style material processing for pitch must NOT distort frequency.
Read F0 candidates on piano / HPSS-harmonic / LPC-synthesis, then compare.

Literature anchors (cascade separation → pitch):
  - HPSS as preprocessor for multipitch / pitched transcription
  - Joint / cascade MSS→PE (e.g. MAJL arXiv:2501.03689; Jointist)
  - Benetos et al. harmonic+percussive joint transcription (ICASSP)

Rescue tag: 506 with no fuse note in ±tol (= structural rescue slot).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from _common import (
    REPO_ROOT,
    build_notes_from_pitches,
    hz_to_midi,
    load_config,
    resolve_cfg_path,
    write_midi,
    write_run,
)


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), always_2d=True, dtype="float32")
    return x.mean(axis=1).astype(np.float32), int(sr)


def cqt_salience_pitch(
    mono: np.ndarray,
    sr: int,
    t_onset: float,
    *,
    delay_s: float,
    post_s: float,
    fmin: float,
    fmax: float,
    bins_per_octave: int,
    n_harmonics: int,
    harmonic_decay: float,
    fallback: int,
) -> tuple[int, dict]:
    t0 = t_onset + delay_s
    t1 = t_onset + post_s
    i0 = max(0, int(round(t0 * sr)))
    i1 = min(len(mono), int(round(t1 * sr)))
    if i1 - i0 < int(0.02 * sr):
        return fallback, {"ok": False, "reason": "short"}
    y = mono[i0:i1].astype(np.float32)
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
    mag = np.mean(C, axis=1)
    freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave)
    best = None
    for f0 in freqs:
        if f0 < fmin or f0 > fmax:
            continue
        score = 0.0
        for h in range(1, n_harmonics + 1):
            target = f0 * h
            if target > freqs[-1] * 1.01:
                break
            j = int(np.argmin(np.abs(freqs - target)))
            score += float(mag[j]) * (harmonic_decay ** (h - 1))
        midi = hz_to_midi(float(f0))
        if best is None or score > best[0]:
            best = (score, float(f0), midi)
    if best is None:
        return fallback, {"ok": False, "reason": "empty"}
    return int(best[2]), {"ok": True, "hz": best[1], "score": best[0], "pitch": int(best[2])}


def soft_eq(a: int, b: int, octave_ok: bool) -> bool:
    if a == b:
        return True
    if octave_ok and abs(a - b) == 12:
        return True
    return False


def pick_by_agreement(
    layer_pitch: dict[str, int],
    layer_meta: dict[str, dict],
    octave_ok: bool,
) -> tuple[int, str, dict]:
    p = layer_pitch.get("piano")
    h = layer_pitch.get("harmonic")
    s = layer_pitch.get("synthesis")
    detail = {"layers": layer_pitch, "scores": {k: m.get("score") for k, m in layer_meta.items()}}

    if p is not None and h is not None and soft_eq(p, h, octave_ok):
        # prefer piano pitch on octave soft-agree
        return int(p), "agree_piano_harmonic", detail
    if p is not None and s is not None and soft_eq(p, s, octave_ok):
        return int(p), "agree_piano_synthesis", detail
    if h is not None and s is not None and soft_eq(h, s, octave_ok):
        return int(h), "agree_harmonic_synthesis", detail
    # no pair agree — take piano if ok else best score
    if p is not None and layer_meta.get("piano", {}).get("ok"):
        return int(p), "piano_fallback", detail
    best_name, best_sc = None, -1.0
    for name, m in layer_meta.items():
        if m.get("ok") and float(m.get("score") or -1) > best_sc:
            best_sc = float(m["score"])
            best_name = name
    if best_name is not None:
        return int(layer_pitch[best_name]), f"score_{best_name}", detail
    return 60, "fallback", detail


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="E11 layer-compare pitch @ 506")
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
    indexed = [(i, t) for i, t in enumerate(all_times) if w0 <= t < w1]
    peaks = [t for _, t in indexed]

    layer_paths = {name: root / path for name, path in cfg["layers"].items()}
    for name, p in layer_paths.items():
        if not p.is_file():
            raise SystemExit(f"missing layer {name}: {p}")

    layers = {}
    sr0 = None
    for name, p in layer_paths.items():
        mono, sr = load_mono(p)
        if sr0 is None:
            sr0 = sr
        elif sr != sr0:
            mono = librosa.resample(mono, orig_sr=sr, target_sr=sr0)
        layers[name] = mono
    sr = int(sr0)

    fuse_path = root / cfg["fuse_ro"]["notes_json"]
    fuse = json.loads(fuse_path.read_text(encoding="utf-8"))
    fuse_win = [n for n in fuse if w0 <= float(n["onset_s"]) < w1]
    fuse_tol = float(cfg["fuse_ro"]["tol_s"])

    def is_rescue(t: float) -> bool:
        return not any(abs(float(n["onset_s"]) - t) <= fuse_tol for n in fuse_win)

    pcfg = cfg["pitch"]
    octave_ok = bool(cfg.get("agree", {}).get("octave_ok", True))
    fallback = int(pcfg.get("fallback_pitch") or 60)

    pitches: list[int] = []
    metas: list[dict] = []
    print(f"E11 layer-compare n_peaks={len(peaks)} layers={list(layers)}", flush=True)
    for _pid, t in indexed:
        layer_pitch: dict[str, int] = {}
        layer_meta: dict[str, dict] = {}
        for name, mono in layers.items():
            pitch, meta = cqt_salience_pitch(
                mono,
                sr,
                t,
                delay_s=float(pcfg["delay_s"]),
                post_s=float(pcfg["post_s"]),
                fmin=float(pcfg["fmin_hz"]),
                fmax=float(pcfg["fmax_hz"]),
                bins_per_octave=int(pcfg["bins_per_octave"]),
                n_harmonics=int(pcfg["n_harmonics"]),
                harmonic_decay=float(pcfg["harmonic_decay"]),
                fallback=fallback,
            )
            layer_pitch[name] = pitch
            layer_meta[name] = meta
        pitch, rule, detail = pick_by_agreement(layer_pitch, layer_meta, octave_ok)
        rescue = is_rescue(t)
        pitches.append(pitch)
        metas.append(
            {
                "ok": True,
                "method": "layer_compare",
                "rule": rule,
                "rescue_slot": rescue,
                "agree_ph": soft_eq(layer_pitch.get("piano", -1), layer_pitch.get("harmonic", -2), False),
                "agree_ph_oct": soft_eq(
                    layer_pitch.get("piano", -1), layer_pitch.get("harmonic", -2), True
                ),
                **detail,
            }
        )

    notes, pitch_meta, n_miss = build_notes_from_pitches(
        indexed, pitches, metas, all_times, layers["piano"], sr, cfg
    )
    # split mids
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    run_id = f"{day}_event_pitch_E11_dir_506_layer_compare_t{int(w0)}_{int(w1)}"
    out_dir = write_run(
        run_id=run_id,
        stage="E11",
        method="layer_compare_cqt",
        cfg_path=cfg_path,
        audio_path=layer_paths["piano"],
        peaks_path=root / cfg["peaks"]["manifest"],
        peaks_key=key,
        n_pilot=len(indexed),
        notes=notes,
        pitch_meta=pitch_meta,
        n_miss=n_miss,
        extra_summary={},
        note=(
            "Separated layers (piano/harmonic/synthesis) CQT salience @ 506; "
            "agreement pick; no tilt/K. Lit: HPSS→pitch cascade; MSS→PE."
        ),
    )

    rescue_notes = [n for n, m in zip(notes, pitch_meta) if m.get("rescue_slot")]
    inter_notes = [n for n, m in zip(notes, pitch_meta) if not m.get("rescue_slot")]
    write_midi(rescue_notes, out_dir / "piano_rescue_only.mid")
    write_midi(inter_notes, out_dir / "piano_intersection_only.mid")

    n_agree = sum(1 for m in pitch_meta if str(m.get("rule", "")).startswith("agree_"))
    n_rescue = sum(1 for m in pitch_meta if m.get("rescue_slot"))
    summary = {
        "n_peaks": len(peaks),
        "n_agree_rules": n_agree,
        "agree_rate": n_agree / len(peaks) if peaks else 0.0,
        "n_rescue_slots": n_rescue,
        "n_intersection_slots": len(peaks) - n_rescue,
        "rule_counts": {},
        "literature": [
            "HPSS as preprocessor for pitched analysis / multipitch",
            "Cascade MSS→pitch (MAJL arXiv:2501.03689; Jointist arXiv:2302.00286)",
            "Benetos/Ewert/Weyde harmonic+percussive transcription ICASSP",
        ],
    }
    for m in pitch_meta:
        r = str(m.get("rule"))
        summary["rule_counts"][r] = summary["rule_counts"].get(r, 0) + 1
    (out_dir / "pitch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # patch manifest summary
    man_path = out_dir / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    man["summary"] = summary
    man["layers"] = {k: str(v) for k, v in layer_paths.items()}
    man["listen"] = {
        "all": "piano_from_506.mid",
        "rescue_only": "piano_rescue_only.mid",
        "intersection_only": "piano_intersection_only.mid",
    }
    man_path.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"wrote {out_dir}  agree={n_agree}/{len(peaks)} "
        f"rescue_slots={n_rescue} rules={summary['rule_counts']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
