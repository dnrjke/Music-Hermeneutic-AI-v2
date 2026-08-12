#!/usr/bin/env python3
"""E12: pitch @ 506 via independent estimators + octave cross-check.

v1 Music Hermeneutic AI hints (README §11-b/c, L_timbre ASA):
  - Correlated estimators agreeing ≠ truth (E11 CQT×layers failed this way)
  - Cross-validate two *independent* methods; ratio≈2 → octave correct
  - Band lens: restrict F0 to melody band (L_spectral spirit)
  - Old-plus-new: score CQT on post-minus-pre spectral energy

Estimators (different families):
  A) CQT harmonic salience on old-plus-new residual (spectral)
  B) librosa.pyin median voiced F0 (time-domain periodicity)

No import from v1 repo — idea only.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
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


def slice_win(mono: np.ndarray, sr: int, t0: float, t1: float) -> np.ndarray:
    i0 = max(0, int(round(t0 * sr)))
    i1 = min(len(mono), int(round(t1 * sr)))
    if i1 <= i0:
        return np.zeros(0, dtype=np.float32)
    return mono[i0:i1].astype(np.float32)


def cqt_mag(y: np.ndarray, sr: int, fmin: float, fmax: float, bins_per_octave: int) -> tuple[np.ndarray, np.ndarray]:
    if len(y) < int(0.02 * sr):
        return np.zeros(0), np.zeros(0)
    n_bins = int(np.ceil(bins_per_octave * np.log2(fmax / fmin)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        C = np.abs(
            librosa.cqt(
                y,
                sr=sr,
                fmin=fmin,
                n_bins=n_bins,
                bins_per_octave=bins_per_octave,
                hop_length=max(64, max(1, len(y) // 8)),
            )
        )
    mag = np.mean(C, axis=1)
    freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave)
    return mag, freqs


def cqt_delta_salience(
    mono: np.ndarray,
    sr: int,
    t_onset: float,
    *,
    pre_s: float,
    delay_s: float,
    post_s: float,
    fmin: float,
    fmax: float,
    bins_per_octave: int,
    n_harmonics: int,
    harmonic_decay: float,
    fallback: int,
) -> tuple[int, dict]:
    """Old-plus-new: salience on max(0, post_cqt - pre_cqt)."""
    y_pre = slice_win(mono, sr, t_onset - pre_s, t_onset)
    y_post = slice_win(mono, sr, t_onset + delay_s, t_onset + post_s)
    mag_post, freqs = cqt_mag(y_post, sr, fmin, fmax, bins_per_octave)
    if len(freqs) == 0:
        return fallback, {"ok": False, "reason": "short_post", "method": "cqt_delta"}
    mag_pre, _ = cqt_mag(y_pre, sr, fmin, fmax, bins_per_octave)
    if len(mag_pre) == len(mag_post):
        mag = np.maximum(0.0, mag_post - mag_pre)
    else:
        mag = mag_post
    if float(np.sum(mag)) <= 1e-12:
        mag = mag_post  # fall back to post-only
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
        return fallback, {"ok": False, "reason": "empty", "method": "cqt_delta"}
    return int(best[2]), {
        "ok": True,
        "method": "cqt_delta",
        "hz": best[1],
        "score": best[0],
        "pitch": int(best[2]),
    }


def pyin_pitch(
    mono: np.ndarray,
    sr: int,
    t_onset: float,
    *,
    delay_s: float,
    post_s: float,
    fmin: float,
    fmax: float,
    frame_length: int,
    hop_length: int,
    fallback: int,
) -> tuple[int, dict]:
    y = slice_win(mono, sr, t_onset + delay_s, t_onset + post_s)
    if len(y) < frame_length // 2:
        return fallback, {"ok": False, "reason": "short", "method": "pyin"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
        )
    f0 = np.asarray(f0, dtype=np.float64)
    vf = np.asarray(voiced_flag, dtype=bool)
    vp = np.asarray(voiced_probs, dtype=np.float64)
    if vf.any():
        sel = f0[vf]
        conf = float(np.nanmean(vp[vf])) if np.any(np.isfinite(vp[vf])) else 0.0
    else:
        sel = f0[np.isfinite(f0)]
        conf = float(np.nanmean(vp[np.isfinite(f0)])) if np.any(np.isfinite(f0)) else 0.0
    if sel.size == 0:
        return fallback, {"ok": False, "reason": "unvoiced", "method": "pyin", "conf": conf}
    hz = float(np.median(sel))
    pitch = hz_to_midi(hz)
    return int(pitch), {
        "ok": True,
        "method": "pyin",
        "hz": hz,
        "pitch": int(pitch),
        "conf": conf,
        "n_voiced": int(vf.sum()) if vf.size else 0,
    }


def cross_pick(
    p_cqt: int,
    m_cqt: dict,
    p_pyin: int,
    m_pyin: dict,
    *,
    octave_prefer: str,
    mismatch_prefer: str,
    fallback: int,
) -> tuple[int, str, dict]:
    detail = {"cqt": m_cqt, "pyin": m_pyin}
    ok_c = bool(m_cqt.get("ok"))
    ok_p = bool(m_pyin.get("ok"))
    if ok_c and ok_p:
        if p_cqt == p_pyin:
            return p_cqt, "agree_exact", detail
        if abs(p_cqt - p_pyin) == 12:
            # §11-b: ratio≈2 → correct toward preferred independent cue
            if octave_prefer == "pyin":
                return p_pyin, "octave_correct_pyin", detail
            return p_cqt, "octave_correct_cqt", detail
        if mismatch_prefer == "pyin":
            return p_pyin, "mismatch_pyin", detail
        return p_cqt, "mismatch_cqt", detail
    if ok_p:
        return p_pyin, "pyin_only", detail
    if ok_c:
        return p_cqt, "cqt_only", detail
    return fallback, "fallback", detail


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="E12 independent octave pitch @ 506")
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

    audio_path = root / cfg["audio"]["path"]
    vel_path = root / cfg["audio"].get("velocity_path", cfg["audio"]["path"])
    mono, sr = load_mono(audio_path)
    vel_mono, vel_sr = load_mono(vel_path)
    if vel_sr != sr:
        vel_mono = librosa.resample(vel_mono, orig_sr=vel_sr, target_sr=sr)

    fuse_path = root / cfg["fuse_ro"]["notes_json"]
    fuse = json.loads(fuse_path.read_text(encoding="utf-8"))
    fuse_win = [n for n in fuse if w0 <= float(n["onset_s"]) < w1]
    fuse_tol = float(cfg["fuse_ro"]["tol_s"])

    def is_rescue(t: float) -> bool:
        return not any(abs(float(n["onset_s"]) - t) <= fuse_tol for n in fuse_win)

    band = cfg["melody_band"]
    fmin = float(band["fmin_hz"])
    fmax = float(band["fmax_hz"])
    pcfg = cfg["pitch"]
    fallback = int(pcfg.get("fallback_pitch") or 60)
    octave_prefer = str(cfg.get("agree", {}).get("octave_prefer", "pyin"))
    mismatch_prefer = str(cfg.get("agree", {}).get("mismatch_prefer", "pyin"))

    pitches: list[int] = []
    metas: list[dict] = []
    print(f"E12 independent-octave n_peaks={len(peaks)} audio={audio_path.name}", flush=True)
    for _pid, t in indexed:
        p_cqt, m_cqt = cqt_delta_salience(
            mono,
            sr,
            t,
            pre_s=float(pcfg["pre_s"]),
            delay_s=float(pcfg["delay_s"]),
            post_s=float(pcfg["post_s"]),
            fmin=fmin,
            fmax=fmax,
            bins_per_octave=int(pcfg["bins_per_octave"]),
            n_harmonics=int(pcfg["n_harmonics"]),
            harmonic_decay=float(pcfg["harmonic_decay"]),
            fallback=fallback,
        )
        p_pyin, m_pyin = pyin_pitch(
            mono,
            sr,
            t,
            delay_s=float(pcfg["delay_s"]),
            post_s=float(pcfg.get("pyin_post_s", pcfg["post_s"])),
            fmin=fmin,
            fmax=fmax,
            frame_length=int(pcfg["pyin_frame_length"]),
            hop_length=int(pcfg["pyin_hop_length"]),
            fallback=fallback,
        )
        pitch, rule, detail = cross_pick(
            p_cqt,
            m_cqt,
            p_pyin,
            m_pyin,
            octave_prefer=octave_prefer,
            mismatch_prefer=mismatch_prefer,
            fallback=fallback,
        )
        pitches.append(pitch)
        metas.append(
            {
                "ok": rule != "fallback",
                "method": "independent_octave",
                "rule": rule,
                "rescue_slot": is_rescue(t),
                "pitch_cqt": p_cqt,
                "pitch_pyin": p_pyin,
                **detail,
            }
        )

    notes, pitch_meta, n_miss = build_notes_from_pitches(
        indexed, pitches, metas, all_times, vel_mono, sr, cfg
    )
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    run_id = f"{day}_event_pitch_E12_dir_506_indep_octave_t{int(w0)}_{int(w1)}"
    out_dir = write_run(
        run_id=run_id,
        stage="E12",
        method="cqt_delta_x_pyin_octave",
        cfg_path=cfg_path,
        audio_path=audio_path,
        peaks_path=root / cfg["peaks"]["manifest"],
        peaks_key=key,
        n_pilot=len(indexed),
        notes=notes,
        pitch_meta=pitch_meta,
        n_miss=n_miss,
        extra_summary={},
        note=(
            "v1 §11-b hint: independent CQTΔ (old-plus-new) vs pyin; "
            "octave cross-correct; melody band. E11 correlated agree discarded."
        ),
    )

    rescue_notes = [n for n, m in zip(notes, pitch_meta) if m.get("rescue_slot")]
    inter_notes = [n for n, m in zip(notes, pitch_meta) if not m.get("rescue_slot")]
    write_midi(rescue_notes, out_dir / "piano_rescue_only.mid")
    write_midi(inter_notes, out_dir / "piano_intersection_only.mid")

    rule_counts: dict[str, int] = {}
    for m in pitch_meta:
        r = str(m.get("rule"))
        rule_counts[r] = rule_counts.get(r, 0) + 1
    n_oct = rule_counts.get("octave_correct_pyin", 0) + rule_counts.get("octave_correct_cqt", 0)
    n_agree = rule_counts.get("agree_exact", 0)
    n_rescue = sum(1 for m in pitch_meta if m.get("rescue_slot"))
    summary = {
        "n_peaks": len(peaks),
        "n_agree_exact": n_agree,
        "n_octave_correct": n_oct,
        "agree_exact_rate": n_agree / len(peaks) if peaks else 0.0,
        "n_rescue_slots": n_rescue,
        "rule_counts": rule_counts,
        "v1_hints": [
            "§11-b independent cross-validate + octave correct",
            "L-49 correlated agree ≠ truth (why E11 failed)",
            "L_spectral melody-band lens",
            "ASA old-plus-new (pre-subtract for CQT)",
        ],
    }
    (out_dir / "pitch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    man_path = out_dir / "manifest.json"
    man_out = json.loads(man_path.read_text(encoding="utf-8"))
    man_out["summary"] = summary
    man_out["listen"] = {
        "all": "piano_from_506.mid",
        "rescue_only": "piano_rescue_only.mid",
        "intersection_only": "piano_intersection_only.mid",
    }
    man_path.write_text(json.dumps(man_out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"wrote {out_dir}  exact={n_agree}/{len(peaks)} oct_fix={n_oct} "
        f"rescue={n_rescue} rules={rule_counts}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
