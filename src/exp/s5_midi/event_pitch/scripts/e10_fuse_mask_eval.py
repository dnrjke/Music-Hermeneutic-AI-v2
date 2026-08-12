#!/usr/bin/env python3
"""E10: 506 evaluation with pitch — clip⊕harmonic RO × masks.

Significance: first way to hear 506 *with* pitch using go-locked fuse pitches.
Limit: only 506 peaks that intersect fuse get a note → true 506 recall has gaps.

Event count:
  - 506 peak anchors: unchanged (n_peaks)
  - MIDI notes (1:1 pick): n_matched ≤ n_peaks  (empties = fuse miss)
  - MIDI notes (poly): can exceed n_peaks

No package imports from midi_fuse — path RO only.
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
from scipy.ndimage import maximum_filter1d

from _common import OUT_ROOT, REPO_ROOT, load_config, resolve_cfg_path, sha256_file, write_midi


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), always_2d=True, dtype="float32")
    return x.mean(axis=1).astype(np.float32), int(sr)


def normalize01(x: np.ndarray) -> np.ndarray:
    x = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
    pos = x[x > 0]
    hi = float(np.percentile(pos, 99.5)) if len(pos) else 1.0
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
    gate = soft506_mask(len(mono), sr, peaks, pre_s, post_s)
    y = mono * (0.05 + 0.95 * gate)
    hop = 256
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    env = np.interp(
        np.arange(len(mono)),
        librosa.frames_to_samples(np.arange(len(rms)), hop_length=hop),
        rms,
    ).astype(np.float32)
    return normalize01(env * gate)


def superflux_envelope(mono: np.ndarray, sr: int) -> np.ndarray:
    hop = 256
    S = np.abs(librosa.stft(mono, n_fft=2048, hop_length=hop)) ** 2
    mel = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=128, fmin=27.5)
    logmel = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
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


def mask_at(mask: np.ndarray, sr: int, t: float) -> float:
    i = int(np.clip(round(t * sr), 0, len(mask) - 1))
    return float(mask[i])


def pick_one_per_506(
    fuse: list[dict],
    peaks: list[float],
    mask: np.ndarray,
    sr: int,
    tol_s: float,
) -> tuple[list[dict], list[dict], dict]:
    """Return (notes, miss_meta, stats). Onset snapped to 506 time for eval mid."""
    notes: list[dict] = []
    misses: list[dict] = []
    used: set[int] = set()
    for t in peaks:
        cands: list[tuple[float, int, dict]] = []
        for i, n in enumerate(fuse):
            if abs(float(n["onset_s"]) - t) > tol_s:
                continue
            m = mask_at(mask, sr, float(n["onset_s"]))
            score = m * (float(n.get("velocity", 64)) / 127.0)
            cands.append((score, i, n))
        if not cands:
            misses.append({"onset_s": t, "reason": "no_fuse_in_tol"})
            continue
        cands.sort(key=lambda x: -x[0])
        chosen = None
        for sc, i, n in cands:
            if i not in used:
                chosen = (sc, i, n)
                break
        if chosen is None:
            # all candidates already used — still take best (allow multi-peak share)
            sc, i, n = cands[0]
            chosen = (sc, i, n)
        sc, i, n = chosen
        used.add(i)
        dur = max(0.04, float(n["offset_s"]) - float(n["onset_s"]))
        notes.append(
            {
                "onset_s": t,  # snap to 506 for event-aligned eval
                "offset_s": t + dur,
                "pitch": int(n["pitch"]),
                "velocity": int(n.get("velocity", 64)),
                "source": f"fuse:{n.get('source', '?')}",
                "fuse_onset_s": float(n["onset_s"]),
                "mask": mask_at(mask, sr, float(n["onset_s"])),
                "score": float(sc),
                "anchor_506": t,
            }
        )
    stats = {
        "n_peaks": len(peaks),
        "n_matched": len(notes),
        "n_miss": len(misses),
        "match_rate": len(notes) / len(peaks) if peaks else 0.0,
        "note": "n_peaks fixed; MIDI 1:1 count = n_matched (misses lack fuse pitch)",
    }
    return notes, misses, stats


def pick_poly(
    fuse: list[dict],
    peaks: list[float],
    mask: np.ndarray,
    sr: int,
    tol_s: float,
) -> tuple[list[dict], dict]:
    """All fuse notes within tol of any 506; onset kept as fuse time."""
    out: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for n in fuse:
        t = float(n["onset_s"])
        near = [p for p in peaks if abs(t - p) <= tol_s]
        if not near:
            continue
        key = (int(round(t * 1000)), int(n["pitch"]))
        if key in seen:
            continue
        seen.add(key)
        m = mask_at(mask, sr, t)
        out.append(
            {
                **{k: n[k] for k in ("onset_s", "offset_s", "pitch", "velocity")},
                "source": f"fuse_poly:{n.get('source', '?')}",
                "mask": m,
                "score": m * (float(n.get("velocity", 64)) / 127.0),
                "anchors_506": near,
            }
        )
    out.sort(key=lambda x: float(x["onset_s"]))
    stats = {
        "n_peaks": len(peaks),
        "n_notes": len(out),
        "note": "poly: note count can exceed n_peaks; peaks unchanged",
    }
    return out, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="E10 fuse×mask 506 eval")
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

    fuse_path = root / cfg["fuse"]["notes_json"]
    fuse_all = json.loads(fuse_path.read_text(encoding="utf-8"))
    fuse = [n for n in fuse_all if w0 <= float(n["onset_s"]) < w1]

    piano, sr = load_mono(root / cfg["audio"]["piano"])
    k_env, sr_k = load_mono(root / cfg["audio"]["k_env"])
    if sr_k != sr:
        k_env = librosa.resample(k_env, orig_sr=sr_k, target_sr=sr)

    pre = float(cfg["gate_for_stem_env"]["pre_s"])
    post = float(cfg["gate_for_stem_env"]["post_s"])
    print("building masks...", flush=True)
    masks = {
        "soft506": soft506_mask(len(piano), sr, all_times, pre, post),
        "gated_stem_env": gated_stem_envelope(piano, sr, all_times, pre, post),
    }
    masks["sf_kenv_x_gated"] = normalize01(
        superflux_envelope(k_env, sr) * masks["gated_stem_env"]
    )

    tol = float(cfg["select"]["tol_s"])
    also_poly = bool(cfg["select"].get("also_poly", True))
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    run_id = f"{day}_event_pitch_E10_dir_506_fuse_mask_t{int(w0)}_{int(w1)}"
    out_dir = OUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "n_506_peaks_pilot": len(peaks),
        "n_fuse_notes_pilot": len(fuse),
        "tol_s": tol,
        "event_count_policy": {
            "peaks": "fixed = conservative_kenv_agree_only in window",
            "midi_1to1": "n_matched ≤ n_peaks; misses = no fuse in ±tol",
            "midi_poly": "can exceed n_peaks",
        },
        "masks": {},
    }

    want = list(cfg.get("masks", {}).get("names") or masks.keys())
    primary_name = "soft506"
    for name in want:
        m = masks[name]
        notes, misses, stats = pick_one_per_506(fuse, peaks, m, sr, tol)
        write_midi(notes, out_dir / f"eval_1to1_{name}.mid")
        (out_dir / f"notes_1to1_{name}.json").write_text(
            json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_dir / f"misses_1to1_{name}.json").write_text(
            json.dumps(misses, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary["masks"][name] = {"one_to_one": stats}
        print(
            f"  1:1 ×{name}: matched={stats['n_matched']}/{stats['n_peaks']} "
            f"miss={stats['n_miss']} ({100*stats['match_rate']:.0f}%)",
            flush=True,
        )
        if also_poly:
            poly, pst = pick_poly(fuse, peaks, m, sr, tol)
            write_midi(poly, out_dir / f"eval_poly_{name}.mid")
            (out_dir / f"notes_poly_{name}.json").write_text(
                json.dumps(poly, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            summary["masks"][name]["poly"] = pst
            print(f"  poly×{name}: n={pst['n_notes']}", flush=True)

    # primary listen = soft506 1:1 (clearest 506-tied selector)
    import shutil

    src_mid = out_dir / f"eval_1to1_{primary_name}.mid"
    shutil.copy(src_mid, out_dir / "piano_from_506.mid")
    shutil.copy(out_dir / f"notes_1to1_{primary_name}.json", out_dir / "notes.json")

    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "track": "event_pitch",
        "stage": "E10",
        "method": "fuse_ro_x_mask_506_eval",
        "note": (
            "506 eval with pitch via clip⊕harmonic RO. "
            "Gaps where fuse has no note near 506. Peak count fixed."
        ),
        "fuse_notes": str(fuse_path),
        "fuse_sha256": sha256_file(fuse_path),
        "config_path": str(cfg_path),
        "summary": summary,
        "listen_primary": f"piano_from_506.mid (= eval_1to1_{primary_name})",
        "listen_alts": [
            "eval_1to1_gated_stem_env.mid",
            "eval_1to1_sf_kenv_x_gated.mid",
            "eval_poly_soft506.mid",
        ],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "pitch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_ROOT / "latest_e10.txt").write_text(str(out_dir.resolve()), encoding="utf-8")
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
