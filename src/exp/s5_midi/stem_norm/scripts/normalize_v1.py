#!/usr/bin/env python3
"""norm_v1_attack_lowblend: piano sustain duck + lowband perc blend.

No s4_piano imports. Reads Dir stems from repo out/ only.
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
from scipy.signal import butter, sosfiltfilt

REPO_ROOT = Path(__file__).resolve().parents[5]
OUT_ROOT = Path(__file__).resolve().parents[1] / "out"

PIANO = REPO_ROOT / "out" / "stems" / "Dir" / "bs_roformer" / "piano.wav"
PERC = REPO_ROOT / "out" / "stems" / "Dir" / "event_sculpt" / "hpss_percussive.wav"

# Fixed v1 params (manifested)
PARAMS = {
    "recipe": "norm_v1_attack_lowblend",
    "rms_win": 2048,
    "rms_hop": 256,
    "norm_block_s": 2.0,
    "duck_strength": 0.40,  # 0=no duck, 1=full soft-mask
    "otsu_fallback_percentile": 50.0,
    "lowpass_hz": 200.0,
    "perc_blend": 0.30,
    "peak_target": 0.95,
}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _rms_frames(mono: np.ndarray, win: int, hop: int) -> np.ndarray:
    if mono.ndim != 1:
        raise ValueError("mono required")
    n = len(mono)
    if n < win:
        return np.array([float(np.sqrt(np.mean(mono**2) + 1e-12))], dtype=np.float64)
    frames = []
    for i in range(0, n - win + 1, hop):
        frames.append(float(np.sqrt(np.mean(mono[i : i + win] ** 2) + 1e-12)))
    return np.asarray(frames, dtype=np.float64)


def _block_p99_norm(env: np.ndarray, sr_frames: float, block_s: float) -> np.ndarray:
    block = max(1, int(round(block_s * sr_frames)))
    out = np.empty_like(env)
    for i in range(0, len(env), block):
        seg = env[i : i + block]
        p = float(np.percentile(seg, 99)) if seg.size else 1.0
        p = max(p, 1e-12)
        out[i : i + block] = seg / p
    return out


def _otsu(positive: np.ndarray) -> float:
    hist, bin_edges = np.histogram(positive, bins=256)
    hist = hist.astype(np.float64)
    if hist.sum() <= 0:
        return float(np.median(positive))
    prob = hist / hist.sum()
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * (bin_edges[:-1] + bin_edges[1:]) * 0.5)
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    k = int(np.nanargmax(sigma_b))
    return float(0.5 * (bin_edges[k] + bin_edges[k + 1]))


def _interp_to_samples(frames: np.ndarray, n_samples: int, hop: int) -> np.ndarray:
    if frames.size == 0:
        return np.ones(n_samples, dtype=np.float32)
    x_f = np.arange(frames.size, dtype=np.float64) * hop
    x_s = np.arange(n_samples, dtype=np.float64)
    return np.interp(x_s, x_f, frames, left=frames[0], right=frames[-1]).astype(
        np.float32
    )


def sustain_duck(
    stereo: np.ndarray, sr: int, params: dict
) -> tuple[np.ndarray, dict]:
    """Mild soft-mask duck: gain = (1-s) + s * mask; attacks stay near 1."""
    y = np.asarray(stereo, dtype=np.float32)
    mono = y.mean(axis=1)
    win = int(params["rms_win"])
    hop = int(params["rms_hop"])
    strength = float(params["duck_strength"])

    env = _rms_frames(mono, win, hop)
    sr_frames = sr / hop
    env_norm = _block_p99_norm(env, sr_frames, float(params["norm_block_s"]))
    positive = env_norm[env_norm > 0]
    if positive.size < 8:
        mask_frames = np.ones_like(env_norm)
        thr = 1.0
    else:
        thr = _otsu(positive)
        if thr <= 0:
            thr = float(np.percentile(positive, params["otsu_fallback_percentile"]))
        mask_frames = np.clip(env_norm / max(thr, 1e-12), 0.0, 1.0)

    mask = _interp_to_samples(mask_frames, y.shape[0], hop)
    gain = ((1.0 - strength) + strength * mask).astype(np.float32)
    out = (y * gain[:, None]).astype(np.float32)
    meta = {
        "otsu_thr": float(thr),
        "duck_strength": strength,
        "gain_mean": float(gain.mean()),
        "gain_p10": float(np.percentile(gain, 10)),
        "gain_p50": float(np.percentile(gain, 50)),
        "mask_mean": float(mask.mean()),
    }
    return out, meta


def lowpass_stereo(stereo: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    y = np.asarray(stereo, dtype=np.float32)
    nyq = 0.5 * sr
    wn = min(cutoff_hz / nyq, 0.99)
    sos = butter(4, wn, btype="low", output="sos")
    chans = []
    for c in range(y.shape[1]):
        chans.append(sosfiltfilt(sos, y[:, c]).astype(np.float32))
    return np.column_stack(chans).astype(np.float32)


def peak_normalize(stereo: np.ndarray, target: float) -> tuple[np.ndarray, float]:
    peak = float(np.max(np.abs(stereo))) + 1e-12
    scale = target / peak
    return (stereo * scale).astype(np.float32), scale


def normalize_v1(
    piano: np.ndarray, perc: np.ndarray, sr: int, params: dict
) -> tuple[np.ndarray, dict]:
    if piano.shape != perc.shape:
        n = min(piano.shape[0], perc.shape[0])
        piano = piano[:n]
        perc = perc[:n]
        if piano.shape[1] != perc.shape[1]:
            raise RuntimeError("channel mismatch piano vs perc")

    ducked, duck_meta = sustain_duck(piano, sr, params)
    perc_low = lowpass_stereo(perc, sr, float(params["lowpass_hz"]))
    blend = float(params["perc_blend"])
    mixed = (ducked + blend * perc_low).astype(np.float32)
    out, scale = peak_normalize(mixed, float(params["peak_target"]))
    meta = {
        "params": params,
        "duck": duck_meta,
        "perc_blend": blend,
        "lowpass_hz": float(params["lowpass_hz"]),
        "peak_scale": scale,
        "out_peak": float(np.max(np.abs(out))),
        "out_rms": float(np.sqrt(np.mean(out**2))),
    }
    return out, meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="stem_norm v1")
    ap.add_argument("--repo-root", type=Path, default=None)
    ap.add_argument("--piano", type=Path, default=None)
    ap.add_argument("--perc", type=Path, default=None)
    args = ap.parse_args(argv)

    root = (args.repo_root or REPO_ROOT).resolve()
    piano_path = args.piano or (root / PIANO.relative_to(REPO_ROOT))
    perc_path = args.perc or (root / PERC.relative_to(REPO_ROOT))
    if not piano_path.is_file():
        raise SystemExit(f"missing piano: {piano_path}")
    if not perc_path.is_file():
        raise SystemExit(f"missing perc: {perc_path}")

    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    run_id = f"{day}_stem_norm_v1_attack_lowblend"
    out_dir = OUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    piano, sr = sf.read(str(piano_path), always_2d=True, dtype="float32")
    perc, sr_p = sf.read(str(perc_path), always_2d=True, dtype="float32")
    if sr != sr_p:
        raise SystemExit(f"sr mismatch {sr} vs {sr_p}")

    out, proc_meta = normalize_v1(piano, perc, sr, PARAMS)
    out_wav = out_dir / "normalized.wav"
    sf.write(str(out_wav), out, sr, subtype="FLOAT")
    # Stable alias for clean_amt config (repo-relative)
    stable = OUT_ROOT / "normalized_v1.wav"
    sf.write(str(stable), out, sr, subtype="FLOAT")

    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "track": "stem_norm",
        "recipe": PARAMS["recipe"],
        "inputs": {
            "piano": {"path": str(piano_path), "sha256": sha256_file(piano_path)},
            "perc": {"path": str(perc_path), "sha256": sha256_file(perc_path)},
        },
        "outputs": {
            "normalized_wav": "normalized.wav",
            "stable_alias": str(stable.relative_to(root)).replace("\\", "/"),
            "sha256": sha256_file(out_wav),
            "sample_rate": sr,
            "frames": int(out.shape[0]),
            "channels": int(out.shape[1]),
            "duration_s": float(out.shape[0] / sr),
        },
        "process": proc_meta,
        "notes": "No s4 import; harmonic not blended in v1",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {out_dir}  peak={proc_meta['out_peak']:.4f} rms={proc_meta['out_rms']:.4f}")
    print(f"stable {stable}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
