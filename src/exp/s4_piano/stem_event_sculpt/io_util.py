"""Paths and FLOAT WAV I/O for stem event sculpt."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SR = 44100
N_FFT = 2048
HOP = 256

SOURCE_PIANO = ROOT / "out" / "stems" / "Dir" / "bs_roformer" / "piano.wav"
OUTPUT_DIR = ROOT / "out" / "stems" / "Dir" / "event_sculpt"
LISTEN_PEAK_LIMIT = 0.98


def click_wav_name(stem: str, n_peaks: int) -> str:
    """Click sonify filename with peak count (convention from 2026-08-10).

    Examples:
      click_wav_name("lpc_o24_residual_sf_adaptive", 406)
        -> "lpc_o24_residual_sf_adaptive_클릭_p406.wav"
      click_wav_name("foo_클릭", 10)  # strips trailing _클릭 if present
        -> "foo_클릭_p10.wav"
    """
    base = stem
    if base.lower().endswith(".wav"):
        base = base[:-4]
    if base.endswith("_클릭"):
        base = base[: -len("_클릭")]
    n = int(n_peaks)
    if n < 0:
        raise ValueError(f"n_peaks must be >= 0, got {n}")
    return f"{base}_클릭_p{n}.wav"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_stereo(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != SR:
        raise RuntimeError(f"{path}: sample rate {sample_rate} != {SR}")
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise RuntimeError(f"{path}: expected stereo, got shape {audio.shape}")
    return audio.astype(np.float32, copy=False), sample_rate


def write_float_wav(path: Path, audio: np.ndarray, sample_rate: int = SR) -> None:
    """Write FLOAT WAV with libsndfile PEAK timestamp zeroed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.asarray(audio, dtype=np.float32)
    if out.ndim == 1:
        out = out[:, None]
    sf.write(path, out, sample_rate, subtype="FLOAT")
    with path.open("r+b") as handle:
        header = handle.read(12)
        if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise RuntimeError(f"unexpected WAV container: {path}")
        while True:
            chunk_header = handle.read(8)
            if len(chunk_header) < 8:
                break
            chunk_id = chunk_header[:4]
            chunk_size = int.from_bytes(chunk_header[4:], "little")
            payload_start = handle.tell()
            if chunk_id == b"PEAK" and chunk_size >= 8:
                handle.seek(payload_start + 4)
                handle.write(b"\x00\x00\x00\x00")
                return
            handle.seek(payload_start + chunk_size + (chunk_size % 2))


def soft_limit_for_listen(audio: np.ndarray) -> np.ndarray:
    """File-level soft limit only; no internal gain staging."""
    out = np.asarray(audio, dtype=np.float32).copy()
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > LISTEN_PEAK_LIMIT:
        out *= LISTEN_PEAK_LIMIT / peak
    return out


def clip_for_listen(audio: np.ndarray) -> np.ndarray:
    """Hard clip to [-1, 1] without global scale-down (preserves LUFS)."""
    return np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)


def audio_stats(audio: np.ndarray) -> dict[str, float]:
    arr = np.asarray(audio, dtype=np.float64)
    return {
        "peak": float(np.max(np.abs(arr))) if arr.size else 0.0,
        "rms": float(np.sqrt(np.mean(arr * arr))) if arr.size else 0.0,
    }


def write_listening_wav(
    path: Path,
    audio: np.ndarray,
    sample_rate: int = SR,
    *,
    limit_mode: str = "soft_scale",
) -> dict[str, Any]:
    if limit_mode == "soft_scale":
        limited = soft_limit_for_listen(audio)
    elif limit_mode == "clip":
        limited = clip_for_listen(audio)
    else:
        raise ValueError(f"unknown limit_mode={limit_mode}")
    write_float_wav(path, limited, sample_rate)
    stats = audio_stats(limited)
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256_file(path),
        "frames": int(limited.shape[0]),
        "channels": int(limited.shape[1]) if limited.ndim == 2 else 1,
        "duration_s": float(limited.shape[0] / sample_rate),
        "limit_mode": limit_mode,
        **stats,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
