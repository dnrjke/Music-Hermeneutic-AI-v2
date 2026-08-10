"""Spectral tilt: attenuate lows, boost highs (fixed power-law on STFT mag)."""
from __future__ import annotations

import librosa
import numpy as np

from io_util import HOP, N_FFT, SR

TILT_PARAMS = {
    "n_fft": N_FFT,
    "hop_length": HOP,
    "f_ref_hz": 1000.0,
    "f_floor_hz": 80.0,
    "alpha": 1.0,  # |S| *= (f/f_ref)^alpha ; alpha>0 → high up, low down
}


def spectral_tilt(stereo: np.ndarray) -> tuple[np.ndarray, dict]:
    """Apply fixed frequency tilt independently per channel. Shape preserved."""
    y = np.asarray(stereo, dtype=np.float32)
    n_fft = TILT_PARAMS["n_fft"]
    hop = TILT_PARAMS["hop_length"]
    f_ref = TILT_PARAMS["f_ref_hz"]
    f_floor = TILT_PARAMS["f_floor_hz"]
    alpha = TILT_PARAMS["alpha"]

    freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    f_eff = np.maximum(freqs, f_floor)
    gain = (f_eff / f_ref) ** alpha
    gain = gain.astype(np.float64)
    # DC bin: keep small
    gain[0] = gain[1] if len(gain) > 1 else 1.0

    out_ch = []
    for ch in range(y.shape[1]):
        stft = librosa.stft(y[:, ch], n_fft=n_fft, hop_length=hop, center=True)
        tilted = stft * gain[:, None]
        rec = librosa.istft(
            tilted, hop_length=hop, length=y.shape[0], center=True
        )
        out_ch.append(np.asarray(rec, dtype=np.float32))

    out = np.column_stack(out_ch).astype(np.float32)
    meta = {
        "f_ref_hz": f_ref,
        "f_floor_hz": f_floor,
        "alpha": alpha,
        "gain_at_100hz": float((max(100.0, f_floor) / f_ref) ** alpha),
        "gain_at_1000hz": 1.0,
        "gain_at_4000hz": float((4000.0 / f_ref) ** alpha),
    }
    return out, meta
