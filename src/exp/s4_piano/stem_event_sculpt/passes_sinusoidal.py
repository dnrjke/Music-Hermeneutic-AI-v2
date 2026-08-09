"""Sinusoidal residual pass: fixed spectral peak tonal reconstruction."""
from __future__ import annotations

import librosa
import numpy as np

from io_util import HOP, N_FFT

SINE_PARAMS = {
    "n_fft": N_FFT,
    "hop_length": HOP,
    "percentile": 90.0,
}


def _tonal_channel(mono: np.ndarray) -> np.ndarray:
    n_fft = SINE_PARAMS["n_fft"]
    hop = SINE_PARAMS["hop_length"]
    percentile = SINE_PARAMS["percentile"]

    y = np.asarray(mono, dtype=np.float32)
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop, center=True)
    mag = np.abs(stft)
    phase = np.angle(stft)

    # Frequency-axis local maxima (interior bins only)
    left = mag[1:-1] > mag[:-2]
    right = mag[1:-1] >= mag[2:]
    local_max = np.zeros_like(mag, dtype=bool)
    local_max[1:-1] = left & right

    # Frame-wise magnitude threshold at fixed percentile
    # mag shape: (freq, time)
    thresh = np.percentile(mag, percentile, axis=0, keepdims=True)
    keep = local_max & (mag >= thresh)

    tonal_stft = np.zeros_like(stft)
    tonal_stft[keep] = mag[keep] * np.exp(1j * phase[keep])

    tonal = librosa.istft(
        tonal_stft,
        hop_length=hop,
        length=len(y),
        center=True,
    )
    return np.asarray(tonal, dtype=np.float32)


def sinusoidal_components(stereo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (tonal, residual=y-tonal), same shape as stereo float32."""
    y = np.asarray(stereo, dtype=np.float32)
    tonals = []
    for ch in range(y.shape[1]):
        tonals.append(_tonal_channel(y[:, ch]))
    tonal = np.column_stack(tonals).astype(np.float32)
    residual = (y - tonal).astype(np.float32)
    if tonal.shape != y.shape or residual.shape != y.shape:
        raise RuntimeError(
            f"Sine shape mismatch: in={y.shape} T={tonal.shape} R={residual.shape}"
        )
    return tonal, residual
