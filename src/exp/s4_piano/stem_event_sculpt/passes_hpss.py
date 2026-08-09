"""HPSS pass: fixed anisotropic median separation."""
from __future__ import annotations

import librosa
import numpy as np

from io_util import HOP, N_FFT

HPSS_PARAMS = {
    "kernel_size": 31,
    "power": 2.0,
    "margin": 1.0,
    "n_fft": N_FFT,
    "hop_length": HOP,
}


def hpss_components(stereo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (harmonic, percussive), same shape as stereo float32."""
    y = np.asarray(stereo, dtype=np.float32)
    # librosa expects shape (n,) or (channels, n)
    y_t = y.T
    harmonic_t, percussive_t = librosa.effects.hpss(
        y_t,
        kernel_size=HPSS_PARAMS["kernel_size"],
        power=HPSS_PARAMS["power"],
        margin=HPSS_PARAMS["margin"],
        n_fft=HPSS_PARAMS["n_fft"],
        hop_length=HPSS_PARAMS["hop_length"],
    )
    harmonic = np.asarray(harmonic_t.T, dtype=np.float32)
    percussive = np.asarray(percussive_t.T, dtype=np.float32)
    if harmonic.shape != y.shape or percussive.shape != y.shape:
        raise RuntimeError(
            f"HPSS shape mismatch: in={y.shape} H={harmonic.shape} P={percussive.shape}"
        )
    return harmonic, percussive
