"""Perceptual envelope helpers for tilt follow-ups (K-weight / sine lowband).

Fixed rules only — BS.1770-4 K-weighting coeffs via pyloudnorm;
sine lowband cutoff = TILT_PARAMS f_ref_hz (no new free parameters).
"""
from __future__ import annotations

import librosa
import numpy as np
from pyloudnorm.iirfilter import IIRfilter

from io_util import HOP, N_FFT, SR
from passes_tilt import TILT_PARAMS

# BS.1770-4 K-weighting (same as pyloudnorm Meter filter_class="K-weighting")
K_WEIGHT_PARAMS = {
    "high_shelf_gain_db": 4.0,
    "high_shelf_Q": float(1.0 / np.sqrt(2.0)),
    "high_shelf_fc_hz": 1500.0,
    "high_pass_Q": 0.5,
    "high_pass_fc_hz": 38.0,
}


def k_weight_mono(mono: np.ndarray, sample_rate: int = SR) -> np.ndarray:
    """Apply BS.1770-4 K-weighting to a mono signal (high shelf then highpass)."""
    x = np.asarray(mono, dtype=np.float64)
    shelf = IIRfilter(
        K_WEIGHT_PARAMS["high_shelf_gain_db"],
        K_WEIGHT_PARAMS["high_shelf_Q"],
        K_WEIGHT_PARAMS["high_shelf_fc_hz"],
        sample_rate,
        "high_shelf",
    )
    hipass = IIRfilter(
        0.0,
        K_WEIGHT_PARAMS["high_pass_Q"],
        K_WEIGHT_PARAMS["high_pass_fc_hz"],
        sample_rate,
        "high_pass",
    )
    y = shelf.apply_filter(x)
    y = hipass.apply_filter(y)
    return np.asarray(y, dtype=np.float32)


def sine_lowband_mono(
    stereo: np.ndarray,
    *,
    f_ref_hz: float | None = None,
    n_fft: int = N_FFT,
    hop_length: int = HOP,
) -> tuple[np.ndarray, dict]:
    """Reconstruct mono energy from STFT bins with f < f_ref (tilt reference).

    Per channel: zero bins at f >= f_ref, istft, then mean across channels.
    """
    y = np.asarray(stereo, dtype=np.float32)
    f_ref = float(
        TILT_PARAMS["f_ref_hz"] if f_ref_hz is None else f_ref_hz
    )
    freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    keep = freqs < f_ref

    chans = []
    for ch in range(y.shape[1]):
        stft = librosa.stft(
            y[:, ch], n_fft=n_fft, hop_length=hop_length, center=True
        )
        masked = np.zeros_like(stft)
        masked[keep] = stft[keep]
        rec = librosa.istft(
            masked, hop_length=hop_length, length=y.shape[0], center=True
        )
        chans.append(np.asarray(rec, dtype=np.float32))
    low = np.mean(np.column_stack(chans), axis=1).astype(np.float32)
    meta = {
        "f_ref_hz": f_ref,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "n_bins_kept": int(np.count_nonzero(keep)),
        "n_bins_total": int(len(freqs)),
    }
    return low, meta
