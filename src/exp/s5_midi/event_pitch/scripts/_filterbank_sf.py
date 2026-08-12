"""Filterbank SuperFlux (2D) — Böck-style, no frequency sum until MIDI map."""
from __future__ import annotations

import librosa
import numpy as np
from scipy.ndimage import maximum_filter1d

from _common import hz_to_midi


def midi_to_hz(m: int) -> float:
    return float(440.0 * (2.0 ** ((m - 69) / 12.0)))


def triangular_filterbank(
    n_freq: int,
    sr: int,
    n_fft: int,
    centers_hz: np.ndarray,
) -> np.ndarray:
    """Return (n_bands, n_freq) triangular weights on linear FFT bins."""
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    n_bands = len(centers_hz)
    fb = np.zeros((n_bands, n_freq), dtype=np.float64)
    # edges: geometric midpoints between centers, with ends
    edges = np.zeros(n_bands + 2, dtype=np.float64)
    edges[0] = max(1.0, centers_hz[0] / (2.0 ** (1.0 / 24.0)))
    edges[-1] = min(sr / 2.0 - 1.0, centers_hz[-1] * (2.0 ** (1.0 / 24.0)))
    for i in range(n_bands):
        if i == 0:
            edges[1] = centers_hz[0]
        else:
            edges[i + 1] = np.sqrt(centers_hz[i - 1] * centers_hz[i])
    # rebuild edges properly: left, center, right for each band
    for i, c in enumerate(centers_hz):
        left = edges[0] if i == 0 else np.sqrt(centers_hz[i - 1] * c)
        right = edges[-1] if i == n_bands - 1 else np.sqrt(c * centers_hz[i + 1])
        for fi, f in enumerate(freqs):
            if left < f <= c:
                fb[i, fi] = (f - left) / max(c - left, 1e-12)
            elif c < f < right:
                fb[i, fi] = (right - f) / max(right - c, 1e-12)
    # normalize
    s = fb.sum(axis=1, keepdims=True)
    s[s < 1e-12] = 1.0
    return fb / s


def quartertone_centers(fmin: float, fmax: float, bands_per_octave: int) -> np.ndarray:
    n = int(np.floor(bands_per_octave * np.log2(fmax / fmin))) + 1
    return fmin * (2.0 ** (np.arange(n) / float(bands_per_octave)))


def piano88_centers(midi_lo: int, midi_hi: int) -> tuple[np.ndarray, np.ndarray]:
    midis = np.arange(midi_lo, midi_hi + 1, dtype=np.int32)
    return np.array([midi_to_hz(int(m)) for m in midis], dtype=np.float64), midis


def superflux_2d(
    mono: np.ndarray,
    sr: int,
    centers_hz: np.ndarray,
    *,
    n_fft: int,
    hop_length: int,
    lag: int,
    max_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sf[T, n_bands], times[T]."""
    S = np.abs(librosa.stft(mono, n_fft=n_fft, hop_length=hop_length)) ** 2
    fb = triangular_filterbank(S.shape[0], sr, n_fft, centers_hz)
    band = fb @ S  # (n_bands, T)
    logB = librosa.power_to_db(band, ref=np.max, top_db=80.0)
    size = int(2 * max_size + 1)
    ref = maximum_filter1d(logB, size=size, axis=0, mode="nearest")
    diff = logB[:, lag:] - ref[:, :-lag]
    sf = np.maximum(0.0, diff)
    sf = np.pad(sf, ((0, 0), (lag, 0)), mode="constant")
    times = librosa.frames_to_time(np.arange(sf.shape[1]), sr=sr, hop_length=hop_length)
    return sf.T, times  # (T, n_bands), (T,)


def aggregate_to_midi(
    sf_band: np.ndarray, centers_hz: np.ndarray, midi_lo: int = 0, midi_hi: int = 127
) -> np.ndarray:
    """sf_band (T, B) -> (T, 128) by nearest MIDI."""
    midis = np.array([hz_to_midi(float(f)) for f in centers_hz], dtype=np.int32)
    out = np.zeros((sf_band.shape[0], 128), dtype=np.float64)
    for bi, m in enumerate(midis):
        if midi_lo <= m <= midi_hi:
            out[:, m] += sf_band[:, bi]
    return out


def pick_at_time(
    sf_pitch: np.ndarray,
    times: np.ndarray,
    t: float,
    half_win: int,
    midi_min: int,
    midi_max: int,
    fallback: int,
    method: str,
) -> tuple[int, dict]:
    from _common import argmax_pitch_vector

    idx = int(np.argmin(np.abs(times - t)))
    i0 = max(0, idx - half_win)
    i1 = min(sf_pitch.shape[0], idx + half_win + 1)
    vec = np.mean(sf_pitch[i0:i1], axis=0)
    pitch, meta = argmax_pitch_vector(vec, 0, midi_min, midi_max, fallback, method)
    meta["frame"] = idx
    meta["frames"] = [i0, i1]
    return pitch, meta
