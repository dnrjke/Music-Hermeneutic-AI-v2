"""SuperFlux 온셋 포락선 추출. v1 features.py에서 이식.

전대역 + 3밴드(low/mid/high) 온셋 강도 포락선을 계산한다.
에너지 계열 금지([D-07]) — SuperFlux는 스펙트럼 변화를 재므로
리미팅에도 온셋 구조가 남는다.
"""
from __future__ import annotations

import numpy as np

from config import (
    SR, N_FFT, HOP, N_MELS, FMIN,
    SUPERFLUX_LAG, SUPERFLUX_MAX, ONSET_BANDS,
)


def superflux_envelope(mono: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """전대역 SuperFlux 온셋 강도 포락선.

    Returns:
        (envelope, times) — 둘 다 (n_frames,) shape
    """
    import librosa

    S = np.abs(librosa.stft(mono, n_fft=N_FFT, hop_length=HOP))
    mel = librosa.feature.melspectrogram(
        S=S ** 2, sr=SR, n_mels=N_MELS, fmin=FMIN
    )
    logmel = librosa.power_to_db(mel, ref=np.max, top_db=80.0)

    sf_kw = dict(
        sr=SR, hop_length=HOP, lag=SUPERFLUX_LAG,
        max_size=SUPERFLUX_MAX, detrend=True,
    )
    n = S.shape[1]
    env = librosa.onset.onset_strength(S=logmel, **sf_kw)
    env = librosa.util.fix_length(env, size=n)
    times = librosa.frames_to_time(np.arange(n), sr=SR, hop_length=HOP)
    return env, times


def band_envelopes(mono: np.ndarray) -> dict[str, np.ndarray]:
    """대역별 SuperFlux 온셋 포락선.

    Returns:
        {"low": ..., "mid": ..., "high": ...} — 각 (n_frames,)
        전대역과 동일한 프레임 수.
    """
    import librosa

    S = np.abs(librosa.stft(mono, n_fft=N_FFT, hop_length=HOP))
    mel = librosa.feature.melspectrogram(
        S=S ** 2, sr=SR, n_mels=N_MELS, fmin=FMIN
    )
    logmel = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    n = S.shape[1]

    mel_f = librosa.mel_frequencies(n_mels=N_MELS, fmin=FMIN, fmax=SR / 2)
    edges: list[int] = [0]
    labels: list[str] = []
    for lab, lo, hi in ONSET_BANDS:
        idx = np.flatnonzero((mel_f >= lo) & (mel_f < hi))
        assert len(idx), f"온셋 대역 {lab}에 mel 밴드가 없다"
        edges.append(int(idx[-1]) + 1)
        labels.append(lab)

    sf_kw = dict(
        sr=SR, hop_length=HOP, lag=SUPERFLUX_LAG,
        max_size=SUPERFLUX_MAX, detrend=True,
    )
    multi = librosa.onset.onset_strength_multi(S=logmel, channels=edges, **sf_kw)
    result: dict[str, np.ndarray] = {}
    for lab, env in zip(labels, multi):
        result[lab] = librosa.util.fix_length(env, size=n)
    return result
