"""오디오 적재 + LUFS 정규화. v1 features.py에서 이식."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from config import SR, TARGET_LUFS


def read_raw(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """정규화 이전의 스테레오."""
    import soundfile as sf

    x, sr = sf.read(str(path), always_2d=True, dtype="float32")
    if sr != SR:
        import soxr
        x = soxr.resample(x, sr, SR, quality="HQ")
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    return x[:, 0], x[:, 1]


def lufs_normalize(
    L: np.ndarray, R: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """파일 단위 LUFS 정규화. 프레임 단위는 절대 금물 ([D-17])."""
    import pyloudnorm as pyln

    x = np.stack([L, R], axis=1)
    meter = pyln.Meter(SR)
    loud = meter.integrated_loudness(x)
    if np.isfinite(loud):
        x = pyln.normalize.loudness(x, loud, TARGET_LUFS).astype(np.float32)
    return x[:, 0], x[:, 1]


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """스테레오 적재 + LUFS 정규화."""
    return lufs_normalize(*read_raw(path))


def load_mono(path: Path) -> np.ndarray:
    """모노 적재. 0.5 * (L + R)."""
    L, R = load(path)
    return 0.5 * (L + R)


def duration_s(path: Path) -> float:
    """파일 길이(초). 전체를 읽지 않는다."""
    import soundfile as sf
    info = sf.info(str(path))
    return info.duration
