"""LPC whitening residual pass: fixed frame-wise Levinson excitation."""
from __future__ import annotations

import librosa
import numpy as np
from scipy.signal import lfilter

LPC_PARAMS = {
    "order": 24,
    "frame": 2048,
    "hop": 512,
    "pre_emphasis": 0.97,
}


def _pre_emphasize(x: np.ndarray, coef: float) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    out[1:] = x[1:] - coef * x[:-1]
    return out


def _lpc_channel(
    mono: np.ndarray,
    *,
    order: int,
    frame: int,
    hop: int,
    pre_emphasis: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Overlap-add LPC residual and all-pole synthesis for one channel."""
    x = np.asarray(mono, dtype=np.float64)
    n = len(x)
    emphasized = _pre_emphasize(x, pre_emphasis)
    window = np.hanning(frame).astype(np.float64)

    residual = np.zeros(n, dtype=np.float64)
    synthesis = np.zeros(n, dtype=np.float64)
    norm = np.zeros(n, dtype=np.float64)

    if n < order + 1:
        return x.astype(np.float32), np.zeros(n, dtype=np.float32)

    for start in range(0, max(n - order, 1), hop):
        end = min(start + frame, n)
        seg = emphasized[start:end]
        if len(seg) < order + 1:
            break
        if len(seg) < frame:
            padded = np.zeros(frame, dtype=np.float64)
            padded[: len(seg)] = seg
            seg_w = padded * window
            valid = len(seg)
        else:
            seg_w = seg * window
            valid = frame

        if float(np.max(np.abs(seg_w))) < 1e-12:
            a = np.zeros(order + 1, dtype=np.float64)
            a[0] = 1.0
        else:
            try:
                a = librosa.lpc(seg_w, order=order)
            except Exception:
                a = np.zeros(order + 1, dtype=np.float64)
                a[0] = 1.0

        a = np.asarray(a, dtype=np.float64)
        exc = lfilter(a, [1.0], seg_w)
        synth = lfilter([1.0], a, exc)

        residual[start : start + valid] += exc[:valid]
        synthesis[start : start + valid] += synth[:valid]
        norm[start : start + valid] += window[:valid]

    nonzero = norm > 1e-12
    residual[nonzero] /= norm[nonzero]
    synthesis[nonzero] /= norm[nonzero]
    return residual.astype(np.float32), synthesis.astype(np.float32)


def lpc_components(
    stereo: np.ndarray,
    *,
    order: int | None = None,
    frame: int | None = None,
    hop: int | None = None,
    pre_emphasis: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (residual, synthesis), same shape as stereo float32."""
    order_v = LPC_PARAMS["order"] if order is None else int(order)
    frame_v = LPC_PARAMS["frame"] if frame is None else int(frame)
    hop_v = LPC_PARAMS["hop"] if hop is None else int(hop)
    pre_v = (
        LPC_PARAMS["pre_emphasis"] if pre_emphasis is None else float(pre_emphasis)
    )

    y = np.asarray(stereo, dtype=np.float32)
    residuals = []
    synths = []
    for ch in range(y.shape[1]):
        res, syn = _lpc_channel(
            y[:, ch],
            order=order_v,
            frame=frame_v,
            hop=hop_v,
            pre_emphasis=pre_v,
        )
        residuals.append(res)
        synths.append(syn)
    residual = np.column_stack(residuals).astype(np.float32)
    synthesis = np.column_stack(synths).astype(np.float32)
    if residual.shape != y.shape or synthesis.shape != y.shape:
        raise RuntimeError(
            f"LPC shape mismatch: in={y.shape} R={residual.shape} S={synthesis.shape}"
        )
    return residual, synthesis
