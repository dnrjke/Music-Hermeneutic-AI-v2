"""adaptive pipeline 전체 길이 소니파이 — pipeline.py와 동일한 탐지.

생성: 전체_adaptive_클릭.wav (3kHz base)
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

import numpy as np
import soundfile as sf
from config import SR, MIN_EVENT_GAP_S, audio_paths, OUT_DIR
from audio_io import load_mono, duration_s
from onset import superflux_envelope, band_envelopes
from peak_pick import peaks_adaptive


def _click(sr, freq, dur_ms=12.0, amp=0.7):
    n = int(sr * dur_ms / 1000)
    t = np.arange(n, dtype=np.float32) / sr
    env = np.exp(-t * 1000 / dur_ms)
    return (amp * env * np.sin(2 * np.pi * freq * t)).astype(np.float32)


click = _click(SR, 3000.0)

TRACK_ALIAS = {
    "cry": "cry", "Grievous": "GL", "Viyella": "VN", "Swift": "SS",
}


def get_alias(name):
    for key, alias in TRACK_ALIAS.items():
        if key in name:
            return alias
    return name[:8]


for p in audio_paths():
    print(f"▸ {p.name}")
    mono = load_mono(p)
    dur = duration_s(p)
    env, times = superflux_envelope(mono)
    bands = band_envelopes(mono)

    pk = peaks_adaptive(env, times, bands, dur)

    out = mono.copy()
    for t in pk:
        idx = int(t * SR)
        end = min(idx + len(click), len(out))
        n = end - idx
        if n > 0:
            out[idx:end] += click[:n]
    out = np.clip(out, -1, 1)

    alias = get_alias(p.name)
    dest = OUT_DIR / "sonify" / alias
    dest.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest / "전체_adaptive_클릭.wav"), out, SR)
    print(f"  adaptive: {len(pk)}")

print(f"\n출력: {OUT_DIR / 'sonify'}")
