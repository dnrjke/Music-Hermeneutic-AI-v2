"""Dir 트랙 adaptive 소니파이."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

import numpy as np
import soundfile as sf
from pathlib import Path
from audio_io import load_mono, duration_s
from onset import superflux_envelope, band_envelopes
from peak_pick import peaks_adaptive
from config import SR, OUT_DIR

p = Path(r"E:\game\Music Hermeneutic AI v2\audio\102 - Dir.wav")
print(f"▸ {p.name}")
mono = load_mono(p)
dur = duration_s(p)
print(f"  길이: {dur:.1f}s")

env, times = superflux_envelope(mono)
bands = band_envelopes(mono)
pk = peaks_adaptive(env, times, bands, dur)
print(f"  adaptive: {len(pk)}")

def _click(sr, freq, dur_ms=12.0, amp=0.7):
    n = int(sr * dur_ms / 1000)
    t = np.arange(n, dtype=np.float32) / sr
    e = np.exp(-t * 1000 / dur_ms)
    return (amp * e * np.sin(2 * np.pi * freq * t)).astype(np.float32)

click = _click(SR, 3000.0)
out = mono.copy()
for t in pk:
    idx = int(t * SR)
    end = min(idx + len(click), len(out))
    n = end - idx
    if n > 0:
        out[idx:end] += click[:n]
out = np.clip(out, -1, 1)

dest = OUT_DIR / "sonify" / "Dir"
dest.mkdir(parents=True, exist_ok=True)
sf.write(str(dest / "전체_adaptive_클릭.wav"), out, SR)
print(f"  출력: {dest / '전체_adaptive_클릭.wav'}")
