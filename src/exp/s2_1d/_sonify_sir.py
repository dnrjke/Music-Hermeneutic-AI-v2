"""SIR(u3) 청각 검증 파일 생성.

각 구간 폴더에 추가:
  SIR_클릭.wav           — SIR base(3kHz) + rescued(5kHz)
  SIR_구조만_클릭.wav     — SIR rescued만 (5kHz)
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import soundfile as sf
from config import audio_paths, MIN_EVENT_GAP_S, SR, OUT_DIR
from audio_io import load_mono, duration_s
from onset import superflux_envelope, band_envelopes
from peak_pick import peaks, otsu
from _diag_spectral_identity import (
    sir_rescue, local_maxima_mask, band_uniformity, band_coincidence
)


def _click(sr, freq, dur_ms=12.0, amp=0.7):
    n = int(sr * dur_ms / 1000)
    t = np.arange(n, dtype=np.float32) / sr
    env = np.exp(-t * 1000 / dur_ms)
    return (amp * env * np.sin(2 * np.pi * freq * t)).astype(np.float32)


click_base = _click(SR, 3000.0)
click_rescue = _click(SR, 5000.0, dur_ms=15.0, amp=0.8)

TRACK_ALIAS = {
    "cry": "cry", "Grievous": "GL", "Viyella": "VN", "Swift": "SS",
}
SEGMENTS = {
    "cry":  [(0, 4), (4, 8), (16, 20), (56, 60)],
    "GL":   [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20)],
    "VN":   [(0, 4), (16, 20), (56, 60), (200, 204), (268, 272)],
    "SS":   [(0, 4), (4, 8), (16, 20), (56, 60)],
}


def get_alias(name):
    for key, alias in TRACK_ALIAS.items():
        if key in name:
            return alias
    return name[:8]


for p in audio_paths():
    print(f"\n{'='*60}")
    print(f"▸ {p.name}")
    mono = load_mono(p)
    dur = duration_s(p)
    env, times = superflux_envelope(mono)
    bands = band_envelopes(mono)

    pk_base = peaks(env, times)
    base_set = set(pk_base.tolist())

    pk_sir, n_rescued, u_thr = sir_rescue(env, bands, times, mode="u3")
    sir_set = set(pk_sir.tolist())
    rescued_set = sir_set - base_set

    alias = get_alias(p.name)
    segs = SEGMENTS.get(alias, [(0, 4), (16, 20)])

    for s0_s, s1_s in segs:
        if s1_s > dur:
            continue
        s0 = int(s0_s * SR)
        s1 = min(int(s1_s * SR), len(mono))
        seg = mono[s0:s1].copy()
        seg_label = f"{s0_s:.0f}-{s1_s:.0f}s"

        seg_dir = OUT_DIR / "sonify" / alias / seg_label
        seg_dir.mkdir(parents=True, exist_ok=True)

        sir_in = pk_sir[(pk_sir >= s0_s) & (pk_sir < s1_s)]

        # SIR 클릭 (base=3kHz, rescued=5kHz)
        out = seg.copy()
        for t in sir_in:
            idx = int((t - s0_s) * SR)
            c = click_rescue if t in rescued_set else click_base
            end = min(idx + len(c), len(out))
            n = end - idx
            if n > 0:
                out[idx:end] += c[:n]
        sf.write(str(seg_dir / "SIR_클릭.wav"),
                 np.clip(out, -1, 1), SR)

        # SIR 구조만
        out = seg.copy()
        n_resc = 0
        for t in sir_in:
            if t in rescued_set:
                idx = int((t - s0_s) * SR)
                end = min(idx + len(click_rescue), len(out))
                n = end - idx
                if n > 0:
                    out[idx:end] += click_rescue[:n]
                n_resc += 1
        sf.write(str(seg_dir / "SIR_구조만_클릭.wav"),
                 np.clip(out, -1, 1), SR)

        n_base = int(sum(1 for t in sir_in if t in base_set))
        print(f"  {alias}/{seg_label}/  SIR={len(sir_in)}  (base={n_base}, 구조={n_resc})")

print(f"\n출력: {OUT_DIR / 'sonify'}")
