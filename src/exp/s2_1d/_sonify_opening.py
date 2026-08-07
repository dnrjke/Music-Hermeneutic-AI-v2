"""opening 잔차 탐지 청각 검증 파일 생성.

각 구간 폴더에 추가:
  opening_클릭.wav          — 잔차 Otsu 피크 (3kHz)
  opening+Q1_클릭.wav       — 잔차 1차(3kHz) + 잔차 Q1 구조(5kHz)
  opening+Q1_구조만_클릭.wav — 잔차 Q1 구조 피크만 (5kHz)
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import soundfile as sf
from scipy.ndimage import grey_opening
from config import audio_paths, MIN_EVENT_GAP_S, WINDOW_S, SR, OUT_DIR
from audio_io import load_mono, duration_s
from onset import superflux_envelope, band_envelopes
from peak_pick import peaks, otsu


KERNEL = 5


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def opening_q1(env_full, bands, times, dur, kernel=KERNEL,
               window_s=WINDOW_S, min_gap_s=MIN_EVENT_GAP_S):
    opened = grey_opening(env_full, size=kernel)
    residual = np.maximum(env_full - opened, 0)

    pk_base = peaks(residual, times, min_gap_s=min_gap_s)
    if len(pk_base) < 1:
        return pk_base, pk_base, set()

    time_to_idx = {t: i for i, t in enumerate(times)}
    band_names = ["low", "mid", "high"]
    thr = otsu(residual[np.isfinite(residual)])

    all_lm = []
    for i in range(1, len(residual) - 1):
        if residual[i] >= residual[i-1] and residual[i] > residual[i+1]:
            all_lm.append(i)

    rescued_set = set()
    n_windows = int(np.ceil(dur / window_s))
    for wi in range(n_windows):
        t0, t1 = wi * window_s, (wi + 1) * window_s
        win_base = [t for t in pk_base if t0 <= t < t1]
        if len(win_base) < 2:
            continue

        profiles = []
        for t in win_base:
            i = time_to_idx[t]
            profiles.append(np.array([bands[b][i] for b in band_names]))
        prototype = np.median(profiles, axis=0)
        sims = [cosine_sim(p, prototype) for p in profiles]
        sim_floor = float(np.percentile(sims, 25))

        for i in all_lm:
            t = times[i]
            if not (t0 <= t < t1):
                continue
            if residual[i] > thr or t in set(win_base):
                continue
            p = np.array([bands[b][i] for b in band_names])
            sim = cosine_sim(p, prototype)
            if sim >= sim_floor:
                rescued_set.add(t)

    all_cands = [(t, residual[time_to_idx[t]]) for t in pk_base]
    for t in rescued_set:
        all_cands.append((t, residual[time_to_idx[t]]))
    all_cands.sort(key=lambda x: -x[1])

    selected = []
    for t, v in all_cands:
        if all(abs(t - s) >= min_gap_s for s in selected):
            selected.append(t)

    final_rescued = set(t for t in selected if t in rescued_set)
    return pk_base, np.array(sorted(selected)), final_rescued


def _click(sr, freq, dur_ms=12.0, amp=0.7):
    n = int(sr * dur_ms / 1000)
    t = np.arange(n, dtype=np.float32) / sr
    env = np.exp(-t * 1000 / dur_ms)
    return (amp * env * np.sin(2 * np.pi * freq * t)).astype(np.float32)


click_base = _click(SR, 3000.0)
click_rescue = _click(SR, 5000.0, dur_ms=15.0, amp=0.8)

TRACK_ALIAS = {
    "cry": "cry",
    "Grievous": "GL",
    "Viyella": "VN",
    "Swift": "SS",
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

    pk_op_base, pk_op_q1, op_rescued = opening_q1(env, bands, times, dur)

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

        base_in = pk_op_base[(pk_op_base >= s0_s) & (pk_op_base < s1_s)]
        all_in = pk_op_q1[(pk_op_q1 >= s0_s) & (pk_op_q1 < s1_s)]

        # opening 단독
        out = seg.copy()
        for t in base_in:
            idx = int((t - s0_s) * SR)
            end = min(idx + len(click_base), len(out))
            n = end - idx
            if n > 0:
                out[idx:end] += click_base[:n]
        sf.write(str(seg_dir / "opening_클릭.wav"),
                 np.clip(out, -1, 1), SR)

        # opening + Q1
        out = seg.copy()
        for t in all_in:
            idx = int((t - s0_s) * SR)
            c = click_rescue if t in op_rescued else click_base
            end = min(idx + len(c), len(out))
            n = end - idx
            if n > 0:
                out[idx:end] += c[:n]
        sf.write(str(seg_dir / "opening+Q1_클릭.wav"),
                 np.clip(out, -1, 1), SR)

        # opening+Q1 구조만
        out = seg.copy()
        n_resc = 0
        for t in all_in:
            if t in op_rescued:
                idx = int((t - s0_s) * SR)
                end = min(idx + len(click_rescue), len(out))
                n = end - idx
                if n > 0:
                    out[idx:end] += click_rescue[:n]
                n_resc += 1
        sf.write(str(seg_dir / "opening+Q1_구조만_클릭.wav"),
                 np.clip(out, -1, 1), SR)

        print(f"  {alias}/{seg_label}/  opening={len(base_in)}  op+Q1={len(all_in)}  구조={n_resc}")

print(f"\n출력: {OUT_DIR / 'sonify'}")
