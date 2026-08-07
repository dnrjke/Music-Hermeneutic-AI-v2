"""전체 길이 Q1 / SIR 소니파이 생성.

out/sonify/{트랙약칭}/
  전체_Q1_클릭.wav       — Q1 (base 3kHz + 구조 5kHz)
  전체_SIR_클릭.wav      — SIR(u3) (base 3kHz + 구조 5kHz)
  전체_1차탐지_클릭.wav   — Otsu base만 (3kHz)
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import soundfile as sf
from config import audio_paths, MIN_EVENT_GAP_S, SR, OUT_DIR, WINDOW_S
from audio_io import load_mono, duration_s
from onset import superflux_envelope, band_envelopes
from peak_pick import peaks, otsu
from _diag_spectral_identity import sir_rescue


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def q1_rescue(env_full, bands, times, dur,
              window_s=WINDOW_S, min_gap_s=MIN_EVENT_GAP_S):
    band_names = ["low", "mid", "high"]
    pk_base = peaks(env_full, times, min_gap_s=min_gap_s)
    if len(pk_base) < 1:
        return pk_base, set()

    time_to_idx = {t: i for i, t in enumerate(times)}
    thr = otsu(env_full[np.isfinite(env_full)])

    all_lm = []
    for i in range(1, len(env_full) - 1):
        if env_full[i] >= env_full[i-1] and env_full[i] > env_full[i+1]:
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
            if env_full[i] > thr or t in set(win_base):
                continue
            p = np.array([bands[b][i] for b in band_names])
            if cosine_sim(p, prototype) >= sim_floor:
                rescued_set.add(t)

    all_cands = [(t, env_full[time_to_idx[t]]) for t in pk_base]
    for t in rescued_set:
        all_cands.append((t, env_full[time_to_idx[t]]))
    all_cands.sort(key=lambda x: -x[1])
    selected = []
    for t, v in all_cands:
        if all(abs(t - s) >= min_gap_s for s in selected):
            selected.append(t)
    final_rescued = set(t for t in selected if t in rescued_set)
    return np.array(sorted(selected)), final_rescued


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


def get_alias(name):
    for key, alias in TRACK_ALIAS.items():
        if key in name:
            return alias
    return name[:8]


def overlay(mono, pk_times, rescued_set):
    out = mono.copy()
    for t in pk_times:
        idx = int(t * SR)
        c = click_rescue if t in rescued_set else click_base
        end = min(idx + len(c), len(out))
        n = end - idx
        if n > 0:
            out[idx:end] += c[:n]
    return np.clip(out, -1, 1)


for p in audio_paths():
    print(f"\n{'='*60}")
    print(f"▸ {p.name}")
    mono = load_mono(p)
    dur = duration_s(p)
    env, times = superflux_envelope(mono)
    bands = band_envelopes(mono)

    alias = get_alias(p.name)
    dest = OUT_DIR / "sonify" / alias
    dest.mkdir(parents=True, exist_ok=True)

    # base
    pk_base = peaks(env, times)
    base_set = set(pk_base.tolist())

    # Q1
    pk_q1, q1_rescued = q1_rescue(env, bands, times, dur)

    # SIR(u3)
    pk_sir, n_sir_r, _ = sir_rescue(env, bands, times, mode="u3")
    sir_set = set(pk_sir.tolist())
    sir_rescued = sir_set - base_set

    # 1차탐지
    sf.write(str(dest / "전체_1차탐지_클릭.wav"),
             overlay(mono, pk_base, set()), SR)
    print(f"  1차탐지: {len(pk_base)}")

    # Q1
    sf.write(str(dest / "전체_Q1_클릭.wav"),
             overlay(mono, pk_q1, q1_rescued), SR)
    print(f"  Q1: {len(pk_q1)}  (구조={sum(1 for t in pk_q1 if t in q1_rescued)})")

    # SIR
    sf.write(str(dest / "전체_SIR_클릭.wav"),
             overlay(mono, pk_sir, sir_rescued), SR)
    print(f"  SIR: {len(pk_sir)}  (구조={sum(1 for t in pk_sir if t in sir_rescued)})")

print(f"\n출력: {OUT_DIR / 'sonify'}")
