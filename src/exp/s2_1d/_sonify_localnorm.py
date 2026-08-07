"""local norm(16s) 기반 전체 길이 소니파이.

3종 생성:
  전체_norm_클릭.wav       — norm 단독 (3kHz)
  전체_norm+Q1_클릭.wav    — norm base + Q1 rescue (3kHz/5kHz)
  전체_norm+SIR_클릭.wav   — norm base + SIR(u3) rescue (3kHz/5kHz)
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import soundfile as sf
from config import audio_paths, MIN_EVENT_GAP_S, HOP, SR, OUT_DIR, WINDOW_S
from audio_io import load_mono, duration_s
from onset import superflux_envelope, band_envelopes
from peak_pick import peaks, otsu
from _diag_spectral_identity import (
    sir_rescue, local_maxima_mask, band_uniformity, band_coincidence,
    greedy_merge, BAND_NAMES
)

frame_dt = HOP / SR


def local_norm(env, times, block_s=16.0, pct=99.0):
    out = np.zeros_like(env)
    dur = times[-1] + frame_dt
    n_blocks = int(np.ceil(dur / block_s))
    for bi in range(n_blocks):
        t0, t1 = bi * block_s, (bi + 1) * block_s
        mask = (times >= t0) & (times < t1)
        seg = env[mask]
        pos = seg[seg > 0]
        if len(pos) < 5:
            out[mask] = 0
            continue
        p = np.percentile(pos, pct)
        if p < 1e-12:
            out[mask] = 0
            continue
        out[mask] = np.clip(seg / p, 0, None)
    return out


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def q1_on_normed(env_normed, bands, times, dur,
                 window_s=WINDOW_S, min_gap_s=MIN_EVENT_GAP_S):
    """정규화 포락선에서 Q1 rescue."""
    pk_base = peaks(env_normed, times, min_gap_s=min_gap_s)
    if len(pk_base) < 1:
        return pk_base, set()

    time_to_idx = {t: i for i, t in enumerate(times)}
    thr = otsu(env_normed[env_normed > 0])

    all_lm = []
    for i in range(1, len(env_normed) - 1):
        if env_normed[i] >= env_normed[i-1] and env_normed[i] > env_normed[i+1]:
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
            profiles.append(np.array([bands[b][i] for b in BAND_NAMES]))
        prototype = np.median(profiles, axis=0)
        sims = [cosine_sim(pf, prototype) for pf in profiles]
        sim_floor = float(np.percentile(sims, 25))
        wb = set(win_base)
        for i in all_lm:
            t = times[i]
            if not (t0 <= t < t1):
                continue
            if env_normed[i] > thr or t in wb:
                continue
            pf = np.array([bands[b][i] for b in BAND_NAMES])
            if cosine_sim(pf, prototype) >= sim_floor:
                rescued_set.add(t)

    all_cands = [(t, env_normed[time_to_idx[t]]) for t in pk_base]
    for t in rescued_set:
        all_cands.append((t, env_normed[time_to_idx[t]]))
    all_cands.sort(key=lambda x: -x[1])
    selected = []
    for t, v in all_cands:
        if all(abs(t - s) >= min_gap_s for s in selected):
            selected.append(t)
    final_rescued = set(t for t in selected if t in rescued_set)
    return np.array(sorted(selected)), final_rescued


def sir_on_normed(env_normed, bands, times, min_gap_s=MIN_EVENT_GAP_S):
    """정규화 포락선에서 SIR(u3) rescue."""
    fin = env_normed > 0
    v = np.where(fin, env_normed, 0)
    pos = v[v > 0]
    if len(pos) < 10:
        return np.array([]), set()
    thr = otsu(pos)

    lm = np.flatnonzero(local_maxima_mask(v))
    if len(lm) == 0:
        return np.array([]), set()

    u = band_uniformity(bands, mode="u3")
    u_thr = otsu(u[lm])
    coin = band_coincidence(bands)

    base_idx = lm[v[lm] > thr].tolist()
    sub = lm[v[lm] <= thr]
    gate = (u[sub] >= u_thr) & (coin[sub] >= 2)
    rescued_idx = sub[gate].tolist()

    pk = greedy_merge(base_idx + rescued_idx, v, times, min_gap_s)
    rescued_times = set(times[i] for i in rescued_idx)
    n_rescued = set(t for t in pk if t in rescued_times)
    return pk, n_rescued


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

    env_n = local_norm(env, times, block_s=16.0)
    pk_norm = peaks(env_n, times)
    norm_set = set(pk_norm.tolist())

    pk_nq1, q1_rescued = q1_on_normed(env_n, bands, times, dur)
    pk_nsir, sir_rescued = sir_on_normed(env_n, bands, times)

    # norm 단독
    sf.write(str(dest / "전체_norm_클릭.wav"),
             overlay(mono, pk_norm, set()), SR)
    print(f"  norm: {len(pk_norm)}")

    # norm + Q1
    sf.write(str(dest / "전체_norm+Q1_클릭.wav"),
             overlay(mono, pk_nq1, q1_rescued), SR)
    n_q1r = sum(1 for t in pk_nq1 if t in q1_rescued)
    print(f"  norm+Q1: {len(pk_nq1)}  (구조={n_q1r})")

    # norm + SIR
    sf.write(str(dest / "전체_norm+SIR_클릭.wav"),
             overlay(mono, pk_nsir, sir_rescued), SR)
    n_sirr = sum(1 for t in pk_nsir if t in sir_rescued)
    print(f"  norm+SIR: {len(pk_nsir)}  (구조={n_sirr})")

print(f"\n출력: {OUT_DIR / 'sonify'}")
