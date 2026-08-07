"""block-gated adaptive detection 전체 길이 소니파이.

3종 생성:
  전체_gate_클릭.wav         — gate 단독 (3kHz)
  전체_gate+Q1_클릭.wav      — gate + Q1 rescue (3kHz/5kHz)
  전체_gate+norm잔여_클릭.wav — gate + norm에만 있는 피크 보조 표시 (3kHz/5kHz)
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

import numpy as np
import soundfile as sf
from config import SR, HOP, WINDOW_S, MIN_EVENT_GAP_S, audio_paths, OUT_DIR
from audio_io import load_mono, duration_s
from onset import superflux_envelope, band_envelopes
from peak_pick import peaks, otsu

frame_dt = HOP / SR
BAND_NAMES = ["low", "mid", "high"]


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
            continue
        p = np.percentile(pos, pct)
        if p < 1e-12:
            continue
        out[mask] = np.clip(seg / p, 0, None)
    return out


def block_gate_detect(env, times, block_s=16.0, pct=99.0,
                      min_gap_s=MIN_EVENT_GAP_S):
    fin = np.isfinite(env)
    global_thr = otsu(env[fin])

    env_normed = local_norm(env, times, block_s=block_s, pct=pct)
    pk_norm = peaks(env_normed, times, min_gap_s=min_gap_s)
    pk_base = peaks(env, times, min_gap_s=min_gap_s)

    dur = times[-1] + frame_dt
    n_blocks = int(np.ceil(dur / block_s))

    selected_times = set()
    for bi in range(n_blocks):
        t0, t1 = bi * block_s, (bi + 1) * block_s
        mask = (times >= t0) & (times < t1)
        seg = env[mask]
        pos = seg[seg > 0]
        block_pct = float(np.percentile(pos, pct)) if len(pos) >= 5 else 0.0

        if block_pct < global_thr:
            pks = pk_norm[(pk_norm >= t0) & (pk_norm < t1)]
        else:
            pks = pk_base[(pk_base >= t0) & (pk_base < t1)]
        for t in pks:
            selected_times.add(t)

    time_to_idx = {t: i for i, t in enumerate(times)}
    candidates = [(t, float(env[time_to_idx[t]])) for t in selected_times
                  if t in time_to_idx]
    candidates.sort(key=lambda x: -x[1])

    final = []
    for t, v in candidates:
        if all(abs(t - s) >= min_gap_s for s in final):
            final.append(t)

    return np.array(sorted(final))


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def q1_rescue_on(base_pk, env, bands, times, dur,
                 window_s=WINDOW_S, min_gap_s=MIN_EVENT_GAP_S):
    """base 피크 위에 Q1 rescue 적용."""
    if len(base_pk) < 1:
        return base_pk, set()

    time_to_idx = {t: i for i, t in enumerate(times)}
    fin = np.isfinite(env)
    thr = otsu(env[fin])

    all_lm = []
    for i in range(1, len(env) - 1):
        if env[i] >= env[i-1] and env[i] > env[i+1]:
            all_lm.append(i)

    rescued_set = set()
    n_windows = int(np.ceil(dur / window_s))
    for wi in range(n_windows):
        t0, t1 = wi * window_s, (wi + 1) * window_s
        win_base = [t for t in base_pk if t0 <= t < t1]
        if len(win_base) < 2:
            continue
        profiles = []
        for t in win_base:
            i = time_to_idx.get(t)
            if i is None:
                continue
            profiles.append(np.array([bands[b][i] for b in BAND_NAMES]))
        if len(profiles) < 2:
            continue
        prototype = np.median(profiles, axis=0)
        sims = [cosine_sim(pf, prototype) for pf in profiles]
        sim_floor = float(np.percentile(sims, 25))
        wb = set(win_base)
        for i in all_lm:
            t = times[i]
            if not (t0 <= t < t1):
                continue
            if t in wb:
                continue
            pf = np.array([bands[b][i] for b in BAND_NAMES])
            if cosine_sim(pf, prototype) >= sim_floor:
                rescued_set.add(t)

    all_cands = [(t, env[time_to_idx[t]]) for t in base_pk if t in time_to_idx]
    for t in rescued_set:
        idx = time_to_idx.get(t)
        if idx is not None:
            all_cands.append((t, env[idx]))
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


def overlay(mono, pk_times, rescued_set=None):
    out = mono.copy()
    if rescued_set is None:
        rescued_set = set()
    for t in pk_times:
        idx = int(t * SR)
        c = click_rescue if t in rescued_set else click_base
        end = min(idx + len(c), len(out))
        n = end - idx
        if n > 0:
            out[idx:end] += c[:n]
    return np.clip(out, -1, 1)


def count_seg(pk, t0, t1):
    return int(((pk >= t0) & (pk < t1)).sum())


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

    # gate 탐지
    pk_gate = block_gate_detect(env, times)

    # gate + Q1
    pk_gq1, q1_rescued = q1_rescue_on(pk_gate, env, bands, times, dur)

    # norm 전체 (비교용 — gate에 없는 norm 피크를 rescue로 표시)
    env_norm = local_norm(env, times)
    pk_norm = peaks(env_norm, times)
    norm_only = set()
    for t in pk_norm:
        if all(abs(t - g) >= MIN_EVENT_GAP_S for g in pk_gate):
            norm_only.add(t)

    # 1: gate 단독
    sf.write(str(dest / "전체_gate_클릭.wav"),
             overlay(mono, pk_gate), SR)
    print(f"  gate: {len(pk_gate)}")

    # 2: gate + Q1
    sf.write(str(dest / "전체_gate+Q1_클릭.wav"),
             overlay(mono, pk_gq1, q1_rescued), SR)
    n_q1r = len(q1_rescued)
    print(f"  gate+Q1: {len(pk_gq1)} (구조={n_q1r})")

    # 3: gate + norm 잔여 (gate를 base, norm에만 있는 것을 5kHz)
    combined = list(pk_gate)
    for t in norm_only:
        combined.append(t)
    combined.sort()
    sf.write(str(dest / "전체_gate+norm잔여_클릭.wav"),
             overlay(mono, np.array(combined), norm_only), SR)
    print(f"  gate+norm잔여: {len(combined)} (잔여={len(norm_only)})")

    # 구간별 수치
    if "Nightmare" in p.name:
        print(f"\n  VN 밀집 구간 (176-240s) 4초 윈도우별:")
        pk_otsu = peaks(env, times)
        for name, pk in [("Otsu", pk_otsu), ("norm", pk_norm),
                         ("gate", pk_gate), ("gate+Q1", pk_gq1)]:
            detail = []
            for t0 in range(176, 240, 4):
                detail.append(f"{count_seg(pk, t0, t0+4):2d}")
            total = sum(int(x) for x in detail)
            print(f"    {name:>10s}: [{', '.join(detail)}] ={total:4d}")

print(f"\n출력: {OUT_DIR / 'sonify'}")
