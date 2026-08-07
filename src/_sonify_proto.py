"""Q1 프로토타입 rescue 결과 청각 검증 파일 생성."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import soundfile as sf
from config import audio_paths, MIN_EVENT_GAP_S, WINDOW_S, SR, OUT_DIR
from audio_io import load_mono
from onset import superflux_envelope, band_envelopes
from peak_pick import peaks, otsu


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def local_proto_rescue_q1(env_full, bands, times, dur,
                          window_s=WINDOW_S, min_gap_s=MIN_EVENT_GAP_S):
    pk_base = peaks(env_full, times, min_gap_s=min_gap_s)
    if len(pk_base) < 1:
        return pk_base, set()

    time_to_idx = {t: i for i, t in enumerate(times)}
    band_names = ["low", "mid", "high"]
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
            sim = cosine_sim(p, prototype)
            if sim >= sim_floor:
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


# GL 구간 생성
for p in audio_paths("Grievous"):
    mono = load_mono(p)
    dur = len(mono) / SR
    env, times = superflux_envelope(mono)
    bands = band_envelopes(mono)

    pk_base = peaks(env, times)
    pk_all, rescued_times = local_proto_rescue_q1(env, bands, times, dur)

    dest = OUT_DIR / "sonify"
    dest.mkdir(parents=True, exist_ok=True)

    click_base = _click(SR, 3000.0)    # 1차 탐지: 3kHz
    click_rescue = _click(SR, 5000.0, dur_ms=15.0, amp=0.8)  # 구조: 5kHz

    for seg_label, s0_s, s1_s in [("0-4s", 0, 4), ("0-12s", 0, 12),
                                   ("16-20s", 16, 20)]:
        s0 = int(s0_s * SR)
        s1 = min(int(s1_s * SR), len(mono))
        seg = mono[s0:s1].copy()

        # 1차만
        out_base = seg.copy()
        base_in_seg = pk_base[(pk_base >= s0_s) & (pk_base < s1_s)]
        for t in base_in_seg:
            idx = int((t - s0_s) * SR)
            end = min(idx + len(click_base), len(out_base))
            n = end - idx
            if n > 0:
                out_base[idx:end] += click_base[:n]

        # 1차 + Q1 구조 (구조 피크만 5kHz로 표시)
        out_q1 = seg.copy()
        all_in_seg = pk_all[(pk_all >= s0_s) & (pk_all < s1_s)]
        for t in all_in_seg:
            idx = int((t - s0_s) * SR)
            is_rescued = t in rescued_times
            c = click_rescue if is_rescued else click_base
            end = min(idx + len(c), len(out_q1))
            n = end - idx
            if n > 0:
                out_q1[idx:end] += c[:n]

        # 구조 피크만
        out_rescued_only = seg.copy()
        for t in all_in_seg:
            if t in rescued_times:
                idx = int((t - s0_s) * SR)
                end = min(idx + len(click_rescue), len(out_rescued_only))
                n = end - idx
                if n > 0:
                    out_rescued_only[idx:end] += click_rescue[:n]

        stem = p.stem
        sf.write(str(dest / f"{stem}_{seg_label}_base_click.wav"),
                 np.clip(out_base, -1, 1), SR)
        sf.write(str(dest / f"{stem}_{seg_label}_q1_click.wav"),
                 np.clip(out_q1, -1, 1), SR)
        sf.write(str(dest / f"{stem}_{seg_label}_q1_rescued_only.wav"),
                 np.clip(out_rescued_only, -1, 1), SR)

        n_base = len(base_in_seg)
        n_all = len(all_in_seg)
        n_resc = sum(1 for t in all_in_seg if t in rescued_times)
        print(f"GL {seg_label}: 1차={n_base}  Q1전체={n_all}  (구조={n_resc})")

print(f"\n출력: {dest}")
