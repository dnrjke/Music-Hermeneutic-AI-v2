"""block-gated adaptive detection 진단.

Fable 제안: SExtractor 2-pass 전략의 1D 재현.
  - 전역 Otsu로 "밝은 소스" 탐지
  - block 99-pct < global Otsu인 블록은 "전역에 보이지 않음"
    → 그 블록만 local norm 탐지로 교체

D-21 준수: block_99pct와 global_otsu는 이미 승인된 수치.
새 파라미터 0개.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

import numpy as np
from config import SR, HOP, WINDOW_S, MIN_EVENT_GAP_S, audio_paths
from audio_io import load_mono, duration_s
from onset import superflux_envelope, band_envelopes
from peak_pick import peaks, otsu

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
            continue
        p = np.percentile(pos, pct)
        if p < 1e-12:
            continue
        out[mask] = np.clip(seg / p, 0, None)
    return out


def block_gate_detect(env, times, block_s=16.0, pct=99.0,
                      min_gap_s=MIN_EVENT_GAP_S):
    """block-gated adaptive detection.

    1. 전역 Otsu 계산
    2. 각 16s 블록의 99-pct 계산
    3. block_99pct >= global_otsu → base Otsu 탐지
       block_99pct <  global_otsu → local norm 탐지
    4. 전체 합산 후 탐욕적 최소간격 필터
    """
    fin = np.isfinite(env)
    global_thr = otsu(env[fin])

    env_normed = local_norm(env, times, block_s=block_s, pct=pct)
    pk_norm = peaks(env_normed, times, min_gap_s=min_gap_s)

    pk_base = peaks(env, times, min_gap_s=min_gap_s)

    dur = times[-1] + frame_dt
    n_blocks = int(np.ceil(dur / block_s))

    block_info = []
    selected_times = set()

    for bi in range(n_blocks):
        t0, t1 = bi * block_s, (bi + 1) * block_s
        mask = (times >= t0) & (times < t1)
        seg = env[mask]
        pos = seg[seg > 0]

        if len(pos) < 5:
            block_pct = 0.0
        else:
            block_pct = float(np.percentile(pos, pct))

        invisible = block_pct < global_thr
        if invisible:
            pks = pk_norm[(pk_norm >= t0) & (pk_norm < t1)]
            source = "norm"
        else:
            pks = pk_base[(pk_base >= t0) & (pk_base < t1)]
            source = "otsu"

        for t in pks:
            selected_times.add(t)

        block_info.append((bi, t0, t1, block_pct, invisible, source, len(pks)))

    # 탐욕적 최소간격 필터 (진폭 기준)
    time_to_idx = {t: i for i, t in enumerate(times)}
    candidates = []
    for t in selected_times:
        idx = time_to_idx.get(t)
        if idx is not None:
            candidates.append((t, float(env[idx])))

    candidates.sort(key=lambda x: -x[1])
    final = []
    for t, v in candidates:
        if all(abs(t - s) >= min_gap_s for s in final):
            final.append(t)

    return np.array(sorted(final)), block_info, global_thr


SEGMENTS = {
    "Grievous":  [(0, 4, 18), (16, 20, 14)],
    "Nightmare": [(0, 4, None), (176, 180, None), (180, 184, None),
                  (184, 188, None), (188, 192, None), (192, 196, None),
                  (196, 200, None), (200, 204, 6), (264, 268, None),
                  (268, 272, 14)],
    "cry":       [(0, 4, None)],
    "Swift":     [(0, 4, None)],
}


def track_key(name):
    for k in SEGMENTS:
        if k in name:
            return k
    return None


def count_seg(pk, t0, t1):
    return int(((pk >= t0) & (pk < t1)).sum())


for p in audio_paths():
    key = track_key(p.name)
    if key is None:
        continue

    print(f"\n{'='*72}")
    print(f"▸ {p.name}")
    mono = load_mono(p)
    dur = duration_s(p)
    env, times = superflux_envelope(mono)

    pk_otsu = peaks(env, times)
    env_norm = local_norm(env, times)
    pk_norm = peaks(env_norm, times)
    pk_gate, block_info, global_thr = block_gate_detect(env, times)

    segs = SEGMENTS[key]

    # 헤더
    print(f"\n  global Otsu = {global_thr:.4f}")
    print(f"\n  {'방법':>16s}  {'전체':>5s}", end="")
    for t0, t1, h in segs:
        label = f"{t0//60}:{t0%60:02d}"
        print(f"  {label:>6s}", end="")
    print()

    for name, pk in [("Otsu", pk_otsu), ("norm(16s)", pk_norm), ("gate", pk_gate)]:
        line = f"  {name:>16s}  {len(pk):5d}"
        for t0, t1, h in segs:
            if t1 > dur:
                line += f"  {'—':>6s}"
                continue
            c = count_seg(pk, t0, t1)
            s = f"{c}"
            if h is not None and name == "Otsu":
                s += f"({h})"
            line += f"  {s:>6s}"
        print(line)

    # 블록별 gate 결정
    n_invisible = sum(1 for _, _, _, _, inv, _, _ in block_info if inv)
    n_total = len(block_info)
    print(f"\n  블록 현황: {n_total}개 중 invisible={n_invisible}개 "
          f"({n_invisible/n_total*100:.1f}%)")

    if key == "Nightmare":
        print(f"\n  --- VN 블록별 gate 결정 (176-256s 구간) ---")
        for bi, t0, t1, bpct, inv, src, nc in block_info:
            if t0 < 160 or t0 > 260:
                continue
            label = f"{t0//60:.0f}:{t0%60:02.0f}"
            marker = "◀ INVISIBLE" if inv else ""
            print(f"    [{label}] 99pct={bpct:6.3f}  {'norm' if inv else 'otsu':4s}  "
                  f"탐지={nc:3d}  {marker}")

        print(f"\n  --- VN 밀집 구간 (176-240s) 4초 윈도우별 ---")
        for name, pk in [("Otsu", pk_otsu), ("norm(16s)", pk_norm), ("gate", pk_gate)]:
            detail = []
            for wi_t0 in range(176, 240, 4):
                c = count_seg(pk, wi_t0, wi_t0 + 4)
                detail.append(f"{c:2d}")
            total = sum(int(x) for x in detail)
            print(f"    {name:>16s}: [{', '.join(detail)}] ={total:4d}")
