import sys; sys.stdout.reconfigure(encoding="utf-8")
"""SIR (Spectral Identity Rescue) — 천체 별 제거 원리의 음향 응용 진단.

── 핵심 아이디어 ─────────────────────────────────────────────
베일 성운 파이프라인에서 별을 특정한 원리:
    uniformity = ch_min / ch_max
    별   = 광대역(백색, uniformity 높음)  → 노이즈로 간주, 제거
    성운 = 협대역(유색, uniformity 낮음)  → 신호로 보존

음향(하드코어)에서는 이 관계가 **반전**된다:
    참 사건(킥·피아노 해머·신스 어택)의 어택 순간
        = 전 대역(low/mid/high) 동시 스펙트럼 변화 = "백색" 온셋
        → uniformity = band_min / band_max 높음  → **신호**
    비사건(지속음의 미세 변조, 협대역 텍스처, 포락선 리플)
        = 특정 대역에만 국한된 변화 = "유색"
        → uniformity 낮음  → **노이즈**

즉 같은 분류 축(대역 균일도)을 쓰되 라벨만 뒤집는다. 별 제거가
"백색을 버리는" 필터였다면 SIR은 "백색만 구조하는" 필터다.

── Q1 rescue의 과다탐지(cry 0-4s) 원인과 해법 ────────────────
Q1은 윈도우 내 이웃 피크들로 프로토타입(대역 프로파일 중앙값)을
만든다 — **외재적** 기준. 조용한 윈도우에서는 이웃이 적거나 노이즈라
프로토타입 자체가 불안정 → 노이즈 리플까지 유사 판정.
SIR은 후보 **자신의 고유(intrinsic)** 대역 균일도만 본다. 이웃이
필요 없으므로 조용한 구간에서도 기준이 무너지지 않는다.

── 자유 파라미터 0 (D-21) ───────────────────────────────────
uniformity 임계는 고르지 않는다. 트랙 전체 극대점 모집단의
uniformity 분포에 Otsu를 적용 — "광대역 무리 / 협대역 무리"의
자연 분할점. 검출 결과(극대점 모집단)에서 규칙(임계)을 생성한다
(사용자 원리: "검출은 규칙으로 하지 않고, 규칙은 검출 결과에서
생성한다"). 에너지가 아니라 SuperFlux 대역 플럭스의 **비율**만
사용 (D-07).

── 다각도 라인 커널의 유추: 대역 동시 극대(coincidence) ───────
별은 R/G/B 어느 채널에서 봐도 점광원이다. 참 사건도 여러 대역
포락선에서 **동시에**(±1프레임 허용, 1프레임=5.8ms는 시간 해상도이지
자유 파라미터가 아님) 극대점이어야 한다. 2-of-3 다수결 게이트.

── 균일도 척도 2종 ──────────────────────────────────────────
  u3 = band_min / band_max          (성운 파이프라인 원형 그대로)
  u2 = band_2nd_max / band_max      (2채널 백색도)
u3는 3대역 전부의 동시 활동을 요구한다. 그러나 하이햇(저역 0),
피아노(저역 미미) 등 참 사건 다수가 저역 플럭스 0이라 u3=0이 된다
— 저역은 "감도가 없는 채널"인 셈. 별이 감광 채널에서만 보여도
점광원이듯, u2는 "최소 2개 대역이 함께 상승"을 백색으로 본다.
두 척도 모두 진단 출력에 분리도(base/sub 중앙값 차)를 병기한다.

── 비교 방법 ────────────────────────────────────────────────
  A  기존 Otsu (base)
  B  Q1 prototype rescue (현행 최선)
  C  SIR(u3)    : base + [서브임계 극대 ∧ u3 ≥ Otsu(u3) ∧ 2-of-3 동시극대]
  D  SIR(u2)    : base + [서브임계 극대 ∧ u2 ≥ Otsu(u2) ∧ 2-of-3 동시극대]
  E  Q1 ∩ SIR   : Q1이 구조한 것 중 SIR(u2) 게이트 통과분만 (하이브리드)

비교 구간: GL(0-4, 16-20), VN(200-204, 268-272), cry(0-4), SS(0-4)
인간 계수: GL 0-4=18, GL 16-20=14, VN 200-204=6, VN 268-272=14
"""
import numpy as np

from config import audio_paths, MIN_EVENT_GAP_S, WINDOW_S
from audio_io import load_mono, duration_s
from onset import superflux_envelope, band_envelopes
from peak_pick import peaks, otsu

BAND_NAMES = ["low", "mid", "high"]

# 트랙별 비교 구간과 인간 계수 참조값
SEGMENTS = {
    "cry":       [(0, 4)],
    "Grievous":  [(0, 4), (16, 20)],
    "Nightmare": [(200, 204), (268, 272)],
    "Swift":     [(0, 4)],
}
HUMAN = {
    ("Grievous", 0): 18, ("Grievous", 16): 14,
    ("Nightmare", 200): 6, ("Nightmare", 268): 14,
}


def local_maxima_mask(v: np.ndarray) -> np.ndarray:
    """극대점 불리언 마스크 (peaks()와 동일 규칙: 좌 >=, 우 >)."""
    m = np.zeros(len(v), dtype=bool)
    if len(v) >= 3:
        m[1:-1] = (v[1:-1] >= v[:-2]) & (v[1:-1] > v[2:])
    return m


def band_uniformity(bands: dict[str, np.ndarray],
                    mode: str = "u3") -> np.ndarray:
    """프레임별 대역 균일도.

    u3: band_min / band_max — 성운 파이프라인의 uniformity = ch_min/ch_max
        그대로. 3대역 전부 활동해야 백색.
    u2: band_2nd_max / band_max — 2채널 백색도. 최소 2개 대역이 함께
        상승하면 백색 (저역 무감 채널 허용).

    음의 플럭스(해당 대역 온셋 없음)는 0으로 클립.
    전 대역 0이면 u = 0 (순수 무음/노이즈 → 비사건).
    """
    B = np.stack([np.clip(bands[b], 0.0, None) for b in BAND_NAMES])
    Bs = np.sort(B, axis=0)          # 오름차순: [min, 2nd, max]
    bmax = Bs[-1]
    num = Bs[0] if mode == "u3" else Bs[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        u = np.where(bmax > 1e-12, num / bmax, 0.0)
    return u


def band_coincidence(bands: dict[str, np.ndarray]) -> np.ndarray:
    """프레임별: 몇 개 대역에서 ±1프레임 내 극대점인가 (0..3)."""
    n = len(bands[BAND_NAMES[0]])
    count = np.zeros(n, dtype=int)
    for b in BAND_NAMES:
        m = local_maxima_mask(bands[b])
        # ±1프레임 허용 (dilation)
        m_dil = m.copy()
        m_dil[1:] |= m[:-1]
        m_dil[:-1] |= m[1:]
        count += m_dil.astype(int)
    return count


def greedy_merge(cand_idx: list[int], env: np.ndarray, times: np.ndarray,
                 min_gap_s: float) -> np.ndarray:
    """진폭 내림차순 탐욕 선택 + 최소간격 (기존 peaks()와 동일 절차)."""
    selected: list[int] = []
    for i in sorted(set(cand_idx), key=lambda i: -env[i]):
        if all(abs(times[i] - times[j]) >= min_gap_s for j in selected):
            selected.append(i)
    if not selected:
        return np.array([], dtype=np.float64)
    return times[np.sort(np.asarray(selected, dtype=int))]


def sir_rescue(env, bands, times, mode: str = "u2",
               min_gap_s=MIN_EVENT_GAP_S):
    """SIR: base Otsu 피크 + [서브임계 극대 ∧ 백색 ∧ 2-of-3 동시극대].

    백색 판정: uniformity(mode) ≥ Otsu(극대점 모집단의 uniformity).
    Returns (peak_times, n_rescued, u_thr)
    """
    fin = np.isfinite(env)
    v = np.where(fin, env, -np.inf)
    thr = otsu(env[fin])

    lm = np.flatnonzero(local_maxima_mask(v))
    if len(lm) == 0:
        return np.array([], dtype=np.float64), 0, 0.0

    u = band_uniformity(bands, mode=mode)
    # 규칙 생성: 극대점 모집단의 uniformity 분포를 Otsu로 이분
    u_thr = otsu(u[lm])

    base_idx = lm[v[lm] > thr].tolist()
    sub = lm[v[lm] <= thr]

    coin = band_coincidence(bands)
    gate = (u[sub] >= u_thr) & (coin[sub] >= 2)
    rescued_idx = sub[gate].tolist()

    pk = greedy_merge(base_idx + rescued_idx, v, times, min_gap_s)
    rescued_times = set(times[i] for i in rescued_idx)
    n_rescued = sum(1 for t in pk if t in rescued_times)
    return pk, n_rescued, u_thr


def q1_rescue(env, bands, times, dur, window_s=WINDOW_S,
              min_gap_s=MIN_EVENT_GAP_S):
    """현행 Q1 prototype rescue (_diag_proto_q1 재현). Returns (pk, rescued_set)."""
    def cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    pk_base = peaks(env, times, min_gap_s=min_gap_s)
    if len(pk_base) < 1:
        return pk_base, set()

    time_to_idx = {t: i for i, t in enumerate(times)}
    thr = otsu(env[np.isfinite(env)])
    lm = np.flatnonzero(local_maxima_mask(env))

    rescued_set: set[float] = set()
    n_windows = int(np.ceil(dur / window_s))
    for wi in range(n_windows):
        t0, t1 = wi * window_s, (wi + 1) * window_s
        win_base = [t for t in pk_base if t0 <= t < t1]
        if len(win_base) < 2:
            continue
        profiles = [np.array([bands[b][time_to_idx[t]] for b in BAND_NAMES])
                    for t in win_base]
        prototype = np.median(profiles, axis=0)
        sims = [cos(pf, prototype) for pf in profiles]
        sim_floor = float(np.percentile(sims, 25))
        wb = set(win_base)
        for i in lm:
            t = times[i]
            if not (t0 <= t < t1) or env[i] > thr or t in wb:
                continue
            pf = np.array([bands[b][i] for b in BAND_NAMES])
            if cos(pf, prototype) >= sim_floor:
                rescued_set.add(t)

    cand = [time_to_idx[t] for t in pk_base] + \
           [time_to_idx[t] for t in rescued_set]
    pk = greedy_merge(cand, env, times, min_gap_s)
    return pk, rescued_set


def q1_intersect_sir(env, bands, times, dur, mode: str = "u2",
                     min_gap_s=MIN_EVENT_GAP_S):
    """E: Q1이 구조한 후보 중 SIR 게이트(uniformity+coincidence) 통과분만."""
    pk_base = peaks(env, times, min_gap_s=min_gap_s)
    _, rescued_set = q1_rescue(env, bands, times, dur)

    fin = np.isfinite(env)
    lm = np.flatnonzero(local_maxima_mask(np.where(fin, env, -np.inf)))
    u = band_uniformity(bands, mode=mode)
    u_thr = otsu(u[lm])
    coin = band_coincidence(bands)

    time_to_idx = {t: i for i, t in enumerate(times)}
    kept = []
    for t in rescued_set:
        i = time_to_idx[t]
        if u[i] >= u_thr and coin[i] >= 2:
            kept.append(i)

    cand = [time_to_idx[t] for t in pk_base] + kept
    pk = greedy_merge(cand, env, times, min_gap_s)
    return pk


def seg_count(pk: np.ndarray, t0: float, t1: float) -> int:
    return int(((pk >= t0) & (pk < t1)).sum())


def track_key(name: str) -> str | None:
    for k in SEGMENTS:
        if k in name:
            return k
    return None


def main() -> None:
    for p in audio_paths():
        key = track_key(p.name)
        if key is None:
            continue
        segs = SEGMENTS[key]

        print(f"\n{'=' * 64}")
        print(f"▸ {p.name}")
        dur = duration_s(p)
        mono = load_mono(p)
        env, times = superflux_envelope(mono)
        bands = band_envelopes(mono)

        pk_a = peaks(env, times)
        pk_b, _ = q1_rescue(env, bands, times, dur)
        pk_c, nr_c, u3_thr = sir_rescue(env, bands, times, mode="u3")
        pk_d, nr_d, u2_thr = sir_rescue(env, bands, times, mode="u2")
        pk_e = q1_intersect_sir(env, bands, times, dur, mode="u2")

        # 진단: 모집단 uniformity 분포 (u3, u2)
        fin = np.isfinite(env)
        v = np.where(fin, env, -np.inf)
        thr = otsu(env[fin])
        lm = np.flatnonzero(local_maxima_mask(v))
        base_lm = lm[v[lm] > thr]
        sub_lm = lm[v[lm] <= thr]
        print(f"  극대점 {len(lm)}개 (base {len(base_lm)}, sub {len(sub_lm)})")
        for mode, u_thr in [("u3", u3_thr), ("u2", u2_thr)]:
            u = band_uniformity(bands, mode=mode)
            mb, ms = np.median(u[base_lm]), np.median(u[sub_lm])
            print(f"  {mode}: Otsu 임계={u_thr:.3f}  중앙값 base={mb:.3f} "
                  f"sub={ms:.3f}  → 백색/유색 분리도 {mb - ms:+.3f}")

        header = f"  {'방법':24s}: {'전체':>5s}"
        for t0, t1 in segs:
            header += f"  {f'{t0}-{t1}':>9s}"
        print(header)

        rows = [
            ("A 기존(Otsu)", pk_a, ""),
            ("B Q1 rescue", pk_b, ""),
            ("C SIR(u3)", pk_c, f"(rescued={nr_c})"),
            ("D SIR(u2)", pk_d, f"(rescued={nr_d})"),
            ("E Q1∩SIR(u2)", pk_e, ""),
        ]
        for label, pk, note in rows:
            line = f"  {label:24s}: {len(pk):5d}"
            for t0, t1 in segs:
                line += f"  {seg_count(pk, t0, t1):9d}"
            print(f"{line}  {note}")

        # 인간 계수 참조
        ref_cells = []
        has_ref = False
        for t0, t1 in segs:
            h = HUMAN.get((key, t0))
            if h is None:
                ref_cells.append(f"  {'—':>9s}")
            else:
                ref_cells.append(f"  {h:9d}")
                has_ref = True
        if has_ref:
            print(f"  {'(인간 계수)':24s}: {'':5s}" + "".join(ref_cells))


if __name__ == "__main__":
    main()
