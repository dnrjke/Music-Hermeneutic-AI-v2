"""Generate tilt/perc locked-material peak timestamp comparison markdown.

Compares already-locked peak_times (±30ms match). No detector retune.
Series (when present in manifests):
  perc_raw, perc_tilt_high, perc_tilt_k_env, perc_tilt_k_env_adaptive
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TILT_DIR = ROOT / "out" / "stems" / "Dir" / "event_sculpt" / "tilt"
OUT_MD = TILT_DIR / "tilt_material_peak_diff_timestamps.md"
OUT_JSON = TILT_DIR / "tilt_material_peak_diff_timestamps.json"
TOL = 0.03

# Preferred inclusion order; skip key if peak_times missing.
CANDIDATE_KEYS = (
    "perc_raw",
    "perc_tilt_high",
    "perc_tilt_k_env",
    "perc_tilt_k_env_adaptive",
)
# Short labels for tables
SHORT = {
    "perc_raw": "raw",
    "perc_tilt_high": "tilt_high",
    "perc_tilt_k_env": "k_env",
    "perc_tilt_k_env_adaptive": "k_env_ad",
}
# Pairwise baseline among locked materials (previous tilt best / RMS k_env).
BASELINE_KEY = "perc_tilt_k_env"

SOURCES = {
    "perc_raw": "tilt/tilt_manifest.json → peak_times_s.perc_raw",
    "perc_tilt_high": "tilt/tilt_manifest.json → peak_times_s.perc_tilt_high",
    "perc_tilt_k_env": (
        "tilt/tilt_percept_manifest.json → peak_times_s.perc_tilt_k_env"
    ),
    "perc_tilt_k_env_adaptive": (
        "tilt/tilt_k_env_adaptive_manifest.json "
        "→ peak_times_s.perc_tilt_k_env_adaptive"
    ),
}


def _fmt(t: float) -> str:
    if t < 0:
        t = 0.0
    m = int(t // 60)
    s = t - 60 * m
    return f"{m}:{s:06.3f}"


def _fmt_list(times: list[float], *, every: int | None = None) -> str:
    if not times:
        return "_(없음)_"
    lines: list[str] = []
    for i, t in enumerate(times):
        cell = f"`{_fmt(t)}` ({t:.3f}s)"
        if every and i > 0 and i % every == 0:
            lines.append("")
        lines.append(f"- {cell}")
    return "\n".join(lines)


def match(
    a: list[float], b: list[float], *, tol: float = TOL
) -> tuple[list[float], list[float], list[float]]:
    used_b: set[int] = set()
    common_a: list[float] = []
    only_a: list[float] = []
    for ta in a:
        best: int | None = None
        best_d = tol + 1.0
        for jb, tb in enumerate(b):
            if jb in used_b:
                continue
            d = abs(ta - tb)
            if d <= tol and d < best_d:
                best_d = d
                best = jb
        if best is None:
            only_a.append(ta)
        else:
            used_b.add(best)
            common_a.append(ta)
    only_b = [tb for jb, tb in enumerate(b) if jb not in used_b]
    return common_a, only_a, only_b


def load_series() -> tuple[dict[str, list[float]], list[str], list[str]]:
    """Load locked peak times. Returns (series, included_keys, skipped_notes)."""
    tilt = json.loads((TILT_DIR / "tilt_manifest.json").read_text(encoding="utf-8"))
    percept = json.loads(
        (TILT_DIR / "tilt_percept_manifest.json").read_text(encoding="utf-8")
    )
    adaptive = json.loads(
        (TILT_DIR / "tilt_k_env_adaptive_manifest.json").read_text(encoding="utf-8")
    )

    pool: dict[str, list[float]] = {}
    for key, times in tilt.get("peak_times_s", {}).items():
        pool[key] = [float(t) for t in times]
    for key, times in percept.get("peak_times_s", {}).items():
        if key not in pool:
            pool[key] = [float(t) for t in times]
    for key, times in adaptive.get("peak_times_s", {}).items():
        if key not in pool:
            pool[key] = [float(t) for t in times]

    included: list[str] = []
    skipped: list[str] = []
    series: dict[str, list[float]] = {}
    for key in CANDIDATE_KEYS:
        if key in pool and pool[key]:
            series[key] = pool[key]
            included.append(key)
        else:
            skipped.append(f"{key}: peak_times missing or empty")

    # Minimum contract from advisory option 1
    required = ("perc_tilt_k_env", "perc_tilt_k_env_adaptive")
    missing_req = [k for k in required if k not in series]
    if missing_req:
        raise RuntimeError(
            "required locked series missing peak_times: " + ", ".join(missing_req)
        )
    return series, included, skipped


def cluster_presence(
    series: dict[str, list[float]],
    keys: tuple[str, ...] | list[str],
) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    for k in keys:
        for t in series[k]:
            best_i: int | None = None
            best_d = TOL + 1.0
            for i, cl in enumerate(clusters):
                if k in cl["orders"]:
                    continue
                d = abs(t - cl["rep"])
                if d <= TOL and d < best_d:
                    best_d = d
                    best_i = i
            if best_i is None:
                clusters.append({"rep": t, "orders": {k}, "times": {k: t}})
            else:
                clusters[best_i]["orders"].add(k)
                clusters[best_i]["times"][k] = t
                vals = list(clusters[best_i]["times"].values())
                clusters[best_i]["rep"] = sum(vals) / len(vals)
    clusters.sort(key=lambda c: c["rep"])
    return clusters


def main() -> None:
    series, keys, skipped = load_series()
    key_tuple = tuple(keys)
    counts = {k: len(series[k]) for k in key_tuple}
    baseline = BASELINE_KEY if BASELINE_KEY in series else key_tuple[0]
    base_short = SHORT.get(baseline, baseline)

    pairwise: dict[str, Any] = {}
    for k in key_tuple:
        if k == baseline:
            continue
        _c, only_other, only_base = match(series[k], series[baseline])
        pairwise[k] = {
            "common": len(_c),
            "only_other": only_other,
            "only_base": only_base,
        }

    neighbors: list[dict[str, Any]] = []
    for a, b in zip(key_tuple, key_tuple[1:]):
        _c, only_a, only_b = match(series[a], series[b])
        neighbors.append(
            {
                "a": a,
                "b": b,
                "common": len(_c),
                "only_a": only_a,
                "only_b": only_b,
            }
        )

    clusters = cluster_presence(series, key_tuple)
    by_mask: dict[tuple[int, ...], list[float]] = defaultdict(list)
    for cl in clusters:
        mask = tuple(1 if o in cl["orders"] else 0 for o in key_tuple)
        by_mask[mask].append(float(cl["rep"]))

    only_one = {
        k: by_mask[tuple(1 if o == k else 0 for o in key_tuple)] for k in key_tuple
    }
    n_series = len(key_tuple)
    all_present = by_mask[tuple([1] * n_series)]
    n_disagreement = len(clusters) - len(all_present)

    payload = {
        "tol_s": TOL,
        "note": (
            "Locked perc/tilt materials only; peak_times from existing manifests. "
            "No detector retune."
        ),
        "baseline": baseline,
        "series_keys": list(key_tuple),
        "skipped": skipped,
        "sources": {k: SOURCES[k] for k in key_tuple if k in SOURCES},
        "counts": counts,
        "all_present_n": len(all_present),
        "disagreement_n": n_disagreement,
        "n_clusters_total": len(clusters),
        "pairwise_vs_baseline": {
            k: {
                "common": v["common"],
                "only_other_n": len(v["only_other"]),
                "only_baseline_n": len(v["only_base"]),
                "only_other_s": v["only_other"],
                "only_baseline_s": v["only_base"],
            }
            for k, v in pairwise.items()
        },
        "neighbors": [
            {
                "a": n["a"],
                "b": n["b"],
                "common": n["common"],
                "only_a_n": len(n["only_a"]),
                "only_b_n": len(n["only_b"]),
                "only_a_s": n["only_a"],
                "only_b_s": n["only_b"],
            }
            for n in neighbors
        ],
        "only_one_series_s": only_one,
        "presence_pattern_counts": [
            {
                "series": [key_tuple[i] for i, bit in enumerate(mask) if bit],
                "n": len(times),
            }
            for mask, times in sorted(
                by_mask.items(), key=lambda x: (-len(x[1]), x[0])
            )
            if sum(mask) > 0
        ],
    }
    OUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines: list[str] = []
    lines.append("# tilt/perc 잠금 소재 피크 타임스탬프 차이")
    lines.append("")
    lines.append(
        "이미 잠긴 perc/tilt 클릭 시리즈의 **어느 시각에 어떤 소재만 피크가 "
        "찍히는지** 보기 위한 문서. 검출기 재튜닝 없음."
    )
    lines.append("")
    lines.append("## 규칙")
    lines.append("")
    lines.append(f"- 매칭 허용: **±{int(TOL * 1000)}ms** (본선 관례와 동일)")
    lines.append("- 피크: 기존 manifest `peak_times_s`만 사용")
    lines.append(
        f"- 페어와이즈 기준선: **{SHORT.get(baseline, baseline)}** "
        f"(`{baseline}`)"
    )
    lines.append("- 시각 표기: `m:ss.mmm` (초 단위 괄호 병기)")
    lines.append(f"- 기계용 전체 목록: `{OUT_JSON.name}`")
    lines.append("")
    lines.append("## 포함 시리즈")
    lines.append("")
    lines.append("| key | short | peaks | source |")
    lines.append("|-----|-------|------:|--------|")
    for k in key_tuple:
        lines.append(
            f"| `{k}` | {SHORT.get(k, k)} | {counts[k]} | "
            f"{SOURCES.get(k, '?')} |"
        )
    lines.append("")
    if skipped:
        lines.append("### 스킵")
        lines.append("")
        for note in skipped:
            lines.append(f"- {note}")
        lines.append("")
    lines.append(
        f"{n_series}개 시리즈 모두 일치(±{int(TOL * 1000)}ms): "
        f"**{len(all_present)}**"
    )
    lines.append(f"불일치(전원이 아닌 클러스터): **{n_disagreement}**")
    lines.append(f"클러스터 합계: **{len(clusters)}**")
    lines.append("")

    lines.append(f"## vs {base_short} (기준) — 요약")
    lines.append("")
    lines.append("| 비교 | 공통 | 비교측만 | 기준만 |")
    lines.append("|------|-----:|---------:|-------:|")
    for k in key_tuple:
        if k == baseline:
            continue
        v = pairwise[k]
        sk = SHORT.get(k, k)
        lines.append(
            f"| {sk} ↔ {base_short} | {v['common']} | "
            f"{len(v['only_other'])} | {len(v['only_base'])} |"
        )
    lines.append("")

    lines.append("## 인접 시리즈 — 요약")
    lines.append("")
    lines.append("| 구간 | 공통 | 왼쪽만 | 오른쪽만 |")
    lines.append("|------|-----:|-------:|---------:|")
    for n in neighbors:
        a_s = SHORT.get(n["a"], n["a"])
        b_s = SHORT.get(n["b"], n["b"])
        lines.append(
            f"| {a_s} → {b_s} | {n['common']} | "
            f"{len(n['only_a'])} | {len(n['only_b'])} |"
        )
    lines.append("")

    lines.append("## 한 시리즈에만 있는 피크 (전용)")
    lines.append("")
    lines.append(
        "클러스터 기준: 같은 ±30ms 구간에 다른 포함 시리즈 피크가 **전혀** 없을 때."
    )
    lines.append("")
    for k in key_tuple:
        times = only_one[k]
        sk = SHORT.get(k, k)
        lines.append(f"### {sk} (`{k}`) 전용 — {len(times)}개")
        lines.append("")
        lines.append(_fmt_list(times))
        lines.append("")

    lines.append(f"## vs {base_short} — 전용 타임스탬프 상세")
    lines.append("")
    for k in key_tuple:
        if k == baseline:
            continue
        v = pairwise[k]
        sk = SHORT.get(k, k)
        lines.append(f"### {sk} ↔ {base_short}")
        lines.append("")
        lines.append(
            f"공통 {v['common']} / **{sk}만 {len(v['only_other'])}** / "
            f"**{base_short}만 {len(v['only_base'])}**"
        )
        lines.append("")
        lines.append(f"#### {sk}에만 있음")
        lines.append("")
        lines.append(_fmt_list(v["only_other"]))
        lines.append("")
        lines.append(f"#### {base_short}에만 있음 (이 비교에서)")
        lines.append("")
        lines.append(_fmt_list(v["only_base"]))
        lines.append("")

    lines.append("## 인접 시리즈 — 전용 타임스탬프 상세")
    lines.append("")
    for n in neighbors:
        a, b = n["a"], n["b"]
        a_s, b_s = SHORT.get(a, a), SHORT.get(b, b)
        lines.append(f"### {a_s} → {b_s}")
        lines.append("")
        lines.append(
            f"공통 {n['common']} / **{a_s}만 {len(n['only_a'])}** / "
            f"**{b_s}만 {len(n['only_b'])}**"
        )
        lines.append("")
        lines.append(f"#### {a_s}에만")
        lines.append("")
        lines.append(_fmt_list(n["only_a"]))
        lines.append("")
        lines.append(f"#### {b_s}에만")
        lines.append("")
        lines.append(_fmt_list(n["only_b"]))
        lines.append("")

    lines.append("## 존재 패턴 빈도 (참고)")
    lines.append("")
    lines.append("±30ms 클러스터가 어떤 시리즈 조합에 나타나는지 상위 목록.")
    lines.append("")
    lines.append("| series | n |")
    lines.append("|--------|--:|")
    for mask, times in sorted(
        by_mask.items(), key=lambda x: (-len(x[1]), x[0])
    ):
        if sum(mask) == 0:
            continue
        labels = "+".join(
            SHORT.get(key_tuple[i], key_tuple[i])
            for i, bit in enumerate(mask)
            if bit
        )
        lines.append(f"| {labels} | {len(times)} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "생성: `src/exp/s4_piano/stem_event_sculpt/gen_tilt_material_peak_diff_doc.py`"
    )

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_JSON}")
    print("included", key_tuple)
    print("counts", counts)
    print(f"all_present={len(all_present)} disagreement={n_disagreement}")
    if skipped:
        print("skipped", skipped)
    for k in key_tuple:
        if k == baseline:
            continue
        v = pairwise[k]
        print(
            f"  {SHORT.get(k, k)}↔{base_short} common={v['common']} "
            f"only_other={len(v['only_other'])} only_base={len(v['only_base'])}"
        )


if __name__ == "__main__":
    main()
