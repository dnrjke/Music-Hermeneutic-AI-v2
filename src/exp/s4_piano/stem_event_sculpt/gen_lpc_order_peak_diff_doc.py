"""Generate LPC-order sf_adaptive peak timestamp comparison markdown.

Compares o4/o6/o8/o12/o24/o36 SuperFlux+peaks_adaptive times (±30ms match).
Baseline for pairwise: o12 adaptive (adopted).
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
PASS2 = ROOT / "out" / "stems" / "Dir" / "event_sculpt" / "pass2"
OUT_MD = PASS2 / "lpc_order_peak_diff_timestamps.md"
OUT_JSON = PASS2 / "lpc_order_peak_diff_timestamps.json"
TOL = 0.03
ORDER_KEYS = ("o4", "o6", "o8", "o12", "o24", "o36")


def _fmt(t: float) -> str:
    """mm:ss.mmm for listen scrubbing."""
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


def load_series() -> dict[str, list[float]]:
    low = json.loads(
        (
            PASS2
            / "lpc_sf_adaptive_on_piano"
            / "lpc_low_and_k_env_on_piano_manifest.json"
        ).read_text(encoding="utf-8")
    )
    o12 = json.loads(
        (PASS2 / "lpc_o12_refine" / "lpc_o12_refine_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    sf = json.loads(
        (PASS2 / "lpc_sf_adaptive" / "lpc_sf_adaptive_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    series = {
        "o4": [float(t) for t in low["peak_times_s"]["o4"]],
        "o6": [float(t) for t in low["peak_times_s"]["o6"]],
        "o8": [float(t) for t in low["peak_times_s"]["o8"]],
        "o12": [float(t) for t in o12["peak_times_s"]["adaptive"]],
        "o24": [float(t) for t in sf["peak_times_s"]["o24"]],
        "o36": [float(t) for t in sf["peak_times_s"]["o36"]],
    }
    return series


def cluster_presence(
    series: dict[str, list[float]],
) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    for k in ORDER_KEYS:
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
    series = load_series()
    counts = {k: len(series[k]) for k in ORDER_KEYS}

    pairwise: dict[str, Any] = {}
    for k in ORDER_KEYS:
        if k == "o12":
            continue
        _c, only_other, only_o12 = match(series[k], series["o12"])
        pairwise[k] = {
            "common": len(_c),
            "only_other": only_other,
            "only_o12": only_o12,
        }

    neighbors: list[dict[str, Any]] = []
    for a, b in zip(ORDER_KEYS, ORDER_KEYS[1:]):
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

    clusters = cluster_presence(series)
    by_mask: dict[tuple[int, ...], list[float]] = defaultdict(list)
    for cl in clusters:
        mask = tuple(1 if o in cl["orders"] else 0 for o in ORDER_KEYS)
        by_mask[mask].append(float(cl["rep"]))

    only_one = {
        k: by_mask[tuple(1 if o == k else 0 for o in ORDER_KEYS)]
        for k in ORDER_KEYS
    }
    all_six = by_mask[tuple([1] * 6)]

    # JSON dump (full times)
    payload = {
        "tol_s": TOL,
        "detector": "superflux_envelope + peaks_adaptive",
        "baseline": "o12 adaptive (lpc_o12_refine)",
        "sources": {
            "o4/o6/o8": "pass2/lpc_sf_adaptive_on_piano/lpc_low_and_k_env_on_piano_manifest.json",
            "o12": "pass2/lpc_o12_refine/lpc_o12_refine_manifest.json → peak_times_s.adaptive",
            "o24/o36": "pass2/lpc_sf_adaptive/lpc_sf_adaptive_manifest.json",
        },
        "counts": counts,
        "pairwise_vs_o12": {
            k: {
                "common": v["common"],
                "only_other_n": len(v["only_other"]),
                "only_o12_n": len(v["only_o12"]),
                "only_other_s": v["only_other"],
                "only_o12_s": v["only_o12"],
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
        "only_one_order_s": only_one,
        "all_six_n": len(all_six),
        "presence_pattern_counts": [
            {
                "orders": [ORDER_KEYS[i] for i, bit in enumerate(mask) if bit],
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

    # Markdown
    lines: list[str] = []
    lines.append("# LPC order별 sf_adaptive 피크 타임스탬프 차이")
    lines.append("")
    lines.append("청취 시 **어느 시각에 어떤 order만 클릭이 찍히는지** 보기 위한 문서.")
    lines.append("")
    lines.append("## 규칙")
    lines.append("")
    lines.append(f"- 매칭 허용: **±{int(TOL*1000)}ms** (본선 관례와 동일)")
    lines.append("- 탐지: SuperFlux + `peaks_adaptive` (동일 검출기, residual order만 다름)")
    lines.append("- 페어와이즈 기준선: **o12 adaptive** (채택 최선, 383)")
    lines.append("- 시각 표기: `m:ss.mmm` (초 단위 괄호 병기)")
    lines.append(f"- 기계용 전체 목록: `{OUT_JSON.name}`")
    lines.append("")
    lines.append("## 피크 수")
    lines.append("")
    lines.append("| order | peaks |")
    lines.append("|------:|------:|")
    for k in ORDER_KEYS:
        lines.append(f"| {k} | {counts[k]} |")
    lines.append("")
    lines.append(f"6개 order 모두 일치(±{int(TOL*1000)}ms): **{len(all_six)}**")
    lines.append("")

    lines.append("## vs o12 (채택 기준) — 요약")
    lines.append("")
    lines.append("| 비교 | 공통 | 비교측만 | o12만 |")
    lines.append("|------|-----:|---------:|------:|")
    for k in ORDER_KEYS:
        if k == "o12":
            continue
        v = pairwise[k]
        lines.append(
            f"| {k} ↔ o12 | {v['common']} | {len(v['only_other'])} | {len(v['only_o12'])} |"
        )
    lines.append("")

    lines.append("## 인접 order — 요약")
    lines.append("")
    lines.append("| 구간 | 공통 | 왼쪽만 | 오른쪽만 |")
    lines.append("|------|-----:|-------:|---------:|")
    for n in neighbors:
        lines.append(
            f"| {n['a']} → {n['b']} | {n['common']} | "
            f"{len(n['only_a'])} | {len(n['only_b'])} |"
        )
    lines.append("")

    lines.append("## 한 order에만 있는 피크 (전용)")
    lines.append("")
    lines.append(
        "클러스터 기준: 같은 ±30ms 구간에 다른 order 피크가 **전혀** 없을 때."
    )
    lines.append("")
    for k in ORDER_KEYS:
        times = only_one[k]
        lines.append(f"### {k} 전용 — {len(times)}개")
        lines.append("")
        lines.append(_fmt_list(times))
        lines.append("")

    lines.append("## vs o12 — 전용 타임스탬프 상세")
    lines.append("")
    for k in ORDER_KEYS:
        if k == "o12":
            continue
        v = pairwise[k]
        lines.append(f"### {k} ↔ o12")
        lines.append("")
        lines.append(
            f"공통 {v['common']} / **{k}만 {len(v['only_other'])}** / "
            f"**o12만 {len(v['only_o12'])}**"
        )
        lines.append("")
        lines.append(f"#### {k}에만 있음")
        lines.append("")
        lines.append(_fmt_list(v["only_other"]))
        lines.append("")
        lines.append("#### o12에만 있음 (이 비교에서)")
        lines.append("")
        lines.append(_fmt_list(v["only_o12"]))
        lines.append("")

    lines.append("## 인접 order — 전용 타임스탬프 상세")
    lines.append("")
    for n in neighbors:
        a, b = n["a"], n["b"]
        lines.append(f"### {a} → {b}")
        lines.append("")
        lines.append(
            f"공통 {n['common']} / **{a}만 {len(n['only_a'])}** / "
            f"**{b}만 {len(n['only_b'])}**"
        )
        lines.append("")
        lines.append(f"#### {a}에만")
        lines.append("")
        lines.append(_fmt_list(n["only_a"]))
        lines.append("")
        lines.append(f"#### {b}에만")
        lines.append("")
        lines.append(_fmt_list(n["only_b"]))
        lines.append("")

    lines.append("## 존재 패턴 빈도 (참고)")
    lines.append("")
    lines.append("±30ms 클러스터가 어떤 order 조합에 나타나는지 상위 목록.")
    lines.append("")
    lines.append("| orders | n |")
    lines.append("|--------|--:|")
    for mask, times in sorted(
        by_mask.items(), key=lambda x: (-len(x[1]), x[0])
    ):
        if sum(mask) == 0:
            continue
        labels = "+".join(ORDER_KEYS[i] for i, bit in enumerate(mask) if bit)
        lines.append(f"| {labels} | {len(times)} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        f"생성: `src/exp/s4_piano/stem_event_sculpt/gen_lpc_order_peak_diff_doc.py`"
    )

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_JSON}")
    print("counts", counts)
    for k in ORDER_KEYS:
        if k == "o12":
            continue
        v = pairwise[k]
        print(
            f"  {k}↔o12 common={v['common']} "
            f"only_{k}={len(v['only_other'])} only_o12={len(v['only_o12'])}"
        )


if __name__ == "__main__":
    main()
