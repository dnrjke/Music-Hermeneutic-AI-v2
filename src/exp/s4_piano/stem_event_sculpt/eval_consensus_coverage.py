"""Batch consensus-coverage eval for s4 Dir click candidates.

Reuses the locked session-10 stem consensus (2+ of BS/Spleeter/Demucs
A-2+positive-rescue peaks, ±30ms) from stem_consensus_metrics.json.

Coverage = one-to-one match common / |consensus|  (same as handoff 82.1%).
Diagnostic only — consensus is model-derived attribution evidence, not GT.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[2]
S4 = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(S4) not in sys.path:
    sys.path.insert(0, str(S4))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from _onset_wtmm_fusion import one_to_one_time_match  # noqa: E402
from io_util import OUTPUT_DIR, write_json  # noqa: E402

ROOT = HERE.parents[3]
CONSENSUS_METRICS = ROOT / "out" / "sonify" / "Dir" / "stem_consensus_metrics.json"
POSDIST_METRICS = ROOT / "out" / "sonify" / "Dir" / "posdist_metrics.json"
PASS2 = OUTPUT_DIR / "pass2"
TILT = OUTPUT_DIR / "tilt"
OUT_DIR = PASS2 / "consensus_coverage"
MATCH_TOL_S = 0.03
MODELS = ("bs_roformer", "spleeter", "demucs")


def _arr(times: list[float] | np.ndarray) -> np.ndarray:
    return np.asarray([float(t) for t in times], dtype=np.float64)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_locked_consensus() -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    if not CONSENSUS_METRICS.exists():
        raise FileNotFoundError(CONSENSUS_METRICS)
    data = _load_json(CONSENSUS_METRICS)
    consensus = _arr(data["stem_consensus"]["times_s"])
    if len(consensus) != int(data["stem_consensus"]["events_2plus"]):
        raise RuntimeError("consensus length mismatch vs events_2plus")
    model_peaks = {
        m: _arr(data["models"][m]["peak_times_s"]) for m in MODELS
    }
    return consensus, model_peaks, data


def collect_candidates() -> dict[str, np.ndarray]:
    """Named peak sets from locked manifests (no detector retune)."""
    cands: dict[str, np.ndarray] = {}

    # Session-10 baselines (sanity: a2_posdist_rescue → 82.1%)
    if POSDIST_METRICS.exists():
        pos = _load_json(POSDIST_METRICS)
        if "a2_posdist_rescue" in pos.get("peak_times_s", {}):
            cands["a2_posdist_rescue_395"] = _arr(
                pos["peak_times_s"]["a2_posdist_rescue"]
            )

    # Locked metrics' own candidate_comparison peaks via recomputed? prefer posdist.
    # Also echo stored coverage later from CONSENSUS file for baselines.

    # tilt / k_env
    kenv_ad = TILT / "tilt_k_env_adaptive_manifest.json"
    if kenv_ad.exists():
        cands["perc_tilt_k_env_adaptive"] = _arr(
            _load_json(kenv_ad)["peak_times_s"]["perc_tilt_k_env_adaptive"]
        )
    percept = TILT / "tilt_percept_manifest.json"
    if percept.exists():
        pts = _load_json(percept)["peak_times_s"]
        if "perc_tilt_k_env" in pts:
            cands["perc_tilt_k_env"] = _arr(pts["perc_tilt_k_env"])
    tilt_man = TILT / "tilt_manifest.json"
    if tilt_man.exists():
        pts = _load_json(tilt_man)["peak_times_s"]
        for key in ("perc_tilt_high", "perc_raw"):
            if key in pts:
                cands[key] = _arr(pts[key])

    tilt_mat = TILT / "tilt_material_peak_diff_timestamps.json"
    if tilt_mat.exists():
        # prefer agreement/disagreement from presence runners if available
        pass
    for kind, path in (
        (
            "tilt_material_agreement",
            TILT / "tilt_material_agreement_on_piano_manifest.json",
        ),
        (
            "tilt_material_disagreement",
            TILT / "tilt_material_disagreement_on_piano_manifest.json",
        ),
    ):
        if path.exists():
            cands[kind] = _arr(_load_json(path)["peak_times_s"])

    # LPC orders
    o12 = PASS2 / "lpc_o12_refine" / "lpc_o12_refine_manifest.json"
    if o12.exists():
        pts = _load_json(o12)["peak_times_s"]
        if "adaptive" in pts:
            cands["lpc_o12_sf_adaptive"] = _arr(pts["adaptive"])
        if "rms_plain" in pts:
            cands["lpc_o12_rms_plain"] = _arr(pts["rms_plain"])

    sf = PASS2 / "lpc_sf_adaptive" / "lpc_sf_adaptive_manifest.json"
    if sf.exists():
        pts = _load_json(sf)["peak_times_s"]
        for key in ("o24", "o36"):
            if key in pts:
                cands[f"lpc_{key}_sf_adaptive"] = _arr(pts[key])

    low = PASS2 / "lpc_sf_adaptive_on_piano" / "lpc_low_and_k_env_on_piano_manifest.json"
    if low.exists():
        pts = _load_json(low)["peak_times_s"]
        for key in ("o4", "o6", "o8"):
            if key in pts:
                cands[f"lpc_{key}_sf_adaptive"] = _arr(pts[key])

    onp = PASS2 / "lpc_sf_adaptive_on_piano"
    for kind, fname in (
        ("lpc_order_agreement", "lpc_order_agreement_on_piano_manifest.json"),
        ("lpc_order_disagreement", "lpc_order_disagreement_on_piano_manifest.json"),
    ):
        path = onp / fname
        if path.exists():
            cands[kind] = _arr(_load_json(path)["peak_times_s"])

    fusion = onp / "fusion_kenv_agree_o12db_on_piano_manifest.json"
    if fusion.exists():
        pts = _load_json(fusion)["peak_times_s"]
        cands["fusion_kenv_agree_o12db"] = _arr(pts["final_unified"])
        if pts.get("conservative_kenv_agree_only"):
            cands["fusion_kenv_agree_only_506"] = _arr(
                pts["conservative_kenv_agree_only"]
            )
        if pts.get("agree_only"):
            cands["fusion_agree_only_layer"] = _arr(pts["agree_only"])
        if pts.get("o12_deburst_extra"):
            cands["fusion_o12db_extra_layer"] = _arr(pts["o12_deburst_extra"])

    cmp_path = onp / "cmp506_vs_395_lowpiano_manifest.json"
    if cmp_path.exists():
        pts = _load_json(cmp_path)["peak_times_s"]
        if pts.get("union_unified"):
            cands["union_506_or_395"] = _arr(pts["union_unified"])
        elif pts.get("common") is not None:
            # fallback if older manifest without union key
            cands["union_506_or_395"] = _arr(
                list(pts.get("common", []))
                + list(pts.get("only_506", []))
                + list(pts.get("only_395", []))
            )
        if pts.get("common"):
            cands["cmp506_395_common"] = _arr(pts["common"])
        if pts.get("only_506"):
            cands["cmp506_only"] = _arr(pts["only_506"])
        if pts.get("only_395"):
            cands["cmp395_only"] = _arr(pts["only_395"])

    cmp_ad = onp / "cmp506_vs_dirAdaptive_lowpiano_manifest.json"
    if cmp_ad.exists():
        pts = _load_json(cmp_ad)["peak_times_s"]
        if pts.get("adaptive"):
            cands["dir_전체_adaptive"] = _arr(pts["adaptive"])
        if pts.get("union"):
            cands["union_506_or_dirAdaptive"] = _arr(pts["union"])
        if pts.get("common"):
            cands["cmp506_dirAdaptive_common"] = _arr(pts["common"])
        if pts.get("only_506"):
            cands["cmp506_vs_dirAdaptive_506only"] = _arr(pts["only_506"])
        if pts.get("only_adaptive"):
            cands["cmp506_vs_dirAdaptive_adaptiveOnly"] = _arr(
                pts["only_adaptive"]
            )

    return cands


def support_count(candidate: np.ndarray, model_peaks: dict[str, np.ndarray]) -> np.ndarray:
    """For each candidate peak, how many stem models support it (±30ms 1-1)."""
    # Approximate with independent masks (same spirit as sonify_consensus support_mask)
    from _onset_source_carving import one_to_one_pair_masks

    votes = np.zeros(len(candidate), dtype=np.int64)
    for model in MODELS:
        left_mask, _ = one_to_one_pair_masks(
            candidate, model_peaks[model], tolerance_s=MATCH_TOL_S
        )
        votes += left_mask.astype(np.int64)
    return votes


def evaluate_one(
    name: str,
    candidate: np.ndarray,
    consensus: np.ndarray,
    model_peaks: dict[str, np.ndarray],
) -> dict[str, Any]:
    common, consensus_only, candidate_only = one_to_one_time_match(
        consensus, candidate
    )
    from _onset_source_carving import one_to_one_pair_masks

    per_model: dict[str, int] = {}
    for m in MODELS:
        mask, _ = one_to_one_pair_masks(
            candidate, model_peaks[m], tolerance_s=MATCH_TOL_S
        )
        per_model[m] = int(np.sum(mask))

    votes = support_count(candidate, model_peaks) if len(candidate) else np.zeros(0)
    vote_hist = {str(v): int(np.sum(votes == v)) for v in range(4)}

    n_cons = len(consensus)
    n_cand = len(candidate)
    return {
        "name": name,
        "peaks": n_cand,
        "consensus_common": int(len(common)),
        "consensus_missed": int(len(consensus_only)),
        "candidate_only": int(len(candidate_only)),
        "consensus_coverage_pct": (
            100.0 * len(common) / n_cons if n_cons else 0.0
        ),
        "candidate_support_pct": (
            100.0 * len(common) / n_cand if n_cand else 0.0
        ),
        "per_model_supported": per_model,
        "candidate_stem_vote_hist": vote_hist,
    }


def render_md(
    rows: list[dict[str, Any]],
    *,
    consensus_n: int,
    locked_echo: dict[str, Any] | None,
) -> str:
    lines: list[str] = []
    lines.append("# Dir stem-합의 포괄률 일괄 평가")
    lines.append("")
    lines.append("## 정의 (세션 10과 동일)")
    lines.append("")
    lines.append(
        "- 참조(분모): BS/Spleeter/Demucs piano stem에 **동일** A-2+positive rescue "
        "적용 → ±30ms로 **2모델 이상** 합의 클러스터 "
        f"(**{consensus_n}** events)."
    )
    lines.append(
        "- 포괄률 = one-to-one ±30ms `common / |consensus|` "
        "(handoff의 82.1%와 같은 식)."
    )
    lines.append(
        "- **진단용**: stem 합의는 attribution 증거이지 absolute GT가 아님."
    )
    lines.append(
        f"- 잠긴 메트릭: `{CONSENSUS_METRICS.relative_to(ROOT).as_posix()}`"
    )
    lines.append("")
    if locked_echo:
        lines.append("## 잠긴 baseline 재확인 (stem_consensus_metrics)")
        lines.append("")
        lines.append("| candidate | peaks | coverage | common | missed | cand-only |")
        lines.append("|-----------|------:|---------:|-------:|-------:|----------:|")
        for name, v in locked_echo.items():
            lines.append(
                f"| {name} | {v['peaks']} | {v['consensus_coverage_pct']:.1f}% | "
                f"{v['consensus_common']} | {v['consensus_missed']} | "
                f"{v['candidate_only']} |"
            )
        lines.append("")

    lines.append("## 현재 산출 후보")
    lines.append("")
    lines.append(
        "| candidate | peaks | **coverage** | common | missed | cand-only | "
        "support% | BS | Spl | Dem |"
    )
    lines.append(
        "|-----------|------:|-------------:|-------:|-------:|----------:|"
        "---------:|---:|----:|----:|"
    )
    for r in rows:
        pm = r["per_model_supported"]
        lines.append(
            f"| `{r['name']}` | {r['peaks']} | "
            f"**{r['consensus_coverage_pct']:.1f}%** | "
            f"{r['consensus_common']} | {r['consensus_missed']} | "
            f"{r['candidate_only']} | {r['candidate_support_pct']:.1f}% | "
            f"{pm['bs_roformer']} | {pm['spleeter']} | {pm['demucs']} |"
        )
    lines.append("")
    lines.append("정렬: coverage 내림차순.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("생성: `src/exp/s4_piano/stem_event_sculpt/eval_consensus_coverage.py`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    consensus, model_peaks, locked = load_locked_consensus()
    locked_echo = locked.get("candidate_comparison", {})

    candidates = collect_candidates()
    if not candidates:
        raise SystemExit("no candidates found")

    # sanity: 395 should reproduce ~82.1
    if "a2_posdist_rescue_395" in candidates:
        sanity = evaluate_one(
            "a2_posdist_rescue_395",
            candidates["a2_posdist_rescue_395"],
            consensus,
            model_peaks,
        )
        expected = locked_echo.get("a2_posdist_rescue", {}).get(
            "consensus_coverage_pct"
        )
        if expected is not None and abs(sanity["consensus_coverage_pct"] - expected) > 0.05:
            raise RuntimeError(
                f"395 coverage mismatch: got {sanity['consensus_coverage_pct']} "
                f"expected {expected}"
            )

    rows = [
        evaluate_one(name, times, consensus, model_peaks)
        for name, times in candidates.items()
    ]
    rows.sort(key=lambda r: (-r["consensus_coverage_pct"], r["peaks"], r["name"]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "s4_sculpt_candidates_vs_stem_consensus",
        "diagnostic_only": True,
        "match_tol_s": MATCH_TOL_S,
        "consensus_source": str(CONSENSUS_METRICS).replace("\\", "/"),
        "consensus_n": int(len(consensus)),
        "consensus_definition": (
            "2+ of 3 piano stems, A-2+positive rescue, ±30ms clusters "
            "(locked session-10 metrics)"
        ),
        "warning": (
            "stem consensus is model-derived attribution evidence, not onset GT"
        ),
        "locked_baseline_echo": locked_echo,
        "candidates": {r["name"]: r for r in rows},
        "ranked": [
            {
                "name": r["name"],
                "peaks": r["peaks"],
                "consensus_coverage_pct": r["consensus_coverage_pct"],
                "consensus_common": r["consensus_common"],
            }
            for r in rows
        ],
    }
    write_json(OUT_DIR / "sculpt_consensus_coverage.json", payload)
    md = render_md(rows, consensus_n=len(consensus), locked_echo=locked_echo)
    (OUT_DIR / "sculpt_consensus_coverage.md").write_text(md, encoding="utf-8")

    print(f"consensus_n={len(consensus)}")
    print(f"candidates={len(rows)}")
    print(f"wrote {OUT_DIR / 'sculpt_consensus_coverage.md'}")
    print()
    print(f"{'candidate':40s} {'n':>5} {'cov%':>7} {'common':>6}")
    for r in rows:
        print(
            f"{r['name']:40s} {r['peaks']:5d} "
            f"{r['consensus_coverage_pct']:6.1f}% {r['consensus_common']:6d}"
        )


if __name__ == "__main__":
    main()
