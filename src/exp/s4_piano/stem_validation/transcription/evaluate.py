"""Compare transcription onset clusters with all fixed s4 candidates."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE.parent))

from sonify_consensus import original_candidates  # noqa: E402


OUTPUT = ROOT / "out" / "transcription" / "Dir"
PRIMARY_TOLERANCE_S = 0.03
SENSITIVITY_TOLERANCES_S = (0.02, 0.05)


def load_cluster_times(model: str) -> np.ndarray:
    path = OUTPUT / f"{model}_onset_clusters_30ms.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(
        [cluster["representative_s"] for cluster in data["clusters"]],
        dtype=np.float64,
    )


def one_to_one_match(
    left: np.ndarray,
    right: np.ndarray,
    tolerance_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]]]:
    candidates: list[tuple[float, int, int]] = []
    for left_index, left_time in enumerate(left):
        lower = int(
            np.searchsorted(right, left_time - tolerance_s, side="left")
        )
        upper = int(
            np.searchsorted(right, left_time + tolerance_s, side="right")
        )
        candidates.extend(
            (
                abs(float(left_time - right[right_index])),
                left_index,
                right_index,
            )
            for right_index in range(lower, upper)
        )
    used_left: set[int] = set()
    used_right: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, left_index, right_index in sorted(candidates):
        if left_index not in used_left and right_index not in used_right:
            used_left.add(left_index)
            used_right.add(right_index)
            pairs.append((left_index, right_index))
    pairs.sort(key=lambda pair: left[pair[0]])
    common = np.asarray(
        [(left[i] + right[j]) / 2.0 for i, j in pairs],
        dtype=np.float64,
    )
    pair_data = [
        {
            "left_s": float(left[i]),
            "right_s": float(right[j]),
            "delta_ms": float((right[j] - left[i]) * 1000.0),
        }
        for i, j in pairs
    ]
    return (
        common,
        np.delete(left, sorted(used_left)),
        np.delete(right, sorted(used_right)),
        pair_data,
    )


def comparison(
    reference: np.ndarray,
    candidate: np.ndarray,
    tolerance_s: float,
) -> dict[str, Any]:
    common, reference_only, candidate_only, pairs = one_to_one_match(
        reference,
        candidate,
        tolerance_s,
    )
    matched = len(common)
    absolute_deltas = [abs(pair["delta_ms"]) for pair in pairs]
    return {
        "tolerance_s": tolerance_s,
        "reference_events": len(reference),
        "candidate_events": len(candidate),
        "matched": matched,
        "reference_only": len(reference_only),
        "candidate_only": len(candidate_only),
        "reference_coverage": (
            matched / len(reference) if len(reference) else None
        ),
        "reference_supported_fraction": (
            matched / len(candidate) if len(candidate) else None
        ),
        "timing_absolute_error_ms": {
            "median": (
                float(np.median(absolute_deltas))
                if absolute_deltas
                else None
            ),
            "p95": (
                float(np.percentile(absolute_deltas, 95))
                if absolute_deltas
                else None
            ),
        },
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    transkun = load_cluster_times("transkun_v2")
    basic = load_cluster_times("basic_pitch")
    candidates = original_candidates()

    model_common, transkun_only, basic_only, model_pairs = one_to_one_match(
        transkun,
        basic,
        PRIMARY_TOLERANCE_S,
    )
    transkun_395_common, transkun_only_395, candidate395_only, pairs_395 = (
        one_to_one_match(
            transkun,
            candidates["a2_posdist_rescue"],
            PRIMARY_TOLERANCE_S,
        )
    )
    _, both_common_missed_395, _, _ = one_to_one_match(
        model_common,
        candidates["a2_posdist_rescue"],
        PRIMARY_TOLERANCE_S,
    )

    comparisons: dict[str, Any] = {}
    for reference_name, reference in (
        ("transkun_v2", transkun),
        ("basic_pitch", basic),
    ):
        comparisons[reference_name] = {}
        for candidate_name, candidate in candidates.items():
            comparisons[reference_name][candidate_name] = {
                "primary_30ms": comparison(
                    reference,
                    candidate,
                    PRIMARY_TOLERANCE_S,
                ),
                "sensitivity": {
                    f"{int(tolerance * 1000)}ms": comparison(
                        reference,
                        candidate,
                        tolerance,
                    )
                    for tolerance in SENSITIVITY_TOLERANCES_S
                },
            }

    metrics = {
        "experiment": "BS stem transcription reference comparison",
        "terminology": {
            "reference_coverage": (
                "matched transcription clusters / transcription clusters"
            ),
            "reference_supported_fraction": (
                "matched candidate events / candidate events"
            ),
            "warning": "These are not true precision or true recall.",
        },
        "fixed_rules": {
            "cluster_window_s": 0.03,
            "primary_match_tolerance_s": PRIMARY_TOLERANCE_S,
            "sensitivity_match_tolerances_s": list(
                SENSITIVITY_TOLERANCES_S
            ),
            "matching": (
                "global greedy one-to-one, ascending absolute time delta"
            ),
        },
        "transcription_agreement_30ms": {
            **comparison(transkun, basic, PRIMARY_TOLERANCE_S),
            "common": len(model_common),
            "transkun_only": len(transkun_only),
            "basic_pitch_only": len(basic_only),
        },
        "comparisons": comparisons,
        "event_roles": {
            "transkun_all": transkun.tolist(),
            "transkun_basic_common": model_common.tolist(),
            "transkun_only": transkun_only.tolist(),
            "basic_pitch_only": basic_only.tolist(),
            "transkun_395_common": transkun_395_common.tolist(),
            "transkun_only_vs_395": transkun_only_395.tolist(),
            "candidate395_only": candidate395_only.tolist(),
            "both_transcriptions_common_missed_by_395": (
                both_common_missed_395.tolist()
            ),
        },
        "matched_pairs": {
            "transkun_vs_basic_pitch": model_pairs,
            "transkun_vs_candidate395": pairs_395,
        },
    }
    path = OUTPUT / "transcription_evaluation.json"
    path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"✓ evaluation: {path}")
    print(
        f"  Transkun={len(transkun)}, Basic Pitch={len(basic)}, "
        f"common={len(model_common)}"
    )
    for name, values in comparisons["transkun_v2"].items():
        result = values["primary_30ms"]
        print(
            f"  {name}: coverage={result['reference_coverage']:.3f}, "
            "supported="
            f"{result['reference_supported_fraction']:.3f}"
        )


if __name__ == "__main__":
    main()
