"""시스템 vs 인간 비교. 설문 응답 수집 후 실행."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from config import SURVEY_DIR, OUT_DIR
from null_model import null_correlation


def load_ground_truth(path: Path | None = None) -> list[dict]:
    if path is None:
        path = SURVEY_DIR / "ground_truth.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_responses(path: Path) -> dict[str, int]:
    """설문 응답을 {clip_id: count} dict로 파싱.

    지원 형식: JSON {"clip_id": count, ...}
    """
    return json.loads(path.read_text(encoding="utf-8"))


def analyse(
    ground_truth: list[dict],
    responses: dict[str, int],
) -> dict:
    """전체 검증 분석."""
    paired = []
    for gt in ground_truth:
        cid = gt["clip_id"]
        if cid in responses:
            paired.append((gt["system_count"], responses[cid]))

    if len(paired) < 3:
        return {"error": f"쌍 {len(paired)}개 — 3개 미만"}

    sys_c = np.array([p[0] for p in paired])
    hum_c = np.array([p[1] for p in paired])

    rho, rho_p = spearmanr(sys_c, hum_c)
    mae = float(np.mean(np.abs(sys_c - hum_c)))
    pm1 = float(np.mean(np.abs(sys_c - hum_c) <= 1))
    pm3 = float(np.mean(np.abs(sys_c - hum_c) <= 3))

    perm = null_correlation(sys_c, hum_c)

    return {
        "n_clips": len(paired),
        "spearman_rho": round(float(rho), 4),
        "spearman_p": round(float(rho_p), 6),
        "mae": round(mae, 2),
        "pm1_accuracy": round(pm1, 4),
        "pm3_accuracy": round(pm3, 4),
        "permutation_test": perm,
        "ceiling_note": "r ~0.85, MAE ~2 (인간 계수의 고유 오차)",
        "floor_note": "r ~0 (무작위 배치)",
    }


def main() -> None:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    if len(sys.argv) < 2:
        print("사용법: python survey_analyse.py <responses.json>")
        sys.exit(1)

    resp_path = Path(sys.argv[1])
    gt = load_ground_truth()
    resp = load_responses(resp_path)
    result = analyse(gt, resp)

    out_path = OUT_DIR / "survey_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
