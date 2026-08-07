"""윈도우별 사건 총 계수. 4초 윈도우, 비중첩, 선언 후 고정 [D-v2-03]."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from config import WINDOW_S


@dataclass
class WindowCount:
    index: int
    start_s: float
    end_s: float
    count: int


def count_events(
    peak_times: np.ndarray,
    duration_s: float,
    window_s: float = WINDOW_S,
) -> list[WindowCount]:
    """비중첩 윈도우별 사건 계수."""
    n_windows = int(np.ceil(duration_s / window_s))
    result: list[WindowCount] = []
    for i in range(n_windows):
        t0 = i * window_s
        t1 = min((i + 1) * window_s, duration_s)
        mask = (peak_times >= t0) & (peak_times < t1)
        result.append(WindowCount(
            index=i, start_s=round(t0, 3), end_s=round(t1, 3),
            count=int(mask.sum()),
        ))
    return result


def summary(counts: list[WindowCount]) -> dict:
    """트랙 요약 통계."""
    vals = np.array([w.count for w in counts])
    return {
        "n_windows": len(counts),
        "total_events": int(vals.sum()),
        "count_min": int(vals.min()),
        "count_max": int(vals.max()),
        "count_median": float(np.median(vals)),
        "count_mean": round(float(vals.mean()), 2),
        "count_std": round(float(vals.std()), 2),
        "count_iqr": round(float(np.percentile(vals, 75) - np.percentile(vals, 25)), 2),
    }


def save(
    track_name: str,
    counts: list[WindowCount],
    stats: dict,
    dest: Path,
) -> Path:
    """윈도우 계수와 통계를 JSON으로 저장."""
    dest.mkdir(parents=True, exist_ok=True)
    stem = track_name.rsplit(".", 1)[0]
    out_path = dest / f"event_counts_{stem}.json"
    payload = {
        "track": track_name,
        "window_s": WINDOW_S,
        "summary": stats,
        "windows": [asdict(w) for w in counts],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    return out_path
