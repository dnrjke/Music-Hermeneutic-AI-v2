"""전체 파이프라인. 탐지 → 계수 → 설문 생성까지 한번에.

사용:
    python pipeline.py                    # 전곡 분석 + 설문 생성
    python pipeline.py --track cry        # 특정 트랙만
    python pipeline.py --no-survey        # 설문 없이 계수만
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from config import audio_paths, OUT_DIR, SURVEY_DIR
from audio_io import load_mono, duration_s
from onset import superflux_envelope, band_envelopes
from peak_pick import peaks, peaks_with_mid, otsu
from counter import count_events, summary, save
from survey_gen import generate_survey, write_survey_template


def run_track(audio_path: Path) -> dict:
    """단일 트랙 전체 파이프라인."""
    name = audio_path.name
    dur = duration_s(audio_path)
    mono = load_mono(audio_path)

    env, times = superflux_envelope(mono)
    bands = band_envelopes(mono)
    thr = otsu(env[np.isfinite(env)])
    pk = peaks_with_mid(env, bands["mid"], times)

    counts = count_events(pk, dur)
    stats = summary(counts)
    save(name, counts, stats, OUT_DIR)

    return {
        "track": name,
        "duration_s": round(dur, 1),
        "n_frames": len(env),
        "otsu_threshold": round(thr, 4),
        "n_events": len(pk),
        "density_per_s": round(len(pk) / dur, 2),
        "window_stats": stats,
        "peak_times": pk,
        "counts": counts,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser(description="Music Hermeneutic AI v2 파이프라인")
    ap.add_argument("--track", help="파일명 일부로 필터")
    ap.add_argument("--no-survey", action="store_true", help="설문 생성 생략")
    args = ap.parse_args()

    paths = audio_paths(args.track)
    if not paths:
        print("오디오 파일을 찾을 수 없습니다")
        sys.exit(1)

    print(f"=== Music Hermeneutic AI v2 — 사건 탐지 파이프라인 ===\n")
    print(f"대상: {len(paths)}곡")

    all_results: list[dict] = []
    for p in paths:
        print(f"\n{'='*60}")
        print(f"▸ {p.name}")
        print(f"{'='*60}")

        r = run_track(p)
        all_results.append(r)

        print(f"  길이: {r['duration_s']}s")
        print(f"  Otsu 임계: {r['otsu_threshold']}")
        print(f"  사건: {r['n_events']}개 ({r['density_per_s']}/s)")
        s = r["window_stats"]
        print(f"  윈도우: {s['n_windows']}개  계수 [{s['count_min']}, {s['count_max']}]"
              f"  중앙 {s['count_median']:.0f}  평균 {s['count_mean']:.1f}±{s['count_std']:.1f}")

        if not args.no_survey:
            survey = generate_survey(p, r["counts"])
            print(f"  설문 클립: {survey['n_clips']}개 → survey/clips/")

    if not args.no_survey:
        tmpl = write_survey_template()
        print(f"\n설문 템플릿: {tmpl}")
        print("⚠ 설문 완료 전 survey/ground_truth.json을 열지 마세요 [D-v2-01]")

    pipeline_out = {
        "tracks": [
            {k: v for k, v in r.items() if k not in ("peak_times", "counts")}
            for r in all_results
        ],
    }
    out_path = OUT_DIR / "pipeline_result.json"
    out_path.write_text(
        json.dumps(pipeline_out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n전체 결과: {out_path}")


if __name__ == "__main__":
    main()
