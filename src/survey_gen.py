"""블라인드 설문 클립 생성. 설문 완료 전 정답 열지 않음 [D-v2-01]."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from config import WINDOW_S, SURVEY_DIR
from counter import WindowCount

FADE_MS = 50
N_CLIPS_PER_TRACK = 12
N_TERCILES = 3
CLIPS_PER_TERCILE = N_CLIPS_PER_TRACK // N_TERCILES


def _anon_id(track_name: str, index: int, salt: str = "v2blind") -> str:
    """익명 클립 ID. 트랙·위치 정보를 숨긴다."""
    raw = f"{salt}:{track_name}:{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:8]


def select_clips(
    counts: list[WindowCount],
    n_per_tercile: int = CLIPS_PER_TERCILE,
    seed: int = 20260807,
) -> list[WindowCount]:
    """사건 밀도 3분위에서 균등 추출."""
    rng = np.random.default_rng(seed)
    vals = np.array([w.count for w in counts])
    terciles = np.percentile(vals, [33.3, 66.7])

    low = [w for w in counts if w.count <= terciles[0]]
    mid = [w for w in counts if terciles[0] < w.count <= terciles[1]]
    high = [w for w in counts if w.count > terciles[1]]

    selected: list[WindowCount] = []
    for group in [low, mid, high]:
        if len(group) <= n_per_tercile:
            selected.extend(group)
        else:
            idx = rng.choice(len(group), size=n_per_tercile, replace=False)
            selected.extend(group[i] for i in sorted(idx))
    return sorted(selected, key=lambda w: w.start_s)


def cut_clip(
    audio_path: Path,
    window: WindowCount,
    clip_id: str,
    dest_dir: Path,
    sr: int = 44100,
) -> Path:
    """4초 클립 절단. 양쪽 페이드."""
    from audio_io import load_mono

    mono = load_mono(audio_path)
    s0 = int(window.start_s * sr)
    s1 = min(int(window.end_s * sr), len(mono))
    clip = mono[s0:s1].copy()

    fade = int(FADE_MS / 1000 * sr)
    if len(clip) > 2 * fade:
        clip[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
        clip[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)

    import soundfile as sf
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / f"{clip_id}.wav"
    sf.write(str(out_path), clip, sr)
    return out_path


def generate_survey(
    audio_path: Path,
    counts: list[WindowCount],
    dest_dir: Path | None = None,
) -> dict:
    """설문 패키지 전체 생성: 클립 + 템플릿 + 정답.

    Returns:
        메타데이터 dict (클립 목록, 정답 경로)
    """
    if dest_dir is None:
        dest_dir = SURVEY_DIR

    track_name = audio_path.name
    selected = select_clips(counts)
    clips_dir = dest_dir / "clips"

    clip_info: list[dict] = []
    for w in selected:
        cid = _anon_id(track_name, w.index)
        clip_path = cut_clip(audio_path, w, cid, clips_dir)
        clip_info.append({
            "clip_id": cid,
            "clip_file": clip_path.name,
        })

    ground_truth: list[dict] = []
    for w, ci in zip(selected, clip_info):
        ground_truth.append({
            "clip_id": ci["clip_id"],
            "track": track_name,
            "window_index": w.index,
            "start_s": w.start_s,
            "end_s": w.end_s,
            "system_count": w.count,
        })

    gt_path = dest_dir / "ground_truth.json"
    existing = []
    if gt_path.exists():
        existing = json.loads(gt_path.read_text(encoding="utf-8"))
    existing.extend(ground_truth)
    gt_path.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "track": track_name,
        "n_clips": len(clip_info),
        "clips": clip_info,
        "ground_truth_path": str(gt_path),
    }


def write_survey_template(dest_dir: Path | None = None) -> Path:
    """설문 템플릿 Markdown 작성."""
    if dest_dir is None:
        dest_dir = SURVEY_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    gt_path = dest_dir / "ground_truth.json"
    if not gt_path.exists():
        return dest_dir / "survey_template.md"

    entries = json.loads(gt_path.read_text(encoding="utf-8"))
    clip_ids = sorted(set(e["clip_id"] for e in entries))

    lines = [
        "# 블라인드 설문 — 사건 계수",
        "",
        "각 클립(4초)을 듣고, **뚜렷한 소리(타격, 건반음, 신스 등)가 몇 번 발생했는지** 숫자로 적어주세요.",
        "",
        "- 배경 노이즈나 잔향은 세지 않습니다",
        "- 뚜렷한 시작이 있는 소리만 셉니다 (타격, 음 시작, 신스 스탭 등)",
        "- 정확하지 않아도 괜찮습니다 — 자연스러운 인상을 적어주세요",
        "",
        "| 클립 | 계수 |",
        "|------|------|",
    ]
    for cid in clip_ids:
        lines.append(f"| {cid}.wav | |")

    lines.extend(["", "---", "설문 완료 후 ground_truth.json을 열어 비교합니다."])

    out_path = dest_dir / "survey_template.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
