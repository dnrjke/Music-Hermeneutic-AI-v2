"""Verify transcription artifacts, event invariants, and sonifications."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pretty_midi
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
OUTPUT = ROOT / "out" / "transcription" / "Dir"
SONIFY = ROOT / "out" / "sonify" / "Dir" / "transcription"
MANIFEST = OUTPUT / "transcription_manifest.json"
EVALUATION = OUTPUT / "transcription_evaluation.json"
SONIFY_MANIFEST = SONIFY / "sonification_manifest.json"
EPSILON_S = 1e-9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_cluster_file(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    clusters = data["clusters"]
    note_total = 0
    previous = -np.inf
    for expected_index, cluster in enumerate(clusters):
        if cluster["cluster_index"] != expected_index:
            raise RuntimeError(f"{path.name}: cluster index 불연속")
        notes = cluster["notes"]
        note_total += len(notes)
        onsets = sorted(float(note["onset_s"]) for note in notes)
        if not onsets:
            raise RuntimeError(f"{path.name}: 빈 cluster")
        span = onsets[-1] - onsets[0]
        if span > data["cluster_window_s"] + EPSILON_S:
            raise RuntimeError(f"{path.name}: 30ms cluster span 초과")
        representative = float(np.median(onsets))
        if abs(representative - cluster["representative_s"]) > EPSILON_S:
            raise RuntimeError(f"{path.name}: median 대표시각 불일치")
        if representative < previous:
            raise RuntimeError(f"{path.name}: cluster 시간 정렬 실패")
        previous = representative
    if note_total != data["note_count"]:
        raise RuntimeError(f"{path.name}: cluster note 합계 불일치")
    if len(clusters) != data["cluster_count"]:
        raise RuntimeError(f"{path.name}: cluster 수 불일치")
    return {
        "note_count": note_total,
        "cluster_count": len(clusters),
        "sha256": sha256(path),
    }


def verify_midi(path: Path, expected_notes: int) -> dict[str, Any]:
    midi = pretty_midi.PrettyMIDI(str(path))
    notes = [
        note
        for instrument in midi.instruments
        for note in instrument.notes
    ]
    if len(notes) != expected_notes:
        raise RuntimeError(
            f"{path.name}: MIDI note {len(notes)} != {expected_notes}"
        )
    for note in notes:
        if not 0 <= note.pitch <= 127:
            raise RuntimeError(f"{path.name}: pitch 범위 오류")
        if not 1 <= note.velocity <= 127:
            raise RuntimeError(f"{path.name}: velocity 범위 오류")
        if note.start < 0 or note.end <= note.start:
            raise RuntimeError(f"{path.name}: note 시간 범위 오류")
    return {
        "note_count": len(notes),
        "instruments": len(midi.instruments),
        "duration_s": float(midi.get_end_time()),
        "sha256": sha256(path),
    }


def verify_audio(
    path: Path,
    expected_base: Path,
) -> dict[str, Any]:
    info = sf.info(path)
    base_info = sf.info(expected_base)
    if info.samplerate != 44_100:
        raise RuntimeError(f"{path.name}: sample rate 오류")
    if info.channels != 1:
        raise RuntimeError(f"{path.name}: mono가 아님")
    if info.frames != base_info.frames:
        raise RuntimeError(
            f"{path.name}: frames {info.frames} != {base_info.frames}"
        )
    if info.subtype != "PCM_16":
        raise RuntimeError(f"{path.name}: subtype {info.subtype}")
    audio, _ = sf.read(path, dtype="float32")
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        raise RuntimeError(f"{path.name}: clipping {peak}")
    return {
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "duration_s": info.duration,
        "subtype": info.subtype,
        "peak": peak,
        "sha256": sha256(path),
    }


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    evaluation = json.loads(EVALUATION.read_text(encoding="utf-8"))
    sonify_manifest = json.loads(
        SONIFY_MANIFEST.read_text(encoding="utf-8")
    )

    deterministic = manifest["determinism"]
    if not deterministic["checked"]:
        raise RuntimeError("재실행 결정성 검사가 수행되지 않았습니다")
    for model in ("transkun_v2", "basic_pitch"):
        for key, value in deterministic[model].items():
            if not value:
                raise RuntimeError(f"{model} 결정성 실패: {key}")

    report: dict[str, Any] = {
        "experiment": "BS stem transcription artifact verification",
        "determinism": deterministic,
        "models": {},
        "evaluation": {},
        "sonifications": {},
    }
    for model in ("transkun_v2", "basic_pitch"):
        model_manifest = manifest["models"][model]
        cluster_path = OUTPUT / model_manifest["outputs"]["clusters"]
        midi_path = OUTPUT / model_manifest["outputs"]["midi"]
        cluster_report = verify_cluster_file(cluster_path)
        midi_report = verify_midi(
            midi_path,
            cluster_report["note_count"],
        )
        report["models"][model] = {
            "clusters": cluster_report,
            "midi": midi_report,
        }

    roles = evaluation["event_roles"]
    if (
        len(roles["transkun_basic_common"])
        + len(roles["transkun_only"])
        != len(roles["transkun_all"])
    ):
        raise RuntimeError("Transkun/Basic 역할 분해 합계 불일치")
    transkun_395_total = (
        len(roles["transkun_395_common"])
        + len(roles["transkun_only_vs_395"])
    )
    if transkun_395_total != len(roles["transkun_all"]):
        raise RuntimeError("Transkun/395 역할 분해 합계 불일치")
    report["evaluation"] = {
        "sha256": sha256(EVALUATION),
        "role_partition_checks": True,
    }

    base_paths = {
        "전체": ROOT / "audio" / "102 - Dir.wav",
        "BS피아노": (
            ROOT / "out" / "stems" / "Dir" / "bs_roformer" / "piano.wav"
        ),
    }
    for filename, item in sonify_manifest["outputs"].items():
        path = SONIFY / filename
        report["sonifications"][filename] = verify_audio(
            path,
            base_paths[item["base"]],
        )

    path = OUTPUT / "transcription_verification.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"✓ verification: {path}")
    print(
        f"  models={len(report['models'])}, "
        f"sonifications={len(report['sonifications'])}"
    )


if __name__ == "__main__":
    main()
