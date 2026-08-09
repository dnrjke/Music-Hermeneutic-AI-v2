"""Transcribe the fixed BS-Roformer piano stem with two independent models."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pretty_midi

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
INPUT = ROOT / "out" / "stems" / "Dir" / "bs_roformer" / "piano.wav"
OUTPUT = ROOT / "out" / "transcription" / "Dir"
RUNTIME = HERE.parent / "runtime"
TRANSKUN_PYTHON = RUNTIME / "venv-transkun" / "Scripts" / "python.exe"
BASIC_PITCH_PYTHON = RUNTIME / "venv-basicpitch" / "Scripts" / "python.exe"
TRANSKUN_PACKAGE = (
    RUNTIME / "venv-transkun" / "Lib" / "site-packages" / "transkun"
)
TRANSKUN_WEIGHT = TRANSKUN_PACKAGE / "pretrained" / "2.0.pt"
TRANSKUN_CONF = TRANSKUN_PACKAGE / "pretrained" / "2.0.conf"
BASIC_PITCH_MODEL = (
    RUNTIME
    / "venv-basicpitch"
    / "Lib"
    / "site-packages"
    / "basic_pitch"
    / "saved_models"
    / "icassp_2022"
    / "nmp"
)
CLUSTER_WINDOW_S = 0.03


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(path: Path) -> tuple[str, dict[str, str]]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    file_hashes = {
        item.relative_to(path).as_posix(): sha256(item)
        for item in files
    }
    digest = hashlib.sha256()
    for relative, file_hash in file_hashes.items():
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), file_hashes


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    print("▸", " ".join(command))
    child_env = os.environ.copy() if env is None else env.copy()
    user_path = os.environ.get("PATH", "")
    if sys.platform == "win32":
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            "Environment",
        ) as key:
            registered_path, _ = winreg.QueryValueEx(key, "Path")
        child_env["PATH"] = os.pathsep.join(
            [user_path, os.path.expandvars(registered_path)]
        )
    subprocess.run(command, check=True, env=child_env)


def midi_events(path: Path) -> list[dict[str, Any]]:
    midi = pretty_midi.PrettyMIDI(str(path))
    events: list[dict[str, Any]] = []
    for instrument_index, instrument in enumerate(midi.instruments):
        for note in instrument.notes:
            events.append(
                {
                    "onset_s": float(note.start),
                    "offset_s": float(note.end),
                    "pitch": int(note.pitch),
                    "velocity": int(note.velocity),
                    "instrument_index": instrument_index,
                    "program": int(instrument.program),
                    "is_drum": bool(instrument.is_drum),
                }
            )
    return sorted(
        events,
        key=lambda event: (
            event["onset_s"],
            event["pitch"],
            event["offset_s"],
        ),
    )


def cluster_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Complete-linkage onset clustering with a fixed 30 ms span."""
    groups: list[list[dict[str, Any]]] = []
    for event in events:
        if (
            not groups
            or event["onset_s"] - groups[-1][0]["onset_s"] > CLUSTER_WINDOW_S
        ):
            groups.append([event])
        else:
            groups[-1].append(event)

    clusters: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        onset_times = sorted(float(event["onset_s"]) for event in group)
        middle = len(onset_times) // 2
        if len(onset_times) % 2:
            representative = onset_times[middle]
        else:
            representative = (
                onset_times[middle - 1] + onset_times[middle]
            ) / 2.0
        clusters.append(
            {
                "cluster_index": index,
                "representative_s": representative,
                "span_s": onset_times[-1] - onset_times[0],
                "note_count": len(group),
                "notes": group,
            }
        )
    return clusters


def write_event_bundle(
    model: str,
    midi_path: Path,
    extra_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    events = midi_events(midi_path)
    clusters = cluster_events(events)
    event_path = OUTPUT / f"{model}_note_events.json"
    cluster_path = OUTPUT / f"{model}_onset_clusters_30ms.json"
    event_payload = {
        "model": model,
        "source": INPUT.as_posix(),
        "midi": midi_path.name,
        "note_count": len(events),
        "events": events,
    }
    if extra_events is not None:
        event_payload["model_raw_note_events"] = extra_events
    cluster_payload = {
        "model": model,
        "cluster_window_s": CLUSTER_WINDOW_S,
        "cluster_rule": "complete linkage: max(onset)-min(onset) <= 30ms",
        "representative": "median note onset",
        "note_count": len(events),
        "cluster_count": len(clusters),
        "clusters": clusters,
    }
    event_path.write_text(
        json.dumps(event_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    cluster_path.write_text(
        json.dumps(cluster_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "notes": events,
        "clusters": clusters,
        "event_path": event_path,
        "cluster_path": cluster_path,
    }


def transcribe_transkun(output_midi: Path) -> None:
    run(
        [
            str(TRANSKUN_PYTHON),
            "-m",
            "transkun.transcribe",
            str(INPUT),
            str(output_midi),
            "--device",
            "cuda",
        ]
    )


def transcribe_basic_pitch(
    output_midi: Path,
) -> list[dict[str, Any]]:
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict

    if Path(ICASSP_2022_MODEL_PATH).resolve() != BASIC_PITCH_MODEL.resolve():
        raise RuntimeError("Basic Pitch 기본 모델 경로가 예상과 다릅니다")
    _, midi, note_events = predict(INPUT)
    midi.write(str(output_midi))
    return [
        {
            "onset_s": float(onset),
            "offset_s": float(offset),
            "pitch": int(pitch),
            "amplitude": float(amplitude),
            "pitch_bends": (
                None
                if pitch_bends is None
                else [int(value) for value in pitch_bends]
            ),
        }
        for onset, offset, pitch, amplitude, pitch_bends in note_events
    ]


def package_version(name: str, python: Path) -> str:
    result = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import importlib.metadata as m;"
                f"print(m.version({name!r}))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def stats(bundle: dict[str, Any]) -> dict[str, Any]:
    notes = bundle["notes"]
    pitches = [event["pitch"] for event in notes]
    velocities = [event["velocity"] for event in notes]
    return {
        "note_count": len(notes),
        "cluster_count": len(bundle["clusters"]),
        "pitch_range": [min(pitches), max(pitches)] if pitches else None,
        "velocity_range": (
            [min(velocities), max(velocities)] if velocities else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--determinism-check",
        action="store_true",
        help="전사를 임시 출력으로 한 번 더 실행해 사건 목록을 비교합니다.",
    )
    args = parser.parse_args()

    if not INPUT.exists():
        raise FileNotFoundError(INPUT)
    for required in (
        TRANSKUN_PYTHON,
        BASIC_PITCH_PYTHON,
        TRANSKUN_WEIGHT,
        TRANSKUN_CONF,
        BASIC_PITCH_MODEL,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    transkun_midi = OUTPUT / "transkun_v2.mid"
    basic_midi = OUTPUT / "basic_pitch.mid"

    transcribe_transkun(transkun_midi)
    transkun_bundle = write_event_bundle("transkun_v2", transkun_midi)
    basic_raw = transcribe_basic_pitch(basic_midi)
    basic_bundle = write_event_bundle(
        "basic_pitch",
        basic_midi,
        extra_events=basic_raw,
    )

    deterministic: dict[str, Any] = {
        "checked": False,
        "comparison": "canonical parsed note-event and cluster JSON hashes",
    }
    if args.determinism_check:
        work = HERE / "work"
        work.mkdir(exist_ok=True)
        rerun_transkun_midi = work / "transkun_v2_rerun.mid"
        rerun_basic_midi = work / "basic_pitch_rerun.mid"
        transcribe_transkun(rerun_transkun_midi)
        rerun_transkun = {
            "notes": midi_events(rerun_transkun_midi),
        }
        rerun_transkun["clusters"] = cluster_events(rerun_transkun["notes"])
        rerun_basic_raw = transcribe_basic_pitch(rerun_basic_midi)
        rerun_basic = {
            "notes": midi_events(rerun_basic_midi),
            "raw": rerun_basic_raw,
        }
        rerun_basic["clusters"] = cluster_events(rerun_basic["notes"])
        deterministic = {
            "checked": True,
            "comparison": "canonical parsed note-event and cluster JSON hashes",
            "transkun_v2": {
                "notes_equal": (
                    canonical_json_hash(transkun_bundle["notes"])
                    == canonical_json_hash(rerun_transkun["notes"])
                ),
                "clusters_equal": (
                    canonical_json_hash(transkun_bundle["clusters"])
                    == canonical_json_hash(rerun_transkun["clusters"])
                ),
            },
            "basic_pitch": {
                "notes_equal": (
                    canonical_json_hash(basic_bundle["notes"])
                    == canonical_json_hash(rerun_basic["notes"])
                ),
                "clusters_equal": (
                    canonical_json_hash(basic_bundle["clusters"])
                    == canonical_json_hash(rerun_basic["clusters"])
                ),
                "raw_events_equal": (
                    canonical_json_hash(basic_raw)
                    == canonical_json_hash(rerun_basic_raw)
                ),
            },
        }

    basic_model_hash, basic_model_files = tree_sha256(BASIC_PITCH_MODEL)
    manifest = {
        "experiment": "BS-Roformer piano stem independent transcription",
        "input": {
            "path": INPUT.relative_to(ROOT).as_posix(),
            "sha256": sha256(INPUT),
        },
        "fixed_comparison": {
            "cluster_window_s": CLUSTER_WINDOW_S,
            "primary_match_tolerance_s": 0.03,
            "sensitivity_match_tolerances_s": [0.02, 0.05],
            "pedal_and_offsets_used_for_onset_evaluation": False,
        },
        "models": {
            "transkun_v2": {
                "role": "primary professional transcription reference",
                "license": "MIT",
                "package": "transkun",
                "package_version": package_version(
                    "transkun",
                    TRANSKUN_PYTHON,
                ),
                "runtime_packages": {
                    "torch": package_version("torch", TRANSKUN_PYTHON),
                    "torchaudio": package_version(
                        "torchaudio",
                        TRANSKUN_PYTHON,
                    ),
                    "ncls": package_version("ncls", TRANSKUN_PYTHON),
                    "ncls_note": (
                        "0.0.39 Windows cp311 wheel; newer ncls releases "
                        "provide no Windows wheel"
                    ),
                },
                "python": "3.11",
                "device": "cuda",
                "checkpoint": {
                    "path": TRANSKUN_WEIGHT.relative_to(HERE.parent).as_posix(),
                    "sha256": sha256(TRANSKUN_WEIGHT),
                },
                "config": {
                    "path": TRANSKUN_CONF.relative_to(HERE.parent).as_posix(),
                    "sha256": sha256(TRANSKUN_CONF),
                    "segment_size_s": 16,
                    "segment_hop_size_s": 8,
                    "cli_overrides": {"device": "cuda"},
                },
                "outputs": {
                    "midi": transkun_midi.name,
                    "midi_sha256": sha256(transkun_midi),
                    "events": transkun_bundle["event_path"].name,
                    "clusters": transkun_bundle["cluster_path"].name,
                },
                **stats(transkun_bundle),
            },
            "basic_pitch": {
                "role": "independent audit transcription",
                "license": "Apache-2.0",
                "package": "basic-pitch",
                "package_version": package_version(
                    "basic-pitch",
                    BASIC_PITCH_PYTHON,
                ),
                "runtime_packages": {
                    "tensorflow": package_version(
                        "tensorflow",
                        BASIC_PITCH_PYTHON,
                    ),
                    "pretty-midi": package_version(
                        "pretty-midi",
                        BASIC_PITCH_PYTHON,
                    ),
                },
                "python": "3.11",
                "backend": "TensorFlow SavedModel (package default)",
                "checkpoint": {
                    "path": BASIC_PITCH_MODEL.relative_to(HERE.parent).as_posix(),
                    "tree_sha256": basic_model_hash,
                    "files": basic_model_files,
                },
                "defaults": {
                    "onset_threshold": 0.5,
                    "frame_threshold": 0.3,
                    "minimum_note_length_ms": 127.7,
                    "minimum_frequency": None,
                    "maximum_frequency": None,
                    "multiple_pitch_bends": False,
                    "melodia_trick": True,
                    "midi_tempo": 120,
                },
                "outputs": {
                    "midi": basic_midi.name,
                    "midi_sha256": sha256(basic_midi),
                    "events": basic_bundle["event_path"].name,
                    "clusters": basic_bundle["cluster_path"].name,
                },
                **stats(basic_bundle),
            },
        },
        "determinism": deterministic,
    }
    manifest_path = OUTPUT / "transcription_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"✓ manifest: {manifest_path}")


if __name__ == "__main__":
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    main()
