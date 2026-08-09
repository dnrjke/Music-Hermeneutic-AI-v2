"""Generate and canonicalize three independent piano stem estimates."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "audio" / "102 - Dir.wav"
RUNTIME = HERE / "runtime"
MODELS = HERE / "models"
WORK = HERE / "work"
RAW = WORK / "raw"
OUTPUT = ROOT / "out" / "stems" / "Dir"
TORCH_PYTHON = RUNTIME / "venv-torch" / "Scripts" / "python.exe"
SPLEETER_PYTHON = RUNTIME / "venv-spleeter" / "Scripts" / "python.exe"
BS_CLI = RUNTIME / "venv-torch" / "Scripts" / "bs-roformer-infer.exe"
STAGED_INPUT = WORK / "input" / "dir.wav"

MODEL_INFO = {
    "bs_roformer": {
        "model": "BS-Roformer SW 6-stem",
        "model_id": "roformer-model-bs-roformer-sw-by-jarredou",
        "role": "strongest public piano benchmark reference",
        "limitation": (
            "checkpoint training provenance and weights license are not stated; "
            "local research diagnostic only"
        ),
    },
    "spleeter": {
        "model": "Spleeter 5-stem",
        "model_id": "spleeter:5stems",
        "role": "independent legacy baseline",
        "limitation": "legacy 11kHz-bandwidth model; piano/music-box leakage possible",
    },
    "demucs": {
        "model": "HTDemucs 6-source",
        "model_id": "htdemucs_6s",
        "role": "independent official open-source baseline",
        "limitation": (
            "official documentation warns that the piano source has substantial "
            "bleeding and artifacts"
        ),
    },
}


def run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    print("▸", " ".join(command))
    subprocess.run(command, cwd=cwd, env=env, check=True)


def stage_input() -> None:
    STAGED_INPUT.parent.mkdir(parents=True, exist_ok=True)
    if not STAGED_INPUT.exists() or STAGED_INPUT.stat().st_size != SOURCE.stat().st_size:
        shutil.copy2(SOURCE, STAGED_INPUT)


def run_bs_roformer(force: bool) -> Path:
    output_dir = RAW / "bs_roformer"
    expected = output_dir / "dir_piano.wav"
    if force and output_dir.exists():
        shutil.rmtree(output_dir)
    if expected.exists():
        return output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["BS_ROFORMER_MODELS_PATH"] = str(MODELS / "bs_roformer")
    env["TORCH_HOME"] = str(MODELS / "torch")
    run(
        [
            str(BS_CLI),
            "--model",
            MODEL_INFO["bs_roformer"]["model_id"],
            "--models_dir",
            str(MODELS / "bs_roformer"),
            "--input_folder",
            str(STAGED_INPUT.parent),
            "--store_dir",
            str(output_dir),
            "--device",
            "cuda",
        ],
        cwd=HERE,
        env=env,
    )
    return output_dir


def run_spleeter(force: bool) -> Path:
    output_dir = RAW / "spleeter"
    expected = output_dir / "piano.wav"
    if force and output_dir.exists():
        shutil.rmtree(output_dir)
    if expected.exists():
        return output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_cwd = MODELS / "spleeter"
    model_cwd.mkdir(parents=True, exist_ok=True)
    run(
        [
            str(SPLEETER_PYTHON),
            str(HERE / "run_spleeter.py"),
            str(STAGED_INPUT),
            str(output_dir),
        ],
        cwd=model_cwd,
        env=os.environ.copy(),
    )
    return output_dir


def run_demucs(force: bool) -> Path:
    root = RAW / "demucs"
    output_dir = root / "htdemucs_6s" / "dir"
    expected = output_dir / "piano.wav"
    if force and root.exists():
        shutil.rmtree(root)
    if expected.exists():
        return output_dir
    root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TORCH_HOME"] = str(MODELS / "torch")
    env["HF_HOME"] = str(MODELS / "huggingface")
    run(
        [
            str(TORCH_PYTHON),
            "-m",
            "demucs",
            "-n",
            "htdemucs_6s",
            "-d",
            "cuda",
            "--shifts",
            "0",
            "--float32",
            "--clip-mode",
            "none",
            "-o",
            str(root),
            str(STAGED_INPUT),
        ],
        cwd=HERE,
        env=env,
    )
    return output_dir


def align_audio(
    audio: np.ndarray,
    sample_rate: int,
    target_rate: int,
    target_frames: int,
    target_channels: int,
) -> np.ndarray:
    if audio.ndim == 1:
        audio = audio[:, None]
    if sample_rate != target_rate:
        audio = soxr.resample(audio, sample_rate, target_rate, quality="HQ")
    if audio.shape[1] != target_channels:
        if target_channels == 1:
            audio = audio.mean(axis=1, keepdims=True)
        elif audio.shape[1] == 1:
            audio = np.repeat(audio, target_channels, axis=1)
        else:
            raise RuntimeError(
                f"channel 정렬 불가: {audio.shape[1]} -> {target_channels}"
            )
    if len(audio) < target_frames:
        audio = np.pad(audio, ((0, target_frames - len(audio)), (0, 0)))
    else:
        audio = audio[:target_frames]
    return np.asarray(audio, dtype=np.float32)


def write_float_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write FLOAT WAV with libsndfile's volatile PEAK timestamp zeroed."""
    sf.write(path, audio, sample_rate, subtype="FLOAT")
    with path.open("r+b") as handle:
        header = handle.read(12)
        if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise RuntimeError(f"예상하지 못한 WAV 컨테이너: {path}")
        while True:
            chunk_header = handle.read(8)
            if len(chunk_header) < 8:
                break
            chunk_id = chunk_header[:4]
            chunk_size = int.from_bytes(chunk_header[4:], "little")
            payload_start = handle.tell()
            if chunk_id == b"PEAK" and chunk_size >= 8:
                handle.seek(payload_start + 4)
                handle.write(b"\x00\x00\x00\x00")
                return
            handle.seek(payload_start + chunk_size + (chunk_size % 2))


def canonicalize(model: str, raw_dir: Path) -> dict[str, object]:
    original, sample_rate = sf.read(SOURCE, dtype="float32", always_2d=True)
    target_frames, target_channels = original.shape
    destination = OUTPUT / model
    destination.mkdir(parents=True, exist_ok=True)

    if model == "bs_roformer":
        raw_stems = sorted(raw_dir.glob("dir_*.wav"))
        stem_name = lambda path: path.stem.removeprefix("dir_")
    else:
        raw_stems = sorted(raw_dir.glob("*.wav"))
        stem_name = lambda path: path.stem

    stems: dict[str, np.ndarray] = {}
    for path in raw_stems:
        name = stem_name(path)
        if name in {"instrumental", "no_piano"}:
            continue
        audio, stem_rate = sf.read(path, dtype="float32", always_2d=True)
        aligned = align_audio(
            audio,
            stem_rate,
            sample_rate,
            target_frames,
            target_channels,
        )
        stems[name] = aligned
        write_float_wav(destination / f"{name}.wav", aligned, sample_rate)

    if "piano" not in stems:
        raise RuntimeError(f"{model}: piano stem이 생성되지 않음")
    residual = original - stems["piano"]
    write_float_wav(destination / "residual.wav", residual, sample_rate)
    return {
        "sample_rate": sample_rate,
        "channels": target_channels,
        "frames": target_frames,
        "duration_s": target_frames / sample_rate,
        "stems": sorted(stems),
        "piano_peak": float(np.max(np.abs(stems["piano"]))),
        "residual_peak": float(np.max(np.abs(residual))),
        "piano_rms": float(np.sqrt(np.mean(stems["piano"] ** 2))),
        "residual_rms": float(np.sqrt(np.mean(residual**2))),
        "file_sha256": {
            **{
                name: sha256(destination / f"{name}.wav")
                for name in sorted(stems)
            },
            "residual": sha256(destination / "residual.wav"),
        },
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_manifest() -> list[dict[str, object]]:
    checkpoints: list[dict[str, object]] = []
    patterns = ("*.ckpt", "*.th", "*.safetensors", "*.index", "*.data-*")
    for root in (MODELS, HERE / "pretrained_models"):
        if not root.exists():
            continue
        for pattern in patterns:
            for path in root.rglob(pattern):
                checkpoints.append(
                    {
                        "path": str(path.relative_to(HERE)),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
    return sorted(checkpoints, key=lambda item: str(item["path"]))


def package_version(python: Path, package: str) -> str:
    command = [
        str(python),
        "-c",
        (
            "import importlib.metadata as m; "
            f"print(m.version({package!r}))"
        ),
    ]
    return subprocess.check_output(command, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_INFO),
        default=list(MODEL_INFO),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for required in (SOURCE, TORCH_PYTHON, SPLEETER_PYTHON, BS_CLI):
        if not required.exists():
            raise FileNotFoundError(required)
    stage_input()

    runners = {
        "bs_roformer": run_bs_roformer,
        "spleeter": run_spleeter,
        "demucs": run_demucs,
    }
    reports: dict[str, object] = {}
    for model in args.models:
        raw_dir = runners[model](args.force)
        reports[model] = canonicalize(model, raw_dir)
        print(f"  {model}: {reports[model]}")

    manifest = {
        "experiment": "s4 piano stem validation",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "diagnostic_only": True,
        "models": MODEL_INFO,
        "resolved_packages": {
            "torch": package_version(TORCH_PYTHON, "torch"),
            "demucs": package_version(TORCH_PYTHON, "demucs"),
            "bs-roformer-infer": package_version(
                TORCH_PYTHON,
                "bs-roformer-infer",
            ),
            "spleeter": package_version(SPLEETER_PYTHON, "spleeter"),
            "tensorflow": package_version(SPLEETER_PYTHON, "tensorflow"),
        },
        "outputs": reports,
        "checkpoints": checkpoint_manifest(),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "stem_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"  manifest: {OUTPUT / 'stem_manifest.json'}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"외부 stem 명령 실패: {exc}", file=sys.stderr)
        raise
