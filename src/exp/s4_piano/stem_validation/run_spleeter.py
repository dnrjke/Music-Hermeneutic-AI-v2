"""Run Spleeter 5-stem inference without an external FFmpeg executable."""
from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
from spleeter.separator import Separator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_wav", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    audio, sample_rate = sf.read(
        args.input_wav,
        dtype="float32",
        always_2d=True,
    )
    if sample_rate != 44_100:
        raise RuntimeError(f"Spleeter 입력 sample rate가 44.1kHz가 아님: {sample_rate}")

    separator = Separator("spleeter:5stems")
    prediction = separator.separate(audio)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stem_name, stem_audio in prediction.items():
        sf.write(
            args.output_dir / f"{stem_name}.wav",
            stem_audio,
            sample_rate,
            subtype="FLOAT",
        )


if __name__ == "__main__":
    main()
