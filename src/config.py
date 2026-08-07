"""상수와 경로. v1 features.py에서 이식한 오디오 파라미터 + v2 고유 설정."""
from __future__ import annotations

from pathlib import Path

# ── 오디오 파라미터 (v1 features.py 그대로)
SR = 44100
N_FFT = 2048          # 46 ms
HOP = 256             # 5.8 ms
N_MELS = 128
FMIN = 20.0
TARGET_LUFS = -23.0

# ── SuperFlux 파라미터 (v1 features.py, Böck & Widmer DAFx-13)
SUPERFLUX_LAG = 2
SUPERFLUX_MAX = 3

# ── 온셋 대역 정의 (v1 features.py ONSET_BANDS)
ONSET_BANDS: list[tuple[str, float, float]] = [
    ("low",  30.0,   120.0),    # 킥
    ("mid",  120.0,  2000.0),   # 신디 레이어
    ("high", 2000.0, 20000.0),  # 하이햇·노이즈·riser
]

# ── v2 고유 상수
WINDOW_S = 4.0              # 계수 윈도우 (초). 선언 후 고정 [D-v2-03]
MIN_EVENT_GAP_S = 0.030     # 30ms. 200BPM 1/64음표(37.5ms) 미만. 물리적 도출

# ── 경로
V2_ROOT = Path(__file__).resolve().parent.parent
V1_AUDIO = Path(r"E:\game\Music Hermeneutic AI\audio\target")
OUT_DIR = V2_ROOT / "out"
SURVEY_DIR = V2_ROOT / "survey"

AUDIO_EXTS = {".wav", ".flac", ".aiff", ".aif", ".w64"}


def audio_paths(track_filter: str | None = None) -> list[Path]:
    """오디오 파일 검색. v2/audio/ 우선, v1 audio/target/ 폴백."""
    v2_audio = V2_ROOT / "audio"
    sources: list[Path] = []
    for d in [v2_audio, V1_AUDIO]:
        if d.exists():
            sources.extend(
                p for p in sorted(d.iterdir())
                if p.suffix.lower() in AUDIO_EXTS
            )
    seen: set[str] = set()
    result: list[Path] = []
    for p in sources:
        if p.name not in seen:
            seen.add(p.name)
            result.append(p)
    if track_filter:
        result = [p for p in result if track_filter in p.name]
    return result
