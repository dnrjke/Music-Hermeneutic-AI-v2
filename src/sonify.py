"""탐지된 사건의 청각 검증 도구.

생성물:
  1. click overlay — 원곡 위에 탐지 시각마다 클릭 삽입
  2. onset stem   — 탐지 시각 주변 실제 음만 추출 (나머지 무음)
  3. mix          — 스테레오: L=원곡, R=클릭트랙
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from config import SR, audio_paths, OUT_DIR, MIN_EVENT_GAP_S
from audio_io import load_mono
from onset import superflux_envelope, band_envelopes
from peak_pick import peaks, peaks_with_mid


def _click(sr: int = SR, freq: float = 3000.0, dur_ms: float = 12.0,
           amp: float = 0.7) -> np.ndarray:
    n = int(sr * dur_ms / 1000)
    t = np.arange(n, dtype=np.float32) / sr
    env = np.exp(-t * 1000 / dur_ms)
    return (amp * env * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def click_overlay(mono: np.ndarray, peak_times: np.ndarray,
                  sr: int = SR) -> np.ndarray:
    out = mono.copy()
    click = _click(sr)
    for t in peak_times:
        idx = int(t * sr)
        end = min(idx + len(click), len(out))
        seg = end - idx
        if seg > 0:
            out[idx:end] += click[:seg]
    return np.clip(out, -1.0, 1.0)


def onset_stem(mono: np.ndarray, peak_times: np.ndarray,
               sr: int = SR, window_ms: float = 80.0) -> np.ndarray:
    """탐지 시각 주변 실제 음만 추출. 페이드 인/아웃 포함."""
    out = np.zeros_like(mono)
    half = int(sr * window_ms / 2000)
    fade = int(sr * 5 / 1000)  # 5ms fade
    for t in peak_times:
        center = int(t * sr)
        s = max(0, center - half)
        e = min(len(mono), center + half)
        seg = mono[s:e].copy()
        if len(seg) > 2 * fade:
            seg[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
            seg[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
        out[s:e] = np.maximum(np.abs(out[s:e]), np.abs(seg)) * np.sign(seg + 1e-30)
    return out


def stereo_mix(mono: np.ndarray, peak_times: np.ndarray,
               sr: int = SR) -> np.ndarray:
    """L=원곡, R=클릭트랙."""
    click_track = np.zeros_like(mono)
    click = _click(sr)
    for t in peak_times:
        idx = int(t * sr)
        end = min(idx + len(click), len(click_track))
        seg = end - idx
        if seg > 0:
            click_track[idx:end] += click[:seg]
    return np.column_stack([mono, np.clip(click_track, -1.0, 1.0)])


def generate(audio_path: Path, start_s: float = 0.0,
             end_s: float | None = None,
             dest_dir: Path | None = None) -> dict[str, Path]:
    """구간 지정 청각 검증 파일 생성."""
    if dest_dir is None:
        dest_dir = OUT_DIR / "sonify"
    dest_dir.mkdir(parents=True, exist_ok=True)

    mono = load_mono(audio_path)
    dur = len(mono) / SR
    if end_s is None:
        end_s = dur

    env, times = superflux_envelope(mono)
    bands = band_envelopes(mono)
    pk = peaks_with_mid(env, bands["mid"], times)

    pk_full = peaks(env, times)
    pk_mid_only = np.array([t for t in pk if t not in set(pk_full.tolist())])

    mask = (pk >= start_s) & (pk < end_s)
    pk_seg = pk[mask]

    s0 = int(start_s * SR)
    s1 = min(int(end_s * SR), len(mono))
    mono_seg = mono[s0:s1]
    pk_local = pk_seg - start_s

    stem = audio_path.stem
    tag = f"{stem}_{start_s:.0f}-{end_s:.0f}s"

    paths = {}

    p = dest_dir / f"{tag}_click.wav"
    sf.write(str(p), click_overlay(mono_seg, pk_local), SR)
    paths["click_overlay"] = p

    p = dest_dir / f"{tag}_stem.wav"
    sf.write(str(p), onset_stem(mono_seg, pk_local), SR)
    paths["onset_stem"] = p

    p = dest_dir / f"{tag}_mix.wav"
    sf.write(str(p), stereo_mix(mono_seg, pk_local), SR)
    paths["stereo_mix"] = p

    # mid-rescued 피크만 따로 표시한 오버레이
    mask_mid = np.isin(pk_seg, pk_mid_only)
    if mask_mid.any():
        pk_rescued_local = pk_seg[mask_mid] - start_s
        p = dest_dir / f"{tag}_mid_rescued.wav"
        mid_click = _click(SR, freq=5000.0, dur_ms=15.0, amp=0.8)
        out = mono_seg.copy()
        for t in pk_rescued_local:
            idx = int(t * SR)
            end = min(idx + len(mid_click), len(out))
            seg = end - idx
            if seg > 0:
                out[idx:end] += mid_click[:seg]
        sf.write(str(p), np.clip(out, -1.0, 1.0), SR)
        paths["mid_rescued_overlay"] = p

    return paths


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser(description="탐지 사건 청각 검증")
    ap.add_argument("--track", help="파일명 일부로 필터")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    args = ap.parse_args()

    paths = audio_paths(args.track)
    if not paths:
        print("오디오 파일을 찾을 수 없습니다")
        sys.exit(1)

    for p in paths:
        print(f"\n▸ {p.name}  [{args.start:.0f}-{args.end or '끝'}s]")
        result = generate(p, args.start, args.end)
        for kind, path in result.items():
            print(f"  {kind:25s}: {path}")


if __name__ == "__main__":
    main()
