"""Multi-track Dir-style pipeline: 전체_adaptive · 506 · 764 sonify pack.

For each track:
  1) BS-Roformer piano stem
  2) 전체_adaptive = SuperFlux + peaks_adaptive on original (load_mono)
  3) 506-style = perc_tilt_k_env_adaptive ∪ LPC-order agreement-only (±30ms)
  4) 764-style = union(506, adaptive) via ±30ms 1:1 match
  5) Sonify on original mix × 0.20 (low bed) + 3 kHz clicks:
       전체_adaptive / 506 / 764 / 전체_adaptive_only / 506_only / common
       (+ optional LPC packs: lpc_agree_only / 764_lpcRescue5k_freqsep)

Outputs:
  out/stems/{alias}/bs_roformer/piano.wav (+ residual)
  out/stems/{alias}/event_sculpt/... intermediates + manifests
  out/sonify/pipeline_764_batch/{alias}/*.wav
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import soxr

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[2]
S4 = HERE.parent
ROOT = HERE.parents[3]
V1_TARGET = Path(r"E:\game\Music Hermeneutic AI\audio\target")
for p in (HERE, S4, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from _onset_wtmm_fusion import one_to_one_time_match  # noqa: E402
from audio_io import duration_s, load_mono, lufs_normalize  # noqa: E402
from config import MIN_EVENT_GAP_S, SR  # noqa: E402
from onset import band_envelopes, superflux_envelope  # noqa: E402
from peak_pick import peaks_adaptive  # noqa: E402

from gen_lpc_order_peak_diff_doc import ORDER_KEYS, TOL, cluster_presence  # noqa: E402
from io_util import (  # noqa: E402
    audio_stats,
    click_wav_name,
    read_stereo,
    sha256_file,
    write_float_wav,
    write_json,
    write_listening_wav,
)
from passes_hpss import hpss_components  # noqa: E402
from passes_lpc import lpc_components  # noqa: E402
from passes_percept import k_weight_mono  # noqa: E402
from passes_tilt import spectral_tilt  # noqa: E402

STEM_VAL = ROOT / "src" / "exp" / "s4_piano" / "stem_validation"
TORCH_PYTHON = STEM_VAL / "runtime" / "venv-torch" / "Scripts" / "python.exe"
BS_CLI = STEM_VAL / "runtime" / "venv-torch" / "Scripts" / "bs-roformer-infer.exe"
BS_MODELS = STEM_VAL / "models" / "bs_roformer"
BS_MODEL_ID = "roformer-model-bs-roformer-sw-by-jarredou"

SONIFY_ROOT = ROOT / "out" / "sonify" / "pipeline_764_batch"
STEMS_ROOT = ROOT / "out" / "stems"

MATCH_TOL_S = 0.03
BED_GAIN = 0.20
CLICK_DUR_MS = 12.0
CLICK_AMP = 0.7
CLICK_HZ = 3000.0
LPC_ORDERS = (4, 6, 8, 12, 24, 36)

TRACKS: list[dict[str, str]] = [
    {
        "alias": "AS",
        "label": "Angelic Sphere",
        "path": str(ROOT / "audio" / "02 Angelic Sphere (Extended Mix).wav"),
    },
    {
        "alias": "FD",
        "label": "FREEDOM DiVE",
        "path": str(ROOT / "audio" / "12. FREEDOM DiVE↓.wav"),
    },
    {
        "alias": "cry",
        "label": "cry of viyella",
        "path": str(V1_TARGET / "01.cry of viyella (2024 ReMaster).wav"),
    },
    {
        "alias": "GL",
        "label": "Grievous Lady",
        "path": str(V1_TARGET / "03.Grievous Lady.wav"),
    },
    {
        "alias": "VN",
        "label": "Viyella Nightmare",
        "path": "",  # resolved at runtime (unicode apostrophe)
    },
    {
        "alias": "SS",
        "label": "Swift Swing",
        "path": str(V1_TARGET / "09.Swift Swing (2024 ReMaster).wav"),
    },
]


def resolve_vn() -> Path:
    for p in sorted(V1_TARGET.iterdir()):
        if p.suffix.lower() == ".wav" and "Viyella" in p.name and "Nightmare" in p.name:
            return p
    raise FileNotFoundError("Viyella Nightmare wav not found in v1 target")


def resolve_tracks(aliases: list[str] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t in TRACKS:
        if aliases and t["alias"] not in aliases:
            continue
        path = Path(t["path"]) if t["path"] else resolve_vn()
        if not path.exists():
            # FREEDOM DiVE may have encoding-sensitive arrow in name
            if t["alias"] == "FD":
                cands = [
                    p
                    for p in (ROOT / "audio").iterdir()
                    if p.suffix.lower() == ".wav" and "FREEDOM" in p.name
                ]
                if not cands:
                    raise FileNotFoundError(t["path"])
                path = cands[0]
            else:
                raise FileNotFoundError(path)
        out.append({**t, "path": path})
    return out


def _click(freq_hz: float = CLICK_HZ) -> np.ndarray:
    n = int(SR * CLICK_DUR_MS / 1000.0)
    t = np.arange(n, dtype=np.float32) / SR
    env = np.exp(-t * 1000.0 / CLICK_DUR_MS)
    return (CLICK_AMP * env * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _overlay(mono: np.ndarray, times: list[float] | np.ndarray, click: np.ndarray) -> np.ndarray:
    out = mono.astype(np.float32, copy=True)
    for t in times:
        idx = int(float(t) * SR)
        end = min(idx + len(click), len(out))
        n = end - idx
        if n > 0:
            out[idx:end] += click[:n]
    return out


def _g_tag(g: float) -> str:
    return f"g{g:.2f}".replace(".", "p")


def ensure_stereo(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return np.column_stack([audio, audio]).astype(np.float32)
    if audio.shape[1] == 1:
        return np.repeat(audio, 2, axis=1).astype(np.float32)
    if audio.shape[1] > 2:
        return audio[:, :2].astype(np.float32)
    return audio.astype(np.float32)


def align_to_original(
    audio: np.ndarray, sample_rate: int, target_rate: int, target_frames: int, target_ch: int
) -> np.ndarray:
    if audio.ndim == 1:
        audio = audio[:, None]
    if sample_rate != target_rate:
        audio = soxr.resample(audio, sample_rate, target_rate, quality="HQ")
    if audio.shape[1] != target_ch:
        if target_ch == 1:
            audio = audio.mean(axis=1, keepdims=True)
        elif audio.shape[1] == 1:
            audio = np.repeat(audio, target_ch, axis=1)
        else:
            audio = audio[:, :target_ch]
    if len(audio) < target_frames:
        audio = np.pad(audio, ((0, target_frames - len(audio)), (0, 0)))
    else:
        audio = audio[:target_frames]
    return np.asarray(audio, dtype=np.float32)


def run_bs_roformer(alias: str, source: Path, *, force: bool) -> Path:
    out_piano = STEMS_ROOT / alias / "bs_roformer" / "piano.wav"
    if out_piano.exists() and not force:
        print(f"  [skip] stem exists: {out_piano}")
        return out_piano
    if not BS_CLI.exists():
        raise FileNotFoundError(BS_CLI)

    work = STEM_VAL / "work" / "batch" / alias
    raw = work / "raw_bs"
    staged_dir = work / "input"
    if force and work.exists():
        shutil.rmtree(work)
    staged_dir.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)

    # clean short name → bs-roformer prefixes outputs with stem
    staged = staged_dir / f"{alias.lower()}.wav"
    if not staged.exists() or staged.stat().st_size != source.stat().st_size:
        shutil.copy2(source, staged)

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["BS_ROFORMER_MODELS_PATH"] = str(BS_MODELS)
    env["TORCH_HOME"] = str(STEM_VAL / "models" / "torch")

    cmd = [
        str(BS_CLI),
        "--model",
        BS_MODEL_ID,
        "--models_dir",
        str(BS_MODELS),
        "--input_folder",
        str(staged_dir),
        "--store_dir",
        str(raw),
        "--device",
        "cuda",
    ]
    print("▸", " ".join(cmd))
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=str(STEM_VAL), env=env, check=True)
    print(f"  bs_roformer done in {time.perf_counter() - t0:.1f}s")

    piano_cands = sorted(raw.glob(f"{alias.lower()}_piano.wav"))
    if not piano_cands:
        piano_cands = sorted(raw.glob("*_piano.wav"))
    if not piano_cands:
        raise RuntimeError(f"no piano stem in {raw}")
    piano_raw = piano_cands[0]

    original, sr0 = sf.read(str(source), dtype="float32", always_2d=True)
    # Canonicalize to project SR so detect/sonify match Dir pipeline.
    if sr0 != SR:
        original = soxr.resample(original, sr0, SR, quality="HQ").astype(np.float32)
        sr0 = SR
    original = ensure_stereo(original)
    target_frames, target_ch = original.shape
    piano_a, sr_p = sf.read(str(piano_raw), dtype="float32", always_2d=True)
    piano = align_to_original(piano_a, sr_p, sr0, target_frames, target_ch)
    residual = original - piano

    dest = STEMS_ROOT / alias / "bs_roformer"
    dest.mkdir(parents=True, exist_ok=True)
    write_float_wav(dest / "piano.wav", piano, sr0)
    write_float_wav(dest / "residual.wav", residual, sr0)
    # also stash other stems if present
    for path in sorted(raw.glob(f"{alias.lower()}_*.wav")):
        name = path.stem.removeprefix(f"{alias.lower()}_")
        if name in {"piano", "instrumental", "no_piano"}:
            continue
        a, sr_a = sf.read(str(path), dtype="float32", always_2d=True)
        aligned = align_to_original(a, sr_a, sr0, target_frames, target_ch)
        write_float_wav(dest / f"{name}.wav", aligned, sr0)

    print(f"  wrote {dest / 'piano.wav'}")
    return dest / "piano.wav"


def sf_adaptive_peaks(mono: np.ndarray) -> np.ndarray:
    dur = float(len(mono) / SR)
    env, times = superflux_envelope(mono)
    bands = band_envelopes(mono)
    return np.asarray(
        peaks_adaptive(env, times, bands, dur, min_gap_s=MIN_EVENT_GAP_S),
        dtype=np.float64,
    )


def compute_adaptive(source: Path) -> np.ndarray:
    mono = load_mono(source)
    return sf_adaptive_peaks(mono)


def compute_kenv_adaptive(piano: np.ndarray, sculpt: Path) -> np.ndarray:
    """HPSS perc → spectral tilt → LUFS → mono mean → K-weight → SF adaptive."""
    harmonic, percussive = hpss_components(piano)
    write_listening_wav(sculpt / "hpss_harmonic.wav", harmonic, SR, limit_mode="clip")
    write_listening_wav(sculpt / "hpss_percussive.wav", percussive, SR, limit_mode="clip")

    tilted_raw, tilt_meta = spectral_tilt(percussive)
    L, R = lufs_normalize(tilted_raw[:, 0], tilted_raw[:, 1])
    tilted = np.column_stack([L, R]).astype(np.float32)
    mono_v2 = tilted.mean(axis=1).astype(np.float32)
    mono_k = k_weight_mono(mono_v2).astype(np.float32)

    tilt_dir = sculpt / "tilt"
    tilt_dir.mkdir(parents=True, exist_ok=True)
    write_listening_wav(tilt_dir / "perc_tilt_k_env_material_mono.wav", mono_k, SR, limit_mode="clip")

    pk = sf_adaptive_peaks(mono_k)
    write_json(
        tilt_dir / "tilt_k_env_adaptive_manifest.json",
        {
            "experiment": "perc_tilt_k_env_superflux_adaptive_batch",
            "tilt_meta": tilt_meta,
            "n_peaks": int(len(pk)),
            "peak_times_s": {"perc_tilt_k_env_adaptive": [float(t) for t in pk]},
        },
    )
    return pk


def compute_lpc_order_peaks(piano: np.ndarray, sculpt: Path) -> dict[str, list[float]]:
    pass2 = sculpt / "pass2"
    low_dir = pass2 / "lpc_low_order"
    high_dir = pass2 / "lpc_orders"
    low_dir.mkdir(parents=True, exist_ok=True)
    high_dir.mkdir(parents=True, exist_ok=True)

    series: dict[str, list[float]] = {}
    for order in LPC_ORDERS:
        print(f"    LPC o{order}…")
        t0 = time.perf_counter()
        residual, synthesis = lpc_components(piano, order=order)
        dest = low_dir if order < 12 else high_dir
        write_listening_wav(
            dest / f"lpc_o{order}_residual.wav", residual, SR, limit_mode="clip"
        )
        write_listening_wav(
            dest / f"lpc_o{order}_synthesis.wav", synthesis, SR, limit_mode="clip"
        )
        res_mono = residual.mean(axis=1).astype(np.float32)
        pk = sf_adaptive_peaks(res_mono)
        series[f"o{order}"] = [float(t) for t in pk]
        print(f"      peaks={len(pk)} ({time.perf_counter() - t0:.1f}s)")
    return series


def agreement_peaks(series: dict[str, list[float]]) -> list[float]:
    clusters = cluster_presence(series)
    all_six = frozenset(ORDER_KEYS)
    kept = [float(cl["rep"]) for cl in clusters if frozenset(cl["orders"]) == all_six]
    kept.sort()
    return kept


def fuse_506(kenv: np.ndarray, agree: list[float]) -> tuple[list[float], list[float]]:
    base = [float(t) for t in kenv]
    agree_only: list[float] = []
    for t in agree:
        if not any(abs(t - a) <= MATCH_TOL_S for a in base):
            base.append(float(t))
            agree_only.append(float(t))
    base.sort()
    agree_only.sort()
    return base, agree_only


def low_bed_from_source(source: Path) -> np.ndarray:
    audio, sr = sf.read(str(source), dtype="float32", always_2d=True)
    if sr != SR:
        audio = soxr.resample(audio, sr, SR, quality="HQ")
    mono = audio.mean(axis=1).astype(np.float32)
    return (mono * np.float32(BED_GAIN)).astype(np.float32)


def sonify_pack(
    alias: str,
    source: Path,
    *,
    adaptive: np.ndarray,
    p506: list[float],
    force: bool,
) -> dict[str, Any]:
    out_dir = SONIFY_ROOT / alias
    out_dir.mkdir(parents=True, exist_ok=True)
    bed = low_bed_from_source(source)
    click = _click(CLICK_HZ)
    gtag = _g_tag(BED_GAIN)

    p506_arr = np.asarray(p506, dtype=np.float64)
    common, only_506, only_ad = one_to_one_time_match(p506_arr, adaptive)
    common_t = [float(t) for t in common]
    only_506_t = [float(t) for t in only_506]
    only_ad_t = [float(t) for t in only_ad]
    union_t = sorted(common_t + only_506_t + only_ad_t)

    variants = {
        "전체_adaptive": [float(t) for t in adaptive],
        "506": [float(t) for t in p506],
        "764": union_t,
        "전체_adaptive_only": only_ad_t,
        "506_only": only_506_t,
        "common": common_t,
    }

    files: dict[str, Any] = {}
    for role, times in variants.items():
        name = f"{alias}_{role}_low_{gtag}_클릭_p{len(times)}.wav"
        path = out_dir / name
        if path.exists() and not force:
            print(f"  [skip] {name}")
            files[role] = {"path": str(path), "n_peaks": len(times), "skipped": True}
            continue
        audio = _overlay(bed, times, click)
        entry = write_listening_wav(path, audio, SR, limit_mode="clip")
        files[role] = {
            **entry,
            "role": role,
            "n_peaks": len(times),
            "bed": "origmix_mono_mean",
            "bed_gain": BED_GAIN,
        }
        print(f"  wrote {name}")

    manifest = {
        "experiment": "pipeline_764_batch",
        "alias": alias,
        "source": str(source).replace("\\", "/"),
        "source_sha256": sha256_file(source),
        "fixed_rules": {
            "전체_adaptive": "load_mono + SuperFlux + peaks_adaptive",
            "506": "kenv_adaptive ∪ LPC-order agreement-only (±30ms outside)",
            "764": "union(506, 전체_adaptive) via ±30ms 1:1 match",
            "only": "exclusive sides of the same match",
            "common": "intersection via ±30ms 1:1 match (both affirm)",
            "bed": f"origmix mono mean × {BED_GAIN}",
            "click_hz": CLICK_HZ,
            "match_tol_s": MATCH_TOL_S,
            "lpc_orders": list(LPC_ORDERS),
            "agreement_tol_s": TOL,
        },
        "counts": {
            "n_adaptive": int(len(adaptive)),
            "n_506": int(len(p506)),
            "n_764": int(len(union_t)),
            "common": int(len(common_t)),
            "only_506": int(len(only_506_t)),
            "only_adaptive": int(len(only_ad_t)),
        },
        "peak_times_s": {
            "adaptive": [float(t) for t in adaptive],
            "p506": [float(t) for t in p506],
            "union": union_t,
            "only_506": only_506_t,
            "only_adaptive": only_ad_t,
            "common": common_t,
        },
        "files": files,
    }
    write_json(out_dir / f"{alias}_pipeline_764_manifest.json", manifest)
    return manifest


def process_track(
    track: dict[str, Any],
    *,
    do_stem: bool,
    do_detect: bool,
    do_sonify: bool,
    force: bool,
) -> dict[str, Any]:
    alias = track["alias"]
    source: Path = track["path"]
    print(f"\n══ {alias} · {track['label']} ══")
    print(f"  source: {source}")

    piano_path = STEMS_ROOT / alias / "bs_roformer" / "piano.wav"
    sculpt = STEMS_ROOT / alias / "event_sculpt"
    detect_manifest = sculpt / "pipeline_detect_manifest.json"

    if do_stem:
        piano_path = run_bs_roformer(alias, source, force=force)

    adaptive: np.ndarray | None = None
    p506: list[float] | None = None

    if do_detect:
        if detect_manifest.exists() and not force:
            data = json.loads(detect_manifest.read_text(encoding="utf-8"))
            adaptive = np.asarray(data["peak_times_s"]["adaptive"], dtype=np.float64)
            p506 = [float(t) for t in data["peak_times_s"]["p506"]]
            print(
                f"  [skip] detect manifest "
                f"(adaptive={len(adaptive)}, 506={len(p506)})"
            )
        else:
            if not piano_path.exists():
                raise FileNotFoundError(piano_path)
            piano, sr = read_stereo(piano_path)
            if sr != SR:
                # resample stereo piano to project SR
                piano = soxr.resample(piano, sr, SR, quality="HQ").astype(np.float32)
            piano = ensure_stereo(piano)
            sculpt.mkdir(parents=True, exist_ok=True)

            print("  ▸ 전체_adaptive")
            t0 = time.perf_counter()
            adaptive = compute_adaptive(source)
            print(f"    n={len(adaptive)} ({time.perf_counter() - t0:.1f}s)")

            print("  ▸ k_env adaptive (502-family)")
            t0 = time.perf_counter()
            kenv = compute_kenv_adaptive(piano, sculpt)
            print(f"    n={len(kenv)} ({time.perf_counter() - t0:.1f}s)")

            print("  ▸ LPC order SF-adaptive")
            t0 = time.perf_counter()
            series = compute_lpc_order_peaks(piano, sculpt)
            print(f"    all orders ({time.perf_counter() - t0:.1f}s)")

            agree = agreement_peaks(series)
            p506, agree_only = fuse_506(kenv, agree)
            print(
                f"  ▸ 506-style: kenv={len(kenv)} + agree_only={len(agree_only)} "
                f"→ {len(p506)}; agreement_all6={len(agree)}"
            )

            payload = {
                "alias": alias,
                "source": str(source).replace("\\", "/"),
                "piano": str(piano_path).replace("\\", "/"),
                "counts": {
                    "adaptive": int(len(adaptive)),
                    "kenv": int(len(kenv)),
                    "agreement_all6": int(len(agree)),
                    "agree_only": int(len(agree_only)),
                    "p506": int(len(p506)),
                    "lpc_orders": {k: len(v) for k, v in series.items()},
                },
                "peak_times_s": {
                    "adaptive": [float(t) for t in adaptive],
                    "kenv": [float(t) for t in kenv],
                    "agreement_all6": agree,
                    "agree_only": agree_only,
                    "p506": p506,
                    "lpc_orders": series,
                },
            }
            write_json(detect_manifest, payload)

    if do_sonify:
        if adaptive is None or p506 is None:
            if not detect_manifest.exists():
                raise FileNotFoundError(detect_manifest)
            data = json.loads(detect_manifest.read_text(encoding="utf-8"))
            adaptive = np.asarray(data["peak_times_s"]["adaptive"], dtype=np.float64)
            p506 = [float(t) for t in data["peak_times_s"]["p506"]]
        assert adaptive is not None and p506 is not None
        return sonify_pack(alias, source, adaptive=adaptive, p506=p506, force=force)

    return {"alias": alias, "status": "ok"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracks",
        nargs="+",
        choices=[t["alias"] for t in TRACKS],
        default=None,
        help="subset of aliases (default: all)",
    )
    parser.add_argument(
        "--step",
        choices=("all", "stem", "detect", "sonify"),
        default="all",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    tracks = resolve_tracks(args.tracks)
    do_stem = args.step in ("all", "stem")
    do_detect = args.step in ("all", "detect")
    do_sonify = args.step in ("all", "sonify")

    print(f"tracks: {[t['alias'] for t in tracks]}")
    print(f"steps: stem={do_stem} detect={do_detect} sonify={do_sonify} force={args.force}")
    print(f"sonify out: {SONIFY_ROOT}")

    summaries: list[dict[str, Any]] = []
    for track in tracks:
        summaries.append(
            process_track(
                track,
                do_stem=do_stem,
                do_detect=do_detect,
                do_sonify=do_sonify,
                force=args.force,
            )
        )

    summary_path = SONIFY_ROOT / "batch_summary.json"
    if do_sonify:
        write_json(
            summary_path,
            {
                "experiment": "pipeline_764_batch",
                "tracks": [
                    {
                        "alias": s.get("alias"),
                        "counts": s.get("counts"),
                        "files": {
                            k: (v.get("path") if isinstance(v, dict) else v)
                            for k, v in (s.get("files") or {}).items()
                        },
                    }
                    for s in summaries
                    if "counts" in s
                ],
            },
        )
        print(f"\nsummary: {summary_path}")
    print("done.")


if __name__ == "__main__":
    main()
