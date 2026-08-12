"""전체_adaptive + original-mix LPC agreement → adaptive_plus → 764 sonify pack.

For each batch track, writes under:
  out/sonify/pipeline_764_batch/{alias}/adaptive_orig_lpc_agree/
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[2]
S4 = HERE.parent
ROOT = HERE.parents[3]
for p in (HERE, S4, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from _onset_wtmm_fusion import one_to_one_time_match  # noqa: E402
from config import MIN_EVENT_GAP_S, SR  # noqa: E402
from gen_lpc_order_peak_diff_doc import ORDER_KEYS, cluster_presence  # noqa: E402
from io_util import write_json, write_listening_wav  # noqa: E402
from onset import band_envelopes, superflux_envelope  # noqa: E402
from passes_lpc import lpc_components  # noqa: E402
from peak_pick import peaks_adaptive  # noqa: E402

ALIASES = ["AS", "FD", "cry", "GL", "VN", "SS"]
LPC_ORDERS = (4, 6, 8, 12, 24, 36)
MATCH_TOL_S = 0.03
BED_GAIN = 0.20
CLICK_DUR_MS = 12.0
CLICK_AMP = 0.7
FREQ_HZ = 3000.0
FREQ_LPC_HZ = 5000.0
SONIFY_ROOT = ROOT / "out" / "sonify" / "pipeline_764_batch"
SUBDIR = "adaptive_orig_lpc_agree"


def ensure_stereo(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return np.column_stack([audio, audio]).astype(np.float32)
    if audio.shape[1] == 1:
        return np.repeat(audio, 2, axis=1).astype(np.float32)
    return audio[:, :2].astype(np.float32)


def load_stereo_sr(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SR:
        audio = soxr.resample(audio, sr, SR, quality="HQ").astype(np.float32)
    return ensure_stereo(audio)


def sf_adaptive_peaks(mono: np.ndarray) -> np.ndarray:
    dur = float(len(mono) / SR)
    env, times = superflux_envelope(mono)
    bands = band_envelopes(mono)
    return np.asarray(
        peaks_adaptive(env, times, bands, dur, min_gap_s=MIN_EVENT_GAP_S),
        dtype=np.float64,
    )


def agreement_peaks(series: dict[str, list[float]]) -> list[float]:
    clusters = cluster_presence(series)
    all_six = frozenset(ORDER_KEYS)
    kept = [float(cl["rep"]) for cl in clusters if frozenset(cl["orders"]) == all_six]
    kept.sort()
    return kept


def add_uncovered(
    base: list[float], candidates: list[float]
) -> tuple[list[float], list[float]]:
    out = list(base)
    added: list[float] = []
    for t in candidates:
        if not any(abs(t - a) <= MATCH_TOL_S for a in out):
            out.append(float(t))
            added.append(float(t))
    out.sort()
    added.sort()
    return out, added


def click(freq_hz: float) -> np.ndarray:
    n = int(SR * CLICK_DUR_MS / 1000.0)
    t = np.arange(n, dtype=np.float32) / SR
    env = np.exp(-t * 1000.0 / CLICK_DUR_MS)
    return (CLICK_AMP * env * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def overlay(mono: np.ndarray, times: list[float], c: np.ndarray) -> np.ndarray:
    out = mono.astype(np.float32, copy=True)
    for t in times:
        idx = int(float(t) * SR)
        end = min(idx + len(c), len(out))
        n = end - idx
        if n > 0:
            out[idx:end] += c[:n]
    return out


def low_bed(stereo: np.ndarray) -> np.ndarray:
    mono = stereo.mean(axis=1).astype(np.float32)
    return (mono * np.float32(BED_GAIN)).astype(np.float32)


def main() -> None:
    c3 = click(FREQ_HZ)
    c5 = click(FREQ_LPC_HZ)
    gtag = f"g{BED_GAIN:.2f}".replace(".", "p")
    batch: list[dict] = []

    for alias in ALIASES:
        t_track = time.perf_counter()
        print(f"\n══ {alias} ══", flush=True)
        son_path = SONIFY_ROOT / alias / f"{alias}_pipeline_764_manifest.json"
        det_path = (
            ROOT / "out" / "stems" / alias / "event_sculpt" / "pipeline_detect_manifest.json"
        )
        son = json.loads(son_path.read_text(encoding="utf-8"))
        det = json.loads(det_path.read_text(encoding="utf-8"))
        source = Path(son["source"])
        adaptive = [float(t) for t in son["peak_times_s"]["adaptive"]]
        p506 = [float(t) for t in det["peak_times_s"]["p506"]]

        out_dir = SONIFY_ROOT / alias / SUBDIR
        out_dir.mkdir(parents=True, exist_ok=True)
        cache_path = out_dir / f"{alias}_orig_lpc_orders_manifest.json"

        if cache_path.exists():
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            series = {
                k: [float(x) for x in v]
                for k, v in cache["peak_times_s"]["lpc_orders"].items()
            }
            print(f"  [cache] {cache_path.name}", flush=True)
            stereo = load_stereo_sr(source)
        else:
            stereo = load_stereo_sr(source)
            print(
                f"  source frames={len(stereo)} ({len(stereo) / SR:.1f}s)",
                flush=True,
            )
            series = {}
            for order in LPC_ORDERS:
                t0 = time.perf_counter()
                print(f"  LPC o{order} on origmix…", flush=True)
                residual, _ = lpc_components(stereo, order=order)
                pk = sf_adaptive_peaks(residual.mean(axis=1).astype(np.float32))
                series[f"o{order}"] = [float(t) for t in pk]
                print(
                    f"    peaks={len(pk)} ({time.perf_counter() - t0:.1f}s)",
                    flush=True,
                )
            write_json(
                cache_path,
                {
                    "experiment": "origmix_lpc_order_sf_adaptive",
                    "alias": alias,
                    "source": str(source).replace("\\", "/"),
                    "orders": list(LPC_ORDERS),
                    "peak_times_s": {"lpc_orders": series},
                    "counts": {k: len(v) for k, v in series.items()},
                },
            )

        agree = agreement_peaks(series)
        adaptive_plus, agree_only = add_uncovered(adaptive, agree)
        print(
            f"  agree_all6={len(agree)} agree_only={len(agree_only)} "
            f"adaptive={len(adaptive)} adaptive_plus={len(adaptive_plus)}",
            flush=True,
        )

        p506_arr = np.asarray(p506, dtype=np.float64)
        adap_plus_arr = np.asarray(adaptive_plus, dtype=np.float64)
        common, only_506, only_ad = one_to_one_time_match(p506_arr, adap_plus_arr)
        common_t = [float(t) for t in common]
        only_506_t = [float(t) for t in only_506]
        only_ad_t = [float(t) for t in only_ad]
        union_t = sorted(common_t + only_506_t + only_ad_t)
        print(
            f"  764=506∪adapPlus → {len(union_t)} "
            f"(c={len(common_t)} 6o={len(only_506_t)} ao={len(only_ad_t)}); "
            f"ref={son['counts']['n_764']}",
            flush=True,
        )

        bed = low_bed(stereo)
        tag = f"{alias}_adapOrigLpc"
        variants = {
            "전체_adaptive_plus": adaptive_plus,
            "origLpc_agree_only": agree_only,
            "764": union_t,
            "전체_adaptive": adaptive,
            "common": common_t,
            "506_only": only_506_t,
            "adaptive_plus_only": only_ad_t,
            "506": p506,
        }
        files: dict = {}
        for role, times in variants.items():
            name = f"{tag}_{role}_low_{gtag}_클릭_p{len(times)}.wav"
            entry = write_listening_wav(
                out_dir / name, overlay(bed, times, c3), SR, limit_mode="clip"
            )
            files[role] = {
                **entry,
                "role": role,
                "n_peaks": len(times),
                "bed_gain": BED_GAIN,
                "click_hz": FREQ_HZ,
            }
            print(f"  wrote {name}", flush=True)

        fs = overlay(bed, union_t, c3)
        fs = overlay(fs, agree_only, c5)
        fs_name = (
            f"{tag}_764_origLpcRescue5k_freqsep_low_{gtag}"
            f"_클릭_p{len(union_t)}_lpc{len(agree_only)}.wav"
        )
        fs_entry = write_listening_wav(out_dir / fs_name, fs, SR, limit_mode="clip")
        files["764_origLpcRescue5k_freqsep"] = {
            **fs_entry,
            "role": "764_origLpcRescue5k_freqsep",
            "n_peaks": len(union_t),
            "n_lpc_highlight": len(agree_only),
            "click_hz_base": FREQ_HZ,
            "click_hz_lpc": FREQ_LPC_HZ,
        }
        print(f"  wrote {fs_name}", flush=True)

        manifest = {
            "experiment": "adaptive_plus_origmix_lpc_agreement",
            "alias": alias,
            "note": (
                "전체_adaptive(원곡) + 원곡 LPC o4..o36 SF-adaptive all-six agreement "
                "중 adaptive ±30ms 밖(agree_only). "
                "764 = 기존 506(piano kenv∪piano-agree) ∪ adaptive_plus."
            ),
            "source": str(source).replace("\\", "/"),
            "subdir": SUBDIR,
            "fixed_rules": {
                "lpc_input": "original mix stereo (not piano stem)",
                "lpc_orders": list(LPC_ORDERS),
                "agree_only_vs": "전체_adaptive",
                "match_tol_s": MATCH_TOL_S,
                "764": "506 ∪ adaptive_plus",
                "506_unchanged": True,
                "bed_gain": BED_GAIN,
            },
            "counts": {
                "lpc_orders": {k: len(v) for k, v in series.items()},
                "agreement_all6": len(agree),
                "agree_only": len(agree_only),
                "n_adaptive": len(adaptive),
                "n_adaptive_plus": len(adaptive_plus),
                "n_506": len(p506),
                "n_764": len(union_t),
                "common": len(common_t),
                "only_506": len(only_506_t),
                "only_adaptive_plus": len(only_ad_t),
                "n_764_agree_ref": int(son["counts"]["n_764"]),
                "delta_764_vs_ref": int(len(union_t) - son["counts"]["n_764"]),
            },
            "peak_times_s": {
                "lpc_orders": series,
                "agreement_all6": agree,
                "agree_only": agree_only,
                "adaptive": adaptive,
                "adaptive_plus": adaptive_plus,
                "p506": p506,
                "union": union_t,
                "common": common_t,
                "only_506": only_506_t,
                "only_adaptive_plus": only_ad_t,
            },
            "files": files,
        }
        man_path = out_dir / f"{alias}_adaptive_orig_lpc_agree_manifest.json"
        write_json(man_path, manifest)

        son.setdefault("variants", {})
        son["variants"]["adaptive_orig_lpc_agree"] = {
            "dir": str(out_dir.relative_to(ROOT)).replace("\\", "/"),
            "manifest": str(man_path.relative_to(ROOT)).replace("\\", "/"),
            "counts": manifest["counts"],
        }
        write_json(son_path, son)

        batch.append(
            {
                "alias": alias,
                "counts": {
                    k: v
                    for k, v in manifest["counts"].items()
                    if k != "lpc_orders"
                },
                "dir": str(out_dir.relative_to(ROOT)).replace("\\", "/"),
            }
        )
        print(f"  done {alias} in {time.perf_counter() - t_track:.1f}s", flush=True)

    summary_path = SONIFY_ROOT / "adaptive_orig_lpc_agree_batch_summary.json"
    write_json(
        summary_path,
        {
            "experiment": "adaptive_plus_origmix_lpc_agreement",
            "subdir": SUBDIR,
            "n_tracks": len(batch),
            "tracks": batch,
        },
    )
    print(f"\nsummary {summary_path}", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
