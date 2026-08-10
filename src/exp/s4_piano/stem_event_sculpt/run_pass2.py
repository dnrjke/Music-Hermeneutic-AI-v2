"""Pass2: percussive refine (A1/A2) + LPC order sweep."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from io_util import (  # noqa: E402
    OUTPUT_DIR,
    SOURCE_PIANO,
    SR,
    audio_stats,
    read_stereo,
    sha256_file,
    write_json,
    write_listening_wav,
)
from passes_lpc import LPC_PARAMS, lpc_components  # noqa: E402
from passes_perc_refine import (  # noqa: E402
    PERC_REFINE_PARAMS,
    attack_release,
    soft_env_gate,
)

PASS2_DIR = OUTPUT_DIR / "pass2"
SOURCE_PERC = OUTPUT_DIR / "hpss_percussive.wav"
LPC_ORDERS = (12, 24, 36)


def run_once() -> dict[str, Any]:
    if not SOURCE_PERC.exists():
        raise FileNotFoundError(
            f"{SOURCE_PERC} missing — run run_passes.py first"
        )
    if not SOURCE_PIANO.exists():
        raise FileNotFoundError(SOURCE_PIANO)

    perc, sr = read_stereo(SOURCE_PERC)
    piano, sr_p = read_stereo(SOURCE_PIANO)
    if sr != SR or sr_p != SR:
        raise RuntimeError("sample rate mismatch")
    if perc.shape[0] != piano.shape[0]:
        raise RuntimeError(
            f"length mismatch perc={perc.shape[0]} piano={piano.shape[0]}"
        )

    PASS2_DIR.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {
        "source_percussive": {
            "role": "pass2_input",
            "path": str(SOURCE_PERC.relative_to(HERE.parents[3])).replace("\\", "/"),
            "sha256": sha256_file(SOURCE_PERC),
            **audio_stats(perc),
            "frames": int(perc.shape[0]),
            "duration_s": float(perc.shape[0] / SR),
        },
        "source_piano": {
            "role": "lpc_sweep_input",
            "path": str(SOURCE_PIANO.relative_to(HERE.parents[3])).replace("\\", "/"),
            "sha256": sha256_file(SOURCE_PIANO),
            **audio_stats(piano),
            "frames": int(piano.shape[0]),
            "duration_s": float(piano.shape[0] / SR),
        },
    }
    runtime: dict[str, float] = {}
    refine_meta: dict[str, Any] = {}

    t0 = time.perf_counter()
    gated, gated_removed, meta_gate = soft_env_gate(perc)
    runtime["soft_env_gate"] = time.perf_counter() - t0
    refine_meta["soft_env_gate"] = meta_gate
    files["perc_soft_gate.wav"] = {
        **write_listening_wav(PASS2_DIR / "perc_soft_gate.wav", gated, SR),
        "role": "perc_refine_a1",
    }
    files["perc_soft_gate_removed.wav"] = {
        **write_listening_wav(
            PASS2_DIR / "perc_soft_gate_removed.wav", gated_removed, SR
        ),
        "role": "perc_refine_a1_removed",
    }

    t0 = time.perf_counter()
    shaped, shaped_removed, meta_ar = attack_release(perc)
    runtime["attack_release"] = time.perf_counter() - t0
    refine_meta["attack_release"] = meta_ar
    files["perc_attack_release.wav"] = {
        **write_listening_wav(PASS2_DIR / "perc_attack_release.wav", shaped, SR),
        "role": "perc_refine_a2",
    }
    files["perc_attack_release_removed.wav"] = {
        **write_listening_wav(
            PASS2_DIR / "perc_attack_release_removed.wav", shaped_removed, SR
        ),
        "role": "perc_refine_a2_removed",
    }

    for order in LPC_ORDERS:
        t0 = time.perf_counter()
        residual, synthesis = lpc_components(piano, order=order)
        runtime[f"lpc_order_{order}"] = time.perf_counter() - t0
        res_name = f"lpc_o{order}_residual.wav"
        syn_name = f"lpc_o{order}_synthesis.wav"
        files[res_name] = {
            **write_listening_wav(PASS2_DIR / res_name, residual, SR),
            "role": "lpc_sweep_residual",
            "order": order,
        }
        files[syn_name] = {
            **write_listening_wav(PASS2_DIR / syn_name, synthesis, SR),
            "role": "lpc_sweep_synthesis",
            "order": order,
        }

    manifest = {
        "experiment": "stem_event_sculpt_pass2",
        "fixed_rules": {
            "perc_input": "out/stems/Dir/event_sculpt/hpss_percussive.wav",
            "lpc_input": "out/stems/Dir/bs_roformer/piano.wav",
            "no_odf_mask": True,
            "no_395_compare_sonify": True,
            "no_click_overlay": True,
            "full_length": True,
            "soft_limit_listen_peak": 0.98,
            "lpc_order_grid": list(LPC_ORDERS),
            "lpc_no_extra_grid_points_after_listen": True,
        },
        "parameters": {
            "perc_refine": PERC_REFINE_PARAMS,
            "lpc_fixed": {
                "frame": LPC_PARAMS["frame"],
                "hop": LPC_PARAMS["hop"],
                "pre_emphasis": LPC_PARAMS["pre_emphasis"],
            },
            "lpc_orders": list(LPC_ORDERS),
        },
        "refine_meta": refine_meta,
        "runtime_s": runtime,
        "files": files,
    }
    write_json(PASS2_DIR / "pass2_manifest.json", manifest)
    return manifest


def determinism_check(manifest: dict[str, Any]) -> dict[str, Any]:
    first = {
        name: entry["sha256"]
        for name, entry in manifest["files"].items()
        if name.endswith(".wav")
    }
    second_manifest = run_once()
    second = {
        name: entry["sha256"]
        for name, entry in second_manifest["files"].items()
        if name.endswith(".wav")
    }
    mismatches = [name for name in first if first[name] != second.get(name)]
    report = {
        "matched": len(mismatches) == 0,
        "compared_files": sorted(first),
        "mismatches": mismatches,
    }
    write_json(PASS2_DIR / "pass2_determinism.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--determinism-check", action="store_true")
    args = parser.parse_args()

    print(f"perc input: {SOURCE_PERC}")
    print(f"piano input: {SOURCE_PIANO}")
    print(f"output: {PASS2_DIR}")
    manifest = run_once()
    for name, entry in manifest["files"].items():
        if name.endswith(".wav"):
            print(
                f"  {name}: peak={entry['peak']:.4f} rms={entry['rms']:.4f} "
                f"sha={entry['sha256'][:12]}"
            )
    print("runtime_s:", {k: round(v, 2) for k, v in manifest["runtime_s"].items()})
    print("refine_meta:", manifest["refine_meta"])

    if args.determinism_check:
        print("determinism-check: second run…")
        report = determinism_check(manifest)
        if not report["matched"]:
            raise SystemExit(f"determinism failed: {report['mismatches']}")
        print("determinism-check: OK")


if __name__ == "__main__":
    main()
