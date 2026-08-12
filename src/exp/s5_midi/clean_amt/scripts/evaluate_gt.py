#!/usr/bin/env python3
"""Evaluate clean_amt notes.json against GT MIDI (onset pitch matching).

Independent of s4_piano. Uses mido only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mido


def load_notes_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def midi_to_notes(path: Path) -> list[dict]:
    mid = mido.MidiFile(path)
    abs_t = 0.0
    active: dict[int, tuple[float, int]] = {}
    notes: list[dict] = []
    for msg in mid:
        abs_t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            active[msg.note] = (abs_t, msg.velocity)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            if msg.note in active:
                onset, vel = active.pop(msg.note)
                notes.append(
                    {
                        "onset_s": onset,
                        "offset_s": abs_t,
                        "pitch": msg.note,
                        "velocity": vel,
                    }
                )
    return notes


def match_onset_f1(
    pred: list[dict],
    ref: list[dict],
    tol_s: float = 0.05,
) -> dict:
    """Greedy 1:1 match on (pitch, onset) within tol_s."""
    ref_used = [False] * len(ref)
    tp = 0
    for p in sorted(pred, key=lambda n: n["onset_s"]):
        best_j = -1
        best_dt = None
        for j, r in enumerate(ref):
            if ref_used[j]:
                continue
            if int(r["pitch"]) != int(p["pitch"]):
                continue
            dt = abs(float(r["onset_s"]) - float(p["onset_s"]))
            if dt <= tol_s and (best_dt is None or dt < best_dt):
                best_dt = dt
                best_j = j
        if best_j >= 0:
            ref_used[best_j] = True
            tp += 1
    fp = len(pred) - tp
    fn = len(ref) - tp
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(ref) if ref else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {
        "tol_s": tol_s,
        "n_pred": len(pred),
        "n_ref": len(ref),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }


def window_notes(notes: list[dict], start_s: float, end_s: float | None) -> list[dict]:
    out = []
    for n in notes:
        onset = float(n["onset_s"])
        if onset < start_s:
            continue
        if end_s is not None and onset >= float(end_s):
            continue
        out.append(n)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="clean_amt evaluate vs GT MIDI")
    ap.add_argument("--run-dir", type=Path, required=True, help="out/<run_id>/")
    ap.add_argument("--gt-midi", type=Path, default=None, help="override GT path")
    ap.add_argument("--tol-ms", type=float, default=50.0)
    ap.add_argument("--start-s", type=float, default=None)
    ap.add_argument("--end-s", type=float, default=None)
    args = ap.parse_args(argv)

    run_dir = args.run_dir.resolve()
    notes_path = run_dir / "notes.json"
    man_path = run_dir / "manifest.json"
    if not notes_path.is_file():
        raise SystemExit(f"missing {notes_path}")

    man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.is_file() else {}
    gt = args.gt_midi
    if gt is None:
        rel = (man.get("input") or {}).get("gt_midi")
        if not rel:
            raise SystemExit("no --gt-midi and manifest.input.gt_midi empty")
        gt = Path(rel)
        if not gt.is_absolute():
            # try repo root = parents[4] from scripts, or cwd
            repo = Path(__file__).resolve().parents[5]
            cand = repo / rel
            gt = cand if cand.is_file() else Path.cwd() / rel
    gt = gt.resolve()
    if not gt.is_file():
        raise SystemExit(f"GT not found: {gt}")

    pred = load_notes_json(notes_path)
    ref = midi_to_notes(gt)

    start_s = args.start_s
    end_s = args.end_s
    if start_s is None:
        start_s = float((man.get("input") or {}).get("start_s") or 0.0)
    if end_s is None:
        end_s = (man.get("input") or {}).get("end_s")

    pred_w = window_notes(pred, start_s, end_s)
    ref_w = window_notes(ref, start_s, end_s)
    # shift pred if transcription was on a cropped file starting at 0
    # assume notes are in absolute time of the source file window origin = start_s
    # If model emits times relative to crop, add start_s:
    # (documented: backends should emit absolute times in source file seconds)

    metrics = match_onset_f1(pred_w, ref_w, tol_s=args.tol_ms / 1000.0)
    metrics["gt_midi"] = str(gt)
    metrics["start_s"] = start_s
    metrics["end_s"] = end_s

    out = run_dir / "metrics.json"
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if man_path.is_file():
        man.setdefault("outputs", {})["metrics_json"] = "metrics.json"
        man_path.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
