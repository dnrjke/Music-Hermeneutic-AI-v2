"""Locked tilt/perc all-series agreement peaks → 3kHz clicks on BS piano.

Clusters raw/tilt_high/k_env/k_env_adaptive (when peak_times exist) at ±30ms.
Keeps every cluster where all included series are present.
Overlay style matches lpc_order_agreement_on_piano packs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tilt_material_presence_clicks import main_for  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--determinism-check", action="store_true")
    args = parser.parse_args()
    main_for("agreement", determinism=args.determinism_check)


if __name__ == "__main__":
    main()
