"""LPC-order disagreement peaks → 3kHz clicks on BS piano.

Clusters o4/o6/o8/o12/o24/o36 SuperFlux+peaks_adaptive times at ±30ms.
Keeps every cluster where NOT all six orders are present (excludes all-six ≈325).
Overlay style matches lpc_*_sf_adaptive_on_piano packs.
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

from lpc_order_presence_clicks import main_for  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--determinism-check", action="store_true")
    args = parser.parse_args()
    main_for("disagreement", determinism=args.determinism_check)


if __name__ == "__main__":
    main()
