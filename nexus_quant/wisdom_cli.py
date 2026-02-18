from __future__ import annotations

import argparse
from pathlib import Path

from .wisdom.curate import curate_wisdom


def wisdom_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="nexus_quant wisdom", add_help=True)
    p.add_argument("--artifacts", default="artifacts", help="Artifacts dir (default: artifacts)")
    p.add_argument("--tail-events", type=int, default=200, help="How many ledger events to summarize (default: 200)")
    args = p.parse_args(argv)

    curate_wisdom(artifacts_dir=Path(args.artifacts), tail_events=int(args.tail_events))
    return 0

