from __future__ import annotations

import argparse
from pathlib import Path

from .learning.critic import critique_recent


def critique_main(*, config_path: Path, artifacts_dir: Path, tail_events: int) -> int:
    critique_recent(config_path=config_path, artifacts_dir=artifacts_dir, tail_events=int(tail_events))
    return 0


def critique_cli(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="nexus_quant critique", add_help=True)
    p.add_argument("--config", required=True, help="Path to JSON config")
    p.add_argument("--artifacts", default="artifacts", help="Artifacts dir (default: artifacts)")
    p.add_argument("--tail-events", type=int, default=200, help="How many ledger events to scan (default: 200)")
    args = p.parse_args(argv)
    return critique_main(config_path=Path(args.config), artifacts_dir=Path(args.artifacts), tail_events=int(args.tail_events))

