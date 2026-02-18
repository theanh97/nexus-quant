from __future__ import annotations

import argparse
from pathlib import Path

from .research.ingest import ingest_path_to_memory


def research_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="nexus_quant research", add_help=True)
    sub = p.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="Ingest local research/source files into long-term memory (with hashes)")
    ing.add_argument("--path", required=True, help="File or folder to ingest")
    ing.add_argument("--artifacts", default="artifacts", help="Artifacts dir (default: artifacts)")
    ing.add_argument("--kind", default="source", help="Memory kind (default: source)")
    ing.add_argument("--tags", default="research", help="Comma-separated tags (default: research)")
    ing.add_argument("--move-to", default="", help="Optional library dir to move ingested files into")
    ing.add_argument("--no-recursive", action="store_true", help="Do not scan folders recursively")

    args = p.parse_args(argv)
    if args.cmd == "ingest":
        artifacts = Path(args.artifacts)
        memory_db = artifacts / "memory" / "memory.db"
        move_to = Path(args.move_to).expanduser() if str(args.move_to or "").strip() else None
        tags = [t.strip() for t in str(args.tags or "").split(",") if t.strip()]
        ingest_path_to_memory(
            path=Path(args.path),
            memory_db=memory_db,
            kind=str(args.kind),
            tags=tags,
            move_to=move_to,
            recursive=not bool(args.no_recursive),
        )
        return 0
    raise SystemExit(f"Unknown research command: {args.cmd}")

