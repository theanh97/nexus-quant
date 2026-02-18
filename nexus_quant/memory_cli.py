from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .memory.store import MemoryStore


def memory_main(argv: List[str]) -> int:
    artifacts = _extract_global_flag(argv, "--artifacts", default="artifacts")

    p = argparse.ArgumentParser(prog="nexus_quant memory", add_help=True)
    sub = p.add_subparsers(dest="subcmd", required=True)

    add_p = sub.add_parser("add", help="Add a memory item")
    add_p.add_argument("--kind", required=True, help="kind: feedback|decision|note|source|risk|todo")
    add_p.add_argument("--tags", default="", help="comma-separated tags")
    add_p.add_argument("--run-id", default=None, help="optional run_id to link evidence")
    add_p.add_argument("--meta", default="{}", help="JSON meta (default: {})")
    add_p.add_argument("--content", default=None, help="content text (if omitted, read stdin)")

    rec_p = sub.add_parser("recent", help="Show recent memory")
    rec_p.add_argument("--kind", default=None, help="filter by kind")
    rec_p.add_argument("--limit", type=int, default=20, help="limit")

    s_p = sub.add_parser("search", help="Search memory")
    s_p.add_argument("--query", required=True, help="search query")
    s_p.add_argument("--kind", default=None, help="filter by kind")
    s_p.add_argument("--limit", type=int, default=20, help="limit")

    st_p = sub.add_parser("stats", help="Memory stats")

    args = p.parse_args(argv)
    artifacts = Path(artifacts)
    db_path = artifacts / "memory" / "memory.db"

    store = MemoryStore(db_path)
    try:
        if args.subcmd == "add":
            content = args.content
            if content is None:
                content = sys.stdin.read()
            content = (content or "").strip()
            if not content:
                raise SystemExit("content is empty")

            tags = [t.strip() for t in (args.tags or "").split(",") if t.strip()]
            meta = _parse_json(args.meta)
            item_id = store.add(
                created_at=_utc_iso(),
                kind=str(args.kind),
                tags=tags,
                content=content,
                meta=meta,
                run_id=str(args.run_id) if args.run_id else None,
            )
            print(json.dumps({"ok": True, "id": item_id, "db": str(db_path)}, indent=2, sort_keys=True))
            return 0

        if args.subcmd == "recent":
            items = store.recent(kind=args.kind, limit=int(args.limit))
            _print_items(items)
            return 0

        if args.subcmd == "search":
            items = store.search(query=str(args.query), kind=args.kind, limit=int(args.limit))
            _print_items(items)
            return 0

        if args.subcmd == "stats":
            print(json.dumps(store.stats(), indent=2, sort_keys=True))
            return 0

        raise SystemExit(f"Unknown memory subcmd: {args.subcmd}")
    finally:
        store.close()


def _parse_json(text: str) -> Dict[str, Any]:
    try:
        v = json.loads(text or "{}")
        if isinstance(v, dict):
            return v
    except Exception:
        pass
    return {}


def _extract_global_flag(argv: List[str], flag: str, default: str) -> str:
    """
    Allow a global flag to appear anywhere in argv (argparse subcommands normally require it before the subcmd).
    """
    value = default
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == flag:
            if i + 1 < len(argv):
                value = argv[i + 1]
                del argv[i : i + 2]
                continue
            del argv[i]
            continue
        if a.startswith(flag + "="):
            value = a.split("=", 1)[1]
            del argv[i]
            continue
        i += 1
    return value


def _print_items(items: List[Any]) -> None:
    for it in items:
        tags = ",".join(it.tags) if it.tags else ""
        run = it.run_id or ""
        print(f"- id={it.id}  ts={it.created_at}  kind={it.kind}  tags={tags}  run_id={run}")
        print(f"  {it.content}")


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
