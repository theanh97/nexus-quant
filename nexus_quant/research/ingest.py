from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..memory.store import MemoryStore
from ..utils.hashing import sha256_bytes


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_text_suffix(path: Path) -> bool:
    return path.suffix.lower() in {".md", ".txt", ".json", ".py", ".html", ".css", ".js", ".yaml", ".yml"}


def _read_text_limited(path: Path, limit_bytes: int = 200_000) -> str:
    data = path.read_bytes()
    if len(data) > limit_bytes:
        data = data[:limit_bytes]
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:
        return data.decode(errors="replace")


def _parse_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    """
    Super-minimal YAML frontmatter parser:
    - Only supports `---` ... `---` at the top and `key: value` pairs.
    Returns (meta, body_without_frontmatter).
    """
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text
    block = text[4:end].strip()
    body = text[end + 5 :]
    meta: Dict[str, Any] = {}
    for ln in block.splitlines():
        if ":" not in ln:
            continue
        k, v = ln.split(":", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        meta[k] = v
    return meta, body


def ingest_path_to_memory(
    *,
    path: Path,
    memory_db: Path,
    kind: str = "source",
    tags: Optional[List[str]] = None,
    move_to: Optional[Path] = None,
    recursive: bool = True,
) -> Dict[str, Any]:
    """
    Verified research ingestion (local-first):
    - hashes every file
    - stores provenance + (optional) text content into long-term memory
    - optionally moves files to a library folder for dedup/curation
    """
    root = Path(path).expanduser()
    if not root.exists():
        raise FileNotFoundError(str(root))

    ms = MemoryStore(memory_db)
    try:
        files: List[Path] = []
        if root.is_file():
            files = [root]
        else:
            it = root.rglob("*") if recursive else root.glob("*")
            for p in it:
                if p.is_file():
                    files.append(p)

        ingested = 0
        skipped = 0
        moved = 0
        items = []

        for p in sorted(files):
            # Skip hidden files and very large binaries by default.
            if any(part.startswith(".") for part in p.parts):
                continue
            b = p.read_bytes()
            sha = sha256_bytes(b)
            size = len(b)

            content = ""
            fm = {}
            if _is_text_suffix(p):
                text = _read_text_limited(p)
                fm, body = _parse_frontmatter(text)
                content = body.strip()
            else:
                content = f"[BINARY] file={p.name} sha256={sha} size_bytes={size}"

            meta = {
                "source_path": str(p),
                "sha256": sha,
                "size_bytes": size,
                "ingested_at": _utc_iso(),
                "frontmatter": fm,
            }

            # Naive de-dup: check recent sources for same sha (fast, local-only).
            # This avoids adding repeated copies in day-to-day workflows.
            recent = ms.search(query=sha, kind=kind, limit=1)
            if recent:
                skipped += 1
                continue

            item_id = ms.add(
                created_at=_utc_iso(),
                kind=str(kind),
                tags=tags or ["research"],
                content=content,
                meta=meta,
                run_id=None,
            )
            ingested += 1
            items.append({"id": item_id, "path": str(p), "sha256": sha, "size_bytes": size})

            if move_to is not None and root.is_dir():
                lib = Path(move_to).expanduser()
                lib.mkdir(parents=True, exist_ok=True)
                target = lib / f"{sha[:12]}__{p.name}"
                try:
                    if target.exists():
                        # Already moved; keep original.
                        pass
                    else:
                        p.rename(target)
                        moved += 1
                except Exception:
                    # Ignore move errors; ingestion already succeeded.
                    pass

        return {"ok": True, "ingested": ingested, "skipped": skipped, "moved": moved, "items": items}
    finally:
        ms.close()

