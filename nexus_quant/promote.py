from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

from .memory.store import MemoryStore


def promote_best_params(*, config_path: Path, best_params_path: Path, artifacts_dir: Path, apply: bool) -> Dict[str, Any]:
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")
    if not best_params_path.exists():
        raise SystemExit(f"Best params not found: {best_params_path}")

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    best = json.loads(best_params_path.read_text(encoding="utf-8"))

    best_params = best.get("params") or {}
    if not isinstance(best_params, dict) or not best_params:
        raise SystemExit("best_params.json has no params to promote")

    before = cfg.get("strategy", {}).get("params", {}) or {}
    after = dict(before)
    after.update(best_params)

    changed = {k: {"before": before.get(k), "after": after.get(k)} for k in sorted(after.keys()) if before.get(k) != after.get(k)}

    out = {
        "ok": True,
        "apply": bool(apply),
        "config_path": str(config_path),
        "best_params_path": str(best_params_path),
        "changed": changed,
        "run_id": best.get("run_id"),
        "run_group_id": best.get("run_group_id"),
        "accepted_at": best.get("accepted_at"),
    }

    if not apply:
        return out

    # Backup current config
    backups = artifacts_dir / "config_backups"
    backups.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backups / f"{config_path.stem}.backup.{ts}.json"
    backup_path.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")

    cfg.setdefault("strategy", {})
    cfg["strategy"].setdefault("params", {})
    cfg["strategy"]["params"] = after

    config_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Record decision to memory.
    mem = MemoryStore(artifacts_dir / "memory" / "memory.db")
    try:
        mem.add(
            created_at=datetime.now(timezone.utc).isoformat(),
            kind="decision",
            tags=["promote", "self_learning"],
            content=f"Promoted best params into config: {config_path.name}",
            meta={
                "config_path": str(config_path),
                "backup_path": str(backup_path),
                "best_params_path": str(best_params_path),
                "changed": changed,
                "run_id": best.get("run_id"),
                "run_group_id": best.get("run_group_id"),
            },
            run_id=str(best.get("run_id")) if best.get("run_id") else None,
        )
    finally:
        mem.close()

    out["backup_path"] = str(backup_path)
    return out

