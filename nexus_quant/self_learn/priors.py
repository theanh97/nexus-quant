from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_priors(memory_dir: Path) -> Dict[str, Any]:
    p = memory_dir / "priors.json"
    if not p.exists():
        return {"version": 1, "updated_at": None, "strategies": {}}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return {"version": 1, "updated_at": None, "strategies": {}}
        obj.setdefault("version", 1)
        obj.setdefault("strategies", {})
        return obj
    except Exception:
        return {"version": 1, "updated_at": None, "strategies": {}}


def save_priors(memory_dir: Path, priors: Dict[str, Any]) -> None:
    p = memory_dir / "priors.json"
    out = dict(priors)
    out["version"] = int(out.get("version") or 1)
    out["updated_at"] = _utc_iso()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")


def update_priors_on_accept(
    *,
    memory_dir: Path,
    strategy_name: str,
    params: Dict[str, Any],
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Update discrete param priors when a candidate is accepted.

    This is intentionally simple:
    - counts per (strategy, param, value)
    - remember last accepted payload for "exploit" sampling
    """
    priors = load_priors(memory_dir)
    strategies = priors.setdefault("strategies", {})
    s = strategies.setdefault(strategy_name, {})

    s["accepted"] = int(s.get("accepted") or 0) + 1
    pc = s.setdefault("params_counts", {})
    for k, v in (params or {}).items():
        vc = pc.setdefault(str(k), {})
        key = json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else str(v)
        vc[key] = int(vc.get(key) or 0) + 1

    s["last_accepted"] = {"params": params, "meta": meta or {}, "ts": _utc_iso()}
    save_priors(memory_dir, priors)
    return priors

