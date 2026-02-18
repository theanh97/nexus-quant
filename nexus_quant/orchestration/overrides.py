from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def overrides_path(artifacts_dir: Path) -> Path:
    return Path(artifacts_dir) / "state" / "overrides.json"


def load_overrides(artifacts_dir: Path) -> Dict[str, Any]:
    """
    Load runtime overrides for "agentic" control without editing the base config file.

    File format (suggested):
    {
      "ts": "...",
      "note": "written by reflection",
      "orion": {"trials": 60},
      "config_overrides": {"self_learn": {"prior_exploit_prob": 0.3}}
    }
    """
    env_flag = os.environ.get("NX_DISABLE_OVERRIDES")
    if env_flag is not None and env_flag.strip().lower() in {"1", "true", "yes", "on"}:
        return {"orion": {}, "config_overrides": {}, "disabled": True}

    p = overrides_path(Path(artifacts_dir))
    if not p.exists():
        return {"orion": {}, "config_overrides": {}}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return {"orion": {}, "config_overrides": {}}
        return {
            "ts": obj.get("ts"),
            "note": obj.get("note"),
            "orion": obj.get("orion") or {},
            "config_overrides": obj.get("config_overrides") or {},
        }
    except Exception:
        return {"orion": {}, "config_overrides": {}, "error": "invalid_json"}


def save_overrides(artifacts_dir: Path, *, orion: Dict[str, Any], config_overrides: Dict[str, Any], note: str = "") -> Path:
    p = overrides_path(Path(artifacts_dir))
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ts": _utc_iso(), "note": str(note or ""), "orion": orion or {}, "config_overrides": config_overrides or {}}
    p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return p

