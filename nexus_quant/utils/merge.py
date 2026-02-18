from __future__ import annotations

from typing import Any, Dict


def deep_merge(base: Any, override: Any) -> Any:
    """
    Recursively merge override into base.

    Rules:
    - dict + dict => merge per key
    - otherwise => override wins (including lists/scalars)
    """
    if isinstance(base, dict) and isinstance(override, dict):
        out: Dict[str, Any] = dict(base)
        for k, v in override.items():
            if k in out:
                out[k] = deep_merge(out[k], v)
            else:
                out[k] = v
        return out
    return override

