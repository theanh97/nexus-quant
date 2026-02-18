from __future__ import annotations

from datetime import datetime, timezone


def parse_iso_utc(ts: str) -> int:
    """
    Parse ISO timestamps and return epoch seconds (UTC).

    Accepts:
    - "2026-01-01T00:00:00Z"
    - "2026-01-01T00:00:00+00:00"
    """
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.astimezone(timezone.utc).timestamp())


def iso_utc(epoch_seconds: int) -> str:
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat()

