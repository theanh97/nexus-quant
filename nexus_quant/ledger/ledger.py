from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class LedgerEvent:
    ts: str
    kind: str
    run_id: str
    run_name: str
    config_sha: str
    code_fingerprint: str
    data_fingerprint: str
    payload: Dict[str, Any]


def append_ledger_event(path: Path, event: LedgerEvent) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(event), sort_keys=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.write("\n")

