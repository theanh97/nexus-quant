from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def print_status(artifacts_dir: Path, tail: int = 6) -> None:
    ledger_path = artifacts_dir / "ledger" / "ledger.jsonl"
    memory_best = artifacts_dir / "memory" / "best_params.json"

    print(f"Artifacts: {artifacts_dir}")
    print(f"Ledger: {ledger_path}")
    if ledger_path.exists():
        events = _tail_jsonl(ledger_path, n=max(1, int(tail)))
        print("")
        print("Recent events:")
        for e in events:
            kind = e.get("kind")
            ts = e.get("ts")
            run_name = e.get("run_name")
            run_id = e.get("run_id")
            payload = e.get("payload") or {}
            if kind == "self_learn":
                accepted = bool(payload.get("accepted"))
                print(f"- {ts}  {kind}  run={run_name}  accepted={accepted}  id={run_id}")
            else:
                dq_ok = payload.get("data_quality_ok")
                print(f"- {ts}  {kind}  run={run_name}  dq_ok={dq_ok}  id={run_id}")
    else:
        print("")
        print("Recent events: (no ledger yet)")

    print("")
    print("Best params:")
    if memory_best.exists():
        d = json.loads(memory_best.read_text(encoding="utf-8"))
        print(f"- strategy: {d.get('strategy')}")
        print(f"- accepted_at: {d.get('accepted_at')}")
        print(f"- params: {json.dumps(d.get('params') or {}, sort_keys=True)}")
    else:
        print("- (none)")


def _tail_jsonl(path: Path, n: int) -> List[Dict[str, Any]]:
    # Small files: just read all. (Optimize later if needed.)
    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines[-n:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out

