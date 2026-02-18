from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ..memory.store import MemoryStore
from ..run import run_one, improve_one
from ..wisdom.curate import curate_wisdom
from .tasks import TaskStore


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class OrionConfig:
    artifacts_dir: Path
    config_path: Path
    trials: int = 30


class Orion:
    """
    Minimal Orion commander:
    - creates tasks for: run -> improve -> handoff
    - executes tasks sequentially (v1)
    - writes to memory + produces a handoff file for human review

    This is the "agentic control plane" seed. We will expand to multi-agent + tool routing next.
    """

    def __init__(self, cfg: OrionConfig) -> None:
        self.cfg = cfg
        self.task_store = TaskStore(cfg.artifacts_dir / "state" / "tasks.db")
        self.memory = MemoryStore(cfg.artifacts_dir / "memory" / "memory.db")

    def close(self) -> None:
        self.task_store.close()
        self.memory.close()

    def bootstrap(self) -> None:
        self.task_store.create(kind="run", payload={"config": str(self.cfg.config_path), "out": str(self.cfg.artifacts_dir)})
        self.task_store.create(
            kind="improve",
            payload={"config": str(self.cfg.config_path), "out": str(self.cfg.artifacts_dir), "trials": int(self.cfg.trials)},
        )
        self.task_store.create(kind="wisdom", payload={"artifacts": str(self.cfg.artifacts_dir)})
        self.task_store.create(kind="handoff", payload={"artifacts": str(self.cfg.artifacts_dir)})

    def run_once(self) -> Dict[str, Any]:
        task = self.task_store.claim_next()
        if not task:
            return {"ok": True, "message": "no pending tasks"}
        try:
            if task.kind == "run":
                run_one(Path(task.payload["config"]), out_dir=Path(task.payload["out"]))
                self.task_store.mark_done(task.id, {"ok": True})
                return {"ok": True, "task": task.kind}
            if task.kind == "improve":
                improve_one(Path(task.payload["config"]), out_dir=Path(task.payload["out"]), trials=int(task.payload.get("trials") or 30))
                self.task_store.mark_done(task.id, {"ok": True})
                return {"ok": True, "task": task.kind}
            if task.kind == "handoff":
                out = self._write_handoff(Path(task.payload["artifacts"]))
                self.task_store.mark_done(task.id, out)
                return {"ok": True, "task": task.kind, "handoff": out}
            if task.kind == "wisdom":
                out = curate_wisdom(artifacts_dir=Path(task.payload["artifacts"]))
                self.task_store.mark_done(task.id, out)
                return {"ok": True, "task": task.kind, "wisdom": out}
            raise ValueError(f"Unknown task kind: {task.kind}")
        except Exception as e:
            self.task_store.mark_failed(task.id, str(e))
            return {"ok": False, "task": task.kind, "error": str(e)}

    def _write_handoff(self, artifacts_dir: Path) -> Dict[str, Any]:
        ledger_path = artifacts_dir / "ledger" / "ledger.jsonl"
        if not ledger_path.exists():
            raise RuntimeError("No ledger found; run at least one cycle first.")
        lines = ledger_path.read_text(encoding="utf-8").splitlines()
        events = []
        for line in lines[-10:]:
            try:
                events.append(json.loads(line))
            except Exception:
                continue

        handoff_dir = artifacts_dir / "handoff"
        handoff_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = handoff_dir / f"handoff.{ts}.md"

        md = []
        md.append("# NEXUS Orion Handoff")
        md.append("")
        md.append(f"- generated_at: `{_utc_iso()}`")
        md.append(f"- artifacts: `{artifacts_dir}`")
        md.append("")
        md.append("## Recent ledger events")
        for e in events:
            kind = e.get("kind")
            md.append(f"- {e.get('ts')}  `{kind}`  run=`{e.get('run_name')}`  id=`{e.get('run_id')}`")
            payload = e.get("payload") or {}
            if kind == "run":
                v = payload.get("verdict") or {}
                md.append(f"  - verdict.pass: `{v.get('pass')}` reasons={json.dumps(v.get('reasons') or [])}")
                md.append(f"  - metrics: {json.dumps(payload.get('metrics') or {}, sort_keys=True)}")
            if kind == "self_learn":
                md.append(f"  - accepted: `{payload.get('accepted')}` objective=`{payload.get('objective')}`")
                if payload.get("ablation_paths"):
                    md.append(f"  - ablation: `{payload['ablation_paths'].get('md')}`")
        md.append("")
        md.append("## Recent user feedback (memory)")
        feedback = self.memory.recent(kind="feedback", limit=5)
        if not feedback:
            md.append("- (none)")
        else:
            for it in feedback:
                tags = ",".join(it.tags) if it.tags else ""
                md.append(f"- {it.created_at}  tags=`{tags}`  {it.content}")
        md.append("")
        md.append("## Questions for human")
        md.append("- Do we promote the accepted params to the main config?")
        md.append("- Any constraints on venue (Binance/Bybit), fee tier, or execution style (maker/taker)?")
        md.append("- Any risk caps (MDD, exposure, turnover) to tighten?")
        md.append("")

        out_path.write_text("\n".join(md), encoding="utf-8")

        # Also store a memory record.
        self.memory.add(
            created_at=_utc_iso(),
            kind="handoff",
            tags=["orion"],
            content=f"Handoff generated: {out_path}",
            meta={"handoff_path": str(out_path)},
            run_id=None,
        )

        return {"handoff_path": str(out_path)}
