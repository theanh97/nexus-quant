from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ..memory.store import MemoryStore
from ..run import run_one, improve_one
from ..wisdom.curate import curate_wisdom
from ..learning.reflection import reflect_and_update
from ..learning.critic import critique_recent
from .overrides import load_overrides
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
        self.task_store.create(kind="research_ingest", payload={"artifacts": str(self.cfg.artifacts_dir)})
        self.task_store.create(
            kind="improve",
            payload={"config": str(self.cfg.config_path), "out": str(self.cfg.artifacts_dir), "trials": int(self.cfg.trials)},
        )
        self.task_store.create(kind="wisdom", payload={"artifacts": str(self.cfg.artifacts_dir)})
        self.task_store.create(kind="reflect", payload={"config": str(self.cfg.config_path), "artifacts": str(self.cfg.artifacts_dir)})
        self.task_store.create(kind="critique", payload={"config": str(self.cfg.config_path), "artifacts": str(self.cfg.artifacts_dir)})

    def run_once(self) -> Dict[str, Any]:
        task = self.task_store.claim_next()
        if not task:
            return {"ok": True, "message": "no pending tasks"}
        try:
            ov = load_overrides(self.cfg.artifacts_dir)
            cfg_override = ov.get("config_overrides") if isinstance(ov, dict) else None
            orion_ov = (ov.get("orion") or {}) if isinstance(ov, dict) else {}

            if task.kind == "run":
                run_one(Path(task.payload["config"]), out_dir=Path(task.payload["out"]), cfg_override=cfg_override)
                self.task_store.mark_done(task.id, {"ok": True})
                return {"ok": True, "task": task.kind}
            if task.kind == "research_ingest":
                out = self._research_ingest(Path(task.payload["artifacts"]))
                self.task_store.mark_done(task.id, out)
                return {"ok": True, "task": task.kind, "research": out}
            if task.kind == "improve":
                trials = int(orion_ov.get("trials") or task.payload.get("trials") or 30)
                improve_one(Path(task.payload["config"]), out_dir=Path(task.payload["out"]), trials=trials, cfg_override=cfg_override)
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
            if task.kind == "reflect":
                out = reflect_and_update(
                    config_path=Path(task.payload["config"]),
                    artifacts_dir=Path(task.payload["artifacts"]),
                    tail_events=200,
                )
                self.task_store.mark_done(task.id, out)
                return {"ok": True, "task": task.kind, "reflect": out}
            if task.kind == "critique":
                out = critique_recent(
                    config_path=Path(task.payload["config"]),
                    artifacts_dir=Path(task.payload["artifacts"]),
                    tail_events=200,
                )
                self.task_store.mark_done(task.id, out)
                # Enqueue next experiments suggested by critic, then handoff.
                enq = self._enqueue_experiments(Path(task.payload["config"]), Path(task.payload["artifacts"]), out.get("next_experiments") or [])
                self.task_store.create(kind="handoff", payload={"artifacts": str(task.payload["artifacts"])})
                return {"ok": True, "task": task.kind, "critique": out, "enqueued_experiments": enq}
            if task.kind == "experiment":
                run_id = run_one(
                    Path(task.payload["config"]),
                    out_dir=Path(task.payload["out"]),
                    cfg_override=task.payload.get("config_overrides") or {},
                )
                self.task_store.mark_done(task.id, {"ok": True, "run_id": run_id})
                return {"ok": True, "task": task.kind, "run_id": run_id, "why": task.payload.get("why"), "experiment_id": task.payload.get("experiment_id")}
            raise ValueError(f"Unknown task kind: {task.kind}")
        except Exception as e:
            self.task_store.mark_failed(task.id, str(e))
            return {"ok": False, "task": task.kind, "error": str(e)}

    def _enqueue_experiments(self, config_path: Path, artifacts_dir: Path, next_experiments: list[dict]) -> Dict[str, Any]:
        max_n = 3
        ov = load_overrides(artifacts_dir)
        if isinstance(ov, dict):
            max_n = int((ov.get("orion") or {}).get("max_experiments_per_critique") or max_n)
        max_n = max(0, min(max_n, 20))

        recent = self.task_store.recent(limit=400)
        existing_ids = {t.payload.get("experiment_id") for t in recent if t.kind == "experiment"}

        created = 0
        skipped = 0
        ids = []
        for it in (next_experiments or [])[:max_n]:
            overrides = it.get("config_overrides") or {}
            why = str(it.get("why") or "")
            raw = json.dumps({"why": why, "overrides": overrides}, sort_keys=True).encode("utf-8")
            exp_id = hashlib.sha256(raw).hexdigest()[:12]
            if exp_id in existing_ids:
                skipped += 1
                continue
            self.task_store.create(
                kind="experiment",
                payload={
                    "experiment_id": exp_id,
                    "why": why,
                    "config": str(config_path),
                    "out": str(artifacts_dir),
                    "config_overrides": overrides,
                },
            )
            created += 1
            ids.append(exp_id)
        return {"created": created, "skipped": skipped, "ids": ids, "max_per_critique": max_n}

    def _research_ingest(self, artifacts_dir: Path) -> Dict[str, Any]:
        """
        Local-first "learn new things":
        - ingest files dropped into research/inbox/ into memory with sha256 provenance
        - does NOT move/delete files by default (conservative)
        """
        try:
            from ..research.ingest import ingest_path_to_memory
        except Exception:
            return {"ok": False, "error": "research_ingest_module_missing"}

        repo_root = Path(__file__).resolve().parents[2]
        inbox = repo_root / "research" / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)

        memory_db = artifacts_dir / "memory" / "memory.db"
        out = ingest_path_to_memory(path=inbox, memory_db=memory_db, kind="source", tags=["research"], move_to=None, recursive=True)
        return out

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
                ov = payload.get("config_overrides") or {}
                if ov:
                    md.append(f"  - config_overrides: {json.dumps(ov, sort_keys=True)}")
            if kind == "self_learn":
                md.append(f"  - accepted: `{payload.get('accepted')}` objective=`{payload.get('objective')}`")
                if payload.get("ablation_paths"):
                    md.append(f"  - ablation: `{payload['ablation_paths'].get('md')}`")
            if kind == "reflection":
                ref = payload.get("reflection") or {}
                sl = ref.get("self_learn") or {}
                cur = sl.get("current") or {}
                prop = sl.get("proposed") or {}
                md.append(f"  - self_learn.accept_rate: `{sl.get('accept_rate')}` no_accept: `{sl.get('consecutive_no_accept')}`")
                md.append(f"  - policy: trials `{cur.get('orion_trials')}` -> `{prop.get('orion_trials')}`, exploit_prob `{cur.get('prior_exploit_prob')}` -> `{prop.get('prior_exploit_prob')}`")
            if kind == "critique":
                flags = payload.get("flags") or []
                nex = payload.get("next_experiments") or []
                md.append(f"  - flags: `{len(flags)}` next_experiments: `{len(nex)}`")
                paths = payload.get("paths") or {}
                if paths.get("md"):
                    md.append(f"  - critique: `{paths.get('md')}`")
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
