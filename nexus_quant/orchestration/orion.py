from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..memory.store import MemoryStore
from ..run import run_one, improve_one
from ..wisdom.curate import curate_wisdom
from ..learning.reflection import reflect_and_update
from ..learning.critic import critique_recent
from ..learning.operational import OperationalLearner
from .overrides import load_overrides
from .tasks import TaskStore
from .opus_rebuttal import OpusRebuttalConfig, persist_opus_rebuttal, run_opus_rebuttal
from ..agents.atlas import AtlasAgent
from ..agents.cipher import CipherAgent
from ..agents.echo import EchoAgent
from ..agents.flux import FluxAgent
from ..agents.base import AgentContext


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
        # Initialize vector store for semantic search
        try:
            from ..memory.vector_store import NexusVectorStore
            self.vector_store = NexusVectorStore(cfg.artifacts_dir)
        except Exception:
            self.vector_store = None
        # Memory store with vector store attached for auto-indexing
        self.memory = MemoryStore(cfg.artifacts_dir / "memory" / "memory.db", vector_store=self.vector_store)
        self.learner = OperationalLearner(cfg.artifacts_dir)

    def close(self) -> None:
        self.task_store.close()
        self.memory.close()
        self.learner.close()
        if self.vector_store:
            self.vector_store.close()

    def bootstrap(self, *, include_improve: bool = True) -> None:
        self.task_store.create(kind="run", payload={"config": str(self.cfg.config_path), "out": str(self.cfg.artifacts_dir)})
        self.task_store.create(kind="research_ingest", payload={"artifacts": str(self.cfg.artifacts_dir)})
        if include_improve:
            self.task_store.create(
                kind="improve",
                payload={"config": str(self.cfg.config_path), "out": str(self.cfg.artifacts_dir), "trials": int(self.cfg.trials)},
            )
        self.task_store.create(kind="wisdom", payload={"artifacts": str(self.cfg.artifacts_dir)})
        self.task_store.create(kind="reflect", payload={"config": str(self.cfg.config_path), "artifacts": str(self.cfg.artifacts_dir)})
        self.task_store.create(kind="critique", payload={"config": str(self.cfg.config_path), "artifacts": str(self.cfg.artifacts_dir)})

    def enqueue_policy_actions(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Enqueue policy-triggered tasks with dedup against recent pending/running tasks.
        """
        recent = self.task_store.recent(limit=400)
        existing = {
            json.dumps({"kind": t.kind, "payload": t.payload}, sort_keys=True)
            for t in recent
            if t.status in {"pending", "running"}
        }
        created = 0
        skipped = 0
        for action in (actions or []):
            kind = str(action.get("kind") or "")
            payload = dict(action.get("payload") or {})
            if not kind:
                skipped += 1
                continue

            if kind == "reflect":
                payload = {"config": str(self.cfg.config_path), "artifacts": str(self.cfg.artifacts_dir), **payload}
            elif kind == "critique":
                payload = {"config": str(self.cfg.config_path), "artifacts": str(self.cfg.artifacts_dir), **payload}
            elif kind == "run":
                payload = {"config": str(self.cfg.config_path), "out": str(self.cfg.artifacts_dir), **payload}
            elif kind == "improve":
                payload = {"config": str(self.cfg.config_path), "out": str(self.cfg.artifacts_dir), "trials": int(self.cfg.trials), **payload}
            elif kind == "wisdom":
                payload = {"artifacts": str(self.cfg.artifacts_dir), **payload}
            elif kind == "research_ingest":
                payload = {"artifacts": str(self.cfg.artifacts_dir), **payload}
            elif kind == "policy_review":
                payload = {"artifacts": str(self.cfg.artifacts_dir), **payload}
            elif kind == "rules_reminder":
                payload = {"artifacts": str(self.cfg.artifacts_dir), **payload}
            elif kind == "agent_run":
                payload = {"config": str(self.cfg.config_path), "out": str(self.cfg.artifacts_dir), **payload}
            else:
                skipped += 1
                continue

            sig = json.dumps({"kind": kind, "payload": payload}, sort_keys=True)
            if sig in existing:
                skipped += 1
                continue
            self.task_store.create(kind=kind, payload=payload)
            existing.add(sig)
            created += 1
        return {"created": created, "skipped": skipped}

    def run_once(self) -> Dict[str, Any]:
        task = self.task_store.claim_next()
        if not task:
            return {"ok": True, "message": "no pending tasks"}

        # ── PREFLIGHT: Check operational lessons before executing ──
        preflight_warnings = self.learner.preflight(
            task_kind=task.kind,
            context=task.payload,
        )
        # Log critical preflight warnings (but don't block — lessons are guidance)
        for w in preflight_warnings:
            if w.severity == "critical":
                self.memory.add(
                    created_at=_utc_iso(),
                    kind="preflight_warning",
                    tags=["operational", w.category, "critical"],
                    content=f"PREFLIGHT [{w.category}]: {w.correction}",
                    meta={"lesson_id": w.lesson_id, "task_kind": task.kind, "pattern": w.pattern_matched},
                    run_id=None,
                )

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
            if task.kind == "agent_run":
                out = self.run_agents()
                self.task_store.mark_done(task.id, {"ok": True, "agents": list(out.keys())})
                return {"ok": True, "task": task.kind, "agents": out}
            if task.kind == "rules_reminder":
                out = self._write_rules_reminder(Path(task.payload["artifacts"]), task.payload)
                self.task_store.mark_done(task.id, out)
                return {"ok": True, "task": task.kind, "rules": out}
            if task.kind == "policy_review":
                out = self._write_policy_review(Path(task.payload["artifacts"]), task.payload)
                self.task_store.mark_done(task.id, out)
                return {"ok": True, "task": task.kind, "review": out}
            raise ValueError(f"Unknown task kind: {task.kind}")
        except Exception as e:
            # ── POSTMORTEM: Extract lessons from failure ──
            postmortem = self.learner.postmortem(
                task_kind=task.kind,
                error=str(e),
                context=task.payload,
            )

            # Constitution Rule 8: Auto-retry once before marking failed
            requeued = self.task_store.requeue_for_retry(task.id, max_retries=1)
            if requeued:
                return {"ok": False, "task": task.kind, "error": str(e), "retrying": True, "retry_count": task.retry_count + 1, "postmortem": postmortem}
            self.task_store.mark_failed(task.id, str(e))
            return {"ok": False, "task": task.kind, "error": str(e), "retrying": False, "postmortem": postmortem}

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

    def _write_rules_reminder(self, artifacts_dir: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
        run_count = int(payload.get("run_count") or 0)
        gate = str(payload.get("gate") or "periodic")
        allow_improve = bool(payload.get("allow_improve", True))
        window_runs = int(payload.get("window_runs") or 0)
        window_size = int(payload.get("window_size") or 100)
        wm_ttl = int(payload.get("working_memory_ttl_runs") or 150)
        prior_half_life = int(payload.get("prior_half_life_runs") or 300)

        state_dir = artifacts_dir / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        out_md_path = state_dir / "current_rules.md"
        out_json_path = state_dir / "current_rules.json"
        reminder_obj = {
            "generated_at": _utc_iso(),
            "gate": gate,
            "run_count": run_count,
            "allow_improve": allow_improve,
            "window_runs": window_runs,
            "window_size": window_size,
            "working_memory_ttl_runs": wm_ttl,
            "prior_half_life_runs": prior_half_life,
            "hard_rules": [
                "No look-ahead and no universe leakage. Data quality fail means stop.",
                "Promote only with credible evidence; in-sample gains alone are invalid.",
                "Keep provenance immutable: config/data/code fingerprints must exist for every run.",
                "Enforce event-driven gates (fast/deep/reset). No uncontrolled optimization loops.",
                "Working memory uses TTL; priors decay over time to avoid stale lock-in.",
            ],
        }
        lines = [
            "# NEXUS Hard Rules Reminder",
            f"- generated_at: `{reminder_obj['generated_at']}`",
            f"- gate: `{gate}`",
            f"- run_count: `{run_count}`",
            f"- improve_allowed: `{allow_improve}`",
            f"- budget_window: `{window_runs}/{window_size}` runs",
            "",
            "## Hard Rules",
            "1. No look-ahead and no universe leakage. Data quality fail means stop.",
            "2. Promote only with credible evidence; in-sample gains alone are invalid.",
            "3. Keep provenance immutable: config/data/code fingerprints must exist for every run.",
            "4. Enforce event-driven gates (fast/deep/reset). No uncontrolled optimization loops.",
            "5. Working memory uses TTL; priors decay over time to avoid stale lock-in.",
            "",
            "## Memory Policy",
            f"- working_memory_ttl_runs: `{wm_ttl}`",
            f"- prior_half_life_runs: `{prior_half_life}`",
        ]
        out_md_path.write_text("\n".join(lines), encoding="utf-8")
        out_json_path.write_text(json.dumps(reminder_obj, indent=2, sort_keys=True), encoding="utf-8")
        self.memory.add(
            created_at=_utc_iso(),
            kind="rules_reminder",
            tags=["orion", "policy", gate],
            content=f"Rules reminder generated at run_count={run_count}, improve_allowed={allow_improve}",
            meta={"rules_md_path": str(out_md_path), "rules_json_path": str(out_json_path), "gate": gate, "run_count": run_count},
            run_id=None,
        )
        return {
            "rules_md_path": str(out_md_path),
            "rules_json_path": str(out_json_path),
            "gate": gate,
            "run_count": run_count,
            "allow_improve": allow_improve,
        }

    def _write_policy_review(self, artifacts_dir: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
        mode = str(payload.get("mode") or "periodic")
        run_count = int(payload.get("run_count") or 0)
        window_runs = int(payload.get("window_runs") or 0)
        window_size = int(payload.get("window_size") or 100)
        review_dir = artifacts_dir / "state" / "policy_reviews"
        review_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_md_path = review_dir / f"policy.{mode}.{ts}.md"
        out_json_path = review_dir / f"policy.{mode}.{ts}.json"

        task_counts = self.task_store.counts()
        runs_dir = artifacts_dir / "runs"
        runs_total = 0
        if runs_dir.exists():
            try:
                runs_total = len([d for d in runs_dir.iterdir() if d.is_dir()])
            except Exception:
                runs_total = 0

        lines = [
            "# NEXUS Policy Review",
            f"- generated_at: `{_utc_iso()}`",
            f"- mode: `{mode}`",
            f"- run_count: `{run_count}`",
            f"- runs_total: `{runs_total}`",
            f"- budget_window: `{window_runs}/{window_size}`",
            "",
            "## Task Counts",
            f"- {json.dumps(task_counts, sort_keys=True)}",
            "",
            "## Auto Recommendations",
        ]
        if mode == "budget_pause":
            lines.extend(
                [
                    "- Pause new `improve` bootstraps until next budget window.",
                    "- Continue `reflect` + `critique` + `wisdom` to consolidate learnings.",
                ]
            )
        elif mode == "reset_gate":
            lines.extend(
                [
                    "- Force `research_ingest` and reset exploration priors assumptions.",
                    "- Require fresh challenger hypotheses before next promotion attempts.",
                ]
            )
        elif mode == "deep_gate":
            lines.extend(
                [
                    "- Run full strategic critique and agent debate before new champion changes.",
                    "- Check whether current best strategy still beats recent challengers.",
                ]
            )
        else:
            lines.append("- Keep loop disciplined and continue monitoring drift and uncertainty.")

        review_obj = {
            "generated_at": _utc_iso(),
            "mode": mode,
            "run_count": run_count,
            "runs_total": runs_total,
            "window_runs": window_runs,
            "window_size": window_size,
            "task_counts": task_counts,
            "recommendations": lines[lines.index("## Auto Recommendations") + 1 :],
        }

        # Auto rebuttal: run Claude Opus on every policy review.
        rebuttal_context = {
            "mode": mode,
            "run_count": run_count,
            "runs_total": runs_total,
            "window_runs": window_runs,
            "window_size": window_size,
            "task_counts": task_counts,
            "recommendations": review_obj.get("recommendations") or [],
            "policy_state": self._read_policy_state(artifacts_dir),
        }
        rebuttal = run_opus_rebuttal(
            topic="Policy review quality gate for autonomous research loop",
            context=rebuttal_context,
            cfg=OpusRebuttalConfig(),
        )
        rebuttal_json_path = review_dir / f"rebuttal.{mode}.{ts}.json"
        rebuttal_text_path = review_dir / f"rebuttal.{mode}.{ts}.txt"
        rebuttal_paths = persist_opus_rebuttal(
            out_json_path=rebuttal_json_path,
            out_text_path=rebuttal_text_path,
            payload=rebuttal,
        )
        rebuttal_parsed = rebuttal.get("parsed") or {}
        rebuttal_verdict = str(rebuttal_parsed.get("verdict") or "").upper()
        policy_state = self._read_policy_state(artifacts_dir)
        policy_cfg = (policy_state.get("config") or {}) if isinstance(policy_state, dict) else {}
        fast_gate_runs = int(policy_cfg.get("fast_gate_runs") or 25)
        deep_gate_runs = int(policy_cfg.get("deep_gate_runs") or 150)
        force_pause_until_runs = None
        if rebuttal_verdict == "ROLLBACK":
            force_pause_until_runs = run_count + max(1, deep_gate_runs)
        elif rebuttal_verdict == "REVISE":
            force_pause_until_runs = run_count + max(1, fast_gate_runs)

        control_path = artifacts_dir / "state" / "research_policy_control.json"
        control_obj = self._read_policy_control(artifacts_dir)
        if force_pause_until_runs is not None:
            current_force = int(control_obj.get("force_pause_until_runs") or 0)
            force_pause_until_runs = max(force_pause_until_runs, current_force)
            control_obj = {
                **control_obj,
                "updated_at": _utc_iso(),
                "source": "opus_rebuttal",
                "mode": mode,
                "run_count": run_count,
                "verdict": rebuttal_verdict,
                "force_pause_until_runs": int(force_pause_until_runs),
                "reason": "Automatic pause from Opus rebuttal verdict",
                "rebuttal_json_path": rebuttal_paths["json_path"],
            }
            control_path.parent.mkdir(parents=True, exist_ok=True)
            control_path.write_text(json.dumps(control_obj, indent=2, sort_keys=True), encoding="utf-8")

        review_obj["opus_rebuttal"] = {
            "ok": bool(rebuttal.get("ok")),
            "returncode": rebuttal.get("returncode"),
            "parsed": rebuttal_parsed,
            "json_path": rebuttal_paths["json_path"],
            "text_path": rebuttal_paths["text_path"],
            "enforced_control": {
                "path": str(control_path),
                "force_pause_until_runs": int(force_pause_until_runs) if force_pause_until_runs is not None else None,
                "verdict": rebuttal_verdict,
            },
        }

        out_md_path.write_text("\n".join(lines), encoding="utf-8")
        out_json_path.write_text(json.dumps(review_obj, indent=2, sort_keys=True), encoding="utf-8")
        self.memory.add(
            created_at=_utc_iso(),
            kind="policy_review",
            tags=["orion", "policy", mode],
            content=f"Policy review generated ({mode}) with run_count={run_count}",
            meta={
                "review_md_path": str(out_md_path),
                "review_json_path": str(out_json_path),
                "mode": mode,
                "task_counts": task_counts,
                "opus_rebuttal_json_path": rebuttal_paths["json_path"],
                "opus_rebuttal_text_path": rebuttal_paths["text_path"],
                "opus_rebuttal_ok": bool(rebuttal.get("ok")),
                "opus_rebuttal_verdict": rebuttal_verdict,
                "policy_control_path": str(control_path),
                "force_pause_until_runs": int(force_pause_until_runs) if force_pause_until_runs is not None else None,
            },
            run_id=None,
        )
        return {
            "review_md_path": str(out_md_path),
            "review_json_path": str(out_json_path),
            "rebuttal_json_path": rebuttal_paths["json_path"],
            "rebuttal_text_path": rebuttal_paths["text_path"],
            "rebuttal_ok": bool(rebuttal.get("ok")),
            "rebuttal_verdict": rebuttal_verdict,
            "force_pause_until_runs": int(force_pause_until_runs) if force_pause_until_runs is not None else None,
            "mode": mode,
            "run_count": run_count,
        }

    def _read_policy_state(self, artifacts_dir: Path) -> Dict[str, Any]:
        p = artifacts_dir / "state" / "research_policy_state.json"
        if not p.exists():
            return {}
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def _read_policy_control(self, artifacts_dir: Path) -> Dict[str, Any]:
        p = artifacts_dir / "state" / "research_policy_control.json"
        if not p.exists():
            return {}
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    # ── Agent orchestration ──────────────────────────────────────────────────

    def _build_agent_context(self) -> AgentContext:
        """Load latest wisdom + metrics + regime for agent context."""
        # Load wisdom from latest.json if exists
        wisdom_path = self.cfg.artifacts_dir / "wisdom" / "latest.json"
        wisdom: Dict[str, Any] = {}
        if wisdom_path.exists():
            try:
                wisdom = json.loads(wisdom_path.read_text("utf-8"))
            except Exception:
                pass

        # Load recent metrics from wisdom or fall back to empty
        recent_metrics: Dict[str, Any] = (wisdom.get("recent") or {}).get("metrics") or {}

        # Load market regime if available
        regime: Dict[str, Any] = {}
        regime_path = self.cfg.artifacts_dir / "state" / "regime.json"
        if regime_path.exists():
            try:
                regime = json.loads(regime_path.read_text("utf-8"))
            except Exception:
                pass

        # Best params
        best_params: Optional[Dict[str, Any]] = None
        best_params_path = self.cfg.artifacts_dir / "memory" / "best_params.json"
        if best_params_path.exists():
            try:
                best_params = json.loads(best_params_path.read_text("utf-8"))
            except Exception:
                pass

        # Policy-aware working memory window (soft forgetting by recentness).
        policy_state = {}
        policy_path = self.cfg.artifacts_dir / "state" / "research_policy_state.json"
        if policy_path.exists():
            try:
                policy_state = json.loads(policy_path.read_text("utf-8"))
            except Exception:
                policy_state = {}
        wm_ttl = 120
        try:
            wm_ttl = int(((policy_state.get("config") or {}).get("working_memory_ttl_runs")) or wm_ttl)
        except Exception:
            wm_ttl = 120
        wm_ttl = max(20, min(wm_ttl, 400))
        prior_half_life = 300
        try:
            prior_half_life = int(((policy_state.get("config") or {}).get("prior_half_life_runs")) or prior_half_life)
        except Exception:
            prior_half_life = 300
        prior_half_life = max(20, min(prior_half_life, 1200))

        hard_rules = []
        current_rules_path = self.cfg.artifacts_dir / "state" / "current_rules.json"
        if current_rules_path.exists():
            try:
                rules_obj = json.loads(current_rules_path.read_text("utf-8"))
                if isinstance(rules_obj, dict):
                    hr = rules_obj.get("hard_rules") or []
                    if isinstance(hr, list):
                        hard_rules = [str(x) for x in hr if str(x).strip()]
            except Exception:
                hard_rules = []

        memory_items = []
        try:
            mem = self.memory.recent(limit=wm_ttl)
            for rank, it in enumerate(mem):
                decay_weight = 0.5 ** (float(rank) / float(max(1, prior_half_life)))
                memory_items.append(
                    {
                        "id": it.id,
                        "created_at": it.created_at,
                        "kind": it.kind,
                        "tags": list(it.tags),
                        "content": (it.content or "")[:500],
                        "run_id": it.run_id,
                        "decay_weight": round(float(decay_weight), 6),
                    }
                )
        except Exception:
            memory_items = []

        # ── Load operational lessons for agents ──
        operational_lessons = []
        try:
            operational_lessons = self.learner.get_agent_lessons(max_lessons=15)
        except Exception:
            pass

        return AgentContext(
            wisdom=wisdom,
            recent_metrics=recent_metrics,
            regime=regime,
            best_params=best_params,
            memory_items=memory_items,
            extra={
                "policy": {
                    "allow_improve": bool(policy_state.get("allow_improve", True)) if isinstance(policy_state, dict) else True,
                    "window_runs": int(policy_state.get("window_runs") or 0) if isinstance(policy_state, dict) else 0,
                    "working_memory_ttl_runs": wm_ttl,
                    "prior_half_life_runs": prior_half_life,
                },
                "hard_rules": hard_rules,
                "operational_lessons": operational_lessons,
            },
        )

    def run_agents(self) -> Dict[str, Any]:
        """
        Run agents using the 3-phase Decision Network pipeline.

        Phase 1 — GENERATE: ATLAS + CIPHER (parallel)
        Phase 2 — VALIDATE: ECHO (cross-validates Phase 1 outputs, dual-model)
        Phase 3 — DECIDE:   FLUX + Synthesis gate (approve/block/escalate)
        """
        from concurrent.futures import ThreadPoolExecutor
        context = self._build_agent_context()
        results: Dict[str, Any] = {"pipeline": "v3-decision-network"}

        def _run_and_log(agent_cls, ctx):
            agent = agent_cls()
            result = agent.run(ctx)
            self.memory.add(
                created_at=_utc_iso(),
                kind="agent_output",
                tags=["agent", result.agent_name, result.phase],
                content=result.raw_response[:1000],
                meta={
                    "agent": result.agent_name,
                    "model": result.model_used,
                    "parsed": result.parsed,
                    "phase": result.phase,
                },
                run_id=None,
            )
            return result

        # ── Phase 1: GENERATE (ATLAS + CIPHER parallel) ──
        phase1: Dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_atlas = pool.submit(_run_and_log, AtlasAgent, context)
            fut_cipher = pool.submit(_run_and_log, CipherAgent, context)
            try:
                atlas_result = fut_atlas.result()
                phase1["atlas"] = atlas_result.parsed
                results["atlas"] = atlas_result.to_dict()
            except Exception as e:
                phase1["atlas"] = {}
                results["atlas"] = {"error": str(e), "fallback_used": True}
            try:
                cipher_result = fut_cipher.result()
                phase1["cipher"] = cipher_result.parsed
                results["cipher"] = cipher_result.to_dict()
            except Exception as e:
                phase1["cipher"] = {}
                results["cipher"] = {"error": str(e), "fallback_used": True}

        # ── Phase 2: VALIDATE (ECHO with Phase 1 context) ──
        context.phase1_results = phase1
        try:
            echo_result = _run_and_log(EchoAgent, context)
            results["echo"] = echo_result.to_dict()
        except Exception as e:
            echo_result = None
            results["echo"] = {"error": str(e), "fallback_used": True}

        # ── Phase 3: DECIDE (FLUX sees everything) ──
        echo_parsed = echo_result.parsed if echo_result else {}
        context.phase1_results["echo"] = echo_parsed
        try:
            flux_result = _run_and_log(FluxAgent, context)
            results["flux"] = flux_result.to_dict()
        except Exception as e:
            results["flux"] = {"error": str(e), "fallback_used": True}

        # ── Synthesis gate ──
        try:
            from ..agents.synthesis import run_decision_gate
            gate = run_decision_gate(
                atlas_output=phase1.get("atlas", {}),
                cipher_output=phase1.get("cipher", {}),
                echo_output=echo_parsed,
                flux_output=flux_result.parsed if flux_result else {},
            )
            results["gate_decision"] = gate.to_dict()
        except Exception as e:
            results["gate_decision"] = {"decision": "block", "error": str(e)}

        return results
