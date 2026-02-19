from __future__ import annotations

"""
NEXUS Autonomous Research Cycle v2.

A full research iteration that:
1. Fetches live market data (Binance funding rates + overview)
2. Searches arxiv for relevant papers
3. Runs ATLAS strategy proposals + CIPHER risk check + ECHO validation
4. Auto-creates Kanban tasks from agent proposals
5. Stores findings in NEXUS memory + diary
6. Returns a structured research report

Designed to run as part of the autonomous loop or on-demand via CLI / dashboard.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception:
        return None


def _read_jsonl_tail(path: Path, n: int = 50) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for ln in path.read_text("utf-8").splitlines()[-n:]:
        try:
            obj = json.loads(ln)
            if isinstance(obj, dict):
                out.append(obj)
        except Exception:
            continue
    return out


class ResearchCycle:
    """
    Orchestrates a full autonomous research iteration.

    Usage:
        cycle = ResearchCycle(artifacts_dir=Path("artifacts"), config_path=Path("configs/run_synthetic_funding.json"))
        report = cycle.run()
    """

    def __init__(
        self,
        artifacts_dir: Path,
        config_path: Optional[Path] = None,
        strategy_topic: str = "crypto perpetual futures funding carry momentum",
        max_workers: int = 3,
    ):
        self.artifacts_dir = Path(artifacts_dir)
        self.config_path = Path(config_path) if config_path else None
        self.strategy_topic = strategy_topic
        self.max_workers = max_workers
        self._report: Dict[str, Any] = {
            "started_at": _now_iso(),
            "market_data": {},
            "arxiv_findings": [],
            "agent_proposals": {},
            "tasks_created": [],
            "memory_stored": 0,
            "errors": [],
        }

    # ── Step 1: Market Intelligence ─────────────────────────────────────

    def _gather_market_data(self) -> Dict[str, Any]:
        try:
            from ..tools.web_research import fetch_binance_funding_rates, fetch_binance_market_overview
            result: Dict[str, Any] = {}
            for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
                rates = fetch_binance_funding_rates(sym, limit=3)
                if rates:
                    result[sym] = {
                        "rates": [float(r.get("fundingRate", 0)) for r in rates],
                        "latest_rate": float(rates[-1].get("fundingRate", 0)),
                        "avg_rate": sum(float(r.get("fundingRate", 0)) for r in rates) / len(rates),
                    }
            overview = fetch_binance_market_overview(limit=10)
            result["top_tickers"] = overview.get("tickers", [])[:5]
            return result
        except Exception as e:
            self._report["errors"].append(f"market_data: {e}")
            return {}

    # ── Step 2: Web Research ─────────────────────────────────────────────

    def _search_arxiv(self) -> List[Dict[str, Any]]:
        try:
            from ..tools.web_research import search_arxiv
            papers = search_arxiv(self.strategy_topic, max_results=3)
            return papers
        except Exception as e:
            self._report["errors"].append(f"arxiv: {e}")
            return []

    # ── Step 3: Load Context ─────────────────────────────────────────────

    def _build_agent_context(self, market_data: Dict[str, Any]) -> Any:
        try:
            from ..agents.base import AgentContext
            # Load latest metrics
            metrics = {}
            runs_dir = self.artifacts_dir / "runs"
            if runs_dir.exists():
                dirs = sorted(runs_dir.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True)
                for d in dirs[:5]:
                    mp = d / "metrics.json"
                    if mp.exists():
                        m = _read_json(mp) or {}
                        metrics = m.get("summary") or {}
                        break

            # Load wisdom
            wisdom = _read_json(self.artifacts_dir / "wisdom" / "latest.json") or {}

            # Load regime
            regime: Dict[str, Any] = {}
            try:
                from ..risk.regime import detect_regime
                if metrics:
                    # Build minimal returns for regime detection
                    regime = {"regime": "unknown", "evidence": "using latest metrics"}
            except Exception:
                pass

            # Load best params
            best_params = None
            if self.config_path and self.config_path.exists():
                cfg = _read_json(self.config_path) or {}
                best_params = cfg.get("strategy", {}).get("params")

            # Inject market data
            extra = {
                "market_data": market_data,
                "research_topic": self.strategy_topic,
            }

            return AgentContext(
                wisdom=wisdom,
                recent_metrics=metrics,
                regime=regime,
                best_params=best_params,
                extra=extra,
            )
        except Exception as e:
            self._report["errors"].append(f"context: {e}")
            return None

    # ── Step 4: Run Agents ───────────────────────────────────────────────

    def _run_atlas(self, context: Any) -> Dict[str, Any]:
        try:
            from ..agents.atlas import AtlasAgent
            agent = AtlasAgent()
            result = agent.run(context)
            return result.parsed or {}
        except Exception as e:
            self._report["errors"].append(f"atlas: {e}")
            return {}

    def _run_cipher(self, context: Any) -> Dict[str, Any]:
        try:
            from ..agents.cipher import CipherAgent
            agent = CipherAgent()
            result = agent.run(context)
            return result.parsed or {}
        except Exception as e:
            self._report["errors"].append(f"cipher: {e}")
            return {}

    def _run_echo(self, context: Any) -> Dict[str, Any]:
        try:
            from ..agents.echo import EchoAgent
            agent = EchoAgent()
            result = agent.run(context)
            return result.parsed or {}
        except Exception as e:
            self._report["errors"].append(f"echo: {e}")
            return {}

    def _run_flux(self, context: Any) -> Dict[str, Any]:
        try:
            from ..agents.flux import FluxAgent
            agent = FluxAgent()
            result = agent.run(context)
            return result.parsed or {}
        except Exception as e:
            self._report["errors"].append(f"flux: {e}")
            return {}

    # ── Step 5: Auto-create Kanban tasks from proposals ─────────────────

    def _create_tasks_from_proposals(
        self,
        atlas_proposals: List[Dict[str, Any]],
        cipher_flags: List[Dict[str, Any]],
        echo_issues: List[Dict[str, Any]],
    ) -> List[str]:
        created: List[str] = []
        try:
            from ..tasks.manager import TaskManager
            tm = TaskManager(self.artifacts_dir)

            # ATLAS proposals → new experiment tasks
            for p in (atlas_proposals or [])[:3]:
                name = str(p.get("name") or "strategy_proposal")
                rationale = str(p.get("rationale") or "")
                priority_num = int(p.get("priority") or 3)
                priority = "high" if priority_num <= 2 else "medium"
                overrides = json.dumps(p.get("config_overrides") or {})
                task = tm.create(
                    title=f"[ATLAS] {name}",
                    description=f"{rationale}\nOverrides: {overrides}",
                    priority=priority,
                    assignee="ATLAS",
                    tags=["experiment", "atlas-proposal"],
                    created_by="NEXUS-autonomous",
                )
                created.append(task.id)

            # CIPHER flags → risk review tasks
            for flag in (cipher_flags or [])[:2]:
                concern = str(flag.get("concern") or str(flag)[:100])
                task = tm.create(
                    title=f"[CIPHER] Risk: {concern[:80]}",
                    description=str(flag),
                    priority="high",
                    assignee="CIPHER",
                    tags=["risk", "cipher-flag"],
                    created_by="NEXUS-autonomous",
                )
                created.append(task.id)

            # ECHO issues → validation tasks
            for issue in (echo_issues or [])[:2]:
                desc = str(issue.get("issue") or str(issue)[:100])
                task = tm.create(
                    title=f"[ECHO] Validate: {desc[:80]}",
                    description=str(issue),
                    priority="medium",
                    assignee="ECHO",
                    tags=["validation", "echo-issue"],
                    created_by="NEXUS-autonomous",
                )
                created.append(task.id)

        except Exception as e:
            self._report["errors"].append(f"task_creation: {e}")
        return created

    # ── Step 6: Store findings in memory ────────────────────────────────

    def _store_findings(
        self,
        market_data: Dict[str, Any],
        arxiv_papers: List[Dict[str, Any]],
        agent_results: Dict[str, Any],
    ) -> int:
        count = 0
        try:
            from ..memory.store import MemoryStore
            db = self.artifacts_dir / "memory" / "memory.db"
            ms = MemoryStore(db)
            now = _now_iso()
            try:
                # Market intelligence
                if market_data:
                    ms.add(
                        created_at=now,
                        kind="market_intelligence",
                        tags=["market", "funding", "auto"],
                        content=json.dumps(market_data, ensure_ascii=False)[:2000],
                        meta={"source": "research_cycle", "ts": now},
                    )
                    count += 1

                # Arxiv papers
                for paper in arxiv_papers[:3]:
                    ms.add(
                        created_at=now,
                        kind="research_paper",
                        tags=["arxiv", "research", "auto"],
                        content=f"{paper.get('title', '')}: {paper.get('summary', '')[:500]}",
                        meta={"source": "arxiv", "url": paper.get("url", ""), "ts": now},
                    )
                    count += 1

                # Agent synthesis
                if agent_results:
                    ms.add(
                        created_at=now,
                        kind="agent_synthesis",
                        tags=["agents", "proposals", "auto"],
                        content=json.dumps(agent_results, ensure_ascii=False)[:2000],
                        meta={"source": "research_cycle", "ts": now},
                    )
                    count += 1
            finally:
                ms.close()

            # Invalidate RAG index
            try:
                (self.artifacts_dir / "brain" / "rag_index.json").unlink(missing_ok=True)
            except Exception:
                pass
        except Exception as e:
            self._report["errors"].append(f"store_findings: {e}")
        return count

    # ── Step 7: Write diary entry ─────────────────────────────────────────

    def _write_diary(
        self,
        market_data: Dict[str, Any],
        arxiv_papers: List[Dict[str, Any]],
        agent_results: Dict[str, Any],
        tasks_created: List[str],
    ) -> None:
        try:
            from ..brain.diary import NexusDiary
            diary = NexusDiary(self.artifacts_dir)

            btc_rate = (market_data.get("BTCUSDT") or {}).get("latest_rate", 0)
            paper_titles = [p.get("title", "") for p in arxiv_papers[:2]]
            atlas = agent_results.get("atlas", {})
            proposals = [p.get("name", "") for p in (atlas.get("proposals") or [])[:3]]

            key_learnings = [
                f"BTC funding rate: {btc_rate:.6f} ({'positive - longs pay' if btc_rate > 0 else 'negative - shorts pay'})",
            ] + [f"Paper: {t[:80]}" for t in paper_titles if t]

            next_plans = [f"Experiment: {p}" for p in proposals if p][:3]
            if not next_plans:
                next_plans = ["Continue parameter optimization", "Monitor regime for changes"]

            diary.write_entry(
                experiments_run=len(tasks_created),
                best_strategy=None,
                best_sharpe=None,
                key_learnings=key_learnings,
                next_plans=next_plans,
                mood="focused",
                goals_progress="Research cycle completed",
                raw_summary=f"Research cycle: {len(arxiv_papers)} papers, {len(tasks_created)} tasks created",
            )
        except Exception as e:
            self._report["errors"].append(f"diary: {e}")

    # ── Step 8: Notify ──────────────────────────────────────────────────

    def _notify(self, summary: str) -> None:
        try:
            from ..comms.notifier import NexusNotifier
            notifier = NexusNotifier(self.artifacts_dir)
            notifier.notify(summary, level="info")
        except Exception:
            pass

    # ── Main orchestrator ───────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """
        Run the full research cycle and return a structured report.
        Steps run in parallel where safe.
        """
        t0 = time.time()

        # Phase A: Parallel data gathering
        market_data: Dict[str, Any] = {}
        arxiv_papers: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_market = pool.submit(self._gather_market_data)
            fut_arxiv = pool.submit(self._search_arxiv)
            market_data = fut_market.result()
            arxiv_papers = fut_arxiv.result()

        self._report["market_data"] = market_data
        self._report["arxiv_findings"] = [
            {"title": p.get("title", ""), "url": p.get("url", ""), "summary": (p.get("summary") or "")[:200]}
            for p in arxiv_papers
        ]

        # Phase B: Build agent context
        context = self._build_agent_context(market_data)

        # Phase C: Parallel agent dispatch
        agent_results: Dict[str, Any] = {}
        if context is not None:
            agent_tasks = {
                "atlas": self._run_atlas,
                "cipher": self._run_cipher,
                "echo": self._run_echo,
                "flux": self._run_flux,
            }
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {pool.submit(fn, context): name for name, fn in agent_tasks.items()}
                for fut in as_completed(futures):
                    name = futures[fut]
                    try:
                        agent_results[name] = fut.result()
                    except Exception as e:
                        self._report["errors"].append(f"agent_{name}: {e}")

        self._report["agent_proposals"] = agent_results

        # Phase D: Create Kanban tasks from proposals
        atlas_proposals = (agent_results.get("atlas") or {}).get("proposals") or []
        cipher_flags = (agent_results.get("cipher") or {}).get("flags") or []
        echo_issues = (agent_results.get("echo") or {}).get("issues") or []
        tasks_created = self._create_tasks_from_proposals(atlas_proposals, cipher_flags, echo_issues)
        self._report["tasks_created"] = tasks_created

        # Phase E: Store findings in memory
        memory_count = self._store_findings(market_data, arxiv_papers, agent_results)
        self._report["memory_stored"] = memory_count

        # Phase F: Write diary entry
        self._write_diary(market_data, arxiv_papers, agent_results, tasks_created)

        # Phase G: Save report to artifacts
        self._report["completed_at"] = _now_iso()
        self._report["duration_sec"] = round(time.time() - t0, 2)

        reports_dir = self.artifacts_dir / "research" / "cycles"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / f"cycle_{_now_iso()[:10]}.json"
        try:
            report_path.write_text(json.dumps(self._report, indent=2, ensure_ascii=False), "utf-8")
        except Exception:
            pass

        # Latest symlink-equivalent: overwrite "latest.json"
        latest_path = self.artifacts_dir / "research" / "latest_cycle.json"
        try:
            latest_path.write_text(json.dumps(self._report, indent=2, ensure_ascii=False), "utf-8")
        except Exception:
            pass

        # Notify
        n_papers = len(arxiv_papers)
        n_tasks = len(tasks_created)
        n_errors = len(self._report["errors"])
        self._notify(
            f"Research cycle done in {self._report['duration_sec']}s: "
            f"{n_papers} papers, {n_tasks} tasks created, {n_errors} errors"
        )

        return self._report
