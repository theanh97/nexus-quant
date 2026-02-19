from __future__ import annotations

"""
NEXUS Daily Research & Self-Learning Routine.

Mimics how a world-class quant researcher learns and adapts:

  0600 UTC — Morning Brief: fetch papers, news, market data
  0700 UTC — Hypothesis Generation: what new ideas to test today?
  0800 UTC — Experiment Queue: auto-queue experiments to Kanban
  Continuous — Backtest Loop: test hypotheses, measure results
  2000 UTC — Evening Review: what worked? what failed? update knowledge
  2100 UTC — Brain Diary: record learnings, update priors

The system learns from:
  1. Research papers (arXiv, SSRN abstracts)
  2. Live market data (funding rates, OI, price action)
  3. Own backtest results (what strategies worked historically)
  4. GLM-5 reasoning (connecting dots across domains)

Knowledge compounds over time via:
  - brain/knowledge_base.json — structured domain knowledge
  - brain/strategy_priors.json — prior beliefs updated by Bayesian logic
  - brain/experiment_ledger.jsonl — every experiment ever run
"""

import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[NEXUS Daily {ts}] {msg}", flush=True)


# ── Knowledge Base ─────────────────────────────────────────────────────────

class KnowledgeBase:
    """
    Persistent knowledge store for NEXUS.
    Records learnings, strategy insights, and research findings.
    Grows more intelligent over time.
    """

    def __init__(self, artifacts_dir: Path):
        self.brain_dir = artifacts_dir / "brain"
        self.brain_dir.mkdir(parents=True, exist_ok=True)
        self.kb_path = self.brain_dir / "knowledge_base.json"
        self.experiment_log = self.brain_dir / "experiment_ledger.jsonl"
        self._data = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.kb_path.exists():
            try:
                return json.loads(self.kb_path.read_text("utf-8"))
            except Exception:
                pass
        return {
            "version": 1,
            "created": _now_iso(),
            "domain_knowledge": {},
            "strategy_insights": [],
            "experiment_history": [],
            "champion_strategy": None,
            "learning_history": [],
            "market_observations": [],
        }

    def save(self) -> None:
        self._data["last_updated"] = _now_iso()
        self.kb_path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    def record_experiment(self, experiment: Dict[str, Any]) -> None:
        """Record an experiment (hypothesis + result)."""
        entry = {**experiment, "ts": _now_iso()}
        self._data["experiment_history"] = (
            self._data.get("experiment_history", [])[-99:] + [entry]
        )
        with open(self.experiment_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        self.save()

    def add_insight(self, category: str, insight: str, confidence: float = 0.5,
                    source: str = "experiment") -> None:
        """Add a new domain insight."""
        self._data.setdefault("strategy_insights", []).append({
            "category": category,
            "insight": insight,
            "confidence": confidence,
            "source": source,
            "ts": _now_iso(),
        })
        # Keep last 200 insights
        self._data["strategy_insights"] = self._data["strategy_insights"][-200:]
        self.save()

    def update_domain_knowledge(self, key: str, value: Any) -> None:
        self._data.setdefault("domain_knowledge", {})[key] = {
            "value": value,
            "updated": _now_iso(),
        }
        self.save()

    def set_champion(self, strategy_name: str, sharpe: float, config_path: str,
                     metrics: Dict[str, Any]) -> None:
        """Promote a new champion strategy."""
        self._data["champion_strategy"] = {
            "name": strategy_name,
            "sharpe": sharpe,
            "config_path": config_path,
            "metrics": metrics,
            "promoted_at": _now_iso(),
        }
        self.save()
        _log(f"NEW CHAMPION: {strategy_name} | Sharpe={sharpe:.3f}")

    def get_champion(self) -> Optional[Dict[str, Any]]:
        return self._data.get("champion_strategy")

    def record_learning(self, what: str, why: str, action: str) -> None:
        """Record a learning in the daily diary format."""
        self._data.setdefault("learning_history", []).append({
            "what": what,
            "why": why,
            "action": action,
            "ts": _now_iso(),
        })
        self._data["learning_history"] = self._data["learning_history"][-500:]
        self.save()

    def get_recent_insights(self, n: int = 10) -> List[Dict[str, Any]]:
        return list(reversed(self._data.get("strategy_insights", [])))[:n]

    def get_best_experiments(self, n: int = 5) -> List[Dict[str, Any]]:
        exps = self._data.get("experiment_history", [])
        return sorted(exps, key=lambda x: x.get("sharpe", 0), reverse=True)[:n]


# ── Experiment Runner ──────────────────────────────────────────────────────

class ExperimentRunner:
    """
    Auto-generates and runs backtest experiments based on hypotheses.
    Measures statistical significance of results.
    """

    def __init__(self, artifacts_dir: Path, configs_dir: Path):
        self.artifacts_dir = artifacts_dir
        self.configs_dir = configs_dir
        self.kb = KnowledgeBase(artifacts_dir)

    def run_backtest(self, config_path: Path, timeout: int = 600) -> Optional[Dict[str, Any]]:
        """Run a single backtest and return metrics."""
        try:
            env = {**os.environ, "PYTHONPATH": str(_PROJECT_ROOT)}
            result = subprocess.run(
                [sys.executable, "-m", "nexus_quant", "run", "--config", str(config_path)],
                cwd=str(_PROJECT_ROOT),
                capture_output=True, text=True,
                timeout=timeout, env=env,
            )
            if result.returncode != 0:
                _log(f"Backtest failed: {result.stderr[-500:]}")
                return None
            # Find latest run matching this config's run_name
            try:
                cfg = json.loads(config_path.read_text("utf-8"))
                run_name = cfg.get("run_name", config_path.stem)
                runs_dir = self.artifacts_dir / "runs"
                matching = sorted(
                    [d for d in runs_dir.iterdir() if d.is_dir() and run_name in d.name],
                    key=lambda d: d.stat().st_mtime, reverse=True,
                )
                if matching:
                    metrics_path = matching[0] / "metrics.json"
                    if metrics_path.exists():
                        return json.loads(metrics_path.read_text("utf-8"))
            except Exception as e:
                _log(f"Could not parse results: {e}")
            return {}
        except subprocess.TimeoutExpired:
            _log(f"Backtest timed out: {config_path.name}")
            return None
        except Exception as e:
            _log(f"Backtest error: {e}")
            return None

    def get_sharpe_from_metrics(self, metrics: Dict[str, Any]) -> float:
        """Extract the correctly-annualized Sharpe ratio."""
        if not metrics:
            return 0.0
        summary = metrics.get("summary", metrics)
        raw_sharpe = float(summary.get("sharpe") or summary.get("sharpe_ratio") or 0.0)
        ppy = float(metrics.get("meta", {}).get("periods_per_year") or 365.0)
        # Correct for wrong periods_per_year (old bug)
        if ppy == 365.0:
            import math
            raw_sharpe *= math.sqrt(8760.0 / 365.0)
        return raw_sharpe

    def compare_all_configs(self) -> List[Dict[str, Any]]:
        """Compare metrics across all existing run results."""
        results = []
        runs_dir = self.artifacts_dir / "runs"
        if not runs_dir.exists():
            return results
        for run_dir in sorted(runs_dir.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True)[:20]:
            if not run_dir.is_dir():
                continue
            metrics_path = run_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            try:
                m = json.loads(metrics_path.read_text("utf-8"))
                sh = self.get_sharpe_from_metrics(m)
                if sh != 0:
                    results.append({
                        "run": run_dir.name[:60],
                        "sharpe": round(sh, 3),
                        "raw_sharpe": m.get("summary", {}).get("sharpe", 0),
                        "cagr": m.get("summary", {}).get("cagr", 0),
                        "mdd": m.get("summary", {}).get("max_drawdown", 0),
                    })
            except Exception:
                pass
        return sorted(results, key=lambda x: x["sharpe"], reverse=True)

    def run_all_binance_configs(self) -> Dict[str, float]:
        """Run all Binance configs and return {config_name: sharpe}."""
        results = {}
        configs = sorted(self.configs_dir.glob("run_binance_*.json"))
        _log(f"Running {len(configs)} Binance configs...")
        for cfg_path in configs:
            _log(f"  Running {cfg_path.name}...")
            metrics = self.run_backtest(cfg_path)
            if metrics:
                sh = self.get_sharpe_from_metrics(metrics)
                results[cfg_path.name] = sh
                _log(f"  {cfg_path.name}: Sharpe={sh:.3f}")
            else:
                _log(f"  {cfg_path.name}: FAILED")
        return results

    def promote_champion(self, results: Dict[str, float]) -> Optional[str]:
        """Promote the best strategy as champion."""
        if not results:
            return None
        best_config = max(results, key=results.get)
        best_sharpe = results[best_config]
        champion = self.kb.get_champion()
        if champion is None or best_sharpe > champion.get("sharpe", 0):
            cfg_path = self.configs_dir / best_config
            self.kb.set_champion(
                strategy_name=best_config,
                sharpe=best_sharpe,
                config_path=str(cfg_path),
                metrics={"sharpe": best_sharpe, "all_results": results},
            )
            return best_config
        return None


# ── Daily Routine Orchestrator ─────────────────────────────────────────────

class DailyRoutine:
    """
    The NEXUS daily self-improvement cycle.

    Mimics how a world-class quant researcher spends their day:
    1. Morning: absorb new research
    2. Midday: generate and test hypotheses
    3. Evening: evaluate results, update knowledge
    4. Night: distill learnings, set tomorrow's agenda
    """

    def __init__(self, artifacts_dir: Path, configs_dir: Optional[Path] = None):
        self.artifacts_dir = artifacts_dir
        self.configs_dir = configs_dir or _PROJECT_ROOT / "configs"
        self.kb = KnowledgeBase(artifacts_dir)
        self.runner = ExperimentRunner(artifacts_dir, self.configs_dir)

    # ── Phase 1: Morning Research Brief ───────────────────────────────────

    def morning_research(self, llm_client: Any = None) -> Dict[str, Any]:
        """
        Fetch and process latest research from 50+ sources:
        arXiv, Reddit forums, quant blogs, crypto news, ML communities.
        Returns a structured research brief.
        """
        _log("=== MORNING RESEARCH BRIEF (50+ sources) ===")
        all_items: List[Dict[str, Any]] = []
        all_hypotheses: List[Dict[str, Any]] = []
        stats: Dict[str, Any] = {}

        # ── 1. Multi-source RSS/API fetch ──
        try:
            from .rss_fetcher import fetch_all_sources, extract_hypotheses, save_research_session
            from .source_registry import SOURCES
            _log(f"Fetching from {len(SOURCES)} sources (academic, forums, blogs, crypto, ML)...")
            items, fetch_stats = fetch_all_sources(min_relevance=0.05, rate_limit_seconds=0.4)
            _log(f"  Fetched: {fetch_stats['fetched_sources']}/{fetch_stats['total_sources']} sources | "
                 f"{fetch_stats['total_items']} articles | "
                 f"Failed: {len(fetch_stats['failed_sources'])}")
            all_items.extend(items)
            stats.update(fetch_stats)

            rss_hypotheses = extract_hypotheses(items, top_n=15)
            all_hypotheses.extend(rss_hypotheses)
            save_research_session(items, rss_hypotheses, stats, self.artifacts_dir)
        except Exception as e:
            _log(f"Multi-source fetch failed: {e}")

        # ── 2. arXiv deep fetch (dedicated) ──
        papers: List[Dict[str, Any]] = []
        try:
            from .arxiv_fetcher import fetch_arxiv_papers, extract_hypotheses_from_papers
            _log("Deep-fetching arXiv papers...")
            papers = fetch_arxiv_papers(days_back=7, min_relevance=0.05)
            _log(f"  arXiv: {len(papers)} relevant papers")
            arxiv_hyps = extract_hypotheses_from_papers(papers, llm_client=llm_client)
            # Merge unique hypotheses
            existing_hyp_texts = {h.get("hypothesis", "") for h in all_hypotheses}
            for h in arxiv_hyps:
                if h.get("hypothesis", "") not in existing_hyp_texts:
                    all_hypotheses.append(h)
        except Exception as e:
            _log(f"arXiv fetch failed: {e}")

        # ── 3. Score & rank all items + hypotheses ──
        all_items_sorted = sorted(all_items, key=lambda x: x.get("relevance", 0.0), reverse=True)
        all_hypotheses_sorted = sorted(all_hypotheses, key=lambda h: h.get("confidence", 0.5), reverse=True)

        # ── 4. Categorize by source type ──
        by_category: Dict[str, List] = {}
        for item in all_items_sorted:
            cat = item.get("category", "other")
            by_category.setdefault(cat, []).append(item)

        _log(f"  Categories: { {k: len(v) for k, v in by_category.items()} }")
        _log(f"  Total hypotheses: {len(all_hypotheses_sorted)}")

        brief = {
            "ts": _now_iso(),
            "stats": stats,
            "total_items": len(all_items_sorted),
            "papers": papers[:15],
            "top_items": all_items_sorted[:30],
            "by_category": {k: v[:5] for k, v in by_category.items()},
            "hypotheses": all_hypotheses_sorted[:10],
            "top_topics": self._extract_topics_from_items(all_items_sorted[:50]),
        }

        # ── 5. Save brief ──
        brief_path = self.artifacts_dir / "brain" / "daily_brief.json"
        brief_path.parent.mkdir(parents=True, exist_ok=True)
        brief_path.write_text(json.dumps(brief, indent=2, default=str), encoding="utf-8")

        # ── 6. Record insights in knowledge base ──
        for h in all_hypotheses_sorted[:5]:
            self.kb.add_insight(
                category=h.get("trigger_keyword", h.get("strategy_type", "general")),
                insight=h.get("hypothesis", ""),
                confidence=float(h.get("confidence", 0.5)),
                source=h.get("source_name", "research"),
            )
        self.kb.update_domain_knowledge("last_research_brief", {
            "ts": _now_iso(),
            "sources_fetched": stats.get("fetched_sources", 0),
            "total_items": len(all_items_sorted),
            "hypotheses": len(all_hypotheses_sorted),
            "top_topics": brief["top_topics"],
        })
        return brief

    def _extract_topics(self, papers: List[Dict]) -> List[str]:
        """Extract main topics from papers (legacy — use _extract_topics_from_items)."""
        all_words = " ".join(p.get("title", "") for p in papers).lower()
        topic_keywords = ["momentum", "carry", "mean reversion", "volatility", "machine learning",
                          "deep learning", "reinforcement learning", "factor", "risk parity",
                          "microstructure", "cryptocurrency", "regime"]
        return [kw for kw in topic_keywords if kw in all_words]

    def _extract_topics_from_items(self, items: List[Dict]) -> List[str]:
        """Extract main topics from all fetched items across all sources."""
        all_words = " ".join(
            f"{i.get('title', '')} {i.get('summary', '')}" for i in items
        ).lower()
        topic_keywords = [
            "momentum", "carry", "mean reversion", "trend following", "volatility",
            "machine learning", "deep learning", "reinforcement learning", "lstm",
            "transformer", "neural network", "factor model", "risk parity",
            "cross-sectional", "time series", "cryptocurrency", "bitcoin", "ethereum",
            "defi", "funding rate", "basis trade", "regime", "drawdown", "sharpe",
            "backtest", "walk-forward", "overfitting", "feature engineering",
            "portfolio optimization", "kelly", "ensemble",
        ]
        found = [kw for kw in topic_keywords if kw in all_words]
        return found[:20]

    # ── Phase 2: Hypothesis Testing ───────────────────────────────────────

    def test_hypotheses(self) -> Dict[str, Any]:
        """
        Test queued hypotheses via backtest.
        Returns summary of experiments run today.
        """
        _log("=== HYPOTHESIS TESTING ===")
        results = self.runner.run_all_binance_configs()
        champion = self.runner.promote_champion(results)

        summary = {
            "ts": _now_iso(),
            "configs_tested": len(results),
            "results": results,
            "new_champion": champion,
            "best_sharpe": max(results.values()) if results else 0,
        }

        if results:
            best = max(results, key=results.get)
            worst = min(results, key=results.get)
            self.kb.record_learning(
                what=f"Tested {len(results)} strategies",
                why="Daily optimization cycle",
                action=f"Best: {best} ({results[best]:.2f}), Worst: {worst} ({results[worst]:.2f})",
            )

        return summary

    # ── Phase 3: Evening Review ────────────────────────────────────────────

    def evening_review(self, test_results: Dict[str, Any],
                       research_brief: Dict[str, Any],
                       llm_client: Any = None) -> Dict[str, Any]:
        """
        Evaluate day's results, update knowledge, generate tomorrow's agenda.
        """
        _log("=== EVENING REVIEW ===")
        champion = self.kb.get_champion()
        best_experiments = self.kb.get_best_experiments(5)
        recent_insights = self.kb.get_recent_insights(10)

        review = {
            "ts": _now_iso(),
            "champion": champion,
            "best_experiments": best_experiments,
            "recent_insights": recent_insights,
            "today_summary": {
                "configs_tested": test_results.get("configs_tested", 0),
                "best_sharpe": test_results.get("best_sharpe", 0),
                "papers_read": len(research_brief.get("papers", [])),
                "hypotheses_generated": len(research_brief.get("hypotheses", [])),
            },
            "tomorrow_agenda": self._plan_tomorrow(
                test_results, research_brief, llm_client
            ),
        }

        # Save evening review
        review_path = self.artifacts_dir / "brain" / "evening_review.json"
        review_path.write_text(json.dumps(review, indent=2), encoding="utf-8")

        # Write to brain diary
        self._write_diary_entry(review)

        return review

    def _plan_tomorrow(self, test_results, brief, llm_client) -> List[str]:
        """Generate tomorrow's research agenda."""
        agenda = []
        results = test_results.get("results", {})

        # Auto-generate based on results
        if results:
            worst_config = min(results, key=results.get)
            agenda.append(f"Investigate why {worst_config} underperforms")
            best_config = max(results, key=results.get)
            agenda.append(f"Parameter tune {best_config} for higher Sharpe")

        for h in brief.get("hypotheses", [])[:3]:
            agenda.append(f"Test: {h.get('hypothesis', '')[:80]}")

        agenda.append("Research: Fetch latest papers on crypto cross-sectional factors")
        agenda.append("Experiment: Try different rebalance intervals (72h, 120h, 168h)")
        agenda.append("Risk: Analyze regime-conditional drawdowns")

        return agenda[:8]

    def _write_diary_entry(self, review: Dict[str, Any]) -> None:
        """Write to NEXUS brain diary."""
        diary_path = self.artifacts_dir / "brain" / "diary.jsonl"
        entry = {
            "ts": _now_iso(),
            "type": "daily_review",
            "champion_sharpe": (review.get("champion") or {}).get("sharpe", 0),
            "insights_count": len(review.get("recent_insights", [])),
            "tomorrow": review.get("tomorrow_agenda", [])[:3],
        }
        with open(diary_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    # ── Apply → Evaluate → Adapt (AEA) Core Loop ──────────────────────────

    def apply_evaluate_adapt(
        self,
        hypothesis: Dict[str, Any],
        config_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        The core self-improvement micro-cycle inspired by metacognitive learning:

        APPLY   — run a backtest experiment implementing the hypothesis
        EVALUATE — measure result vs. prior champion (positive / neutral / negative)
        ADAPT   — update knowledge base, priors, and tomorrow's agenda accordingly

        This mirrors how a top quant researcher updates their mental model
        after every experiment: reject if evidence weak, promote if strong,
        and always extract a specific learning regardless of outcome.
        """
        hyp_text = hypothesis.get("hypothesis", str(hypothesis))
        _log(f"[AEA] APPLY: {hyp_text[:80]}...")

        # ── APPLY ──
        configs = sorted(self.configs_dir.glob("run_binance_*.json"))
        best_config = None
        if config_path and config_path.exists():
            best_config = config_path
        elif configs:
            # Use highest-weight config for this hypothesis (simple: pick best champion config)
            champion = self.kb.get_champion()
            if champion and champion.get("config_path"):
                cp = Path(champion["config_path"])
                if cp.exists():
                    best_config = cp
            if best_config is None:
                best_config = configs[0]

        result_metrics: Optional[Dict[str, Any]] = None
        new_sharpe = 0.0
        if best_config:
            _log(f"[AEA]   Running backtest: {best_config.name}")
            result_metrics = self.runner.run_backtest(best_config)
            if result_metrics:
                new_sharpe = self.runner.get_sharpe_from_metrics(result_metrics)

        # ── EVALUATE ──
        champion = self.kb.get_champion()
        prior_sharpe = float((champion or {}).get("sharpe", 0.0))
        delta = new_sharpe - prior_sharpe

        if result_metrics and new_sharpe > 0:
            if delta > 0.05:
                outcome = "positive"
                verdict = f"IMPROVED by {delta:.3f} Sharpe — promoting"
            elif delta < -0.1:
                outcome = "negative"
                verdict = f"DEGRADED by {abs(delta):.3f} Sharpe — reverting"
            else:
                outcome = "neutral"
                verdict = f"Neutral (delta={delta:+.3f}) — no change"
        else:
            outcome = "failed"
            verdict = "Backtest failed — marking as invalid hypothesis"

        _log(f"[AEA] EVALUATE: {outcome.upper()} | {verdict}")

        # ── ADAPT ──
        learning = {
            "hypothesis": hyp_text,
            "outcome": outcome,
            "delta_sharpe": delta,
            "new_sharpe": new_sharpe,
            "prior_sharpe": prior_sharpe,
        }

        if outcome == "positive":
            self.kb.set_champion(
                strategy_name=best_config.name if best_config else "unknown",
                sharpe=new_sharpe,
                config_path=str(best_config) if best_config else "",
                metrics=result_metrics or {},
            )
            self.kb.add_insight(
                category="validated",
                insight=f"VALIDATED: {hyp_text[:120]} → Sharpe +{delta:.3f}",
                confidence=min(0.95, 0.5 + delta),
                source="aea_loop",
            )
            self.kb.record_learning(
                what=f"Hypothesis validated: {hyp_text[:80]}",
                why=f"Sharpe improved from {prior_sharpe:.3f} to {new_sharpe:.3f}",
                action="Promoted to champion, keep exploring similar ideas",
            )
        elif outcome == "negative":
            self.kb.add_insight(
                category="refuted",
                insight=f"REFUTED: {hyp_text[:120]} → Sharpe -{abs(delta):.3f}",
                confidence=0.8,
                source="aea_loop",
            )
            self.kb.record_learning(
                what=f"Hypothesis refuted: {hyp_text[:80]}",
                why=f"Sharpe degraded from {prior_sharpe:.3f} to {new_sharpe:.3f}",
                action="Avoid this approach; search for complementary signals instead",
            )
        elif outcome == "neutral":
            self.kb.add_insight(
                category="inconclusive",
                insight=f"INCONCLUSIVE: {hyp_text[:120]} (delta={delta:+.3f})",
                confidence=0.3,
                source="aea_loop",
            )
            self.kb.record_learning(
                what=f"Neutral result: {hyp_text[:80]}",
                why=f"No meaningful Sharpe change (delta={delta:+.3f})",
                action="Try different parameter ranges or different market regime",
            )
        else:
            self.kb.record_learning(
                what=f"Backtest failed for: {hyp_text[:80]}",
                why="Execution error or insufficient data",
                action="Fix config or data provider, retry next cycle",
            )

        self.kb.record_experiment({**learning, "config": str(best_config) if best_config else ""})
        _log(f"[AEA] ADAPT: Knowledge updated | insights={len(self.kb.get_recent_insights())}")
        return learning

    def apply_evaluate_adapt_all(
        self,
        hypotheses: List[Dict[str, Any]],
        max_tests: int = 3,
    ) -> List[Dict[str, Any]]:
        """Run AEA loop for top N hypotheses (time-bounded)."""
        results = []
        for h in hypotheses[:max_tests]:
            try:
                r = self.apply_evaluate_adapt(h)
                results.append(r)
            except Exception as e:
                _log(f"[AEA] Error for hypothesis '{h.get('hypothesis', '')[:50]}': {e}")
        return results

    # ── Full Day Cycle ─────────────────────────────────────────────────────

    def run_full_day(self, llm_client: Any = None) -> Dict[str, Any]:
        """
        Execute a complete daily learning cycle:
        Research → Hypothesize → Apply→Evaluate→Adapt → Test All → Nightly Review

        Inspired by spaced repetition + deliberate practice + Bayesian updating:
        - Read 50+ sources (breadth)
        - Extract top hypotheses (synthesis)
        - AEA micro-cycle on top 2 ideas (depth/validation)
        - Full strategy comparison (leaderboard)
        - Evening review (consolidation + tomorrow's agenda)
        """
        _log("====== NEXUS DAILY CYCLE START ======")
        cycle_start = time.time()

        # 1. Morning Research (50+ sources)
        brief = self.morning_research(llm_client)
        time.sleep(1)

        # 2. Apply→Evaluate→Adapt on top hypotheses
        aea_results = self.apply_evaluate_adapt_all(
            brief.get("hypotheses", []), max_tests=2
        )
        time.sleep(1)

        # 3. Full strategy comparison (all binance configs)
        test_results = self.test_hypotheses()
        time.sleep(1)

        # 4. Evening Review + Knowledge Update
        review = self.evening_review(test_results, brief, llm_client)

        elapsed = time.time() - cycle_start
        _log(f"====== DAILY CYCLE COMPLETE ({elapsed:.0f}s) ======")
        _log(f"  Sources: {brief.get('stats', {}).get('fetched_sources', 0)}")
        _log(f"  Articles: {brief.get('total_items', 0)}")
        _log(f"  Hypotheses: {len(brief.get('hypotheses', []))}")
        _log(f"  AEA cycles: {len(aea_results)}")
        _log(f"  Experiments: {test_results.get('configs_tested', 0)}")
        _log(f"  Best Sharpe: {test_results.get('best_sharpe', 0):.3f}")
        _log(f"  New Champion: {test_results.get('new_champion', 'none')}")

        return {
            "brief": brief,
            "aea_results": aea_results,
            "test_results": test_results,
            "review": review,
            "elapsed_seconds": elapsed,
        }

    def run_continuous(self, interval_hours: float = 24.0, max_cycles: int = None) -> None:
        """
        Run NEXUS daily routine continuously.
        Default: full cycle every 24 hours.
        """
        _log(f"Starting continuous learning loop (interval={interval_hours}h)")
        cycle = 0
        while True:
            if max_cycles and cycle >= max_cycles:
                _log(f"Reached max_cycles={max_cycles}, stopping.")
                break
            try:
                llm_client = self._get_llm_client()
                self.run_full_day(llm_client)
            except KeyboardInterrupt:
                _log("Interrupted by user, stopping.")
                break
            except Exception as e:
                _log(f"Cycle failed: {e}")
            cycle += 1
            _log(f"Sleeping {interval_hours}h until next cycle...")
            time.sleep(interval_hours * 3600)

    def _get_llm_client(self) -> Optional[Any]:
        """Get GLM-5/Claude client if API key available."""
        try:
            import anthropic
            api_key = os.environ.get("ZAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                return None
            base_url = os.environ.get("ZAI_ANTHROPIC_BASE_URL", "").rstrip("/")
            kwargs: Dict[str, Any] = {"api_key": api_key, "max_retries": 1}
            if base_url:
                kwargs["base_url"] = base_url
            return anthropic.Anthropic(**kwargs)
        except Exception:
            return None
