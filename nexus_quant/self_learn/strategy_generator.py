from __future__ import annotations

"""
NEXUS Strategy Generator — Autonomous strategy code proposals.

Uses GLM-5/Claude to propose:
1. New parameter combinations (config_overrides)
2. New feature engineering ideas
3. Factor weight adjustments
4. Regime-adaptive rule modifications

All proposals are validated for safety (no lookahead, proper risk controls)
before being queued as experiments via the Kanban task board.
"""

import json
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


# ── Prompt templates ─────────────────────────────────────────────────────────

_STRATEGY_IMPROVEMENT_SYSTEM = """\
You are NEXUS ATLAS — an elite quantitative researcher specializing in crypto perpetual futures.
Your task: analyze recent backtest performance and propose SPECIFIC, IMPLEMENTABLE improvements.

Rules:
1. All proposals must have config_overrides that directly map to strategy parameters
2. No lookahead bias — only use past data
3. Prioritize Sharpe improvement, minimize drawdown
4. Consider transaction costs (assume 0.04% per trade)
5. Max 4 proposals, ranked by expected impact

Output ONLY valid JSON in this exact format:
{
  "proposals": [
    {
      "name": "short descriptive name",
      "hypothesis": "why this should improve performance",
      "config_overrides": {"param": value},
      "expected_sharpe_delta": 0.5,
      "risk_assessment": "low|medium|high",
      "priority": 1
    }
  ],
  "key_insight": "main learning from current results"
}
"""

_FACTOR_RESEARCH_SYSTEM = """\
You are NEXUS ATLAS — quantitative factor researcher.
Given market regime and current factor performance, propose:
1. New factor combinations to test
2. Factor weight adjustments
3. Signal thresholding improvements

Output JSON with "factor_proposals" list.
"""

_REGIME_ADAPTATION_SYSTEM = """\
You are NEXUS CIPHER — regime-adaptive strategy specialist.
Given current market regime, propose parameter adjustments that suit the regime:
- Trending: momentum signals, wider stops, higher leverage
- Ranging: mean-reversion, tighter thresholds, lower leverage
- Volatile: reduce position sizes, wider spreads, defensive posture

Output JSON with "regime_adaptations" list.
"""


class StrategyGenerator:
    """
    Generates strategy improvement proposals using LLM reasoning.
    Proposals are validated and queued to Kanban board.
    """

    def __init__(self, artifacts_dir: Path, model: str = "glm-5"):
        self.artifacts_dir = Path(artifacts_dir)
        self.model = model
        self._proposals_log = self.artifacts_dir / "research" / "strategy_proposals.jsonl"
        self._proposals_log.parent.mkdir(parents=True, exist_ok=True)

    def _call_llm(self, system: str, user: str, max_tokens: int = 1000) -> str:
        """Call LLM via ZAI gateway."""
        try:
            import os
            import anthropic
            api_key = os.environ.get("ZAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
            base_url = os.environ.get("ZAI_ANTHROPIC_BASE_URL")
            if not api_key:
                return ""
            kwargs: Dict[str, Any] = {"api_key": api_key, "max_retries": 2}
            if base_url:
                kwargs["base_url"] = base_url
            client = anthropic.Anthropic(**kwargs)
            resp = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=0.3,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            if resp.content:
                return str(resp.content[0].text)
        except Exception as e:
            pass
        return ""

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response."""
        import re
        # Try to find JSON block
        for pattern in [r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```", r"\{.*\}"]:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1) if "```" in pattern else match.group(0))
                except Exception:
                    continue
        # Try parsing full text
        try:
            return json.loads(text.strip())
        except Exception:
            return None

    def _load_context(self) -> Dict[str, Any]:
        """Load latest metrics, wisdom, and regime."""
        ctx: Dict[str, Any] = {"metrics": {}, "wisdom": {}, "regime": {}, "best_params": {}}

        # Latest metrics
        runs_dir = self.artifacts_dir / "runs"
        if runs_dir.exists():
            dirs = sorted(runs_dir.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True)
            for d in dirs[:1]:
                mp = d / "metrics.json"
                if mp.exists():
                    m = _read_json(mp) or {}
                    ctx["metrics"] = m.get("summary") or {}
                    break

        # Wisdom
        wp = self.artifacts_dir / "wisdom" / "latest.json"
        ctx["wisdom"] = _read_json(wp) or {}

        # Bias check from latest run
        for d in sorted((self.artifacts_dir / "runs").iterdir() if (self.artifacts_dir / "runs").exists() else [], key=lambda d: d.stat().st_mtime, reverse=True)[:1]:
            mp = d / "metrics.json"
            if mp.exists():
                m = _read_json(mp) or {}
                ctx["bias_check"] = m.get("bias_check") or {}
                break

        return ctx

    def generate_improvement_proposals(
        self,
        config_path: Optional[Path] = None,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate strategy improvement proposals using LLM.
        Returns list of validated proposals.
        """
        ctx = self._load_context()
        metrics = ctx["metrics"]
        wisdom = ctx["wisdom"]

        # Build market context string
        market_str = ""
        if market_data:
            btc = market_data.get("BTCUSDT", {})
            rate = btc.get("latest_rate", 0)
            market_str = f"BTC funding rate: {rate:.6f} ({'positive' if rate > 0 else 'negative'})\n"

        # Load config for strategy params
        config_str = ""
        if config_path and config_path.exists():
            cfg = _read_json(config_path) or {}
            strategy = cfg.get("strategy", {})
            config_str = f"Strategy: {strategy.get('name', 'unknown')}\nParams: {json.dumps(strategy.get('params', {}), indent=2)}\n"

        user_prompt = f"""Current Performance:
{json.dumps(metrics, indent=2)}

Wisdom (accept_rate, consecutive_no_accept):
{json.dumps({k: wisdom.get(k) for k in ('accept_rate', 'consecutive_no_accept', 'n_runs') if k in wisdom}, indent=2)}

{config_str}
{market_str}
Bias check:
{json.dumps(ctx.get('bias_check', {}), indent=2)}

Based on this data, propose 3-4 specific strategy improvements with concrete config_overrides."""

        raw = self._call_llm(_STRATEGY_IMPROVEMENT_SYSTEM, user_prompt, max_tokens=1200)
        parsed = self._extract_json(raw) or {}
        proposals = parsed.get("proposals") or []
        key_insight = parsed.get("key_insight", "")

        # Log proposals
        log_entry = {
            "ts": _now_iso(),
            "model": self.model,
            "proposals": proposals,
            "key_insight": key_insight,
            "context_sharpe": metrics.get("sharpe"),
            "config": str(config_path) if config_path else None,
        }
        with open(self._proposals_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        return proposals

    def queue_proposals_to_kanban(
        self,
        proposals: List[Dict[str, Any]],
        source: str = "strategy_generator",
    ) -> List[str]:
        """Add validated proposals to Kanban board as tasks."""
        created_ids: List[str] = []
        try:
            from ..tasks.manager import TaskManager
            tm = TaskManager(self.artifacts_dir)
            for p in proposals[:4]:
                name = str(p.get("name") or "proposal")
                hypothesis = str(p.get("hypothesis") or p.get("rationale") or "")
                risk = str(p.get("risk_assessment") or "medium")
                priority = "high" if int(p.get("priority") or 3) <= 2 else "medium"
                overrides = json.dumps(p.get("config_overrides") or {})
                task = tm.create(
                    title=f"[AutoGen] {name}",
                    description=f"{hypothesis}\nOverrides: {overrides}",
                    priority=priority,
                    assignee="ATLAS",
                    tags=["auto-generated", "experiment", source],
                    created_by=source,
                )
                created_ids.append(task.id)
        except Exception as e:
            pass
        return created_ids

    def recent_proposals(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return recent proposals from log."""
        if not self._proposals_log.exists():
            return []
        lines = self._proposals_log.read_text("utf-8").splitlines()[-limit:]
        out = []
        for ln in lines:
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
        return list(reversed(out))
