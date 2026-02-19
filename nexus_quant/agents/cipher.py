"""
CIPHER: Risk Assessment Agent.

Role: Review current strategy risk metrics (VaR, drawdown, correlation, regime)
and flag any risk concerns. Suggest position limit adjustments.

Uses: claude-sonnet-4-6 via Z.AI gateway (higher accuracy for risk).
"""
from __future__ import annotations

import json
from typing import Any, Dict

from .base import AgentContext, AgentResult, BaseAgent
from .router import extract_json_block, safe_call_llm

_SYSTEM_PROMPT = """\
You are CIPHER, a risk management specialist for algorithmic trading systems.
Your task: review quantitative risk metrics and flag concerns, then recommend position limit adjustments.

Guidelines:
- Severity levels: "ok" (no action needed), "warn" (monitor closely), "critical" (reduce exposure now).
- Be direct and specific. Reference metric values in your flags.
- Do not add commentary outside the JSON block.

Respond with ONLY a JSON block in this exact format:
```json
{
  "risk_flags": ["<flag description>"],
  "recommended_limits": {
    "max_position_pct": <0-1>,
    "max_drawdown_pct": <0-1>
  },
  "severity": "ok" | "warn" | "critical"
}
```
"""

_FALLBACK: Dict[str, Any] = {
    "risk_flags": [],
    "severity": "ok",
    "recommended_limits": {},
}


class CipherAgent(BaseAgent):
    """Risk assessment agent — flags risk concerns and recommends limits."""

    name = "cipher"
    default_model = "claude-sonnet-4-6"

    def run(self, context: AgentContext) -> AgentResult:
        metrics = context.recent_metrics
        regime = context.regime
        wisdom = context.wisdom

        # Pull salient risk fields from metrics
        risk_fields = {
            k: metrics.get(k)
            for k in (
                "max_drawdown", "sharpe", "sortino", "calmar",
                "var_95", "var_99", "win_rate", "total_return",
                "avg_trade_return", "n_trades",
            )
            if metrics.get(k) is not None
        }

        user_prompt = f"""\
## Risk Metrics
```json
{json.dumps(risk_fields, indent=2)}
```

## Full Recent Metrics
```json
{json.dumps(metrics, indent=2)}
```

## Market Regime
```json
{json.dumps(regime, indent=2)}
```

## Wisdom / Accept Rate Context
```json
{json.dumps({k: wisdom.get(k) for k in ("accept_rate", "n_runs", "consecutive_no_accept") if k in wisdom}, indent=2)}
```

Review the above and identify risk concerns. Recommend position limits if any metric breaches typical thresholds \
(e.g. max_drawdown > 20%, sharpe < 0.5, win_rate < 40%). Set severity accordingly.
"""

        raw, fallback_used, error = safe_call_llm(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=self.model,
            max_tokens=1024,
            fallback_result=_FALLBACK,
        )

        if fallback_used:
            parsed = _FALLBACK.copy()
        else:
            parsed = extract_json_block(raw)
            if not parsed or "severity" not in parsed:
                parsed = _FALLBACK.copy()

        # Enforce valid severity
        if parsed.get("severity") not in ("ok", "warn", "critical"):
            parsed["severity"] = "ok"

        return AgentResult(
            agent_name=self.name,
            model_used=self.model,
            raw_response=raw,
            parsed=parsed,
            fallback_used=fallback_used,
            error=error,
        )
