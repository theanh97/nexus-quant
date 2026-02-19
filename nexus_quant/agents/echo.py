"""
ECHO: QA/Validation Agent.

Role: Review backtest results for signs of overfitting, data leakage,
or cherry-picking. Ask hard questions about robustness.

Uses: MiniMax-Text-01 via Z.AI gateway (falls back to anthropic SDK path).
"""
from __future__ import annotations

import json
from typing import Any, Dict

from .base import AgentContext, AgentResult, BaseAgent
from .router import extract_json_block, safe_call_llm

_SYSTEM_PROMPT = """\
You are ECHO, a skeptical quantitative analyst who reviews backtest results critically.
Your task: identify potential issues including overfitting, data leakage, look-ahead bias, \
survivorship bias, regime overfitting, or cherry-picked evaluation windows.

Guidelines:
- overfit_score ranges 0 (no concern) to 10 (severe overfitting).
- verdict: "pass" (results look robust), "warn" (concerns present), "fail" (results not trustworthy).
- Raise hard, specific questions for the researcher to answer.
- Do not add commentary outside the JSON block.

Respond with ONLY a JSON block in this exact format:
```json
{
  "verdict": "pass" | "warn" | "fail",
  "concerns": ["<specific concern>"],
  "questions": ["<hard question for researcher>"],
  "overfit_score": <0-10>
}
```
"""

_FALLBACK: Dict[str, Any] = {
    "verdict": "pass",
    "concerns": [],
    "questions": [],
    "overfit_score": 0,
}


class EchoAgent(BaseAgent):
    """QA/validation agent — flags overfitting and data quality concerns."""

    name = "echo"
    default_model = "MiniMax-Text-01"

    def run(self, context: AgentContext) -> AgentResult:
        metrics = context.recent_metrics
        wisdom = context.wisdom
        best_params = context.best_params or {}

        # Summarize optimization intensity as a proxy for overfit risk
        n_runs = wisdom.get("n_runs") or metrics.get("n_runs") or 0
        accept_rate = wisdom.get("accept_rate")
        consecutive_no_accept = wisdom.get("consecutive_no_accept") or 0

        user_prompt = f"""\
## Backtest / Run Metrics
```json
{json.dumps(metrics, indent=2)}
```

## Optimization History Context
- Total optimization runs so far: {n_runs}
- Accept rate: {accept_rate}
- Consecutive non-accepted runs: {consecutive_no_accept}

## Best Parameters Found
```json
{json.dumps(best_params, indent=2)}
```

## Market Regime at Time of Test
```json
{json.dumps(context.regime, indent=2)}
```

Review the above backtest results with extreme skepticism. \
Score overfitting risk, list concerns, and raise specific hard questions.
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
            if not parsed or "verdict" not in parsed:
                parsed = _FALLBACK.copy()

        # Enforce valid verdict
        if parsed.get("verdict") not in ("pass", "warn", "fail"):
            parsed["verdict"] = "pass"

        # Clamp overfit_score to [0, 10]
        try:
            parsed["overfit_score"] = max(0, min(10, int(parsed.get("overfit_score", 0))))
        except (TypeError, ValueError):
            parsed["overfit_score"] = 0

        return AgentResult(
            agent_name=self.name,
            model_used=self.model,
            raw_response=raw,
            parsed=parsed,
            fallback_used=fallback_used,
            error=error,
        )
