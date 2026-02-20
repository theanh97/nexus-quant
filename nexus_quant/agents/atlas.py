"""
ATLAS: Strategy Research Agent.

Role: Analyze current strategy performance and propose new factor combinations,
parameter adjustments, or entirely new strategy configs to try.

Uses: GLM-5 (ZhipuAI) via Z.AI gateway
"""
from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional

from .base import AgentContext, AgentResult, BaseAgent
from .router import extract_json_block, safe_call_llm

_SYSTEM_PROMPT = """\
You are ATLAS, a quantitative research analyst specializing in crypto perpetual futures.
Your task: analyze strategy performance data and propose specific, actionable improvements.

Guidelines:
- Focus on parameter tuning, new signal combinations, and regime-adaptive adjustments.
- Each proposal must include a concrete config_overrides dict (valid JSON key-value pairs).
- Prioritize proposals by expected impact (1 = highest priority, 5 = lowest).
- Be concise and precise. Do not add commentary outside the JSON block.

Respond with ONLY a JSON block in this exact format:
```json
{
  "proposals": [
    {
      "name": "<short name>",
      "rationale": "<one-sentence rationale>",
      "config_overrides": {"param_key": value},
      "priority": <1-5>
    }
  ]
}
```
"""


def _build_fallback(best_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return top 2 random perturbations of best_params as fallback proposals."""
    proposals: List[Dict[str, Any]] = []
    if not best_params:
        return {"proposals": proposals}

    numeric_keys = [k for k, v in best_params.items() if isinstance(v, (int, float))]
    random.shuffle(numeric_keys)

    for i, key in enumerate(numeric_keys[:2]):
        val = best_params[key]
        # ±10% perturbation
        delta = val * 0.10
        new_val = round(val + (delta if i % 2 == 0 else -delta), 6)
        proposals.append({
            "name": f"perturb_{key}",
            "rationale": f"Deterministic ±10% perturbation of {key} from best_params.",
            "config_overrides": {key: new_val},
            "priority": i + 1,
        })
    return {"proposals": proposals}


class AtlasAgent(BaseAgent):
    """Strategy research agent — proposes parameter / factor improvements."""

    name = "atlas"
    default_model = "gemini-3-pro-preview"
    _smart_task_type = "STRATEGY_RESEARCH"

    def run(self, context: AgentContext) -> AgentResult:
        fallback_parsed = _build_fallback(context.best_params)

        metrics_str = json.dumps(context.recent_metrics, indent=2)
        best_str = json.dumps(context.best_params or {}, indent=2)
        wisdom_summary = json.dumps(
            {k: context.wisdom.get(k) for k in ("accept_rate", "consecutive_no_accept", "n_runs") if k in context.wisdom},
            indent=2,
        )

        user_prompt = f"""\
## Current Strategy Metrics
```json
{metrics_str}
```

## Best Known Parameters
```json
{best_str}
```

## Wisdom Summary
```json
{wisdom_summary}
```

## Regime
```json
{json.dumps(context.regime, indent=2)}
```

Based on the above, propose 2-4 strategy improvements with concrete config_overrides.
"""

        raw, fallback_used, error = safe_call_llm(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=self.model,
            max_tokens=1024,
            fallback_result=fallback_parsed,
        )

        if fallback_used:
            parsed = fallback_parsed
        else:
            parsed = extract_json_block(raw)
            if not parsed or "proposals" not in parsed:
                parsed = fallback_parsed

        return AgentResult(
            agent_name=self.name,
            model_used=self.model,
            raw_response=raw,
            parsed=parsed,
            fallback_used=fallback_used,
            error=error,
            phase="generate",
        )
