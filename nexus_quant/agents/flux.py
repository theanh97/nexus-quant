"""
FLUX: Operations & Task Prioritization Agent.

Role: Look at the current experiment queue, available results, and resource state.
Decide which experiments to run next, and in what order.

Uses: GLM-5 via Z.AI (fast, cost-efficient for scheduling tasks).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from .base import AgentContext, AgentResult, BaseAgent
from .router import extract_json_block, safe_call_llm

_SYSTEM_PROMPT = """\
You are FLUX, an operations coordinator for algorithmic trading research.
Your task: review the current experiment pipeline and decide what to prioritize next.

Guidelines:
- Consider pending task count, recent accept rate, and any unresolved risk flags.
- next_tasks should list 1-5 tasks in priority order (1 = highest).
- suggested_trials: how many optimization trials to run in the next improve cycle (10-200).
- Do not add commentary outside the JSON block.

Respond with ONLY a JSON block in this exact format:
```json
{
  "next_tasks": [
    {
      "kind": "run" | "improve" | "reflect" | "critique" | "wisdom" | "agent_run" | "experiment",
      "priority": <1-5>,
      "reason": "<one-sentence reason>"
    }
  ],
  "suggested_trials": <int>
}
```
"""

_FALLBACK: Dict[str, Any] = {
    "next_tasks": [],
    "suggested_trials": 30,
}


class FluxAgent(BaseAgent):
    """Operations agent — prioritizes experiments and manages research pipeline."""

    name = "flux"
    default_model = "glm-4-long"

    def run(self, context: AgentContext) -> AgentResult:
        wisdom = context.wisdom
        metrics = context.recent_metrics
        extra = context.extra

        # Gather pipeline state from context.extra (populated by caller if available)
        pending_tasks: List[Dict[str, Any]] = extra.get("pending_tasks") or []
        recent_results: List[Dict[str, Any]] = extra.get("recent_results") or []
        accept_rate = wisdom.get("accept_rate") or metrics.get("accept_rate") or "unknown"
        consecutive_no_accept = wisdom.get("consecutive_no_accept") or 0

        user_prompt = f"""\
## Pipeline State
- Pending tasks ({len(pending_tasks)} total):
```json
{json.dumps(pending_tasks[:10], indent=2)}
```
- Recent results ({len(recent_results)} available):
```json
{json.dumps(recent_results[:5], indent=2)}
```

## Optimization Health
- Accept rate: {accept_rate}
- Consecutive non-accepted runs: {consecutive_no_accept}
- Total runs: {wisdom.get("n_runs") or "unknown"}

## Recent Metrics
```json
{json.dumps(metrics, indent=2)}
```

## Regime
```json
{json.dumps(context.regime, indent=2)}
```

Based on the pipeline state above, decide which tasks to prioritize next \
and recommend how many trials to run in the next improve cycle.
"""

        raw, fallback_used, error = safe_call_llm(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=self.model,
            max_tokens=768,
            fallback_result=_FALLBACK,
        )

        if fallback_used:
            parsed = _FALLBACK.copy()
        else:
            parsed = extract_json_block(raw)
            if not parsed or "next_tasks" not in parsed:
                parsed = _FALLBACK.copy()

        # Ensure suggested_trials is a sane integer
        try:
            parsed["suggested_trials"] = max(10, min(200, int(parsed.get("suggested_trials", 30))))
        except (TypeError, ValueError):
            parsed["suggested_trials"] = 30

        return AgentResult(
            agent_name=self.name,
            model_used=self.model,
            raw_response=raw,
            parsed=parsed,
            fallback_used=fallback_used,
            error=error,
        )
