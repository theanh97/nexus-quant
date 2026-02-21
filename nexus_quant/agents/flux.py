"""
FLUX: Decision Gate & Task Prioritization Agent (Phase 3 — Final Arbiter).

Role: Receive ALL Phase 1 + Phase 2 agent outputs, then make the final
go/no-go decision on which proposals to execute.

v1.0 Decision Network upgrade:
- Now sees ATLAS proposals, CIPHER risk flags, AND ECHO validation verdicts
- Acts as the DECISION GATE: blocks risky/overfitted proposals
- Prioritizes approved tasks for the experiment pipeline
- Uses Claude Sonnet 4.6 (upgraded from GLM-5)

Safe fallback: empty task list + conservative trials (system pauses, not proceeds blindly).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from .base import AgentContext, AgentResult, BaseAgent
from .router import extract_json_block, safe_call_llm

_SYSTEM_PROMPT = """\
You are FLUX, the final decision gate for NEXUS algorithmic trading research.

You receive outputs from THREE other agents:
- ATLAS (researcher): proposed strategy experiments
- CIPHER (risk officer): risk severity and flags
- ECHO (QA skeptic): overfitting verdict and proposal reviews

Your task: Make the FINAL decision on what gets executed.

Decision rules:
- If ECHO verdict is "fail" OR CIPHER severity is "critical":
  → BLOCK all proposals, set gate_decision to "block"
- If ECHO verdict is "warn" OR CIPHER severity is "warn":
  → Allow only ECHO-approved proposals, set gate_decision to "partial"
- If ECHO verdict is "pass" AND CIPHER severity is "ok":
  → Allow all proposals, set gate_decision to "approve"

Guidelines:
- next_tasks: list 1-5 tasks in priority order (only APPROVED proposals)
- suggested_trials: how many optimization trials (10-200), reduce if risk/overfit concerns
- gate_decision: "approve" | "partial" | "block"
- blocked_reasons: explain WHY any proposals were blocked
- Be conservative — blocking a bad experiment is better than running it

Respond with ONLY a JSON block in this exact format:
```json
{
  "gate_decision": "approve" | "partial" | "block",
  "blocked_reasons": ["<why blocked, if any>"],
  "next_tasks": [
    {
      "kind": "run" | "improve" | "reflect" | "critique" | "wisdom" | "agent_run" | "experiment",
      "priority": <1-5>,
      "reason": "<one-sentence reason>"
    }
  ],
  "suggested_trials": <int>,
  "escalate_to_human": <true/false>,
  "escalation_reason": "<why human review needed, if escalating>"
}
```
"""

# SAFE fallback: block + escalate when FLUX can't assess
_FALLBACK: Dict[str, Any] = {
    "gate_decision": "block",
    "blocked_reasons": ["FLUX agent unavailable — cannot make decision (API failure)"],
    "next_tasks": [],
    "suggested_trials": 10,
    "escalate_to_human": True,
    "escalation_reason": "FLUX decision gate offline — manual review required",
}


class FluxAgent(BaseAgent):
    """Phase 3 decision gate — approves/blocks proposals based on all agent outputs."""

    name = "flux"
    default_model = "claude-sonnet-4-6"
    _smart_task_type = "EXPERIMENT_DESIGN"

    def run(self, context: AgentContext) -> AgentResult:
        wisdom = context.wisdom
        metrics = context.recent_metrics
        extra = context.extra

        # Phase 1 + Phase 2 results (populated by orchestrator)
        atlas_output = context.phase1_results.get("atlas", {})
        cipher_output = context.phase1_results.get("cipher", {})
        echo_output = context.phase1_results.get("echo", {})

        # Pipeline state
        pending_tasks: List[Dict[str, Any]] = extra.get("pending_tasks") or []
        recent_results: List[Dict[str, Any]] = extra.get("recent_results") or []
        accept_rate = wisdom.get("accept_rate") or metrics.get("accept_rate") or "unknown"
        consecutive_no_accept = wisdom.get("consecutive_no_accept") or 0

        user_prompt = f"""\
## ATLAS Proposals (Phase 1 — Researcher)
```json
{json.dumps(atlas_output.get("proposals", []), indent=2)}
```

## CIPHER Risk Assessment (Phase 1 — Risk Officer)
- Severity: {cipher_output.get("severity", "unknown")}
- Risk flags: {json.dumps(cipher_output.get("risk_flags", []))}
- Recommended limits: {json.dumps(cipher_output.get("recommended_limits", {}))}

## ECHO Validation (Phase 2 — QA Skeptic)
- Verdict: {echo_output.get("verdict", "unknown")}
- Overfit score: {echo_output.get("overfit_score", "?")} / 10
- Concerns: {json.dumps(echo_output.get("concerns", []))}
- Proposal reviews: {json.dumps(echo_output.get("proposal_reviews", []))}
- CIPHER assessment: {echo_output.get("cipher_assessment", "unknown")}
- Dual-model agreement: {echo_output.get("dual_model_agreement", "N/A")}

## Pipeline State
- Pending tasks: {len(pending_tasks)}
- Accept rate: {accept_rate}
- Consecutive non-accepted runs: {consecutive_no_accept}
- Total runs: {wisdom.get("n_runs") or "unknown"}

## Recent Metrics
```json
{json.dumps(metrics, indent=2)}
```

Based on ALL the above agent outputs, make your gate decision. \
Remember: blocking a bad experiment is ALWAYS better than running it.
"""

        raw, fallback_used, error = safe_call_llm(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=self.model,
            max_tokens=1024,
            fallback_result=_FALLBACK,
            smart_task_type=self._smart_task_type,
        )

        if fallback_used:
            parsed = _FALLBACK.copy()
        else:
            parsed = extract_json_block(raw)
            if not parsed or "gate_decision" not in parsed:
                parsed = _FALLBACK.copy()

        # Enforce valid gate_decision
        if parsed.get("gate_decision") not in ("approve", "partial", "block"):
            parsed["gate_decision"] = "block"

        # Ensure suggested_trials is sane
        try:
            parsed["suggested_trials"] = max(10, min(200, int(parsed.get("suggested_trials", 30))))
        except (TypeError, ValueError):
            parsed["suggested_trials"] = 30

        # Ensure escalate fields exist
        parsed.setdefault("escalate_to_human", False)
        parsed.setdefault("escalation_reason", "")
        parsed.setdefault("blocked_reasons", [])

        return AgentResult(
            agent_name=self.name,
            model_used=self.model,
            raw_response=raw,
            parsed=parsed,
            fallback_used=fallback_used,
            error=error,
            phase="decide",
        )
