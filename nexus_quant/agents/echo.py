"""
ECHO: QA/Validation Agent (Phase 2 — Cross-Validator).

Role: Review backtest results AND Phase 1 agent outputs (ATLAS proposals +
CIPHER risk flags) for signs of overfitting, data leakage, or risky proposals.

v1.0 Decision Network upgrade:
- Now receives Phase 1 outputs (ATLAS + CIPHER) for cross-validation
- Uses DUAL-MODEL consensus: Claude Sonnet (primary) + Gemini Pro (secondary)
- Safe fallback: "warn" (not "pass") when API fails
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .base import AgentContext, AgentResult, BaseAgent
from .router import extract_json_block, safe_call_llm, call_llm_dual

_SYSTEM_PROMPT = """\
You are ECHO, a skeptical quantitative analyst who reviews backtest results \
AND strategy proposals from other agents.

Your task has two parts:
1. BACKTEST VALIDATION: Identify overfitting, data leakage, look-ahead bias, \
survivorship bias, regime overfitting, or cherry-picked evaluation windows.
2. PROPOSAL REVIEW: Evaluate whether ATLAS's proposals are sound and whether \
CIPHER's risk assessment is thorough enough. Flag any proposals that could \
lead to overfitting or excessive risk.

Guidelines:
- overfit_score ranges 0 (no concern) to 10 (severe overfitting).
- verdict: "pass" (results look robust), "warn" (concerns present), "fail" (not trustworthy).
- For each ATLAS proposal, indicate "approve", "warn", or "reject" with reason.
- Raise hard, specific questions for the researcher to answer.
- Be adversarial — your job is to PREVENT bad experiments from running.

Respond with ONLY a JSON block in this exact format:
```json
{
  "verdict": "pass" | "warn" | "fail",
  "concerns": ["<specific concern>"],
  "questions": ["<hard question for researcher>"],
  "overfit_score": <0-10>,
  "proposal_reviews": [
    {
      "proposal_name": "<name from ATLAS>",
      "decision": "approve" | "warn" | "reject",
      "reason": "<why>"
    }
  ],
  "cipher_assessment": "adequate" | "insufficient" | "overreacting"
}
```
"""

# SAFE fallback: "warn" not "pass" — system should NOT proceed blindly when API fails
_FALLBACK: Dict[str, Any] = {
    "verdict": "warn",
    "concerns": ["ECHO agent unavailable — cannot validate results (API failure)"],
    "questions": ["Manual review required: ECHO could not assess these results"],
    "overfit_score": 5,
    "proposal_reviews": [],
    "cipher_assessment": "insufficient",
}


class EchoAgent(BaseAgent):
    """QA/validation agent — Phase 2 cross-validator with dual-model consensus."""

    name = "echo"
    default_model = "claude-sonnet-4-6"
    _smart_task_type = "DATA_ANALYSIS"

    def __init__(self, model: Optional[str] = None, use_dual_model: bool = True) -> None:
        super().__init__(model=model)
        self.use_dual_model = use_dual_model

    def run(self, context: AgentContext) -> AgentResult:
        metrics = context.recent_metrics
        wisdom = context.wisdom
        best_params = context.best_params or {}

        n_runs = wisdom.get("n_runs") or metrics.get("n_runs") or 0
        accept_rate = wisdom.get("accept_rate")
        consecutive_no_accept = wisdom.get("consecutive_no_accept") or 0

        # Phase 1 results (populated by orchestrator before calling ECHO)
        atlas_output = context.phase1_results.get("atlas", {})
        cipher_output = context.phase1_results.get("cipher", {})

        atlas_section = ""
        if atlas_output:
            proposals = atlas_output.get("proposals", [])
            atlas_section = f"""
## ATLAS Proposals (Phase 1 — review these critically)
```json
{json.dumps(proposals, indent=2)}
```
"""

        cipher_section = ""
        if cipher_output:
            cipher_section = f"""
## CIPHER Risk Assessment (Phase 1 — is this thorough enough?)
- Severity: {cipher_output.get('severity', 'unknown')}
- Risk flags: {json.dumps(cipher_output.get('risk_flags', []))}
- Recommended limits: {json.dumps(cipher_output.get('recommended_limits', {}))}
"""

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
{atlas_section}{cipher_section}
Review the above with EXTREME SKEPTICISM. Score overfitting risk, evaluate each \
ATLAS proposal, assess whether CIPHER's risk analysis is adequate, and raise \
specific hard questions. Your job is to PREVENT bad experiments from running.
"""

        if self.use_dual_model:
            return self._run_dual_model(user_prompt)
        else:
            return self._run_single_model(user_prompt)

    def _run_single_model(self, user_prompt: str) -> AgentResult:
        raw, fallback_used, error = safe_call_llm(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=self.model,
            max_tokens=1536,
            fallback_result=_FALLBACK,
            smart_task_type=self._smart_task_type,
        )

        parsed = self._parse_response(raw, fallback_used)

        return AgentResult(
            agent_name=self.name,
            model_used=self.model,
            raw_response=raw,
            parsed=parsed,
            fallback_used=fallback_used,
            error=error,
            phase="validate",
        )

    def _run_dual_model(self, user_prompt: str) -> AgentResult:
        """Run Claude + Gemini, then merge verdicts for higher confidence."""
        primary_text, secondary_text, any_fallback, error = call_llm_dual(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            primary_model=self.model,
            secondary_model="gemini-3-flash-preview",
            max_tokens=1536,
            fallback_result=_FALLBACK,
        )

        primary_parsed = self._parse_response(primary_text, False)
        secondary_parsed = self._parse_response(secondary_text, False)

        # Merge verdicts: most conservative wins
        merged = self._merge_dual_verdicts(primary_parsed, secondary_parsed)
        models_used = f"{self.model}+gemini-3-flash-preview"

        return AgentResult(
            agent_name=self.name,
            model_used=models_used,
            raw_response=f"--- PRIMARY ({self.model}) ---\n{primary_text}\n\n--- SECONDARY (gemini-3-flash-preview) ---\n{secondary_text}",
            parsed=merged,
            fallback_used=any_fallback,
            error=error,
            phase="validate",
        )

    def _merge_dual_verdicts(
        self, primary: Dict[str, Any], secondary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge two model verdicts — most conservative wins for safety."""
        verdict_rank = {"fail": 3, "warn": 2, "pass": 1}
        p_rank = verdict_rank.get(primary.get("verdict", "warn"), 2)
        s_rank = verdict_rank.get(secondary.get("verdict", "warn"), 2)

        # Most conservative verdict wins
        if p_rank >= s_rank:
            final_verdict = primary.get("verdict", "warn")
        else:
            final_verdict = secondary.get("verdict", "warn")

        # If models DISAGREE, auto-escalate to at least "warn"
        if primary.get("verdict") != secondary.get("verdict"):
            if final_verdict == "pass":
                final_verdict = "warn"

        # Merge overfit scores: take the higher (more pessimistic)
        p_score = primary.get("overfit_score", 5)
        s_score = secondary.get("overfit_score", 5)
        final_score = max(p_score, s_score)

        # Union of concerns and questions
        concerns = list(set(primary.get("concerns", []) + secondary.get("concerns", [])))
        questions = list(set(primary.get("questions", []) + secondary.get("questions", [])))

        # Proposal reviews: use primary as base, note disagreements
        proposal_reviews = primary.get("proposal_reviews", [])
        secondary_reviews = {
            r.get("proposal_name"): r
            for r in secondary.get("proposal_reviews", [])
        }
        for review in proposal_reviews:
            name = review.get("proposal_name", "")
            if name in secondary_reviews:
                sec_decision = secondary_reviews[name].get("decision", "warn")
                if sec_decision != review.get("decision"):
                    # Models disagree on this proposal — escalate
                    review["decision"] = "warn"
                    review["reason"] = (
                        f"{review.get('reason', '')} "
                        f"[DUAL-MODEL DISAGREEMENT: secondary says '{sec_decision}']"
                    )

        return {
            "verdict": final_verdict,
            "concerns": concerns,
            "questions": questions,
            "overfit_score": final_score,
            "proposal_reviews": proposal_reviews,
            "cipher_assessment": primary.get("cipher_assessment", "insufficient"),
            "dual_model_agreement": primary.get("verdict") == secondary.get("verdict"),
        }

    @staticmethod
    def _parse_response(raw: str, fallback_used: bool) -> Dict[str, Any]:
        if fallback_used:
            return _FALLBACK.copy()

        parsed = extract_json_block(raw)
        if not parsed or "verdict" not in parsed:
            return _FALLBACK.copy()

        # Enforce valid verdict
        if parsed.get("verdict") not in ("pass", "warn", "fail"):
            parsed["verdict"] = "warn"

        # Clamp overfit_score to [0, 10]
        try:
            parsed["overfit_score"] = max(0, min(10, int(parsed.get("overfit_score", 5))))
        except (TypeError, ValueError):
            parsed["overfit_score"] = 5

        # Ensure proposal_reviews exists
        parsed.setdefault("proposal_reviews", [])
        parsed.setdefault("cipher_assessment", "adequate")

        return parsed
