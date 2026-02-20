"""
Synthesis: Decision Gate Logic for the 3-Phase Pipeline.

This is NOT an LLM agent — it's deterministic logic that enforces
the Decision Network rules BEFORE any task is queued.

Rules:
- BLOCK all tasks if ECHO verdict="fail" OR CIPHER severity="critical"
- REDUCE scope if ECHO verdict="warn" OR CIPHER severity="warn"
- APPROVE all if ECHO verdict="pass" AND CIPHER severity="ok"
- ALWAYS block if FLUX gate_decision="block"
- Escalate to human when agents disagree significantly

This prevents the old problem where ECHO could say "fail" but tasks
would still run anyway.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("nexus.synthesis")


class GateDecision:
    """Result of the synthesis decision gate."""
    APPROVE = "approve"
    PARTIAL = "partial"
    BLOCK = "block"

    def __init__(
        self,
        decision: str,
        approved_proposals: List[Dict[str, Any]],
        blocked_proposals: List[Dict[str, Any]],
        reasons: List[str],
        escalate_to_human: bool = False,
        escalation_reason: str = "",
        suggested_trials: int = 30,
        confidence: float = 0.5,
    ):
        self.decision = decision
        self.approved_proposals = approved_proposals
        self.blocked_proposals = blocked_proposals
        self.reasons = reasons
        self.escalate_to_human = escalate_to_human
        self.escalation_reason = escalation_reason
        self.suggested_trials = suggested_trials
        self.confidence = confidence
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "approved_count": len(self.approved_proposals),
            "blocked_count": len(self.blocked_proposals),
            "approved_proposals": self.approved_proposals,
            "blocked_proposals": self.blocked_proposals,
            "reasons": self.reasons,
            "escalate_to_human": self.escalate_to_human,
            "escalation_reason": self.escalation_reason,
            "suggested_trials": self.suggested_trials,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


def run_decision_gate(
    atlas_output: Dict[str, Any],
    cipher_output: Dict[str, Any],
    echo_output: Dict[str, Any],
    flux_output: Dict[str, Any],
) -> GateDecision:
    """
    Deterministic decision gate that synthesizes all agent outputs.

    This is the CORE of the Decision Network — it enforces the rules
    that prevent bad experiments from running.
    """
    reasons: List[str] = []
    proposals = atlas_output.get("proposals", [])

    cipher_severity = cipher_output.get("severity", "warn")
    echo_verdict = echo_output.get("verdict", "warn")
    echo_overfit_score = echo_output.get("overfit_score", 5)
    flux_gate = flux_output.get("gate_decision", "block")
    echo_proposal_reviews = {
        r.get("proposal_name", ""): r
        for r in echo_output.get("proposal_reviews", [])
    }

    # ── Rule 1: Hard block conditions ──
    if echo_verdict == "fail" or cipher_severity == "critical":
        block_reason = []
        if echo_verdict == "fail":
            block_reason.append(f"ECHO verdict=fail (overfit_score={echo_overfit_score}/10)")
        if cipher_severity == "critical":
            block_reason.append(f"CIPHER severity=critical: {cipher_output.get('risk_flags', [])}")
        reasons.extend(block_reason)
        logger.warning("Decision gate: BLOCK — %s", "; ".join(block_reason))
        return GateDecision(
            decision=GateDecision.BLOCK,
            approved_proposals=[],
            blocked_proposals=proposals,
            reasons=reasons,
            escalate_to_human=True,
            escalation_reason=f"Hard block: {'; '.join(block_reason)}",
            suggested_trials=10,
            confidence=0.9,
        )

    # ── Rule 2: FLUX explicitly blocks ──
    if flux_gate == "block":
        flux_reasons = flux_output.get("blocked_reasons", ["FLUX decision: block"])
        reasons.extend(flux_reasons)
        logger.warning("Decision gate: BLOCK (FLUX) — %s", flux_reasons)
        return GateDecision(
            decision=GateDecision.BLOCK,
            approved_proposals=[],
            blocked_proposals=proposals,
            reasons=reasons,
            escalate_to_human=flux_output.get("escalate_to_human", True),
            escalation_reason=flux_output.get("escalation_reason", "FLUX blocked all proposals"),
            suggested_trials=flux_output.get("suggested_trials", 10),
            confidence=0.8,
        )

    # ── Rule 3: Partial — warnings present, filter proposals ──
    if echo_verdict == "warn" or cipher_severity == "warn" or flux_gate == "partial":
        approved = []
        blocked = []
        for p in proposals:
            p_name = p.get("name", "")
            review = echo_proposal_reviews.get(p_name, {})
            review_decision = review.get("decision", "warn")

            if review_decision == "reject":
                blocked.append(p)
                reasons.append(f"Blocked '{p_name}': ECHO rejected — {review.get('reason', '?')}")
            elif review_decision == "warn" and echo_overfit_score >= 7:
                blocked.append(p)
                reasons.append(f"Blocked '{p_name}': ECHO warns + high overfit score ({echo_overfit_score}/10)")
            else:
                approved.append(p)

        if not approved and proposals:
            reasons.append("All proposals blocked by ECHO reviews under warn conditions")

        # Reduce trials when under warning conditions
        base_trials = flux_output.get("suggested_trials", 30)
        reduced_trials = max(10, base_trials // 2)

        logger.info("Decision gate: PARTIAL — approved=%d, blocked=%d", len(approved), len(blocked))
        return GateDecision(
            decision=GateDecision.PARTIAL if approved else GateDecision.BLOCK,
            approved_proposals=approved,
            blocked_proposals=blocked,
            reasons=reasons,
            escalate_to_human=echo_overfit_score >= 8,
            escalation_reason=f"High overfit score ({echo_overfit_score}/10)" if echo_overfit_score >= 8 else "",
            suggested_trials=reduced_trials,
            confidence=0.6,
        )

    # ── Rule 4: All clear — approve all ──
    reasons.append("All agents agree: ECHO=pass, CIPHER=ok, FLUX=approve")
    logger.info("Decision gate: APPROVE — %d proposals", len(proposals))
    return GateDecision(
        decision=GateDecision.APPROVE,
        approved_proposals=proposals,
        blocked_proposals=[],
        reasons=reasons,
        escalate_to_human=False,
        suggested_trials=flux_output.get("suggested_trials", 30),
        confidence=0.9,
    )
