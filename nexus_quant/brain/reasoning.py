"""
NEXUS Reasoning Engine - Uses LLM (GLM-5 via ZAI) to reason about
research questions, explain decisions, and generate diary entries.
"""
from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional


def _call_llm(system_prompt: str, user_message: str, max_tokens: int = 800) -> str:
    """Call LLM via ZAI gateway (GLM-5). Falls back to deterministic summary."""
    api_key = os.environ.get("ZAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    base_url = os.environ.get("ZAI_ANTHROPIC_BASE_URL", "https://api.z.ai/api/anthropic")
    model = os.environ.get("ZAI_DEFAULT_MODEL", "glm-5")

    if not api_key:
        return f"[NEXUS deterministic mode] {user_message[:200]}"

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return resp.content[0].text if resp.content else ""
    except Exception as e:
        return f"[LLM error: {e}]"


def answer_question(
    question: str,
    context: Dict[str, Any],
    persona: str = "",
) -> str:
    """NEXUS answers a user question about its research, using full context."""
    system = (
        f"{persona}\n\n"
        "You have access to your recent research results. "
        "Answer concisely and accurately. If you are uncertain, say so. "
        "Format numbers to 3 decimal places. Use bullet points for lists."
    )
    ctx_str = json.dumps(
        {
            "recent_metrics": context.get("metrics") or {},
            "regime": context.get("regime") or {},
            "active_goals": context.get("goals") or [],
            "last_diary": context.get("last_diary") or {},
            "wisdom_summary": str(context.get("wisdom") or {})[:500],
        },
        indent=2,
    )
    user_msg = f"Context:\n{ctx_str}\n\nQuestion: {question}"
    return _call_llm(system, user_msg, max_tokens=600)


def generate_diary_synthesis(
    experiments_run: int,
    metrics_summary: Dict[str, Any],
    ledger_events: List[Dict[str, Any]],
    goals_summary: str,
    persona: str = "",
) -> Dict[str, Any]:
    """Generate diary learnings + next plans using LLM."""
    system = (
        f"{persona}\n\n"
        "You are writing your own research diary. Be honest, insightful, and specific. "
        "Focus on what the data actually shows. Do not hallucinate metrics."
    )

    recent_learns = [e for e in ledger_events if e.get("kind") == "self_learn"][-5:]
    accepted = [e for e in recent_learns if (e.get("payload") or {}).get("accepted")]

    user_msg = (
        f"Today I ran {experiments_run} experiments.\n"
        f"Best metrics: {json.dumps(metrics_summary, indent=2)}\n"
        f"Self-learning: {len(recent_learns)} attempts, {len(accepted)} accepted.\n"
        f"Goals: {goals_summary}\n\n"
        "Write a JSON response with exactly these keys:\n"
        '{"key_learnings": ["..."], "next_plans": ["..."], "mood": "focused|exploring|concerned|satisfied"}\n'
        "key_learnings: 2-3 specific insights from today's data.\n"
        "next_plans: 2-3 concrete research actions for next cycle.\n"
        "mood: single word reflecting current research state."
    )

    raw = _call_llm(system, user_msg, max_tokens=400)

    try:
        import re
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            parsed = json.loads(m.group())
            return {
                "key_learnings": parsed.get("key_learnings", [raw[:200]]),
                "next_plans": parsed.get("next_plans", ["Continue experiments"]),
                "mood": parsed.get("mood", "focused"),
                "raw": raw,
            }
    except Exception:
        pass

    return {
        "key_learnings": [raw[:300]] if raw else ["Analysis in progress."],
        "next_plans": ["Continue current research direction."],
        "mood": "focused",
        "raw": raw,
    }
