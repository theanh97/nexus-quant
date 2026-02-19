"""
NEXUS Reasoning Engine - Uses LLM (multi-model) to reason about
research questions, explain decisions, and generate diary entries.
"""
from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional


def _call_llm(
    system_prompt: str,
    user_message: str,
    max_tokens: int = 800,
    model: str | None = None,
) -> str:
    """Multi-model LLM router. Supports GLM-5, Claude Sonnet/Opus, GPT-5.2 (Codex), MiniMax."""
    selected = model or os.environ.get("ZAI_DEFAULT_MODEL", "glm-5")

    # ─── ZAI / Anthropic-compatible gateway (GLM-5, Claude models) ───
    if selected in ("glm-5", "claude-sonnet-4-6", "claude-opus-4-6", "glm-4"):
        api_key = os.environ.get("ZAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        base_url = os.environ.get("ZAI_ANTHROPIC_BASE_URL", "https://api.z.ai/api/anthropic")
        # Remove trailing slash if present
        if base_url.endswith("/"):
            base_url = base_url.rstrip("/")
        if not api_key:
            return f"[NEXUS] No API key configured for {selected}."
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
            resp = client.messages.create(
                model=selected,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            return resp.content[0].text if resp.content else ""
        except Exception as e:
            return f"[{selected} error: {e}]"

    # ─── Codex CLI (GPT-5.2) ───
    if selected in ("codex", "gpt-5.2", "gpt4o"):
        codex_bin = "/Applications/Codex.app/Contents/Resources/codex"
        if not __import__("os").path.exists(codex_bin):
            return "[Codex CLI not found. Please ensure Codex.app is installed.]"
        try:
            import subprocess
            combined = f"System: {system_prompt}\n\nUser: {user_message}"
            result = subprocess.run(
                [codex_bin, "exec", "-c", 'sandbox_permissions=["disk-full-read-access"]', "-"],
                input=combined.encode("utf-8"),
                capture_output=True,
                timeout=30,
            )
            out = result.stdout.decode("utf-8", errors="replace").strip()
            return out if out else f"[Codex: no output, stderr={result.stderr.decode()[:200]}]"
        except subprocess.TimeoutExpired:
            return "[Codex: timeout after 30s]"
        except Exception as e:
            return f"[Codex error: {e}]"

    # ─── MiniMax 2.5 ───
    if selected in ("minimax-2.5", "minimax"):
        minimax_key = os.environ.get("MINIMAX_API_KEY", "")
        minimax_group = os.environ.get("MINIMAX_GROUP_ID", "")
        if not minimax_key or not minimax_group:
            return "[MiniMax] MINIMAX_API_KEY or MINIMAX_GROUP_ID not configured in environment."
        try:
            from minimax import Minimax  # type: ignore
            client = Minimax(api_key=minimax_key, group_id=minimax_group)
            resp = client.chat.completions.create(
                model="abab6.5s-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content if resp.choices else ""
        except Exception as e:
            return f"[MiniMax error: {e}]"

    return f"[Unknown model: {selected}]"


def answer_question(
    question: str,
    context: Dict[str, Any],
    persona: str = "",
    model: str | None = None,
) -> str:
    """NEXUS answers a user question about its research, using full context."""
    system = (
        f"{persona}\n\n"
        "You have access to recent research results from NEXUS quant trading system. "
        "Answer concisely and accurately. If you are uncertain, say so. "
        "Format numbers to 3 decimal places. Use bullet points for lists. "
        "When asked about performance, always distinguish IS (in-sample) from OOS (out-of-sample)."
    )
    ctx_str = json.dumps(
        {
            "recent_metrics": context.get("metrics") or {},
            "regime": context.get("regime") or {},
            "active_goals": context.get("goals") or [],
            "last_diary": context.get("last_diary") or {},
            "wisdom_summary": str(context.get("wisdom") or {})[:500],
            "live_market": context.get("live_market") or {},
            "system_state": context.get("system_state") or {},
        },
        indent=2,
    )
    user_msg = f"Context:\n{ctx_str}\n\nQuestion: {question}"
    return _call_llm(system, user_msg, max_tokens=800, model=model)


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
