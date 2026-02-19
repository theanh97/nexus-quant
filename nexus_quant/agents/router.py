"""
AgentRouter: Calls LLM via Z.AI gateway (Anthropic API format).

Supports multiple models:
- GLM-5 (default): via ZAI_DEFAULT_MODEL env
- claude-sonnet-4-6: via ZAI_ANTHROPIC_BASE_URL
- MiniMax: via minimax SDK

Graceful fallback: if ANTHROPIC_API_KEY or ZAI_API_KEY not set,
returns AgentResult with fallback_used=True and deterministic defaults.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional, Tuple

from .base import AgentContext, AgentResult


def _get_api_key() -> Optional[str]:
    """Try ZAI_API_KEY first, then ANTHROPIC_API_KEY."""
    return os.environ.get("ZAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")


def _get_base_url() -> Optional[str]:
    return os.environ.get("ZAI_ANTHROPIC_BASE_URL")


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> str:
    """
    Call LLM via anthropic SDK with Z.AI gateway.
    Returns response text or raises exception.
    """
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic SDK not installed. Run: pip install anthropic")

    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("No API key found (ZAI_API_KEY or ANTHROPIC_API_KEY)")

    base_url = _get_base_url()
    resolved_model = model or os.environ.get("ZAI_DEFAULT_MODEL") or "glm-4-long"

    kwargs: Dict[str, Any] = {"api_key": api_key, "max_retries": 2}
    if base_url:
        kwargs["base_url"] = base_url

    client = anthropic.Anthropic(**kwargs)
    message = client.messages.create(
        model=resolved_model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return str(message.content[0].text)


def safe_call_llm(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    max_tokens: int = 1024,
    fallback_result: Optional[Dict[str, Any]] = None,
) -> Tuple[str, bool, Optional[str]]:
    """
    Call LLM safely, returning (response_text, fallback_used, error).
    If no API key or exception, returns fallback_result as JSON string.
    """
    fallback_json = json.dumps(fallback_result or {})
    api_key = _get_api_key()
    if not api_key:
        return fallback_json, True, "No API key configured"
    try:
        text = call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            max_tokens=max_tokens,
        )
        return text, False, None
    except Exception as e:
        return fallback_json, True, str(e)


def extract_json_block(text: str) -> Dict[str, Any]:
    """Extract first JSON block from LLM response text."""
    # Try ```json ... ``` block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    # Try bare JSON object (handles nested braces one level deep)
    match = re.search(r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    return {}
