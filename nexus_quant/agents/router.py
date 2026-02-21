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


def _smart_task_type_for_model(model: Optional[str] = None):
    """Map agent default_model hint to SmartRouter TaskType."""
    from .smart_router import TaskType
    # Agents set _smart_task_type on their class; here we use a simple heuristic
    return TaskType.QA_CHAT  # default; overridden by callers when possible


def safe_call_llm(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    max_tokens: int = 1024,
    fallback_result: Optional[Dict[str, Any]] = None,
    smart_task_type: Optional[str] = None,
) -> Tuple[str, bool, Optional[str]]:
    """
    Call LLM safely, returning (response_text, fallback_used, error).

    Priority:
    1. Anthropic SDK (if ZAI/ANTHROPIC key available)
    2. SmartRouter (Gemini Flash always works)
    3. Deterministic fallback
    """
    fallback_json = json.dumps(fallback_result or {})

    # Try 1: Anthropic SDK (original path)
    api_key = _get_api_key()
    if api_key:
        try:
            text = call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                max_tokens=max_tokens,
            )
            return text, False, None
        except Exception:
            pass  # Fall through to SmartRouter

    # Try 2: SmartRouter (Gemini Flash always available)
    try:
        from .smart_router import SmartRouter, TaskType
        router = SmartRouter()
        tt = TaskType.QA_CHAT
        if smart_task_type:
            tt = TaskType(smart_task_type)
        text = router.call(task_type=tt, system=system_prompt, user=user_prompt, max_tokens=max_tokens)
        if text and text.strip():
            return text, False, None
    except Exception as e:
        pass  # Fall through to deterministic fallback

    return fallback_json, True, "All LLM paths failed (anthropic SDK + SmartRouter)"


def call_llm_dual(
    system_prompt: str,
    user_prompt: str,
    primary_model: Optional[str] = None,
    secondary_model: Optional[str] = None,
    max_tokens: int = 1024,
    fallback_result: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, bool, Optional[str]]:
    """
    Call two LLM models with the same prompt and return both responses.

    Used for critical cross-validation (e.g. ECHO QA agent).
    Returns (primary_text, secondary_text, any_fallback, error).
    Priority: anthropic SDK → SmartRouter → fallback.
    """
    fallback_json = json.dumps(fallback_result or {})
    primary_text = fallback_json
    secondary_text = fallback_json
    any_fallback = False
    errors: list = []

    # Primary model call: try anthropic SDK first, then SmartRouter
    api_key = _get_api_key()
    if api_key:
        try:
            primary_text = call_llm(system_prompt, user_prompt, model=primary_model, max_tokens=max_tokens)
        except Exception as e:
            errors.append(f"primary_anthropic({primary_model}): {e}")
    if primary_text == fallback_json:
        # Fallback to SmartRouter for primary
        try:
            from .smart_router import SmartRouter, TaskType
            router = SmartRouter()
            primary_text = router.call(task_type=TaskType.DATA_ANALYSIS, system=system_prompt, user=user_prompt, max_tokens=max_tokens)
            if not primary_text or not primary_text.strip():
                primary_text = fallback_json
                any_fallback = True
        except Exception as e:
            any_fallback = True
            errors.append(f"primary_smart({primary_model}): {e}")

    # Secondary model call via SmartRouter (always use Flash for cross-validation)
    if secondary_model:
        try:
            from .smart_router import SmartRouter, GOOGLE_GEMINI_FLASH
            router = SmartRouter()
            secondary_text = router._call_with_spec(
                GOOGLE_GEMINI_FLASH,
                task_type=None,
                system=system_prompt,
                user=user_prompt,
                max_tokens=max_tokens,
            )
            if not secondary_text or not secondary_text.strip():
                secondary_text = primary_text
        except Exception as e:
            secondary_text = primary_text
            errors.append(f"secondary({secondary_model}): {e}")
    else:
        secondary_text = primary_text

    error_str = "; ".join(errors) if errors else None
    return primary_text, secondary_text, any_fallback, error_str


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
