"""
NEXUS Smart Router - Assigns the optimal LLM to each task type.

Model assignments based on empirical benchmarks:
- Claude Sonnet 4.6: Architecture, code review, complex debugging (PM role)
- GPT-5.2 (Codex): Code generation, strategy math, refactoring
- GLM-5 (ZAI): Q&A, diary synthesis, quick summaries (low cost)
- MiniMax: High-throughput monitoring, real-time alerts (when configured)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class TaskType(str, Enum):
    CODE_GENERATION = "CODE_GENERATION"
    CODE_REVIEW = "CODE_REVIEW"
    STRATEGY_RESEARCH = "STRATEGY_RESEARCH"
    RISK_ANALYSIS = "RISK_ANALYSIS"
    QA_CHAT = "QA_CHAT"
    DIARY_SYNTHESIS = "DIARY_SYNTHESIS"
    MONITORING_ALERT = "MONITORING_ALERT"
    DATA_ANALYSIS = "DATA_ANALYSIS"
    EXPERIMENT_DESIGN = "EXPERIMENT_DESIGN"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: str
    base_url: str
    api_key_env: str
    max_tokens: int
    cost_tier: str  # "low" | "medium" | "high"
    strengths: List[str]


_DEFAULT_ZAI_ANTHROPIC_BASE_URL = "https://api.z.ai/api/anthropic"


def _zai_base_url() -> str:
    return os.environ.get("ZAI_ANTHROPIC_BASE_URL") or _DEFAULT_ZAI_ANTHROPIC_BASE_URL


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


ZAI_GLM5 = ModelSpec(
    name="glm-5",
    provider="zai",
    base_url=_zai_base_url(),
    api_key_env="ZAI_API_KEY",
    max_tokens=4096,
    cost_tier="low",
    strengths=["q&a", "summaries", "diary synthesis", "cheap throughput"],
)

ZAI_CLAUDE_SONNET_46 = ModelSpec(
    name="claude-sonnet-4-6",
    provider="zai",
    base_url=_zai_base_url(),
    api_key_env="ZAI_API_KEY",
    max_tokens=4096,
    cost_tier="high",
    strengths=["architecture", "code review", "complex debugging", "risk reasoning"],
)

OPENAI_GPT52_CODEX = ModelSpec(
    name="gpt-5.2",
    provider="openai",
    base_url="https://api.openai.com/v1",
    api_key_env="OPENAI_API_KEY",
    max_tokens=4096,
    cost_tier="high",
    strengths=["code generation", "refactoring", "strategy math"],
)


ROUTING_TABLE: Dict[TaskType, ModelSpec] = {
    TaskType.CODE_GENERATION: OPENAI_GPT52_CODEX,
    TaskType.STRATEGY_RESEARCH: ZAI_CLAUDE_SONNET_46,
    TaskType.CODE_REVIEW: ZAI_CLAUDE_SONNET_46,
    TaskType.RISK_ANALYSIS: ZAI_CLAUDE_SONNET_46,
    TaskType.DATA_ANALYSIS: ZAI_CLAUDE_SONNET_46,
    TaskType.QA_CHAT: ZAI_GLM5,
    TaskType.DIARY_SYNTHESIS: ZAI_GLM5,
    TaskType.MONITORING_ALERT: ZAI_GLM5,
    TaskType.EXPERIMENT_DESIGN: ZAI_GLM5,
}


class SmartRouter:
    def __init__(
        self,
        routing_table: Optional[Dict[TaskType, ModelSpec]] = None,
        fallback_model: Optional[ModelSpec] = None,
        log_path: Optional[Path] = None,
    ) -> None:
        self._routing_table = dict(routing_table or ROUTING_TABLE)
        self._fallback_model = fallback_model or ZAI_GLM5
        project_root = Path(__file__).resolve().parents[2]
        self._log_path = log_path or (project_root / "artifacts" / "brain" / "routing_log.jsonl")

    def get_model_spec(self, task_type: TaskType) -> Optional[ModelSpec]:
        """Return the routed ModelSpec for this task type (may be None on error)."""
        try:
            return self.route(task_type)
        except Exception:
            return None

    def route(self, task_type: TaskType) -> ModelSpec:
        """
        Return the best available model spec for this task type.

        For CODE_GENERATION, prefer OpenAI if OPENAI_API_KEY is configured,
        otherwise route to GLM-5 via ZAI.
        """
        spec = self._routing_table.get(task_type, self._fallback_model)
        if task_type == TaskType.CODE_GENERATION and spec.provider == "openai":
            if not os.environ.get(spec.api_key_env):
                spec = self._fallback_model

        if spec.provider == "zai":
            spec = replace(spec, base_url=_zai_base_url())
        return spec

    def call(self, task_type: TaskType, system: str, user: str, max_tokens: int = 500) -> str:
        primary = self.route(task_type)
        try:
            text = self._call_with_spec(primary, task_type=task_type, system=system, user=user, max_tokens=max_tokens)
            self._log_event(
                {
                    "ts": _now_iso(),
                    "task_type": task_type.value,
                    "model": primary.name,
                    "provider": primary.provider,
                    "base_url": primary.base_url,
                    "max_tokens": min(max_tokens, primary.max_tokens),
                    "fallback_used": False,
                }
            )
            return text
        except Exception as e:
            self._log_event(
                {
                    "ts": _now_iso(),
                    "task_type": task_type.value,
                    "model": primary.name,
                    "provider": primary.provider,
                    "base_url": primary.base_url,
                    "max_tokens": min(max_tokens, primary.max_tokens),
                    "fallback_used": True,
                    "error": str(e),
                }
            )

        fallback = replace(self._fallback_model, base_url=_zai_base_url())
        if fallback == primary:
            raise RuntimeError(f"Primary model failed and fallback is identical: {primary.name}")
        text = self._call_with_spec(fallback, task_type=task_type, system=system, user=user, max_tokens=max_tokens)
        self._log_event(
            {
                "ts": _now_iso(),
                "task_type": task_type.value,
                "model": fallback.name,
                "provider": fallback.provider,
                "base_url": fallback.base_url,
                "max_tokens": min(max_tokens, fallback.max_tokens),
                "fallback_used": True,
                "fallback_for": primary.name,
            }
        )
        return text

    def _call_with_spec(self, spec: ModelSpec, task_type: TaskType, system: str, user: str, max_tokens: int) -> str:
        if spec.provider == "openai":
            models_to_try = [spec.name]
            if task_type == TaskType.CODE_GENERATION:
                # If GPT-5.2 isn't available, try gpt-4o before falling back to GLM-5.
                models_to_try.append("gpt-4o")
            return self._call_openai(spec, system=system, user=user, max_tokens=max_tokens, models=models_to_try)
        if spec.provider == "zai":
            return self._call_anthropic(spec, system=system, user=user, max_tokens=max_tokens)
        raise ValueError(f"Unknown provider: {spec.provider}")

    def _call_anthropic(self, spec: ModelSpec, system: str, user: str, max_tokens: int) -> str:
        try:
            import anthropic  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError("anthropic SDK not installed. Run: pip install anthropic") from e

        api_key = os.environ.get(spec.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key env var: {spec.api_key_env}")

        kwargs: Dict[str, Any] = {"api_key": api_key, "max_retries": 2}
        if spec.base_url:
            kwargs["base_url"] = spec.base_url

        client = anthropic.Anthropic(**kwargs)
        resp = client.messages.create(
            model=spec.name,
            max_tokens=min(max_tokens, spec.max_tokens),
            temperature=0.3,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if not getattr(resp, "content", None):
            return ""
        return str(resp.content[0].text)

    def _call_openai(
        self,
        spec: ModelSpec,
        system: str,
        user: str,
        max_tokens: int,
        models: List[str],
    ) -> str:
        api_key = os.environ.get(spec.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key env var: {spec.api_key_env}")

        url = spec.base_url.rstrip("/") + "/chat/completions"
        last_error: Optional[Exception] = None
        for model in models:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": min(max_tokens, spec.max_tokens),
                "temperature": 0.3,
            }
            req = urllib.request.Request(
                url=url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )

            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    body = resp.read().decode("utf-8")
                data = json.loads(body)
                text = (
                    (data.get("choices") or [{}])[0].get("message", {}).get("content")
                    or (data.get("choices") or [{}])[0].get("text")
                    or ""
                )
                return str(text)
            except urllib.error.HTTPError as e:
                body = ""
                try:
                    body = e.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                model_not_found = False
                try:
                    err = json.loads(body).get("error") or {}
                    code = (err.get("code") or "").lower()
                    msg = (err.get("message") or "").lower()
                    if "model" in code or "not_found" in code or "model" in msg or "not found" in msg:
                        model_not_found = True
                except Exception:
                    pass
                if e.code in (400, 404) and model_not_found and model != models[-1]:
                    last_error = e
                    continue
                raise RuntimeError(f"OpenAI HTTP {e.code}: {body[:500]}") from e
            except urllib.error.URLError as e:
                last_error = e
                break
            except Exception as e:
                last_error = e
                break

        raise RuntimeError(f"OpenAI call failed for models={models}: {last_error}") from last_error

    def _log_event(self, event: Dict[str, Any]) -> None:
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception:
            # Best-effort logging only.
            return
