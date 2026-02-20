"""
NEXUS Smart Router v1.0 — Decision Network model assignments.

Cost-optimized model stack (Feb 2026):
- Claude Sonnet 4.6: CIPHER (risk) + ECHO primary (QA) — precision-critical [PAID via ZAI]
- Gemini 3 Pro (preview): ATLAS (research) + FLUX (decision gate) [FREE]
- Gemini 3 Pro (preview): ECHO secondary (dual-model cross-validation) [FREE]
- GPT-5.3 Codex: Code generation tasks [PAID via OpenAI CLI]
- Gemini 3 Flash: QA chat, monitoring, diary synthesis [FREE]
- GLM-5 (ZAI): Fallback only [PAID, low cost]

Decision Network phases:
  Phase 1 — ATLAS (Gemini) + CIPHER (Claude) in parallel
  Phase 2 — ECHO (Claude + Gemini dual-model cross-validation)
  Phase 3 — FLUX (Gemini) + deterministic synthesis gate

Per-cycle cost: ~2 Claude calls + 3 Gemini calls (free) = ~50% cheaper than all-Claude.
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

OPENAI_GPT53_CODEX = ModelSpec(
    name="gpt-5.3-codex",
    provider="openai",
    base_url="https://api.openai.com/v1",
    api_key_env="OPENAI_API_KEY",
    max_tokens=4096,
    cost_tier="high",
    strengths=["code generation", "refactoring", "strategy math", "agentic coding"],
)

# ─── Google Gemini (OpenAI-compatible endpoint) — FREE tier ───
_GOOGLE_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"

# Gemini 3 Pro (preview) — latest free model, strong reasoning
GOOGLE_GEMINI_PRO = ModelSpec(
    name="gemini-3-pro-preview",
    provider="google",
    base_url=_GOOGLE_OPENAI_BASE,
    api_key_env="GEMINI_API_KEY",
    max_tokens=4096,
    cost_tier="free",
    strengths=["complex reasoning", "long context", "cross-verification", "multimodal"],
)

# Gemini 3 Flash (preview) — fast free model
GOOGLE_GEMINI_FLASH = ModelSpec(
    name="gemini-3-flash-preview",
    provider="google",
    base_url=_GOOGLE_OPENAI_BASE,
    api_key_env="GEMINI_API_KEY",
    max_tokens=4096,
    cost_tier="free",
    strengths=["fast", "high throughput", "monitoring", "cost-effective"],
)

# Gemini 2.5 stable — fallback if preview models unavailable
GOOGLE_GEMINI_25_PRO = ModelSpec(
    name="gemini-2.5-pro",
    provider="google",
    base_url=_GOOGLE_OPENAI_BASE,
    api_key_env="GEMINI_API_KEY",
    max_tokens=4096,
    cost_tier="free",
    strengths=["stable", "complex reasoning", "long context"],
)


# ─── Gemini CLI (local binary, OAuth — no API key needed) — FREE tier ───

GEMINI_CLI_PRO = ModelSpec(
    name="gemini-cli-pro",
    provider="gemini-cli",
    base_url="",
    api_key_env="",
    max_tokens=4096,
    cost_tier="free",
    strengths=["complex reasoning", "long context", "cross-verification", "no API key"],
)

GEMINI_CLI_FLASH = ModelSpec(
    name="gemini-cli-flash",
    provider="gemini-cli",
    base_url="",
    api_key_env="",
    max_tokens=4096,
    cost_tier="free",
    strengths=["fast", "high throughput", "no API key"],
)


# ── v1.0 Decision Network — Fallback Chain per Task Type ──
#
# Each task type has an ORDERED list of models to try.
# First model = primary. If it fails → try next → next → ...
# To change model assignments: just reorder or swap entries in the list.
#
FALLBACK_CHAINS: Dict[TaskType, List[ModelSpec]] = {
    # Phase 1: ATLAS — creative proposals, cost-effective
    TaskType.STRATEGY_RESEARCH: [
        GOOGLE_GEMINI_PRO,        # 1st: Gemini 3 Pro [FREE]
        ZAI_CLAUDE_SONNET_46,     # 2nd: Claude Sonnet [PAID]
        GOOGLE_GEMINI_25_PRO,     # 3rd: Gemini 2.5 stable [FREE]
        ZAI_GLM5,                 # 4th: GLM-5 [low cost]
    ],
    # Phase 1: CIPHER — risk precision, Claude first
    TaskType.RISK_ANALYSIS: [
        ZAI_CLAUDE_SONNET_46,     # 1st: Claude Sonnet [PAID] — cannot afford mistakes
        GOOGLE_GEMINI_PRO,        # 2nd: Gemini 3 Pro [FREE]
        GOOGLE_GEMINI_25_PRO,     # 3rd: Gemini 2.5 stable [FREE]
        ZAI_GLM5,                 # 4th: GLM-5 [low cost]
    ],
    # Phase 2: ECHO primary — overfitting detection, most critical
    TaskType.DATA_ANALYSIS: [
        ZAI_CLAUDE_SONNET_46,     # 1st: Claude Sonnet [PAID] — strongest reasoning
        GOOGLE_GEMINI_PRO,        # 2nd: Gemini 3 Pro [FREE]
        GOOGLE_GEMINI_25_PRO,     # 3rd: Gemini 2.5 stable [FREE]
        ZAI_GLM5,                 # 4th: GLM-5 [low cost]
    ],
    # Phase 3: FLUX — decision gate, structured input
    TaskType.EXPERIMENT_DESIGN: [
        GOOGLE_GEMINI_PRO,        # 1st: Gemini 3 Pro [FREE]
        ZAI_CLAUDE_SONNET_46,     # 2nd: Claude Sonnet [PAID]
        GOOGLE_GEMINI_25_PRO,     # 3rd: Gemini 2.5 stable [FREE]
        ZAI_GLM5,                 # 4th: GLM-5 [low cost]
    ],
    # Code generation
    TaskType.CODE_GENERATION: [
        OPENAI_GPT53_CODEX,       # 1st: GPT-5.3 Codex [PAID]
        GOOGLE_GEMINI_PRO,        # 2nd: Gemini 3 Pro [FREE]
        ZAI_CLAUDE_SONNET_46,     # 3rd: Claude Sonnet [PAID]
        ZAI_GLM5,                 # 4th: GLM-5 [low cost]
    ],
    # Non-agent tasks
    TaskType.CODE_REVIEW: [
        GOOGLE_GEMINI_PRO,        # 1st: Gemini 3 Pro [FREE]
        GOOGLE_GEMINI_25_PRO,     # 2nd: Gemini 2.5 stable [FREE]
        ZAI_GLM5,                 # 3rd: GLM-5 [low cost]
    ],
    TaskType.QA_CHAT: [
        GOOGLE_GEMINI_FLASH,      # 1st: Gemini Flash [FREE]
        GOOGLE_GEMINI_PRO,        # 2nd: Gemini 3 Pro [FREE]
        ZAI_GLM5,                 # 3rd: GLM-5 [low cost]
    ],
    TaskType.DIARY_SYNTHESIS: [
        GOOGLE_GEMINI_FLASH,      # 1st: Gemini Flash [FREE]
        GOOGLE_GEMINI_25_PRO,     # 2nd: Gemini 2.5 stable [FREE]
        ZAI_GLM5,                 # 3rd: GLM-5 [low cost]
    ],
    TaskType.MONITORING_ALERT: [
        GOOGLE_GEMINI_FLASH,      # 1st: Gemini Flash [FREE]
        GOOGLE_GEMINI_PRO,        # 2nd: Gemini 3 Pro [FREE]
        ZAI_GLM5,                 # 3rd: GLM-5 [low cost]
    ],
}

# Legacy compatibility: ROUTING_TABLE maps to first entry in each chain
ROUTING_TABLE: Dict[TaskType, ModelSpec] = {
    tt: chain[0] for tt, chain in FALLBACK_CHAINS.items()
}

# ECHO uses dual-model cross-validation: Claude (primary) + Gemini 3 Pro (secondary)
ECHO_SECONDARY_MODEL: ModelSpec = GOOGLE_GEMINI_PRO


class SmartRouter:
    def __init__(
        self,
        routing_table: Optional[Dict[TaskType, ModelSpec]] = None,
        fallback_model: Optional[ModelSpec] = None,
        log_path: Optional[Path] = None,
    ) -> None:
        self._routing_table = dict(routing_table or ROUTING_TABLE)
        # Prefer Gemini Flash (free) as fallback if API key available, else GLM-5
        if fallback_model:
            self._fallback_model = fallback_model
        elif os.environ.get("GEMINI_API_KEY"):
            self._fallback_model = GOOGLE_GEMINI_FLASH
        else:
            self._fallback_model = ZAI_GLM5
        project_root = Path(__file__).resolve().parents[2]
        self._log_path = log_path or (project_root / "artifacts" / "brain" / "routing_log.jsonl")

    def get_model_spec(self, task_type: TaskType) -> Optional[ModelSpec]:
        """Return the routed ModelSpec for this task type (may be None on error)."""
        try:
            return self.route(task_type)
        except Exception:
            return None

    @staticmethod
    def _gemini_cli_available() -> bool:
        import shutil
        return bool(shutil.which("gemini") or os.path.isfile("/opt/homebrew/bin/gemini"))

    def route(self, task_type: TaskType) -> ModelSpec:
        """
        Return the best available model spec for this task type.

        Auto-fallback logic:
        - CODE_GENERATION: OpenAI → GLM-5 if no key
        - Gemini API → Gemini CLI if no GEMINI_API_KEY but CLI available
        """
        spec = self._routing_table.get(task_type, self._fallback_model)
        if task_type == TaskType.CODE_GENERATION and spec.provider == "openai":
            if not os.environ.get(spec.api_key_env):
                spec = self._fallback_model

        # Auto-fallback: Gemini API → Gemini CLI when no API key
        if spec.provider == "google" and not os.environ.get(spec.api_key_env):
            if self._gemini_cli_available():
                cli_spec = GEMINI_CLI_PRO if "pro" in spec.name else GEMINI_CLI_FLASH
                spec = cli_spec

        if spec.provider == "zai":
            spec = replace(spec, base_url=_zai_base_url())
        return spec

    def get_chain(self, task_type: TaskType) -> List[ModelSpec]:
        """Get the full fallback chain for a task type (for inspection/override)."""
        chain = FALLBACK_CHAINS.get(task_type, [self._fallback_model])
        # Resolve availability: skip models without API keys
        resolved: List[ModelSpec] = []
        for spec in chain:
            s = spec
            if s.provider == "zai":
                s = replace(s, base_url=_zai_base_url())
            if s.provider == "google" and not os.environ.get(s.api_key_env):
                if self._gemini_cli_available():
                    s = GEMINI_CLI_PRO if "pro" in s.name else GEMINI_CLI_FLASH
                else:
                    continue  # skip unavailable
            if s.provider == "openai" and not os.environ.get(s.api_key_env):
                continue  # skip unavailable
            resolved.append(s)
        if not resolved:
            resolved.append(replace(self._fallback_model, base_url=_zai_base_url()))
        return resolved

    def call(self, task_type: TaskType, system: str, user: str, max_tokens: int = 500) -> str:
        """
        Call LLM with automatic fallback chain.

        Tries each model in the chain until one succeeds.
        Logs every attempt (success and failure) for observability.
        """
        chain = self.get_chain(task_type)
        last_error: Optional[Exception] = None

        for i, spec in enumerate(chain):
            try:
                text = self._call_with_spec(spec, task_type=task_type, system=system, user=user, max_tokens=max_tokens)
                self._log_event({
                    "ts": _now_iso(),
                    "task_type": task_type.value,
                    "model": spec.name,
                    "provider": spec.provider,
                    "max_tokens": min(max_tokens, spec.max_tokens),
                    "chain_position": i,
                    "chain_length": len(chain),
                    "fallback_used": i > 0,
                })
                return text
            except Exception as e:
                last_error = e
                self._log_event({
                    "ts": _now_iso(),
                    "task_type": task_type.value,
                    "model": spec.name,
                    "provider": spec.provider,
                    "max_tokens": min(max_tokens, spec.max_tokens),
                    "chain_position": i,
                    "chain_length": len(chain),
                    "fallback_used": True,
                    "error": str(e)[:200],
                })
                continue  # try next in chain

        raise RuntimeError(
            f"All {len(chain)} models failed for {task_type.value}: {last_error}"
        )

    def _call_with_spec(self, spec: ModelSpec, task_type: TaskType, system: str, user: str, max_tokens: int) -> str:
        if spec.provider == "openai":
            models_to_try = [spec.name]
            if task_type == TaskType.CODE_GENERATION:
                models_to_try.append("gpt-5.2")
            return self._call_openai(spec, system=system, user=user, max_tokens=max_tokens, models=models_to_try)
        if spec.provider == "google":
            models_to_try = [spec.name]
            if "pro" in spec.name:
                models_to_try.append("gemini-2.5-pro")  # stable fallback
            return self._call_openai(spec, system=system, user=user, max_tokens=max_tokens, models=models_to_try)
        if spec.provider == "gemini-cli":
            return self._call_gemini_cli(spec, system=system, user=user)
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

    @staticmethod
    def _call_gemini_cli(spec: ModelSpec, system: str, user: str) -> str:
        """Call Gemini via local CLI binary (uses OAuth, no API key)."""
        import shutil
        import subprocess

        gemini_bin = shutil.which("gemini") or "/opt/homebrew/bin/gemini"
        if not os.path.isfile(gemini_bin):
            raise RuntimeError("Gemini CLI not installed")

        cli_model = "gemini-2.5-pro" if "pro" in spec.name else "gemini-2.5-flash"
        combined = f"SYSTEM: {system}\n\nUSER: {user}"

        result = subprocess.run(
            [gemini_bin, "-p", combined, "-m", cli_model, "-o", "text"],
            capture_output=True,
            timeout=90,
            cwd=os.environ.get("HOME", "/tmp"),
        )
        text = result.stdout.decode(errors="replace").strip()
        # Strip Gemini CLI boilerplate
        lines = text.splitlines()
        clean = [l for l in lines if not l.startswith("Loaded cached") and not l.startswith("Hook registry")]
        text = "\n".join(clean).strip()
        if not text:
            raise RuntimeError(f"Gemini CLI returned empty (stderr: {result.stderr.decode(errors='replace')[:200]})")
        return text

    def _log_event(self, event: Dict[str, Any]) -> None:
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception:
            # Best-effort logging only.
            return
