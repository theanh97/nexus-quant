"""
Base agent interface for NEXUS multi-agent system.

All agents:
- Receive structured context (wisdom checkpoint + metrics + regime)
- Return structured output (JSON-serializable dict)
- Log output to ledger with provenance
- Fallback to deterministic mode if LLM unavailable
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentContext:
    """Structured context passed to every agent."""
    wisdom: Dict[str, Any]          # Latest wisdom checkpoint
    recent_metrics: Dict[str, Any]  # Last run metrics
    regime: Dict[str, Any]          # Current market regime
    best_params: Optional[Dict[str, Any]] = None
    memory_items: List[Dict[str, Any]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Structured output from an agent."""
    agent_name: str
    model_used: str
    raw_response: str
    parsed: Dict[str, Any]
    fallback_used: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "model_used": self.model_used,
            "raw_response": self.raw_response[:2000],  # truncate for ledger
            "parsed": self.parsed,
            "fallback_used": self.fallback_used,
            "error": self.error,
        }


class BaseAgent:
    """Base class for all NEXUS agents."""

    name: str = "base"
    default_model: str = "glm-4-long"
    # Map agent name → smart router TaskType (imported lazily)
    _smart_task_type: Optional[str] = None

    def __init__(self, model: Optional[str] = None) -> None:
        if model:
            self.model = model
        else:
            # Try SmartRouter to pick optimal model per task type
            resolved = self._resolve_model_via_smart_router()
            self.model = resolved or os.environ.get("ZAI_DEFAULT_MODEL") or self.default_model

    def _resolve_model_via_smart_router(self) -> Optional[str]:
        """Ask SmartRouter for the optimal model for this agent's task type."""
        if not self._smart_task_type:
            return None
        try:
            from ..agents.smart_router import SmartRouter, TaskType
            router = SmartRouter()
            spec = router.get_model_spec(TaskType(self._smart_task_type))
            if spec:
                return spec.name
        except Exception:
            pass
        return None

    def call_llm_via_router(
        self,
        system_prompt: str,
        user_prompt: str,
        task_type_str: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> str:
        """Call LLM using SmartRouter for optimal cost/quality routing."""
        try:
            from ..agents.smart_router import SmartRouter, TaskType
            tt_str = task_type_str or self._smart_task_type
            tt = TaskType(tt_str) if tt_str else None
            router = SmartRouter()
            return router.call(tt, system_prompt, user_prompt, max_tokens=max_tokens)
        except Exception:
            # Fallback to direct call
            from .router import call_llm
            return call_llm(system_prompt, user_prompt, model=self.model, max_tokens=max_tokens)

    def run(self, context: AgentContext) -> AgentResult:
        raise NotImplementedError
