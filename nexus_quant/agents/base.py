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

    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model or os.environ.get("ZAI_DEFAULT_MODEL") or self.default_model

    def run(self, context: AgentContext) -> AgentResult:
        raise NotImplementedError
