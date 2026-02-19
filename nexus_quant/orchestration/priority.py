"""
ExperimentPrioritizer: Ranks experiment proposals by expected value.

Uses:
- FLUX agent suggestions (LLM-based priority)
- Historical acceptance rate by config dimension
- Exploration/exploitation balance
"""
from __future__ import annotations

import json
import hashlib
import random
from typing import Any, Dict, List, Optional

from .scheduler import ExperimentSpec


def _exp_id(config_overrides: Dict[str, Any], why: str = "") -> str:
    raw = json.dumps({"why": why, "overrides": config_overrides}, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:12]


def build_experiment_specs(
    proposals: List[Dict[str, Any]],
    *,
    max_experiments: int = 10,
    deduplicate: bool = True,
    existing_ids: Optional[set] = None,
) -> List[ExperimentSpec]:
    """
    Convert raw proposals (from critic/ATLAS) to ExperimentSpec list.
    
    Each proposal: {"config_overrides": {...}, "why": str, "priority": int (1-10)}
    """
    existing_ids = existing_ids or set()
    seen_ids = set()
    specs = []
    
    for prop in proposals[:max_experiments * 2]:  # oversample, then trim
        overrides = prop.get("config_overrides") or {}
        why = str(prop.get("why") or "")
        priority = int(prop.get("priority") or 5)
        exp_id = _exp_id(overrides, why)
        
        if deduplicate and exp_id in existing_ids:
            continue
        if exp_id in seen_ids:
            continue
        seen_ids.add(exp_id)
        
        specs.append(ExperimentSpec(
            experiment_id=exp_id,
            config_overrides=overrides,
            priority=priority,
            why=why,
        ))
        
        if len(specs) >= max_experiments:
            break
    
    return sorted(specs, key=lambda s: s.priority)


def score_experiments(
    specs: List[ExperimentSpec],
    historical_results: List[Dict[str, Any]],
) -> List[ExperimentSpec]:
    """
    Re-score experiments based on historical performance of similar overrides.
    Higher scoring experiments get lower priority number (run first).
    """
    if not historical_results:
        return specs
    
    # Build lookup: config dimension -> mean Sharpe
    dim_scores: Dict[str, float] = {}
    for r in historical_results:
        overrides = r.get("config_overrides") or {}
        sharpe = float(r.get("sharpe") or 0.0)
        passed = bool(r.get("pass"))
        for k, v in overrides.items():
            key = f"{k}={v}"
            existing = dim_scores.get(key, [])
            if not isinstance(existing, list):
                existing = []
            existing.append(sharpe if passed else -1.0)
            dim_scores[key] = existing
    
    # Average scores
    avg_scores = {k: sum(v)/len(v) for k, v in dim_scores.items() if isinstance(v, list) and v}
    
    def _score_spec(spec: ExperimentSpec) -> float:
        scores = []
        for k, v in spec.config_overrides.items():
            key = f"{k}={v}"
            if key in avg_scores:
                scores.append(avg_scores[key])
        return sum(scores) / len(scores) if scores else 0.0
    
    # Re-assign priority based on score (higher score = lower priority number = run first)
    scored = [(spec, _score_spec(spec)) for spec in specs]
    scored.sort(key=lambda x: -x[1])  # sort by score descending
    
    result = []
    for i, (spec, _) in enumerate(scored):
        result.append(ExperimentSpec(
            experiment_id=spec.experiment_id,
            config_overrides=spec.config_overrides,
            priority=i + 1,
            why=spec.why,
            tags=spec.tags,
        ))
    return result
