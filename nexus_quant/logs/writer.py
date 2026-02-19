"""
Analysis Log Writer — persistent research journal for NEXUS.

Writes structured analysis entries to artifacts/logs/analysis.jsonl.
Each entry captures thinking, findings, decisions, and audit results
that would otherwise be lost when the terminal session closes.

Usage:
    from nexus_quant.logs import log_analysis, log_finding, log_decision

    log_analysis("V1 2025 full year: Sharpe 0.102", source="claude", category="backtest")
    log_finding("Orderflow signal DECAYED", details="Sharpe -1.14 in 2025", severity="critical")
    log_decision("V1 standalone reinstated as champion", rationale="Ensemble hurt by decayed signals")
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


_DEFAULT_LOG_DIR = Path("artifacts/logs")
_LOG_FILE = "analysis.jsonl"


def _ensure_dir(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / _LOG_FILE


def _write_entry(entry: Dict[str, Any], log_dir: Optional[Path] = None) -> None:
    """Append a single entry to the analysis log."""
    ld = log_dir or _DEFAULT_LOG_DIR
    fp = _ensure_dir(ld)
    with open(fp, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def log_analysis(
    content: str,
    *,
    source: str = "claude",
    category: str = "analysis",
    title: str = "",
    tags: Optional[List[str]] = None,
    log_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Log a general analysis entry (thinking, evaluation, comparison)."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "epoch": time.time(),
        "type": "analysis",
        "source": source,
        "category": category,
        "title": title or content[:80],
        "content": content,
        "tags": tags or [],
    }
    _write_entry(entry, log_dir)
    return entry


def log_finding(
    title: str,
    *,
    details: str = "",
    severity: str = "info",
    source: str = "claude",
    metrics: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    log_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Log a research finding (backtest result, signal discovery, etc)."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "epoch": time.time(),
        "type": "finding",
        "source": source,
        "severity": severity,
        "title": title,
        "details": details,
        "metrics": metrics or {},
        "tags": tags or [],
    }
    _write_entry(entry, log_dir)
    return entry


def log_decision(
    decision: str,
    *,
    rationale: str = "",
    source: str = "claude",
    alternatives: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    log_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Log a strategic decision (champion selection, signal rejection, etc)."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "epoch": time.time(),
        "type": "decision",
        "source": source,
        "title": decision,
        "rationale": rationale,
        "alternatives": alternatives or [],
        "tags": tags or [],
    }
    _write_entry(entry, log_dir)
    return entry


def log_audit(
    question: str,
    *,
    responses: Optional[Dict[str, str]] = None,
    consensus: str = "",
    source: str = "multi-model",
    tags: Optional[List[str]] = None,
    log_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Log a multi-model audit result."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "epoch": time.time(),
        "type": "audit",
        "source": source,
        "title": question[:80],
        "question": question,
        "responses": responses or {},
        "consensus": consensus,
        "tags": tags or [],
    }
    _write_entry(entry, log_dir)
    return entry


def read_log(
    log_dir: Optional[Path] = None,
    n: int = 100,
    entry_type: Optional[str] = None,
    source: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Read the last N entries from the analysis log, optionally filtered."""
    ld = log_dir or _DEFAULT_LOG_DIR
    fp = ld / _LOG_FILE
    if not fp.exists():
        return []
    lines = fp.read_text(encoding="utf-8").splitlines()
    entries = []
    for line in lines:
        try:
            e = json.loads(line)
            if entry_type and e.get("type") != entry_type:
                continue
            if source and e.get("source") != source:
                continue
            entries.append(e)
        except (json.JSONDecodeError, TypeError):
            continue
    return entries[-n:]
