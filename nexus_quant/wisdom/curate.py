from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..memory.store import MemoryStore


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ts_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_jsonl(path: Path, *, max_lines: int = 5000) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()[-max_lines:]
    out = []
    for ln in lines:
        try:
            obj = json.loads(ln)
            if isinstance(obj, dict):
                out.append(obj)
        except Exception:
            continue
    return out


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def curate_wisdom(
    *,
    artifacts_dir: Path,
    tail_events: int = 200,
    max_events_scan: int = 5000,
) -> Dict[str, Any]:
    """
    Compile long-horizon "wisdom" from ledger + memory into versioned checkpoints.

    This is intentionally LLM-free. It creates structured artifacts that a future
    LLM router can consume safely (with provenance).
    """
    artifacts_dir = Path(artifacts_dir)
    ledger_path = artifacts_dir / "ledger" / "ledger.jsonl"
    events = _read_jsonl(ledger_path, max_lines=max_events_scan)
    tail = events[-max(1, min(int(tail_events), 2000)) :]

    runs = [e for e in tail if e.get("kind") == "run"]
    learns = [e for e in tail if e.get("kind") == "self_learn"]

    # Recent performance snapshot
    recent_run = runs[-1] if runs else None
    recent_metrics = ((recent_run or {}).get("payload") or {}).get("metrics") or {}
    recent_verdict = ((recent_run or {}).get("payload") or {}).get("verdict") or {}

    # Self-learn stats
    accepted = [e for e in learns if bool(((e.get("payload") or {}).get("accepted")))]
    accept_rate = (len(accepted) / float(len(learns))) if learns else 0.0

    # Approx uplift distribution on holdout (when payload is available).
    uplifts = []
    for e in accepted:
        p = e.get("payload") or {}
        base = (p.get("baseline") or {}).get("holdout") or {}
        cand = (p.get("best_candidate") or {}).get("holdout") or {}
        obj = str(p.get("objective") or "median_calmar")
        b = _safe_float(base.get(obj))
        c = _safe_float(cand.get(obj))
        u = (c - b) / (abs(b) + 1e-9)
        uplifts.append(float(u))

    uplift_summary = {
        "count": len(uplifts),
        "median": round(float(statistics.median(uplifts)), 6) if uplifts else 0.0,
        "p25": round(float(statistics.quantiles(uplifts, n=4)[0]), 6) if len(uplifts) >= 4 else 0.0,
        "p75": round(float(statistics.quantiles(uplifts, n=4)[2]), 6) if len(uplifts) >= 4 else 0.0,
    }

    # Failure reasons (run verdict + learn rejects)
    reason_counts: Dict[str, int] = {}
    for e in runs:
        v = ((e.get("payload") or {}).get("verdict") or {})
        for r in (v.get("reasons") or []):
            reason_counts[str(r)] = int(reason_counts.get(str(r)) or 0) + 1

    # Proposal-level reject reasons (optional)
    proposals_path = artifacts_dir / "memory" / "proposals.jsonl"
    proposals = _read_jsonl(proposals_path, max_lines=2000)
    proposal_rejects = 0
    for pr in proposals:
        if str(pr.get("verdict") or "") != "reject":
            continue
        proposal_rejects += 1
        for r in (pr.get("reasons") or []):
            reason_counts[str(r)] = int(reason_counts.get(str(r)) or 0) + 1

    top_reasons = sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:12]

    # Long-term memory snapshot
    mem_db = artifacts_dir / "memory" / "memory.db"
    memory_stats = {}
    recent_feedback = []
    if mem_db.exists():
        ms = MemoryStore(mem_db)
        try:
            memory_stats = ms.stats()
            recent_feedback = [
                {"created_at": it.created_at, "tags": it.tags, "content": it.content[:400]}
                for it in ms.recent(kind="feedback", limit=5)
            ]
        finally:
            ms.close()

    # Latest accepted params (if any)
    best_params_path = artifacts_dir / "memory" / "best_params.json"
    best_params = None
    if best_params_path.exists():
        try:
            best_params = json.loads(best_params_path.read_text(encoding="utf-8"))
        except Exception:
            best_params = None

    priors_path = artifacts_dir / "memory" / "priors.json"
    priors = None
    if priors_path.exists():
        try:
            priors = json.loads(priors_path.read_text(encoding="utf-8"))
        except Exception:
            priors = None

    wisdom = {
        "ts": _utc_iso(),
        "artifacts_dir": str(artifacts_dir),
        "ledger": {"path": str(ledger_path), "tail_events": len(tail), "runs": len(runs), "self_learn": len(learns)},
        "recent": {
            "run_id": (recent_run or {}).get("run_id"),
            "run_name": (recent_run or {}).get("run_name"),
            "metrics": recent_metrics,
            "verdict": recent_verdict,
        },
        "self_learn": {
            "events": len(learns),
            "accepted": len(accepted),
            "accept_rate": round(float(accept_rate), 6),
            "uplift_holdout": uplift_summary,
            "proposal_rejects_scanned": int(proposal_rejects),
        },
        "top_reasons": [{"reason": r, "count": c} for r, c in top_reasons],
        "memory": {"db": str(mem_db), "stats": memory_stats, "recent_feedback": recent_feedback},
        "best_params": best_params,
        "priors": priors,
    }

    # Write artifacts (versioned + latest pointers)
    out_dir = artifacts_dir / "wisdom"
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tsid = _ts_id()

    ckpt_json = ckpt_dir / f"wisdom.{tsid}.json"
    ckpt_md = ckpt_dir / f"wisdom.{tsid}.md"
    latest_json = out_dir / "latest.json"
    latest_md = out_dir / "latest.md"

    ckpt_json.write_text(json.dumps(wisdom, indent=2, sort_keys=True), encoding="utf-8")
    latest_json.write_text(json.dumps(wisdom, indent=2, sort_keys=True), encoding="utf-8")

    md = render_wisdom_md(wisdom)
    ckpt_md.write_text(md, encoding="utf-8")
    latest_md.write_text(md, encoding="utf-8")

    return {"ok": True, "wisdom_path": str(ckpt_json), "wisdom_md": str(ckpt_md)}


def render_wisdom_md(wisdom: Dict[str, Any]) -> str:
    recent = wisdom.get("recent") or {}
    sl = wisdom.get("self_learn") or {}
    upl = sl.get("uplift_holdout") or {}
    lines = []
    lines.append("# NEXUS Quant Wisdom Checkpoint")
    lines.append("")
    lines.append(f"- generated_at: `{wisdom.get('ts')}`")
    lines.append(f"- artifacts_dir: `{wisdom.get('artifacts_dir')}`")
    lines.append("")
    lines.append("## Snapshot")
    lines.append(f"- last_run_id: `{recent.get('run_id')}`")
    lines.append(f"- last_run_name: `{recent.get('run_name')}`")
    v = recent.get("verdict") or {}
    lines.append(f"- verdict.pass: `{v.get('pass')}`  reasons={json.dumps(v.get('reasons') or [])}")
    lines.append("")
    lines.append("## Key Metrics (last run)")
    m = recent.get("metrics") or {}
    for k in sorted(m.keys()):
        lines.append(f"- {k}: `{m[k]}`")
    lines.append("")
    lines.append("## Self-Learning (tail window)")
    lines.append(f"- events: `{sl.get('events')}`  accepted: `{sl.get('accepted')}`  accept_rate: `{sl.get('accept_rate')}`")
    lines.append(f"- uplift_holdout.median: `{upl.get('median')}`  p25: `{upl.get('p25')}`  p75: `{upl.get('p75')}`  n: `{upl.get('count')}`")
    lines.append("")
    lines.append("## Top Failure/Reject Reasons")
    top = wisdom.get("top_reasons") or []
    if not top:
        lines.append("- (none)")
    else:
        for it in top:
            lines.append(f"- {it.get('reason')}: `{it.get('count')}`")
    lines.append("")
    lines.append("## Recent User Feedback (memory)")
    fb = ((wisdom.get("memory") or {}).get("recent_feedback") or [])
    if not fb:
        lines.append("- (none)")
    else:
        for it in fb:
            tags = ",".join(it.get("tags") or [])
            lines.append(f"- {it.get('created_at')}  tags=`{tags}`  {it.get('content')}")
    lines.append("")
    lines.append("## Notes")
    lines.append("- This file is produced without an LLM (deterministic), to be safe-by-default.")
    lines.append("- It is meant as input to a future NEXUS multi-agent brain/router.")
    lines.append("")
    return "\n".join(lines)

