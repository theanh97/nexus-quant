from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..ledger.ledger import LedgerEvent, append_ledger_event
from ..memory.store import MemoryStore
from ..orchestration.overrides import load_overrides, save_overrides
from ..utils.hashing import sha256_text
from ..utils.merge import deep_merge


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


def _count_reasons(items: List[Dict[str, Any]], *, key: str = "reasons") -> Dict[str, int]:
    c: Dict[str, int] = {}
    for it in items:
        for r in (it.get(key) or []):
            rr = str(r)
            c[rr] = int(c.get(rr) or 0) + 1
    return c


def _top_counts(counts: Dict[str, int], n: int = 12) -> List[Dict[str, Any]]:
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
    return [{"reason": r, "count": int(c)} for r, c in items]


def reflect_and_update(
    *,
    config_path: Path,
    artifacts_dir: Path,
    tail_events: int = 200,
) -> Dict[str, Any]:
    """
    Deterministic "critical thinking" loop (LLM-free):
    - analyze ledger + proposals
    - detect stuck patterns
    - propose and apply safe runtime overrides (trials + exploit_prob)
    - write reflection artifacts + memory item + ledger event
    """
    artifacts_dir = Path(artifacts_dir)
    config_path = Path(config_path)

    ledger_path = artifacts_dir / "ledger" / "ledger.jsonl"
    events = _read_jsonl(ledger_path, max_lines=5000)
    tail = events[-max(1, min(int(tail_events), 2000)) :]

    # Identify a reference event for provenance.
    ref = tail[-1] if tail else {}
    ref_run_id = str(ref.get("run_id") or "")
    ref_run_name = str(ref.get("run_name") or config_path.stem)
    ref_config_sha = str(ref.get("config_sha") or "")
    ref_code_fp = str(ref.get("code_fingerprint") or "")
    ref_data_fp = str(ref.get("data_fingerprint") or "")

    # Load config (with current overrides applied) for context.
    base_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    ov = load_overrides(artifacts_dir)
    cfg_override = (ov.get("config_overrides") or {}) if isinstance(ov, dict) else {}
    eff_cfg = deep_merge(base_cfg, cfg_override) if cfg_override else base_cfg
    self_learn_cfg = eff_cfg.get("self_learn") or {}
    cur_exploit = float(self_learn_cfg.get("prior_exploit_prob", 0.7))

    cur_trials = 30
    if isinstance(ov, dict):
        cur_trials = int((ov.get("orion") or {}).get("trials") or cur_trials)

    # Self-learn stagnation: consecutive events with accepted=false.
    learns = [e for e in tail if e.get("kind") == "self_learn"]
    consecutive_no_accept = 0
    for e in reversed(learns):
        acc = bool((e.get("payload") or {}).get("accepted"))
        if acc:
            break
        consecutive_no_accept += 1

    accept_rate = (sum(1 for e in learns if bool((e.get("payload") or {}).get("accepted"))) / float(len(learns))) if learns else 0.0

    # Proposals reject reasons (more granular than event-level).
    proposals_path = artifacts_dir / "memory" / "proposals.jsonl"
    proposals = _read_jsonl(proposals_path, max_lines=2000)
    reject_props = [p for p in proposals if str(p.get("verdict") or "") == "reject"]
    proposal_reason_counts = _count_reasons(reject_props)

    # Run verdict reasons (from run events).
    runs = [e for e in tail if e.get("kind") == "run"]
    run_reason_counts: Dict[str, int] = {}
    for r in runs:
        v = ((r.get("payload") or {}).get("verdict") or {})
        for rr in (v.get("reasons") or []):
            run_reason_counts[str(rr)] = int(run_reason_counts.get(str(rr)) or 0) + 1

    # Safe auto-updates (do NOT change locked gates automatically).
    proposed_trials = cur_trials
    proposed_exploit = cur_exploit
    actions: List[str] = []

    if consecutive_no_accept >= 3:
        proposed_trials = int(min(max(cur_trials * 2, 60), 400))
        proposed_exploit = max(0.1, cur_exploit - 0.2)
        actions.append("stuck_detected: increase trials and reduce exploit_prob to explore more")

    if consecutive_no_accept == 0 and accept_rate >= 0.3:
        # If we're consistently finding accepted params, bias more to exploitation.
        proposed_exploit = min(0.9, cur_exploit + 0.05)
        actions.append("healthy_accept_rate: slightly increase exploit_prob")

    # Extra deterministic hints (no auto-apply, just record).
    top_prop = _top_counts(proposal_reason_counts, n=5)
    for it in top_prop:
        if it["reason"] in {"train_mdd_gate_fail", "holdout_mdd_gate_fail"}:
            actions.append("hint: many MDD gate fails -> bias search toward lower leverage / inverse_vol")
            break
    for it in top_prop:
        if it["reason"] in {"train_turnover_gate_fail", "holdout_turnover_gate_fail"}:
            actions.append("hint: many turnover gate fails -> bias toward longer rebalance interval / smaller k_per_side")
            break

    # Apply safe overrides by writing artifacts/state/overrides.json
    orion_overrides = dict((ov.get("orion") or {}) if isinstance(ov, dict) else {})
    config_overrides = dict(cfg_override or {})

    # Only update these keys automatically (safe envelope).
    orion_overrides["trials"] = int(proposed_trials)
    config_overrides.setdefault("self_learn", {})
    config_overrides["self_learn"]["prior_exploit_prob"] = float(round(proposed_exploit, 6))

    overrides_path = save_overrides(
        artifacts_dir,
        orion=orion_overrides,
        config_overrides=config_overrides,
        note="auto-updated by reflection (safe envelope: trials, prior_exploit_prob)",
    )

    # Write reflection artifacts
    out_dir = artifacts_dir / "wisdom" / "reflections"
    out_dir.mkdir(parents=True, exist_ok=True)
    tsid = _ts_id()
    out_json = out_dir / f"reflection.{tsid}.json"
    out_md = out_dir / f"reflection.{tsid}.md"

    payload = {
        "ts": _utc_iso(),
        "config_path": str(config_path),
        "reference": {
            "run_id": ref_run_id,
            "run_name": ref_run_name,
            "config_sha": ref_config_sha,
            "code_fingerprint": ref_code_fp,
            "data_fingerprint": ref_data_fp,
        },
        "self_learn": {
            "events_scanned": len(learns),
            "accept_rate": round(float(accept_rate), 6),
            "consecutive_no_accept": int(consecutive_no_accept),
            "current": {"prior_exploit_prob": float(cur_exploit), "orion_trials": int(cur_trials)},
            "proposed": {"prior_exploit_prob": float(round(proposed_exploit, 6)), "orion_trials": int(proposed_trials)},
        },
        "top_proposal_reject_reasons": _top_counts(proposal_reason_counts, n=12),
        "top_run_verdict_reasons": _top_counts(run_reason_counts, n=12),
        "actions": actions,
        "overrides_path": str(overrides_path),
    }

    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    out_md.write_text(_render_md(payload), encoding="utf-8")

    # Memory record
    mem_db = artifacts_dir / "memory" / "memory.db"
    ms = MemoryStore(mem_db)
    try:
        ms.add(
            created_at=_utc_iso(),
            kind="reflection",
            tags=["orion", "self_learn", "policy"],
            content=f"Reflection: no_accept={consecutive_no_accept}, accept_rate={round(float(accept_rate), 4)}, overrides={overrides_path.name}",
            meta={"reflection_json": str(out_json), "reflection_md": str(out_md)},
            run_id=ref_run_id or None,
        )
    finally:
        ms.close()

    # Ledger event (append-only)
    if ledger_path.exists():
        ev = LedgerEvent(
            ts=_utc_iso(),
            kind="reflection",
            run_id=ref_run_id or f"reflection.{tsid}",
            run_name=ref_run_name,
            config_sha=ref_config_sha or sha256_text(json.dumps(eff_cfg, sort_keys=True)),
            code_fingerprint=ref_code_fp,
            data_fingerprint=ref_data_fp,
            payload={"reflection": payload, "paths": {"json": str(out_json), "md": str(out_md)}},
        )
        append_ledger_event(ledger_path, ev)

    return {"ok": True, "reflection_json": str(out_json), "reflection_md": str(out_md), "overrides": str(overrides_path)}


def _render_md(payload: Dict[str, Any]) -> str:
    sl = payload.get("self_learn") or {}
    cur = sl.get("current") or {}
    prop = sl.get("proposed") or {}
    lines = []
    lines.append("# NEXUS Reflection (Deterministic)")
    lines.append("")
    lines.append(f"- ts: `{payload.get('ts')}`")
    lines.append(f"- config_path: `{payload.get('config_path')}`")
    ref = payload.get("reference") or {}
    lines.append(f"- ref.run_id: `{ref.get('run_id')}`  run_name: `{ref.get('run_name')}`")
    lines.append("")
    lines.append("## Self-Learn Health")
    lines.append(f"- events_scanned: `{sl.get('events_scanned')}`")
    lines.append(f"- accept_rate: `{sl.get('accept_rate')}`")
    lines.append(f"- consecutive_no_accept: `{sl.get('consecutive_no_accept')}`")
    lines.append("")
    lines.append("## Safe Policy Update (auto-applied)")
    lines.append(f"- orion.trials: `{cur.get('orion_trials')}` -> `{prop.get('orion_trials')}`")
    lines.append(f"- self_learn.prior_exploit_prob: `{cur.get('prior_exploit_prob')}` -> `{prop.get('prior_exploit_prob')}`")
    lines.append("")
    lines.append("## Top Proposal Reject Reasons")
    top = payload.get("top_proposal_reject_reasons") or []
    if not top:
        lines.append("- (none)")
    else:
        for it in top:
            lines.append(f"- {it.get('reason')}: `{it.get('count')}`")
    lines.append("")
    lines.append("## Top Run Verdict Reasons")
    top2 = payload.get("top_run_verdict_reasons") or []
    if not top2:
        lines.append("- (none)")
    else:
        for it in top2:
            lines.append(f"- {it.get('reason')}: `{it.get('count')}`")
    lines.append("")
    lines.append("## Actions")
    acts = payload.get("actions") or []
    if not acts:
        lines.append("- (none)")
    else:
        for a in acts:
            lines.append(f"- {a}")
    lines.append("")
    lines.append("## Paths")
    lines.append(f"- overrides: `{payload.get('overrides_path')}`")
    lines.append("")
    return "\n".join(lines)

