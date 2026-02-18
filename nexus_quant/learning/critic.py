from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..ledger.ledger import LedgerEvent, append_ledger_event
from ..memory.store import MemoryStore
from ..orchestration.overrides import load_overrides
from ..utils.hashing import sha256_text


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ts_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


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


def _latest_by_prefix(dir_path: Path, prefix: str, suffix: str) -> Optional[Path]:
    if not dir_path.exists():
        return None
    items = sorted([p for p in dir_path.glob(f"{prefix}*{suffix}") if p.is_file()])
    return items[-1] if items else None


def critique_recent(*, config_path: Path, artifacts_dir: Path, tail_events: int = 200) -> Dict[str, Any]:
    """
    Deterministic "critic" loop (LLM-free).

    It continuously challenges assumptions using *evidence*:
    - compares strategy vs baselines
    - flags overfit symptoms (stability, turnover, cost sensitivity)
    - proposes next experiments (as config override snippets)
    - records an auditable critique artifact + memory + ledger event
    """
    artifacts_dir = Path(artifacts_dir)
    config_path = Path(config_path)
    ledger_path = artifacts_dir / "ledger" / "ledger.jsonl"
    events = _read_jsonl(ledger_path, max_lines=5000)
    tail = events[-max(1, min(int(tail_events), 2000)) :] if events else []

    # Pick the most recent run event as the anchor.
    run_ev = None
    for e in reversed(tail):
        if e.get("kind") == "run":
            run_ev = e
            break

    if not run_ev:
        return {"ok": False, "error": "no_run_event_found"}

    run_id = str(run_ev.get("run_id") or "")
    run_name = str(run_ev.get("run_name") or config_path.stem)
    payload = run_ev.get("payload") or {}
    run_root = Path(str((payload.get("paths") or {}).get("run_root") or ""))
    if not run_root.exists():
        # Fallback: best effort guess.
        run_root = artifacts_dir / "runs" / run_id

    cfg_used = _read_json(run_root / "config.json") or _read_json(config_path)
    metrics = _read_json(run_root / "metrics.json")
    result = _read_json(run_root / "result.json")

    summary = metrics.get("summary") or {}
    baselines = metrics.get("baselines") or {}
    walk = metrics.get("walk_forward") or {}
    verdict = metrics.get("verdict") or {}

    # Optional: last self-learn event info.
    learn_ev = None
    for e in reversed(tail):
        if e.get("kind") == "self_learn":
            learn_ev = e
            break
    learn_payload = (learn_ev or {}).get("payload") or {}

    best_params = None
    best_path = artifacts_dir / "memory" / "best_params.json"
    if best_path.exists():
        best_params = _read_json(best_path)

    ablation_latest = None
    abl_path = artifacts_dir / "memory" / "ablation_latest.json"
    if abl_path.exists():
        ablation_latest = _read_json(abl_path)

    ov = load_overrides(artifacts_dir)

    # Heuristics: "what would a quant reviewer push back on?"
    flags: List[Dict[str, Any]] = []
    next_experiments: List[Dict[str, Any]] = []

    # Baseline comparisons (choose a couple to avoid noise).
    eq_bh = baselines.get("equal_weight_buy_hold") or {}
    btc_bh = baselines.get("btc_buy_hold") or {}
    s_calmar = _safe_float(summary.get("calmar"))
    s_sharpe = _safe_float(summary.get("sharpe"))

    if eq_bh:
        if s_calmar < _safe_float(eq_bh.get("calmar")):
            flags.append({"level": "warn", "code": "underperform_equal_weight_bh_calmar", "detail": {"strategy_calmar": s_calmar, "baseline_calmar": _safe_float(eq_bh.get("calmar"))}})
        if s_sharpe < _safe_float(eq_bh.get("sharpe")):
            flags.append({"level": "warn", "code": "underperform_equal_weight_bh_sharpe", "detail": {"strategy_sharpe": s_sharpe, "baseline_sharpe": _safe_float(eq_bh.get("sharpe"))}})

    if btc_bh:
        if s_calmar < _safe_float(btc_bh.get("calmar")):
            flags.append({"level": "warn", "code": "underperform_btc_bh_calmar", "detail": {"strategy_calmar": s_calmar, "baseline_calmar": _safe_float(btc_bh.get("calmar"))}})

    # Risk/cost/stability flags
    if not bool(verdict.get("pass", True)):
        flags.append({"level": "error", "code": "verdict_fail", "detail": {"reasons": verdict.get("reasons") or [], "values": verdict.get("values") or {}, "thresholds": verdict.get("thresholds") or {}}})

    turnover_max = _safe_float(summary.get("turnover_max"))
    if turnover_max > 0.0 and turnover_max >= 2.0:
        flags.append({"level": "warn", "code": "high_turnover_max", "detail": {"turnover_max": turnover_max}})

    mdd = _safe_float(summary.get("max_drawdown"))
    if mdd >= 0.35:
        flags.append({"level": "warn", "code": "high_drawdown", "detail": {"max_drawdown": mdd}})

    corr = _safe_float(summary.get("corr"))
    beta = _safe_float(summary.get("beta"))
    if abs(corr) >= 0.85 and abs(beta) >= 0.7:
        flags.append({"level": "warn", "code": "beta_like_exposure", "detail": {"corr": corr, "beta": beta}})

    st = walk.get("stability") or {}
    frac_prof = _safe_float(st.get("fraction_profitable"))
    if st and frac_prof > 0 and frac_prof < 0.6:
        flags.append({"level": "warn", "code": "low_walk_forward_profit_fraction", "detail": {"fraction_profitable": frac_prof, "windows": st.get("windows")}})

    # PnL breakdown sanity (funding strategies can hide price risk).
    breakdown = (result.get("breakdown") or {})
    if breakdown:
        price_pnl = _safe_float(breakdown.get("price_pnl"))
        funding_pnl = _safe_float(breakdown.get("funding_pnl"))
        cost_pnl = _safe_float(breakdown.get("cost_pnl"))
        if abs(cost_pnl) > abs(price_pnl + funding_pnl) * 0.5 and (price_pnl + funding_pnl) != 0.0:
            flags.append({"level": "warn", "code": "costs_dominate_pnl", "detail": {"price_pnl": price_pnl, "funding_pnl": funding_pnl, "cost_pnl": cost_pnl}})

    # Next experiments: deterministic suggestions based on strategy family.
    strat = (cfg_used.get("strategy") or {}).get("name") or ""
    params = (cfg_used.get("strategy") or {}).get("params") or {}

    if str(strat) == "funding_carry_perp_v1":
        # Turnover control
        if turnover_max >= 2.0 or "turnover_max_gt_threshold" in (verdict.get("reasons") or []):
            next_experiments.append({"why": "reduce turnover", "config_overrides": {"strategy": {"params": {"k_per_side": max(1, int(params.get("k_per_side") or 2) - 1), "target_gross_leverage": max(0.4, float(params.get("target_gross_leverage") or 1.0) * 0.8)}}}})
            next_experiments.append({"why": "rebalance less often", "config_overrides": {"strategy": {"params": {"rebalance_on_funding": False, "rebalance_interval_bars": 24}}}})

        # Signal ablations even if not accepted
        next_experiments.append({"why": "ablation: funding-only", "config_overrides": {"strategy": {"params": {"use_basis_proxy": False, "basis_weight": 0.0}}}})
        next_experiments.append({"why": "ablation: basis-only", "config_overrides": {"strategy": {"params": {"use_basis_proxy": True, "basis_weight": 1.0, "k_per_side": int(params.get("k_per_side") or 2)}}}})

    if str(strat) == "momentum_xs_v1":
        next_experiments.append({"why": "reduce turnover (slower rebalance)", "config_overrides": {"strategy": {"params": {"rebalance_interval_bars": 48}}}})
        next_experiments.append({"why": "longer lookback to reduce noise", "config_overrides": {"strategy": {"params": {"lookback_bars": 1440}}}})

    if str(strat) == "mean_reversion_xs_v1":
        next_experiments.append({"why": "reduce crash risk (lower leverage)", "config_overrides": {"strategy": {"params": {"target_gross_leverage": max(0.4, float(params.get("target_gross_leverage") or 1.0) * 0.7)}}}})
        next_experiments.append({"why": "less frequent rebalance", "config_overrides": {"strategy": {"params": {"rebalance_interval_bars": 24}}}})

    if str(strat) == "multi_factor_xs_v1":
        next_experiments.append({"why": "turnover control (slower rebalance)", "config_overrides": {"strategy": {"params": {"rebalance_interval_bars": 48}}}})
        next_experiments.append({"why": "stress: more costs", "config_overrides": {"costs": {"cost_multiplier": 2.0}}})
        next_experiments.append({"why": "factor ablation: no momentum", "config_overrides": {"strategy": {"params": {"w_momentum": 0.0}}}})
        next_experiments.append({"why": "factor ablation: no mean-reversion", "config_overrides": {"strategy": {"params": {"w_mean_reversion": 0.0}}}})

    # Cost sensitivity suggestion (universal)
    cur_slip = float(((cfg_used.get("execution") or {}).get("slippage_bps") or 3.0))
    next_experiments.append({"why": "cost sensitivity: +50% slippage", "config_overrides": {"execution": {"slippage_bps": cur_slip * 1.5}}})

    critique = {
        "ts": _utc_iso(),
        "run": {"run_id": run_id, "run_name": run_name, "run_root": str(run_root)},
        "strategy": {"name": strat, "params": params},
        "summary": summary,
        "verdict": verdict,
        "baselines": {"btc_buy_hold": btc_bh, "equal_weight_buy_hold": eq_bh},
        "walk_forward": walk,
        "self_learn": {
            "last": {
                "accepted": learn_payload.get("accepted"),
                "objective": learn_payload.get("objective"),
                "min_uplift": learn_payload.get("min_uplift"),
                "prior_exploit_prob": learn_payload.get("prior_exploit_prob"),
            },
            "best_params": best_params,
            "ablation_latest": ablation_latest,
        },
        "overrides": ov,
        "flags": flags,
        "next_experiments": next_experiments,
    }

    out_dir = artifacts_dir / "wisdom" / "critiques"
    out_dir.mkdir(parents=True, exist_ok=True)
    tsid = _ts_id()
    out_json = out_dir / f"critique.{tsid}.json"
    out_md = out_dir / f"critique.{tsid}.md"
    out_json.write_text(json.dumps(critique, indent=2, sort_keys=True), encoding="utf-8")
    out_md.write_text(_render_md(critique), encoding="utf-8")
    (out_dir / "latest.json").write_text(json.dumps(critique, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "latest.md").write_text(_render_md(critique), encoding="utf-8")

    # Memory record
    mem_db = artifacts_dir / "memory" / "memory.db"
    ms = MemoryStore(mem_db)
    try:
        ms.add(
            created_at=_utc_iso(),
            kind="critique",
            tags=["echo", "review", "verified"],
            content=f"Critique for {run_name}: flags={len(flags)} next_experiments={len(next_experiments)}",
            meta={"critique_json": str(out_json), "critique_md": str(out_md), "run_id": run_id},
            run_id=run_id or None,
        )
    finally:
        ms.close()

    # Ledger event (append-only)
    if ledger_path.exists():
        ev = LedgerEvent(
            ts=_utc_iso(),
            kind="critique",
            run_id=run_id or f"critique.{tsid}",
            run_name=run_name,
            config_sha=str(run_ev.get("config_sha") or sha256_text(json.dumps(cfg_used, sort_keys=True))),
            code_fingerprint=str(run_ev.get("code_fingerprint") or ""),
            data_fingerprint=str(run_ev.get("data_fingerprint") or ""),
            payload={"paths": {"json": str(out_json), "md": str(out_md)}, "flags": flags, "next_experiments": next_experiments},
        )
        append_ledger_event(ledger_path, ev)

    return {"ok": True, "critique_json": str(out_json), "critique_md": str(out_md), "next_experiments": next_experiments, "flags": flags}


def _render_md(c: Dict[str, Any]) -> str:
    run = c.get("run") or {}
    strat = c.get("strategy") or {}
    flags = c.get("flags") or []
    nex = c.get("next_experiments") or []

    lines = []
    lines.append("# NEXUS Critique (Deterministic)")
    lines.append("")
    lines.append(f"- ts: `{c.get('ts')}`")
    lines.append(f"- run_id: `{run.get('run_id')}`")
    lines.append(f"- run_name: `{run.get('run_name')}`")
    lines.append(f"- run_root: `{run.get('run_root')}`")
    lines.append("")
    lines.append("## Strategy")
    lines.append(f"- name: `{strat.get('name')}`")
    lines.append(f"- params: `{json.dumps(strat.get('params') or {}, sort_keys=True)}`")
    lines.append("")
    lines.append("## Verdict")
    v = c.get("verdict") or {}
    lines.append(f"- pass: `{v.get('pass')}`  reasons={json.dumps(v.get('reasons') or [])}")
    lines.append("")
    lines.append("## Key Metrics")
    s = c.get("summary") or {}
    for k in ["total_return", "cagr", "volatility", "sharpe", "sortino", "max_drawdown", "calmar", "turnover_avg", "turnover_max", "beta", "corr"]:
        if k in s:
            lines.append(f"- {k}: `{s.get(k)}`")
    lines.append("")
    lines.append("## Flags (pushback)")
    if not flags:
        lines.append("- (none)")
    else:
        for f in flags:
            lines.append(f"- [{f.get('level')}] {f.get('code')}: `{json.dumps(f.get('detail') or {}, sort_keys=True)}`")
    lines.append("")
    lines.append("## Next Experiments (suggested overrides)")
    if not nex:
        lines.append("- (none)")
    else:
        for it in nex[:12]:
            lines.append(f"- why: {it.get('why')}")
            lines.append(f"  overrides: `{json.dumps(it.get('config_overrides') or {}, sort_keys=True)}`")
    lines.append("")
    lines.append("## Notes")
    lines.append("- This is LLM-free. It is conservative: it only uses evidence artifacts.")
    lines.append("- Treat `next_experiments` as a queue; run them one-by-one under locked benchmark + gates.")
    lines.append("")
    return "\n".join(lines)
