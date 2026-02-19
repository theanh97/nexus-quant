from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .utils.hashing import sha256_bytes, sha256_text, file_sha256
from .data.providers.registry import make_provider
from .data.quality import validate_dataset
from .strategies.registry import make_strategy
from .backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from .backtest.costs import cost_model_from_config
from .evaluation.benchmark import BenchmarkConfig, run_benchmark_pack_v1
from .ledger.ledger import LedgerEvent, append_ledger_event
from .self_learn.search import improve_strategy_params
from .utils.merge import deep_merge


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _hash_run_identity(config_text: str) -> str:
    return sha256_text(config_text)[:16]


def _code_fingerprint() -> str:
    """
    Lightweight code identity even without git:
    hash of a small set of core modules (expand later if needed).
    """
    roots = [
        Path(__file__).resolve(),
        Path(__file__).resolve().parent / "backtest" / "engine.py",
        Path(__file__).resolve().parent / "evaluation" / "metrics.py",
        Path(__file__).resolve().parent / "strategies" / "funding_carry.py",
    ]
    parts = []
    for p in roots:
        if p.exists():
            parts.append(f"{p.name}:{file_sha256(p)}")
    return sha256_text("\n".join(parts))[:16]


def run_one(config_path: Path, out_dir: Path, *, cfg_override: Optional[Dict[str, Any]] = None) -> str:
    base_text = config_path.read_text(encoding="utf-8")
    cfg = json.loads(base_text)
    if cfg_override:
        cfg = deep_merge(cfg, cfg_override)
    cfg_text = json.dumps(cfg, sort_keys=True)

    seed = int(cfg.get("seed", 0))
    run_name = str(cfg.get("run_name") or config_path.stem)
    config_id = _hash_run_identity(cfg_text)
    code_fp = _code_fingerprint()
    ts_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_group_id = f"{run_name}.{config_id}"
    run_id = f"{run_group_id}.{code_fp}.{ts_id}"

    run_root = out_dir / "runs" / run_id
    _ensure_dir(run_root)
    _ensure_dir(out_dir / "ledger")

    provider = make_provider(cfg["data"], seed=seed)
    dataset = provider.load()
    expected_step = None
    if "bar_interval_minutes" in (cfg.get("data") or {}):
        expected_step = int(cfg["data"]["bar_interval_minutes"]) * 60
    dq = validate_dataset(dataset, expected_step_seconds=expected_step)
    if not dq["ok"]:
        raise SystemExit(f"Data quality gate failed: {dq['issues'][:3]}")

    strategy = make_strategy(cfg["strategy"])
    costs_cfg = cfg.get("costs") or {}
    exec_cfg = cfg.get("execution") or {}
    venue_cfg = cfg.get("venue") or {}
    bt_cfg = BacktestConfig(costs=cost_model_from_config(costs_cfg, execution_cfg=exec_cfg, venue_cfg=venue_cfg))
    engine = BacktestEngine(bt_cfg)
    result = engine.run(dataset=dataset, strategy=strategy, seed=seed)
    result.code_fingerprint = code_fp

    bench_cfg = BenchmarkConfig.from_dict(cfg.get("benchmark") or {})
    bench = run_benchmark_pack_v1(dataset=dataset, result=result, bench_cfg=bench_cfg)

    # Run bias checks
    try:
        from .validation.bias_checker import run_full_bias_check

        returns = result.returns if hasattr(result, "returns") else result.to_dict().get("returns", [])
        if returns:
            bias = run_full_bias_check(
                returns,
                n_params_searched=int(cfg.get("self_learn", {}).get("n_trials", 1)),
                periods_per_year=8760.0,
            )
            bench["bias_check"] = {
                "verdict": bias["overall_verdict"],
                "confidence": bias["confidence_score"],
                "flags": bias["flags"],
                "sharpe_tstat": bias["checks"]["sharpe_sig"].get("t_stat"),
                "significant_95": bias["checks"]["sharpe_sig"].get("significant_95"),
                "likely_overfit": bias["checks"]["overfitting"].get("likely_overfit"),
            }
    except Exception as e:
        bench["bias_check"] = {"error": str(e)}
    bench["verdict"] = _compute_verdict(bench, risk_cfg=cfg.get("risk") or {}, data_quality_ok=bool(dq["ok"]))

    (run_root / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")
    (run_root / "data_quality.json").write_text(json.dumps(dq, indent=2, sort_keys=True), encoding="utf-8")
    (run_root / "metrics.json").write_text(json.dumps(bench, indent=2, sort_keys=True), encoding="utf-8")
    (run_root / "result.json").write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    report = _render_report_md(cfg, result, bench)
    (run_root / "report.md").write_text(report, encoding="utf-8")

    event = LedgerEvent(
        ts=_utcnow_iso(),
        kind="run",
        run_id=run_id,
        run_name=run_name,
        config_sha=sha256_text(cfg_text),
        code_fingerprint=code_fp,
        data_fingerprint=dataset.fingerprint,
        payload={
            "run_group_id": run_group_id,
            "config_path": str(config_path),
            "config_overrides": cfg_override or {},
            "venue": cfg.get("venue") or {},
            "execution": cfg.get("execution") or {},
            "costs": cfg.get("costs") or {},
            "data": cfg.get("data") or {},
            "strategy": cfg["strategy"]["name"],
            "metrics": bench.get("summary") or {},
            "bias_check": bench.get("bias_check") or {},
            "data_quality_ok": bool(dq["ok"]),
            "verdict": bench.get("verdict") or {},
            "paths": {
                "run_root": str(run_root),
                "report": str(run_root / "report.md"),
            },
        },
    )
    append_ledger_event(out_dir / "ledger" / "ledger.jsonl", event)
    return run_id


def improve_one(
    config_path: Path, out_dir: Path, trials: int = 30, *, cfg_override: Optional[Dict[str, Any]] = None
) -> str:
    base_text = config_path.read_text(encoding="utf-8")
    cfg = json.loads(base_text)
    if cfg_override:
        cfg = deep_merge(cfg, cfg_override)
    cfg_text = json.dumps(cfg, sort_keys=True)
    seed = int(cfg.get("seed", 0))
    run_name = str(cfg.get("run_name") or config_path.stem)
    config_id = _hash_run_identity(cfg_text)
    code_fp = _code_fingerprint()
    ts_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_group_id = f"{run_name}.{config_id}"
    run_id = f"{run_group_id}.{code_fp}.{ts_id}"

    _ensure_dir(out_dir / "runs")
    _ensure_dir(out_dir / "ledger")
    _ensure_dir(out_dir / "memory")

    provider = make_provider(cfg["data"], seed=seed)
    dataset = provider.load()
    expected_step = None
    if "bar_interval_minutes" in (cfg.get("data") or {}):
        expected_step = int(cfg["data"]["bar_interval_minutes"]) * 60
    dq = validate_dataset(dataset, expected_step_seconds=expected_step)
    if not dq["ok"]:
        raise SystemExit(f"Data quality gate failed: {dq['issues'][:3]}")
    (out_dir / "memory" / "data_quality.json").write_text(json.dumps(dq, indent=2, sort_keys=True), encoding="utf-8")

    base_strategy_cfg = cfg["strategy"]
    base_strategy = make_strategy(base_strategy_cfg)

    costs_cfg = cfg.get("costs") or {}
    exec_cfg = cfg.get("execution") or {}
    venue_cfg = cfg.get("venue") or {}
    bt_cfg = BacktestConfig(costs=cost_model_from_config(costs_cfg, execution_cfg=exec_cfg, venue_cfg=venue_cfg))
    engine = BacktestEngine(bt_cfg)

    bench_cfg = BenchmarkConfig.from_dict(cfg.get("benchmark") or {})

    improvement = improve_strategy_params(
        dataset=dataset,
        engine=engine,
        base_strategy_cfg=base_strategy_cfg,
        bench_cfg=bench_cfg,
        risk_cfg=cfg.get("risk") or {},
        self_learn_cfg=cfg.get("self_learn") or {},
        run_config_sha=sha256_text(cfg_text),
        code_fingerprint=code_fp,
        run_id=run_id,
        run_group_id=run_group_id,
        trials=int(trials),
        seed=seed,
        memory_dir=out_dir / "memory",
    )

    # Always record the self-learn event (even if no improvement found).
    event = LedgerEvent(
        ts=_utcnow_iso(),
        kind="self_learn",
        run_id=run_id,
        run_name=run_name,
        config_sha=sha256_text(cfg_text),
        code_fingerprint=code_fp,
        data_fingerprint=dataset.fingerprint,
        payload={
            **improvement,
            "run_group_id": run_group_id,
            "config_path": str(config_path),
            "config_overrides": cfg_override or {},
            "venue": cfg.get("venue") or {},
            "execution": cfg.get("execution") or {},
            "costs": cfg.get("costs") or {},
            "data": cfg.get("data") or {},
        },
    )
    append_ledger_event(out_dir / "ledger" / "ledger.jsonl", event)
    return run_id


def _compute_verdict(bench: Dict[str, Any], risk_cfg: Dict[str, Any], data_quality_ok: bool) -> Dict[str, Any]:
    reasons = []
    summary = bench.get("summary") or {}
    mdd = float(summary.get("max_drawdown", 0.0) or 0.0)
    turnover_max = float(summary.get("turnover_max", 0.0) or 0.0)

    max_mdd = float(risk_cfg.get("max_drawdown") or 0.35)
    max_turnover = float(risk_cfg.get("max_turnover_per_rebalance") or 2.0)

    if not data_quality_ok:
        reasons.append("data_quality_fail")
    if mdd > max_mdd:
        reasons.append("mdd_gt_threshold")
    if turnover_max > max_turnover:
        reasons.append("turnover_max_gt_threshold")

    return {
        "pass": len(reasons) == 0,
        "reasons": reasons,
        "thresholds": {"max_drawdown": max_mdd, "max_turnover_per_rebalance": max_turnover},
        "values": {"max_drawdown": round(mdd, 6), "turnover_max": round(turnover_max, 6)},
    }


def _render_report_md(cfg: Dict[str, Any], result: BacktestResult, bench: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"# NEXUS Quant Report: {cfg.get('run_name')}")
    lines.append("")
    lines.append("## Identity")
    lines.append(f"- Strategy: `{cfg['strategy']['name']}`")
    lines.append(f"- Data fingerprint: `{result.data_fingerprint}`")
    lines.append(f"- Code fingerprint: `{result.code_fingerprint}`")
    lines.append("")
    lines.append("## Venue & Execution")
    venue = cfg.get("venue") or {}
    execution = cfg.get("execution") or {}
    costs = cfg.get("costs") or {}
    if venue:
        lines.append(f"- venue: `{json.dumps(venue, sort_keys=True)}`")
    if execution:
        lines.append(f"- execution: `{json.dumps(execution, sort_keys=True)}`")
    if costs:
        # Keep costs explicit; this is part of the audit trail.
        lines.append(f"- costs: `{json.dumps(costs, sort_keys=True)}`")
    lines.append("")
    lines.append("## Summary (Benchmark v1)")
    summary = bench.get("summary") or {}
    for k in sorted(summary.keys()):
        lines.append(f"- {k}: `{summary[k]}`")
    lines.append("")
    lines.append("## PnL Breakdown")
    lines.append(f"- Price PnL: `{result.breakdown.get('price_pnl', 0.0):.6f}`")
    lines.append(f"- Funding PnL: `{result.breakdown.get('funding_pnl', 0.0):.6f}`")
    lines.append(f"- Cost PnL: `{result.breakdown.get('cost_pnl', 0.0):.6f}`")
    lines.append("")
    lines.append("## Notes")
    lines.append("- This report is reproducible given the same config, seed, and dataset provider.")
    lines.append("- Treat synthetic results as a pipeline demo; switch to local data for real research.")
    lines.append("")
    return "\n".join(lines)
