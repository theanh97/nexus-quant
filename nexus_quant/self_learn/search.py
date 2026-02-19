from __future__ import annotations

import json
import os
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..backtest.engine import BacktestEngine
from ..data.schema import MarketDataset
from ..evaluation.metrics import equity_from_returns, max_drawdown, summarize
from ..strategies.registry import make_strategy
from ..utils.hashing import sha256_text
from .priors import load_priors, update_priors_on_accept


def improve_strategy_params(
    dataset: MarketDataset,
    engine: BacktestEngine,
    base_strategy_cfg: Dict[str, Any],
    bench_cfg: Any,
    risk_cfg: Dict[str, Any],
    self_learn_cfg: Dict[str, Any],
    run_config_sha: str,
    code_fingerprint: str,
    run_id: str,
    run_group_id: str,
    trials: int,
    seed: int,
    memory_dir: Path,
) -> Dict[str, Any]:
    """
    Verified self-learning (L1):
    - propose candidate params
    - evaluate on train split
    - verify on holdout split
    - record evidence to memory_dir
    """
    enabled = bool(self_learn_cfg.get("enabled", True))
    env_flag = os.environ.get("NX_ENABLE_SELF_LEARN")
    if env_flag is not None:
        enabled = env_flag.strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return {"enabled": False, "reason": "disabled"}

    objective = str(self_learn_cfg.get("objective") or "median_calmar")
    min_uplift = float(self_learn_cfg.get("min_uplift") or 0.0)
    holdout_fraction = float(self_learn_cfg.get("holdout_fraction") or 0.2)
    holdout_fraction = min(max(holdout_fraction, 0.05), 0.5)

    require_stress = bool(self_learn_cfg.get("require_stress_pass", True))
    stress_mult = float(self_learn_cfg.get("stress_cost_multiplier", 2.0))
    stress_mult = min(max(stress_mult, 1.0), 10.0)
    min_uplift_stress = float(self_learn_cfg.get("min_uplift_stress", min_uplift))

    priors_enabled = bool(self_learn_cfg.get("priors_enabled", True))
    prior_exploit_prob = float(self_learn_cfg.get("prior_exploit_prob", 0.7))
    prior_exploit_prob = min(max(prior_exploit_prob, 0.0), 1.0)

    rng = random.Random(seed + 9991)
    memory_dir.mkdir(parents=True, exist_ok=True)
    proposals_path = memory_dir / "proposals.jsonl"
    best_path = memory_dir / "best_params.json"

    priors = load_priors(memory_dir) if priors_enabled else {"version": 1, "updated_at": None, "strategies": {}}
    priors_by_strategy = (priors.get("strategies") or {}) if isinstance(priors, dict) else {}

    # Baseline run
    base_strategy = make_strategy(base_strategy_cfg)
    base_res = engine.run(dataset=dataset, strategy=base_strategy, seed=seed)
    base_eval = _eval_split(base_res, dataset=dataset, bench_cfg=bench_cfg, holdout_fraction=holdout_fraction)

    # Risk gates (baseline can still fail; learning should not make it worse).
    gates = _risk_gates(risk_cfg)

    # Candidate search
    best_train = None
    best_train_score = float("-inf")
    best_accept = None
    best_accept_score = float("-inf")
    history: List[Dict[str, Any]] = []

    for t in range(int(trials)):
        cand_cfg = _sample_candidate_cfg(
            base_strategy_cfg=base_strategy_cfg,
            dataset=dataset,
            rng=rng,
            priors=priors_by_strategy.get(str(base_strategy_cfg.get("name") or "")),
            exploit_prob=prior_exploit_prob,
        )
        cand_strategy = make_strategy(cand_cfg)
        cand_res = engine.run(dataset=dataset, strategy=cand_strategy, seed=seed + t + 1)
        cand_eval = _eval_split(cand_res, dataset=dataset, bench_cfg=bench_cfg, holdout_fraction=holdout_fraction)

        verdict = _verdict(
            base_eval=base_eval,
            cand_eval=cand_eval,
            objective=objective,
            min_uplift=min_uplift,
            gates=gates,
        )

        rec = {
            "trial": t,
            "strategy": cand_cfg.get("name"),
            "params": cand_cfg.get("params", {}),
            "objective": objective,
            "score_train": cand_eval["train"].get(objective, 0.0),
            "score_holdout": cand_eval["holdout"].get(objective, 0.0),
            "mdd_train": cand_eval["train"].get("max_drawdown", 0.0),
            "mdd_holdout": cand_eval["holdout"].get("max_drawdown", 0.0),
            "verdict": verdict["verdict"],
            "reasons": verdict["reasons"],
        }
        history.append(rec)
        _append_jsonl(proposals_path, rec)

        # Track best-by-train among gate-passers (holdout not used for optimization).
        train_score = float(cand_eval["train"].get(objective, 0.0))
        if verdict["passes_gates_train"] and train_score > best_train_score:
            best_train_score = train_score
            best_train = {"cfg": cand_cfg, "eval": cand_eval, "verdict": verdict}

        # Also keep the best candidate that PASSES holdout verification.
        # This uses holdout as a pass/fail gate, not as an optimization target.
        if verdict["verdict"] == "accept" and train_score > best_accept_score:
            best_accept_score = train_score
            best_accept = {"cfg": cand_cfg, "eval": cand_eval, "verdict": verdict}

    out = {
        "enabled": True,
        "objective": objective,
        "min_uplift": min_uplift,
        "holdout_fraction": holdout_fraction,
        "require_stress_pass": require_stress,
        "stress_cost_multiplier": stress_mult,
        "min_uplift_stress": min_uplift_stress,
        "priors_enabled": priors_enabled,
        "prior_exploit_prob": prior_exploit_prob,
        "baseline": {
            "strategy": base_strategy_cfg.get("name"),
            "params": base_strategy_cfg.get("params", {}),
            "train": base_eval["train"],
            "holdout": base_eval["holdout"],
        },
        "best_candidate": None,
        "accepted": False,
        "trials": int(trials),
    }

    best = best_accept or best_train
    if best is None:
        out["best_candidate"] = {"reason": "no_candidate_passed_gates"}
        return out

    out["best_candidate"] = {
        "strategy": best["cfg"].get("name"),
        "params": best["cfg"].get("params", {}),
        "train": best["eval"]["train"],
        "holdout": best["eval"]["holdout"],
        "verdict": best["verdict"]["verdict"],
        "reasons": best["verdict"]["reasons"],
    }

    # Accept only if at least one candidate passed holdout verification.
    out["accepted"] = best_accept is not None and best is best_accept

    # Optional stress gate: re-run both baseline and accepted candidate under higher transaction costs.
    if out["accepted"] and require_stress:
        stress = _stress_verify(
            dataset=dataset,
            engine=engine,
            base_strategy_cfg=base_strategy_cfg,
            cand_strategy_cfg=best["cfg"],
            bench_cfg=bench_cfg,
            holdout_fraction=holdout_fraction,
            objective=objective,
            min_uplift=min_uplift_stress,
            stress_mult=stress_mult,
            seed=seed,
        )
        out["stress"] = stress
        if not bool(stress.get("pass", False)):
            out["accepted"] = False
            out.setdefault("reject_reasons", []).extend(stress.get("reasons", []))

    if out["accepted"]:
        ts = _ts_id()
        payload = {
            "accepted_at": _utc_iso(),
            "run_id": run_id,
            "run_group_id": run_group_id,
            "run_config_sha": run_config_sha,
            "code_fingerprint": code_fingerprint,
            "strategy": out["best_candidate"]["strategy"],
            "params": out["best_candidate"]["params"],
            "baseline": out["baseline"],
            "train": out["best_candidate"]["train"],
            "holdout": out["best_candidate"]["holdout"],
            "evidence": {
                "data_fingerprint": dataset.fingerprint,
                "provider": dataset.provider,
                "strategy_cfg_sha": sha256_text(json.dumps(base_strategy_cfg, sort_keys=True)),
            },
        }
        versioned = memory_dir / f"best_params.{ts}.json"
        versioned.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        best_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        out["best_params_path"] = str(versioned)

        ablation = _ablation_report(
            base=out["baseline"],
            cand=out["best_candidate"],
            objective=objective,
            stress=out.get("stress"),
            meta={
                "accepted_at": payload["accepted_at"],
                "run_id": run_id,
                "run_group_id": run_group_id,
                "run_config_sha": run_config_sha,
                "code_fingerprint": code_fingerprint,
                "data_fingerprint": dataset.fingerprint,
                "data_provider": dataset.provider,
            },
        )
        abl_dir = memory_dir / "ablations"
        abl_dir.mkdir(parents=True, exist_ok=True)
        abl_json = abl_dir / f"ablation.{ts}.json"
        abl_md = abl_dir / f"ablation.{ts}.md"
        abl_json.write_text(json.dumps(ablation, indent=2, sort_keys=True), encoding="utf-8")
        abl_md.write_text(_render_ablation_md(ablation), encoding="utf-8")
        (memory_dir / "ablation_latest.json").write_text(json.dumps(ablation, indent=2, sort_keys=True), encoding="utf-8")
        out["ablation_paths"] = {"json": str(abl_json), "md": str(abl_md)}

        # Update priors (best-effort) for long-horizon self-improvement.
        try:
            update_priors_on_accept(
                memory_dir=memory_dir,
                strategy_name=str(payload.get("strategy") or ""),
                params=dict(payload.get("params") or {}),
                meta={
                    "accepted_at": payload.get("accepted_at"),
                    "run_id": payload.get("run_id"),
                    "run_group_id": payload.get("run_group_id"),
                    "run_config_sha": payload.get("run_config_sha"),
                    "code_fingerprint": payload.get("code_fingerprint"),
                    "data_fingerprint": payload.get("evidence", {}).get("data_fingerprint"),
                    "objective": out.get("objective"),
                },
            )
        except Exception:
            pass

    return out


def _risk_gates(risk_cfg: Dict[str, Any]) -> Dict[str, float]:
    return {
        "max_drawdown": float(risk_cfg.get("max_drawdown") or 0.35),
        "max_turnover_per_rebalance": float(risk_cfg.get("max_turnover_per_rebalance") or 2.0),
    }


def _objective_value(summary: Dict[str, Any], objective: str) -> float:
    if objective == "median_calmar":
        return float(summary.get("median_calmar", 0.0))
    if objective == "median_sharpe":
        return float(summary.get("median_sharpe", 0.0))
    return float(summary.get(objective, 0.0))


def _verdict(
    base_eval: Dict[str, Any],
    cand_eval: Dict[str, Any],
    objective: str,
    min_uplift: float,
    gates: Dict[str, float],
) -> Dict[str, Any]:
    reasons: List[str] = []

    base_train = base_eval["train"]
    cand_train = cand_eval["train"]
    base_hold = base_eval["holdout"]
    cand_hold = cand_eval["holdout"]

    mdd_train_ok = float(cand_train.get("max_drawdown", 0.0)) <= float(gates["max_drawdown"])
    mdd_hold_ok = float(cand_hold.get("max_drawdown", 0.0)) <= float(gates["max_drawdown"])
    turn_train_ok = float(cand_train.get("turnover_max", 0.0)) <= float(gates["max_turnover_per_rebalance"])
    turn_hold_ok = float(cand_hold.get("turnover_max", 0.0)) <= float(gates["max_turnover_per_rebalance"])

    passes_train = mdd_train_ok and turn_train_ok
    passes_hold = mdd_hold_ok and turn_hold_ok

    if not mdd_train_ok:
        reasons.append("train_mdd_gate_fail")
    if not mdd_hold_ok:
        reasons.append("holdout_mdd_gate_fail")
    if not turn_train_ok:
        reasons.append("train_turnover_gate_fail")
    if not turn_hold_ok:
        reasons.append("holdout_turnover_gate_fail")

    base_train_score = _objective_value(base_train, objective)
    cand_train_score = _objective_value(cand_train, objective)
    base_hold_score = _objective_value(base_hold, objective)
    cand_hold_score = _objective_value(cand_hold, objective)

    # Require uplift on holdout for acceptance.
    uplift_hold = (cand_hold_score - base_hold_score) / (abs(base_hold_score) + 1e-9)
    if uplift_hold < min_uplift:
        reasons.append("holdout_uplift_below_threshold")

    if passes_train and passes_hold and uplift_hold >= min_uplift:
        return {"verdict": "accept", "reasons": reasons, "passes_gates_train": True}

    # Keep track of candidates that pass train gates for best-of selection.
    return {"verdict": "reject", "reasons": reasons, "passes_gates_train": passes_train}


def _eval_split(
    res: Any, dataset: MarketDataset, bench_cfg: Any, holdout_fraction: float
) -> Dict[str, Dict[str, Any]]:
    rs = list(res.returns)
    split = int(len(rs) * (1.0 - float(holdout_fraction)))
    split = max(10, min(split, len(rs) - 10))

    periods_per_year = float(getattr(bench_cfg, "periods_per_year", 365.0))

    train_r = rs[:split]
    hold_r = rs[split:]

    train_eq = equity_from_returns(train_r, start=1.0)
    hold_eq = equity_from_returns(hold_r, start=1.0)

    trades = list(getattr(res, "trades", []) or [])
    train_trades = []
    hold_trades = []
    for t in trades:
        try:
            idx = int(t.get("idx") or 0)
        except Exception:
            idx = 0
        # Trade at bar idx impacts return index (idx-1).
        if idx > 0 and (idx - 1) < split:
            train_trades.append(t)
        elif idx > 0:
            hold_trades.append(t)

    train = summarize(train_r, train_eq, periods_per_year=periods_per_year, trades=train_trades)
    holdout = summarize(hold_r, hold_eq, periods_per_year=periods_per_year, trades=hold_trades)

    # Add simple windowed medians (objective-friendly) using bench walk-forward config.
    wf = getattr(bench_cfg, "walk_forward", None)
    if wf and getattr(wf, "enabled", False):
        win = int(getattr(wf, "window_bars", 0) or 0)
        step = int(getattr(wf, "step_bars", 0) or 0)
        if win > 0 and step > 0:
            train.update(_window_medians(train_r, periods_per_year=periods_per_year, window=win, step=step))
            holdout.update(_window_medians(hold_r, periods_per_year=periods_per_year, window=win, step=step))

    return {"train": _round_summary(train), "holdout": _round_summary(holdout)}


def _window_medians(returns: List[float], periods_per_year: float, window: int, step: int) -> Dict[str, Any]:
    if len(returns) < window:
        return {"median_calmar": 0.0, "median_sharpe": 0.0}
    calmars = []
    sharpes = []
    eq = equity_from_returns(returns, start=1.0)
    # equity is len+1, returns is len
    for start in range(0, max(0, len(returns) - window + 1), step):
        r = returns[start : start + window]
        e = eq[start : start + window + 1]
        s = summarize(r, e, periods_per_year=periods_per_year, trades=None)
        calmars.append(float(s.get("calmar", 0.0)))
        sharpes.append(float(s.get("sharpe", 0.0)))
    med_calmar = statistics.median(calmars) if calmars else 0.0
    med_sharpe = statistics.median(sharpes) if sharpes else 0.0
    return {"median_calmar": float(med_calmar), "median_sharpe": float(med_sharpe)}


def _weighted_choice(rng: random.Random, counts: Dict[str, Any]) -> Optional[str]:
    items = []
    for k, v in (counts or {}).items():
        try:
            c = int(v)
        except Exception:
            c = 0
        if c > 0:
            items.append((str(k), c))
    if not items:
        return None
    total = sum(c for _, c in items)
    r = rng.randint(1, total)
    acc = 0
    for k, c in items:
        acc += c
        if r <= acc:
            return k
    return items[-1][0]


def _sample_candidate_cfg(
    base_strategy_cfg: Dict[str, Any],
    dataset: MarketDataset,
    rng: random.Random,
    priors: Optional[Dict[str, Any]] = None,
    exploit_prob: float = 0.0,
) -> Dict[str, Any]:
    name = str(base_strategy_cfg.get("name") or "")
    base_params = dict(base_strategy_cfg.get("params") or {})
    pc = (priors or {}).get("params_counts") if isinstance(priors, dict) else None

    def pick(param: str, *, cast, fallback: List[Any]) -> Any:
        if pc and rng.random() < float(exploit_prob):
            k = _weighted_choice(rng, pc.get(param) or {})
            if k is not None:
                try:
                    return cast(k)
                except Exception:
                    pass
        return rng.choice(fallback)

    if name == "funding_carry_perp_v1":
        max_k = max(1, len(dataset.symbols) // 2)
        k = pick("k_per_side", cast=int, fallback=[1, 2, 3, 4, 5])
        k = max(1, min(k, max_k))
        cand = dict(base_params)
        cand["k_per_side"] = k
        cand["basis_weight"] = pick("basis_weight", cast=float, fallback=[0.0, 0.2, 0.5, 0.8])
        cand["vol_lookback_bars"] = pick("vol_lookback_bars", cast=int, fallback=[24, 48, 72, 96, 168])
        cand["risk_weighting"] = pick("risk_weighting", cast=str, fallback=["equal", "inverse_vol"])
        cand["target_gross_leverage"] = pick("target_gross_leverage", cast=float, fallback=[0.6, 0.8, 1.0, 1.2, 1.6])
        return {"name": name, "params": cand}

    if name == "momentum_xs_v1":
        max_k = max(1, len(dataset.symbols) // 2)
        k = pick("k_per_side", cast=int, fallback=[1, 2, 3, 4])
        k = max(1, min(k, max_k))
        cand = dict(base_params)
        cand["k_per_side"] = k
        cand["lookback_bars"] = pick("lookback_bars", cast=int, fallback=[168, 336, 720, 1440])
        cand["rebalance_interval_bars"] = pick("rebalance_interval_bars", cast=int, fallback=[24, 48, 168])
        cand["risk_weighting"] = pick("risk_weighting", cast=str, fallback=["equal", "inverse_vol"])
        cand["target_gross_leverage"] = pick("target_gross_leverage", cast=float, fallback=[0.6, 0.8, 1.0, 1.2])
        return {"name": name, "params": cand}

    if name == "mean_reversion_xs_v1":
        max_k = max(1, len(dataset.symbols) // 2)
        k = pick("k_per_side", cast=int, fallback=[1, 2, 3, 4])
        k = max(1, min(k, max_k))
        cand = dict(base_params)
        cand["k_per_side"] = k
        cand["lookback_bars"] = pick("lookback_bars", cast=int, fallback=[6, 12, 24, 48, 72])
        cand["rebalance_interval_bars"] = pick("rebalance_interval_bars", cast=int, fallback=[6, 12, 24])
        cand["risk_weighting"] = pick("risk_weighting", cast=str, fallback=["equal", "inverse_vol"])
        cand["target_gross_leverage"] = pick("target_gross_leverage", cast=float, fallback=[0.6, 0.8, 1.0, 1.2])
        return {"name": name, "params": cand}

    if name == "multi_factor_xs_v1":
        max_k = max(1, len(dataset.symbols) // 2)
        k = pick("k_per_side", cast=int, fallback=[1, 2, 3, 4])
        k = max(1, min(k, max_k))
        cand = dict(base_params)
        cand["k_per_side"] = k

        cand["w_funding"] = pick("w_funding", cast=float, fallback=[0.5, 0.8, 1.0, 1.2, 1.6])
        cand["w_basis"] = pick("w_basis", cast=float, fallback=[0.0, 0.2, 0.5, 0.8])
        cand["w_momentum"] = pick("w_momentum", cast=float, fallback=[0.0, 0.2, 0.5, 0.8])
        cand["w_mean_reversion"] = pick("w_mean_reversion", cast=float, fallback=[0.0, 0.2, 0.5, 0.8])

        cand["momentum_lookback_bars"] = pick("momentum_lookback_bars", cast=int, fallback=[168, 336, 720, 1440])
        cand["mean_reversion_lookback_bars"] = pick("mean_reversion_lookback_bars", cast=int, fallback=[6, 12, 24, 48, 72])
        cand["rebalance_interval_bars"] = pick("rebalance_interval_bars", cast=int, fallback=[8, 24, 48, 168])
        cand["rebalance_on_funding"] = pick("rebalance_on_funding", cast=lambda x: str(x).lower() in {"1", "true", "yes", "on"}, fallback=[False, True])
        cand["use_basis_proxy"] = pick("use_basis_proxy", cast=lambda x: str(x).lower() in {"1", "true", "yes", "on"}, fallback=[True, False])
        cand["risk_weighting"] = pick("risk_weighting", cast=str, fallback=["equal", "inverse_vol"])
        cand["target_gross_leverage"] = pick("target_gross_leverage", cast=float, fallback=[0.6, 0.8, 1.0, 1.2])
        return {"name": name, "params": cand}

    if name in ("nexus_alpha_v1", "nexus_alpha_v1_vol_scaled", "nexus_alpha_v1_regime"):
        max_k = max(1, len(dataset.symbols) // 2)
        k = pick("k_per_side", cast=int, fallback=[1, 2, 3])
        k = max(1, min(k, max_k))
        cand = dict(base_params)
        cand["k_per_side"] = k
        cand["w_carry"] = pick("w_carry", cast=float, fallback=[0.15, 0.25, 0.35, 0.45, 0.55])
        cand["w_mom"] = pick("w_mom", cast=float, fallback=[0.25, 0.35, 0.45, 0.55, 0.65])
        cand["w_confirm"] = pick("w_confirm", cast=float, fallback=[0.0, 0.10, 0.20, 0.30])
        cand["w_mean_reversion"] = pick("w_mean_reversion", cast=float, fallback=[0.0, 0.05, 0.10, 0.20, 0.30])
        cand["w_vol_momentum"] = pick("w_vol_momentum", cast=float, fallback=[0.0, 0.05, 0.10])
        cand["w_funding_trend"] = pick("w_funding_trend", cast=float, fallback=[0.0, 0.05, 0.10])
        cand["rebalance_interval_bars"] = pick("rebalance_interval_bars", cast=int, fallback=[72, 120, 168, 240, 336])
        cand["target_gross_leverage"] = pick("target_gross_leverage", cast=float, fallback=[0.20, 0.30, 0.35, 0.45, 0.55])
        cand["mr_lookback_bars"] = pick("mr_lookback_bars", cast=int, fallback=[24, 36, 48, 72, 96])
        cand["risk_weighting"] = pick("risk_weighting", cast=str, fallback=["equal", "inverse_vol"])
        return {"name": name, "params": cand}

    # Unknown strategy: no-op
    return {"name": name, "params": base_params}


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True))
        f.write("\n")


def _round_summary(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = round(v, 6)
        else:
            out[k] = v
    return out


def _utc_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _ts_id() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _stress_verify(
    dataset: MarketDataset,
    engine: BacktestEngine,
    base_strategy_cfg: Dict[str, Any],
    cand_strategy_cfg: Dict[str, Any],
    bench_cfg: Any,
    holdout_fraction: float,
    objective: str,
    min_uplift: float,
    stress_mult: float,
    seed: int,
) -> Dict[str, Any]:
    from ..backtest.engine import BacktestConfig, BacktestEngine as _Engine

    stress_engine = _Engine(
        BacktestConfig(costs=engine.cfg.costs.with_multiplier(float(stress_mult)))
    )

    base_res = stress_engine.run(dataset=dataset, strategy=make_strategy(base_strategy_cfg), seed=seed + 10001)
    cand_res = stress_engine.run(dataset=dataset, strategy=make_strategy(cand_strategy_cfg), seed=seed + 10002)
    base_eval = _eval_split(base_res, dataset=dataset, bench_cfg=bench_cfg, holdout_fraction=holdout_fraction)
    cand_eval = _eval_split(cand_res, dataset=dataset, bench_cfg=bench_cfg, holdout_fraction=holdout_fraction)

    reasons = []
    base_hold = base_eval["holdout"]
    cand_hold = cand_eval["holdout"]

    obj_key = objective if objective in cand_hold else None
    if obj_key is None:
        # Backward-compat objective resolution
        if objective == "median_calmar" and "median_calmar" in cand_hold:
            obj_key = "median_calmar"
        elif objective == "median_sharpe" and "median_sharpe" in cand_hold:
            obj_key = "median_sharpe"
        elif "calmar" in cand_hold:
            obj_key = "calmar"
        elif "sharpe" in cand_hold:
            obj_key = "sharpe"
        else:
            obj_key = "calmar"
    base_score = float(base_hold.get(obj_key, 0.0))
    cand_score = float(cand_hold.get(obj_key, 0.0))
    uplift = (cand_score - base_score) / (abs(base_score) + 1e-9)
    if uplift < float(min_uplift):
        reasons.append("stress_holdout_uplift_below_threshold")

    return {
        "pass": len(reasons) == 0,
        "reasons": reasons,
        "objective_key": obj_key,
        "min_uplift": float(min_uplift),
        "baseline_holdout": base_hold,
        "candidate_holdout": cand_hold,
        "uplift_holdout": round(uplift, 6),
        "stress_cost_multiplier": float(stress_mult),
    }


def _ablation_report(
    base: Dict[str, Any],
    cand: Dict[str, Any],
    objective: str,
    stress: Optional[Dict[str, Any]],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    b_train = base.get("train") or {}
    b_hold = base.get("holdout") or {}
    c_train = cand.get("train") or {}
    c_hold = cand.get("holdout") or {}

    def uplift(a: float, b: float) -> float:
        return (a - b) / (abs(b) + 1e-9)

    key = objective if objective in c_hold else ("median_calmar" if "median_calmar" in c_hold else "calmar")

    out = {
        "meta": meta,
        "objective": objective,
        "objective_key_used": key,
        "baseline": {"strategy": base.get("strategy"), "params": base.get("params"), "train": b_train, "holdout": b_hold},
        "candidate": {"strategy": cand.get("strategy"), "params": cand.get("params"), "train": c_train, "holdout": c_hold},
        "uplift": {
            "train": round(uplift(float(c_train.get(key, 0.0)), float(b_train.get(key, 0.0))), 6),
            "holdout": round(uplift(float(c_hold.get(key, 0.0)), float(b_hold.get(key, 0.0))), 6),
        },
        "stress": stress,
    }
    return out


def _render_ablation_md(ablation: Dict[str, Any]) -> str:
    meta = ablation.get("meta") or {}
    base = ablation.get("baseline") or {}
    cand = ablation.get("candidate") or {}
    upl = ablation.get("uplift") or {}

    def pick(d: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
        out = {}
        for k in keys:
            if k in d:
                out[k] = d[k]
        return out

    klist = ["total_return", "cagr", "sharpe", "sortino", "max_drawdown", "calmar", "median_calmar", "median_sharpe"]
    lines = []
    lines.append("# Ablation Report (baseline vs accepted candidate)")
    lines.append("")
    lines.append("## Meta")
    for k in ["accepted_at", "run_id", "run_group_id", "run_config_sha", "code_fingerprint", "data_provider", "data_fingerprint"]:
        if k in meta:
            lines.append(f"- {k}: `{meta[k]}`")
    lines.append("")
    lines.append("## Uplift")
    lines.append(f"- train: `{upl.get('train')}`")
    lines.append(f"- holdout: `{upl.get('holdout')}`")
    lines.append("")
    lines.append("## Baseline (train/holdout)")
    btrain = (base.get("train") or {})
    bhold = (base.get("holdout") or {})
    lines.append(f"- strategy: `{base.get('strategy')}`")
    lines.append(f"- params: `{json.dumps(base.get('params') or {}, sort_keys=True)}`")
    lines.append(f"- train: `{json.dumps(pick(btrain, klist), sort_keys=True)}`")
    lines.append(f"- holdout: `{json.dumps(pick(bhold, klist), sort_keys=True)}`")
    lines.append("")
    lines.append("## Candidate (train/holdout)")
    ctrain = (cand.get("train") or {})
    chold = (cand.get("holdout") or {})
    lines.append(f"- strategy: `{cand.get('strategy')}`")
    lines.append(f"- params: `{json.dumps(cand.get('params') or {}, sort_keys=True)}`")
    lines.append(f"- train: `{json.dumps(pick(ctrain, klist), sort_keys=True)}`")
    lines.append(f"- holdout: `{json.dumps(pick(chold, klist), sort_keys=True)}`")
    lines.append("")

    stress = ablation.get("stress")
    if stress:
        lines.append("## Stress (higher costs)")
        lines.append(f"- pass: `{stress.get('pass')}`")
        lines.append(f"- uplift_holdout: `{stress.get('uplift_holdout')}`")
        lines.append(f"- reasons: `{json.dumps(stress.get('reasons') or [])}`")
        lines.append("")

    return "\n".join(lines)
