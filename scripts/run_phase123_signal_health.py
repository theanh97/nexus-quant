#!/usr/bin/env python3
"""
Phase 123: Signal Health Monitor & Attribution Analysis
=========================================================
Runs each P91b sub-signal independently across 5 years + computes:
1. Per-signal Sharpe per year (standalone)
2. Signal contribution & marginal value in ensemble
3. Correlation matrix (inter-signal diversification)
4. Health assessment with degradation alerts
5. Optimal weight suggestion based on recent performance

Critical finding from Phase 118: V1 = 2.76 Sharpe in 2026 OOS,
idio signals turned negative. Must track signal health continuously.
"""
import json
import os
import signal as _signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from nexus_quant.backtest.engine import BacktestConfig, BacktestEngine
from nexus_quant.backtest.costs import cost_model_from_config
from nexus_quant.data.providers.registry import make_provider
from nexus_quant.strategies.registry import make_strategy

# ── Timeout protection ──
_partial_results = {}

def _timeout_handler(signum, frame):
    print("\n⏰ TIMEOUT — saving partial results...")
    save_report(_partial_results, partial=True)
    sys.exit(0)

_signal.signal(_signal.SIGALRM, _timeout_handler)
_signal.alarm(600)  # 10 min max

# ── Load production config for signal definitions ──
PROD_CFG = json.loads((ROOT / "configs" / "production_p91b_champion.json").read_text())

SYMBOLS = PROD_CFG["data"]["symbols"]
ENSEMBLE_WEIGHTS = PROD_CFG["ensemble"]["weights"]
SIGNAL_DEFS = PROD_CFG["ensemble"]["signals"]

YEAR_RANGES = {
    "2021": ("2021-02-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
}

OUT_DIR = ROOT / "artifacts" / "phase123"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_signal_config(sig_key: str, start: str, end: str) -> dict:
    """Build a complete backtest config for a single sub-signal."""
    sig_def = SIGNAL_DEFS[sig_key]
    return {
        "run_name": f"health_{sig_key}",
        "seed": 42,
        "data": {
            "provider": "binance_rest_v1",
            "symbols": SYMBOLS,
            "start": start,
            "end": end,
            "bar_interval": "1h",
            "cache_dir": ".cache/binance_rest",
        },
        "execution": {"style": "taker", "slippage_bps": 3.0},
        "strategy": {
            "name": sig_def["strategy"],
            "params": sig_def["params"],
        },
        "costs": {"fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0},
    }


def run_backtest(cfg: dict) -> dict:
    """Run a single backtest, return metrics + equity curve."""
    provider = make_provider(cfg["data"], seed=cfg.get("seed", 42))
    dataset = provider.load()

    cost_cfg = cfg.get("costs", {})
    cost_model = cost_model_from_config(cost_cfg)
    bt_cfg = BacktestConfig(costs=cost_model)

    strat = make_strategy(cfg["strategy"])
    engine = BacktestEngine(bt_cfg)
    result = engine.run(dataset, strat)

    # Compute Sharpe from hourly returns
    rets = np.array(result.returns, dtype=np.float64)
    sharpe = sharpe_from_returns(rets)

    # Max drawdown from equity curve
    eq = np.array(result.equity_curve, dtype=np.float64)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.maximum(peak, 1e-8)
    mdd = float(np.min(dd)) * 100

    return {
        "sharpe": sharpe,
        "max_drawdown": round(mdd, 2),
        "equity_curve": list(result.equity_curve),
        "returns": list(result.returns),
    }


def sharpe_from_returns(rets: np.ndarray) -> float:
    """Annualized Sharpe from hourly returns."""
    if len(rets) < 100:
        return 0.0
    mu = float(np.mean(rets)) * 8760
    sigma = float(np.std(rets)) * np.sqrt(8760)
    return round(mu / sigma, 4) if sigma > 1e-12 else 0.0


def save_report(data: dict, partial: bool = False) -> None:
    """Save report to JSON."""
    data["partial"] = partial
    data["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path = OUT_DIR / "signal_health_report.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → Saved: {path}")


def main():
    global _partial_results
    t0 = time.time()
    print("=" * 70)
    print("Phase 123: Signal Health Monitor & Attribution Analysis")
    print("=" * 70)

    # ── 1. Per-signal per-year standalone Sharpe ──
    print("\n[1/4] Per-signal per-year standalone Sharpe...")
    signal_sharpes = {}
    signal_eq_curves = {}  # sig_key -> year -> equity_curve

    for sig_key in SIGNAL_DEFS:
        print(f"\n  {sig_key} ({SIGNAL_DEFS[sig_key]['strategy']}):")
        signal_sharpes[sig_key] = {}
        signal_eq_curves[sig_key] = {}

        for year, (start, end) in YEAR_RANGES.items():
            try:
                cfg = make_signal_config(sig_key, start, end)
                result = run_backtest(cfg)
                sharpe = result["sharpe"]
                mdd = result["max_drawdown"]
                signal_sharpes[sig_key][year] = sharpe
                signal_eq_curves[sig_key][year] = result["equity_curve"]

                tag = "✓" if sharpe > 0.5 else ("⚠" if sharpe > 0 else "✗")
                print(f"    {year}: Sharpe={sharpe:7.3f}  MDD={mdd:6.2f}%  {tag}")
            except Exception as e:
                print(f"    {year}: ERROR — {e}")
                signal_sharpes[sig_key][year] = None
                signal_eq_curves[sig_key][year] = []

    _partial_results = {
        "phase": 123,
        "description": "Signal Health Monitor",
        "signal_sharpes": signal_sharpes,
        "ensemble_weights": ENSEMBLE_WEIGHTS,
    }

    # ── 2. Signal contribution (ensemble weighted) ──
    print("\n[2/4] Signal contribution analysis...")
    contributions = {}

    for year in YEAR_RANGES:
        # Extract returns for each signal
        sig_returns = {}
        min_len = float("inf")

        for sig_key in SIGNAL_DEFS:
            eq = signal_eq_curves.get(sig_key, {}).get(year, [])
            if len(eq) > 1:
                eq_arr = np.array(eq, dtype=np.float64)
                rets = np.diff(eq_arr) / eq_arr[:-1]
                sig_returns[sig_key] = rets
                min_len = min(min_len, len(rets))

        if min_len < 100 or min_len == float("inf"):
            contributions[year] = {"error": "insufficient data"}
            continue

        # Trim to same length
        for k in sig_returns:
            sig_returns[k] = sig_returns[k][:int(min_len)]

        # Compute ensemble returns
        ens_rets = np.zeros(int(min_len))
        for sig_key, w in ENSEMBLE_WEIGHTS.items():
            if sig_key in sig_returns:
                ens_rets += w * sig_returns[sig_key]

        ens_sharpe = sharpe_from_returns(ens_rets)

        year_contrib = {"ensemble_sharpe": ens_sharpe}
        for sig_key in SIGNAL_DEFS:
            if sig_key not in sig_returns:
                continue
            w = ENSEMBLE_WEIGHTS[sig_key]
            weighted = w * sig_returns[sig_key]

            # Leave-one-out marginal contribution
            loo = ens_rets - weighted
            loo_sharpe = sharpe_from_returns(loo)
            marginal = ens_sharpe - loo_sharpe

            # Return correlation with ensemble
            corr = float(np.corrcoef(sig_returns[sig_key], ens_rets)[0, 1])

            year_contrib[sig_key] = {
                "standalone": signal_sharpes[sig_key].get(year, 0),
                "weight": w,
                "marginal": round(marginal, 4),
                "corr_with_ensemble": round(corr, 4),
            }

        contributions[year] = year_contrib
        print(f"  {year}: Ensemble Sharpe = {ens_sharpe:.3f}")
        for sig_key in SIGNAL_DEFS:
            if sig_key in year_contrib and isinstance(year_contrib[sig_key], dict):
                c = year_contrib[sig_key]
                print(f"    {sig_key:12s}: stand={c['standalone']:6.3f}  marg={c['marginal']:+6.3f}  ρ={c['corr_with_ensemble']:5.3f}")

    _partial_results["contributions"] = contributions

    # ── 3. Inter-signal correlation ──
    print("\n[3/4] Inter-signal correlation matrix...")
    correlations = {}

    for year in YEAR_RANGES:
        sig_rets = {}
        min_len = float("inf")
        for sig_key in SIGNAL_DEFS:
            eq = signal_eq_curves.get(sig_key, {}).get(year, [])
            if len(eq) > 1:
                eq_arr = np.array(eq, dtype=np.float64)
                sig_rets[sig_key] = np.diff(eq_arr) / eq_arr[:-1]
                min_len = min(min_len, len(sig_rets[sig_key]))

        if min_len < 100 or min_len == float("inf"):
            continue

        keys = sorted(sig_rets.keys())
        matrix = {}
        for i, k1 in enumerate(keys):
            for k2 in keys[i:]:
                r1 = sig_rets[k1][:int(min_len)]
                r2 = sig_rets[k2][:int(min_len)]
                c = float(np.corrcoef(r1, r2)[0, 1])
                matrix[f"{k1}__{k2}"] = round(c, 4)

        off_diag = [v for k, v in matrix.items()
                    if k.split("__")[0] != k.split("__")[1]]
        avg_corr = float(np.mean(off_diag)) if off_diag else 0
        correlations[year] = {"matrix": matrix, "avg_pairwise": round(avg_corr, 4)}
        print(f"  {year}: avg pairwise ρ = {avg_corr:.3f}")

    _partial_results["correlations"] = correlations

    # ── 4. Health assessment ──
    print("\n[4/4] Signal health assessment...")
    health = {}
    alerts = []

    for sig_key in SIGNAL_DEFS:
        sharpes = signal_sharpes[sig_key]
        all_vals = [v for v in sharpes.values() if v is not None]
        recent = [sharpes.get(y) for y in ["2024", "2025"] if sharpes.get(y) is not None]
        early = [sharpes.get(y) for y in ["2021", "2022", "2023"] if sharpes.get(y) is not None]

        avg_all = float(np.mean(all_vals)) if all_vals else 0
        avg_recent = float(np.mean(recent)) if recent else 0
        avg_early = float(np.mean(early)) if early else 0

        # Trend: recent vs early
        if avg_early != 0 and early:
            trend = (avg_recent - avg_early) / max(abs(avg_early), 0.1)
        else:
            trend = 0

        # Status
        if avg_recent < 0:
            status = "CRITICAL"
            alerts.append(f"ALERT: {sig_key} negative recent Sharpe ({avg_recent:.3f})")
        elif avg_recent < 0.5:
            status = "DEGRADED"
            alerts.append(f"WARNING: {sig_key} recent Sharpe < 0.5 ({avg_recent:.3f})")
        elif trend < -0.30:
            status = "DECLINING"
            alerts.append(f"NOTICE: {sig_key} declining ({trend:+.1%} vs historical)")
        else:
            status = "HEALTHY"

        # Marginal contribution trend
        marg_recent = []
        for y in ["2024", "2025"]:
            c = contributions.get(y, {}).get(sig_key, {})
            if isinstance(c, dict) and "marginal" in c:
                marg_recent.append(c["marginal"])
        avg_marginal = float(np.mean(marg_recent)) if marg_recent else 0

        health[sig_key] = {
            "status": status,
            "avg_sharpe_all": round(avg_all, 4),
            "avg_sharpe_recent": round(avg_recent, 4),
            "avg_sharpe_early": round(avg_early, 4),
            "trend_pct": round(trend * 100, 1),
            "avg_marginal_recent": round(avg_marginal, 4),
            "yearly": {k: v for k, v in sharpes.items() if v is not None},
        }

        icon = {"HEALTHY": "✓", "DECLINING": "↘", "DEGRADED": "⚠", "CRITICAL": "✗"}.get(status, "?")
        print(f"  {icon} {sig_key:12s}: {status:10s}  all={avg_all:6.3f}  recent={avg_recent:6.3f}  trend={trend:+.1%}  marg={avg_marginal:+.4f}")

    # Optimal weights (recent-Sharpe-proportional, floored at 5%)
    print("\n  Suggested weights (recent-performance-weighted, floor=5%):")
    raw = {}
    for sig_key in SIGNAL_DEFS:
        s = health[sig_key]["avg_sharpe_recent"]
        raw[sig_key] = max(s, 0.05)  # floor so no signal gets zero

    total = sum(raw.values())
    optimal = {k: round(v / total, 4) for k, v in raw.items()}

    for k in SIGNAL_DEFS:
        cur = ENSEMBLE_WEIGHTS[k]
        sug = optimal[k]
        delta = sug - cur
        print(f"    {k:12s}: current={cur:.4f}  suggested={sug:.4f}  ({delta:+.4f})")

    _partial_results.update({
        "health": health,
        "alerts": alerts,
        "optimal_weights": optimal,
        "current_weights": dict(ENSEMBLE_WEIGHTS),
    })

    # ── Summary ──
    elapsed = time.time() - t0
    _partial_results["elapsed_seconds"] = round(elapsed, 1)

    # Compute signal stability score (% of years with Sharpe > 0.5)
    stability = {}
    for sig_key in SIGNAL_DEFS:
        vals = [v for v in signal_sharpes[sig_key].values() if v is not None]
        above_05 = sum(1 for v in vals if v > 0.5)
        stability[sig_key] = round(above_05 / max(len(vals), 1) * 100, 1)
    _partial_results["stability_pct_above_05"] = stability

    save_report(_partial_results, partial=False)

    print(f"\n{'='*70}")
    print(f"Phase 123 complete in {elapsed:.0f}s")
    if alerts:
        print(f"\n⚠ {len(alerts)} alerts:")
        for a in alerts:
            print(f"  - {a}")
    else:
        print("  All signals healthy.")

    # Signal health summary table
    print(f"\n  Signal Health Summary:")
    print(f"  {'Signal':12s} {'Status':10s} {'All':>7s} {'Recent':>7s} {'Trend':>7s} {'Stability':>10s} {'Weight':>7s}")
    for sig_key in SIGNAL_DEFS:
        h = health[sig_key]
        print(f"  {sig_key:12s} {h['status']:10s} {h['avg_sharpe_all']:7.3f} {h['avg_sharpe_recent']:7.3f} {h['trend_pct']:+6.1f}% {stability[sig_key]:9.1f}% {ENSEMBLE_WEIGHTS[sig_key]:7.4f}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
