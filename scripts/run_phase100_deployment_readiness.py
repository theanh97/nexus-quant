#!/usr/bin/env python3
"""
Phase 100: Deployment Readiness — Cost Sensitivity + Leverage Optimization
===========================================================================
Final pre-deployment tests:

  A) CONTINUOUS vs PER-YEAR: Run P91b as one continuous 5yr backtest
     to reconcile with per-year numbers.

  B) COST SENSITIVITY: How robust is P91b to different cost assumptions?
     Test fee_rate × slippage_rate grid.

  C) LEVERAGE OPTIMIZATION: Is 0.35 target_gross_leverage optimal?
     Sweep from 0.15 to 0.60.

  D) EXPECTED LIVE PERFORMANCE: Conservative estimate with realistic costs.
"""

import copy, json, math, os, sys, time
from pathlib import Path

PROJ = "/Users/qtmobile/Desktop/Nexus - Quant Trading "
sys.path.insert(0, PROJ)

import numpy as np

from nexus_quant.data.providers.registry import make_provider
from nexus_quant.strategies.registry import make_strategy
from nexus_quant.backtest.engine import BacktestConfig, BacktestEngine
from nexus_quant.backtest.costs import cost_model_from_config

OUT_DIR = os.path.join(PROJ, "artifacts", "phase100")
os.makedirs(OUT_DIR, exist_ok=True)

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
]

YEARS = ["2021", "2022", "2023", "2024", "2025"]
YEAR_RANGES = {
    "2021": ("2021-01-01", "2022-01-01"),
    "2022": ("2022-01-01", "2023-01-01"),
    "2023": ("2023-01-01", "2024-01-01"),
    "2024": ("2024-01-01", "2025-01-01"),
    "2025": ("2025-01-01", "2026-01-01"),
}
FULL_RANGE = ("2021-01-01", "2026-01-01")
OOS_RANGE = ("2026-01-01", "2026-02-20")

P91B_WEIGHTS = {"v1": 0.2747, "i460bw168": 0.1967, "i415bw216": 0.3247, "f144": 0.2039}

def make_signals(lev_override=None):
    sigs = {
        "v1": {"name": "nexus_alpha_v1", "params": {
            "k_per_side": 2, "w_carry": 0.35, "w_mom": 0.45, "w_mean_reversion": 0.20,
            "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 72,
            "vol_lookback_bars": 168, "target_gross_leverage": 0.35, "rebalance_interval_bars": 60}},
        "i460bw168": {"name": "idio_momentum_alpha", "params": {
            "k_per_side": 4, "lookback_bars": 460, "beta_window_bars": 168,
            "target_gross_leverage": 0.30, "rebalance_interval_bars": 48}},
        "i415bw216": {"name": "idio_momentum_alpha", "params": {
            "k_per_side": 4, "lookback_bars": 415, "beta_window_bars": 216,
            "target_gross_leverage": 0.30, "rebalance_interval_bars": 48}},
        "f144": {"name": "funding_momentum_alpha", "params": {
            "k_per_side": 2, "funding_lookback_bars": 144, "direction": "contrarian",
            "target_gross_leverage": 0.25, "rebalance_interval_bars": 24}},
    }
    if lev_override is not None:
        for sig in sigs.values():
            sig["params"]["target_gross_leverage"] = lev_override
    return sigs


def log(msg):
    print(f"[P100] {msg}", flush=True)


def compute_sharpe(returns, bars_per_year=8760):
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size < 50:
        return 0.0
    std = float(np.std(arr))
    if std <= 0:
        return 0.0
    return float(np.mean(arr) / std * np.sqrt(bars_per_year))


def blend_returns(sig_rets, weights):
    keys = sorted(weights.keys())
    n = min(len(sig_rets.get(k, [])) for k in keys)
    if n == 0:
        return []
    R = np.zeros((len(keys), n), dtype=np.float64)
    W = np.array([weights[k] for k in keys], dtype=np.float64)
    for i, k in enumerate(keys):
        R[i, :] = sig_rets[k][:n]
    return (W @ R).tolist()


def run_signal(sig_cfg, start, end, fee_rate=0.0005, slippage_rate=0.0003):
    data_cfg = {
        "provider": "binance_rest_v1",
        "symbols": SYMBOLS,
        "start": start, "end": end,
        "bar_interval": "1h",
        "cache_dir": ".cache/binance_rest",
    }
    costs_cfg = {"fee_rate": fee_rate, "slippage_rate": slippage_rate}
    exec_cfg = {"style": "taker", "slippage_bps": 3.0}

    provider = make_provider(data_cfg, seed=42)
    dataset = provider.load()
    strategy = make_strategy({"name": sig_cfg["name"], "params": copy.deepcopy(sig_cfg["params"])})
    cost_model = cost_model_from_config(costs_cfg, execution_cfg=exec_cfg)
    engine = BacktestEngine(BacktestConfig(costs=cost_model))
    result = engine.run(dataset=dataset, strategy=strategy, seed=42)
    return result.returns


def run_blend(start, end, signal_cfgs=None, fee_rate=0.0005, slippage_rate=0.0003):
    cfgs = signal_cfgs or make_signals()
    sig_rets = {}
    for sig_key in sorted(P91B_WEIGHTS.keys()):
        try:
            rets = run_signal(cfgs[sig_key], start, end, fee_rate, slippage_rate)
        except Exception as exc:
            log(f"    {sig_key} ERROR: {exc}")
            rets = []
        sig_rets[sig_key] = rets
    return blend_returns(sig_rets, P91B_WEIGHTS)


def compute_yearly_from_continuous(returns, bars_per_year_list):
    """Split continuous returns into per-year chunks and compute Sharpe."""
    idx = 0
    yearly_sharpes = {}
    for year, n_bars in zip(YEARS, bars_per_year_list):
        chunk = returns[idx:idx + n_bars]
        yearly_sharpes[year] = round(compute_sharpe(chunk, 8760), 4)
        idx += n_bars
    return yearly_sharpes


if __name__ == "__main__":
    t0 = time.time()
    report = {"phase": 100}

    # ════════════════════════════════════
    # SECTION A: Continuous 5yr backtest
    # ════════════════════════════════════
    log("=" * 60)
    log("SECTION A: Continuous 5yr backtest (2021-2025)")
    log("=" * 60)

    cfgs = make_signals()
    continuous_rets = run_blend(FULL_RANGE[0], FULL_RANGE[1], cfgs)
    overall_sharpe = round(compute_sharpe(continuous_rets, 8760), 4)
    log(f"  5yr continuous Sharpe = {overall_sharpe}")

    # Split into per-year
    total_bars = len(continuous_rets)
    bars_2021 = 8759
    bars_2022 = 8759
    bars_2023 = 8759
    bars_2024 = 8783
    bars_2025 = total_bars - bars_2021 - bars_2022 - bars_2023 - bars_2024
    log(f"  Total bars={total_bars}, 2025 gets {bars_2025} bars")
    yearly_cont = compute_yearly_from_continuous(
        continuous_rets, [bars_2021, bars_2022, bars_2023, bars_2024, bars_2025]
    )
    vals = list(yearly_cont.values())
    log(f"  Per-year: {yearly_cont}")
    log(f"  AVG={sum(vals)/len(vals):.4f}, MIN={min(vals):.4f}")

    # Per-year (separate backtests)
    log("\n  Per-year separate backtests:")
    yearly_sep = {}
    for year in YEARS:
        start, end = YEAR_RANGES[year]
        rets = run_blend(start, end, cfgs)
        s = round(compute_sharpe(rets, 8760), 4)
        yearly_sep[year] = s
        log(f"    {year}: Sharpe={s}")
    vals_sep = list(yearly_sep.values())

    report["continuous_vs_peryear"] = {
        "continuous_5yr_sharpe": overall_sharpe,
        "continuous_yearly": yearly_cont,
        "continuous_avg": round(sum(vals) / len(vals), 4),
        "continuous_min": round(min(vals), 4),
        "peryear_yearly": yearly_sep,
        "peryear_avg": round(sum(vals_sep) / len(vals_sep), 4),
        "peryear_min": round(min(vals_sep), 4),
    }

    # ════════════════════════════════════
    # SECTION B: Cost sensitivity
    # ════════════════════════════════════
    log("\n" + "=" * 60)
    log("SECTION B: Cost sensitivity")
    log("=" * 60)

    COST_GRID = [
        ("ultra_low",  0.0002, 0.0001),   # maker fees, tight spreads
        ("low",        0.0003, 0.0002),   # VIP taker
        ("baseline",   0.0005, 0.0003),   # standard
        ("high",       0.0007, 0.0005),   # worst case
        ("ultra_high", 0.0010, 0.0007),   # retail + wide spreads
    ]

    cost_results = {}
    for label, fee, slip in COST_GRID:
        log(f"\n  {label} (fee={fee:.4f}, slip={slip:.4f})")
        yearly = {}
        for year in YEARS:
            start, end = YEAR_RANGES[year]
            rets = run_blend(start, end, cfgs, fee_rate=fee, slippage_rate=slip)
            s = round(compute_sharpe(rets, 8760), 4)
            yearly[year] = s
        vals = list(yearly.values())
        avg_s = round(sum(vals) / len(vals), 4)
        min_s = round(min(vals), 4)
        log(f"    AVG={avg_s:.4f}, MIN={min_s:.4f}")
        cost_results[label] = {
            "fee_rate": fee, "slippage_rate": slip,
            "yearly": yearly, "avg": avg_s, "min": min_s,
        }
    report["cost_sensitivity"] = cost_results

    # ════════════════════════════════════
    # SECTION C: Leverage optimization
    # ════════════════════════════════════
    log("\n" + "=" * 60)
    log("SECTION C: Leverage optimization")
    log("=" * 60)

    LEV_GRID = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
    lev_results = {}
    for lev in LEV_GRID:
        label = f"lev_{int(lev*100)}"
        log(f"\n  {label} (target_gross_leverage={lev:.2f})")
        cfgs_lev = make_signals(lev_override=lev)
        yearly = {}
        for year in YEARS:
            start, end = YEAR_RANGES[year]
            rets = run_blend(start, end, cfgs_lev)
            s = round(compute_sharpe(rets, 8760), 4)
            yearly[year] = s
        vals = list(yearly.values())
        avg_s = round(sum(vals) / len(vals), 4)
        min_s = round(min(vals), 4)
        log(f"    AVG={avg_s:.4f}, MIN={min_s:.4f}")
        lev_results[label] = {
            "leverage": lev,
            "yearly": yearly, "avg": avg_s, "min": min_s,
        }
    report["leverage_optimization"] = lev_results

    # ════════════════════════════════════
    # SECTION D: Expected live estimate
    # ════════════════════════════════════
    log("\n" + "=" * 60)
    log("SECTION D: Expected live performance")
    log("=" * 60)

    # Use high-cost estimate for conservative projection
    high_cost = cost_results.get("high", {})
    baseline_cost = cost_results.get("baseline", {})

    report["live_estimate"] = {
        "conservative": {
            "note": "high costs (fee=0.07%, slip=0.05%)",
            "expected_sharpe_range": f"{high_cost.get('min', 0):.2f} to {high_cost.get('avg', 0):.2f}",
            "is_min": high_cost.get("min", 0),
            "is_avg": high_cost.get("avg", 0),
            "live_degradation": "40-50% from IS",
            "expected_live_sharpe": round(high_cost.get("min", 0) * 0.6, 2),
        },
        "realistic": {
            "note": "baseline costs (fee=0.05%, slip=0.03%)",
            "is_min": baseline_cost.get("min", 0),
            "is_avg": baseline_cost.get("avg", 0),
            "expected_live_sharpe": round(baseline_cost.get("min", 0) * 0.6, 2),
        },
    }

    # ════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════
    elapsed = round(time.time() - t0, 1)
    report["elapsed_seconds"] = elapsed

    out_path = os.path.join(OUT_DIR, "phase100_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    log("\n" + "=" * 60)
    log(f"Phase 100 COMPLETE in {elapsed}s → {out_path}")
    log("=" * 60)

    log("\nCost Sensitivity:")
    log(f"  {'Label':>12} | {'Fee':>6} | {'Slip':>6} | {'AVG':>8} | {'MIN':>8}")
    log(f"  {'-'*12}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}")
    for label, data in cost_results.items():
        log(f"  {label:>12} | {data['fee_rate']:>6.4f} | {data['slippage_rate']:>6.4f} | {data['avg']:>8.4f} | {data['min']:>8.4f}")

    log("\nLeverage Optimization:")
    log(f"  {'Leverage':>10} | {'AVG':>8} | {'MIN':>8}")
    log(f"  {'-'*10}-+-{'-'*8}-+-{'-'*8}")
    for label, data in lev_results.items():
        marker = " ★" if data['leverage'] == 0.35 else ""
        log(f"  {data['leverage']:>10.2f} | {data['avg']:>8.4f} | {data['min']:>8.4f}{marker}")
