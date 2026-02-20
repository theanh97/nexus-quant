#!/usr/bin/env python3
"""
Phase 98: Timeframe + Long-Bias Experiments
============================================
Two fundamentally different axes we haven't explored:

  A) 4-HOUR TIMEFRAME: Same P91b signals on 4h bars.
     - 4h_match_hours: adjust lookback bars to match same wall-clock hours
       (336h → 84 bars, 72h → 18 bars, 168h → 42 bars, 60h → 15 bars)
     - 4h_match_bars: keep same bar counts as 1h (longer wall-clock lookback)
       (336 bars → 1344h, 72 bars → 288h, etc.)

  B) LONG-BIAS OVERLAY: P91b is dollar-neutral. Crypto has massive positive drift.
     Add a beta-exposure overlay: blended_return + beta × market_return
     Test beta = 0.05, 0.10, 0.15, 0.20

  C) Compare all variants on Sharpe, MIN, and 2026 OOS.
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

OUT_DIR = os.path.join(PROJ, "artifacts", "phase98")
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
OOS_RANGE = ("2026-01-01", "2026-02-20")

# P91b champion weights
P91B_WEIGHTS = {"v1": 0.2747, "i460bw168": 0.1967, "i415bw216": 0.3247, "f144": 0.2039}

# ── Signal configs for 1h (baseline) ──
SIGNALS_1H = {
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

# ── Signal configs for 4h (match wall-clock hours) ──
SIGNALS_4H_MATCH_HOURS = {
    "v1": {"name": "nexus_alpha_v1", "params": {
        "k_per_side": 2, "w_carry": 0.35, "w_mom": 0.45, "w_mean_reversion": 0.20,
        "momentum_lookback_bars": 84, "mean_reversion_lookback_bars": 18,
        "vol_lookback_bars": 42, "target_gross_leverage": 0.35, "rebalance_interval_bars": 15}},
    "i460bw168": {"name": "idio_momentum_alpha", "params": {
        "k_per_side": 4, "lookback_bars": 115, "beta_window_bars": 42,
        "target_gross_leverage": 0.30, "rebalance_interval_bars": 12}},
    "i415bw216": {"name": "idio_momentum_alpha", "params": {
        "k_per_side": 4, "lookback_bars": 104, "beta_window_bars": 54,
        "target_gross_leverage": 0.30, "rebalance_interval_bars": 12}},
    "f144": {"name": "funding_momentum_alpha", "params": {
        "k_per_side": 2, "funding_lookback_bars": 36, "direction": "contrarian",
        "target_gross_leverage": 0.25, "rebalance_interval_bars": 6}},
}

# ── Signal configs for 4h (keep same bar counts = longer lookback in hours) ──
SIGNALS_4H_MATCH_BARS = copy.deepcopy(SIGNALS_1H)
# Same bar counts, but 4h bars → 4x longer in wall clock


def log(msg):
    print(f"[P98] {msg}", flush=True)


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


def run_signal(sig_cfg, symbols, start, end, bar_interval="1h"):
    data_cfg = {
        "provider": "binance_rest_v1",
        "symbols": symbols,
        "start": start, "end": end,
        "bar_interval": bar_interval,
        "cache_dir": ".cache/binance_rest",
    }
    costs_cfg = {"fee_rate": 0.0005, "slippage_rate": 0.0003}
    exec_cfg = {"style": "taker", "slippage_bps": 3.0}

    provider = make_provider(data_cfg, seed=42)
    dataset = provider.load()
    strategy = make_strategy({"name": sig_cfg["name"], "params": copy.deepcopy(sig_cfg["params"])})
    cost_model = cost_model_from_config(costs_cfg, execution_cfg=exec_cfg)
    engine = BacktestEngine(BacktestConfig(costs=cost_model))
    result = engine.run(dataset=dataset, strategy=strategy, seed=42)
    return result.returns


def get_market_returns(symbols, start, end, bar_interval="1h"):
    """Equal-weighted index returns for the symbol basket."""
    data_cfg = {
        "provider": "binance_rest_v1",
        "symbols": symbols,
        "start": start, "end": end,
        "bar_interval": bar_interval,
        "cache_dir": ".cache/binance_rest",
    }
    provider = make_provider(data_cfg, seed=42)
    dataset = provider.load()
    n_bars = len(dataset.perp_close[symbols[0]])
    mkt_rets = []
    for t in range(1, n_bars):
        bar_ret = 0.0
        for s in symbols:
            c_prev = float(dataset.perp_close[s][t - 1])
            c_now = float(dataset.perp_close[s][t])
            if c_prev > 0:
                bar_ret += (c_now / c_prev - 1.0)
        bar_ret /= len(symbols)
        mkt_rets.append(bar_ret)
    return mkt_rets


def run_blend(signal_cfgs, symbols, start, end, bar_interval="1h"):
    """Run all 4 signals and blend with P91b weights. Return blended returns."""
    sig_rets = {}
    for sig_key in sorted(P91B_WEIGHTS.keys()):
        try:
            rets = run_signal(signal_cfgs[sig_key], symbols, start, end, bar_interval)
        except Exception as exc:
            log(f"    {sig_key} ERROR: {exc}")
            rets = []
        sig_rets[sig_key] = rets
    return blend_returns(sig_rets, P91B_WEIGHTS)


# ════════════════════════════════════════
# MAIN
# ════════════════════════════════════════
if __name__ == "__main__":
    t0 = time.time()
    report = {"phase": 98, "experiments": {}}

    # ════════════════════════════════════
    # SECTION A: 1h baseline (verify P91b)
    # ════════════════════════════════════
    log("=" * 60)
    log("SECTION A: P91b baseline (1h) — verify")
    log("=" * 60)

    baseline_sharpes = {}
    baseline_rets_by_year = {}
    for year in YEARS:
        start, end = YEAR_RANGES[year]
        log(f"  1h_baseline / {year}")
        rets = run_blend(SIGNALS_1H, SYMBOLS, start, end, "1h")
        s = compute_sharpe(rets, 8760)
        baseline_sharpes[year] = round(s, 4)
        baseline_rets_by_year[year] = rets
        log(f"    Sharpe={s:.4f}")

    # OOS
    log("  1h_baseline / 2026 OOS")
    oos_rets_1h = run_blend(SIGNALS_1H, SYMBOLS, OOS_RANGE[0], OOS_RANGE[1], "1h")
    oos_1h = round(compute_sharpe(oos_rets_1h, 8760), 4)
    log(f"    OOS Sharpe={oos_1h:.4f}")

    vals = list(baseline_sharpes.values())
    report["experiments"]["1h_baseline"] = {
        "yearly": baseline_sharpes,
        "avg": round(sum(vals) / len(vals), 4),
        "min": round(min(vals), 4),
        "oos_2026": oos_1h,
        "bar_interval": "1h",
    }

    # ════════════════════════════════════
    # SECTION B: 4h match-hours
    # ════════════════════════════════════
    log("=" * 60)
    log("SECTION B: 4h timeframe (match wall-clock hours)")
    log("=" * 60)

    sharpes_4h_mh = {}
    for year in YEARS:
        start, end = YEAR_RANGES[year]
        log(f"  4h_match_hours / {year}")
        rets = run_blend(SIGNALS_4H_MATCH_HOURS, SYMBOLS, start, end, "4h")
        s = compute_sharpe(rets, 2190)  # 8760/4 = 2190 4h-bars per year
        sharpes_4h_mh[year] = round(s, 4)
        log(f"    Sharpe={s:.4f}")

    log("  4h_match_hours / 2026 OOS")
    oos_rets_4h_mh = run_blend(SIGNALS_4H_MATCH_HOURS, SYMBOLS, OOS_RANGE[0], OOS_RANGE[1], "4h")
    oos_4h_mh = round(compute_sharpe(oos_rets_4h_mh, 2190), 4)
    log(f"    OOS Sharpe={oos_4h_mh:.4f}")

    vals = list(sharpes_4h_mh.values())
    report["experiments"]["4h_match_hours"] = {
        "yearly": sharpes_4h_mh,
        "avg": round(sum(vals) / len(vals), 4),
        "min": round(min(vals), 4),
        "oos_2026": oos_4h_mh,
        "bar_interval": "4h",
        "note": "lookback params scaled to match same wall-clock hours as 1h",
    }

    # ════════════════════════════════════
    # SECTION C: 4h match-bars
    # ════════════════════════════════════
    log("=" * 60)
    log("SECTION C: 4h timeframe (keep same bar counts = 4x longer lookback)")
    log("=" * 60)

    sharpes_4h_mb = {}
    for year in YEARS:
        start, end = YEAR_RANGES[year]
        log(f"  4h_match_bars / {year}")
        rets = run_blend(SIGNALS_4H_MATCH_BARS, SYMBOLS, start, end, "4h")
        s = compute_sharpe(rets, 2190)
        sharpes_4h_mb[year] = round(s, 4)
        log(f"    Sharpe={s:.4f}")

    log("  4h_match_bars / 2026 OOS")
    oos_rets_4h_mb = run_blend(SIGNALS_4H_MATCH_BARS, SYMBOLS, OOS_RANGE[0], OOS_RANGE[1], "4h")
    oos_4h_mb = round(compute_sharpe(oos_rets_4h_mb, 2190), 4)
    log(f"    OOS Sharpe={oos_4h_mb:.4f}")

    vals = list(sharpes_4h_mb.values())
    report["experiments"]["4h_match_bars"] = {
        "yearly": sharpes_4h_mb,
        "avg": round(sum(vals) / len(vals), 4),
        "min": round(min(vals), 4),
        "oos_2026": oos_4h_mb,
        "bar_interval": "4h",
        "note": "same bar counts as 1h → 4x longer lookback in wall clock",
    }

    # ════════════════════════════════════
    # SECTION D: Long-bias overlay
    # ════════════════════════════════════
    log("=" * 60)
    log("SECTION D: Long-bias overlay (P91b + beta × market)")
    log("=" * 60)

    BETAS = [0.05, 0.10, 0.15, 0.20]
    for beta in BETAS:
        label = f"longbias_{int(beta*100)}pct"
        log(f"\n  {label}")
        sharpes_lb = {}

        for year in YEARS:
            start, end = YEAR_RANGES[year]
            log(f"    {year}")
            # Use cached 1h baseline returns
            p91b_rets = baseline_rets_by_year[year]
            mkt_rets = get_market_returns(SYMBOLS, start, end, "1h")
            n = min(len(p91b_rets), len(mkt_rets))
            if n < 50:
                sharpes_lb[year] = 0.0
                continue
            combined = [p91b_rets[i] + beta * mkt_rets[i] for i in range(n)]
            s = compute_sharpe(combined, 8760)
            sharpes_lb[year] = round(s, 4)
            log(f"      Sharpe={s:.4f}")

        # OOS
        log(f"    2026 OOS")
        mkt_rets_oos = get_market_returns(SYMBOLS, OOS_RANGE[0], OOS_RANGE[1], "1h")
        n_oos = min(len(oos_rets_1h), len(mkt_rets_oos))
        if n_oos >= 50:
            combined_oos = [oos_rets_1h[i] + beta * mkt_rets_oos[i] for i in range(n_oos)]
            oos_lb = round(compute_sharpe(combined_oos, 8760), 4)
        else:
            oos_lb = 0.0
        log(f"      OOS Sharpe={oos_lb:.4f}")

        vals = list(sharpes_lb.values())
        report["experiments"][label] = {
            "yearly": sharpes_lb,
            "avg": round(sum(vals) / len(vals), 4),
            "min": round(min(vals), 4),
            "oos_2026": oos_lb,
            "beta": beta,
            "note": f"P91b + {beta:.0%} market exposure",
        }

    # ════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════
    elapsed = round(time.time() - t0, 1)
    report["elapsed_seconds"] = elapsed

    out_path = os.path.join(OUT_DIR, "phase98_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    log("\n" + "=" * 60)
    log(f"Phase 98 COMPLETE in {elapsed}s → {out_path}")
    log("=" * 60)

    log(f"\n{'Experiment':>20} | {'AVG':>8} | {'MIN':>8} | {'OOS':>8}")
    log(f"{'-'*20}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for name, data in report["experiments"].items():
        log(f"{name:>20} | {data['avg']:>8.4f} | {data['min']:>8.4f} | {data['oos_2026']:>8.4f}")

    # Identify best
    best_name = max(report["experiments"],
                    key=lambda k: (report["experiments"][k]["avg"] + report["experiments"][k]["min"]) / 2)
    best = report["experiments"][best_name]
    log(f"\n  Best (balanced): {best_name} — AVG={best['avg']:.4f}, MIN={best['min']:.4f}")
    baseline = report["experiments"]["1h_baseline"]
    log(f"  vs 1h baseline:  ΔAVG={best['avg'] - baseline['avg']:+.4f}, ΔMIN={best['min'] - baseline['min']:+.4f}")
