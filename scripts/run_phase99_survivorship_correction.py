#!/usr/bin/env python3
"""
Phase 99: Survivorship-Bias Correction
========================================
Our 2021 results use CURRENT 10 coins, but some (SOL, AVAX, DOT) had
limited futures in early 2021. This inflates 2021 Sharpe.

Tests:
  A) Dynamic universe per year based on actual perp listing dates
  B) Conservative universe (only coins with full year data)
  C) Ultra-conservative (only BTC/ETH/BNB/XRP + 2 mid-cap)
  D) Compare corrected Sharpe vs current (especially 2021)

If 2021 Sharpe drops significantly → our MIN may shift from 2022 to 2021,
which changes our understanding of the champion's true robustness.
"""

import copy, json, os, sys, time
from pathlib import Path

PROJ = "/Users/qtmobile/Desktop/Nexus - Quant Trading "
sys.path.insert(0, PROJ)

import numpy as np

from nexus_quant.data.providers.registry import make_provider
from nexus_quant.strategies.registry import make_strategy
from nexus_quant.backtest.engine import BacktestConfig, BacktestEngine
from nexus_quant.backtest.costs import cost_model_from_config

OUT_DIR = os.path.join(PROJ, "artifacts", "phase99")
os.makedirs(OUT_DIR, exist_ok=True)

# Approximate Binance PERP listing dates (conservative estimates)
# If a coin's perp was listed mid-year, exclude from that year
PERP_AVAILABLE = {
    "BTCUSDT":  "2019-09-01",  # always available
    "ETHUSDT":  "2019-09-01",
    "BNBUSDT":  "2020-02-01",
    "XRPUSDT":  "2020-01-01",
    "ADAUSDT":  "2020-07-01",
    "LINKUSDT": "2020-07-01",
    "DOGEUSDT": "2021-05-01",  # mid-2021 — exclude from 2021 conservative
    "SOLUSDT":  "2021-04-01",  # mid-2021 — exclude from 2021 conservative
    "AVAXUSDT": "2021-09-01",  # late 2021 — exclude from 2021
    "DOTUSDT":  "2020-12-01",  # available but lower liquidity early 2021
    "LTCUSDT":  "2020-01-01",
    "UNIUSDT":  "2021-01-01",
}

FULL_10 = [
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

P91B_WEIGHTS = {"v1": 0.2747, "i460bw168": 0.1967, "i415bw216": 0.3247, "f144": 0.2039}

SIGNALS = {
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


def log(msg):
    print(f"[P99] {msg}", flush=True)


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


def run_signal(sig_cfg, symbols, start, end):
    data_cfg = {
        "provider": "binance_rest_v1",
        "symbols": symbols,
        "start": start, "end": end,
        "bar_interval": "1h",
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


def run_blend(symbols, start, end, signal_cfgs=None, k_adj=None):
    """Run all 4 signals, optionally adjust k_per_side, blend with P91b weights."""
    cfgs = copy.deepcopy(signal_cfgs or SIGNALS)
    if k_adj is not None:
        for sig_key in cfgs:
            cfgs[sig_key]["params"]["k_per_side"] = min(
                k_adj, len(symbols) // 2
            )
    sig_rets = {}
    for sig_key in sorted(P91B_WEIGHTS.keys()):
        try:
            rets = run_signal(cfgs[sig_key], symbols, start, end)
        except Exception as exc:
            log(f"    {sig_key} ERROR: {exc}")
            rets = []
        sig_rets[sig_key] = rets
    return blend_returns(sig_rets, P91B_WEIGHTS)


def get_available_symbols(year_start_str):
    """Return symbols that had active perps BEFORE the year start."""
    from datetime import datetime
    year_start = datetime.strptime(year_start_str, "%Y-%m-%d")
    available = []
    for sym, listing_date_str in PERP_AVAILABLE.items():
        if sym not in FULL_10:
            continue
        listing_date = datetime.strptime(listing_date_str, "%Y-%m-%d")
        if listing_date < year_start:
            available.append(sym)
    return sorted(available)


# ════════════════════════════════════════
# Universes to test
# ════════════════════════════════════════
def get_universes():
    return {
        # A) Full 10 (current, may have survivorship bias in 2021)
        "full_10": {year: FULL_10 for year in YEARS},
        # B) Dynamic: only coins available before year start
        "dynamic": {year: get_available_symbols(YEAR_RANGES[year][0]) for year in YEARS},
        # C) Conservative: only coins with confirmed full-year perp data
        "conservative": {
            "2021": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "LINKUSDT", "DOTUSDT"],
            "2022": FULL_10,  # all 10 had perps by 2022
            "2023": FULL_10,
            "2024": FULL_10,
            "2025": FULL_10,
        },
        # D) Ultra-conservative: only mega-cap
        "ultra_conservative": {
            "2021": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"],
            "2022": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "LINKUSDT", "DOGEUSDT", "SOLUSDT"],
            "2023": FULL_10,
            "2024": FULL_10,
            "2025": FULL_10,
        },
    }


if __name__ == "__main__":
    t0 = time.time()
    report = {"phase": 99, "universes": {}, "comparison": {}}

    universes = get_universes()

    for uni_name, year_syms in universes.items():
        log("=" * 60)
        log(f"Testing universe: {uni_name}")
        log("=" * 60)

        yearly_sharpes = {}
        for year in YEARS:
            syms = year_syms[year]
            start, end = YEAR_RANGES[year]
            log(f"  {uni_name} / {year} — {len(syms)} symbols: {', '.join(syms)}")

            rets = run_blend(syms, start, end)
            s = round(compute_sharpe(rets, 8760), 4)
            yearly_sharpes[year] = s
            log(f"    Sharpe={s:.4f}")

        # OOS 2026 (always use full 10 for OOS)
        log(f"  {uni_name} / 2026 OOS")
        oos_rets = run_blend(FULL_10, OOS_RANGE[0], OOS_RANGE[1])
        oos_s = round(compute_sharpe(oos_rets, 8760), 4)
        log(f"    OOS Sharpe={oos_s:.4f}")

        vals = list(yearly_sharpes.values())
        report["universes"][uni_name] = {
            "yearly": yearly_sharpes,
            "avg": round(sum(vals) / len(vals), 4),
            "min": round(min(vals), 4),
            "oos_2026": oos_s,
            "symbols_per_year": {y: year_syms[y] for y in YEARS},
            "n_symbols_per_year": {y: len(year_syms[y]) for y in YEARS},
        }

    # Comparison: delta from full_10 baseline
    baseline = report["universes"]["full_10"]
    for uni_name, data in report["universes"].items():
        if uni_name == "full_10":
            continue
        report["comparison"][f"{uni_name}_vs_full_10"] = {
            "delta_avg": round(data["avg"] - baseline["avg"], 4),
            "delta_min": round(data["min"] - baseline["min"], 4),
            "yearly_delta": {
                y: round(data["yearly"][y] - baseline["yearly"][y], 4)
                for y in YEARS
            },
        }

    elapsed = round(time.time() - t0, 1)
    report["elapsed_seconds"] = elapsed

    out_path = os.path.join(OUT_DIR, "phase99_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    log("\n" + "=" * 60)
    log(f"Phase 99 COMPLETE in {elapsed}s → {out_path}")
    log("=" * 60)

    log(f"\n{'Universe':>20} | {'2021':>7} | {'2022':>7} | {'2023':>7} | {'2024':>7} | {'2025':>7} | {'AVG':>7} | {'MIN':>7}")
    log(f"{'-'*20}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
    for uni_name, data in report["universes"].items():
        y = data["yearly"]
        log(f"{uni_name:>20} | {y['2021']:>7.3f} | {y['2022']:>7.3f} | {y['2023']:>7.3f} | {y['2024']:>7.3f} | {y['2025']:>7.3f} | {data['avg']:>7.3f} | {data['min']:>7.3f}")

    log(f"\nKey question: Does 2021 Sharpe drop significantly with corrected universe?")
    full_2021 = baseline["yearly"]["2021"]
    for uni_name in ["dynamic", "conservative", "ultra_conservative"]:
        corr_2021 = report["universes"][uni_name]["yearly"]["2021"]
        delta = corr_2021 - full_2021
        log(f"  {uni_name}: 2021 Sharpe={corr_2021:.3f} (Δ={delta:+.3f})")
