#!/usr/bin/env python3
"""Phase 60A: 4 new alpha signal categories — initial 5-year OOS test."""
import json
import math
import statistics
import subprocess
import sys
import copy
from pathlib import Path

OUT_DIR = "artifacts/phase60a"
YEARS = ["2021", "2022", "2023", "2024", "2025"]
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT"]

YEAR_RANGES = {
    "2021": ("2021-01-01", "2022-01-01"),
    "2022": ("2022-01-01", "2023-01-01"),
    "2023": ("2023-01-01", "2024-01-01"),
    "2024": ("2024-01-01", "2025-01-01"),
    "2025": ("2025-01-01", "2026-01-01"),
}

BASE_CONFIG = {
    "seed": 42,
    "venue": {"name": "binance_usdm", "kind": "crypto_perp", "vip_tier": 0},
    "data": {"provider": "binance_rest_v1", "symbols": SYMBOLS, "bar_interval": "1h", "cache_dir": ".cache/binance_rest"},
    "execution": {"style": "taker", "slippage_bps": 3.0},
    "costs": {"fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0},
    "benchmark": {"version": "v1", "walk_forward": {"enabled": True}},
    "risk": {"max_drawdown": 0.30, "max_turnover_per_rebalance": 0.8, "max_gross_leverage": 0.7, "max_position_per_symbol": 0.3},
    "self_learn": {"enabled": False}
}

# 4 new signals with default params
SIGNALS = [
    (
        "multitf_mom",
        "multitf_momentum_alpha",
        {
            "k_per_side": 2,
            "w_24h": 0.10,
            "w_72h": 0.20,
            "w_168h": 0.35,
            "w_336h": 0.35,
            "vol_lookback_bars": 168,
            "target_gross_leverage": 0.35,
            "rebalance_interval_bars": 48,
        },
    ),
    (
        "sharpe_ratio",
        "sharpe_ratio_alpha",
        {
            "k_per_side": 2,
            "lookback_bars": 168,
            "vol_lookback_bars": 168,
            "target_gross_leverage": 0.35,
            "rebalance_interval_bars": 48,
        },
    ),
    (
        "price_level",
        "price_level_alpha",
        {
            "k_per_side": 2,
            "level_lookback_bars": 504,
            "vol_lookback_bars": 168,
            "target_gross_leverage": 0.30,
            "rebalance_interval_bars": 48,
        },
    ),
    (
        "amihud_illiq",
        "amihud_illiquidity_alpha",
        {
            "k_per_side": 2,
            "lookback_bars": 168,
            "mom_lookback_bars": 168,
            "use_interaction": True,
            "vol_lookback_bars": 168,
            "target_gross_leverage": 0.30,
            "rebalance_interval_bars": 48,
        },
    ),
]


def compute_metrics(result_path: str) -> dict:
    d = json.load(open(result_path))
    eq = d.get("equity_curve", [])
    rets = d.get("returns", [])
    if not eq or not rets or len(rets) < 100:
        return {"sharpe": 0, "cagr": 0, "max_dd": 0, "error": "insufficient data"}
    n_years = len(rets) / 8760.0
    cagr = (eq[-1] ** (1.0 / n_years) - 1.0) if n_years > 0 and eq[-1] > 0 else 0
    peak = eq[0]
    max_dd = 0
    for v in eq:
        if v > peak:
            peak = v
        dd = 1 - v / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    mu = statistics.mean(rets)
    sd = statistics.pstdev(rets)
    sharpe = (mu / sd) * math.sqrt(8760) if sd > 0 else 0
    return {"sharpe": round(sharpe, 3), "cagr": round(cagr * 100, 2), "max_dd": round(max_dd * 100, 2)}


def make_config(run_name: str, year: str, strategy_name: str, params: dict) -> str:
    cfg = copy.deepcopy(BASE_CONFIG)
    start, end = YEAR_RANGES[year]
    cfg["run_name"] = run_name
    cfg["data"]["start"] = start
    cfg["data"]["end"] = end
    cfg["strategy"] = {"name": strategy_name, "params": params}
    path = f"/tmp/phase60a_{run_name}_{year}.json"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def run_backtest(config_path: str, run_name: str) -> dict:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "nexus_quant", "run", "--config", config_path, "--out", OUT_DIR],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            return {"error": result.stderr[-200:]}
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}

    runs_dir = Path(OUT_DIR) / "runs"
    if not runs_dir.exists():
        return {"error": "no runs dir"}
    matching = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith(run_name)],
        key=lambda d: d.stat().st_mtime,
    )
    if matching:
        rp = matching[-1] / "result.json"
        if rp.exists():
            return compute_metrics(str(rp))
    return {"error": "no result"}


def main():
    print("=" * 80)
    print("  PHASE 60A: 4 NEW ALPHA SIGNALS — 5-YEAR OOS")
    print("=" * 80, flush=True)

    all_results = {}

    for signal_name, strategy_name, params in SIGNALS:
        print(f"\n{'='*70}")
        print(f"  >> {signal_name} ({strategy_name})")
        print(f"{'='*70}", flush=True)

        year_results = {}
        for year in YEARS:
            run_name = f"{signal_name}_{year}"
            config_path = make_config(run_name, year, strategy_name, params)
            m = run_backtest(config_path, run_name)
            year_results[year] = m
            s = m.get("sharpe", "ERR")
            c = m.get("cagr", "?")
            print(f"    {year}: Sharpe={s}, CAGR={c}%", flush=True)

        sharpes = [y.get("sharpe", 0) for y in year_results.values() if isinstance(y.get("sharpe"), (int, float))]
        avg = round(sum(sharpes) / len(sharpes), 3) if sharpes else 0
        mn = round(min(sharpes), 3) if sharpes else 0
        pos_years = sum(1 for s in sharpes if s > 0)
        print(f"  -> AVG Sharpe={avg}, MIN={mn}, {pos_years}/5 positive", flush=True)
        year_results["_avg_sharpe"] = avg
        year_results["_min_sharpe"] = mn
        year_results["_pos_years"] = pos_years
        all_results[signal_name] = year_results

    # Summary
    print(f"\n{'='*80}")
    print(f"  FINAL SUMMARY — PHASE 60A")
    print(f"{'='*80}")
    print(f"  {'Signal':<20} {'AVG':>7} {'MIN':>7} {'Pos':>5}  [year-by-year]")
    print(f"  {'-'*70}")
    for name, r in sorted(all_results.items(), key=lambda x: x[1].get("_avg_sharpe", -99), reverse=True):
        sharpes_str = " ".join(f"{y}:{r.get(y, {}).get('sharpe', '?')}" for y in YEARS)
        print(f"  {name:<20} {r['_avg_sharpe']:>7.3f} {r['_min_sharpe']:>7.3f} {r['_pos_years']:>5}  [{sharpes_str}]")

    # Save
    out_path = Path(OUT_DIR) / "phase60a_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")
    print(f"\nNext: run Phase 60B if any signal has AVG > -0.3 or 3+ positive years")


if __name__ == "__main__":
    main()
