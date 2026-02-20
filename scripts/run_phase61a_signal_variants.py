#!/usr/bin/env python3
"""Phase 61A: EWMA Sharpe + Sortino Ratio variants — 5-year OOS test."""
import json
import math
import statistics
import subprocess
import sys
import copy
from pathlib import Path

OUT_DIR = "artifacts/phase61a"
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

# Sharpe champion for comparison
SR_CHAMPION_PARAMS = {
    "k_per_side": 2, "lookback_bars": 336, "vol_lookback_bars": 168,
    "target_gross_leverage": 0.35, "rebalance_interval_bars": 48,
}


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
        if v > peak: peak = v
        dd = 1 - v / peak if peak > 0 else 0
        if dd > max_dd: max_dd = dd
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
    path = f"/tmp/phase61a_{run_name}_{year}.json"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def sweep_variant(name: str, strategy_name: str, params: dict) -> dict:
    year_results = {}
    for year in YEARS:
        run_name = f"{name}_{year}"
        config_path = make_config(run_name, year, strategy_name, params)
        try:
            result = subprocess.run(
                [sys.executable, "-m", "nexus_quant", "run", "--config", config_path, "--out", OUT_DIR],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0:
                year_results[year] = {"error": result.stderr[-100:]}
                print(f"    {year}: ERROR", flush=True)
                continue
        except subprocess.TimeoutExpired:
            year_results[year] = {"error": "timeout"}
            continue

        runs_dir = Path(OUT_DIR) / "runs"
        if not runs_dir.exists():
            year_results[year] = {"error": "no runs dir"}
            continue
        matching = sorted(
            [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith(run_name)],
            key=lambda d: d.stat().st_mtime,
        )
        if matching:
            rp = matching[-1] / "result.json"
            if rp.exists():
                m = compute_metrics(str(rp))
                year_results[year] = m
                print(f"    {year}: Sharpe={m.get('sharpe', '?')}", flush=True)
                continue
        year_results[year] = {"error": "no result"}
        print(f"    {year}: no result", flush=True)

    sharpes = [y.get("sharpe", 0) for y in year_results.values() if isinstance(y.get("sharpe"), (int, float))]
    avg = round(sum(sharpes) / len(sharpes), 3) if sharpes else 0
    mn = round(min(sharpes), 3) if sharpes else 0
    year_results["_avg_sharpe"] = avg
    year_results["_min_sharpe"] = mn
    return year_results


def main():
    print("=" * 80)
    print("  PHASE 61A: EWMA SHARPE + SORTINO RATIO VARIANTS")
    print("=" * 80, flush=True)

    all_results = {}

    # ====================================================================
    # BASELINE: Sharpe Ratio 336h (Phase 60 champion)
    # ====================================================================
    print(f"\n{'='*70}")
    print("  BASELINE: Sharpe Ratio 336h (Phase 60 champion)")
    print(f"{'='*70}", flush=True)
    r = sweep_variant("sr_336h_base", "sharpe_ratio_alpha", SR_CHAMPION_PARAMS)
    all_results["sr_336h_base"] = r
    print(f"  -> AVG={r['_avg_sharpe']}, MIN={r['_min_sharpe']}")

    # ====================================================================
    # 1. EWMA SHARPE — lambda sweep
    # ====================================================================
    print(f"\n{'='*70}")
    print("  EWMA SHARPE ALPHA — LAMBDA SWEEP")
    print(f"{'='*70}", flush=True)

    ewma_variants = [
        # (name, lookback, lambda, lev)
        ("ewma_l99_336",   336, 0.99, 0.35),  # very slow decay (nearly equal-weight)
        ("ewma_l98_336",   336, 0.98, 0.35),  # slow decay (half-life ~34h)
        ("ewma_l97_336",   336, 0.97, 0.35),  # medium decay (half-life ~23h)
        ("ewma_l95_336",   336, 0.95, 0.35),  # fast decay (half-life ~13h)
        ("ewma_l98_240",   240, 0.98, 0.35),  # shorter lookback
        ("ewma_l98_504",   504, 0.98, 0.35),  # longer lookback
        ("ewma_l97_240",   240, 0.97, 0.35),  # shorter + faster
        ("ewma_l98_r24",   336, 0.98, 0.35),  # different rebal (set below)
        ("ewma_l98_lev30", 336, 0.98, 0.30),  # lower leverage
        ("ewma_l98_lev40", 336, 0.98, 0.40),  # higher leverage
    ]

    for name, lb, lam, lev in ewma_variants:
        rebal = 24 if name == "ewma_l98_r24" else 48
        params = {
            "k_per_side": 2,
            "lookback_bars": lb,
            "ewma_lambda": lam,
            "vol_lookback_bars": 168,
            "target_gross_leverage": lev,
            "rebalance_interval_bars": rebal,
        }
        print(f"\n  >> {name} (lb={lb} lambda={lam} lev={lev} rebal={rebal})", flush=True)
        r = sweep_variant(name, "ewma_sharpe_alpha", params)
        all_results[name] = r
        print(f"     AVG={r['_avg_sharpe']}, MIN={r['_min_sharpe']}", flush=True)

    # ====================================================================
    # 2. SORTINO RATIO — lookback sweep
    # ====================================================================
    print(f"\n{'='*70}")
    print("  SORTINO RATIO ALPHA — PARAMETER SWEEP")
    print(f"{'='*70}", flush=True)

    sortino_variants = [
        # (name, lookback, lev, rebal)
        ("sort_168",   168, 0.35, 48),
        ("sort_240",   240, 0.35, 48),
        ("sort_336",   336, 0.35, 48),   # same as sr_336h but Sortino
        ("sort_504",   504, 0.35, 48),
        ("sort_720",   720, 0.35, 48),
        ("sort_336_r24", 336, 0.35, 24),
        ("sort_336_lev30", 336, 0.30, 48),
        ("sort_336_k3",    336, 0.35, 48),   # k=3 (set below)
    ]

    for name, lb, lev, rebal in sortino_variants:
        k = 3 if name == "sort_336_k3" else 2
        params = {
            "k_per_side": k,
            "lookback_bars": lb,
            "target_return": 0.0,
            "vol_lookback_bars": 168,
            "target_gross_leverage": lev,
            "rebalance_interval_bars": rebal,
        }
        print(f"\n  >> {name} (lb={lb} lev={lev} rebal={rebal} k={k})", flush=True)
        r = sweep_variant(name, "sortino_alpha", params)
        all_results[name] = r
        print(f"     AVG={r['_avg_sharpe']}, MIN={r['_min_sharpe']}", flush=True)

    # ====================================================================
    # SUMMARY
    # ====================================================================
    print(f"\n{'='*80}")
    print(f"  TOP 15 VARIANTS BY AVG SHARPE")
    print(f"{'='*80}")
    ranked = sorted(all_results.items(), key=lambda x: x[1].get("_avg_sharpe", -99), reverse=True)
    for i, (name, r) in enumerate(ranked[:15]):
        sharpes_str = " | ".join(f"{y}:{r.get(y, {}).get('sharpe', '?')}" for y in YEARS)
        pos = sum(1 for y in YEARS if isinstance(r.get(y, {}).get("sharpe"), (int, float)) and r[y]["sharpe"] > 0)
        print(f"  {i+1:>2}. {name:<24} AVG={r['_avg_sharpe']:>7.3f} MIN={r['_min_sharpe']:>7.3f} pos={pos}  [{sharpes_str}]")

    print(f"\n{'='*80}")
    print(f"  VARIANTS WITH 5 POSITIVE YEARS")
    print(f"{'='*80}")
    for name, r in ranked:
        pos = sum(1 for y in YEARS if isinstance(r.get(y, {}).get("sharpe"), (int, float)) and r[y]["sharpe"] > 0)
        if pos == 5:
            sharpes_str = " | ".join(f"{y}:{r.get(y, {}).get('sharpe', '?')}" for y in YEARS)
            print(f"  {name:<24} AVG={r['_avg_sharpe']:>7.3f} MIN={r['_min_sharpe']:>7.3f}  [{sharpes_str}]")

    # Save
    out_path = Path(OUT_DIR) / "phase61a_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
