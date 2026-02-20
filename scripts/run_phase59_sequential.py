#!/usr/bin/env python3
"""Phase 59: Run all backtests sequentially by year (to cache data once per year)."""
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

STRATEGIES = [
    "lead_lag_alpha",
    "volume_reversal_alpha",
    "basis_momentum_alpha",
    "vol_breakout_alpha",
    "rs_acceleration_alpha",
]

YEARS = ["2021", "2022", "2023", "2024", "2025"]
OUT_DIR = "artifacts/phase59"
CONFIGS_DIR = "configs"


def compute_metrics(result_path: str) -> dict:
    d = json.load(open(result_path))
    eq = d.get("equity_curve", [])
    rets = d.get("returns", [])
    if not eq or not rets or len(rets) < 100:
        return {"sharpe": 0, "cagr": 0, "max_dd": 0, "total_return": 0, "error": "insufficient data"}
    total_return = eq[-1] / eq[0] - 1.0 if eq[0] > 0 else 0
    n_hours = len(rets)
    n_years = n_hours / 8760.0
    cagr = (eq[-1] ** (1.0 / n_years) - 1.0) if n_years > 0 and eq[-1] > 0 else 0
    peak = eq[0]
    max_dd = 0
    for v in eq:
        if v > peak: peak = v
        dd = 1 - v / peak if peak > 0 else 0
        if dd > max_dd: max_dd = dd
    if len(rets) > 1:
        mu = statistics.mean(rets)
        sd = statistics.pstdev(rets)
        sharpe = (mu / sd) * math.sqrt(8760) if sd > 0 else 0
    else:
        sharpe = 0
    return {"sharpe": round(sharpe, 3), "cagr": round(cagr * 100, 2), "max_dd": round(max_dd * 100, 2), "total_return": round(total_return * 100, 2)}


def find_latest_result(strategy: str, year: str) -> str | None:
    runs_dir = Path(OUT_DIR) / "runs"
    if not runs_dir.exists():
        return None
    prefix = f"{strategy}_{year}"
    matching = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)],
        key=lambda d: d.stat().st_mtime,
    )
    if matching:
        result = matching[-1] / "result.json"
        if result.exists():
            return str(result)
    return None


def run_backtest(strategy: str, year: str) -> dict:
    config = f"{CONFIGS_DIR}/run_{strategy}_{year}.json"
    if not Path(config).exists():
        return {"error": f"config not found: {config}"}

    print(f"\n  >> {strategy} / {year} ...", flush=True)

    try:
        result = subprocess.run(
            [sys.executable, "-m", "nexus_quant", "run", "--config", config, "--out", OUT_DIR],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            err_msg = result.stderr[-300:] if result.stderr else "unknown error"
            print(f"     FAIL: {err_msg[:100]}")
            return {"error": err_msg[:200]}
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}

    result_path = find_latest_result(strategy, year)
    if result_path:
        metrics = compute_metrics(result_path)
        print(f"     Sharpe={metrics.get('sharpe')}, CAGR={metrics.get('cagr')}%, MDD={metrics.get('max_dd')}%", flush=True)
        return metrics
    return {"error": "no result file"}


def main():
    all_results: dict = {}

    # Run by YEAR first (so data gets cached once, shared by all strategies)
    for year in YEARS:
        print(f"\n{'='*60}")
        print(f"  YEAR {year}")
        print(f"{'='*60}", flush=True)

        for strategy in STRATEGIES:
            if strategy not in all_results:
                all_results[strategy] = {}
            metrics = run_backtest(strategy, year)
            all_results[strategy][year] = metrics

        # Small delay between years to avoid rate limit
        if year != YEARS[-1]:
            print(f"\n  [sleeping 5s between years...]", flush=True)
            time.sleep(5)

    # Save
    out_path = Path(OUT_DIR) / "phase59_all_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'='*80}")
    print(f"  PHASE 59A — SHARPE RATIO SUMMARY")
    print(f"{'='*80}")
    print(f"{'Strategy':<28} {'2021':>8} {'2022':>8} {'2023':>8} {'2024':>8} {'2025':>8} {'AVG':>8} {'MIN':>8}")
    print("-" * 80)

    for strategy in STRATEGIES:
        sharpes = []
        row = f"{strategy:<28}"
        for year in YEARS:
            m = all_results.get(strategy, {}).get(year, {})
            s = m.get("sharpe", "ERR")
            row += f" {s:>8}"
            if isinstance(s, (int, float)):
                sharpes.append(s)
        if sharpes:
            avg = round(sum(sharpes) / len(sharpes), 3)
            mn = round(min(sharpes), 3)
            row += f" {avg:>8} {mn:>8}"
        print(row)

    print(f"\n{'='*80}")
    print(f"  CAGR SUMMARY")
    print(f"{'='*80}")
    print(f"{'Strategy':<28} {'2021':>8} {'2022':>8} {'2023':>8} {'2024':>8} {'2025':>8}")
    print("-" * 80)
    for strategy in STRATEGIES:
        row = f"{strategy:<28}"
        for year in YEARS:
            m = all_results.get(strategy, {}).get(year, {})
            c = m.get("cagr", "ERR")
            row += f" {c:>7}%"
        print(row)

    print(f"\n{'='*80}")
    print(f"  MAX DRAWDOWN SUMMARY")
    print(f"{'='*80}")
    print(f"{'Strategy':<28} {'2021':>8} {'2022':>8} {'2023':>8} {'2024':>8} {'2025':>8}")
    print("-" * 80)
    for strategy in STRATEGIES:
        row = f"{strategy:<28}"
        for year in YEARS:
            m = all_results.get(strategy, {}).get(year, {})
            d = m.get("max_dd", "ERR")
            row += f" {d:>7}%"
        print(row)

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
