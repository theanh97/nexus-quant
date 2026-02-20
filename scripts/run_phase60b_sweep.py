#!/usr/bin/env python3
"""Phase 60B: Parameter sweep on best Phase 60A performers."""
import json
import math
import statistics
import subprocess
import sys
import copy
from pathlib import Path

OUT_DIR = "artifacts/phase60b"
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
    path = f"/tmp/phase60b_{run_name}_{year}.json"
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
            print(f"    {year}: TIMEOUT", flush=True)
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
    print("  PHASE 60B: PARAMETER SWEEP")
    print("=" * 80, flush=True)

    all_results = {}

    # ====================================================================
    # 1. MULTI-TF MOMENTUM — weight variations
    # ====================================================================
    print("\n" + "="*70)
    print("  MULTI-TIMEFRAME MOMENTUM — WEIGHT SWEEP")
    print("="*70, flush=True)

    multitf_variants = [
        # (name, w24, w72, w168, w336, rebal, lev)
        ("mtf_eq",         0.25, 0.25, 0.25, 0.25, 48, 0.35),  # equal weights
        ("mtf_slow",       0.05, 0.10, 0.40, 0.45, 48, 0.35),  # slow-heavy
        ("mtf_fast",       0.30, 0.35, 0.20, 0.15, 48, 0.35),  # fast-heavy
        ("mtf_mid",        0.10, 0.30, 0.40, 0.20, 48, 0.35),  # mid-weighted
        ("mtf_336only",    0.00, 0.00, 0.00, 1.00, 48, 0.35),  # only 336h
        ("mtf_168only",    0.00, 0.00, 1.00, 0.00, 48, 0.35),  # only 168h (like V1)
        ("mtf_24_168",     0.20, 0.00, 0.40, 0.40, 48, 0.35),  # skip 72h
        ("mtf_slow_r24",   0.05, 0.10, 0.40, 0.45, 24, 0.35),  # slower rebal
        ("mtf_slow_r72",   0.05, 0.10, 0.40, 0.45, 72, 0.35),  # faster rebal
        ("mtf_slow_k3",    0.05, 0.10, 0.40, 0.45, 48, 0.35),  # k=3 (set below)
        ("mtf_slow_lev25", 0.05, 0.10, 0.40, 0.45, 48, 0.25),  # lower lev
        ("mtf_slow_lev45", 0.05, 0.10, 0.40, 0.45, 48, 0.45),  # higher lev
    ]

    for name, w24, w72, w168, w336, rebal, lev in multitf_variants:
        k = 3 if name == "mtf_slow_k3" else 2
        params = {
            "k_per_side": k,
            "w_24h": w24,
            "w_72h": w72,
            "w_168h": w168,
            "w_336h": w336,
            "vol_lookback_bars": 168,
            "target_gross_leverage": lev,
            "rebalance_interval_bars": rebal,
        }
        print(f"\n  >> {name} (w=[{w24},{w72},{w168},{w336}] rebal={rebal} lev={lev} k={k})", flush=True)
        r = sweep_variant(name, "multitf_momentum_alpha", params)
        all_results[name] = r
        print(f"     AVG={r['_avg_sharpe']}, MIN={r['_min_sharpe']}", flush=True)

    # ====================================================================
    # 2. SHARPE RATIO ALPHA — lookback variations
    # ====================================================================
    print("\n" + "="*70)
    print("  SHARPE RATIO ALPHA — LOOKBACK SWEEP")
    print("="*70, flush=True)

    sharpe_variants = [
        # (name, lookback, rebal, lev)
        ("sr_72h",   72,  48, 0.35),
        ("sr_168h",  168, 48, 0.35),  # default
        ("sr_336h",  336, 48, 0.35),
        ("sr_504h",  504, 48, 0.35),
        ("sr_720h",  720, 48, 0.35),
        ("sr_168_r24", 168, 24, 0.35),
        ("sr_168_r72", 168, 72, 0.35),
        ("sr_336_r48", 336, 48, 0.35),
        ("sr_168_lev25", 168, 48, 0.25),
        ("sr_168_k3",    168, 48, 0.35),  # k=3
    ]

    for name, lb, rebal, lev in sharpe_variants:
        k = 3 if name == "sr_168_k3" else 2
        params = {
            "k_per_side": k,
            "lookback_bars": lb,
            "vol_lookback_bars": 168,
            "target_gross_leverage": lev,
            "rebalance_interval_bars": rebal,
        }
        print(f"\n  >> {name} (lb={lb} rebal={rebal} lev={lev} k={k})", flush=True)
        r = sweep_variant(name, "sharpe_ratio_alpha", params)
        all_results[name] = r
        print(f"     AVG={r['_avg_sharpe']}, MIN={r['_min_sharpe']}", flush=True)

    # ====================================================================
    # 3. PRICE LEVEL ALPHA — lookback variations
    # ====================================================================
    print("\n" + "="*70)
    print("  PRICE LEVEL ALPHA — LOOKBACK SWEEP")
    print("="*70, flush=True)

    pl_variants = [
        # (name, level_lb, rebal, lev)
        ("pl_168h",  168,  48, 0.30),   # 1 week
        ("pl_336h",  336,  48, 0.30),   # 2 weeks
        ("pl_504h",  504,  48, 0.30),   # 3 weeks (default)
        ("pl_720h",  720,  48, 0.30),   # 1 month
        ("pl_1440h", 1440, 48, 0.30),   # 2 months
        ("pl_504_r24",  504, 24, 0.30),
        ("pl_504_r72",  504, 72, 0.30),
        ("pl_504_lev35", 504, 48, 0.35),
        ("pl_720_k3",    720, 48, 0.30),  # k=3
    ]

    for name, level_lb, rebal, lev in pl_variants:
        k = 3 if name == "pl_720_k3" else 2
        params = {
            "k_per_side": k,
            "level_lookback_bars": level_lb,
            "vol_lookback_bars": 168,
            "target_gross_leverage": lev,
            "rebalance_interval_bars": rebal,
        }
        print(f"\n  >> {name} (level_lb={level_lb} rebal={rebal} lev={lev} k={k})", flush=True)
        r = sweep_variant(name, "price_level_alpha", params)
        all_results[name] = r
        print(f"     AVG={r['_avg_sharpe']}, MIN={r['_min_sharpe']}", flush=True)

    # ====================================================================
    # 4. AMIHUD ILLIQUIDITY — parameter sweep
    # ====================================================================
    print("\n" + "="*70)
    print("  AMIHUD ILLIQUIDITY ALPHA — PARAMETER SWEEP")
    print("="*70, flush=True)

    amihud_variants = [
        # (name, lb, mom_lb, interaction, rebal, lev)
        ("am_base",     168, 168, True,  48, 0.30),  # default
        ("am_no_int",   168, 168, False, 48, 0.30),  # no interaction (pure illiquidity)
        ("am_lb72",     72,  168, True,  48, 0.30),
        ("am_lb336",    336, 168, True,  48, 0.30),
        ("am_mom72",    168, 72,  True,  48, 0.30),
        ("am_mom336",   168, 336, True,  48, 0.30),
        ("am_r24",      168, 168, True,  24, 0.30),
        ("am_r72",      168, 168, True,  72, 0.30),
    ]

    for name, lb, mom_lb, interaction, rebal, lev in amihud_variants:
        params = {
            "k_per_side": 2,
            "lookback_bars": lb,
            "mom_lookback_bars": mom_lb,
            "use_interaction": interaction,
            "vol_lookback_bars": 168,
            "target_gross_leverage": lev,
            "rebalance_interval_bars": rebal,
        }
        print(f"\n  >> {name} (lb={lb} mom_lb={mom_lb} int={interaction} rebal={rebal})", flush=True)
        r = sweep_variant(name, "amihud_illiquidity_alpha", params)
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
    print(f"  VARIANTS WITH 3+ POSITIVE YEARS")
    print(f"{'='*80}")
    for name, r in ranked:
        pos = sum(1 for y in YEARS if isinstance(r.get(y, {}).get("sharpe"), (int, float)) and r[y]["sharpe"] > 0)
        if pos >= 3:
            sharpes_str = " | ".join(f"{y}:{r.get(y, {}).get('sharpe', '?')}" for y in YEARS)
            print(f"  {name:<24} AVG={r['_avg_sharpe']:>7.3f} MIN={r['_min_sharpe']:>7.3f} pos={pos}  [{sharpes_str}]")

    # Save
    out_path = Path(OUT_DIR) / "phase60b_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
