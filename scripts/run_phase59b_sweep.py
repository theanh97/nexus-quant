#!/usr/bin/env python3
"""Phase 59B: Parameter sweep for Vol Breakout + Volume Reversal + new signals."""
import json
import math
import statistics
import subprocess
import sys
import time
import copy
from pathlib import Path

OUT_DIR = "artifacts/phase59b"
CONFIGS_DIR = "configs"

YEARS = ["2021", "2022", "2023", "2024", "2025"]
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT"]

BASE_CONFIG = {
    "seed": 42,
    "venue": {"name": "binance_usdm", "kind": "crypto_perp", "vip_tier": 0},
    "data": {
        "provider": "binance_rest_v1",
        "symbols": SYMBOLS,
        "bar_interval": "1h",
        "cache_dir": ".cache/binance_rest"
    },
    "execution": {"style": "taker", "slippage_bps": 3.0},
    "costs": {"fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0},
    "benchmark": {"version": "v1", "walk_forward": {"enabled": True}},
    "risk": {"max_drawdown": 0.30, "max_turnover_per_rebalance": 0.8, "max_gross_leverage": 0.7, "max_position_per_symbol": 0.3},
    "self_learn": {"enabled": False}
}

YEAR_RANGES = {
    "2021": ("2021-01-01", "2022-01-01"),
    "2022": ("2022-01-01", "2023-01-01"),
    "2023": ("2023-01-01", "2024-01-01"),
    "2024": ("2024-01-01", "2025-01-01"),
    "2025": ("2025-01-01", "2026-01-01"),
}


def compute_metrics(result_path: str) -> dict:
    d = json.load(open(result_path))
    eq = d.get("equity_curve", [])
    rets = d.get("returns", [])
    if not eq or not rets or len(rets) < 100:
        return {"sharpe": 0, "cagr": 0, "max_dd": 0, "error": "insufficient data"}
    total_return = eq[-1] / eq[0] - 1.0 if eq[0] > 0 else 0
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
    """Create a temp config and return its path."""
    cfg = copy.deepcopy(BASE_CONFIG)
    start, end = YEAR_RANGES[year]
    cfg["run_name"] = run_name
    cfg["data"]["start"] = start
    cfg["data"]["end"] = end
    cfg["strategy"] = {"name": strategy_name, "params": params}

    path = f"/tmp/phase59b_{run_name}_{year}.json"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def run_backtest(config_path: str, run_name: str, year: str) -> dict:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "nexus_quant", "run", "--config", config_path, "--out", OUT_DIR],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            return {"error": result.stderr[-100:]}
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}

    runs_dir = Path(OUT_DIR) / "runs"
    if not runs_dir.exists():
        return {"error": "no runs dir"}
    prefix = f"{run_name}"
    matching = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)],
        key=lambda d: d.stat().st_mtime,
    )
    if matching:
        result_path = matching[-1] / "result.json"
        if result_path.exists():
            return compute_metrics(str(result_path))
    return {"error": "no result"}


def sweep_variant(name: str, strategy_name: str, params: dict) -> dict:
    """Run one parameter variant across all 5 years."""
    year_results = {}
    for year in YEARS:
        run_name = f"{name}_{year}"
        config_path = make_config(run_name, year, strategy_name, params)
        metrics = run_backtest(config_path, run_name, year)
        year_results[year] = metrics
        s = metrics.get("sharpe", "ERR")
        print(f"    {year}: Sharpe={s}", flush=True)

    sharpes = [y.get("sharpe", 0) for y in year_results.values() if isinstance(y.get("sharpe"), (int, float))]
    avg = round(sum(sharpes) / len(sharpes), 3) if sharpes else 0
    mn = round(min(sharpes), 3) if sharpes else 0
    year_results["_avg_sharpe"] = avg
    year_results["_min_sharpe"] = mn
    return year_results


def main():
    all_results = {}

    # ========================================================================
    # 1. Vol Breakout parameter sweep
    # ========================================================================
    print("\n" + "=" * 70)
    print("  VOL BREAKOUT — PARAMETER SWEEP")
    print("=" * 70, flush=True)

    vol_breakout_variants = [
        # (name, vol_short, vol_long, compression_thr, return_lb, rebalance)
        ("vb_base",       24,  168, 0.7,  12, 24),
        ("vb_short48",    48,  168, 0.7,  12, 24),
        ("vb_long336",    24,  336, 0.7,  12, 24),
        ("vb_thr05",      24,  168, 0.5,  12, 24),
        ("vb_thr09",      24,  168, 0.9,  12, 24),
        ("vb_ret24",      24,  168, 0.7,  24, 24),
        ("vb_ret48",      24,  168, 0.7,  48, 24),
        ("vb_rebal48",    24,  168, 0.7,  12, 48),
        ("vb_rebal12",    24,  168, 0.7,  12, 12),
        ("vb_k3",         24,  168, 0.7,  12, 24),  # k=3
        ("vb_lev20",      24,  168, 0.7,  12, 24),  # leverage 0.20
        ("vb_lev40",      24,  168, 0.7,  12, 24),  # leverage 0.40
        ("vb_best_combo", 48,  336, 0.5,  24, 48),  # combine best ideas
    ]

    for name, vs, vl, thr, rlb, rebal in vol_breakout_variants:
        print(f"\n  >> {name} (vs={vs} vl={vl} thr={thr} ret={rlb} rebal={rebal})", flush=True)
        params = {
            "k_per_side": 3 if name == "vb_k3" else 2,
            "vol_short_bars": vs,
            "vol_long_bars": vl,
            "compression_threshold": thr,
            "return_lookback_bars": rlb,
            "vol_lookback_bars": 168,
            "target_gross_leverage": 0.20 if name == "vb_lev20" else (0.40 if name == "vb_lev40" else 0.30),
            "rebalance_interval_bars": rebal,
        }
        result = sweep_variant(name, "vol_breakout_alpha", params)
        all_results[name] = result
        print(f"     AVG={result['_avg_sharpe']}, MIN={result['_min_sharpe']}", flush=True)

    # ========================================================================
    # 2. Volume Reversal parameter sweep
    # ========================================================================
    print("\n" + "=" * 70)
    print("  VOLUME REVERSAL — PARAMETER SWEEP")
    print("=" * 70, flush=True)

    vol_rev_variants = [
        # (name, vol_lb, ret_lb, rebalance)
        ("vr_base",    168,  24, 24),
        ("vr_vol72",    72,  24, 24),
        ("vr_vol336",  336,  24, 24),
        ("vr_ret12",   168,  12, 24),
        ("vr_ret48",   168,  48, 24),
        ("vr_rebal48", 168,  24, 48),
        ("vr_rebal12", 168,  24, 12),
        ("vr_combo1",   72,  48, 48),
        ("vr_combo2",  336,  12, 12),
    ]

    for name, vlb, rlb, rebal in vol_rev_variants:
        print(f"\n  >> {name} (vol_lb={vlb} ret_lb={rlb} rebal={rebal})", flush=True)
        params = {
            "k_per_side": 2,
            "volume_lookback_bars": vlb,
            "return_lookback_bars": rlb,
            "vol_lookback_bars": 168,
            "target_gross_leverage": 0.30,
            "rebalance_interval_bars": rebal,
        }
        result = sweep_variant(name, "volume_reversal_alpha", params)
        all_results[name] = result
        print(f"     AVG={result['_avg_sharpe']}, MIN={result['_min_sharpe']}", flush=True)

    # ========================================================================
    # 3. RS Acceleration with tweaks
    # ========================================================================
    print("\n" + "=" * 70)
    print("  RS ACCELERATION — PARAMETER SWEEP")
    print("=" * 70, flush=True)

    rs_variants = [
        # (name, rs_short, rs_long, rebalance)
        ("rs_base",     72,  336, 48),
        ("rs_s24",      24,  336, 48),
        ("rs_s168",    168,  336, 48),
        ("rs_l168",     72,  168, 48),
        ("rs_l672",     72,  672, 48),
        ("rs_rebal24",  72,  336, 24),
        ("rs_rebal96",  72,  336, 96),
        ("rs_combo",   168,  672, 96),
    ]

    for name, rs_s, rs_l, rebal in rs_variants:
        print(f"\n  >> {name} (rs_s={rs_s} rs_l={rs_l} rebal={rebal})", flush=True)
        params = {
            "k_per_side": 2,
            "rs_short_bars": rs_s,
            "rs_long_bars": rs_l,
            "vol_lookback_bars": 168,
            "target_gross_leverage": 0.30,
            "rebalance_interval_bars": rebal,
        }
        result = sweep_variant(name, "rs_acceleration_alpha", params)
        all_results[name] = result
        print(f"     AVG={result['_avg_sharpe']}, MIN={result['_min_sharpe']}", flush=True)

    # ========================================================================
    # 4. NEW SIGNALS: Taker Buy, Funding Contrarian, MR+Funding
    # ========================================================================
    print("\n" + "=" * 70)
    print("  NEW SIGNALS — TAKER BUY / FUNDING CONTRARIAN / MR+FUNDING")
    print("=" * 70, flush=True)

    # Taker Buy Ratio
    for name, lb, rebal in [("tb_48_24", 48, 24), ("tb_24_12", 24, 12), ("tb_72_48", 72, 48), ("tb_168_24", 168, 24)]:
        print(f"\n  >> {name}", flush=True)
        params = {
            "k_per_side": 2,
            "ratio_lookback_bars": lb,
            "vol_lookback_bars": 168,
            "target_gross_leverage": 0.30,
            "rebalance_interval_bars": rebal,
        }
        result = sweep_variant(name, "taker_buy_alpha", params)
        all_results[name] = result
        print(f"     AVG={result['_avg_sharpe']}, MIN={result['_min_sharpe']}", flush=True)

    # Funding Contrarian
    for name, mom_lb, rebal in [("fc_168_48", 168, 48), ("fc_336_60", 336, 60), ("fc_72_24", 72, 24), ("fc_168_24", 168, 24)]:
        print(f"\n  >> {name}", flush=True)
        params = {
            "k_per_side": 2,
            "momentum_lookback_bars": mom_lb,
            "vol_lookback_bars": 168,
            "target_gross_leverage": 0.30,
            "rebalance_interval_bars": rebal,
        }
        result = sweep_variant(name, "funding_contrarian_alpha", params)
        all_results[name] = result
        print(f"     AVG={result['_avg_sharpe']}, MIN={result['_min_sharpe']}", flush=True)

    # MR + Funding Filter
    for name, mr_lb, rebal in [("mrf_48_24", 48, 24), ("mrf_24_12", 24, 12), ("mrf_72_48", 72, 48), ("mrf_72_24", 72, 24), ("mrf_48_48", 48, 48)]:
        print(f"\n  >> {name}", flush=True)
        params = {
            "k_per_side": 2,
            "mr_lookback_bars": mr_lb,
            "vol_lookback_bars": 168,
            "target_gross_leverage": 0.30,
            "rebalance_interval_bars": rebal,
        }
        result = sweep_variant(name, "mr_funding_alpha", params)
        all_results[name] = result
        print(f"     AVG={result['_avg_sharpe']}, MIN={result['_min_sharpe']}", flush=True)

    # Save
    out_path = Path(OUT_DIR) / "phase59b_sweep_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Top results
    print(f"\n{'='*80}")
    print(f"  TOP 10 VARIANTS BY AVG SHARPE")
    print(f"{'='*80}")
    ranked = sorted(all_results.items(), key=lambda x: x[1].get("_avg_sharpe", -99), reverse=True)
    for i, (name, r) in enumerate(ranked[:10]):
        sharpes_str = " | ".join(f"{y}:{r.get(y, {}).get('sharpe', '?')}" for y in YEARS)
        print(f"  {i+1}. {name:<20} AVG={r['_avg_sharpe']:>7} MIN={r['_min_sharpe']:>7}  [{sharpes_str}]")

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
