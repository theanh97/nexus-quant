#!/usr/bin/env python3
"""Phase 59C: Deep optimization — Hybrid Alpha + best variant tuning."""
import json
import math
import statistics
import subprocess
import sys
import copy
from pathlib import Path

OUT_DIR = "artifacts/phase59c"
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
    path = f"/tmp/phase59c_{run_name}_{year}.json"
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
            return {"error": result.stderr[-100:]}
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


def sweep_variant(name: str, strategy_name: str, params: dict) -> dict:
    year_results = {}
    for year in YEARS:
        run_name = f"{name}_{year}"
        config_path = make_config(run_name, year, strategy_name, params)
        metrics = run_backtest(config_path, run_name)
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
    # 1. HYBRID ALPHA — Grid search
    # ========================================================================
    print("\n" + "=" * 70)
    print("  HYBRID ALPHA — GRID SEARCH")
    print("=" * 70, flush=True)

    hybrid_variants = [
        # (name, w_vb, w_vr, w_fc, vs, vl, thr, ret_lb, vol_lb_vol, mom_lb, rebal, k)
        ("hyb_base",        0.40, 0.30, 0.30, 24,  168, 0.7, 48,  168, 336, 48, 2),
        ("hyb_vb_heavy",    0.60, 0.20, 0.20, 24,  168, 0.7, 48,  168, 336, 48, 2),
        ("hyb_vr_heavy",    0.20, 0.50, 0.30, 24,  168, 0.7, 48,  168, 336, 48, 2),
        ("hyb_fc_heavy",    0.20, 0.20, 0.60, 24,  168, 0.7, 48,  168, 336, 48, 2),
        ("hyb_equal",       0.33, 0.33, 0.34, 24,  168, 0.7, 48,  168, 336, 48, 2),
        # Best vol breakout params
        ("hyb_vb_opt",      0.50, 0.25, 0.25, 24,  168, 0.7, 48,  168, 336, 48, 2),
        # Slower rebalance
        ("hyb_rebal60",     0.40, 0.30, 0.30, 24,  168, 0.7, 48,  168, 336, 60, 2),
        ("hyb_rebal96",     0.40, 0.30, 0.30, 24,  168, 0.7, 48,  168, 336, 96, 2),
        # Longer vol window
        ("hyb_vlong336",    0.40, 0.30, 0.30, 24,  336, 0.7, 48,  168, 336, 48, 2),
        # Different return lookbacks
        ("hyb_ret24",       0.40, 0.30, 0.30, 24,  168, 0.7, 24,  168, 336, 48, 2),
        ("hyb_ret72",       0.40, 0.30, 0.30, 24,  168, 0.7, 72,  168, 336, 48, 2),
        # k=3
        ("hyb_k3",          0.40, 0.30, 0.30, 24,  168, 0.7, 48,  168, 336, 48, 3),
        # Lower leverage
        ("hyb_lev20",       0.40, 0.30, 0.30, 24,  168, 0.7, 48,  168, 336, 48, 2),
        # Higher leverage
        ("hyb_lev40",       0.40, 0.30, 0.30, 24,  168, 0.7, 48,  168, 336, 48, 2),
        # Shorter mom for FC
        ("hyb_mom168",      0.40, 0.30, 0.30, 24,  168, 0.7, 48,  168, 168, 48, 2),
        # Compression threshold
        ("hyb_thr05",       0.40, 0.30, 0.30, 24,  168, 0.5, 48,  168, 336, 48, 2),
        ("hyb_thr09",       0.40, 0.30, 0.30, 24,  168, 0.9, 48,  168, 336, 48, 2),
        # Vol reversal with shorter vol window
        ("hyb_volrev72",    0.40, 0.30, 0.30, 24,  168, 0.7, 48,   72, 336, 48, 2),
        ("hyb_volrev336",   0.40, 0.30, 0.30, 24,  168, 0.7, 48,  336, 336, 48, 2),
    ]

    for name, w_vb, w_vr, w_fc, vs, vl, thr, ret_lb, volrev_lb, mom_lb, rebal, k in hybrid_variants:
        print(f"\n  >> {name} (vb={w_vb} vr={w_vr} fc={w_fc} ret={ret_lb} rebal={rebal})", flush=True)
        lev = 0.20 if name == "hyb_lev20" else (0.40 if name == "hyb_lev40" else 0.30)
        params = {
            "k_per_side": k,
            "w_vol_breakout": w_vb,
            "w_vol_reversal": w_vr,
            "w_fund_contrarian": w_fc,
            "vol_short_bars": vs,
            "vol_long_bars": vl,
            "compression_threshold": thr,
            "return_lookback_bars": ret_lb,
            "volume_lookback_bars": volrev_lb,
            "momentum_lookback_bars": mom_lb,
            "vol_lookback_bars": 168,
            "target_gross_leverage": lev,
            "rebalance_interval_bars": rebal,
        }
        result = sweep_variant(name, "hybrid_alpha", params)
        all_results[name] = result
        print(f"     AVG={result['_avg_sharpe']}, MIN={result['_min_sharpe']}", flush=True)

    # ========================================================================
    # 2. Vol Breakout DEEP tuning (around vb_ret48 which was best: AVG -0.027)
    # ========================================================================
    print("\n" + "=" * 70)
    print("  VOL BREAKOUT — DEEP TUNING (around vb_ret48)")
    print("=" * 70, flush=True)

    vb_deep = [
        # (name, vs, vl, thr, ret_lb, rebal, lev, k)
        ("vb2_base48",       24, 168, 0.7,  48, 24, 0.30, 2),
        ("vb2_ret48_r48",    24, 168, 0.7,  48, 48, 0.30, 2),
        ("vb2_ret48_r36",    24, 168, 0.7,  48, 36, 0.30, 2),
        ("vb2_ret48_lev25",  24, 168, 0.7,  48, 24, 0.25, 2),
        ("vb2_ret48_lev35",  24, 168, 0.7,  48, 24, 0.35, 2),
        ("vb2_ret36",        24, 168, 0.7,  36, 24, 0.30, 2),
        ("vb2_ret60",        24, 168, 0.7,  60, 24, 0.30, 2),
        ("vb2_ret48_vs48",   48, 168, 0.7,  48, 24, 0.30, 2),
        ("vb2_ret48_vl336",  24, 336, 0.7,  48, 24, 0.30, 2),
        ("vb2_ret48_t06",    24, 168, 0.6,  48, 24, 0.30, 2),
        ("vb2_ret48_t08",    24, 168, 0.8,  48, 24, 0.30, 2),
        ("vb2_combo_best",   48, 336, 0.6,  48, 36, 0.30, 2),
    ]

    for name, vs, vl, thr, ret_lb, rebal, lev, k in vb_deep:
        print(f"\n  >> {name} (vs={vs} vl={vl} thr={thr} ret={ret_lb} rebal={rebal} lev={lev})", flush=True)
        params = {
            "k_per_side": k,
            "vol_short_bars": vs,
            "vol_long_bars": vl,
            "compression_threshold": thr,
            "return_lookback_bars": ret_lb,
            "vol_lookback_bars": 168,
            "target_gross_leverage": lev,
            "rebalance_interval_bars": rebal,
        }
        result = sweep_variant(name, "vol_breakout_alpha", params)
        all_results[name] = result
        print(f"     AVG={result['_avg_sharpe']}, MIN={result['_min_sharpe']}", flush=True)

    # Save
    out_path = Path(OUT_DIR) / "phase59c_deep_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Top results
    print(f"\n{'='*80}")
    print(f"  TOP 15 VARIANTS BY AVG SHARPE")
    print(f"{'='*80}")
    ranked = sorted(all_results.items(), key=lambda x: x[1].get("_avg_sharpe", -99), reverse=True)
    for i, (name, r) in enumerate(ranked[:15]):
        sharpes_str = " | ".join(f"{y}:{r.get(y, {}).get('sharpe', '?')}" for y in YEARS)
        print(f"  {i+1}. {name:<22} AVG={r['_avg_sharpe']:>7} MIN={r['_min_sharpe']:>7}  [{sharpes_str}]")

    # Also show signals that are positive in most years (even if avg is low)
    print(f"\n{'='*80}")
    print(f"  VARIANTS WITH 3+ POSITIVE YEARS")
    print(f"{'='*80}")
    for name, r in ranked:
        pos_years = sum(1 for y in YEARS if isinstance(r.get(y, {}).get("sharpe"), (int, float)) and r[y]["sharpe"] > 0)
        if pos_years >= 3:
            sharpes_str = " | ".join(f"{y}:{r.get(y, {}).get('sharpe', '?')}" for y in YEARS)
            print(f"  {name:<22} AVG={r['_avg_sharpe']:>7} MIN={r['_min_sharpe']:>7} pos_years={pos_years}  [{sharpes_str}]")

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
