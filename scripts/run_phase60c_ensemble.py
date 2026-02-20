#!/usr/bin/env python3
"""Phase 60C: Correlation analysis + ensemble testing for Phase 60 champions.

Tests:
1. V1-Long vs Multi-TF Momentum: correlation analysis
2. V1-Long vs Sharpe Ratio Alpha: correlation analysis
3. Ensemble combinations: V1 + MultiTF, V1 + Sharpe, V1 + VB + MultiTF (3-way)
"""
import json
import math
import statistics
import subprocess
import sys
import copy
from pathlib import Path

OUT_DIR = "artifacts/phase60c"
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

# V1-Long champion
V1_LONG_PARAMS = {
    "k_per_side": 2,
    "w_carry": 0.35, "w_mom": 0.45, "w_confirm": 0.0,
    "w_mean_reversion": 0.20, "w_vol_momentum": 0.0, "w_funding_trend": 0.0,
    "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 72,
    "vol_lookback_bars": 168,
    "target_portfolio_vol": 0.0, "use_min_variance": False,
    "target_gross_leverage": 0.35, "min_gross_leverage": 0.05, "max_gross_leverage": 0.65,
    "rebalance_interval_bars": 60, "strict_agreement": False
}

# Vol Breakout champion (Phase 59)
VB_CHAMPION_PARAMS = {
    "k_per_side": 2,
    "vol_short_bars": 24, "vol_long_bars": 168,
    "compression_threshold": 0.7, "return_lookback_bars": 48,
    "vol_lookback_bars": 168, "target_gross_leverage": 0.30,
    "rebalance_interval_bars": 48,
}

# Multi-TF Momentum champion (mtf_slow from Phase 60B): AVG 0.543, ALL 5 YEARS POSITIVE
MULTITF_BASE_PARAMS = {
    "k_per_side": 2,
    "w_24h": 0.05, "w_72h": 0.10, "w_168h": 0.40, "w_336h": 0.45,
    "vol_lookback_bars": 168,
    "target_gross_leverage": 0.35,
    "rebalance_interval_bars": 48,
}

# Sharpe Ratio Alpha champion (sr_336h from Phase 60B): AVG 0.846, ALL 5 YEARS POSITIVE
SHARPE_BASE_PARAMS = {
    "k_per_side": 2,
    "lookback_bars": 336,
    "vol_lookback_bars": 168,
    "target_gross_leverage": 0.35,
    "rebalance_interval_bars": 48,
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
    return {"sharpe": round(sharpe, 3), "cagr": round(cagr * 100, 2), "max_dd": round(max_dd * 100, 2), "returns": rets, "equity": eq}


def make_config(run_name: str, year: str, strategy_name: str, params: dict) -> str:
    cfg = copy.deepcopy(BASE_CONFIG)
    start, end = YEAR_RANGES[year]
    cfg["run_name"] = run_name
    cfg["data"]["start"] = start
    cfg["data"]["end"] = end
    cfg["strategy"] = {"name": strategy_name, "params": params}
    path = f"/tmp/phase60c_{run_name}_{year}.json"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def run_and_get_results(run_name: str, year: str, strategy_name: str, params: dict) -> dict:
    config_path = make_config(run_name, year, strategy_name, params)
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


def correlation(xs, ys):
    n = min(len(xs), len(ys))
    if n < 100:
        return 0.0
    xs, ys = xs[:n], ys[:n]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sx, sy = statistics.pstdev(xs), statistics.pstdev(ys)
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    return cov / (sx * sy)


def ensemble_returns(rets_dict, weights_dict):
    """Combine multiple return series with weights. weights_dict: {name: weight}"""
    names = list(weights_dict.keys())
    min_len = min(len(rets_dict[n]) for n in names if n in rets_dict)
    combined = []
    for i in range(min_len):
        val = sum(weights_dict[n] * rets_dict[n][i] for n in names if n in rets_dict)
        combined.append(val)
    return combined


def metrics_from_returns(rets):
    eq = [1.0]
    for r in rets:
        eq.append(eq[-1] * (1.0 + r))
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


def main():
    print("=" * 80)
    print("  PHASE 60C: CORRELATION ANALYSIS + ENSEMBLE")
    print("=" * 80, flush=True)

    # Load best params from Phase 60B if available
    b_path = Path("artifacts/phase60b/phase60b_results.json")
    if b_path.exists():
        b_results = json.loads(b_path.read_text())
        # Find best multi-tf variant
        mtf_variants = {k: v for k, v in b_results.items() if k.startswith("mtf_")}
        if mtf_variants:
            best_mtf = max(mtf_variants.items(), key=lambda x: x[1].get("_avg_sharpe", -99))
            print(f"  Best MultiTF from 60B: {best_mtf[0]} (AVG={best_mtf[1]['_avg_sharpe']})")
        # Find best sharpe variant
        sr_variants = {k: v for k, v in b_results.items() if k.startswith("sr_")}
        if sr_variants:
            best_sr = max(sr_variants.items(), key=lambda x: x[1].get("_avg_sharpe", -99))
            print(f"  Best SharpeRatio from 60B: {best_sr[0]} (AVG={best_sr[1]['_avg_sharpe']})")
    else:
        print("  Phase 60B results not found, using default params")

    # Run all strategies across all years
    results = {s: {} for s in ["v1_long", "vol_breakout", "multitf_mom", "sharpe_ratio"]}
    strat_configs = [
        ("v1_long",     "v1long_c",    "nexus_alpha_v1",        V1_LONG_PARAMS),
        ("vol_breakout","vb_champ_c",  "vol_breakout_alpha",    VB_CHAMPION_PARAMS),
        ("multitf_mom", "mtf_c",       "multitf_momentum_alpha", MULTITF_BASE_PARAMS),
        ("sharpe_ratio","sr_c",        "sharpe_ratio_alpha",    SHARPE_BASE_PARAMS),
    ]

    for strat_key, run_prefix, strategy_name, params in strat_configs:
        print(f"\n{'='*70}")
        print(f"  Running: {strat_key}", flush=True)
        print(f"{'='*70}", flush=True)
        for year in YEARS:
            run_name = f"{run_prefix}_{year}"
            m = run_and_get_results(run_name, year, strategy_name, params)
            results[strat_key][year] = m
            s = m.get("sharpe", "ERR")
            print(f"    {year}: Sharpe={s}", flush=True)

    # Correlation analysis
    print(f"\n{'='*80}")
    print(f"  CORRELATION ANALYSIS")
    print(f"{'='*80}", flush=True)

    all_rets = {k: [] for k in results}
    for year in YEARS:
        year_rets = {}
        for k in results:
            r = results[k].get(year, {})
            rets = r.get("returns", [])
            if rets:
                year_rets[k] = rets

        if len(year_rets) >= 2:
            keys = list(year_rets.keys())
            n = min(len(year_rets[k]) for k in keys)
            for k in keys:
                all_rets[k].extend(year_rets[k][:n])

            # Print pairwise correlations for this year
            for i, k1 in enumerate(keys):
                for k2 in keys[i+1:]:
                    c = correlation(year_rets[k1], year_rets[k2])
                    print(f"  {year}: corr({k1[:10]}, {k2[:10]}) = {c:.3f}")

    print(f"\n  OVERALL CORRELATIONS:")
    keys_with_data = [k for k in all_rets if all_rets[k]]
    for i, k1 in enumerate(keys_with_data):
        for k2 in keys_with_data[i+1:]:
            c = correlation(all_rets[k1], all_rets[k2])
            print(f"  corr({k1[:15]}, {k2[:15]}) = {c:.4f}")

    # Ensemble testing
    print(f"\n{'='*80}")
    print(f"  ENSEMBLE TESTING")
    print(f"{'='*80}", flush=True)

    # 2-way ensembles
    print(f"\n  -- 2-WAY: V1-Long + Multi-TF Momentum --")
    for w_v1, w_mtf, label in [(0.90, 0.10, "90/10"), (0.80, 0.20, "80/20"), (0.70, 0.30, "70/30"), (0.60, 0.40, "60/40")]:
        year_sharpes = []
        for year in YEARS:
            v1_r = results["v1_long"].get(year, {}).get("returns", [])
            mtf_r = results["multitf_mom"].get(year, {}).get("returns", [])
            if v1_r and mtf_r:
                combined = ensemble_returns({"v1": v1_r, "mtf": mtf_r}, {"v1": w_v1, "mtf": w_mtf})
                m = metrics_from_returns(combined)
                print(f"    {label} | {year}: Sharpe={m['sharpe']}, MDD={m['max_dd']}%", flush=True)
                year_sharpes.append(m["sharpe"])
        if year_sharpes:
            avg = round(sum(year_sharpes) / len(year_sharpes), 3)
            mn = round(min(year_sharpes), 3)
            print(f"    {label} -> AVG={avg}, MIN={mn}")

    print(f"\n  -- 2-WAY: V1-Long + Sharpe Ratio --")
    for w_v1, w_sr, label in [(0.90, 0.10, "90/10"), (0.80, 0.20, "80/20"), (0.70, 0.30, "70/30")]:
        year_sharpes = []
        for year in YEARS:
            v1_r = results["v1_long"].get(year, {}).get("returns", [])
            sr_r = results["sharpe_ratio"].get(year, {}).get("returns", [])
            if v1_r and sr_r:
                combined = ensemble_returns({"v1": v1_r, "sr": sr_r}, {"v1": w_v1, "sr": w_sr})
                m = metrics_from_returns(combined)
                print(f"    {label} | {year}: Sharpe={m['sharpe']}, MDD={m['max_dd']}%", flush=True)
                year_sharpes.append(m["sharpe"])
        if year_sharpes:
            avg = round(sum(year_sharpes) / len(year_sharpes), 3)
            mn = round(min(year_sharpes), 3)
            print(f"    {label} -> AVG={avg}, MIN={mn}")

    print(f"\n  -- 3-WAY: V1-Long + Vol Breakout + Multi-TF --")
    for w_v1, w_vb, w_mtf, label in [
        (0.70, 0.15, 0.15, "70/15/15"),
        (0.65, 0.20, 0.15, "65/20/15"),
        (0.60, 0.20, 0.20, "60/20/20"),
        (0.70, 0.10, 0.20, "70/10/20"),
        (0.60, 0.15, 0.25, "60/15/25"),
    ]:
        year_sharpes = []
        for year in YEARS:
            v1_r = results["v1_long"].get(year, {}).get("returns", [])
            vb_r = results["vol_breakout"].get(year, {}).get("returns", [])
            mtf_r = results["multitf_mom"].get(year, {}).get("returns", [])
            if v1_r and vb_r and mtf_r:
                combined = ensemble_returns(
                    {"v1": v1_r, "vb": vb_r, "mtf": mtf_r},
                    {"v1": w_v1, "vb": w_vb, "mtf": w_mtf}
                )
                m = metrics_from_returns(combined)
                year_sharpes.append(m["sharpe"])
        if year_sharpes:
            avg = round(sum(year_sharpes) / len(year_sharpes), 3)
            mn = round(min(year_sharpes), 3)
            pos = sum(1 for s in year_sharpes if s > 0)
            print(f"    {label}: AVG={avg}, MIN={mn}, {pos}/5 positive")

    print(f"\n  -- 4-WAY: V1-Long + Vol Breakout + Multi-TF + Sharpe --")
    for w_v1, w_vb, w_mtf, w_sr, label in [
        (0.65, 0.15, 0.10, 0.10, "65/15/10/10"),
        (0.60, 0.15, 0.15, 0.10, "60/15/15/10"),
        (0.55, 0.15, 0.15, 0.15, "55/15/15/15"),
    ]:
        year_sharpes = []
        for year in YEARS:
            v1_r = results["v1_long"].get(year, {}).get("returns", [])
            vb_r = results["vol_breakout"].get(year, {}).get("returns", [])
            mtf_r = results["multitf_mom"].get(year, {}).get("returns", [])
            sr_r = results["sharpe_ratio"].get(year, {}).get("returns", [])
            if v1_r and vb_r and mtf_r and sr_r:
                combined = ensemble_returns(
                    {"v1": v1_r, "vb": vb_r, "mtf": mtf_r, "sr": sr_r},
                    {"v1": w_v1, "vb": w_vb, "mtf": w_mtf, "sr": w_sr}
                )
                m = metrics_from_returns(combined)
                year_sharpes.append(m["sharpe"])
        if year_sharpes:
            avg = round(sum(year_sharpes) / len(year_sharpes), 3)
            mn = round(min(year_sharpes), 3)
            pos = sum(1 for s in year_sharpes if s > 0)
            print(f"    {label}: AVG={avg}, MIN={mn}, {pos}/5 positive")

    # Save results
    save_data = {
        "strategies": {
            k: {y: {kk: vv for kk, vv in results[k][y].items() if kk not in ("returns", "equity")} for y in YEARS}
            for k in results
        },
        "correlations_overall": {},
    }
    for i, k1 in enumerate(keys_with_data):
        for k2 in keys_with_data[i+1:]:
            save_data["correlations_overall"][f"{k1}_vs_{k2}"] = correlation(all_rets[k1], all_rets[k2])

    out_path = Path(OUT_DIR) / "phase60c_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
