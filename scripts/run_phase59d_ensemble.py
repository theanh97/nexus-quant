#!/usr/bin/env python3
"""Phase 59D: Correlation analysis + Ensemble test (V1-Long + Vol Breakout)."""
import json
import math
import statistics
import subprocess
import sys
import copy
from pathlib import Path

OUT_DIR = "artifacts/phase59d"
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

# V1-Long champion params
V1_LONG_PARAMS = {
    "k_per_side": 2,
    "w_carry": 0.35,
    "w_mom": 0.45,
    "w_confirm": 0.0,
    "w_mean_reversion": 0.20,
    "w_vol_momentum": 0.0,
    "w_funding_trend": 0.0,
    "momentum_lookback_bars": 336,
    "mean_reversion_lookback_bars": 72,
    "vol_lookback_bars": 168,
    "target_portfolio_vol": 0.0,
    "use_min_variance": False,
    "target_gross_leverage": 0.35,
    "min_gross_leverage": 0.05,
    "max_gross_leverage": 0.65,
    "rebalance_interval_bars": 60,
    "strict_agreement": False
}

# Vol Breakout champion params (vb2_ret48_r48)
VB_CHAMPION_PARAMS = {
    "k_per_side": 2,
    "vol_short_bars": 24,
    "vol_long_bars": 168,
    "compression_threshold": 0.7,
    "return_lookback_bars": 48,
    "vol_lookback_bars": 168,
    "target_gross_leverage": 0.30,
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
    path = f"/tmp/phase59d_{run_name}_{year}.json"
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
    """Pearson correlation between two return series."""
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


def ensemble_returns(rets_a, rets_b, w_a, w_b):
    """Combine two return series with weights."""
    n = min(len(rets_a), len(rets_b))
    return [w_a * rets_a[i] + w_b * rets_b[i] for i in range(n)]


def metrics_from_returns(rets):
    """Compute Sharpe/CAGR/MDD from combined returns."""
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
    print("  PHASE 59D: CORRELATION ANALYSIS + ENSEMBLE")
    print("=" * 80, flush=True)

    v1_results = {}
    vb_results = {}

    # Run both strategies for all years
    for year in YEARS:
        print(f"\n--- {year} ---", flush=True)

        print(f"  V1-Long...", flush=True)
        v1 = run_and_get_results(f"v1long_{year}", year, "nexus_alpha_v1", V1_LONG_PARAMS)
        v1_results[year] = v1
        print(f"    Sharpe={v1.get('sharpe', 'ERR')}", flush=True)

        print(f"  Vol Breakout...", flush=True)
        vb = run_and_get_results(f"vb_champ_{year}", year, "vol_breakout_alpha", VB_CHAMPION_PARAMS)
        vb_results[year] = vb
        print(f"    Sharpe={vb.get('sharpe', 'ERR')}", flush=True)

    # Correlation analysis
    print(f"\n{'='*80}")
    print(f"  CORRELATION ANALYSIS")
    print(f"{'='*80}", flush=True)

    all_v1_rets = []
    all_vb_rets = []
    for year in YEARS:
        v1_rets = v1_results[year].get("returns", [])
        vb_rets = vb_results[year].get("returns", [])
        if v1_rets and vb_rets:
            n = min(len(v1_rets), len(vb_rets))
            corr = correlation(v1_rets[:n], vb_rets[:n])
            print(f"  {year}: correlation = {corr:.3f}")
            all_v1_rets.extend(v1_rets[:n])
            all_vb_rets.extend(vb_rets[:n])

    if all_v1_rets and all_vb_rets:
        overall_corr = correlation(all_v1_rets, all_vb_rets)
        print(f"  OVERALL: correlation = {overall_corr:.3f}")

    # Ensemble testing
    print(f"\n{'='*80}")
    print(f"  ENSEMBLE TESTING (V1-Long + Vol Breakout)")
    print(f"{'='*80}", flush=True)

    weight_combos = [
        (0.90, 0.10, "90/10"),
        (0.80, 0.20, "80/20"),
        (0.70, 0.30, "70/30"),
        (0.60, 0.40, "60/40"),
        (0.50, 0.50, "50/50"),
    ]

    ensemble_results = {}
    for w_v1, w_vb, label in weight_combos:
        print(f"\n  Ensemble {label} (V1={w_v1}, VB={w_vb}):", flush=True)
        year_sharpes = []
        for year in YEARS:
            v1_rets = v1_results[year].get("returns", [])
            vb_rets = vb_results[year].get("returns", [])
            if v1_rets and vb_rets:
                combined = ensemble_returns(v1_rets, vb_rets, w_v1, w_vb)
                m = metrics_from_returns(combined)
                print(f"    {year}: Sharpe={m['sharpe']}, CAGR={m['cagr']}%, MDD={m['max_dd']}%", flush=True)
                year_sharpes.append(m["sharpe"])

        if year_sharpes:
            avg = round(sum(year_sharpes) / len(year_sharpes), 3)
            mn = round(min(year_sharpes), 3)
            print(f"    -> AVG Sharpe={avg}, MIN Sharpe={mn}", flush=True)
            ensemble_results[label] = {"avg_sharpe": avg, "min_sharpe": mn, "year_sharpes": year_sharpes}

    # Comparison summary
    print(f"\n{'='*80}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*80}")

    v1_sharpes = [v1_results[y].get("sharpe", 0) for y in YEARS if isinstance(v1_results[y].get("sharpe"), (int, float))]
    vb_sharpes = [vb_results[y].get("sharpe", 0) for y in YEARS if isinstance(vb_results[y].get("sharpe"), (int, float))]

    if v1_sharpes:
        print(f"  V1-Long standalone:  AVG={sum(v1_sharpes)/len(v1_sharpes):.3f}, MIN={min(v1_sharpes):.3f}")
    if vb_sharpes:
        print(f"  VB standalone:       AVG={sum(vb_sharpes)/len(vb_sharpes):.3f}, MIN={min(vb_sharpes):.3f}")

    for label, er in ensemble_results.items():
        print(f"  Ensemble {label}:     AVG={er['avg_sharpe']:.3f}, MIN={er['min_sharpe']:.3f}")

    # Save all results
    save_data = {
        "v1_long": {y: {k: v for k, v in v1_results[y].items() if k not in ("returns", "equity")} for y in YEARS},
        "vol_breakout": {y: {k: v for k, v in vb_results[y].items() if k not in ("returns", "equity")} for y in YEARS},
        "ensemble": ensemble_results,
        "correlation": {y: correlation(v1_results[y].get("returns", []), vb_results[y].get("returns", [])) for y in YEARS if v1_results[y].get("returns") and vb_results[y].get("returns")},
    }

    out_path = Path(OUT_DIR) / "phase59d_ensemble_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
