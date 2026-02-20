#!/usr/bin/env python3
"""Phase 61B: Idiosyncratic Momentum + Sortino ensemble testing."""
import json
import math
import statistics
import subprocess
import sys
import copy
from pathlib import Path

OUT_DIR = "artifacts/phase61b"
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


def make_config(run_name, year, strategy_name, params):
    cfg = copy.deepcopy(BASE_CONFIG)
    start, end = YEAR_RANGES[year]
    cfg["run_name"] = run_name
    cfg["data"]["start"] = start
    cfg["data"]["end"] = end
    cfg["strategy"] = {"name": strategy_name, "params": params}
    path = f"/tmp/phase61b_{run_name}_{year}.json"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def run_all_years(prefix, strategy_name, params):
    results = {}
    for year in YEARS:
        run_name = f"{prefix}_{year}"
        path = make_config(run_name, year, strategy_name, params)
        try:
            r = subprocess.run(
                [sys.executable, "-m", "nexus_quant", "run", "--config", path, "--out", OUT_DIR],
                capture_output=True, text=True, timeout=600,
            )
            if r.returncode != 0:
                results[year] = {"error": r.stderr[-100:]}
                print(f"    {year}: ERROR", flush=True)
                continue
        except subprocess.TimeoutExpired:
            results[year] = {"error": "timeout"}
            continue
        runs_dir = Path(OUT_DIR) / "runs"
        matching = sorted(
            [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith(run_name)],
            key=lambda d: d.stat().st_mtime,
        ) if runs_dir.exists() else []
        if matching:
            rp = matching[-1] / "result.json"
            if rp.exists():
                m = compute_metrics(str(rp))
                results[year] = m
                print(f"    {year}: Sharpe={m.get('sharpe', '?')}", flush=True)
                continue
        results[year] = {"error": "no result"}
        print(f"    {year}: no result", flush=True)

    sharpes = [y.get("sharpe", 0) for y in results.values() if isinstance(y.get("sharpe"), (int, float))]
    avg = round(sum(sharpes) / len(sharpes), 3) if sharpes else 0
    mn = round(min(sharpes), 3) if sharpes else 0
    pos = sum(1 for s in sharpes if s > 0)
    print(f"    -> AVG={avg}, MIN={mn}, {pos}/5 positive", flush=True)
    return results


def correlation(xs, ys):
    n = min(len(xs), len(ys))
    if n < 100: return 0.0
    xs, ys = xs[:n], ys[:n]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sx, sy = statistics.pstdev(xs), statistics.pstdev(ys)
    if sx < 1e-12 or sy < 1e-12: return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    return cov / (sx * sy)


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
    print("  PHASE 61B: IDIOSYNCRATIC MOMENTUM + SORTINO ENSEMBLE")
    print("=" * 80, flush=True)

    # ====================================================================
    # 1. Idiosyncratic Momentum — 5-year test
    # ====================================================================
    print(f"\n{'='*70}")
    print("  IDIOSYNCRATIC MOMENTUM ALPHA — INITIAL TEST")
    print(f"{'='*70}", flush=True)

    idio_variants = [
        ("idio_336_168", 336, 168, 0.30, 48),   # default
        ("idio_168_72",  168, 72,  0.30, 48),   # shorter windows
        ("idio_336_72",  336, 72,  0.30, 48),   # long lookback, short beta window
        ("idio_504_168", 504, 168, 0.30, 48),   # longer lookback
        ("idio_336_168_r24", 336, 168, 0.30, 24),  # more frequent rebal
        ("idio_336_168_lev35", 336, 168, 0.35, 48), # higher leverage
    ]

    idio_results = {}
    for name, lb, beta_w, lev, rebal in idio_variants:
        params = {
            "k_per_side": 2,
            "lookback_bars": lb,
            "beta_window_bars": beta_w,
            "vol_lookback_bars": 168,
            "target_gross_leverage": lev,
            "rebalance_interval_bars": rebal,
        }
        print(f"\n  >> {name} (lb={lb} beta_w={beta_w} lev={lev} rebal={rebal})", flush=True)
        r = run_all_years(name, "idio_momentum_alpha", params)
        idio_results[name] = r

    # ====================================================================
    # 2. Sortino in ensemble with V1-Long
    # ====================================================================
    print(f"\n{'='*70}")
    print("  SORTINO 504h — ENSEMBLE WITH V1-LONG")
    print(f"{'='*70}", flush=True)

    sort504_params = {
        "k_per_side": 2, "lookback_bars": 504, "target_return": 0.0,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.35, "rebalance_interval_bars": 48,
    }
    print("\n  Running Sortino 504h:", flush=True)
    sort504_results = run_all_years("sort504_ens", "sortino_alpha", sort504_params)

    print("\n  Running V1-Long:", flush=True)
    v1_results = run_all_years("v1_61b", "nexus_alpha_v1", V1_LONG_PARAMS)

    # Correlation analysis
    print(f"\n{'='*70}")
    print("  CORRELATION ANALYSIS")
    print(f"{'='*70}", flush=True)

    all_v1 = []
    all_sort504 = []
    best_idio_rets = {}

    for year in YEARS:
        v1_rets = v1_results.get(year, {}).get("returns", [])
        sort_rets = sort504_results.get(year, {}).get("returns", [])
        if v1_rets:
            all_v1.extend(v1_rets[:8760])
        if sort_rets:
            all_sort504.extend(sort_rets[:8760])

    # Best idio variant (collect returns from 336_168)
    for year in YEARS:
        r = idio_results.get("idio_336_168", {}).get(year, {})
        if r.get("returns"):
            best_idio_rets[year] = r["returns"]

    corr_v1_sort = correlation(all_v1, all_sort504)
    print(f"  corr(V1-Long, Sortino504) = {corr_v1_sort:.4f}")

    all_idio = []
    for yr in YEARS:
        if yr in best_idio_rets:
            all_idio.extend(best_idio_rets[yr][:8760])
    if all_idio and all_v1:
        corr_v1_idio = correlation(all_v1[:len(all_idio)], all_idio)
        print(f"  corr(V1-Long, IdioMom336) = {corr_v1_idio:.4f}")

    # Ensemble: V1 + Sortino 504h
    print(f"\n  -- V1-Long + Sortino 504h --")
    for w_v1, w_sr, label in [(0.90, 0.10, "90/10"), (0.80, 0.20, "80/20"), (0.70, 0.30, "70/30")]:
        year_sharpes = []
        for year in YEARS:
            v1_r = v1_results.get(year, {}).get("returns", [])
            sr_r = sort504_results.get(year, {}).get("returns", [])
            if v1_r and sr_r:
                n = min(len(v1_r), len(sr_r))
                combined = [w_v1 * v1_r[i] + w_sr * sr_r[i] for i in range(n)]
                m = metrics_from_returns(combined)
                year_sharpes.append(m["sharpe"])
        if year_sharpes:
            avg = round(sum(year_sharpes) / len(year_sharpes), 3)
            mn = round(min(year_sharpes), 3)
            pos = sum(1 for s in year_sharpes if s > 0)
            sharpes_str = " ".join(f"{s:.2f}" for s in year_sharpes)
            print(f"  {label}: AVG={avg}, MIN={mn}, {pos}/5  [{sharpes_str}]")

    # Summary
    print(f"\n{'='*80}")
    print(f"  PHASE 61B SUMMARY — ALL NEW VARIANTS")
    print(f"{'='*80}")
    all_results = dict(idio_results)
    all_results["sort504_ens"] = sort504_results
    ranked = []
    for name, yr_dict in all_results.items():
        sharpes = [yr_dict.get(y, {}).get("sharpe", 0) for y in YEARS if isinstance(yr_dict.get(y, {}).get("sharpe"), (int, float))]
        if sharpes:
            avg = round(sum(sharpes) / len(sharpes), 3)
            mn = round(min(sharpes), 3)
            pos = sum(1 for s in sharpes if s > 0)
            ranked.append((name, avg, mn, pos, sharpes))

    ranked.sort(key=lambda x: x[1], reverse=True)
    for name, avg, mn, pos, sharpes in ranked:
        sharpes_str = " ".join(f"{s:.2f}" for s in sharpes)
        print(f"  {name:<25} AVG={avg:>7.3f} MIN={mn:>7.3f} pos={pos}  [{sharpes_str}]")

    # Save
    out_path = Path(OUT_DIR) / "phase61b_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({k: {y: {kk: vv for kk, vv in v.items() if kk != "returns"} for y, v in yr_d.items() if isinstance(v, dict)} for k, yr_d in all_results.items()}, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
