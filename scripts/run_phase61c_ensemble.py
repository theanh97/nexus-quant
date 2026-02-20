#!/usr/bin/env python3
"""Phase 61C: Idiosyncratic Momentum ensemble analysis — the new champion?"""
import json
import math
import statistics
import subprocess
import sys
import copy
from pathlib import Path

OUT_DIR = "artifacts/phase61c"
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

VB_CHAMPION_PARAMS = {
    "k_per_side": 2,
    "vol_short_bars": 24, "vol_long_bars": 168,
    "compression_threshold": 0.7, "return_lookback_bars": 48,
    "vol_lookback_bars": 168, "target_gross_leverage": 0.30,
    "rebalance_interval_bars": 48,
}

SR_CHAMPION_PARAMS = {
    "k_per_side": 2, "lookback_bars": 336, "vol_lookback_bars": 168,
    "target_gross_leverage": 0.35, "rebalance_interval_bars": 48,
}

IDIO_BEST_PARAMS = {
    "k_per_side": 2,
    "lookback_bars": 336, "beta_window_bars": 72,
    "vol_lookback_bars": 168, "target_gross_leverage": 0.30,
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


def make_config(run_name, year, strategy_name, params):
    cfg = copy.deepcopy(BASE_CONFIG)
    start, end = YEAR_RANGES[year]
    cfg["run_name"] = run_name
    cfg["data"]["start"] = start
    cfg["data"]["end"] = end
    cfg["strategy"] = {"name": strategy_name, "params": params}
    path = f"/tmp/phase61c_{run_name}_{year}.json"
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
                s = m.get("sharpe", "?")
                print(f"    {year}: Sharpe={s}", flush=True)
                continue
        results[year] = {"error": "no result"}
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
    print("  PHASE 61C: IDIOSYNCRATIC MOMENTUM ENSEMBLE ANALYSIS")
    print("=" * 80, flush=True)

    # Run all 4 strategies
    strats = {
        "v1_long":     ("v1_61c",   "nexus_alpha_v1",        V1_LONG_PARAMS),
        "vol_breakout":("vb_61c",   "vol_breakout_alpha",    VB_CHAMPION_PARAMS),
        "sharpe_ratio":("sr_61c",   "sharpe_ratio_alpha",    SR_CHAMPION_PARAMS),
        "idio_mom":    ("idio_61c", "idio_momentum_alpha",   IDIO_BEST_PARAMS),
    }

    results = {}
    for key, (prefix, sname, params) in strats.items():
        print(f"\n{'='*60}")
        print(f"  Running: {key}", flush=True)
        print(f"{'='*60}", flush=True)
        r = run_all_years(prefix, sname, params)
        sharpes = [r.get(y, {}).get("sharpe", 0) for y in YEARS if isinstance(r.get(y, {}).get("sharpe"), (int, float))]
        avg = round(sum(sharpes)/len(sharpes), 3) if sharpes else 0
        mn = round(min(sharpes), 3) if sharpes else 0
        pos = sum(1 for s in sharpes if s > 0)
        print(f"  -> AVG={avg}, MIN={mn}, {pos}/5 positive", flush=True)
        results[key] = r

    # Correlation matrix
    print(f"\n{'='*80}")
    print(f"  OVERALL CORRELATION MATRIX")
    print(f"{'='*80}", flush=True)

    all_rets = {k: [] for k in results}
    for year in YEARS:
        yr_rets = {}
        for k in results:
            rets = results[k].get(year, {}).get("returns", [])
            if rets:
                yr_rets[k] = rets

        if len(yr_rets) >= 2:
            n = min(len(yr_rets[k]) for k in yr_rets)
            for k in yr_rets:
                all_rets[k].extend(yr_rets[k][:n])

    keys = [k for k in all_rets if all_rets[k]]
    for i, k1 in enumerate(keys):
        for k2 in keys[i+1:]:
            c = correlation(all_rets[k1], all_rets[k2])
            print(f"  corr({k1[:15]:<15}, {k2[:15]:<15}) = {c:.4f}")

    # Ensemble testing
    print(f"\n{'='*80}")
    print(f"  2-WAY: V1-Long + Idiosyncratic Momentum")
    print(f"{'='*80}", flush=True)
    print(f"  {'V1/Idio':<15} {'AVG':>7} {'MIN':>7} {'Pos':>5}  year-by-year")

    best_2way_idio = {"label": "", "avg": 0, "min": 0}
    for w_v1 in [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]:
        w_idio = round(1.0 - w_v1, 2)
        label = f"{int(w_v1*100)}/{int(w_idio*100)}"
        year_sharpes = []
        for year in YEARS:
            v1_r = results["v1_long"].get(year, {}).get("returns", [])
            id_r = results["idio_mom"].get(year, {}).get("returns", [])
            if v1_r and id_r:
                n = min(len(v1_r), len(id_r))
                combined = [w_v1 * v1_r[i] + w_idio * id_r[i] for i in range(n)]
                m = metrics_from_returns(combined)
                year_sharpes.append(m["sharpe"])
        if year_sharpes:
            avg = round(sum(year_sharpes) / len(year_sharpes), 3)
            mn = round(min(year_sharpes), 3)
            pos = sum(1 for s in year_sharpes if s > 0)
            sharpes_str = " ".join(f"{s:.2f}" for s in year_sharpes)
            print(f"  {label:<15} {avg:>7.3f} {mn:>7.3f} {pos:>5}  [{sharpes_str}]")
            if avg > best_2way_idio["avg"]:
                best_2way_idio = {"label": label, "avg": avg, "min": mn}

    print(f"\n  Best V1+IdioMom: {best_2way_idio['label']} → AVG={best_2way_idio['avg']}, MIN={best_2way_idio['min']}")

    print(f"\n{'='*80}")
    print(f"  3-WAY: V1-Long + Sharpe Ratio + Idiosyncratic Momentum")
    print(f"{'='*80}", flush=True)
    weight_combos = [
        (0.55, 0.25, 0.20), (0.55, 0.20, 0.25), (0.50, 0.25, 0.25),
        (0.60, 0.20, 0.20), (0.50, 0.30, 0.20), (0.50, 0.20, 0.30),
        (0.60, 0.15, 0.25), (0.60, 0.25, 0.15), (0.65, 0.15, 0.20),
        (0.65, 0.20, 0.15), (0.70, 0.15, 0.15),
    ]
    print(f"  {'V1/SR/Idio':<20} {'AVG':>7} {'MIN':>7} {'Pos':>5}")
    best_3way = {"label": "", "avg": 0, "min": 0}
    for w_v1, w_sr, w_idio in weight_combos:
        label = f"{int(w_v1*100)}/{int(w_sr*100)}/{int(w_idio*100)}"
        year_sharpes = []
        for year in YEARS:
            v1_r = results["v1_long"].get(year, {}).get("returns", [])
            sr_r = results["sharpe_ratio"].get(year, {}).get("returns", [])
            id_r = results["idio_mom"].get(year, {}).get("returns", [])
            if v1_r and sr_r and id_r:
                n = min(len(v1_r), len(sr_r), len(id_r))
                combined = [w_v1*v1_r[i] + w_sr*sr_r[i] + w_idio*id_r[i] for i in range(n)]
                m = metrics_from_returns(combined)
                year_sharpes.append(m["sharpe"])
        if year_sharpes:
            avg = round(sum(year_sharpes) / len(year_sharpes), 3)
            mn = round(min(year_sharpes), 3)
            pos = sum(1 for s in year_sharpes if s > 0)
            sharpes_str = " ".join(f"{s:.2f}" for s in year_sharpes)
            print(f"  {label:<20} {avg:>7.3f} {mn:>7.3f} {pos:>5}  [{sharpes_str}]")
            if avg > best_3way["avg"]:
                best_3way = {"label": label, "avg": avg, "min": mn}

    print(f"\n  Best 3-way V1+SR+Idio: {best_3way['label']} → AVG={best_3way['avg']}, MIN={best_3way['min']}")

    print(f"\n{'='*80}")
    print(f"  4-WAY: V1 + SR + VB + Idio")
    print(f"{'='*80}", flush=True)
    combos_4way = [
        (0.50, 0.20, 0.10, 0.20), (0.55, 0.15, 0.10, 0.20),
        (0.50, 0.15, 0.15, 0.20), (0.55, 0.20, 0.10, 0.15),
        (0.60, 0.15, 0.10, 0.15), (0.50, 0.20, 0.15, 0.15),
    ]
    print(f"  {'V1/SR/VB/Idio':<22} {'AVG':>7} {'MIN':>7} {'Pos':>5}")
    best_4way = {"label": "", "avg": 0, "min": 0}
    for w_v1, w_sr, w_vb, w_idio in combos_4way:
        label = f"{int(w_v1*100)}/{int(w_sr*100)}/{int(w_vb*100)}/{int(w_idio*100)}"
        year_sharpes = []
        for year in YEARS:
            v1_r = results["v1_long"].get(year, {}).get("returns", [])
            sr_r = results["sharpe_ratio"].get(year, {}).get("returns", [])
            vb_r = results["vol_breakout"].get(year, {}).get("returns", [])
            id_r = results["idio_mom"].get(year, {}).get("returns", [])
            if v1_r and sr_r and vb_r and id_r:
                n = min(len(v1_r), len(sr_r), len(vb_r), len(id_r))
                combined = [w_v1*v1_r[i] + w_sr*sr_r[i] + w_vb*vb_r[i] + w_idio*id_r[i] for i in range(n)]
                m = metrics_from_returns(combined)
                year_sharpes.append(m["sharpe"])
        if year_sharpes:
            avg = round(sum(year_sharpes) / len(year_sharpes), 3)
            mn = round(min(year_sharpes), 3)
            pos = sum(1 for s in year_sharpes if s > 0)
            sharpes_str = " ".join(f"{s:.2f}" for s in year_sharpes)
            print(f"  {label:<22} {avg:>7.3f} {mn:>7.3f} {pos:>5}  [{sharpes_str}]")
            if avg > best_4way["avg"]:
                best_4way = {"label": label, "avg": avg, "min": mn}

    # Final summary
    print(f"\n{'='*80}")
    print(f"  PHASE 61C FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"  STANDALONE:")
    for k in results:
        sharpes = [results[k].get(y, {}).get("sharpe", 0) for y in YEARS if isinstance(results[k].get(y, {}).get("sharpe"), (int, float))]
        if sharpes:
            print(f"    {k:<20} AVG={sum(sharpes)/len(sharpes):.3f} MIN={min(sharpes):.3f} [{' '.join(f'{s:.2f}' for s in sharpes)}]")
    print(f"  ENSEMBLES:")
    print(f"    V1+IdioMom best:    {best_2way_idio['label']} AVG={best_2way_idio['avg']} MIN={best_2way_idio['min']}")
    print(f"    V1+SR+Idio best:    {best_3way['label']} AVG={best_3way['avg']} MIN={best_3way['min']}")
    print(f"    V1+SR+VB+Idio best: {best_4way['label']} AVG={best_4way['avg']} MIN={best_4way['min']}")
    print(f"\n  REFERENCE (Phase 60 champion):")
    print(f"    V1+SR 70/30: AVG=1.184 MIN=1.076")
    print(f"    V1+SR+VB 60/25/15: AVG=1.208 MIN=0.821")

    # Save
    out_path = Path(OUT_DIR) / "phase61c_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"best_2way_idio": best_2way_idio, "best_3way": best_3way, "best_4way": best_4way}, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
