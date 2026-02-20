#!/usr/bin/env python3
"""Phase 60D: Deep optimization of Sharpe Ratio Alpha + ensemble weight sweep."""
import json
import math
import statistics
import subprocess
import sys
import copy
from pathlib import Path

OUT_DIR = "artifacts/phase60d"
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
    path = f"/tmp/phase60d_{run_name}_{year}.json"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def run_and_get(run_name, year, strategy_name, params):
    path = make_config(run_name, year, strategy_name, params)
    try:
        r = subprocess.run(
            [sys.executable, "-m", "nexus_quant", "run", "--config", path, "--out", OUT_DIR],
            capture_output=True, text=True, timeout=600,
        )
        if r.returncode != 0:
            return {"error": r.stderr[-100:]}
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
    print("  PHASE 60D: SHARPE RATIO DEEP OPTIMIZATION + ENSEMBLE WEIGHT SWEEP")
    print("=" * 80, flush=True)

    all_results = {}
    sr_year_rets = {}
    v1_year_rets = {}
    vb_year_rets = {}

    # ====================================================================
    # 1. Sharpe Ratio Alpha — deep parameter sweep
    # ====================================================================
    print("\n" + "="*70)
    print("  SHARPE RATIO ALPHA — DEEP PARAM SWEEP")
    print("="*70, flush=True)

    sr_variants = [
        # (name, lookback, rebal, lev, k)
        ("sr_240h", 240, 48, 0.35, 2),
        ("sr_288h", 288, 48, 0.35, 2),
        ("sr_336h", 336, 48, 0.35, 2),   # champion
        ("sr_384h", 384, 48, 0.35, 2),
        ("sr_432h", 432, 48, 0.35, 2),
        ("sr_480h", 480, 48, 0.35, 2),
        ("sr_336_r36", 336, 36, 0.35, 2),
        ("sr_336_r60", 336, 60, 0.35, 2),
        ("sr_336_lev30", 336, 48, 0.30, 2),
        ("sr_336_lev40", 336, 48, 0.40, 2),
        ("sr_336_k3", 336, 48, 0.35, 3),
        ("sr_336_k1", 336, 48, 0.35, 1),
    ]

    for name, lb, rebal, lev, k in sr_variants:
        params = {
            "k_per_side": k,
            "lookback_bars": lb,
            "vol_lookback_bars": 168,
            "target_gross_leverage": lev,
            "rebalance_interval_bars": rebal,
        }
        print(f"\n  >> {name} (lb={lb} rebal={rebal} lev={lev} k={k})", flush=True)
        yr_sharpes = []
        yr_rets = {}
        for year in YEARS:
            m = run_and_get(f"{name}_{year}", year, "sharpe_ratio_alpha", params)
            all_results[f"sr_{name}"] = all_results.get(f"sr_{name}", {})
            all_results[f"sr_{name}"][year] = m
            s = m.get("sharpe", "ERR")
            print(f"    {year}: Sharpe={s}", flush=True)
            if isinstance(s, (int, float)):
                yr_sharpes.append(s)
            if m.get("returns"):
                yr_rets[year] = m["returns"]
        avg = round(sum(yr_sharpes) / len(yr_sharpes), 3) if yr_sharpes else 0
        mn = round(min(yr_sharpes), 3) if yr_sharpes else 0
        pos = sum(1 for s in yr_sharpes if s > 0)
        print(f"     -> AVG={avg}, MIN={mn}, {pos}/5 positive", flush=True)
        if yr_rets:
            sr_year_rets[name] = yr_rets

    # ====================================================================
    # 2. Run V1-Long and Vol Breakout for ensemble
    # ====================================================================
    print("\n" + "="*70)
    print("  RUNNING V1-LONG + VOL BREAKOUT FOR ENSEMBLE BASE")
    print("="*70, flush=True)

    for year in YEARS:
        m = run_and_get(f"v1_d_{year}", year, "nexus_alpha_v1", V1_LONG_PARAMS)
        v1_year_rets[year] = m.get("returns", [])
        print(f"  V1-Long {year}: Sharpe={m.get('sharpe', 'ERR')}", flush=True)

    for year in YEARS:
        m = run_and_get(f"vb_d_{year}", year, "vol_breakout_alpha", VB_CHAMPION_PARAMS)
        vb_year_rets[year] = m.get("returns", [])
        print(f"  VolBreakout {year}: Sharpe={m.get('sharpe', 'ERR')}", flush=True)

    # ====================================================================
    # 3. Ensemble sweep: V1-Long + Sr_336h at all weights
    # ====================================================================
    print("\n" + "="*70)
    print("  ENSEMBLE SWEEP: V1-Long + Sharpe Ratio 336h")
    print("="*70, flush=True)

    sr_champion_rets = sr_year_rets.get("sr_336h", {})
    if not sr_champion_rets:
        # fall back to re-running if not in cache
        sr_champion_rets = {}
        sr_params = {"k_per_side": 2, "lookback_bars": 336, "vol_lookback_bars": 168,
                     "target_gross_leverage": 0.35, "rebalance_interval_bars": 48}
        for year in YEARS:
            m = run_and_get(f"sr336_{year}", year, "sharpe_ratio_alpha", sr_params)
            if m.get("returns"):
                sr_champion_rets[year] = m["returns"]

    best_2way = {"label": "V1only", "avg": 0, "min": 0}
    print(f"\n  {'Weight V1/SR':<15} {'AVG':>7} {'MIN':>7} {'Pos':>5}  year-by-year")
    for w_v1 in [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]:
        w_sr = round(1.0 - w_v1, 2)
        label = f"{int(w_v1*100)}/{int(w_sr*100)}"
        year_sharpes = []
        mdds = []
        for year in YEARS:
            v1_r = v1_year_rets.get(year, [])
            sr_r = sr_champion_rets.get(year, [])
            if not v1_r or not sr_r:
                continue
            n = min(len(v1_r), len(sr_r))
            combined = [w_v1 * v1_r[i] + w_sr * sr_r[i] for i in range(n)]
            m = metrics_from_returns(combined)
            year_sharpes.append(m["sharpe"])
            mdds.append(m["max_dd"])
        if year_sharpes:
            avg = round(sum(year_sharpes) / len(year_sharpes), 3)
            mn = round(min(year_sharpes), 3)
            mdd_avg = round(sum(mdds) / len(mdds), 1)
            pos = sum(1 for s in year_sharpes if s > 0)
            sharpes_str = " ".join(f"{s:.2f}" for s in year_sharpes)
            print(f"  {label:<15} {avg:>7.3f} {mn:>7.3f} {pos:>5}  [{sharpes_str}] MDD_avg={mdd_avg}%")
            if avg > best_2way["avg"] or (avg >= best_2way["avg"] and mn > best_2way["min"]):
                best_2way = {"label": label, "avg": avg, "min": mn}

    print(f"\n  BEST 2-WAY: V1/{best_2way['label']} → AVG={best_2way['avg']}, MIN={best_2way['min']}")

    # ====================================================================
    # 4. 3-way ensemble: V1 + Sharpe Ratio + Vol Breakout
    # ====================================================================
    print("\n" + "="*70)
    print("  3-WAY ENSEMBLE: V1 + Sharpe Ratio + Vol Breakout")
    print("="*70, flush=True)

    best_3way = {"label": "", "avg": 0, "min": 0}
    weight_combos_3way = [
        (0.70, 0.20, 0.10), (0.65, 0.25, 0.10), (0.65, 0.20, 0.15),
        (0.60, 0.25, 0.15), (0.60, 0.30, 0.10), (0.55, 0.30, 0.15),
        (0.50, 0.35, 0.15), (0.55, 0.25, 0.20), (0.60, 0.20, 0.20),
        (0.70, 0.15, 0.15), (0.65, 0.15, 0.20),
    ]
    print(f"  {'V1/SR/VB':<18} {'AVG':>7} {'MIN':>7} {'Pos':>5}")
    for w_v1, w_sr, w_vb in weight_combos_3way:
        label = f"{int(w_v1*100)}/{int(w_sr*100)}/{int(w_vb*100)}"
        year_sharpes = []
        for year in YEARS:
            v1_r = v1_year_rets.get(year, [])
            sr_r = sr_champion_rets.get(year, [])
            vb_r = vb_year_rets.get(year, [])
            if not v1_r or not sr_r or not vb_r:
                continue
            n = min(len(v1_r), len(sr_r), len(vb_r))
            combined = [w_v1 * v1_r[i] + w_sr * sr_r[i] + w_vb * vb_r[i] for i in range(n)]
            m = metrics_from_returns(combined)
            year_sharpes.append(m["sharpe"])
        if year_sharpes:
            avg = round(sum(year_sharpes) / len(year_sharpes), 3)
            mn = round(min(year_sharpes), 3)
            pos = sum(1 for s in year_sharpes if s > 0)
            sharpes_str = " ".join(f"{s:.2f}" for s in year_sharpes)
            print(f"  {label:<18} {avg:>7.3f} {mn:>7.3f} {pos:>5}  [{sharpes_str}]")
            if avg > best_3way["avg"] or (avg >= best_3way["avg"] and mn > best_3way["min"]):
                best_3way = {"label": label, "avg": avg, "min": mn}

    print(f"\n  BEST 3-WAY: V1/SR/VB={best_3way['label']} → AVG={best_3way['avg']}, MIN={best_3way['min']}")

    # Save
    out_path = Path(OUT_DIR) / "phase60d_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    # Final champion recommendation
    print(f"\n{'='*80}")
    print(f"  PHASE 60D RECOMMENDATION")
    print(f"{'='*80}")
    print(f"  Champion 2-way ensemble: V1-Long / Sharpe Ratio = {best_2way['label']}")
    print(f"    AVG Sharpe: {best_2way['avg']}")
    print(f"    MIN Sharpe: {best_2way['min']}")
    print(f"  Champion 3-way ensemble: V1 / SR / VB = {best_3way['label']}")
    print(f"    AVG Sharpe: {best_3way['avg']}")
    print(f"    MIN Sharpe: {best_3way['min']}")


if __name__ == "__main__":
    main()
