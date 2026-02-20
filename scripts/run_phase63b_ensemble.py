#!/usr/bin/env python3
"""
Phase 63B: Proper ensemble analysis with new 437h optimal parameters.

Phase 63A found: optimal lookback = 437h for BOTH SR Alpha and Idio Momentum.
This phase runs proper return-level ensemble blending (same approach as Phase 61C).

Key improvement: idio_437 standalone AVG=1.200 vs idio_336=1.039 (+15.5%)
                 sr_437 standalone AVG=1.151 vs sr_336=0.846 (+36%)
"""

import json
import math
import statistics
import subprocess
import sys
import copy
from pathlib import Path

OUT_DIR = "artifacts/phase63b"
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
    "self_learn": {"enabled": False},
}

# Champion parameters from Phase 63
V1_PARAMS = {
    "k_per_side": 2,
    "w_carry": 0.35, "w_mom": 0.45, "w_confirm": 0.0,
    "w_mean_reversion": 0.20, "w_vol_momentum": 0.0, "w_funding_trend": 0.0,
    "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 72,
    "vol_lookback_bars": 168,
    "target_portfolio_vol": 0.0, "use_min_variance": False,
    "target_gross_leverage": 0.35, "min_gross_leverage": 0.05, "max_gross_leverage": 0.65,
    "rebalance_interval_bars": 60, "strict_agreement": False,
}

SR_437_PARAMS = {
    "k_per_side": 2,
    "lookback_bars": 437,
    "vol_lookback_bars": 168,
    "target_gross_leverage": 0.35,
    "rebalance_interval_bars": 48,
}

IDIO_437_PARAMS = {
    "k_per_side": 2,
    "lookback_bars": 437,
    "beta_window_bars": 72,
    "vol_lookback_bars": 168,
    "target_gross_leverage": 0.30,
    "rebalance_interval_bars": 48,
}

# Also test V1 with updated momentum_lookback_bars=437 (since V1 uses 336h for momentum)
V1_437_PARAMS = {**V1_PARAMS, "momentum_lookback_bars": 437}


def compute_metrics(result_path: str) -> dict:
    d = json.load(open(result_path))
    eq = d.get("equity_curve", [])
    rets = d.get("returns", [])
    if not eq or not rets or len(rets) < 100:
        return {"sharpe": 0, "error": "insufficient data"}
    mu = statistics.mean(rets)
    sd = statistics.pstdev(rets)
    sharpe = (mu / sd) * math.sqrt(8760) if sd > 0 else 0
    return {"sharpe": round(sharpe, 3), "returns": rets}


def make_config(run_name: str, year: str, strategy_name: str, params: dict) -> str:
    cfg = copy.deepcopy(BASE_CONFIG)
    start, end = YEAR_RANGES[year]
    cfg["run_name"] = run_name
    cfg["data"]["start"] = start
    cfg["data"]["end"] = end
    cfg["strategy"] = {"name": strategy_name, "params": params}
    path = f"/tmp/phase63b_{run_name}_{year}.json"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def run_strategy_all_years(label: str, strategy_name: str, params: dict) -> dict:
    """Run a strategy across all 5 years, return {year: {sharpe, returns}}."""
    print(f"\n{'='*60}")
    print(f"  Running: {label} ({strategy_name})", flush=True)
    print(f"{'='*60}")
    year_results = {}
    for year in YEARS:
        run_name = f"{label}_{year}"
        config_path = make_config(run_name, year, strategy_name, params)
        try:
            result = subprocess.run(
                [sys.executable, "-m", "nexus_quant", "run", "--config", config_path, "--out", OUT_DIR],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0:
                year_results[year] = {"error": "run failed", "sharpe": 0, "returns": []}
                print(f"    {year}: ERROR", flush=True)
                continue
        except subprocess.TimeoutExpired:
            year_results[year] = {"error": "timeout", "sharpe": 0, "returns": []}
            continue

        runs_dir = Path(OUT_DIR) / "runs"
        if not runs_dir.exists():
            year_results[year] = {"error": "no runs dir", "sharpe": 0, "returns": []}
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
        year_results[year] = {"error": "no result", "sharpe": 0, "returns": []}

    sharpes = [year_results[y].get("sharpe", 0) for y in YEARS if isinstance(year_results.get(y, {}).get("sharpe"), (int, float))]
    avg = round(sum(sharpes) / len(sharpes), 3) if sharpes else 0
    mn = round(min(sharpes), 3) if sharpes else 0
    pos = sum(1 for s in sharpes if s > 0)
    print(f"\n  -> AVG={avg}, MIN={mn}, {pos}/5 positive", flush=True)
    year_results["_avg"] = avg
    year_results["_min"] = mn
    year_results["_pos"] = pos
    return year_results


def metrics_from_returns(rets: list) -> dict:
    if not rets or len(rets) < 10:
        return {"sharpe": 0}
    mu = statistics.mean(rets)
    sd = statistics.pstdev(rets)
    sharpe = (mu / sd) * math.sqrt(8760) if sd > 0 else 0
    return {"sharpe": round(sharpe, 3)}


def compute_ensemble(strategies: dict, weights: dict) -> dict:
    """Blend return series by weight and compute ensemble metrics per year."""
    year_sharpes = []
    for year in YEARS:
        all_rets = None
        min_len = None
        for name, data in strategies.items():
            rets = data.get(year, {}).get("returns", [])
            if not rets:
                all_rets = None
                break
            if min_len is None or len(rets) < min_len:
                min_len = len(rets)
        if all_rets is None and min_len is None:
            year_sharpes.append(None)
            continue

        # Align all return series to min length and blend
        blended = []
        for i in range(min_len):
            r = 0.0
            for name, data in strategies.items():
                rets = data.get(year, {}).get("returns", [])
                w = weights.get(name, 0)
                if i < len(rets):
                    r += w * rets[i]
            blended.append(r)

        m = metrics_from_returns(blended)
        year_sharpes.append(m["sharpe"])

    valid = [s for s in year_sharpes if s is not None]
    avg = round(sum(valid) / len(valid), 3) if valid else 0
    mn = round(min(valid), 3) if valid else 0
    pos = sum(1 for s in valid if s > 0)
    return {"avg": avg, "min": mn, "pos": pos, "year_sharpes": year_sharpes}


def main():
    print("=" * 80)
    print("  PHASE 63B: PROPER ENSEMBLE WITH 437H OPTIMAL PARAMETERS")
    print("=" * 80)
    print("  Phase 63A: sr_437 AVG=1.151, idio_437 AVG=1.200 (both 5/5 positive)")
    print("  Now: proper return-level blending to find best ensembles")
    print("=" * 80, flush=True)

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # Run all strategies
    v1_data = run_strategy_all_years("v1_long", "nexus_alpha_v1", V1_PARAMS)
    sr_data = run_strategy_all_years("sr_437", "sharpe_ratio_alpha", SR_437_PARAMS)
    idio_data = run_strategy_all_years("idio_437", "idio_momentum_alpha", IDIO_437_PARAMS)
    v1_437_data = run_strategy_all_years("v1_437", "nexus_alpha_v1", V1_437_PARAMS)

    # ─── Correlation analysis ───
    print("\n" + "=" * 80)
    print("  CORRELATION ANALYSIS (V1, SR_437, Idio_437)")
    print("=" * 80)

    strategies = {
        "v1_long": v1_data,
        "sr_437": sr_data,
        "idio_437": idio_data,
        "v1_437": v1_437_data,
    }

    def get_all_returns(data):
        rets = []
        for year in YEARS:
            rets.extend(data.get(year, {}).get("returns", []))
        return rets

    def pearson_corr(x, y):
        n = min(len(x), len(y))
        if n < 10:
            return 0
        x, y = x[:n], y[:n]
        mx, my = sum(x) / n, sum(y) / n
        cov = sum((x[i] - mx) * (y[i] - my) for i in range(n)) / n
        vx = sum((xi - mx) ** 2 for xi in x) / n
        vy = sum((yi - my) ** 2 for yi in y) / n
        if vx <= 0 or vy <= 0:
            return 0
        return round(cov / math.sqrt(vx * vy), 4)

    all_rets = {k: get_all_returns(v) for k, v in strategies.items()}
    strat_names = list(strategies.keys())
    for i, n1 in enumerate(strat_names):
        for n2 in strat_names[i + 1:]:
            c = pearson_corr(all_rets[n1], all_rets[n2])
            print(f"  corr({n1:<15}, {n2:<15}) = {c}")

    # ─── 2-way ensembles ───
    print("\n" + "=" * 80)
    print("  2-WAY ENSEMBLES: V1 + Idio_437")
    print("=" * 80)
    print(f"  {'V1/Idio':>15}  {'AVG':>7}  {'MIN':>7}  {'Pos':>4}  year-by-year")

    best_2way = {"label": None, "avg": 0, "min": 0}
    for w1, w2 in [(1.0, 0.0), (0.90, 0.10), (0.85, 0.15), (0.80, 0.20),
                   (0.75, 0.25), (0.70, 0.30), (0.65, 0.35), (0.60, 0.40),
                   (0.55, 0.45), (0.50, 0.50)]:
        ens = compute_ensemble(
            {"v1": v1_data, "idio": idio_data},
            {"v1": w1, "idio": w2}
        )
        yby = [f"{s:.2f}" if s is not None else "?" for s in ens["year_sharpes"]]
        print(f"  {str(int(w1*100))+'/'+str(int(w2*100)):>15}  {ens['avg']:>7.3f}  {ens['min']:>7.3f}  {ens['pos']:>4}  [{' '.join(yby)}]")
        if ens["avg"] > best_2way["avg"] and ens["pos"] == 5:
            best_2way = {"label": f"{int(w1*100)}/{int(w2*100)}", "avg": ens["avg"], "min": ens["min"]}

    print(f"\n  Best V1+Idio_437: {best_2way['label']} → AVG={best_2way['avg']}, MIN={best_2way['min']}")

    # ─── 3-way: V1 + SR_437 + Idio_437 ───
    print("\n" + "=" * 80)
    print("  3-WAY: V1 + SR_437 + Idio_437")
    print("=" * 80)
    print(f"  {'V1/SR/Idio':>20}  {'AVG':>7}  {'MIN':>7}  {'Pos':>4}")

    three_way_configs = [
        (0.70, 0.15, 0.15), (0.65, 0.15, 0.20), (0.65, 0.20, 0.15),
        (0.60, 0.20, 0.20), (0.60, 0.15, 0.25), (0.60, 0.25, 0.15),
        (0.55, 0.20, 0.25), (0.55, 0.25, 0.20), (0.55, 0.15, 0.30),
        (0.50, 0.25, 0.25), (0.50, 0.20, 0.30), (0.50, 0.30, 0.20),
    ]

    best_3way = {"label": None, "avg": 0, "min": 0}
    for w1, w2, w3 in three_way_configs:
        ens = compute_ensemble(
            {"v1": v1_data, "sr": sr_data, "idio": idio_data},
            {"v1": w1, "sr": w2, "idio": w3}
        )
        label = f"{int(w1*100)}/{int(w2*100)}/{int(w3*100)}"
        print(f"  {label:>20}  {ens['avg']:>7.3f}  {ens['min']:>7.3f}  {ens['pos']:>4}")
        if ens["avg"] > best_3way["avg"] and ens["pos"] == 5:
            best_3way = {"label": label, "avg": ens["avg"], "min": ens["min"]}

    print(f"\n  Best 3-way V1+SR_437+Idio_437: {best_3way['label']} → AVG={best_3way['avg']}, MIN={best_3way['min']}")

    # ─── V1_437 (updated momentum lookback in V1 itself) ───
    print("\n" + "=" * 80)
    print("  V1_437 BASELINE (momentum_lookback updated to 437h)")
    print("=" * 80)
    v1_437_sharpes = [v1_437_data.get(y, {}).get("sharpe", 0) for y in YEARS]
    print(f"  V1_437: AVG={v1_437_data['_avg']}, MIN={v1_437_data['_min']}, pos={v1_437_data['_pos']}/5")
    print(f"  Year-by-year: {v1_437_sharpes}")

    # Ensemble V1_437 + Idio_437
    print("\n  V1_437 + Idio_437 ensemble:")
    for w1, w2 in [(0.70, 0.30), (0.65, 0.35), (0.60, 0.40), (0.55, 0.45), (0.50, 0.50)]:
        ens = compute_ensemble(
            {"v1": v1_437_data, "idio": idio_data},
            {"v1": w1, "idio": w2}
        )
        print(f"  V1_437={int(w1*100)}%/Idio_437={int(w2*100)}%: AVG={ens['avg']}, MIN={ens['min']}, pos={ens['pos']}/5")

    # ─── Final Summary ───
    print("\n" + "=" * 80)
    print("  PHASE 63B FINAL SUMMARY")
    print("=" * 80)
    print("\n  STANDALONE PERFORMANCE:")
    print(f"    v1_long:   AVG={v1_data['_avg']:.3f}  MIN={v1_data['_min']:.3f}  [{[v1_data.get(y,{}).get('sharpe','?') for y in YEARS]}]")
    print(f"    sr_437:    AVG={sr_data['_avg']:.3f}  MIN={sr_data['_min']:.3f}  [{[sr_data.get(y,{}).get('sharpe','?') for y in YEARS]}]")
    print(f"    idio_437:  AVG={idio_data['_avg']:.3f}  MIN={idio_data['_min']:.3f}  [{[idio_data.get(y,{}).get('sharpe','?') for y in YEARS]}]")
    print(f"    v1_437:    AVG={v1_437_data['_avg']:.3f}  MIN={v1_437_data['_min']:.3f}  [{[v1_437_data.get(y,{}).get('sharpe','?') for y in YEARS]}]")

    print(f"\n  BEST ENSEMBLES:")
    if best_2way["label"]:
        print(f"    V1+Idio_437 best 2-way: {best_2way['label']} → AVG={best_2way['avg']}, MIN={best_2way['min']}")
    if best_3way["label"]:
        print(f"    V1+SR_437+Idio_437 best 3-way: {best_3way['label']} → AVG={best_3way['avg']}, MIN={best_3way['min']}")

    print("\n  COMPARISON TO PHASE 61 CHAMPIONS:")
    print("    Phase 61 V1+IdioMom_336 65/35: AVG=1.261, MIN=1.107")
    print("    Phase 61 V1+IdioMom_336 50/50: AVG=1.277, MIN=1.059")

    # Save configs for best ensembles
    if best_2way["label"]:
        w1_str, w2_str = best_2way["label"].split("/")
        w1_val = float(w1_str) / 100
        w2_val = float(w2_str) / 100
        config_2way = {
            "_comment": f"Phase 63 Champion: V1({w1_str}%) + Idio_437({w2_str}%). AVG={best_2way['avg']}, MIN={best_2way['min']}, 5/5 positive. lookback=437h=18.2 days (discovered via sensitivity analysis).",
            "run_name": f"ensemble_v1_idio437_{w1_str}{w2_str}",
            "seed": 42,
            "venue": {"name": "binance_usdm", "kind": "crypto_perp", "vip_tier": 0},
            "data": {
                "provider": "binance_rest_v1", "symbols": SYMBOLS,
                "start": "2023-01-01", "end": "2024-01-01",
                "bar_interval": "1h", "cache_dir": ".cache/binance_rest"
            },
            "execution": {"style": "taker", "slippage_bps": 3.0},
            "_strategies": {
                "v1_long": {"weight": w1_val, "name": "nexus_alpha_v1", "params": V1_PARAMS},
                "idio_437": {"weight": w2_val, "name": "idio_momentum_alpha", "params": IDIO_437_PARAMS},
            },
            "costs": {"fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0},
            "benchmark": {"version": "v1", "walk_forward": {"enabled": True}},
            "risk": {"max_drawdown": 0.30, "max_turnover_per_rebalance": 0.8, "max_gross_leverage": 0.7, "max_position_per_symbol": 0.3},
            "self_learn": {"enabled": False},
        }
        cfg_path = f"configs/run_ensemble_v1_idio437_{w1_str}{w2_str}.json"
        with open(cfg_path, "w") as f:
            json.dump(config_2way, f, indent=2)
        print(f"\n  Saved champion config: {cfg_path}")

    # Save results
    out = {
        "standalone": {
            "v1_long": {"avg": v1_data["_avg"], "min": v1_data["_min"], "pos": v1_data["_pos"]},
            "sr_437": {"avg": sr_data["_avg"], "min": sr_data["_min"], "pos": sr_data["_pos"]},
            "idio_437": {"avg": idio_data["_avg"], "min": idio_data["_min"], "pos": idio_data["_pos"]},
            "v1_437": {"avg": v1_437_data["_avg"], "min": v1_437_data["_min"], "pos": v1_437_data["_pos"]},
        },
        "best_2way": best_2way,
        "best_3way": best_3way,
    }
    out_path = Path(OUT_DIR) / "phase63b_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
