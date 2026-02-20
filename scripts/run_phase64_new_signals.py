#!/usr/bin/env python3
"""
Phase 64: Three New Signal Directions

Motivated by Phase 62-63 findings and academic literature:

A. Pure Momentum Alpha — Raw cross-sectional momentum (lb=437h)
   Test: does beta-hedging in Idio Momentum actually help?

B. Skip-Gram Momentum Alpha — Momentum skipping recent 24h
   Test: does avoiding 1-day reversal improve performance?

C. Funding Momentum Alpha — Cumulative funding rate as cross-section signal
   Test: does funding rate create exploitable cross-sectional spread?

For each signal:
1. Run all 5 OOS years (2021-2025)
2. Compute correlation with existing champions (V1, SR_437, Idio_437)
3. Test ensemble combinations with current champions

Academic foundation:
- Pure momentum: Jegadeesh & Titman (1993); Liu et al. (2022) confirm for crypto 28-35d
- Skip-gram: 1-day reversal documented in crypto by Liu & Timmermann (2013)
- Funding: Bailey & López de Prado (2012); funding predicts contrarian reversal
"""

import json
import math
import statistics
import subprocess
import sys
import copy
from pathlib import Path

OUT_DIR = "artifacts/phase64"
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

# Known champion params (Phase 63)
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

IDIO_437_PARAMS = {
    "k_per_side": 2,
    "lookback_bars": 437,
    "beta_window_bars": 72,
    "vol_lookback_bars": 168,
    "target_gross_leverage": 0.30,
    "rebalance_interval_bars": 48,
}


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
    path = f"/tmp/phase64_{run_name}_{year}.json"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def run_strategy(label: str, strategy_name: str, params: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"  Running: {label}", flush=True)
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
                print(f"    {year}: ERROR - {result.stderr[-100:]}", flush=True)
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
    year_results["_avg"] = avg
    year_results["_min"] = mn
    year_results["_pos"] = pos
    print(f"  -> AVG={avg}, MIN={mn}, {pos}/5 positive", flush=True)
    return year_results


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


def get_all_returns(data: dict) -> list:
    rets = []
    for year in YEARS:
        rets.extend(data.get(year, {}).get("returns", []))
    return rets


def metrics_from_returns(rets: list) -> float:
    if not rets or len(rets) < 10:
        return 0.0
    mu = statistics.mean(rets)
    sd = statistics.pstdev(rets)
    return round((mu / sd) * math.sqrt(8760), 3) if sd > 0 else 0.0


def blend_ensemble(strategies: dict, weights: dict) -> dict:
    year_sharpes = []
    for year in YEARS:
        min_len = None
        valid = True
        for name, data in strategies.items():
            rets = data.get(year, {}).get("returns", [])
            if not rets:
                valid = False
                break
            if min_len is None or len(rets) < min_len:
                min_len = len(rets)
        if not valid or min_len is None:
            year_sharpes.append(None)
            continue
        blended = []
        for i in range(min_len):
            r = sum(weights.get(n, 0) * strategies[n].get(year, {}).get("returns", [])[i]
                    for n in strategies if i < len(strategies[n].get(year, {}).get("returns", [])))
            blended.append(r)
        year_sharpes.append(metrics_from_returns(blended))
    valid_s = [s for s in year_sharpes if s is not None]
    return {
        "avg": round(sum(valid_s) / len(valid_s), 3) if valid_s else 0,
        "min": round(min(valid_s), 3) if valid_s else 0,
        "pos": sum(1 for s in valid_s if s > 0),
        "yby": year_sharpes,
    }


def main():
    print("=" * 80)
    print("  PHASE 64: THREE NEW SIGNAL DIRECTIONS")
    print("=" * 80)
    print("  A: Pure Momentum (no beta-hedge) vs Idio Momentum")
    print("  B: Skip-Gram Momentum (skip 24h) vs Idio Momentum")
    print("  C: Funding Rate Cross-Section vs current champions")
    print("=" * 80, flush=True)

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    all_results = {}

    # ─── A: Pure Momentum Alpha ───
    print("\n" + "=" * 70)
    print("  A: PURE MOMENTUM ALPHA (lb=437h, no beta-hedge)")
    print("=" * 70, flush=True)
    print("  Hypothesis: if Idio > Pure, beta-hedging adds genuine value")
    print("  If Pure ≈ Idio, beta-hedging is noise")

    pure_variants = [
        ("pure_437", {"k_per_side": 2, "lookback_bars": 437, "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 48}),
        ("pure_336", {"k_per_side": 2, "lookback_bars": 336, "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 48}),
        ("pure_504", {"k_per_side": 2, "lookback_bars": 504, "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 48}),
    ]
    pure_results = {}
    for name, params in pure_variants:
        r = run_strategy(name, "pure_momentum_alpha", params)
        pure_results[name] = r
        all_results[name] = r
    best_pure = max(pure_results.items(), key=lambda x: x[1]["_avg"])
    print(f"\n  Best Pure Momentum: {best_pure[0]} AVG={best_pure[1]['_avg']}")
    print(f"  Reference Idio_437: AVG=1.200 (Phase 63)")

    # ─── B: Skip-Gram Momentum ───
    print("\n" + "=" * 70)
    print("  B: SKIP-GRAM MOMENTUM ALPHA (lb=437h, skip=24h)")
    print("=" * 70, flush=True)
    print("  Hypothesis: skipping 1-day reversal improves signal quality")

    skip_variants = [
        ("skip_24", {"k_per_side": 2, "lookback_bars": 437, "skip_bars": 24, "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 48}),
        ("skip_48", {"k_per_side": 2, "lookback_bars": 437, "skip_bars": 48, "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 48}),
        ("skip_72", {"k_per_side": 2, "lookback_bars": 437, "skip_bars": 72, "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 48}),
        ("skip_0",  {"k_per_side": 2, "lookback_bars": 437, "skip_bars": 0,  "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 48}),  # control
    ]
    skip_results = {}
    for name, params in skip_variants:
        r = run_strategy(name, "skip_gram_momentum_alpha", params)
        skip_results[name] = r
        all_results[name] = r
    best_skip = max(skip_results.items(), key=lambda x: x[1]["_avg"])
    print(f"\n  Best Skip-Gram: {best_skip[0]} AVG={best_skip[1]['_avg']}")
    print(f"  Skip-0 (control) = pure momentum: AVG={skip_results.get('skip_0', {}).get('_avg', '?')}")

    # ─── C: Funding Momentum Alpha ───
    print("\n" + "=" * 70)
    print("  C: FUNDING RATE CROSS-SECTION (contrarian direction)")
    print("=" * 70, flush=True)
    print("  Hypothesis: high funding = crowded longs = future underperformance")

    fund_variants = [
        ("fund_cont_48",  {"k_per_side": 2, "funding_lookback_bars": 48,  "direction": "contrarian", "vol_lookback_bars": 168, "target_gross_leverage": 0.25, "rebalance_interval_bars": 24}),
        ("fund_cont_168", {"k_per_side": 2, "funding_lookback_bars": 168, "direction": "contrarian", "vol_lookback_bars": 168, "target_gross_leverage": 0.25, "rebalance_interval_bars": 24}),
        ("fund_cont_336", {"k_per_side": 2, "funding_lookback_bars": 336, "direction": "contrarian", "vol_lookback_bars": 168, "target_gross_leverage": 0.25, "rebalance_interval_bars": 48}),
        ("fund_mom_168",  {"k_per_side": 2, "funding_lookback_bars": 168, "direction": "momentum",   "vol_lookback_bars": 168, "target_gross_leverage": 0.25, "rebalance_interval_bars": 24}),
    ]
    fund_results = {}
    for name, params in fund_variants:
        r = run_strategy(name, "funding_momentum_alpha", params)
        fund_results[name] = r
        all_results[name] = r
    best_fund = max(fund_results.items(), key=lambda x: x[1]["_avg"])
    print(f"\n  Best Funding Signal: {best_fund[0]} AVG={best_fund[1]['_avg']}")

    # ─── Correlation + Ensemble Analysis ───
    print("\n" + "=" * 80)
    print("  CORRELATION ANALYSIS")
    print("=" * 80, flush=True)

    # Run V1 and Idio_437 fresh for correlation (use cached data)
    v1_data = run_strategy("v1_base", "nexus_alpha_v1", V1_PARAMS)
    idio_data = run_strategy("idio_437_base", "idio_momentum_alpha", IDIO_437_PARAMS)
    all_results["v1"] = v1_data
    all_results["idio_437"] = idio_data

    all_rets = {
        "v1": get_all_returns(v1_data),
        "idio_437": get_all_returns(idio_data),
        "pure_437": get_all_returns(pure_results.get("pure_437", {})),
        "skip_24": get_all_returns(skip_results.get("skip_24", {})),
        "fund_cont_168": get_all_returns(fund_results.get("fund_cont_168", {})),
    }

    strat_names = list(all_rets.keys())
    for i, n1 in enumerate(strat_names):
        for n2 in strat_names[i + 1:]:
            if all_rets[n1] and all_rets[n2]:
                c = pearson_corr(all_rets[n1], all_rets[n2])
                print(f"  corr({n1:<20}, {n2:<20}) = {c}")

    # ─── Ensemble with best new signal ───
    print("\n" + "=" * 80)
    print("  ENSEMBLE: V1 + Idio_437 + BEST NEW SIGNAL")
    print("=" * 80, flush=True)

    best_new_name = max(
        [("pure_437", pure_results.get("pure_437", {"_avg": 0})),
         ("skip_24", skip_results.get("skip_24", {"_avg": 0})),
         ("fund_cont_168", fund_results.get("fund_cont_168", {"_avg": 0}))],
        key=lambda x: x[1]["_avg"]
    )
    print(f"\n  Best new signal: {best_new_name[0]} AVG={best_new_name[1]['_avg']}")

    best_new_data = all_results.get(best_new_name[0], {})
    for w1, w2, w3 in [(0.60, 0.30, 0.10), (0.55, 0.30, 0.15), (0.50, 0.30, 0.20),
                        (0.60, 0.25, 0.15), (0.55, 0.25, 0.20)]:
        ens = blend_ensemble(
            {"v1": v1_data, "idio": idio_data, "new": best_new_data},
            {"v1": w1, "idio": w2, "new": w3}
        )
        print(f"  V1={int(w1*100)}/Idio_437={int(w2*100)}/{best_new_name[0]}={int(w3*100)}: AVG={ens['avg']}, MIN={ens['min']}, pos={ens['pos']}/5")

    # ─── Final Summary ───
    print("\n" + "=" * 80)
    print("  PHASE 64 FINAL SUMMARY")
    print("=" * 80)
    print("\n  STANDALONE PERFORMANCE:")
    print(f"    REFERENCE Idio_437:  AVG=1.200, MIN=0.505 (Phase 63)")
    print(f"    REFERENCE V1-Long:   AVG=1.125, MIN=0.948")
    for name, r in [("pure_437", pure_results.get("pure_437", {})),
                     ("skip_24", skip_results.get("skip_24", {})),
                     ("fund_cont_168", fund_results.get("fund_cont_168", {}))]:
        yby = [r.get(y, {}).get("sharpe", "?") for y in YEARS]
        print(f"    {name:<25}: AVG={r.get('_avg', '?')}, MIN={r.get('_min', '?')}, pos={r.get('_pos', '?')}/5 | {yby}")

    print("\n  KEY QUESTIONS ANSWERED:")
    pure_avg = pure_results.get("pure_437", {}).get("_avg", 0)
    idio_ref = 1.200
    if pure_avg > idio_ref * 0.95:
        print(f"  Q: Does beta-hedging help? A: MARGINAL — pure={pure_avg:.3f} ≈ idio={idio_ref:.3f}")
    elif pure_avg > idio_ref * 0.80:
        print(f"  Q: Does beta-hedging help? A: YES — idio={idio_ref:.3f} > pure={pure_avg:.3f} (+{(idio_ref-pure_avg)/pure_avg*100:.1f}%)")
    else:
        print(f"  Q: Does beta-hedging help? A: SIGNIFICANTLY — idio={idio_ref:.3f} >> pure={pure_avg:.3f}")

    skip_best_avg = best_skip[1]["_avg"]
    pure_skip0 = skip_results.get("skip_0", {}).get("_avg", pure_avg)
    if skip_best_avg > pure_skip0 * 1.05:
        print(f"  Q: Does reversal skip help? A: YES — skip_best={skip_best_avg:.3f} > no_skip={pure_skip0:.3f}")
    else:
        print(f"  Q: Does reversal skip help? A: NO — skip_best={skip_best_avg:.3f} ≈ no_skip={pure_skip0:.3f}")

    # Save
    def strip_returns(d):
        if isinstance(d, dict):
            return {k: strip_returns(v) for k, v in d.items() if k != "returns"}
        return d
    out_path = Path(OUT_DIR) / "phase64_results.json"
    with open(out_path, "w") as f:
        json.dump(strip_returns(all_results), f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
