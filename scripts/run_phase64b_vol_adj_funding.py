#!/usr/bin/env python3
"""
Phase 64B: Vol-Adjusted Momentum + Funding Signal (Fixed)

Builds on Phase 64 findings:
  - Pure Momentum (pure_437):  AVG=1.032, 5/5 positive (confirmed baseline)
  - Skip-Gram (skip_24):       AVG=0.869 (skip DOES NOT help — short-term reversal
                                           effect negligible in concentrated universe)
  - Funding Momentum:          FAILED — KeyError: integer bar indexing vs timestamp keys

This script:
  A. Vol-Adjusted Momentum Alpha — score = total_return / realized_vol (endpoint IR)
     • Different from SR Alpha (path Sharpe) vs this (endpoint information ratio)
     • Hypothesis: vol-normalization at return-level adds different info than per-bar
     • Tests: lb=437h (known optimum), lb=336h, lb=504h

  B. Funding Momentum Alpha (FIXED) — cumulative funding via timeline-aligned access
     • Bug fixed: no longer uses integer bar indices; uses dataset.last_funding_rate_before()
     • Re-tests contrarian and momentum directions
     • Tests lookback: 48h, 168h, 336h

  C. 4-Way Ensemble — V1 + Idio_437 + Vol_Adj + best new signal
     • Phase 63 champion: V1(50%) + SR_437(20%) + Idio_437(30%), AVG=1.380, MIN=1.063
     • Can we exceed 1.380 AVG with vol_adjusted or funding?

Academic context for Vol-Adjusted Momentum:
  - IR = total_return / realized_vol is equivalent to Jensen's alpha concept
  - Differs from Sharpe in that it measures ENDPOINT efficiency not PATH consistency
  - Ghysels et al. (2005): volatility-scaled signals improve timing in equity markets
  - In crypto: concentrated universe = faster mean reversion of vol ranking vs equity
"""

import json
import math
import statistics
import subprocess
import sys
import copy
from pathlib import Path

OUT_DIR = "artifacts/phase64b"
YEARS = ["2021", "2022", "2023", "2024", "2025"]
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
           "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT"]

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
    "data": {
        "provider": "binance_rest_v1",
        "symbols": SYMBOLS,
        "bar_interval": "1h",
        "cache_dir": ".cache/binance_rest",
    },
    "execution": {"style": "taker", "slippage_bps": 3.0},
    "costs": {"fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0},
    "benchmark": {"version": "v1", "walk_forward": {"enabled": True}},
    "risk": {
        "max_drawdown": 0.30,
        "max_turnover_per_rebalance": 0.8,
        "max_gross_leverage": 0.7,
        "max_position_per_symbol": 0.3,
    },
    "self_learn": {"enabled": False},
}

# Phase 63 champion params (frozen reference)
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

SR_437_PARAMS = {
    "k_per_side": 2,
    "lookback_bars": 437,
    "vol_lookback_bars": 168,
    "target_gross_leverage": 0.35,
    "rebalance_interval_bars": 48,
}

# Phase 64 known results (from prior run, no need to re-run)
PHASE64_KNOWN = {
    "pure_437": {"_avg": 1.032, "_min": 0.365, "_pos": 5},
    "skip_24":  {"_avg": 0.869, "_min": -0.098, "_pos": 4},
}


def compute_metrics(result_path: str) -> dict:
    d = json.load(open(result_path))
    rets = d.get("returns", [])
    if not rets or len(rets) < 100:
        return {"sharpe": 0.0, "error": "insufficient data", "returns": []}
    mu = statistics.mean(rets)
    sd = statistics.pstdev(rets)
    sharpe = (mu / sd) * math.sqrt(8760) if sd > 0 else 0.0
    return {"sharpe": round(sharpe, 3), "returns": rets}


def make_config(run_name: str, year: str, strategy_name: str, params: dict) -> str:
    cfg = copy.deepcopy(BASE_CONFIG)
    start, end = YEAR_RANGES[year]
    cfg["run_name"] = run_name
    cfg["data"]["start"] = start
    cfg["data"]["end"] = end
    cfg["strategy"] = {"name": strategy_name, "params": params}
    path = f"/tmp/phase64b_{run_name}_{year}.json"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def run_strategy(label: str, strategy_name: str, params: dict) -> dict:
    print(f"\n{'─'*60}")
    print(f"  Running: {label}", flush=True)
    year_results = {}
    for year in YEARS:
        run_name = f"{label}_{year}"
        config_path = make_config(run_name, year, strategy_name, params)
        try:
            result = subprocess.run(
                [sys.executable, "-m", "nexus_quant", "run",
                 "--config", config_path, "--out", OUT_DIR],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0:
                year_results[year] = {"error": "run failed", "sharpe": 0.0, "returns": []}
                print(f"    {year}: ERROR — {result.stderr[-200:]}", flush=True)
                continue
        except subprocess.TimeoutExpired:
            year_results[year] = {"error": "timeout", "sharpe": 0.0, "returns": []}
            print(f"    {year}: TIMEOUT", flush=True)
            continue

        runs_dir = Path(OUT_DIR) / "runs"
        if not runs_dir.exists():
            year_results[year] = {"error": "no runs dir", "sharpe": 0.0, "returns": []}
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
        year_results[year] = {"error": "no result", "sharpe": 0.0, "returns": []}
        print(f"    {year}: no result file", flush=True)

    sharpes = [
        year_results[y].get("sharpe", 0.0)
        for y in YEARS
        if isinstance(year_results.get(y, {}).get("sharpe"), (int, float))
    ]
    avg = round(sum(sharpes) / len(sharpes), 3) if sharpes else 0.0
    mn = round(min(sharpes), 3) if sharpes else 0.0
    pos = sum(1 for s in sharpes if s > 0)
    year_results["_avg"] = avg
    year_results["_min"] = mn
    year_results["_pos"] = pos
    print(f"  → AVG={avg}, MIN={mn}, {pos}/5 positive", flush=True)
    return year_results


def pearson_corr(x, y):
    n = min(len(x), len(y))
    if n < 50:
        return 0.0
    x, y = x[:n], y[:n]
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((x[i] - mx) * (y[i] - my) for i in range(n)) / n
    vx = sum((xi - mx) ** 2 for xi in x) / n
    vy = sum((yi - my) ** 2 for yi in y) / n
    if vx <= 0 or vy <= 0:
        return 0.0
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
    year_sharpes = {}
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
            year_sharpes[year] = None
            continue
        blended = [
            sum(
                weights.get(n, 0) * strategies[n].get(year, {}).get("returns", [])[i]
                for n in strategies
                if i < len(strategies[n].get(year, {}).get("returns", []))
            )
            for i in range(min_len)
        ]
        year_sharpes[year] = metrics_from_returns(blended)

    valid_s = [s for s in year_sharpes.values() if s is not None]
    return {
        "avg": round(sum(valid_s) / len(valid_s), 3) if valid_s else 0.0,
        "min": round(min(valid_s), 3) if valid_s else 0.0,
        "pos": sum(1 for s in valid_s if s > 0),
        "yby": [year_sharpes.get(y) for y in YEARS],
    }


def save_champion_config(run_name: str, comment: str, strategy_weights: dict) -> None:
    """Save a 3-way ensemble config file for champion configurations."""
    cfg = {
        "_comment": comment,
        "run_name": run_name,
        "seed": 42,
        "venue": {"name": "binance_usdm", "kind": "crypto_perp", "vip_tier": 0},
        "data": {
            "provider": "binance_rest_v1",
            "symbols": SYMBOLS,
            "start": "2023-01-01", "end": "2024-01-01",
            "bar_interval": "1h", "cache_dir": ".cache/binance_rest",
        },
        "execution": {"style": "taker", "slippage_bps": 3.0},
        "_strategies": strategy_weights,
        "costs": {"fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0},
        "benchmark": {"version": "v1", "walk_forward": {"enabled": True}},
        "risk": {
            "max_drawdown": 0.30, "max_turnover_per_rebalance": 0.8,
            "max_gross_leverage": 0.7, "max_position_per_symbol": 0.3,
        },
        "self_learn": {"enabled": False},
    }
    out_path = Path("configs") / f"{run_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  Config saved: {out_path}")


def main():
    print("=" * 80)
    print("  PHASE 64B: VOL-ADJUSTED MOMENTUM + FUNDING SIGNAL (FIXED)")
    print("=" * 80)
    print("  Phase 63 champion: V1(50%)+SR_437(20%)+Idio_437(30%): AVG=1.380, MIN=1.063")
    print("  Phase 64 context: pure_437=1.032, skip_24=0.869, funding=FAILED (fixed now)")
    print("=" * 80, flush=True)

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    all_results = {}

    # ─── A: Vol-Adjusted Momentum Alpha ───────────────────────────────────────
    print("\n" + "═" * 70)
    print("  A: VOL-ADJUSTED MOMENTUM ALPHA")
    print("  Score = total_return(lb) / realized_vol(lb)")
    print("  Hypothesis: endpoint IR > path Sharpe for cross-sectional ranking?")
    print("═" * 70, flush=True)

    vol_adj_variants = [
        ("vol_adj_437", {
            "k_per_side": 2, "lookback_bars": 437, "signal_vol_bars": 437,
            "vol_lookback_bars": 168, "target_gross_leverage": 0.30,
            "rebalance_interval_bars": 48,
        }),
        ("vol_adj_336", {
            "k_per_side": 2, "lookback_bars": 336, "signal_vol_bars": 336,
            "vol_lookback_bars": 168, "target_gross_leverage": 0.30,
            "rebalance_interval_bars": 48,
        }),
        ("vol_adj_504", {
            "k_per_side": 2, "lookback_bars": 504, "signal_vol_bars": 504,
            "vol_lookback_bars": 168, "target_gross_leverage": 0.30,
            "rebalance_interval_bars": 48,
        }),
        # Asymmetric: long return window, short vol window (captures momentum with current vol)
        ("vol_adj_437_v168", {
            "k_per_side": 2, "lookback_bars": 437, "signal_vol_bars": 168,
            "vol_lookback_bars": 168, "target_gross_leverage": 0.30,
            "rebalance_interval_bars": 48,
        }),
    ]

    vol_adj_results = {}
    for name, params in vol_adj_variants:
        r = run_strategy(name, "vol_adjusted_momentum_alpha", params)
        vol_adj_results[name] = r
        all_results[name] = r

    best_vol_adj = max(vol_adj_results.items(), key=lambda x: x[1]["_avg"])
    print(f"\n  Best Vol-Adjusted: {best_vol_adj[0]} AVG={best_vol_adj[1]['_avg']}")
    print(f"  Reference Pure Momentum (437): AVG=1.032")
    print(f"  Reference SR Alpha (437):      AVG=1.151 (Phase 63)")
    print(f"  Reference Idio Momentum (437): AVG=1.200 (Phase 63)")

    vol_adj_best_avg = best_vol_adj[1]["_avg"]
    sr_ref = 1.151
    pure_ref = 1.032
    print(f"\n  DIAGNOSIS:")
    if vol_adj_best_avg > sr_ref:
        print(f"  → Vol-Adj ({vol_adj_best_avg:.3f}) > SR Alpha ({sr_ref:.3f}): endpoint IR beats path Sharpe!")
    elif vol_adj_best_avg > pure_ref * 1.05:
        print(f"  → Vol-Adj ({vol_adj_best_avg:.3f}) in [Pure, SR]: vol normalization helps but path Sharpe is better")
    else:
        print(f"  → Vol-Adj ({vol_adj_best_avg:.3f}) ≈ Pure ({pure_ref:.3f}): vol-normalization at return level doesn't help")

    # ─── B: Funding Momentum Alpha (FIXED) ────────────────────────────────────
    print("\n" + "═" * 70)
    print("  B: FUNDING MOMENTUM ALPHA (BUG FIXED)")
    print("  Previous failure: KeyError from integer indexing on timestamp-keyed dict")
    print("  Fix: uses dataset.last_funding_rate_before(symbol, timeline[idx]) via timeline")
    print("═" * 70, flush=True)

    fund_variants = [
        # Contrarian variants (high funding → crowded longs → expect underperformance)
        ("fund_cont_48",  {
            "k_per_side": 2, "funding_lookback_bars": 48,
            "direction": "contrarian", "vol_lookback_bars": 168,
            "target_gross_leverage": 0.25, "rebalance_interval_bars": 24,
        }),
        ("fund_cont_168", {
            "k_per_side": 2, "funding_lookback_bars": 168,
            "direction": "contrarian", "vol_lookback_bars": 168,
            "target_gross_leverage": 0.25, "rebalance_interval_bars": 24,
        }),
        ("fund_cont_336", {
            "k_per_side": 2, "funding_lookback_bars": 336,
            "direction": "contrarian", "vol_lookback_bars": 168,
            "target_gross_leverage": 0.25, "rebalance_interval_bars": 48,
        }),
        # Momentum variant (high funding → strong demand → continues)
        ("fund_mom_168",  {
            "k_per_side": 2, "funding_lookback_bars": 168,
            "direction": "momentum", "vol_lookback_bars": 168,
            "target_gross_leverage": 0.25, "rebalance_interval_bars": 24,
        }),
    ]

    fund_results = {}
    for name, params in fund_variants:
        r = run_strategy(name, "funding_momentum_alpha", params)
        fund_results[name] = r
        all_results[name] = r

    best_fund = max(fund_results.items(), key=lambda x: x[1]["_avg"])
    print(f"\n  Best Funding Signal: {best_fund[0]} AVG={best_fund[1]['_avg']}")
    all_fund_avgs = {n: r["_avg"] for n, r in fund_results.items()}
    print(f"  All funding results: {all_fund_avgs}")

    # Academic prediction: contrarian should win (Liu et al. 2021)
    cont_best = max(
        [(n, r) for n, r in fund_results.items() if "cont" in n],
        key=lambda x: x[1]["_avg"],
        default=(None, {"_avg": 0}),
    )
    mom_best = max(
        [(n, r) for n, r in fund_results.items() if "mom" in n],
        key=lambda x: x[1]["_avg"],
        default=(None, {"_avg": 0}),
    )
    print(f"\n  Contrarian best: {cont_best[0]} AVG={cont_best[1]['_avg']}")
    print(f"  Momentum best:   {mom_best[0]} AVG={mom_best[1]['_avg']}")
    if cont_best[1]["_avg"] > mom_best[1]["_avg"]:
        print(f"  → Academic prediction CONFIRMED: contrarian wins")
    else:
        print(f"  → Academic prediction REJECTED: momentum direction stronger")

    # ─── Run Reference Strategies ──────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  Running reference strategies for correlation analysis...")
    print("═" * 70, flush=True)
    v1_data = run_strategy("v1_ref", "nexus_alpha_v1", V1_PARAMS)
    idio_data = run_strategy("idio_437_ref", "idio_momentum_alpha", IDIO_437_PARAMS)
    sr_data = run_strategy("sr_437_ref", "sharpe_ratio_alpha", SR_437_PARAMS)
    all_results["v1"] = v1_data
    all_results["idio_437"] = idio_data
    all_results["sr_437"] = sr_data

    # ─── Correlation Analysis ──────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("  CORRELATION ANALYSIS (full 5-year concatenated returns)")
    print("═" * 80, flush=True)

    best_vol_adj_data = vol_adj_results[best_vol_adj[0]]
    best_fund_data = fund_results[best_fund[0]]

    all_rets = {
        "v1":           get_all_returns(v1_data),
        "idio_437":     get_all_returns(idio_data),
        "sr_437":       get_all_returns(sr_data),
        "vol_adj_437":  get_all_returns(vol_adj_results.get("vol_adj_437", {})),
        "best_fund":    get_all_returns(best_fund_data),
    }

    strat_names = list(all_rets.keys())
    print("\n  Pairwise correlations:")
    for i, n1 in enumerate(strat_names):
        for n2 in strat_names[i + 1:]:
            if all_rets[n1] and all_rets[n2]:
                c = pearson_corr(all_rets[n1], all_rets[n2])
                print(f"    corr({n1:<22}, {n2:<22}) = {c:+.4f}")

    # ─── Ensemble Experiments ──────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("  ENSEMBLE EXPERIMENTS")
    print("  Phase 63 champion baseline: V1(50%)+SR_437(20%)+Idio_437(30%) AVG=1.380")
    print("═" * 80, flush=True)

    # Current champion reproduced
    champ_3way = blend_ensemble(
        {"v1": v1_data, "sr": sr_data, "idio": idio_data},
        {"v1": 0.50, "sr": 0.20, "idio": 0.30},
    )
    print(f"\n  [BASELINE] V1(50%)+SR_437(20%)+Idio_437(30%): AVG={champ_3way['avg']}, MIN={champ_3way['min']}, pos={champ_3way['pos']}/5")
    print(f"    Year-by-year: {champ_3way['yby']}")

    # Test vol_adj as replacement/addition for SR_437
    print("\n  --- Adding Vol-Adj Momentum ---")
    for w_v1, w_idio, w_sr, w_va in [
        (0.50, 0.25, 0.10, 0.15),
        (0.50, 0.20, 0.10, 0.20),
        (0.45, 0.30, 0.10, 0.15),
        (0.50, 0.30, 0.00, 0.20),  # replace SR with vol_adj
        (0.55, 0.30, 0.00, 0.15),  # V1+Idio+VolAdj (no SR)
        (0.45, 0.25, 0.15, 0.15),  # full 4-way
    ]:
        ens = blend_ensemble(
            {"v1": v1_data, "idio": idio_data, "sr": sr_data, "va": best_vol_adj_data},
            {"v1": w_v1, "idio": w_idio, "sr": w_sr, "va": w_va},
        )
        tag = f"V1={int(w_v1*100)}/Idio={int(w_idio*100)}/SR={int(w_sr*100)}/VA={int(w_va*100)}"
        print(f"    {tag}: AVG={ens['avg']}, MIN={ens['min']}, pos={ens['pos']}/5")

    # Test funding if meaningful (AVG > 0.5)
    if best_fund["_avg"] > 0.5:
        print(f"\n  --- Adding Funding ({best_fund[0]}, AVG={best_fund['_avg']}) ---")
        for w_v1, w_idio, w_sr, w_f in [
            (0.50, 0.25, 0.15, 0.10),
            (0.55, 0.25, 0.10, 0.10),
            (0.50, 0.20, 0.20, 0.10),
        ]:
            ens = blend_ensemble(
                {"v1": v1_data, "idio": idio_data, "sr": sr_data, "fund": best_fund_data},
                {"v1": w_v1, "idio": w_idio, "sr": w_sr, "fund": w_f},
            )
            tag = f"V1={int(w_v1*100)}/Idio={int(w_idio*100)}/SR={int(w_sr*100)}/Fund={int(w_f*100)}"
            print(f"    {tag}: AVG={ens['avg']}, MIN={ens['min']}, pos={ens['pos']}/5")

    # Full 4-way: V1+Idio+SR+VolAdj (most diversified)
    print("\n  --- 4-Way Ensemble: V1 + Idio_437 + SR_437 + Vol_Adj ---")
    for w_v1, w_idio, w_sr, w_va in [
        (0.45, 0.25, 0.15, 0.15),
        (0.50, 0.20, 0.15, 0.15),
        (0.45, 0.30, 0.15, 0.10),
        (0.40, 0.30, 0.15, 0.15),
        (0.40, 0.25, 0.20, 0.15),
    ]:
        ens = blend_ensemble(
            {"v1": v1_data, "idio": idio_data, "sr": sr_data, "va": best_vol_adj_data},
            {"v1": w_v1, "idio": w_idio, "sr": w_sr, "va": w_va},
        )
        tag = f"V1={int(w_v1*100)}/Idio={int(w_idio*100)}/SR={int(w_sr*100)}/VA={int(w_va*100)}"
        champ_flag = " *** NEW CHAMPION ***" if ens["avg"] > 1.380 else ""
        print(f"    {tag}: AVG={ens['avg']}, MIN={ens['min']}, pos={ens['pos']}/5{champ_flag}")

    # ─── Best New Champion Auto-Save ──────────────────────────────────────────
    print("\n" + "═" * 80)
    print("  CHAMPION CANDIDATE SEARCH")
    print("═" * 80, flush=True)

    candidates = []

    # Systematically search 4-way weights
    for w_v1 in [0.40, 0.45, 0.50, 0.55]:
        for w_idio in [0.20, 0.25, 0.30]:
            for w_sr in [0.10, 0.15, 0.20]:
                w_va = round(1.0 - w_v1 - w_idio - w_sr, 2)
                if 0.05 <= w_va <= 0.25:
                    ens = blend_ensemble(
                        {"v1": v1_data, "idio": idio_data, "sr": sr_data, "va": best_vol_adj_data},
                        {"v1": w_v1, "idio": w_idio, "sr": w_sr, "va": w_va},
                    )
                    candidates.append((w_v1, w_idio, w_sr, w_va, ens["avg"], ens["min"], ens["pos"]))

    candidates.sort(key=lambda x: x[4], reverse=True)

    print("\n  Top 5 candidates (by AVG Sharpe):")
    for i, (w_v1, w_idio, w_sr, w_va, avg, mn, pos) in enumerate(candidates[:5]):
        champ_flag = " ← NEW CHAMPION" if avg > 1.380 else ""
        print(f"  [{i+1}] V1={int(w_v1*100)}/Idio={int(w_idio*100)}/SR={int(w_sr*100)}/VA={int(w_va*100)}: "
              f"AVG={avg}, MIN={mn}, pos={pos}/5{champ_flag}")

    if candidates and candidates[0][4] > 1.380:
        w_v1, w_idio, w_sr, w_va, avg, mn, pos = candidates[0]
        run_name = f"ensemble_v1_idio437_sr437_va_{int(w_v1*100)}{int(w_idio*100)}{int(w_sr*100)}{int(w_va*100)}"
        comment = (
            f"Phase 64B Champion (4-Way): V1({int(w_v1*100)}%) + Idio_437({int(w_idio*100)}%) + "
            f"SR_437({int(w_sr*100)}%) + VolAdj({int(w_va*100)}%). "
            f"AVG={avg}, MIN={mn}, pos={pos}/5. "
            f"Beats Phase 63 champion (AVG=1.380). "
            f"vol_adjusted_momentum uses lb=437h signal_vol=437h."
        )
        strategy_weights = {
            "v1_long": {
                "weight": w_v1, "name": "nexus_alpha_v1", "params": V1_PARAMS,
            },
            "idio_437": {
                "weight": w_idio, "name": "idio_momentum_alpha", "params": IDIO_437_PARAMS,
            },
            "sr_437": {
                "weight": w_sr, "name": "sharpe_ratio_alpha", "params": SR_437_PARAMS,
            },
            "vol_adj_437": {
                "weight": w_va, "name": "vol_adjusted_momentum_alpha",
                "params": vol_adj_variants[0][1],  # vol_adj_437 params
            },
        }
        save_champion_config(run_name, comment, strategy_weights)
    else:
        print("\n  No 4-way ensemble beats Phase 63 champion (AVG=1.380)")
        print("  → Vol-adjusted momentum may be too correlated with SR_437 to add diversification")

    # ─── Final Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  PHASE 64B FINAL SUMMARY")
    print("=" * 80)

    print("\n  STANDALONE PERFORMANCE (all signals tested):")
    print(f"    CHAMPION  Idio_437:         AVG=1.200 (Phase 63)")
    print(f"    CHAMPION  SR_437:           AVG=1.151 (Phase 63)")
    print(f"    baseline  Pure_437:         AVG=1.032 (Phase 64)")

    vol_adj_437_r = vol_adj_results.get("vol_adj_437", {})
    print(f"\n    NEW: vol_adj_437:          AVG={vol_adj_437_r.get('_avg', '?')}, MIN={vol_adj_437_r.get('_min', '?')}, pos={vol_adj_437_r.get('_pos', '?')}/5")
    for n, r in vol_adj_results.items():
        yby = [r.get(y, {}).get("sharpe", "?") for y in YEARS]
        print(f"    {n:<30} AVG={r['_avg']}, MIN={r['_min']} | {yby}")

    print(f"\n    Funding Signal Results (fixed):")
    for n, r in fund_results.items():
        yby = [r.get(y, {}).get("sharpe", "?") for y in YEARS]
        print(f"    {n:<30} AVG={r['_avg']}, MIN={r['_min']} | {yby}")

    print("\n  KEY INSIGHTS:")
    # Vol-adj vs SR comparison
    va437_avg = vol_adj_results.get("vol_adj_437", {}).get("_avg", 0)
    print(f"  1. Vol-Adj (endpoint IR) vs SR Alpha (path Sharpe):")
    print(f"     vol_adj_437={va437_avg:.3f} vs sr_437=1.151 → ", end="")
    if va437_avg > 1.200:
        print("endpoint IR BEST signal found so far!")
    elif va437_avg > 1.151:
        print("endpoint IR slightly better than path Sharpe")
    elif va437_avg > 1.032:
        print("vol-normalization helps but path Sharpe is better")
    else:
        print("vol-normalization at return level doesn't help vs pure momentum")

    print(f"  2. Funding signal: {'WORKS' if best_fund['_avg'] > 0.3 else 'WEAK/FAILS'} (best={best_fund[0]}, AVG={best_fund['_avg']:.3f})")
    print(f"     Academic prediction (contrarian): {'CONFIRMED' if cont_best[1]['_avg'] > mom_best[1]['_avg'] else 'REJECTED'}")

    print(f"  3. Best ensemble AVG: {candidates[0][4] if candidates else 'N/A'}")
    if candidates and candidates[0][4] > 1.380:
        print(f"     → NEW OVERALL CHAMPION: {candidates[0]}")
    else:
        print(f"     → Phase 63 champion remains best: AVG=1.380")

    # Save results
    def strip_returns(d):
        if isinstance(d, dict):
            return {k: strip_returns(v) for k, v in d.items() if k != "returns"}
        return d

    out_path = Path(OUT_DIR) / "phase64b_results.json"
    with open(out_path, "w") as f:
        json.dump(strip_returns(all_results), f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
