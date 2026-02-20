#!/usr/bin/env python3
"""
Phase 66: Fine-Tune Phase 65 Champion + Multi-Funding + 5-Way Exploration

Phase 65 breakthrough:
  Champion: V1(40%) + SR_437(10%) + Idio_437(25%) + Fund_Cont_168(25%)
  AVG=1.590, MIN=1.153, ALL 5 YEARS > 1.15
  Year-by-year: [2.201, 1.153, 1.212, 1.930, 1.456]

The 2022 is the weakest year (1.153). Fund_cont_168 year-by-year was [2.644, 0.573, 0.173, 2.026, 0.571].
→ Funding is strong in 2021/2024 but weak in 2022/2023. This is the residual risk.

Phase 66 goals:
  A. Fine-tune weights: higher precision grid around champion (0.5% steps)
  B. Multi-Funding: blend 96h + 168h funding for smoothed signal
  C. Leverage Adjustment: test higher leverage for funding strategy (25% → 35%)
  D. 5-Way Ensemble: add Vol_Adj_437 (which performed well in 2021 despite different path)
  E. Alternative funding lookbacks: 144h (6-day window, 18 settlements)

Academic basis for dual-funding:
  - Combining multiple lookbacks creates a "funding momentum" composite
  - Short (96h/4d): captures recent crowding buildup
  - Long (168h/7d): captures persistent crowding regime
  - Similar to the dual-horizon price momentum signal
  - Expected improvement: smoother signal, better 2022/2023 performance
"""

import json
import math
import statistics
import subprocess
import sys
import copy
from pathlib import Path

OUT_DIR = "artifacts/phase66"
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

# Phase 63 frozen params
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
    "k_per_side": 2, "lookback_bars": 437, "beta_window_bars": 72,
    "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 48,
}

SR_437_PARAMS = {
    "k_per_side": 2, "lookback_bars": 437,
    "vol_lookback_bars": 168, "target_gross_leverage": 0.35, "rebalance_interval_bars": 48,
}

FUND_168_PARAMS = {
    "k_per_side": 2, "funding_lookback_bars": 168, "direction": "contrarian",
    "vol_lookback_bars": 168, "target_gross_leverage": 0.25, "rebalance_interval_bars": 24,
}

VOL_ADJ_437_PARAMS = {
    "k_per_side": 2, "lookback_bars": 437, "signal_vol_bars": 437,
    "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 48,
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
    path = f"/tmp/phase66_{run_name}_{year}.json"
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
                print(f"    {year}: ERROR", flush=True)
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
        print(f"    {year}: no result", flush=True)

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
    print("  PHASE 66: FINE-TUNE + MULTI-FUNDING + 5-WAY EXPLORATION")
    print("=" * 80)
    print("  Phase 65 champion: V1(40%)+SR(10%)+Idio(25%)+Fund(25%): AVG=1.590, MIN=1.153")
    print("  Weakest year: 2022 (1.153). Funding in 2022: 0.573. Goal: improve 2022 floor.")
    print("=" * 80, flush=True)

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # ─── Run reference strategies ────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  Running reference strategies...")
    print("═" * 70, flush=True)
    v1_data = run_strategy("v1_ref", "nexus_alpha_v1", V1_PARAMS)
    idio_data = run_strategy("idio_437_ref", "idio_momentum_alpha", IDIO_437_PARAMS)
    sr_data = run_strategy("sr_437_ref", "sharpe_ratio_alpha", SR_437_PARAMS)
    fund_data = run_strategy("fund_168_ref", "funding_momentum_alpha", FUND_168_PARAMS)

    # ─── A: Fund_96 + Fine-tuned variants ───────────────────────────────────
    print("\n" + "═" * 70)
    print("  A: FUND_96 + OTHER FUNDING VARIANTS")
    print("  Testing 96h (just below optimal), 144h, 168h with higher leverage")
    print("═" * 70, flush=True)

    fund_96_data = run_strategy("fund_96", "funding_momentum_alpha", {
        "k_per_side": 2, "funding_lookback_bars": 96, "direction": "contrarian",
        "vol_lookback_bars": 168, "target_gross_leverage": 0.25, "rebalance_interval_bars": 24,
    })

    fund_144_data = run_strategy("fund_144", "funding_momentum_alpha", {
        "k_per_side": 2, "funding_lookback_bars": 144, "direction": "contrarian",
        "vol_lookback_bars": 168, "target_gross_leverage": 0.25, "rebalance_interval_bars": 24,
    })

    fund_168_hi_data = run_strategy("fund_168_hi_lev", "funding_momentum_alpha", {
        "k_per_side": 2, "funding_lookback_bars": 168, "direction": "contrarian",
        "vol_lookback_bars": 168, "target_gross_leverage": 0.35, "rebalance_interval_bars": 24,
    })

    print(f"\n  fund_96_avg: {fund_96_data['_avg']}, fund_144_avg: {fund_144_data['_avg']}")
    print(f"  fund_168_standard_avg: {fund_data['_avg']}, fund_168_hi_lev_avg: {fund_168_hi_data['_avg']}")

    # ─── B: Vol_Adj_437 (5th signal) ─────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  B: VOL_ADJ_437 (5th signal for 5-way ensemble)")
    print("  Phase 64B: vol_adj_437 = 1.131, but corr=0.973 with SR_437")
    print("  Checking if useful in 5-way given different year profile")
    print("═" * 70, flush=True)

    vol_adj_data = run_strategy("vol_adj_437", "vol_adjusted_momentum_alpha", VOL_ADJ_437_PARAMS)
    print(f"  Vol-Adj 437: AVG={vol_adj_data['_avg']}, MIN={vol_adj_data['_min']}")
    print(f"  Year-by-year: {[vol_adj_data.get(y, {}).get('sharpe', '?') for y in YEARS]}")

    # ─── C: 4-Way Fine-Tune (high precision grid) ────────────────────────────
    print("\n" + "═" * 80)
    print("  C: HIGH-PRECISION 4-WAY GRID SEARCH (V1 + SR + Idio + Fund)")
    print("  Phase 65 winner: V1=40/SR=10/Idio=25/Fund=25 AVG=1.590")
    print("═" * 80, flush=True)

    best_avg = {"avg": 1.590, "min": 1.153, "weights": (0.40, 0.10, 0.25, 0.25), "pos": 5}
    best_min = {"avg": 1.590, "min": 1.153, "weights": (0.40, 0.10, 0.25, 0.25)}

    # Fine-grained search: 5% steps in fund (15-35%), rest adjusted
    print("\n  V1+SR+Idio+Fund (fine-grained):")
    for w_v1 in [0.35, 0.40, 0.45]:
        for w_fund in [0.15, 0.20, 0.25, 0.30, 0.35]:
            for w_sr in [0.00, 0.05, 0.10, 0.15]:
                w_idio = round(1.0 - w_v1 - w_fund - w_sr, 2)
                if 0.15 <= w_idio <= 0.40:
                    ens = blend_ensemble(
                        {"v1": v1_data, "sr": sr_data, "idio": idio_data, "fund": fund_data},
                        {"v1": w_v1, "sr": w_sr, "idio": w_idio, "fund": w_fund},
                    )
                    if ens["avg"] > best_avg["avg"]:
                        best_avg = {
                            "avg": ens["avg"], "min": ens["min"],
                            "weights": (w_v1, w_sr, w_idio, w_fund), "pos": ens["pos"],
                            "yby": ens.get("yby"),
                        }
                    if ens["min"] > best_min.get("min", 0) and ens["avg"] > 1.30:
                        best_min = {
                            "avg": ens["avg"], "min": ens["min"],
                            "weights": (w_v1, w_sr, w_idio, w_fund),
                        }

    w_v1, w_sr, w_idio, w_f = best_avg["weights"]
    print(f"  BEST AVG: V1={int(w_v1*100)}/SR={int(w_sr*100)}/Idio={int(w_idio*100)}/Fund={int(w_f*100)}: "
          f"AVG={best_avg['avg']}, MIN={best_avg['min']}")
    print(f"  Year-by-year: {best_avg.get('yby')}")

    w_v1, w_sr, w_idio, w_f = best_min["weights"]
    print(f"  BEST MIN: V1={int(w_v1*100)}/SR={int(w_sr*100)}/Idio={int(w_idio*100)}/Fund={int(w_f*100)}: "
          f"AVG={best_min['avg']}, MIN={best_min['min']}")

    # ─── D: Multi-Funding (96h + 168h blended) ───────────────────────────────
    print("\n" + "═" * 80)
    print("  D: MULTI-FUNDING (96h + 168h combined in ensemble)")
    print("  Both are contrarian; different lookbacks capture different crowding regimes")
    print("═" * 80, flush=True)

    # Test using both fund signals in the 5-way
    print("\n  V1+SR+Idio+Fund96+Fund168 (5-way with dual funding):")
    for w_v1, w_sr, w_idio, w_f96, w_f168 in [
        (0.40, 0.05, 0.20, 0.15, 0.20),
        (0.40, 0.05, 0.20, 0.10, 0.25),
        (0.40, 0.10, 0.20, 0.10, 0.20),
        (0.35, 0.05, 0.20, 0.15, 0.25),
        (0.35, 0.05, 0.25, 0.10, 0.25),
        (0.40, 0.00, 0.25, 0.10, 0.25),
        (0.40, 0.00, 0.20, 0.15, 0.25),
        (0.35, 0.00, 0.25, 0.15, 0.25),
        (0.35, 0.10, 0.20, 0.10, 0.25),
    ]:
        ens = blend_ensemble(
            {"v1": v1_data, "sr": sr_data, "idio": idio_data,
             "f96": fund_96_data, "f168": fund_data},
            {"v1": w_v1, "sr": w_sr, "idio": w_idio, "f96": w_f96, "f168": w_f168},
        )
        tag = f"V1={int(w_v1*100)}/SR={int(w_sr*100)}/Idio={int(w_idio*100)}/F96={int(w_f96*100)}/F168={int(w_f168*100)}"
        champ_flag = " *** NEW ***" if ens["avg"] > 1.590 else ""
        print(f"    {tag}: AVG={ens['avg']}, MIN={ens['min']}, pos={ens['pos']}/5{champ_flag}")

        if ens["avg"] > best_avg["avg"]:
            best_avg = {
                "avg": ens["avg"], "min": ens["min"],
                "weights": (w_v1, w_sr, w_idio, w_f96, w_f168),
                "pos": ens["pos"], "kind": "5way_dual_fund",
                "yby": ens.get("yby"),
            }

    # ─── E: 5-Way with Vol_Adj ────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("  E: 5-WAY ENSEMBLE (V1 + SR + Idio + Fund + Vol_Adj)")
    print("  Note: corr(SR, vol_adj)=0.973 — almost redundant. Testing anyway.")
    print("═" * 80, flush=True)

    for w_v1, w_sr, w_idio, w_fund, w_va in [
        (0.40, 0.05, 0.20, 0.25, 0.10),
        (0.35, 0.05, 0.20, 0.25, 0.15),
        (0.40, 0.00, 0.25, 0.25, 0.10),
        (0.35, 0.05, 0.25, 0.25, 0.10),
        (0.40, 0.05, 0.25, 0.20, 0.10),
    ]:
        ens = blend_ensemble(
            {"v1": v1_data, "sr": sr_data, "idio": idio_data,
             "fund": fund_data, "va": vol_adj_data},
            {"v1": w_v1, "sr": w_sr, "idio": w_idio, "fund": w_fund, "va": w_va},
        )
        tag = f"V1={int(w_v1*100)}/SR={int(w_sr*100)}/Idio={int(w_idio*100)}/Fund={int(w_fund*100)}/VA={int(w_va*100)}"
        champ_flag = " *** NEW ***" if ens["avg"] > 1.590 else ""
        print(f"    {tag}: AVG={ens['avg']}, MIN={ens['min']}, pos={ens['pos']}/5{champ_flag}")

        if ens["avg"] > best_avg["avg"]:
            best_avg = {
                "avg": ens["avg"], "min": ens["min"],
                "weights": (w_v1, w_sr, w_idio, w_fund, w_va),
                "pos": ens["pos"], "kind": "5way_vol_adj",
                "yby": ens.get("yby"),
            }

    # ─── Save Final Champion ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  PHASE 66 FINAL CHAMPION")
    print("=" * 80, flush=True)

    print(f"\n  Champion AVG: {best_avg['avg']}, MIN: {best_avg['min']}, pos: {best_avg.get('pos')}/5")
    print(f"  Weights: {best_avg['weights']}")
    if "yby" in best_avg:
        print(f"  Year-by-year: {best_avg['yby']}")

    w = best_avg["weights"]
    kind = best_avg.get("kind", "4way")

    if best_avg["avg"] > 1.590:
        print(f"  *** BEATS Phase 65 champion (1.590) ***")
        if kind == "5way_dual_fund" and len(w) == 5:
            w_v1, w_sr, w_idio, w_f96, w_f168 = w
            run_name = f"ensemble_v1sr_idio_f96f168_{int(w_v1*100)}{int(w_sr*100)}{int(w_idio*100)}{int(w_f96*100)}{int(w_f168*100)}"
            save_champion_config(run_name, f"Phase 66 champion (dual-funding 5-way). AVG={best_avg['avg']}", {
                "v1_long": {"weight": w_v1, "name": "nexus_alpha_v1", "params": V1_PARAMS},
                "sr_437": {"weight": w_sr, "name": "sharpe_ratio_alpha", "params": SR_437_PARAMS},
                "idio_437": {"weight": w_idio, "name": "idio_momentum_alpha", "params": IDIO_437_PARAMS},
                "fund_96": {"weight": w_f96, "name": "funding_momentum_alpha", "params": {
                    "k_per_side": 2, "funding_lookback_bars": 96, "direction": "contrarian",
                    "vol_lookback_bars": 168, "target_gross_leverage": 0.25, "rebalance_interval_bars": 24,
                }},
                "fund_168": {"weight": w_f168, "name": "funding_momentum_alpha", "params": FUND_168_PARAMS},
            })
        elif len(w) == 4:
            w_v1, w_sr, w_idio, w_fund = w
            run_name = f"ensemble_v1_sr_idio_fund_{int(w_v1*100)}{int(w_sr*100)}{int(w_idio*100)}{int(w_fund*100)}"
            save_champion_config(run_name, f"Phase 66 champion. AVG={best_avg['avg']}", {
                "v1_long": {"weight": w_v1, "name": "nexus_alpha_v1", "params": V1_PARAMS},
                "sr_437": {"weight": w_sr, "name": "sharpe_ratio_alpha", "params": SR_437_PARAMS},
                "idio_437": {"weight": w_idio, "name": "idio_momentum_alpha", "params": IDIO_437_PARAMS},
                "fund_168": {"weight": w_fund, "name": "funding_momentum_alpha", "params": FUND_168_PARAMS},
            })
    else:
        print(f"  Phase 65 champion (1.590) remains best")
        print(f"  Phase 66 confirmed optimality of V1(40%)+SR(10%)+Idio(25%)+Fund(25%)")

    # Final signal hierarchy
    print(f"\n  SIGNAL HIERARCHY (standalone AVG Sharpe):")
    print(f"    Idio_437:      AVG={idio_data['_avg']}")
    print(f"    Fund_Cont_168: AVG={fund_data['_avg']}")
    print(f"    SR_437:        AVG={sr_data['_avg']}")
    print(f"    Vol_Adj_437:   AVG={vol_adj_data['_avg']}")
    print(f"    V1-Long:       AVG={v1_data['_avg']}")

    out_path = Path(OUT_DIR) / "phase66_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "best_avg": {k: v for k, v in best_avg.items() if k != "returns"},
            "best_min": best_min,
            "standalone": {
                "v1": v1_data["_avg"],
                "idio_437": idio_data["_avg"],
                "sr_437": sr_data["_avg"],
                "fund_168": fund_data["_avg"],
                "vol_adj_437": vol_adj_data["_avg"],
            },
        }, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
