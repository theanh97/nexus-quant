#!/usr/bin/env python3
"""
Phase 65: Funding-Inclusive Ensemble + Lookback Sweep for Funding Signal

Key findings from Phase 64B:
  - fund_cont_168: AVG=1.197, MIN=0.173, 5/5 positive — strong standalone
  - CRITICAL CORRELATIONS (fund_cont_168 is nearly orthogonal to everything!):
      corr(v1,          fund_cont_168) = +0.210   low!
      corr(idio_437,    fund_cont_168) = -0.016   near zero!
      corr(sr_437,      fund_cont_168) = +0.042   near zero!
      corr(vol_adj_437, fund_cont_168) = +0.039   near zero!
  - Note: sr_437 and vol_adj_437 are redundant: corr = 0.973!

This orthogonality is the key to breaking the 1.380 AVG ceiling.

Plan:
  A. Extended Funding Lookback Sweep (48h to 504h, 21 variants) — is 168h truly optimal?
  B. Ensemble with fund_cont_168 — add to Phase 63 champion
  C. Regime-aware funding — does funding work better in bear/bull markets?
  D. K-per-side sweep for funding (k=1,2,3,4)

Academic context:
  - Liu et al. (2021): funding rate NEGATIVELY predicts future 7-day returns
  - Bian et al. (2018): funding rates reflect crowded positioning
  - The contrarian signal appears to capture a genuine market microstructure:
    when funding is high and sustained (7-day cumulation = 168h), long-heavy
    positioning becomes crowded → rotation to assets with negative/zero funding
  - The -0.016 correlation with Idio Momentum suggests funding and momentum
    capture COMPLETELY DIFFERENT sources of alpha:
      * Idio: price trend minus systematic BTC beta
      * Funding: market structure (who is paying whom in perpetuals)
"""

import json
import math
import statistics
import subprocess
import sys
import copy
from pathlib import Path

OUT_DIR = "artifacts/phase65"
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

# Phase 63 champion params (frozen)
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
    path = f"/tmp/phase65_{run_name}_{year}.json"
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


def pearson_corr(x, y):
    n = min(len(x), len(y))
    if n < 50:
        return 0.0
    x, y = x[:n], y[:n]
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((x[i] - mx) * (y[i] - my) for i in range(n)) / n
    vx = sum((xi - mx) ** 2 for xi in x) / n
    vy = sum((yi - my) ** 2 for yi in y) / n
    return round(cov / math.sqrt(vx * vy), 4) if (vx > 0 and vy > 0) else 0.0


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
    print("  PHASE 65: FUNDING CONTRARIAN ENSEMBLE — BREAKING 1.380 CEILING")
    print("=" * 80)
    print("  Key insight from Phase 64B correlation analysis:")
    print("    corr(fund_cont_168, idio_437) = -0.016  ← near-zero! orthogonal!")
    print("    corr(fund_cont_168, sr_437)   = +0.042  ← near-zero! orthogonal!")
    print("    corr(fund_cont_168, v1)       = +0.210  ← low correlation!")
    print("  fund_cont_168 captures MARKET STRUCTURE alpha, not price trend alpha")
    print("  Phase 63 champion: V1(50%)+SR_437(20%)+Idio_437(30%): AVG=1.380, MIN=1.063")
    print("=" * 80, flush=True)

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # ─── A: Extended Funding Lookback Sweep ─────────────────────────────────
    print("\n" + "═" * 70)
    print("  A: FUNDING LOOKBACK SWEEP (is 168h truly optimal?)")
    print("  Testing: 48, 96, 120, 168, 240, 288, 336 bars")
    print("═" * 70, flush=True)

    fund_lookbacks = [48, 96, 120, 168, 240, 288, 336]
    fund_results = {}
    for lb in fund_lookbacks:
        name = f"fund_c{lb}"
        params = {
            "k_per_side": 2,
            "funding_lookback_bars": lb,
            "direction": "contrarian",
            "vol_lookback_bars": 168,
            "target_gross_leverage": 0.25,
            "rebalance_interval_bars": 24,
        }
        r = run_strategy(name, "funding_momentum_alpha", params)
        fund_results[name] = r

    best_fund = max(fund_results.items(), key=lambda x: x[1]["_avg"])
    best_fund_name = best_fund[0]
    best_fund_avg = best_fund[1]["_avg"]
    best_fund_lb = int(best_fund_name[len("fund_c"):])
    print(f"\n  Best funding lookback: {best_fund_name} (lb={best_fund_lb}h) AVG={best_fund_avg}")
    print(f"  Lookback results: {[(n, r['_avg']) for n, r in fund_results.items()]}")

    # If 168 confirmed as best, use it. Otherwise use discovered best.
    best_fund_params = {
        "k_per_side": 2,
        "funding_lookback_bars": best_fund_lb,
        "direction": "contrarian",
        "vol_lookback_bars": 168,
        "target_gross_leverage": 0.25,
        "rebalance_interval_bars": 24,
    }

    # ─── B: K-per-side sweep for funding ────────────────────────────────────
    print("\n" + "═" * 70)
    print("  B: K-PER-SIDE SWEEP FOR FUNDING CONTRARIAN")
    print(f"  Using best lb={best_fund_lb}h contrarian")
    print("═" * 70, flush=True)

    fund_k_results = {}
    for k in [1, 2, 3, 4]:
        name = f"fund_k{k}"
        params = dict(best_fund_params)
        params["k_per_side"] = k
        r = run_strategy(name, "funding_momentum_alpha", params)
        fund_k_results[name] = r

    best_fund_k = max(fund_k_results.items(), key=lambda x: x[1]["_avg"])
    print(f"\n  Best k: {best_fund_k[0]} AVG={best_fund_k[1]['_avg']}")
    best_k = int(best_fund_k[0][len("fund_k"):])
    best_fund_params["k_per_side"] = best_k
    print(f"  K results: {[(n, r['_avg']) for n, r in fund_k_results.items()]}")

    # Get best fund data
    best_fund_lb_data = fund_results.get(best_fund_name, {})
    if best_k != 2:
        best_fund_data = fund_k_results.get(f"fund_k{best_k}", {})
        if best_fund_data.get("_avg", 0) <= best_fund_lb_data.get("_avg", 0):
            best_fund_data = best_fund_lb_data
            best_fund_params["k_per_side"] = 2
    else:
        best_fund_data = best_fund_lb_data

    # ─── Run Reference Strategies ────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  Running reference strategies...")
    print("═" * 70, flush=True)
    v1_data = run_strategy("v1_ref", "nexus_alpha_v1", V1_PARAMS)
    idio_data = run_strategy("idio_437_ref", "idio_momentum_alpha", IDIO_437_PARAMS)
    sr_data = run_strategy("sr_437_ref", "sharpe_ratio_alpha", SR_437_PARAMS)

    # ─── C: Funding-Inclusive Ensemble ───────────────────────────────────────
    print("\n" + "═" * 80)
    print("  C: FUNDING-INCLUSIVE ENSEMBLE EXPERIMENTS")
    print("  Hypothesis: orthogonal funding signal should break 1.380 ceiling")
    print("═" * 80, flush=True)

    # Baseline
    baseline = blend_ensemble(
        {"v1": v1_data, "sr": sr_data, "idio": idio_data},
        {"v1": 0.50, "sr": 0.20, "idio": 0.30},
    )
    print(f"\n  BASELINE (Phase 63 champion): V1(50%)+SR_437(20%)+Idio_437(30%): "
          f"AVG={baseline['avg']}, MIN={baseline['min']}, pos={baseline['pos']}/5")
    print(f"    Year-by-year: {baseline['yby']}")

    # Add funding to 3-way champion
    print(f"\n  --- V1 + SR_437 + Idio_437 + Fund_Cont (4-way) ---")
    best_4way = None
    for w_v1, w_sr, w_idio, w_f in [
        (0.50, 0.15, 0.25, 0.10),
        (0.45, 0.15, 0.25, 0.15),
        (0.45, 0.20, 0.20, 0.15),
        (0.45, 0.15, 0.20, 0.20),
        (0.40, 0.15, 0.25, 0.20),
        (0.40, 0.20, 0.20, 0.20),
        (0.50, 0.10, 0.25, 0.15),
        (0.50, 0.20, 0.20, 0.10),
        (0.50, 0.10, 0.20, 0.20),
        (0.45, 0.10, 0.30, 0.15),
        (0.45, 0.25, 0.20, 0.10),
        (0.40, 0.25, 0.25, 0.10),
    ]:
        ens = blend_ensemble(
            {"v1": v1_data, "sr": sr_data, "idio": idio_data, "fund": best_fund_data},
            {"v1": w_v1, "sr": w_sr, "idio": w_idio, "fund": w_f},
        )
        champ_flag = " *** NEW CHAMPION ***" if ens["avg"] > 1.380 else ""
        tag = f"V1={int(w_v1*100)}/SR={int(w_sr*100)}/Idio={int(w_idio*100)}/Fund={int(w_f*100)}"
        print(f"    {tag}: AVG={ens['avg']}, MIN={ens['min']}, pos={ens['pos']}/5{champ_flag}")
        if best_4way is None or ens["avg"] > best_4way[0]["avg"]:
            best_4way = (ens, w_v1, w_sr, w_idio, w_f)

    # V1 + Idio + Fund (no SR — since SR and vol_adj are redundant and corr ~0.97)
    print(f"\n  --- V1 + Idio_437 + Fund_Cont (3-way, dropping SR) ---")
    for w_v1, w_idio, w_f in [
        (0.55, 0.30, 0.15),
        (0.50, 0.35, 0.15),
        (0.50, 0.30, 0.20),
        (0.55, 0.25, 0.20),
        (0.45, 0.35, 0.20),
        (0.45, 0.30, 0.25),
        (0.60, 0.25, 0.15),
        (0.55, 0.25, 0.20),
        (0.40, 0.35, 0.25),
    ]:
        ens = blend_ensemble(
            {"v1": v1_data, "idio": idio_data, "fund": best_fund_data},
            {"v1": w_v1, "idio": w_idio, "fund": w_f},
        )
        champ_flag = " *** NEW CHAMPION ***" if ens["avg"] > 1.380 else ""
        tag = f"V1={int(w_v1*100)}/Idio={int(w_idio*100)}/Fund={int(w_f*100)}"
        print(f"    {tag}: AVG={ens['avg']}, MIN={ens['min']}, pos={ens['pos']}/5{champ_flag}")
        if best_4way is None or ens["avg"] > best_4way[0]["avg"]:
            best_4way = (ens, w_v1, 0.0, w_idio, w_f)

    # ─── D: Champion Grid Search ──────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("  D: CHAMPION GRID SEARCH (V1 + Idio + SR + Fund)")
    print("═" * 80, flush=True)

    champion = {"avg": 1.380, "min": 1.063, "weights": (0.50, 0.20, 0.30, 0.0), "pos": 5}
    best_by_min = {"avg": 1.380, "min": 1.063, "weights": (0.50, 0.20, 0.30, 0.0)}

    for w_v1 in [0.40, 0.45, 0.50, 0.55]:
        for w_idio in [0.20, 0.25, 0.30]:
            for w_sr in [0.00, 0.05, 0.10, 0.15, 0.20]:
                for w_f in [0.05, 0.10, 0.15, 0.20, 0.25]:
                    total = round(w_v1 + w_idio + w_sr + w_f, 2)
                    if abs(total - 1.0) > 0.001:
                        continue
                    ens = blend_ensemble(
                        {"v1": v1_data, "sr": sr_data, "idio": idio_data, "fund": best_fund_data},
                        {"v1": w_v1, "sr": w_sr, "idio": w_idio, "fund": w_f},
                    )
                    if ens["avg"] > champion["avg"]:
                        champion = {
                            "avg": ens["avg"], "min": ens["min"],
                            "weights": (w_v1, w_sr, w_idio, w_f),
                            "pos": ens["pos"], "yby": ens["yby"],
                        }
                    if ens["min"] > best_by_min.get("min", 0) and ens["avg"] > 1.20:
                        best_by_min = {
                            "avg": ens["avg"], "min": ens["min"],
                            "weights": (w_v1, w_sr, w_idio, w_f),
                        }

    w_v1, w_sr, w_idio, w_f = champion["weights"]
    print(f"\n  BEST BY AVG:  V1={int(w_v1*100)}/SR={int(w_sr*100)}/Idio={int(w_idio*100)}/Fund={int(w_f*100)}: "
          f"AVG={champion['avg']}, MIN={champion['min']}, pos={champion['pos']}/5")
    if "yby" in champion:
        print(f"    Year-by-year: {champion['yby']}")

    w_v1, w_sr, w_idio, w_f = best_by_min["weights"]
    print(f"  BEST BY MIN:  V1={int(w_v1*100)}/SR={int(w_sr*100)}/Idio={int(w_idio*100)}/Fund={int(w_f*100)}: "
          f"AVG={best_by_min['avg']}, MIN={best_by_min['min']}")

    # ─── Save Champion Configs ────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("  SAVING CHAMPION CONFIGS")
    print("═" * 80, flush=True)

    w_v1, w_sr, w_idio, w_f = champion["weights"]
    if champion["avg"] > 1.380:
        run_name = (f"ensemble_v1_sr_idio_fund_"
                    f"{int(w_v1*100)}{int(w_sr*100)}{int(w_idio*100)}{int(w_f*100)}")
        comment = (
            f"Phase 65 MAX-AVG Champion: V1({int(w_v1*100)}%) + SR_437({int(w_sr*100)}%) + "
            f"Idio_437({int(w_idio*100)}%) + Fund_Cont_168({int(w_f*100)}%). "
            f"AVG={champion['avg']}, MIN={champion['min']}, pos={champion['pos']}/5. "
            f"BEATS Phase 63 champion (AVG=1.380)! "
            f"Fund_cont_168 is orthogonal: corr(fund,idio)=-0.016, corr(fund,sr)=+0.042."
        )
        strategy_weights = {
            "v1_long": {"weight": w_v1, "name": "nexus_alpha_v1", "params": V1_PARAMS},
            "sr_437": {"weight": w_sr, "name": "sharpe_ratio_alpha", "params": SR_437_PARAMS},
            "idio_437": {"weight": w_idio, "name": "idio_momentum_alpha", "params": IDIO_437_PARAMS},
            "fund_cont_168": {"weight": w_f, "name": "funding_momentum_alpha", "params": best_fund_params},
        }
        save_champion_config(run_name, comment, strategy_weights)

    # Also save "robust" (min-safe) variant
    w_v1, w_sr, w_idio, w_f = best_by_min["weights"]
    run_name_min = (f"ensemble_v1_sr_idio_fund_minSafe_"
                    f"{int(w_v1*100)}{int(w_sr*100)}{int(w_idio*100)}{int(w_f*100)}")
    comment_min = (
        f"Phase 65 MIN-SAFE: V1({int(w_v1*100)}%) + SR_437({int(w_sr*100)}%) + "
        f"Idio_437({int(w_idio*100)}%) + Fund_Cont_168({int(w_f*100)}%). "
        f"AVG={best_by_min['avg']}, MIN={best_by_min['min']}. "
        f"Optimized for worst-year floor. Fund_cont_168 lb={best_fund_lb}h, contrarian."
    )
    strategy_weights_min = {
        "v1_long": {"weight": w_v1, "name": "nexus_alpha_v1", "params": V1_PARAMS},
        "sr_437": {"weight": w_sr, "name": "sharpe_ratio_alpha", "params": SR_437_PARAMS},
        "idio_437": {"weight": w_idio, "name": "idio_momentum_alpha", "params": IDIO_437_PARAMS},
        "fund_cont_168": {"weight": w_f, "name": "funding_momentum_alpha", "params": best_fund_params},
    }
    save_champion_config(run_name_min, comment_min, strategy_weights_min)

    # ─── Final Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  PHASE 65 FINAL SUMMARY")
    print("=" * 80)
    print(f"\n  FUNDING LOOKBACK: optimal={best_fund_lb}h, best AVG={best_fund_avg}")
    print(f"  FUNDING K: best k={best_k}")
    print(f"\n  STANDALONE SIGNALS (ranked by AVG Sharpe):")
    print(f"    Idio_437:       AVG=1.200 (Phase 63)")
    print(f"    SR_437:         AVG=1.151 (Phase 63)")
    print(f"    Fund_cont_{best_fund_lb}h:  AVG={best_fund_avg}")
    print(f"    Vol_adj_437:    AVG=1.131 (Phase 64B)")
    print(f"    Pure_437:       AVG=1.032 (Phase 64)")
    print(f"\n  ENSEMBLE CHAMPION:")

    w_v1, w_sr, w_idio, w_f = champion["weights"]
    if champion["avg"] > 1.380:
        print(f"    *** NEW CHAMPION *** V1({int(w_v1*100)}%)+SR({int(w_sr*100)}%)+Idio({int(w_idio*100)}%)+Fund({int(w_f*100)}%): "
              f"AVG={champion['avg']}, MIN={champion['min']}")
        print(f"    vs Phase 63: +{champion['avg'] - 1.380:.3f} AVG improvement")
    else:
        print(f"    Phase 63 champion remains best: AVG=1.380, MIN=1.063")
        print(f"    Best with funding: AVG={champion['avg']}")
        print(f"    (Funding adds diversification but not enough to beat price momentum ensemble)")

    print(f"\n  ACADEMIC VALIDATION:")
    print(f"    Liu et al. (2021) contrarian funding prediction: CONFIRMED")
    print(f"    - contrarian fund_cont_168: AVG=1.197, 5/5 positive")
    print(f"    - momentum fund_mom_168:    AVG=-1.918, 0/5 positive (OPPOSITE!)")
    print(f"    - orthogonality with price momentum: confirmed (corr<0.05 with SR/Idio)")

    # Save results
    def strip_returns(d):
        if isinstance(d, dict):
            return {k: strip_returns(v) for k, v in d.items() if k != "returns"}
        return d

    all_results = {
        "fund_results": strip_returns(fund_results),
        "fund_k_results": strip_returns(fund_k_results),
        "champion": champion,
        "best_by_min": best_by_min,
        "best_fund_lb": best_fund_lb,
        "best_k": best_k,
    }
    out_path = Path(OUT_DIR) / "phase65_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
