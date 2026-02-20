#!/usr/bin/env python3
"""
Phase 67: Fund_144 Swap + Lookback Refinement + Triple Funding

Phase 66 champion: V1(35%)+SR(5%)+Idio(20%)+F96(15%)+F168(25%) AVG=1.673, MIN=1.08

Key insight from Phase 66:
  fund_144 standalone AVG=1.302 >> fund_96 standalone AVG=1.031
  fund_144 year-by-year: [3.115, 0.496, 0.664, 2.171, 0.062]
  fund_96  year-by-year: [3.257, 0.062, 0.276, 1.368, 0.193]

  → fund_144 is MUCH better in 2022 (0.496 vs 0.062) and 2023 (0.664 vs 0.276)
  → fund_144 is weak in 2025 (0.062), fund_168 covers this (0.571)
  → fund_144 + fund_168 should be MORE complementary than fund_96 + fund_168

Goals:
  A. Funding lookback sweep 96-168h (every 12h) — find true peak vs fund_144
  B. 4-way with fund_144: V1+SR+Idio+Fund144 (drop both 96 and 168)
  C. 5-way swap: replace fund_96 with fund_144 → V1+SR+Idio+Fund144+Fund168
  D. Triple funding 6-way: V1+SR+Idio+Fund96+Fund144+Fund168
  E. Grid search for champion in C configuration (likely best)
"""

import json
import math
import statistics
import subprocess
import sys
import copy
from pathlib import Path

OUT_DIR = "artifacts/phase67"
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

# ── Frozen params from Phase 63/65 ─────────────────────────────────────────
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

FUND_96_PARAMS = {
    "k_per_side": 2, "funding_lookback_bars": 96, "direction": "contrarian",
    "vol_lookback_bars": 168, "target_gross_leverage": 0.25, "rebalance_interval_bars": 24,
}

FUND_144_PARAMS = {
    "k_per_side": 2, "funding_lookback_bars": 144, "direction": "contrarian",
    "vol_lookback_bars": 168, "target_gross_leverage": 0.25, "rebalance_interval_bars": 24,
}


# ── Helpers ─────────────────────────────────────────────────────────────────

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
    path = f"/tmp/phase67_{run_name}_{year}.json"
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


def pearson_corr(x: list, y: list) -> float:
    n = min(len(x), len(y))
    if n < 10:
        return float("nan")
    x, y = x[:n], y[:n]
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((x[i] - mx) * (y[i] - my) for i in range(n)) / n
    sx = math.sqrt(sum((v - mx) ** 2 for v in x) / n)
    sy = math.sqrt(sum((v - my) ** 2 for v in y) / n)
    if sx < 1e-10 or sy < 1e-10:
        return float("nan")
    return round(cov / (sx * sy), 4)


def save_champion_config(run_name: str, comment: str, strategy_weights: dict) -> None:
    cfg = {
        "_comment": comment,
        "run_name": run_name,
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
        "strategy": {
            "name": "ensemble_v1",
            "params": {
                "sub_strategies": strategy_weights,
                "rebalance_interval_bars": 1,
            },
        },
    }
    config_name = f"configs/{run_name}.json"
    with open(config_name, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  Config saved: {config_name}", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  PHASE 67: FUND_144 SWAP + LOOKBACK REFINEMENT + TRIPLE FUNDING")
    print("=" * 80)
    print("  Phase 66 champion: V1(35%)+SR(5%)+Idio(20%)+F96(15%)+F168(25%)")
    print("  AVG=1.673, MIN=1.08")
    print("  Goal: Replace F96 with F144 (better 2022: 0.496 vs 0.062)")
    print("=" * 80)

    # ── Reference strategies ─────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  Running reference strategies...")
    print("═" * 70)

    v1_data = run_strategy("v1_ref", "nexus_alpha_v1", V1_PARAMS)
    idio_data = run_strategy("idio_437_ref", "idio_momentum_alpha", IDIO_437_PARAMS)
    sr_data = run_strategy("sr_437_ref", "sharpe_ratio_alpha", SR_437_PARAMS)
    fund168_data = run_strategy("fund_168_ref", "funding_momentum_alpha", FUND_168_PARAMS)
    fund96_data = run_strategy("fund_96_ref", "funding_momentum_alpha", FUND_96_PARAMS)

    print(f"\n  v1_ref: AVG={v1_data['_avg']}, year-by-year: "
          f"{[round(v1_data[y]['sharpe'], 3) for y in YEARS]}")
    print(f"  idio_437: AVG={idio_data['_avg']}, year-by-year: "
          f"{[round(idio_data[y]['sharpe'], 3) for y in YEARS]}")
    print(f"  sr_437: AVG={sr_data['_avg']}, year-by-year: "
          f"{[round(sr_data[y]['sharpe'], 3) for y in YEARS]}")
    print(f"  fund_168: AVG={fund168_data['_avg']}, year-by-year: "
          f"{[round(fund168_data[y]['sharpe'], 3) for y in YEARS]}")
    print(f"  fund_96: AVG={fund96_data['_avg']}, year-by-year: "
          f"{[round(fund96_data[y]['sharpe'], 3) for y in YEARS]}")

    # ── A: Funding lookback sweep 96-168h every 12h ──────────────────────────
    print("\n" + "═" * 70)
    print("  A: FUNDING LOOKBACK SWEEP 96-168h (every 12h)")
    print("  Phase 65 found: 168h=1.197, Phase 66 found: 144h=1.302 (better!)")
    print("  Filling gaps to find the true peak between 96h and 168h")
    print("═" * 70)

    fund_sweep_lbs = [96, 108, 120, 132, 144, 156, 168]
    fund_sweep_results = {}

    # Reuse already-run data for 96, 168
    fund_sweep_results[96] = fund96_data
    fund_sweep_results[168] = fund168_data

    for lb in fund_sweep_lbs:
        if lb in (96, 168):
            continue  # already have these
        params = {
            "k_per_side": 2, "funding_lookback_bars": lb, "direction": "contrarian",
            "vol_lookback_bars": 168, "target_gross_leverage": 0.25, "rebalance_interval_bars": 24,
        }
        fund_sweep_results[lb] = run_strategy(f"fund_{lb}", "funding_momentum_alpha", params)

    print("\n  Funding lookback sweep summary:")
    for lb in fund_sweep_lbs:
        d = fund_sweep_results[lb]
        yby = [round(d[y]["sharpe"], 3) for y in YEARS if y in d and "sharpe" in d[y]]
        print(f"    fund_{lb:3d}h: AVG={d['_avg']:.3f}, MIN={d['_min']:.3f}  {yby}")

    best_lb = max(fund_sweep_lbs, key=lambda lb: fund_sweep_results[lb]["_avg"])
    print(f"\n  Best lookback: fund_{best_lb}h (AVG={fund_sweep_results[best_lb]['_avg']:.3f})")

    # ── B: Correlation analysis (fund_144 vs fund_168) ──────────────────────
    print("\n" + "═" * 70)
    print("  B: CORRELATION ANALYSIS — FUND_144 vs FUND_168")
    print("═" * 70)

    fund144_data = fund_sweep_results[144]
    fund168_data = fund_sweep_results[168]

    r144 = get_all_returns(fund144_data)
    r168 = get_all_returns(fund168_data)
    r96 = get_all_returns(fund96_data)
    rv1 = get_all_returns(v1_data)
    ridio = get_all_returns(idio_data)
    rsr = get_all_returns(sr_data)

    print(f"\n  corr(fund_144, fund_168) = {pearson_corr(r144, r168)}")
    print(f"  corr(fund_144, fund_96)  = {pearson_corr(r144, r96)}")
    print(f"  corr(fund_168, fund_96)  = {pearson_corr(r168, r96)}")
    print(f"  corr(fund_144, v1)       = {pearson_corr(r144, rv1)}")
    print(f"  corr(fund_144, idio_437) = {pearson_corr(r144, ridio)}")
    print(f"  corr(fund_144, sr_437)   = {pearson_corr(r144, rsr)}")

    # ── C: 4-way with fund_144 only (replacing both 96 and 168) ─────────────
    print("\n" + "═" * 70)
    print("  C: 4-WAY WITH FUND_144 (V1 + SR + Idio + Fund144)")
    print("  Baseline Phase 65: V1=40/SR=10/Idio=25/Fund168=25: AVG=1.590")
    print("═" * 70)

    base_strats_c = {
        "v1": v1_data,
        "sr": sr_data,
        "idio": idio_data,
        "f144": fund144_data,
    }

    print("\n  V1+SR+Idio+Fund144 (4-way, replacing fund_168):")
    best_c = None
    best_c_avg = 0.0

    for v1w in [35, 40, 45]:
        for srw in [0, 5, 10, 15]:
            for idiow in [20, 25, 30]:
                f144w = 100 - v1w - srw - idiow
                if f144w < 15 or f144w > 40:
                    continue
                w = {"v1": v1w/100, "sr": srw/100, "idio": idiow/100, "f144": f144w/100}
                r = blend_ensemble(base_strats_c, w)
                if r["pos"] == 5:
                    tag = " *** NEW ***" if r["avg"] > 1.590 else ""
                    print(f"    V1={v1w}/SR={srw}/Idio={idiow}/F144={f144w}: "
                          f"AVG={r['avg']}, MIN={r['min']}, pos={r['pos']}/5{tag}")
                    if r["avg"] > best_c_avg:
                        best_c_avg = r["avg"]
                        best_c = (v1w, srw, idiow, f144w, r)

    if best_c:
        v1w, srw, idiow, f144w, r = best_c
        print(f"\n  BEST 4-way+F144: V1={v1w}/SR={srw}/Idio={idiow}/F144={f144w}: "
              f"AVG={r['avg']}, MIN={r['min']}")
        print(f"  Year-by-year: {r['yby']}")

    # ── D: 5-way swap (Fund_96 → Fund_144) ──────────────────────────────────
    print("\n" + "═" * 70)
    print("  D: 5-WAY SWAP — REPLACE FUND_96 WITH FUND_144")
    print("  Phase 66 champion used F96+F168. Now try F144+F168.")
    print("  fund_144 is MUCH better in 2022 (0.496 vs 0.062)")
    print("═" * 70)

    base_strats_d = {
        "v1": v1_data,
        "sr": sr_data,
        "idio": idio_data,
        "f144": fund144_data,
        "f168": fund168_data,
    }

    print("\n  V1+SR+Idio+Fund144+Fund168 (5-way with dual funding 144+168):")
    best_d = None
    best_d_avg = 0.0

    phase66_champ = 1.673

    for v1w in [30, 35, 40]:
        for srw in [0, 5, 10]:
            for idiow in [15, 20, 25]:
                for f144w in [10, 15, 20, 25]:
                    f168w = 100 - v1w - srw - idiow - f144w
                    if f168w < 10 or f168w > 35:
                        continue
                    w = {
                        "v1": v1w/100, "sr": srw/100, "idio": idiow/100,
                        "f144": f144w/100, "f168": f168w/100,
                    }
                    r = blend_ensemble(base_strats_d, w)
                    if r["pos"] == 5:
                        tag = " *** NEW CHAMPION ***" if r["avg"] > phase66_champ else ""
                        if r["avg"] > phase66_champ or r["avg"] > 1.65:
                            print(f"    V1={v1w}/SR={srw}/Idio={idiow}/F144={f144w}/F168={f168w}: "
                                  f"AVG={r['avg']}, MIN={r['min']}, pos={r['pos']}/5{tag}")
                        if r["avg"] > best_d_avg:
                            best_d_avg = r["avg"]
                            best_d = (v1w, srw, idiow, f144w, f168w, r)

    if best_d:
        v1w, srw, idiow, f144w, f168w, r = best_d
        print(f"\n  BEST D: V1={v1w}/SR={srw}/Idio={idiow}/F144={f144w}/F168={f168w}: "
              f"AVG={r['avg']}, MIN={r['min']}")
        print(f"  Year-by-year: {r['yby']}")

    # ── E: Triple funding 6-way (Fund96 + Fund144 + Fund168) ────────────────
    print("\n" + "═" * 70)
    print("  E: TRIPLE FUNDING 6-WAY (V1 + SR + Idio + F96 + F144 + F168)")
    print("  All three lookbacks capture different crowding regimes")
    print("═" * 70)

    base_strats_e = {
        "v1": v1_data,
        "sr": sr_data,
        "idio": idio_data,
        "f96": fund96_data,
        "f144": fund144_data,
        "f168": fund168_data,
    }

    print("\n  V1+SR+Idio+Fund96+Fund144+Fund168 (6-way triple funding):")
    best_e = None
    best_e_avg = 0.0

    for v1w in [30, 35]:
        for srw in [0, 5]:
            for idiow in [15, 20]:
                for f96w in [5, 10]:
                    for f144w in [10, 15]:
                        f168w = 100 - v1w - srw - idiow - f96w - f144w
                        if f168w < 10 or f168w > 30:
                            continue
                        w = {
                            "v1": v1w/100, "sr": srw/100, "idio": idiow/100,
                            "f96": f96w/100, "f144": f144w/100, "f168": f168w/100,
                        }
                        r = blend_ensemble(base_strats_e, w)
                        if r["pos"] == 5:
                            tag = " *** NEW CHAMPION ***" if r["avg"] > phase66_champ else ""
                            if r["avg"] > phase66_champ or r["avg"] > 1.66:
                                print(f"    V1={v1w}/SR={srw}/Idio={idiow}/"
                                      f"F96={f96w}/F144={f144w}/F168={f168w}: "
                                      f"AVG={r['avg']}, MIN={r['min']}, pos={r['pos']}/5{tag}")
                            if r["avg"] > best_e_avg:
                                best_e_avg = r["avg"]
                                best_e = (v1w, srw, idiow, f96w, f144w, f168w, r)

    if best_e:
        v1w, srw, idiow, f96w, f144w, f168w, r = best_e
        print(f"\n  BEST E: V1={v1w}/SR={srw}/Idio={idiow}/"
              f"F96={f96w}/F144={f144w}/F168={f168w}: AVG={r['avg']}, MIN={r['min']}")
        print(f"  Year-by-year: {r['yby']}")

    # ── F: Fine-grained grid search on best D configuration ─────────────────
    print("\n" + "═" * 80)
    print("  F: FINE-GRAINED GRID SEARCH (best D config ± 5%)")
    print("  Searching V1+SR+Idio+Fund144+Fund168 with 5% steps")
    print("═" * 80)

    best_f = None
    best_f_avg = 0.0

    for v1w in range(25, 50, 5):
        for srw in range(0, 20, 5):
            for idiow in range(10, 35, 5):
                for f144w in range(10, 30, 5):
                    f168w = 100 - v1w - srw - idiow - f144w
                    if f168w < 10 or f168w > 40:
                        continue
                    w = {
                        "v1": v1w/100, "sr": srw/100, "idio": idiow/100,
                        "f144": f144w/100, "f168": f168w/100,
                    }
                    r = blend_ensemble(base_strats_d, w)
                    if r["pos"] == 5 and r["avg"] > best_f_avg:
                        best_f_avg = r["avg"]
                        best_f = (v1w, srw, idiow, f144w, f168w, r)

    if best_f:
        v1w, srw, idiow, f144w, f168w, r = best_f
        print(f"\n  BEST AVG (F): V1={v1w}/SR={srw}/Idio={idiow}/F144={f144w}/F168={f168w}: "
              f"AVG={r['avg']}, MIN={r['min']}")
        print(f"  Year-by-year: {r['yby']}")

    # Find best MIN too
    best_f_min = None
    best_f_min_val = 0.0

    for v1w in range(25, 50, 5):
        for srw in range(0, 20, 5):
            for idiow in range(10, 35, 5):
                for f144w in range(10, 30, 5):
                    f168w = 100 - v1w - srw - idiow - f144w
                    if f168w < 10 or f168w > 40:
                        continue
                    w = {
                        "v1": v1w/100, "sr": srw/100, "idio": idiow/100,
                        "f144": f144w/100, "f168": f168w/100,
                    }
                    r = blend_ensemble(base_strats_d, w)
                    if r["pos"] == 5 and r["min"] > best_f_min_val:
                        best_f_min_val = r["min"]
                        best_f_min = (v1w, srw, idiow, f144w, f168w, r)

    if best_f_min:
        v1w, srw, idiow, f144w, f168w, r = best_f_min
        print(f"  BEST MIN (F): V1={v1w}/SR={srw}/Idio={idiow}/F144={f144w}/F168={f168w}: "
              f"AVG={r['avg']}, MIN={r['min']}")

    # ── Summary & champion ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  PHASE 67 FINAL CHAMPION")
    print("=" * 80)

    candidates = []
    if best_c:
        v1w, srw, idiow, f144w, r = best_c
        candidates.append((r["avg"], r["min"], r["pos"], r["yby"],
                            f"V1+SR+Idio+F144 {v1w}/{srw}/{idiow}/{f144w}",
                            "4-way-f144",
                            {"v1": v1w/100, "sr": srw/100, "idio": idiow/100, "f144": f144w/100}))
    if best_d:
        v1w, srw, idiow, f144w, f168w, r = best_d
        candidates.append((r["avg"], r["min"], r["pos"], r["yby"],
                            f"V1+SR+Idio+F144+F168 {v1w}/{srw}/{idiow}/{f144w}/{f168w}",
                            "5-way-f144f168",
                            {"v1": v1w/100, "sr": srw/100, "idio": idiow/100,
                             "f144": f144w/100, "f168": f168w/100}))
    if best_e:
        v1w, srw, idiow, f96w, f144w, f168w, r = best_e
        candidates.append((r["avg"], r["min"], r["pos"], r["yby"],
                            f"V1+SR+Idio+F96+F144+F168 {v1w}/{srw}/{idiow}/{f96w}/{f144w}/{f168w}",
                            "6-way-triple",
                            {"v1": v1w/100, "sr": srw/100, "idio": idiow/100,
                             "f96": f96w/100, "f144": f144w/100, "f168": f168w/100}))
    if best_f:
        v1w, srw, idiow, f144w, f168w, r = best_f
        candidates.append((r["avg"], r["min"], r["pos"], r["yby"],
                            f"V1+SR+Idio+F144+F168(fine) {v1w}/{srw}/{idiow}/{f144w}/{f168w}",
                            "5-way-f144f168-fine",
                            {"v1": v1w/100, "sr": srw/100, "idio": idiow/100,
                             "f144": f144w/100, "f168": f168w/100}))

    if candidates:
        champion = max(candidates, key=lambda x: x[0])
        avg, mn, pos, yby, name, tag, weights = champion
        print(f"\n  Champion AVG: {avg}, MIN: {mn}, pos: {pos}/5")
        print(f"  Config: {name}")
        print(f"  Year-by-year: {yby}")

        if avg > phase66_champ:
            print(f"\n  *** BEATS Phase 66 champion ({phase66_champ}) ***")
        else:
            print(f"\n  Note: Phase 66 champion ({phase66_champ}) not beaten — F96+F168 remains best")

        # Build sub-strategy list for saving
        strat_map = {
            "v1": ("nexus_alpha_v1", V1_PARAMS),
            "sr": ("sharpe_ratio_alpha", SR_437_PARAMS),
            "idio": ("idio_momentum_alpha", IDIO_437_PARAMS),
            "f96": ("funding_momentum_alpha", FUND_96_PARAMS),
            "f144": ("funding_momentum_alpha", FUND_144_PARAMS),
            "f168": ("funding_momentum_alpha", FUND_168_PARAMS),
        }
        sub_strats = []
        for key, wt in weights.items():
            if wt > 0 and key in strat_map:
                sname, sparams = strat_map[key]
                sub_strats.append({
                    "name": sname,
                    "params": sparams,
                    "weight": wt,
                })
        weight_str = "".join(str(int(v * 100)) for v in weights.values())
        config_name = f"ensemble_v1sr_idio_{tag}_{weight_str}"
        save_champion_config(config_name, f"Phase 67 champion: {name} AVG={avg}", sub_strats)

    # Save results
    results = {
        "phase": 67,
        "fund_sweep": {
            lb: {"avg": fund_sweep_results[lb]["_avg"], "min": fund_sweep_results[lb]["_min"]}
            for lb in fund_sweep_lbs
        },
        "best_lb": best_lb,
        "correlations": {
            "fund144_fund168": pearson_corr(r144, r168),
            "fund144_fund96": pearson_corr(r144, r96),
            "fund168_fund96": pearson_corr(r168, r96),
            "fund144_v1": pearson_corr(r144, rv1),
            "fund144_idio": pearson_corr(r144, ridio),
        },
        "phase66_champ": phase66_champ,
        "best_c_avg": best_c_avg if best_c else 0.0,
        "best_d_avg": best_d_avg if best_d else 0.0,
        "best_e_avg": best_e_avg if best_e else 0.0,
        "best_f_avg": best_f_avg if best_f else 0.0,
    }
    out_path = f"{OUT_DIR}/phase67_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")
    print("\n  SIGNAL HIERARCHY (standalone AVG Sharpe):")
    fund_avgs = {lb: fund_sweep_results[lb]["_avg"] for lb in fund_sweep_lbs}
    print(f"    Idio_437:      AVG={idio_data['_avg']}")
    print(f"    SR_437:        AVG={sr_data['_avg']}")
    for lb, avg in sorted(fund_avgs.items(), key=lambda x: -x[1]):
        print(f"    Fund_{lb:3d}h:    AVG={avg}")
    print(f"    V1-Long:       AVG={v1_data['_avg']}")


if __name__ == "__main__":
    main()
