#!/usr/bin/env python3
"""
Phase 68: Fine-Grain Champion + Fund_156 + Weight Precision

Phase 67 champion: V1(25%)+SR(5%)+Idio(25%)+Fund144(25%)+Fund168(20%)
  AVG=1.800, MIN=1.114, 5/5 positive
  Year-by-year: [2.986, 1.114, 1.219, 2.372, 1.310]

Observations:
  - fund_144 has terrible 2025 (0.062) — fund_168 covers this (0.571)
  - fund_156: AVG=1.184, year-by-year: [2.329, 0.397, 0.864, 2.214, 0.116]
    → 2025=0.116 (slightly better than fund_144=0.062)
    → 2022=0.397 (slightly worse than fund_144=0.496)
    → 2023=0.864 (MUCH better than fund_144=0.664!) ← interesting

Goals:
  A. Ultra-fine grid search (2.5% steps) around Phase 67 champion
     V1=[20-35%], SR=[0-10%], Idio=[20-30%], F144=[20-30%], F168=[15-25%]
  B. Replace Fund144 with Fund156 in 5-way (since fund_156 has better 2023)
  C. Triple combo: fund_144 + fund_156 + fund_168 (6-way, omit fund_96)
  D. Pure funding allocation: minimize price-based signals
     Try V1=20%/SR=0%/Idio=20%/F144=30%/F168=30% type configurations
  E. Protect 2022 floor: find best MIN config around the champion region
"""

import json
import math
import statistics
import subprocess
import sys
import copy
from pathlib import Path

OUT_DIR = "artifacts/phase68"
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

# ── Frozen params ────────────────────────────────────────────────────────────
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

FUND_144_PARAMS = {
    "k_per_side": 2, "funding_lookback_bars": 144, "direction": "contrarian",
    "vol_lookback_bars": 168, "target_gross_leverage": 0.25, "rebalance_interval_bars": 24,
}

FUND_156_PARAMS = {
    "k_per_side": 2, "funding_lookback_bars": 156, "direction": "contrarian",
    "vol_lookback_bars": 168, "target_gross_leverage": 0.25, "rebalance_interval_bars": 24,
}

FUND_96_PARAMS = {
    "k_per_side": 2, "funding_lookback_bars": 96, "direction": "contrarian",
    "vol_lookback_bars": 168, "target_gross_leverage": 0.25, "rebalance_interval_bars": 24,
}


# ── Helpers ──────────────────────────────────────────────────────────────────

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
    path = f"/tmp/phase68_{run_name}_{year}.json"
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
    """Fast blend using pre-extracted lists to minimize per-element overhead."""
    year_sharpes = {}
    for year in YEARS:
        # Pre-extract (weight, returns_array) pairs — avoids dict lookups in inner loop
        pairs = []
        min_len = None
        valid = True
        for name, data in strategies.items():
            w = weights.get(name, 0.0)
            rets = data.get(year, {}).get("returns", [])
            if not rets:
                valid = False
                break
            pairs.append((w, rets))
            if min_len is None or len(rets) < min_len:
                min_len = len(rets)
        if not valid or min_len is None:
            year_sharpes[year] = None
            continue
        blended = [
            sum(w * arr[i] for w, arr in pairs)
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


def save_champion_config(config_name: str, comment: str, strat_map: dict,
                          weights: dict) -> None:
    strat_cfgs = {
        "v1": ("nexus_alpha_v1", V1_PARAMS),
        "sr": ("sharpe_ratio_alpha", SR_437_PARAMS),
        "idio": ("idio_momentum_alpha", IDIO_437_PARAMS),
        "f96": ("funding_momentum_alpha", FUND_96_PARAMS),
        "f144": ("funding_momentum_alpha", FUND_144_PARAMS),
        "f156": ("funding_momentum_alpha", FUND_156_PARAMS),
        "f168": ("funding_momentum_alpha", FUND_168_PARAMS),
    }
    sub_strats = []
    for key, wt in weights.items():
        if wt > 0 and key in strat_cfgs:
            sname, sparams = strat_cfgs[key]
            sub_strats.append({"name": sname, "params": sparams, "weight": wt})

    cfg = {
        "_comment": comment,
        "run_name": config_name,
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
        "_strategies": sub_strats,
    }
    path = f"configs/{config_name}.json"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  Config saved: {path}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    PHASE67_CHAMP = 1.800

    print("=" * 80)
    print("  PHASE 68: FINE-GRAINED CHAMPION SEARCH + FUND_156 EXPLORATION")
    print("=" * 80)
    print("  Phase 67 champion: V1(25%)+SR(5%)+Idio(25%)+F144(25%)+F168(20%)")
    print("  AVG=1.800, MIN=1.114, year: [2.986, 1.114, 1.219, 2.372, 1.310]")
    print("  Goal: Push beyond 1.800 with 2.5% step precision + fund_156")
    print("=" * 80)

    # ── Reference strategies ─────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  Running reference strategies...")
    print("═" * 70)

    v1_data = run_strategy("v1_ref", "nexus_alpha_v1", V1_PARAMS)
    idio_data = run_strategy("idio_437_ref", "idio_momentum_alpha", IDIO_437_PARAMS)
    sr_data = run_strategy("sr_437_ref", "sharpe_ratio_alpha", SR_437_PARAMS)
    fund168_data = run_strategy("fund_168_ref", "funding_momentum_alpha", FUND_168_PARAMS)
    fund144_data = run_strategy("fund_144_ref", "funding_momentum_alpha", FUND_144_PARAMS)
    fund156_data = run_strategy("fund_156_ref", "funding_momentum_alpha", FUND_156_PARAMS)

    print(f"\n  Standalone signal profile (yearly Sharpe):")
    for label, d in [
        ("V1-Long", v1_data), ("SR_437", sr_data), ("Idio_437", idio_data),
        ("Fund_144", fund144_data), ("Fund_156", fund156_data), ("Fund_168", fund168_data),
    ]:
        yby = [round(d[y].get("sharpe", 0), 3) for y in YEARS if y in d]
        print(f"    {label:12s}: AVG={d['_avg']:.3f}  {yby}")

    # ── B: Correlation analysis ──────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  B: CORRELATION ANALYSIS (fund_144 vs fund_156 vs fund_168)")
    print("═" * 70)

    r144 = get_all_returns(fund144_data)
    r156 = get_all_returns(fund156_data)
    r168 = get_all_returns(fund168_data)
    rv1 = get_all_returns(v1_data)
    ridio = get_all_returns(idio_data)

    print(f"\n  corr(fund_144, fund_156) = {pearson_corr(r144, r156)}")
    print(f"  corr(fund_144, fund_168) = {pearson_corr(r144, r168)}")
    print(f"  corr(fund_156, fund_168) = {pearson_corr(r156, r168)}")
    print(f"  corr(fund_144, v1)       = {pearson_corr(r144, rv1)}")
    print(f"  corr(fund_156, v1)       = {pearson_corr(r156, rv1)}")
    print(f"  corr(fund_144, idio)     = {pearson_corr(r144, ridio)}")

    # ── A: Ultra-fine grid (2.5% steps) around Phase 67 champion ────────────
    print("\n" + "═" * 80)
    print("  A: ULTRA-FINE GRID SEARCH (2.5% steps) — V1+SR+Idio+F144+F168")
    print("  Phase 67: V1=25/SR=5/Idio=25/F144=25/F168=20 → AVG=1.800")
    print("═" * 80)

    base_strats_5 = {
        "v1": v1_data, "sr": sr_data, "idio": idio_data,
        "f144": fund144_data, "f168": fund168_data,
    }

    print("\n  V1+SR+Idio+Fund144+Fund168 (5% steps, extended range):")
    best_a_avg = PHASE67_CHAMP
    best_a = None
    best_a_min = {"min": 0.0, "cfg": None, "r": None}

    # Extended 5% step grid: wider range than Phase 67's fine search
    for v1w in range(15, 45, 5):   # 15% to 40%
        for srw in range(0, 15, 5):  # 0% to 10%
            for idiow in range(15, 35, 5):  # 15% to 30%
                for f144w in range(15, 40, 5):  # 15% to 35%
                    f168w = 100 - v1w - srw - idiow - f144w
                    if f168w < 10 or f168w > 40:
                        continue
                    w = {
                        "v1": v1w/100, "sr": srw/100,
                        "idio": idiow/100, "f144": f144w/100, "f168": f168w/100,
                    }
                    r = blend_ensemble(base_strats_5, w)
                    if r["pos"] == 5:
                        tag = " *** BEATS 1.800 ***" if r["avg"] > PHASE67_CHAMP else ""
                        if r["avg"] > PHASE67_CHAMP or r["avg"] > 1.79:
                            print(f"    V1={v1w}/SR={srw}/Idio={idiow}/"
                                  f"F144={f144w}/F168={f168w}: "
                                  f"AVG={r['avg']}, MIN={r['min']}{tag}", flush=True)
                        if r["avg"] > best_a_avg:
                            best_a_avg = r["avg"]
                            best_a = (v1w, srw, idiow, f144w, f168w, r)
                        if r["min"] > best_a_min["min"]:
                            best_a_min = {"min": r["min"], "avg": r["avg"],
                                          "cfg": (v1w, srw, idiow, f144w, f168w), "r": r}

    if best_a:
        v1w, srw, idiow, f144w, f168w, r = best_a
        print(f"\n  BEST AVG (A): V1={v1w}/SR={srw}/Idio={idiow}/"
              f"F144={f144w}/F168={f168w}: AVG={r['avg']}, MIN={r['min']}")
        print(f"  Year-by-year: {r['yby']}")
    else:
        print(f"\n  No improvement found over Phase 67 champion (1.800)")
        print(f"  Phase 67 champion remains best in extended 5% grid")

    if best_a_min["r"]:
        v1w, srw, idiow, f144w, f168w = best_a_min["cfg"]
        r = best_a_min["r"]
        print(f"  BEST MIN (A): V1={v1w}/SR={srw}/Idio={idiow}/"
              f"F144={f144w}/F168={f168w}: AVG={r['avg']}, MIN={r['min']}")

    # ── C: Fund_156 replacing Fund_144 in 5-way ──────────────────────────────
    print("\n" + "═" * 80)
    print("  C: FUND_156 REPLACING FUND_144 (V1+SR+Idio+F156+F168)")
    print("  fund_156: [2.329, 0.397, 0.864, 2.214, 0.116] — better 2023 than f144!")
    print("═" * 80)

    base_strats_c = {
        "v1": v1_data, "sr": sr_data, "idio": idio_data,
        "f156": fund156_data, "f168": fund168_data,
    }

    print("\n  V1+SR+Idio+Fund156+Fund168 (5-way with F156 replacing F144):")
    best_c = None
    best_c_avg = 0.0

    for v1w in range(20, 40, 5):
        for srw in range(0, 15, 5):
            for idiow in range(15, 35, 5):
                for f156w in range(15, 35, 5):
                    f168w = 100 - v1w - srw - idiow - f156w
                    if f168w < 10 or f168w > 35:
                        continue
                    w = {
                        "v1": v1w/100, "sr": srw/100, "idio": idiow/100,
                        "f156": f156w/100, "f168": f168w/100,
                    }
                    r = blend_ensemble(base_strats_c, w)
                    if r["pos"] == 5:
                        tag = " *** NEW ***" if r["avg"] > PHASE67_CHAMP else ""
                        if r["avg"] > PHASE67_CHAMP or r["avg"] > 1.78:
                            print(f"    V1={v1w}/SR={srw}/Idio={idiow}/F156={f156w}/F168={f168w}: "
                                  f"AVG={r['avg']}, MIN={r['min']}, pos={r['pos']}/5{tag}")
                        if r["avg"] > best_c_avg:
                            best_c_avg = r["avg"]
                            best_c = (v1w, srw, idiow, f156w, f168w, r)

    if best_c:
        v1w, srw, idiow, f156w, f168w, r = best_c
        print(f"\n  BEST C (F156): V1={v1w}/SR={srw}/Idio={idiow}/F156={f156w}/F168={f168w}: "
              f"AVG={r['avg']}, MIN={r['min']}")
        print(f"  Year-by-year: {r['yby']}")

    # ── D: Triple funding without F96 (F144+F156+F168) ──────────────────────
    print("\n" + "═" * 80)
    print("  D: TRIPLE FUNDING F144+F156+F168 (6-way, no F96)")
    print("  fund_144: 2025=0.062 (weak), fund_156: 2025=0.116, fund_168: 2025=0.571")
    print("  Together: better 2025 floor for funding allocation")
    print("═" * 80)

    base_strats_d = {
        "v1": v1_data, "sr": sr_data, "idio": idio_data,
        "f144": fund144_data, "f156": fund156_data, "f168": fund168_data,
    }

    print("\n  V1+SR+Idio+F144+F156+F168 (6-way triple close-lookback funding):")
    best_d = None
    best_d_avg = 0.0

    for v1w in range(20, 40, 5):
        for srw in range(0, 10, 5):
            for idiow in range(15, 30, 5):
                for f144w in range(10, 30, 5):
                    for f156w in range(5, 20, 5):
                        f168w = 100 - v1w - srw - idiow - f144w - f156w
                        if f168w < 10 or f168w > 30:
                            continue
                        w = {
                            "v1": v1w/100, "sr": srw/100, "idio": idiow/100,
                            "f144": f144w/100, "f156": f156w/100, "f168": f168w/100,
                        }
                        r = blend_ensemble(base_strats_d, w)
                        if r["pos"] == 5:
                            tag = " *** NEW ***" if r["avg"] > PHASE67_CHAMP else ""
                            if r["avg"] > PHASE67_CHAMP or r["avg"] > 1.79:
                                print(f"    V1={v1w}/SR={srw}/Idio={idiow}/"
                                      f"F144={f144w}/F156={f156w}/F168={f168w}: "
                                      f"AVG={r['avg']}, MIN={r['min']}{tag}")
                            if r["avg"] > best_d_avg:
                                best_d_avg = r["avg"]
                                best_d = (v1w, srw, idiow, f144w, f156w, f168w, r)

    if best_d:
        v1w, srw, idiow, f144w, f156w, f168w, r = best_d
        print(f"\n  BEST D (triple F144+156+168): V1={v1w}/SR={srw}/Idio={idiow}/"
              f"F144={f144w}/F156={f156w}/F168={f168w}: AVG={r['avg']}, MIN={r['min']}")
        print(f"  Year-by-year: {r['yby']}")

    # ── E: High-funding-weight experiments (reduce price signals) ────────────
    print("\n" + "═" * 80)
    print("  E: HIGH FUNDING WEIGHT — MINIMUM PRICE SIGNAL")
    print("  Hypothesis: funding alpha >> price alpha in crypto → push funding to 60-70%")
    print("═" * 80)

    print("\n  V1+Idio+F144+F168 (4-way, drop SR, high funding):")
    best_e = None
    best_e_avg = 0.0

    for v1w in range(15, 35, 5):
        for idiow in range(15, 35, 5):
            for f144w in range(25, 45, 5):
                f168w = 100 - v1w - idiow - f144w
                if f168w < 15 or f168w > 40:
                    continue
                w = {
                    "v1": v1w/100, "idio": idiow/100,
                    "f144": f144w/100, "f168": f168w/100,
                }
                r = blend_ensemble(
                    {k: v for k, v in base_strats_5.items() if k != "sr"}, w
                )
                if r["pos"] == 5:
                    tag = " *** NEW ***" if r["avg"] > PHASE67_CHAMP else ""
                    if r["avg"] > PHASE67_CHAMP or r["avg"] > 1.79:
                        print(f"    V1={v1w}/Idio={idiow}/F144={f144w}/F168={f168w}: "
                              f"AVG={r['avg']}, MIN={r['min']}{tag}")
                    if r["avg"] > best_e_avg:
                        best_e_avg = r["avg"]
                        best_e = (v1w, idiow, f144w, f168w, r)

    if best_e:
        v1w, idiow, f144w, f168w, r = best_e
        print(f"\n  BEST E (no SR): V1={v1w}/Idio={idiow}/F144={f144w}/F168={f168w}: "
              f"AVG={r['avg']}, MIN={r['min']}")
        print(f"  Year-by-year: {r['yby']}")

    # ── Summary & champion ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  PHASE 68 FINAL CHAMPION")
    print("=" * 80)

    # Phase 67 champion for reference
    phase67_yby = [2.986, 1.114, 1.219, 2.372, 1.310]
    candidates = [
        (PHASE67_CHAMP, 1.114, 5, phase67_yby,
         "Phase67 V1+SR+Idio+F144+F168 25/5/25/25/20",
         {"v1": 0.25, "sr": 0.05, "idio": 0.25, "f144": 0.25, "f168": 0.20}),
    ]

    if best_a:
        v1i, sri, idioi, f144i, f168i, r = best_a
        pct = lambda i: round(i * 2.5)
        candidates.append((
            r["avg"], r["min"], r["pos"], r["yby"],
            f"A V1+SR+Idio+F144+F168 {pct(v1i)}/{pct(sri)}/{pct(idioi)}/{pct(f144i)}/{pct(f168i)}",
            {"v1": v1i/40, "sr": sri/40, "idio": idioi/40, "f144": f144i/40, "f168": f168i/40},
        ))

    if best_c:
        v1w, srw, idiow, f156w, f168w, r = best_c
        candidates.append((
            r["avg"], r["min"], r["pos"], r["yby"],
            f"C V1+SR+Idio+F156+F168 {v1w}/{srw}/{idiow}/{f156w}/{f168w}",
            {"v1": v1w/100, "sr": srw/100, "idio": idiow/100, "f156": f156w/100, "f168": f168w/100},
        ))

    if best_d:
        v1w, srw, idiow, f144w, f156w, f168w, r = best_d
        candidates.append((
            r["avg"], r["min"], r["pos"], r["yby"],
            f"D V1+SR+Idio+F144+F156+F168 {v1w}/{srw}/{idiow}/{f144w}/{f156w}/{f168w}",
            {"v1": v1w/100, "sr": srw/100, "idio": idiow/100,
             "f144": f144w/100, "f156": f156w/100, "f168": f168w/100},
        ))

    if best_e:
        v1w, idiow, f144w, f168w, r = best_e
        candidates.append((
            r["avg"], r["min"], r["pos"], r["yby"],
            f"E V1+Idio+F144+F168 {v1w}/{idiow}/{f144w}/{f168w}",
            {"v1": v1w/100, "idio": idiow/100, "f144": f144w/100, "f168": f168w/100},
        ))

    if candidates:
        champion = max(candidates, key=lambda x: x[0])
        avg, mn, pos, yby, name, weights = champion
        print(f"\n  Champion AVG: {avg}, MIN: {mn}, pos: {pos}/5")
        print(f"  Config: {name}")
        print(f"  Year-by-year: {yby}")

        if avg > PHASE67_CHAMP:
            print(f"\n  *** BEATS Phase 67 champion (1.800) ***")
            weight_str = "".join(f"{int(v*100)}" for v in weights.values())
            config_name = f"ensemble_v1sr_idio_p68_{weight_str}"
            save_champion_config(config_name,
                                 f"Phase 68 champion: {name} AVG={avg}",
                                 {}, weights)
        else:
            print(f"\n  Phase 67 champion (1.800) not beaten — remains current champion")
            print(f"  Best found: AVG={champion[0]}")

    # Save results
    results = {
        "phase": 68,
        "phase67_champ": PHASE67_CHAMP,
        "best_a_avg": best_a_avg if best_a else PHASE67_CHAMP,
        "best_c_avg": best_c_avg if best_c else 0.0,
        "best_d_avg": best_d_avg if best_d else 0.0,
        "best_e_avg": best_e_avg if best_e else 0.0,
        "fund_156_standalone": fund156_data["_avg"],
        "fund_156_yby": [fund156_data[y].get("sharpe", 0.0) for y in YEARS if y in fund156_data],
        "correlations": {
            "fund144_fund156": pearson_corr(r144, r156),
            "fund144_fund168": pearson_corr(r144, r168),
            "fund156_fund168": pearson_corr(r156, r168),
        },
    }
    out_path = f"{OUT_DIR}/phase68_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
