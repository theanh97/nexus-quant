#!/usr/bin/env python3
"""
Phase 77: k=3 I437 Fine Grid + F160 + bw336 Combinations

Phase 76 discoveries:
  NEW BALANCED CHAMPION: k=3 for I437 + Phase 74 weights
    V1=17.5%, I437_bw168_k3=12.5%, I600_bw168_k2=22.5%, F144=47.5%, F168=0%
    AVG=1.914, MIN=1.191 — strictly dominates Phase 73 balanced (1.909/1.170)
  k=3 for I437 + Phase 73 weights: AVG=1.923, MIN=1.153 (higher AVG, lower MIN)
  F160 standalone: AVG=1.252, 2022=0.970 (best 2022 in funding family, beats F144 2022=0.496)
  k=3 for BOTH idio (P73 weights): AVG=1.916, MIN=1.036

Phase 77 directions:
  A. Save Phase 76 k=3 champion config (missed in Phase 76)
  B. Fine 2.5% grid around k=3 I437 (P74-style, F168=0)
     — Also test 5-signal grid adding small F168 allocation
  C. F160 + k=3 I437: F160's 2022=0.970 may push MIN floor up
     — Grid over F144/F160 splits with k=3 I437 at P74 weight structure
  D. bw336 + k=3 I437: bw336 gives higher AVG standalone;
     does k=3 diversification help the MIN (Phase 75 bw336 had MIN=0.972/1.166)?
  E. k=3 for I600 at P74 weights: Phase 76 only tested P73 (1.916/1.036)
  F. Summary: update champion table
"""

import json
import math
import statistics
import subprocess
import sys
import copy
from pathlib import Path

OUT_DIR = "artifacts/phase77"
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

FUND_144_PARAMS = {
    "k_per_side": 2, "funding_lookback_bars": 144, "direction": "contrarian",
    "vol_lookback_bars": 168, "target_gross_leverage": 0.25, "rebalance_interval_bars": 24,
}

FUND_160_PARAMS = {
    "k_per_side": 2, "funding_lookback_bars": 160, "direction": "contrarian",
    "vol_lookback_bars": 168, "target_gross_leverage": 0.25, "rebalance_interval_bars": 24,
}

FUND_168_PARAMS = {
    "k_per_side": 2, "funding_lookback_bars": 168, "direction": "contrarian",
    "vol_lookback_bars": 168, "target_gross_leverage": 0.25, "rebalance_interval_bars": 24,
}

def make_idio_params(lookback: int, beta_window: int, k: int = 2) -> dict:
    return {
        "k_per_side": k, "lookback_bars": lookback, "beta_window_bars": beta_window,
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
    path = f"/tmp/phase77_{run_name}_{year}.json"
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
    yby = [round(year_results.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]
    year_results["_avg"] = avg
    year_results["_min"] = mn
    year_results["_pos"] = pos
    print(f"  → AVG={avg}, MIN={mn}, {pos}/5 positive", flush=True)
    print(f"  → YbY: {yby}", flush=True)
    return year_results


def blend_ensemble(strategies: dict, weights: dict) -> dict:
    year_sharpes = {}
    for year in YEARS:
        pairs = []
        min_len = None
        valid = True
        for name, data in strategies.items():
            w = weights.get(name, 0.0)
            if w == 0.0:
                continue
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
        blended = [sum(w * arr[i] for w, arr in pairs) for i in range(min_len)]
        mu = statistics.mean(blended)
        sd = statistics.pstdev(blended)
        s = round((mu / sd) * math.sqrt(8760), 3) if sd > 0 else 0.0
        year_sharpes[year] = s

    valid_s = [s for s in year_sharpes.values() if s is not None]
    return {
        "avg": round(sum(valid_s) / len(valid_s), 3) if valid_s else 0.0,
        "min": round(min(valid_s), 3) if valid_s else 0.0,
        "pos": sum(1 for s in valid_s if s > 0),
        "yby": [year_sharpes.get(y) for y in YEARS],
    }


def save_champion(config_name: str, comment: str, strategy_list: list) -> None:
    cfg = {
        "_comment": comment,
        "run_name": config_name,
        **{k: v for k, v in BASE_CONFIG.items()},
        "_strategies": strategy_list,
    }
    with open(f"configs/{config_name}.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  Saved: configs/{config_name}.json", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PHASE 77: k=3 I437 Fine Grid + F160 + bw336 Combinations")
print("=" * 70)

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# Prior champions
P73 = {"avg": 1.909, "min": 1.170}  # V1(15)+I437_k2(15)+I600_k2(20)+F144(45)+F168(5)
P74 = {"avg": 1.902, "min": 1.199}  # V1(17.5)+I437_k2(12.5)+I600_k2(22.5)+F144(47.5)+F168(0)
P75_AVGMAX = {"avg": 1.934, "min": 0.972}  # V1(12.5)+I437_bw336(20)+I600_bw168(17.5)+F144(40)+F168(10)
P76_BALANCED = {"avg": 1.914, "min": 1.191}  # V1(17.5)+I437_bw168_k3(12.5)+I600_bw168(22.5)+F144(47.5)

# ─── SECTION A: Save Phase 76 champion config ─────────────────────────────────
print("\n" + "═" * 70)
print("SECTION A: Saving Phase 76 k=3 balanced champion config")
print("═" * 70)

save_champion(
    "ensemble_p76_k3i437_balanced",
    "Phase 76 balanced champion: V1(17.5%)+I437_bw168_k3(12.5%)+I600_bw168_k2(22.5%)+F144(47.5%). AVG=1.914, MIN=1.191",
    [
        {
            "name": "nexus_alpha_v1",
            "params": V1_PARAMS,
            "weight": 0.175,
        },
        {
            "name": "idio_momentum_alpha",
            "params": make_idio_params(437, 168, k=3),
            "weight": 0.125,
        },
        {
            "name": "idio_momentum_alpha",
            "params": make_idio_params(600, 168, k=2),
            "weight": 0.225,
        },
        {
            "name": "funding_momentum_alpha",
            "params": FUND_144_PARAMS,
            "weight": 0.475,
        },
    ],
)

# ─── SECTION B: Reference strategies ─────────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION B: Reference strategies")
print("═" * 70)

print("  Running V1 reference...", flush=True)
v1_data = run_strategy("v1_ref", "nexus_alpha_v1", V1_PARAMS)

print("  Running I437_bw168_k3 (Phase 76 balanced champion core)...", flush=True)
i437_k3 = run_strategy("idio_lb437_bw168_k3", "idio_momentum_alpha", make_idio_params(437, 168, k=3))

print("  Running I437_bw168_k2 (reference)...", flush=True)
i437_k2 = run_strategy("idio_lb437_bw168", "idio_momentum_alpha", make_idio_params(437, 168, k=2))

print("  Running I437_bw336_k3 (bw336 + k3 combo)...", flush=True)
i437_bw336_k3 = run_strategy("idio_lb437_bw336_k3", "idio_momentum_alpha", make_idio_params(437, 336, k=3))

print("  Running I600_bw168_k2 (reference)...", flush=True)
i600_k2 = run_strategy("idio_lb600_bw168", "idio_momentum_alpha", make_idio_params(600, 168, k=2))

print("  Running I600_bw168_k3...", flush=True)
i600_k3 = run_strategy("idio_lb600_bw168_k3", "idio_momentum_alpha", make_idio_params(600, 168, k=3))

print("  Running F144 reference...", flush=True)
f144_data = run_strategy("fund_144_ref", "funding_momentum_alpha", FUND_144_PARAMS)

print("  Running F160 reference...", flush=True)
f160_data = run_strategy("fund_160_ref", "funding_momentum_alpha", FUND_160_PARAMS)

print("  Running F168 reference...", flush=True)
f168_data = run_strategy("fund_168_ref", "funding_momentum_alpha", FUND_168_PARAMS)

# Confirm Phase 76 champion baseline
strats_p76_balanced = {
    "v1": v1_data, "idio_437_k3": i437_k3, "idio_600": i600_k2, "f144": f144_data,
}
p76_champ_verify = blend_ensemble(strats_p76_balanced, {"v1": 0.175, "idio_437_k3": 0.125, "idio_600": 0.225, "f144": 0.475})
print(f"\n  Phase 76 champion verification: AVG={p76_champ_verify['avg']}, MIN={p76_champ_verify['min']}")
print(f"  YbY: {p76_champ_verify['yby']}")

strats_p73_k3 = {
    "v1": v1_data, "idio_437_k3": i437_k3, "idio_600": i600_k2, "f144": f144_data, "f168": f168_data,
}
p73_k3_verify = blend_ensemble(strats_p73_k3, {"v1": 0.15, "idio_437_k3": 0.15, "idio_600": 0.20, "f144": 0.45, "f168": 0.05})
print(f"  Phase 73 weights + k3 I437: AVG={p73_k3_verify['avg']}, MIN={p73_k3_verify['min']}")
print(f"  YbY: {p73_k3_verify['yby']}")

# ─── SECTION C: Fine 2.5% grid around k=3 I437 ─────────────────────────────
print("\n" + "═" * 70)
print("SECTION C: Fine 2.5% grid for k=3 I437 + k2 I600")
print("  Center: V1=17.5, I437_k3=12.5, I600_k2=22.5, F144=47.5, F168=0")
print("  Goal: find AVG>1.914 AND MIN>1.191 simultaneously")
print("═" * 70)

STEP = 2.5
strats_c = {
    "v1": v1_data, "idio_437_k3": i437_k3, "idio_600": i600_k2,
    "f144": f144_data, "f168": f168_data,
}

grid_results_c = []
v1_vals  = [x / 100 for x in range(1000, 2750, 250)]   # 10% to 27.5%
i437_vals = [x / 100 for x in range(750, 2000, 250)]   # 7.5% to 19.5%
i600_vals = [x / 100 for x in range(1500, 3250, 250)]  # 15% to 32.5%
f168_vals = [0.0, 0.025, 0.05]                         # 0%, 2.5%, 5%

total_tested = 0
for wv1 in v1_vals:
    for wi437 in i437_vals:
        for wi600 in i600_vals:
            for wf168 in f168_vals:
                wf144 = round(1.0 - wv1 - wi437 - wi600 - wf168, 4)
                if wf144 < 0.35 or wf144 > 0.575:
                    continue
                if wv1 + wi437 + wi600 + wf168 + wf144 > 1.001:
                    continue
                wts = {"v1": wv1, "idio_437_k3": wi437, "idio_600": wi600,
                       "f144": wf144, "f168": wf168}
                r = blend_ensemble(strats_c, wts)
                grid_results_c.append((wv1, wi437, wi600, wf144, wf168, r))
                total_tested += 1

print(f"\n  Tested {total_tested} configurations")

# Top 10 by AVG
grid_results_c.sort(key=lambda x: x[5]["avg"], reverse=True)
print(f"\n  Top 10 by AVG (k=3 I437, 2.5% grid):")
print(f"       V1   I437_k3  I600   F144   F168      AVG      MIN")
print(f"  {'─'*60}")
for wv1, wi437, wi600, wf144, wf168, r in grid_results_c[:10]:
    print(f"    {wv1*100:5.1f}   {wi437*100:5.1f}  {wi600*100:5.1f}  {wf144*100:5.1f}  {wf168*100:4.1f}   {r['avg']:.3f}   {r['min']:.3f}")

# Best MIN with AVG > 1.900
min_filtered = [(x, r) for *x, r in [(*row[:5], row[5]) for row in grid_results_c]
                if r["avg"] > 1.900]
min_filtered.sort(key=lambda x: x[1]["min"], reverse=True)
print(f"\n  Best MIN (AVG > 1.900):")
print(f"       V1   I437_k3  I600   F144   F168      AVG      MIN")
print(f"  {'─'*60}")
for (wv1, wi437, wi600, wf144, wf168), r in min_filtered[:10]:
    yby_str = [round(v, 3) if v is not None else None for v in r["yby"]]
    print(f"    {wv1*100:5.1f}   {wi437*100:5.1f}  {wi600*100:5.1f}  {wf144*100:5.1f}  {wf168*100:4.1f}   {r['avg']:.3f}   {r['min']:.3f}  YbY:{yby_str}")

# Check for dominant configurations (AVG > P76 AND MIN > P76)
dominant = [(x, r) for x, r in min_filtered if r["avg"] > P76_BALANCED["avg"] and r["min"] > P76_BALANCED["min"]]
if dominant:
    print(f"\n  ★ STRICTLY DOMINANT (AVG>{P76_BALANCED['avg']} AND MIN>{P76_BALANCED['min']}):")
    for (wv1, wi437, wi600, wf144, wf168), r in dominant[:5]:
        print(f"    V1={wv1*100:.1f}%, I437_k3={wi437*100:.1f}%, I600={wi600*100:.1f}%, F144={wf144*100:.1f}%, F168={wf168*100:.1f}%")
        print(f"    AVG={r['avg']}, MIN={r['min']}, YbY={r['yby']}")
else:
    print(f"\n  No config strictly dominates Phase 76 (AVG={P76_BALANCED['avg']}, MIN={P76_BALANCED['min']})")

# ─── SECTION D: F160 + k=3 I437 ─────────────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION D: F160 + k=3 I437 combinations")
print(f"  F160 standalone: AVG={f160_data['_avg']}, MIN={f160_data['_min']}")
yby_f160 = [round(f160_data.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]
print(f"  YbY: {yby_f160}  ← 2022={yby_f160[1]} is key advantage")
print("═" * 70)

strats_d = {
    "v1": v1_data, "idio_437_k3": i437_k3, "idio_600": i600_k2,
    "f144": f144_data, "f160": f160_data, "f168": f168_data,
}

# Phase 76 champion + F160 replacing F168=0 → small F160 allocation
r_p76_f160_5 = blend_ensemble(strats_d, {"v1": 0.175, "idio_437_k3": 0.125, "idio_600": 0.225, "f144": 0.450, "f160": 0.025})
print(f"\n  P76-k3 + F160(2.5%): AVG={r_p76_f160_5['avg']}, MIN={r_p76_f160_5['min']}")
r_p76_f160_10 = blend_ensemble(strats_d, {"v1": 0.175, "idio_437_k3": 0.125, "idio_600": 0.225, "f144": 0.425, "f160": 0.05})
print(f"  P76-k3 + F160(5%):   AVG={r_p76_f160_10['avg']}, MIN={r_p76_f160_10['min']}")

# Grid over F144/F160 ratio (P74 total funding ~47.5%, k=3 I437)
print(f"\n  F144/F160 grid (V1=17.5, I437_k3=12.5, I600=22.5, F_total varies):")
print(f"       F144   F160      AVG      MIN   YbY")
print(f"  {'─'*60}")
f160_grid = []
for f_total_pct in [45, 47.5, 50]:
    for f144_frac in [0.6, 0.7, 0.8, 0.9, 1.0]:
        wf144 = round(f_total_pct * f144_frac / 100, 4)
        wf160 = round(f_total_pct * (1 - f144_frac) / 100, 4)
        wrest = round(0.175 + 0.125 + 0.225, 4)
        if abs(wrest + wf144 + wf160 - 1.0) > 0.005:
            continue
        wts = {"v1": 0.175, "idio_437_k3": 0.125, "idio_600": 0.225,
               "f144": wf144, "f160": wf160}
        r = blend_ensemble(strats_d, wts)
        f160_grid.append((wf144, wf160, r))

f160_grid.sort(key=lambda x: x[2]["avg"], reverse=True)
for wf144, wf160, r in f160_grid:
    yby_str = [round(v, 3) if v is not None else None for v in r["yby"]]
    print(f"    {wf144*100:5.1f}   {wf160*100:4.1f}   {r['avg']:.3f}   {r['min']:.3f}  {yby_str}")

# Phase 73 weights + k=3 I437 + F160 replacing F168
r_p73_k3_f160 = blend_ensemble(strats_d, {"v1": 0.15, "idio_437_k3": 0.15, "idio_600": 0.20, "f144": 0.45, "f160": 0.05})
print(f"\n  P73 weights + k3 I437 + F160(5%): AVG={r_p73_k3_f160['avg']}, MIN={r_p73_k3_f160['min']}")
print(f"  YbY: {r_p73_k3_f160['yby']}")

# ─── SECTION E: bw336 + k=3 I437 ─────────────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION E: bw336 + k=3 I437 combinations")
print(f"  I437_bw336_k3 standalone: AVG={i437_bw336_k3['_avg']}, MIN={i437_bw336_k3['_min']}")
yby_bw336_k3 = [round(i437_bw336_k3.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]
print(f"  YbY: {yby_bw336_k3}")
yby_bw168_k3 = [round(i437_k3.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]
print(f"  I437_bw168_k3 standalone: AVG={i437_k3['_avg']}, MIN={i437_k3['_min']}")
print(f"  YbY: {yby_bw168_k3}")
print("═" * 70)

# Replace I437_bw168_k3 with I437_bw336_k3 in P76 balanced config
strats_e = {
    "v1": v1_data, "idio_437_bw336_k3": i437_bw336_k3, "idio_600": i600_k2,
    "f144": f144_data, "f168": f168_data,
}
# P74 weights with bw336_k3
r_bw336_k3_p74 = blend_ensemble(strats_e, {"v1": 0.175, "idio_437_bw336_k3": 0.125, "idio_600": 0.225, "f144": 0.475})
print(f"\n  bw336_k3 at P74 weights: AVG={r_bw336_k3_p74['avg']}, MIN={r_bw336_k3_p74['min']}")
print(f"  YbY: {r_bw336_k3_p74['yby']}")

# P73 weights with bw336_k3
r_bw336_k3_p73 = blend_ensemble(strats_e, {"v1": 0.15, "idio_437_bw336_k3": 0.15, "idio_600": 0.20, "f144": 0.45, "f168": 0.05})
print(f"  bw336_k3 at P73 weights: AVG={r_bw336_k3_p73['avg']}, MIN={r_bw336_k3_p73['min']}")
print(f"  YbY: {r_bw336_k3_p73['yby']}")

# Quick grid: bw336_k3 AVG-max direction (P75 had V1=12.5, I437_bw336=20, I600_bw168=17.5, F144=40, F168=10)
strats_e2 = {
    "v1": v1_data, "idio_437_bw336_k3": i437_bw336_k3, "idio_600": i600_k2,
    "f144": f144_data, "f168": f168_data,
}
r_bw336_k3_avgmax = blend_ensemble(strats_e2, {"v1": 0.125, "idio_437_bw336_k3": 0.20, "idio_600": 0.175, "f144": 0.40, "f168": 0.10})
print(f"\n  bw336_k3 at P75 AVG-max weights: AVG={r_bw336_k3_avgmax['avg']}, MIN={r_bw336_k3_avgmax['min']}")
print(f"  YbY: {r_bw336_k3_avgmax['yby']}")

# ─── SECTION F: k=3 I600 at P74 weights ──────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION F: k=3 for I600 at P74 weights")
print(f"  I600_bw168_k3 standalone: AVG={i600_k3['_avg']}, MIN={i600_k3['_min']}")
yby_i600_k3 = [round(i600_k3.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]
print(f"  YbY: {yby_i600_k3}")
print("═" * 70)

strats_f = {
    "v1": v1_data, "idio_437_k3": i437_k3, "idio_600_k3": i600_k3,
    "f144": f144_data, "f168": f168_data,
}
# Phase 74 weights with both k=3
r_both_k3_p74 = blend_ensemble(strats_f, {"v1": 0.175, "idio_437_k3": 0.125, "idio_600_k3": 0.225, "f144": 0.475})
print(f"\n  k=3 BOTH idio at P74 weights: AVG={r_both_k3_p74['avg']}, MIN={r_both_k3_p74['min']}")
print(f"  YbY: {r_both_k3_p74['yby']}")

# Phase 73 weights with both k=3 (confirmed in Phase 76: 1.916/1.036)
r_both_k3_p73 = blend_ensemble(strats_f, {"v1": 0.15, "idio_437_k3": 0.15, "idio_600_k3": 0.20, "f144": 0.45, "f168": 0.05})
print(f"  k=3 BOTH idio at P73 weights: AVG={r_both_k3_p73['avg']}, MIN={r_both_k3_p73['min']}")
print(f"  YbY: {r_both_k3_p73['yby']}")

# k=3 I437 + k2 I600 vs k2 I437 + k3 I600 (asymmetric)
strats_asym = {
    "v1": v1_data, "idio_437_k2": i437_k2, "idio_600_k3": i600_k3,
    "f144": f144_data, "f168": f168_data,
}
r_k2_437_k3_600_p74 = blend_ensemble(strats_asym, {"v1": 0.175, "idio_437_k2": 0.125, "idio_600_k3": 0.225, "f144": 0.475})
print(f"\n  k2 I437 + k3 I600 at P74 weights: AVG={r_k2_437_k3_600_p74['avg']}, MIN={r_k2_437_k3_600_p74['min']}")
print(f"  YbY: {r_k2_437_k3_600_p74['yby']}")

# ─── SECTION G: Save champions and final summary ──────────────────────────────
print("\n" + "═" * 70)
print("SECTION G: Phase 77 Summary")
print("═" * 70)

# Collect all results for champion comparison
all_candidates = []

# Section C: best from fine grid
if grid_results_c:
    top_avg_c = grid_results_c[0]
    all_candidates.append(("C:grid_avg_max", top_avg_c[5], top_avg_c[:5]))
    if min_filtered:
        top_min_c = min_filtered[0]
        all_candidates.append(("C:grid_min_max", top_min_c[1], top_min_c[0]))

# Section D: F160 combos
all_candidates.append(("D:p76_f160_25", r_p76_f160_5, None))
all_candidates.append(("D:p73_k3_f160", r_p73_k3_f160, None))
if f160_grid:
    top_f160 = f160_grid[0]
    all_candidates.append(("D:f160_grid_top", top_f160[2], None))

# Section E: bw336 + k3
all_candidates.append(("E:bw336_k3_p74", r_bw336_k3_p74, None))
all_candidates.append(("E:bw336_k3_p73", r_bw336_k3_p73, None))
all_candidates.append(("E:bw336_k3_avgmax", r_bw336_k3_avgmax, None))

# Section F: k3 for I600
all_candidates.append(("F:both_k3_p74", r_both_k3_p74, None))
all_candidates.append(("F:k2i437_k3i600_p74", r_k2_437_k3_600_p74, None))

print(f"\n  Champion landscape:")
print(f"    Phase 73 sweet spot: AVG={P73['avg']}, MIN={P73['min']} (V1+I437_k2+I600_k2+F144+F168)")
print(f"    Phase 74 balanced:   AVG={P74['avg']}, MIN={P74['min']} (V1+I437_k2+I600_k2+F144)")
print(f"    Phase 75 AVG-max:    AVG={P75_AVGMAX['avg']}, MIN={P75_AVGMAX['min']} (V1+I437_bw336+I600+F144+F168)")
print(f"    Phase 76 balanced:   AVG={P76_BALANCED['avg']}, MIN={P76_BALANCED['min']} (V1+I437_k3+I600_k2+F144)")
print(f"\n  Phase 77 candidates:")
print(f"  {'Label':<25}  {'AVG':>7}  {'MIN':>7}")
print(f"  {'─'*45}")
for label, r, _ in sorted(all_candidates, key=lambda x: x[1]["avg"], reverse=True):
    flag = ""
    if r["avg"] > P76_BALANCED["avg"] and r["min"] > P76_BALANCED["min"]:
        flag = " ★ DOMINANT"
    elif r["avg"] > P75_AVGMAX["avg"]:
        flag = " ★ AVG-MAX"
    print(f"  {label:<25}  {r['avg']:>7.3f}  {r['min']:>7.3f}{flag}")

# Save any new champions
new_balanced = None
new_avgmax = None

# Check for new balanced champion (dominates P76 balanced)
for label, r, wts_tuple in all_candidates:
    if r["avg"] > P76_BALANCED["avg"] and r["min"] > P76_BALANCED["min"]:
        if new_balanced is None or r["min"] > new_balanced[1]["min"]:
            new_balanced = (label, r, wts_tuple)
    if r["avg"] > P75_AVGMAX["avg"]:
        if new_avgmax is None or r["avg"] > new_avgmax[1]["avg"]:
            new_avgmax = (label, r, wts_tuple)

# Save best from fine grid if it's a champion
if min_filtered and len(min_filtered) > 0:
    (wv1, wi437, wi600, wf144, wf168), r_best = min_filtered[0]
    yby_best = [round(v, 3) if v is not None else None for v in r_best["yby"]]
    print(f"\n  Best balanced from Section C grid:")
    print(f"    V1={wv1*100:.1f}%, I437_k3={wi437*100:.1f}%, I600={wi600*100:.1f}%, F144={wf144*100:.1f}%, F168={wf168*100:.1f}%")
    print(f"    AVG={r_best['avg']}, MIN={r_best['min']}, YbY={yby_best}")

    if r_best["avg"] > P76_BALANCED["avg"] and r_best["min"] > P76_BALANCED["min"]:
        save_champion(
            "ensemble_p77_k3i437_balanced",
            f"Phase 77 balanced champion: V1({wv1*100:.1f}%)+I437_bw168_k3({wi437*100:.1f}%)+I600_bw168_k2({wi600*100:.1f}%)+F144({wf144*100:.1f}%)+F168({wf168*100:.1f}%). AVG={r_best['avg']}, MIN={r_best['min']}",
            [
                {"name": "nexus_alpha_v1", "params": V1_PARAMS, "weight": wv1},
                {"name": "idio_momentum_alpha", "params": make_idio_params(437, 168, k=3), "weight": wi437},
                {"name": "idio_momentum_alpha", "params": make_idio_params(600, 168, k=2), "weight": wi600},
                {"name": "funding_momentum_alpha", "params": FUND_144_PARAMS, "weight": wf144},
                {"name": "funding_momentum_alpha", "params": FUND_168_PARAMS, "weight": wf168},
            ],
        )
    elif r_best["min"] > P76_BALANCED["min"]:
        # Save as "high MIN" variant even if AVG doesn't dominate
        save_champion(
            "ensemble_p77_k3i437_highmin",
            f"Phase 77 high-MIN variant: V1({wv1*100:.1f}%)+I437_bw168_k3({wi437*100:.1f}%)+I600_bw168_k2({wi600*100:.1f}%)+F144({wf144*100:.1f}%)+F168({wf168*100:.1f}%). AVG={r_best['avg']}, MIN={r_best['min']}",
            [
                {"name": "nexus_alpha_v1", "params": V1_PARAMS, "weight": wv1},
                {"name": "idio_momentum_alpha", "params": make_idio_params(437, 168, k=3), "weight": wi437},
                {"name": "idio_momentum_alpha", "params": make_idio_params(600, 168, k=2), "weight": wi600},
                {"name": "funding_momentum_alpha", "params": FUND_144_PARAMS, "weight": wf144},
                {"name": "funding_momentum_alpha", "params": FUND_168_PARAMS, "weight": wf168},
            ],
        )

# Check if bw336_k3 broke AVG-max record
if r_bw336_k3_avgmax["avg"] > P75_AVGMAX["avg"]:
    print(f"\n  ★ NEW AVG-MAX: bw336_k3 at P75 weights: AVG={r_bw336_k3_avgmax['avg']}, MIN={r_bw336_k3_avgmax['min']}")
    save_champion(
        "ensemble_p77_bw336k3_avgmax",
        f"Phase 77 AVG-max: V1(12.5%)+I437_bw336_k3(20%)+I600_bw168_k2(17.5%)+F144(40%)+F168(10%). AVG={r_bw336_k3_avgmax['avg']}, MIN={r_bw336_k3_avgmax['min']}",
        [
            {"name": "nexus_alpha_v1", "params": V1_PARAMS, "weight": 0.125},
            {"name": "idio_momentum_alpha", "params": make_idio_params(437, 336, k=3), "weight": 0.20},
            {"name": "idio_momentum_alpha", "params": make_idio_params(600, 168, k=2), "weight": 0.175},
            {"name": "funding_momentum_alpha", "params": FUND_144_PARAMS, "weight": 0.40},
            {"name": "funding_momentum_alpha", "params": FUND_168_PARAMS, "weight": 0.10},
        ],
    )

print("\n" + "=" * 70)
print("PHASE 77 COMPLETE")
print("=" * 70)
