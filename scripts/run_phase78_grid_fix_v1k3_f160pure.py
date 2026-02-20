#!/usr/bin/env python3
"""
Phase 78: Fixed Fine Grid + V1 k=3 + F160 Pure Replacement

Phase 77 findings:
  - Phase 76 champion VERIFIED: V1(17.5%)+I437_bw168_k3(12.5%)+I600_bw168_k2(22.5%)+F144(47.5%)
    AVG=1.914, MIN=1.191 — YbY: [3.483, 1.206, 1.191, 2.49, 1.199]
  - F160 as additive: trades MIN for AVG (not worth it for balanced goal)
  - bw336+k3 in ensemble: weaker MIN than bw168+k3
  - k3 for I600: destroys 2022 MIN (drops to 1.051)
  - Section C grid bug: integer range produced percentages not fractions → 0 configs tested

Phase 78 directions:
  A. FIXED fine 2.5% grid around P76 champion (fractions: 0.175, 0.125, etc.)
     - 4-signal (F168=0 like P76) and 5-signal (add small F168)
     - Goal: find strict improvement over P76 (AVG>1.914 AND MIN>1.191)
  B. k=3 for V1 strategy (nexus_alpha_v1): never tested
     - V1 standalone with k_per_side=3
     - V1_k3 in ensemble at various weights
  C. F160 as PRIMARY funding signal (replace F144 entirely)
     - F144 2022=0.496; F160 2022=0.970 → huge 2022 improvement
     - Does F160-dominant ensemble beat F144-dominant for MIN?
     - Grid: V1+I437_k3+I600+F160 (no F144)
  D. Between P73 and P76: fine grid in the AVG/MIN trade-off gap
     - P73 weights+k3: V1=15, I437_k3=15, I600=20, F144=45, F168=5 → 1.923/1.153
     - P76 weights+k3: V1=17.5, I437_k3=12.5, I600=22.5, F144=47.5, F168=0 → 1.914/1.191
     - Is there a point with AVG between these AND MIN better than 1.191?
"""

import json
import math
import statistics
import subprocess
import sys
import copy
from pathlib import Path

OUT_DIR = "artifacts/phase78"
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

V1_PARAMS_K2 = {
    "k_per_side": 2,
    "w_carry": 0.35, "w_mom": 0.45, "w_confirm": 0.0,
    "w_mean_reversion": 0.20, "w_vol_momentum": 0.0, "w_funding_trend": 0.0,
    "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 72,
    "vol_lookback_bars": 168,
    "target_portfolio_vol": 0.0, "use_min_variance": False,
    "target_gross_leverage": 0.35, "min_gross_leverage": 0.05, "max_gross_leverage": 0.65,
    "rebalance_interval_bars": 60, "strict_agreement": False,
}

V1_PARAMS_K3 = {
    **V1_PARAMS_K2,
    "k_per_side": 3,
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
    path = f"/tmp/phase78_{run_name}_{year}.json"
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


# Helper: generate 2.5% step grid values as fractions
def pct_range(lo_pct: int, hi_pct_exclusive: int, step_pct: int = 250) -> list:
    """Generate weight fractions from lo_pct% to hi_pct_exclusive% in step_pct/10 % steps.
    All values in tenths of a percent: step_pct=250 → 2.5% steps.
    lo_pct, hi_pct_exclusive in units of 10ths of a percent (e.g., 1000 = 10%, 2750 = 27.5%).
    """
    return [x / 10000 for x in range(lo_pct, hi_pct_exclusive, step_pct)]


# ═════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PHASE 78: Fixed Fine Grid + V1 k=3 + F160 Pure Replacement")
print("=" * 70)

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

P73 = {"avg": 1.909, "min": 1.170}
P74 = {"avg": 1.902, "min": 1.199}
P75_AVGMAX = {"avg": 1.934, "min": 0.972}
P76_BALANCED = {"avg": 1.914, "min": 1.191}  # V1(17.5%)+I437_bw168_k3(12.5%)+I600_bw168_k2(22.5%)+F144(47.5%)

# ─── SECTION A: Fixed 2.5% grid ───────────────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION A: FIXED fine 2.5% grid around P76 champion")
print("  Bug fix: weights now correctly in fractions (0.175 not 17.5)")
print("  Center: V1=0.175, I437_k3=0.125, I600=0.225, F144=0.475, F168=0.0")
print("═" * 70)

print("  Running reference strategies...", flush=True)
print("  Running V1...", flush=True)
v1_data = run_strategy("v1_ref", "nexus_alpha_v1", V1_PARAMS_K2)
print("  Running I437_bw168_k3...", flush=True)
i437_k3 = run_strategy("idio_lb437_bw168_k3", "idio_momentum_alpha", make_idio_params(437, 168, k=3))
print("  Running I600_bw168_k2...", flush=True)
i600_k2 = run_strategy("idio_lb600_bw168", "idio_momentum_alpha", make_idio_params(600, 168, k=2))
print("  Running F144...", flush=True)
f144_data = run_strategy("fund_144_ref", "funding_momentum_alpha", FUND_144_PARAMS)
print("  Running F168...", flush=True)
f168_data = run_strategy("fund_168_ref", "funding_momentum_alpha", FUND_168_PARAMS)
print("  Running F160...", flush=True)
f160_data = run_strategy("fund_160_ref", "funding_momentum_alpha", FUND_160_PARAMS)

# Verify P76 champion baseline
strats_base = {"v1": v1_data, "idio_437_k3": i437_k3, "idio_600": i600_k2,
               "f144": f144_data, "f168": f168_data}
p76_verify = blend_ensemble(strats_base, {"v1": 0.175, "idio_437_k3": 0.125, "idio_600": 0.225, "f144": 0.475})
print(f"\n  P76 champion verify: AVG={p76_verify['avg']}, MIN={p76_verify['min']}")
print(f"  YbY: {p76_verify['yby']}")

# Fixed grid: weights as proper fractions
# V1 in [10%, 25%], I437_k3 in [7.5%, 20%], I600 in [15%, 32.5%]
# F168 in [0%, 5%], F144 = 1 - sum
v1_vals   = pct_range(1000, 2750, 250)   # 10% to 25% in 2.5% steps
i437_vals = pct_range(750, 2000, 250)    # 7.5% to 17.5% in 2.5% steps
i600_vals = pct_range(1500, 3250, 250)   # 15% to 32.5% in 2.5% steps
f168_vals = [0.0, 0.025, 0.05]          # 0%, 2.5%, 5%

grid_results = []
total_tested = 0

for wv1 in v1_vals:
    for wi437 in i437_vals:
        for wi600 in i600_vals:
            for wf168 in f168_vals:
                wf144 = round(1.0 - wv1 - wi437 - wi600 - wf168, 4)
                if wf144 < 0.35 or wf144 > 0.575:
                    continue
                wts = {"v1": wv1, "idio_437_k3": wi437, "idio_600": wi600,
                       "f144": wf144, "f168": wf168}
                r = blend_ensemble(strats_base, wts)
                grid_results.append((wv1, wi437, wi600, wf144, wf168, r))
                total_tested += 1

print(f"\n  Tested {total_tested} configurations")

grid_results.sort(key=lambda x: x[5]["avg"], reverse=True)
print(f"\n  Top 15 by AVG:")
print(f"       V1%  I437_k3%  I600%  F144%  F168%    AVG     MIN")
print(f"  {'─'*65}")
for row in grid_results[:15]:
    wv1, wi437, wi600, wf144, wf168, r = row
    print(f"    {wv1*100:4.1f}  {wi437*100:5.1f}  {wi600*100:5.1f}  {wf144*100:5.1f}  {wf168*100:4.1f}  {r['avg']:.3f}  {r['min']:.3f}")

# Best MIN with AVG > 1.900
high_avg = [(row, row[5]) for row in grid_results if row[5]["avg"] > 1.900]
high_avg_sorted_by_min = sorted(high_avg, key=lambda x: x[1]["min"], reverse=True)
print(f"\n  Best MIN where AVG > 1.900 ({len(high_avg)} configs):")
print(f"       V1%  I437_k3%  I600%  F144%  F168%    AVG     MIN   YbY")
print(f"  {'─'*80}")
for (wv1, wi437, wi600, wf144, wf168, r), _ in high_avg_sorted_by_min[:10]:
    yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
    print(f"    {wv1*100:4.1f}  {wi437*100:5.1f}  {wi600*100:5.1f}  {wf144*100:5.1f}  {wf168*100:4.1f}  {r['avg']:.3f}  {r['min']:.3f}  {yby_r}")

# Check for dominance over P76
dominant = [(wv1, wi437, wi600, wf144, wf168, r)
            for wv1, wi437, wi600, wf144, wf168, r in grid_results
            if r["avg"] > P76_BALANCED["avg"] and r["min"] > P76_BALANCED["min"]]
if dominant:
    print(f"\n  ★ STRICTLY DOMINANT over P76 ({len(dominant)} configs):")
    for wv1, wi437, wi600, wf144, wf168, r in dominant[:5]:
        yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
        print(f"    V1={wv1*100:.1f}%, I437_k3={wi437*100:.1f}%, I600={wi600*100:.1f}%, F144={wf144*100:.1f}%, F168={wf168*100:.1f}%")
        print(f"    AVG={r['avg']}, MIN={r['min']}, YbY={yby_r}")
else:
    print(f"\n  No config strictly dominates P76 (AVG={P76_BALANCED['avg']}, MIN={P76_BALANCED['min']})")

# ─── SECTION B: V1 k=3 ────────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION B: V1 k=3 (nexus_alpha_v1 with k_per_side=3) — NEVER TESTED")
print("═" * 70)

v1_k3_data = run_strategy("v1_k3_ref", "nexus_alpha_v1", V1_PARAMS_K3)

yby_v1k2 = [round(v1_data.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]
yby_v1k3 = [round(v1_k3_data.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]
print(f"\n  V1 k=2: AVG={v1_data['_avg']}, MIN={v1_data['_min']}, YbY={yby_v1k2}")
print(f"  V1 k=3: AVG={v1_k3_data['_avg']}, MIN={v1_k3_data['_min']}, YbY={yby_v1k3}")

# V1_k3 in P76 champion configuration
strats_v1k3 = {"v1_k3": v1_k3_data, "idio_437_k3": i437_k3, "idio_600": i600_k2,
               "f144": f144_data, "f168": f168_data}
r_v1k3_p76 = blend_ensemble(strats_v1k3, {"v1_k3": 0.175, "idio_437_k3": 0.125, "idio_600": 0.225, "f144": 0.475})
print(f"\n  V1_k3 + I437_k3 at P76 weights: AVG={r_v1k3_p76['avg']}, MIN={r_v1k3_p76['min']}")
print(f"  YbY: {r_v1k3_p76['yby']}")

r_v1k3_p73 = blend_ensemble(strats_v1k3, {"v1_k3": 0.15, "idio_437_k3": 0.15, "idio_600": 0.20, "f144": 0.45, "f168": 0.05})
print(f"  V1_k3 + I437_k3 at P73 weights: AVG={r_v1k3_p73['avg']}, MIN={r_v1k3_p73['min']}")
print(f"  YbY: {r_v1k3_p73['yby']}")

# V1_k3 with original k=2 idio (to isolate V1 k3 effect)
strats_v1k3_idio_k2 = {"v1_k3": v1_k3_data, "idio_437_k2": run_strategy("idio_lb437_bw168", "idio_momentum_alpha", make_idio_params(437, 168, k=2)) if False else None}
# Just use i437 from base
strats_v1k3_combo = {"v1_k3": v1_k3_data, "idio_437_k2": {"2021": {"returns": i437_k3.get("2021", {}).get("returns", [])}}  }
# Simpler: use strats_base + replace v1 with v1_k3
strats_base_v1k3 = {"v1": v1_k3_data, "idio_437_k3": i437_k3, "idio_600": i600_k2,
                    "f144": f144_data, "f168": f168_data}

# Run a quick grid for V1_k3
print(f"\n  Quick grid: V1_k3 weight sensitivity at P76-style:")
v1k3_grid = []
for wv1 in [0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25]:
    wf144 = round(1.0 - wv1 - 0.125 - 0.225, 4)
    if wf144 < 0.35 or wf144 > 0.575:
        continue
    r = blend_ensemble(strats_v1k3, {"v1_k3": wv1, "idio_437_k3": 0.125, "idio_600": 0.225, "f144": wf144})
    v1k3_grid.append((wv1, r))
    yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
    print(f"    V1_k3={wv1*100:.1f}%, I437_k3=12.5%, I600=22.5%, F144={wf144*100:.1f}%: AVG={r['avg']}, MIN={r['min']}  {yby_r}")

# ─── SECTION C: F160 as PRIMARY funding signal ────────────────────────────────
print("\n" + "═" * 70)
print("SECTION C: F160 as PRIMARY funding signal (replace F144 entirely)")
print(f"  F160 standalone: AVG={f160_data['_avg']}, MIN={f160_data['_min']}")
yby_f160 = [round(f160_data.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]
yby_f144 = [round(f144_data.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]
print(f"  YbY F160: {yby_f160}  ← 2022={yby_f160[1]}")
print(f"  YbY F144: {yby_f144}  ← 2022={yby_f144[1]}")
print(f"  F160 has 2022={yby_f160[1]} vs F144={yby_f144[1]} (+{yby_f160[1]-yby_f144[1]:.3f})")
print("═" * 70)

strats_f160_only = {"v1": v1_data, "idio_437_k3": i437_k3, "idio_600": i600_k2, "f160": f160_data}

# Direct replacement: use same absolute weight as F144 (47.5%)
r_f160_only_p76 = blend_ensemble(strats_f160_only, {"v1": 0.175, "idio_437_k3": 0.125, "idio_600": 0.225, "f160": 0.475})
print(f"\n  F160 only at P76 weights (47.5%): AVG={r_f160_only_p76['avg']}, MIN={r_f160_only_p76['min']}")
print(f"  YbY: {r_f160_only_p76['yby']}")

# Grid: V1, I437_k3, I600, F160 (4-signal, no F144)
print(f"\n  4-signal grid (F160 only, no F144):")
print(f"       V1%  I437_k3%  I600%  F160%    AVG     MIN   YbY")
print(f"  {'─'*70}")
f160_grid_results = []
for wv1 in pct_range(1000, 2750, 250):
    for wi437 in pct_range(750, 2000, 250):
        for wi600 in pct_range(1500, 3250, 250):
            wf160 = round(1.0 - wv1 - wi437 - wi600, 4)
            if wf160 < 0.35 or wf160 > 0.575:
                continue
            wts = {"v1": wv1, "idio_437_k3": wi437, "idio_600": wi600, "f160": wf160}
            r = blend_ensemble(strats_f160_only, wts)
            f160_grid_results.append((wv1, wi437, wi600, wf160, r))

f160_grid_results.sort(key=lambda x: x[4]["avg"], reverse=True)
print(f"  Tested {len(f160_grid_results)} F160-only configurations")
print(f"\n  Top 10 by AVG:")
for wv1, wi437, wi600, wf160, r in f160_grid_results[:10]:
    yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
    print(f"    {wv1*100:4.1f}  {wi437*100:5.1f}  {wi600*100:5.1f}  {wf160*100:5.1f}  {r['avg']:.3f}  {r['min']:.3f}  {yby_r}")

# Best MIN with AVG > 1.850 (lower bar for F160-only since it's a new config space)
f160_high_avg = [(row, row[4]) for row in f160_grid_results if row[4]["avg"] > 1.850]
f160_high_avg_by_min = sorted(f160_high_avg, key=lambda x: x[1]["min"], reverse=True)
print(f"\n  Best MIN (F160-only, AVG > 1.850, {len(f160_high_avg)} configs):")
for (wv1, wi437, wi600, wf160, r), _ in f160_high_avg_by_min[:5]:
    yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
    print(f"    V1={wv1*100:.1f}%, I437_k3={wi437*100:.1f}%, I600={wi600*100:.1f}%, F160={wf160*100:.1f}%: AVG={r['avg']}, MIN={r['min']}  {yby_r}")

# ─── SECTION D: Between P73 and P76 — gap exploration ────────────────────────
print("\n" + "═" * 70)
print("SECTION D: P73→P76 gradient — find AVG>1.914 AND MIN>1.191")
print("  P73+k3: V1=15%, I437_k3=15%, I600=20%, F144=45%, F168=5% → 1.923/1.153")
print("  P76:    V1=17.5%, I437_k3=12.5%, I600=22.5%, F144=47.5%, F168=0% → 1.914/1.191")
print("  Sweep between them: does any point dominate P76 on both metrics?")
print("═" * 70)

# Interpolate between P73 and P76 weights in 10 steps
# P73 weights: v1=0.15, i437=0.15, i600=0.20, f144=0.45, f168=0.05
# P76 weights: v1=0.175, i437=0.125, i600=0.225, f144=0.475, f168=0.0
p73_wts = {"v1": 0.15, "idio_437_k3": 0.15, "idio_600": 0.20, "f144": 0.45, "f168": 0.05}
p76_wts = {"v1": 0.175, "idio_437_k3": 0.125, "idio_600": 0.225, "f144": 0.475, "f168": 0.0}

print(f"\n  Linear interpolation from P73+k3 to P76+k3:")
print(f"      α    V1%  I437%  I600%  F144%  F168%    AVG     MIN   YbY")
print(f"  {'─'*75}")
interp_results = []
for i in range(11):
    alpha = i / 10.0  # 0 = P73+k3, 1 = P76
    wts_interp = {k: round((1 - alpha) * p73_wts[k] + alpha * p76_wts.get(k, 0.0), 4)
                  for k in p73_wts}
    # Round to nearest 0.25% step to avoid floating point issues
    wts_interp = {k: round(v * 1000) / 1000 for k, v in wts_interp.items()}
    r = blend_ensemble(strats_base, wts_interp)
    interp_results.append((alpha, wts_interp, r))
    yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
    print(f"    {alpha:.1f}  {wts_interp['v1']*100:4.1f}  {wts_interp['idio_437_k3']*100:5.1f}  {wts_interp['idio_600']*100:5.1f}  {wts_interp['f144']*100:5.1f}  {wts_interp['f168']*100:4.1f}  {r['avg']:.3f}  {r['min']:.3f}  {yby_r}")

# ─── SECTION E: Grand Summary ─────────────────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION E: Phase 78 Summary")
print("═" * 70)

print(f"\n  Champion landscape entering Phase 78:")
print(f"    Phase 75 AVG-max:  AVG=1.934, MIN=0.972")
print(f"    Phase 76 balanced: AVG=1.914, MIN=1.191 ← current best balanced")
print(f"    Phase 73 sweet:    AVG=1.909, MIN=1.170")
print(f"    Phase 74 balanced: AVG=1.902, MIN=1.199")

print(f"\n  Section A (Fixed grid) — top config: ", end="")
if grid_results:
    top_a = grid_results[0]
    print(f"AVG={top_a[5]['avg']}, MIN={top_a[5]['min']}")
    print(f"    V1={top_a[0]*100:.1f}%, I437_k3={top_a[1]*100:.1f}%, I600={top_a[2]*100:.1f}%, F144={top_a[3]*100:.1f}%, F168={top_a[4]*100:.1f}%")
else:
    print("N/A")

print(f"\n  Section B (V1 k=3) — V1_k3 standalone: AVG={v1_k3_data['_avg']}, MIN={v1_k3_data['_min']}")
if v1k3_grid:
    best_v1k3 = max(v1k3_grid, key=lambda x: x[1]["avg"])
    print(f"    Best in ensemble: V1_k3={best_v1k3[0]*100:.1f}% → AVG={best_v1k3[1]['avg']}, MIN={best_v1k3[1]['min']}")

print(f"\n  Section C (F160 pure) — best at P76 weights: AVG={r_f160_only_p76['avg']}, MIN={r_f160_only_p76['min']}")
if f160_grid_results:
    best_f160 = f160_grid_results[0]
    print(f"    Best F160-only grid: AVG={best_f160[4]['avg']}, MIN={best_f160[4]['min']}")
    print(f"    V1={best_f160[0]*100:.1f}%, I437_k3={best_f160[1]*100:.1f}%, I600={best_f160[2]*100:.1f}%, F160={best_f160[3]*100:.1f}%")

print(f"\n  Section D (P73→P76 gradient): best MIN near 1.191?")
best_interp = max(interp_results, key=lambda x: x[2]["min"])
print(f"    α={best_interp[0]:.1f}: AVG={best_interp[2]['avg']}, MIN={best_interp[2]['min']}")

# Save new champions
new_saved = []

if grid_results:
    (wv1, wi437, wi600, wf144, wf168, r_best) = high_avg_sorted_by_min[0][0] if high_avg_sorted_by_min else grid_results[0]
    if r_best["avg"] > P76_BALANCED["avg"] and r_best["min"] > P76_BALANCED["min"]:
        name = "ensemble_p78_k3i437_balanced"
        save_champion(
            name,
            f"Phase 78 balanced champion: V1({wv1*100:.1f}%)+I437_bw168_k3({wi437*100:.1f}%)+I600({wi600*100:.1f}%)+F144({wf144*100:.1f}%)+F168({wf168*100:.1f}%). AVG={r_best['avg']}, MIN={r_best['min']}",
            [
                {"name": "nexus_alpha_v1", "params": V1_PARAMS_K2, "weight": wv1},
                {"name": "idio_momentum_alpha", "params": make_idio_params(437, 168, k=3), "weight": wi437},
                {"name": "idio_momentum_alpha", "params": make_idio_params(600, 168, k=2), "weight": wi600},
                {"name": "funding_momentum_alpha", "params": FUND_144_PARAMS, "weight": wf144},
                {"name": "funding_momentum_alpha", "params": FUND_168_PARAMS, "weight": wf168},
            ],
        )
        new_saved.append(name)

if not new_saved:
    print(f"\n  No config strictly dominates P76 balanced champion.")
    print(f"  Phase 76 champion (AVG=1.914, MIN=1.191) remains the best balanced config.")

print("\n" + "=" * 70)
print("PHASE 78 COMPLETE")
print("=" * 70)
