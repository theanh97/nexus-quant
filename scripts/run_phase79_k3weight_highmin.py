#!/usr/bin/env python3
"""
Phase 79: I437_k3 Weight Extension + High-MIN Frontier

Phase 78 discoveries:
  NEW BALANCED CHAMPION: V1(17.5%)+I437_bw168_k3(17.5%)+I600_bw168(20%)+F144(45%)+F168(0%)
    AVG=1.919, MIN=1.206, YbY=[3.446, 1.206, 1.207, 2.489, 1.249]
  5 strictly dominant configs all used I437_k3=17.5% (vs 12.5% in P76)
  Top AVG (1.931) also had I437_k3=17.5%
  High-MIN config: V1=20%, I437_k3=17.5%, I600=20%, F144=42.5% → 1.907/1.223
  Phase 78 grid capped I437_k3 at 17.5% — need to explore 20%, 22.5%, 25%

Phase 79 directions:
  A. I437_k3 weight sweep: 15% to 27.5% at 2.5% steps
     - Fix V1=17.5%, I600=20%, F144=remaining; show optimal I437_k3 weight
     - Also test V1=12.5% (AVG-max region)
  B. Extended 2.5% grid: I437_k3 up to 25%, V1 up to 25%
     - Does I437_k3=20% or 22.5% find new dominant configs?
  C. High-MIN frontier: V1 in [20%, 25%], I437_k3 in [15%, 22.5%]
     - Current best MIN=1.223 has AVG=1.907 — can AVG reach 1.910 with MIN>1.220?
  D. Summary: full Pareto frontier update
"""

import json
import math
import statistics
import subprocess
import sys
import copy
from pathlib import Path

OUT_DIR = "artifacts/phase79"
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
    path = f"/tmp/phase79_{run_name}_{year}.json"
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


def pct_range(lo_10ths: int, hi_10ths_exclusive: int, step_10ths: int = 250) -> list:
    """Generate weight fractions in units of 10ths of a percent (250 = 2.5%)."""
    return [x / 10000 for x in range(lo_10ths, hi_10ths_exclusive, step_10ths)]


# ═════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PHASE 79: I437_k3 Weight Extension + High-MIN Frontier")
print("=" * 70)

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

P78_BALANCED = {"avg": 1.919, "min": 1.206}  # V1=17.5%, I437_k3=17.5%, I600=20%, F144=45%
P75_AVGMAX = {"avg": 1.934, "min": 0.972}

# ─── Reference runs ───────────────────────────────────────────────────────────
print("\n  Running reference strategies...", flush=True)
v1_data = run_strategy("v1_ref", "nexus_alpha_v1", V1_PARAMS)
i437_k3 = run_strategy("idio_lb437_bw168_k3", "idio_momentum_alpha", make_idio_params(437, 168, k=3))
i600_k2 = run_strategy("idio_lb600_bw168", "idio_momentum_alpha", make_idio_params(600, 168, k=2))
f144_data = run_strategy("fund_144_ref", "funding_momentum_alpha", FUND_144_PARAMS)
f168_data = run_strategy("fund_168_ref", "funding_momentum_alpha", FUND_168_PARAMS)

# Verify P78 champion
strats_base = {"v1": v1_data, "idio_437_k3": i437_k3, "idio_600": i600_k2,
               "f144": f144_data, "f168": f168_data}
p78_verify = blend_ensemble(strats_base, {"v1": 0.175, "idio_437_k3": 0.175, "idio_600": 0.20, "f144": 0.45})
print(f"\n  P78 champion verify: AVG={p78_verify['avg']}, MIN={p78_verify['min']}")
print(f"  YbY: {p78_verify['yby']}")

# ─── SECTION A: I437_k3 weight sweep ─────────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION A: I437_k3 weight sweep (15% to 27.5%)")
print("  Fix V1=17.5%, I600=20%; derive F144=1-V1-I437_k3-I600; F168=0")
print("═" * 70)

print(f"\n  V1=17.5% fixed, I437_k3 sweep:")
print(f"  I437_k3%  I600%  F144%    AVG     MIN   YbY")
print(f"  {'─'*60}")
sweep_v175 = []
for wi437 in pct_range(1500, 2750, 250):   # 15% to 25%
    wf144 = round(1.0 - 0.175 - wi437 - 0.20, 4)
    if wf144 < 0.35 or wf144 > 0.575:
        continue
    r = blend_ensemble(strats_base, {"v1": 0.175, "idio_437_k3": wi437, "idio_600": 0.20, "f144": wf144})
    sweep_v175.append((wi437, r))
    yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
    print(f"  {wi437*100:5.1f}  20.0  {wf144*100:5.1f}   {r['avg']:.3f}  {r['min']:.3f}  {yby_r}")

print(f"\n  V1=12.5% fixed, I437_k3 sweep (AVG-max region):")
print(f"  I437_k3%  I600%  F144%  F168%   AVG     MIN   YbY")
print(f"  {'─'*65}")
sweep_v125 = []
for wi437 in pct_range(1500, 2750, 250):   # 15% to 25%
    for wi600 in [0.20, 0.225]:
        for wf168 in [0.0, 0.05, 0.10]:
            wf144 = round(1.0 - 0.125 - wi437 - wi600 - wf168, 4)
            if wf144 < 0.35 or wf144 > 0.575:
                continue
            r = blend_ensemble(strats_base, {"v1": 0.125, "idio_437_k3": wi437, "idio_600": wi600,
                                             "f144": wf144, "f168": wf168})
            sweep_v125.append((wi437, wi600, wf168, wf144, r))
            if r["avg"] > 1.920:
                yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
                print(f"  {wi437*100:5.1f}  {wi600*100:4.1f}  {wf144*100:5.1f}  {wf168*100:4.1f}  {r['avg']:.3f}  {r['min']:.3f}  {yby_r}")

if sweep_v125:
    best_v125 = max(sweep_v125, key=lambda x: x[4]["avg"])
    print(f"  → Best V1=12.5% sweep: AVG={best_v125[4]['avg']}, MIN={best_v125[4]['min']}")

# ─── SECTION B: Extended grid (I437_k3 up to 25%) ────────────────────────────
print("\n" + "═" * 70)
print("SECTION B: Extended 2.5% grid (I437_k3 up to 25%, broader coverage)")
print("═" * 70)

v1_vals   = pct_range(1000, 2750, 250)    # 10% to 25%
i437_vals = pct_range(1250, 2750, 250)    # 12.5% to 25%  ← extended from 17.5%
i600_vals = pct_range(1500, 3000, 250)    # 15% to 27.5%
f168_vals = [0.0, 0.025, 0.05]

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

# Best MIN with AVG > 1.910 (stricter bar now)
high_avg = [(row, row[5]) for row in grid_results if row[5]["avg"] > 1.910]
high_avg_by_min = sorted(high_avg, key=lambda x: x[1]["min"], reverse=True)
print(f"\n  Best MIN where AVG > 1.910 ({len(high_avg)} configs):")
print(f"       V1%  I437_k3%  I600%  F144%  F168%    AVG     MIN   YbY")
print(f"  {'─'*80}")
for (wv1, wi437, wi600, wf144, wf168, r), _ in high_avg_by_min[:10]:
    yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
    print(f"    {wv1*100:4.1f}  {wi437*100:5.1f}  {wi600*100:5.1f}  {wf144*100:5.1f}  {wf168*100:4.1f}  {r['avg']:.3f}  {r['min']:.3f}  {yby_r}")

# Strictly dominant over P78
dominant = [(wv1, wi437, wi600, wf144, wf168, r)
            for wv1, wi437, wi600, wf144, wf168, r in grid_results
            if r["avg"] > P78_BALANCED["avg"] and r["min"] > P78_BALANCED["min"]]
if dominant:
    print(f"\n  ★ STRICTLY DOMINANT over P78 ({len(dominant)} configs):")
    for wv1, wi437, wi600, wf144, wf168, r in dominant[:5]:
        yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
        print(f"    V1={wv1*100:.1f}%, I437_k3={wi437*100:.1f}%, I600={wi600*100:.1f}%, F144={wf144*100:.1f}%, F168={wf168*100:.1f}%")
        print(f"    AVG={r['avg']}, MIN={r['min']}, YbY={yby_r}")
else:
    print(f"\n  No config strictly dominates P78 (AVG={P78_BALANCED['avg']}, MIN={P78_BALANCED['min']})")

# ─── SECTION C: High-MIN frontier ────────────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION C: High-MIN frontier (V1 in 20-27.5%, I437_k3 in 15-25%)")
print("  Best known: V1=20%, I437_k3=17.5%, I600=20%, F144=42.5% → 1.907/1.223")
print("  Goal: AVG > 1.910 with MIN > 1.220")
print("═" * 70)

highmin_results = []
for wv1 in pct_range(2000, 2750, 250):     # 20% to 25%
    for wi437 in pct_range(1250, 2500, 250):  # 12.5% to 22.5%
        for wi600 in pct_range(1500, 2750, 250):   # 15% to 25%
            for wf168 in [0.0, 0.025, 0.05]:
                wf144 = round(1.0 - wv1 - wi437 - wi600 - wf168, 4)
                if wf144 < 0.35 or wf144 > 0.575:
                    continue
                wts = {"v1": wv1, "idio_437_k3": wi437, "idio_600": wi600,
                       "f144": wf144, "f168": wf168}
                r = blend_ensemble(strats_base, wts)
                highmin_results.append((wv1, wi437, wi600, wf144, wf168, r))

highmin_results.sort(key=lambda x: x[5]["min"], reverse=True)
print(f"\n  Top MIN configs (V1 >= 20%):")
print(f"       V1%  I437_k3%  I600%  F144%  F168%    AVG     MIN   YbY")
print(f"  {'─'*80}")
for wv1, wi437, wi600, wf144, wf168, r in highmin_results[:15]:
    yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
    print(f"    {wv1*100:4.1f}  {wi437*100:5.1f}  {wi600*100:5.1f}  {wf144*100:5.1f}  {wf168*100:4.1f}  {r['avg']:.3f}  {r['min']:.3f}  {yby_r}")

# Best MIN with AVG > 1.900
highmin_filtered = [(x, r) for *x, r in [(wv1, wi437, wi600, wf144, wf168, r)
                    for wv1, wi437, wi600, wf144, wf168, r in highmin_results]
                    if r["avg"] > 1.900]
# Re-sort: pick items with AVG > 1.900 sorted by MIN
# Actually filter differently
highmin_filtered = []
for wv1, wi437, wi600, wf144, wf168, r in highmin_results:
    if r["avg"] > 1.900:
        highmin_filtered.append((wv1, wi437, wi600, wf144, wf168, r))
highmin_filtered.sort(key=lambda x: x[5]["min"], reverse=True)

print(f"\n  Best MIN where AVG > 1.900 ({len(highmin_filtered)} configs):")
for wv1, wi437, wi600, wf144, wf168, r in highmin_filtered[:10]:
    yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
    print(f"    V1={wv1*100:.1f}%, I437_k3={wi437*100:.1f}%, I600={wi600*100:.1f}%, F144={wf144*100:.1f}%, F168={wf168*100:.1f}%: AVG={r['avg']}, MIN={r['min']}  {yby_r}")

# ─── SECTION D: Full Pareto frontier ─────────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION D: Full AVG/MIN Pareto Frontier (all grid results)")
print("═" * 70)

# Combine all grid results and map the Pareto frontier
all_results = list(grid_results) + list(highmin_results)
# Deduplicate (same config might appear twice)
seen = set()
unique_results = []
for row in all_results:
    key = (round(row[0]*1000), round(row[1]*1000), round(row[2]*1000), round(row[3]*1000), round(row[4]*1000))
    if key not in seen:
        seen.add(key)
        unique_results.append(row)

# Compute Pareto frontier (non-dominated set)
pareto = []
for i, row_i in enumerate(unique_results):
    dominated = False
    ri = row_i[5]
    for j, row_j in enumerate(unique_results):
        if i == j:
            continue
        rj = row_j[5]
        if rj["avg"] >= ri["avg"] and rj["min"] >= ri["min"] and (rj["avg"] > ri["avg"] or rj["min"] > ri["min"]):
            dominated = True
            break
    if not dominated:
        pareto.append(row_i)

pareto.sort(key=lambda x: x[5]["avg"], reverse=True)
print(f"\n  AVG/MIN Pareto frontier ({len(pareto)} non-dominated configs):")
print(f"       V1%  I437_k3%  I600%  F144%  F168%    AVG     MIN   YbY")
print(f"  {'─'*80}")
for wv1, wi437, wi600, wf144, wf168, r in pareto[:20]:
    yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
    flag = " ★" if r["avg"] > P78_BALANCED["avg"] and r["min"] > P78_BALANCED["min"] else ""
    print(f"    {wv1*100:4.1f}  {wi437*100:5.1f}  {wi600*100:5.1f}  {wf144*100:5.1f}  {wf168*100:4.1f}  {r['avg']:.3f}  {r['min']:.3f}  {yby_r}{flag}")

# ─── SECTION E: Save champions ────────────────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION E: Phase 79 Summary & Champions")
print("═" * 70)

print(f"\n  Champion landscape:")
print(f"    Phase 75 AVG-max:  AVG=1.934, MIN=0.972")
print(f"    Phase 78 balanced: AVG=1.919, MIN=1.206 ← entering this phase")
print(f"    Phase 78 best MIN (AVG>1.900): AVG~1.907, MIN=1.223")

# Check for new champions
new_balanced = None
for wv1, wi437, wi600, wf144, wf168, r in unique_results:
    if r["avg"] > P78_BALANCED["avg"] and r["min"] > P78_BALANCED["min"]:
        if new_balanced is None or (r["avg"] + r["min"] > new_balanced[5]["avg"] + new_balanced[5]["min"]):
            new_balanced = (wv1, wi437, wi600, wf144, wf168, r)

new_avgmax = None
for wv1, wi437, wi600, wf144, wf168, r in unique_results:
    if r["avg"] > P75_AVGMAX["avg"]:
        if new_avgmax is None or r["avg"] > new_avgmax[5]["avg"]:
            new_avgmax = (wv1, wi437, wi600, wf144, wf168, r)

if new_balanced:
    wv1, wi437, wi600, wf144, wf168, r = new_balanced
    yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
    print(f"\n  ★ NEW BALANCED CHAMPION:")
    print(f"    V1={wv1*100:.1f}%, I437_k3={wi437*100:.1f}%, I600={wi600*100:.1f}%, F144={wf144*100:.1f}%, F168={wf168*100:.1f}%")
    print(f"    AVG={r['avg']}, MIN={r['min']}, YbY={yby_r}")
    save_champion(
        "ensemble_p79_k3i437_balanced",
        f"Phase 79 balanced champion: V1({wv1*100:.1f}%)+I437_bw168_k3({wi437*100:.1f}%)+I600_bw168_k2({wi600*100:.1f}%)+F144({wf144*100:.1f}%)+F168({wf168*100:.1f}%). AVG={r['avg']}, MIN={r['min']}",
        [
            {"name": "nexus_alpha_v1", "params": V1_PARAMS, "weight": wv1},
            {"name": "idio_momentum_alpha", "params": make_idio_params(437, 168, k=3), "weight": wi437},
            {"name": "idio_momentum_alpha", "params": make_idio_params(600, 168, k=2), "weight": wi600},
            {"name": "funding_momentum_alpha", "params": FUND_144_PARAMS, "weight": wf144},
            {"name": "funding_momentum_alpha", "params": FUND_168_PARAMS, "weight": wf168},
        ],
    )
else:
    print(f"\n  No config strictly dominates P78 balanced champion (AVG={P78_BALANCED['avg']}, MIN={P78_BALANCED['min']})")
    # Save best P79 config anyway (best balanced from this run)
    if high_avg_by_min:
        (wv1, wi437, wi600, wf144, wf168, r), _ = high_avg_by_min[0]
        print(f"  Best balanced from Section B grid: AVG={r['avg']}, MIN={r['min']}")
        yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
        print(f"    V1={wv1*100:.1f}%, I437_k3={wi437*100:.1f}%, I600={wi600*100:.1f}%, F144={wf144*100:.1f}%, F168={wf168*100:.1f}%")
        print(f"    YbY: {yby_r}")

if new_avgmax:
    wv1, wi437, wi600, wf144, wf168, r = new_avgmax
    print(f"\n  ★ NEW AVG-MAX: AVG={r['avg']}, MIN={r['min']}")
    print(f"    V1={wv1*100:.1f}%, I437_k3={wi437*100:.1f}%, I600={wi600*100:.1f}%, F144={wf144*100:.1f}%, F168={wf168*100:.1f}%")

if highmin_filtered:
    wv1, wi437, wi600, wf144, wf168, r = highmin_filtered[0]
    print(f"\n  Best high-MIN with AVG > 1.900: AVG={r['avg']}, MIN={r['min']}")
    print(f"    V1={wv1*100:.1f}%, I437_k3={wi437*100:.1f}%, I600={wi600*100:.1f}%, F144={wf144*100:.1f}%, F168={wf168*100:.1f}%")
    print(f"    YbY: {[round(v, 3) if v is not None else None for v in r['yby']]}")

print("\n" + "=" * 70)
print("PHASE 79 COMPLETE")
print("=" * 70)
