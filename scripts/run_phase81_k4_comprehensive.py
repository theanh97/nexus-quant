#!/usr/bin/env python3
"""
Phase 81: I437 k=4 Comprehensive Grid — Pareto Frontier for k=4 Ensemble

Phase 80 BREAKTHROUGH:
  I437 k=4 standalone: AVG=1.623, 2022=1.281 (vs k3: 1.227, 2022=0.635!)
  P78 weights + I437_k4: AVG=2.015, MIN=1.168 — FIRST TIME ABOVE 2.0!
    YbY: [3.546, 1.366, 1.168, 2.667, 1.326]
  Trade-off vs P78 k3 (1.919/1.206): +0.096 AVG, -0.038 MIN (2023 bottleneck)

Phase 81 directions:
  A. I437_k4 weight sweep: find optimal weight (P80 tested only 17.5%)
     - Heavier k4: push AVG higher (can we reach 2.05+?)
     - Lighter k4: improve 2023 MIN (approaching P78 1.206?)
  B. Full 2.5% grid with I437_k4 replacing k3
     - Map the complete Pareto frontier for k4-based configs
     - Focus on (V1, I437_k4, I600, F144, F168)
  C. Dual-k strategy: I437_k3 + I437_k4 simultaneously
     - k3 provides better 2023 floor; k4 provides better 2022 profile
     - Can we get BOTH AVG>1.95 AND MIN>1.190?
  D. I437_k4 + I600_k4: does k=4 for I600 help?
     (Phase 77 tested I600_k3 — it hurt 2023. k4 might be different)
  E. Summary: new Pareto frontier with k=4
"""

import json
import math
import statistics
import subprocess
import sys
import copy
from pathlib import Path

OUT_DIR = "artifacts/phase81"
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

def make_fund_params(lb: int, k: int = 2) -> dict:
    return {
        "k_per_side": k, "funding_lookback_bars": lb, "direction": "contrarian",
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
    path = f"/tmp/phase81_{run_name}_{year}.json"
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
    return [x / 10000 for x in range(lo_10ths, hi_10ths_exclusive, step_10ths)]


# ═════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PHASE 81: I437 k=4 Comprehensive Grid — Pareto Frontier")
print("=" * 70)

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

P78_BALANCED = {"avg": 1.919, "min": 1.206}
P79_AVGMAX   = {"avg": 1.934, "min": 1.098}
P80_K4_P78WTS = {"avg": 2.015, "min": 1.168}  # V1=17.5%, I437_k4=17.5%, I600=20%, F144=45%

# ─── Reference runs ───────────────────────────────────────────────────────────
print("\n  Running reference strategies...", flush=True)
v1_data = run_strategy("v1_ref", "nexus_alpha_v1", V1_PARAMS)
i437_k3 = run_strategy("idio_lb437_bw168_k3", "idio_momentum_alpha", make_idio_params(437, 168, k=3))
i437_k4 = run_strategy("idio_lb437_bw168_k4", "idio_momentum_alpha", make_idio_params(437, 168, k=4))
i600_k2 = run_strategy("idio_lb600_bw168", "idio_momentum_alpha", make_idio_params(600, 168, k=2))
i600_k4 = run_strategy("idio_lb600_bw168_k4", "idio_momentum_alpha", make_idio_params(600, 168, k=4))
f144_k2 = run_strategy("fund_144_k2", "funding_momentum_alpha", make_fund_params(144, k=2))
f168_k2 = run_strategy("fund_168_k2", "funding_momentum_alpha", make_fund_params(168, k=2))

# Verify P80 finding
strats_base_k4 = {"v1": v1_data, "idio_437_k4": i437_k4, "idio_600": i600_k2,
                  "f144": f144_k2, "f168": f168_k2}
p80_verify = blend_ensemble(strats_base_k4, {"v1": 0.175, "idio_437_k4": 0.175, "idio_600": 0.20, "f144": 0.45})
print(f"\n  P80 k4 verify: AVG={p80_verify['avg']}, MIN={p80_verify['min']}")
print(f"  YbY: {p80_verify['yby']}")

# ─── SECTION A: I437_k4 weight sweep ──────────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION A: I437_k4 weight sweep (5% to 30%)")
print("  Fix V1=17.5%, I600=20%; derive F144=1-V1-I437_k4-I600; F168=0")
print("  P80 baseline: I437_k4=17.5% → 2.015/1.168")
print("═" * 70)

sweep_results = []
print(f"\n  I437_k4%  F144%    AVG     MIN   YbY")
print(f"  {'─'*60}")
for wi437 in [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25, 0.275, 0.30]:
    wf144 = round(1.0 - 0.175 - wi437 - 0.20, 4)
    if wf144 < 0.35 or wf144 > 0.575:
        continue
    r = blend_ensemble(strats_base_k4, {"v1": 0.175, "idio_437_k4": wi437, "idio_600": 0.20, "f144": wf144})
    sweep_results.append((wi437, r))
    yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
    flag = " ★ AVG>2.0" if r["avg"] >= 2.0 else ("")
    print(f"  {wi437*100:5.1f}  {wf144*100:5.1f}   {r['avg']:.3f}  {r['min']:.3f}  {yby_r}{flag}")

# ─── SECTION B: Full 2.5% grid with I437_k4 ──────────────────────────────────
print("\n" + "═" * 70)
print("SECTION B: Full 2.5% grid — I437_k4 replacing I437_k3")
print("═" * 70)

v1_vals   = pct_range(1000, 2500, 250)    # 10% to 22.5%
i437_vals = pct_range(750, 2750, 250)     # 7.5% to 25%
i600_vals = pct_range(1500, 3000, 250)    # 15% to 27.5%
f168_vals = [0.0, 0.025, 0.05]

grid_k4 = []
total_k4 = 0

for wv1 in v1_vals:
    for wi437 in i437_vals:
        for wi600 in i600_vals:
            for wf168 in f168_vals:
                wf144 = round(1.0 - wv1 - wi437 - wi600 - wf168, 4)
                if wf144 < 0.35 or wf144 > 0.575:
                    continue
                wts = {"v1": wv1, "idio_437_k4": wi437, "idio_600": wi600,
                       "f144": wf144, "f168": wf168}
                r = blend_ensemble(strats_base_k4, wts)
                grid_k4.append((wv1, wi437, wi600, wf144, wf168, r))
                total_k4 += 1

print(f"\n  Tested {total_k4} configurations")
grid_k4.sort(key=lambda x: x[5]["avg"], reverse=True)

print(f"\n  Top 15 by AVG:")
print(f"       V1%  I437_k4%  I600%  F144%  F168%    AVG     MIN")
print(f"  {'─'*65}")
for row in grid_k4[:15]:
    wv1, wi437, wi600, wf144, wf168, r = row
    print(f"    {wv1*100:4.1f}  {wi437*100:5.1f}  {wi600*100:5.1f}  {wf144*100:5.1f}  {wf168*100:4.1f}  {r['avg']:.3f}  {r['min']:.3f}")

# Best MIN with AVG > 1.950 (higher bar since k4 can reach 2.0+)
high_avg_k4 = [(row, row[5]) for row in grid_k4 if row[5]["avg"] > 1.950]
high_avg_by_min_k4 = sorted(high_avg_k4, key=lambda x: x[1]["min"], reverse=True)
print(f"\n  Best MIN where AVG > 1.950 ({len(high_avg_k4)} configs):")
print(f"       V1%  I437_k4%  I600%  F144%  F168%    AVG     MIN   YbY")
print(f"  {'─'*80}")
for (wv1, wi437, wi600, wf144, wf168, r), _ in high_avg_by_min_k4[:10]:
    yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
    print(f"    {wv1*100:4.1f}  {wi437*100:5.1f}  {wi600*100:5.1f}  {wf144*100:5.1f}  {wf168*100:4.1f}  {r['avg']:.3f}  {r['min']:.3f}  {yby_r}")

# Best MIN with AVG > 1.900 (broader view)
print(f"\n  Best MIN where AVG > 1.900 (top 10 by MIN):")
high_avg_900 = [(row, row[5]) for row in grid_k4 if row[5]["avg"] > 1.900]
high_avg_900_by_min = sorted(high_avg_900, key=lambda x: x[1]["min"], reverse=True)
print(f"       V1%  I437_k4%  I600%  F144%  F168%    AVG     MIN   YbY")
print(f"  {'─'*80}")
for (wv1, wi437, wi600, wf144, wf168, r), _ in high_avg_900_by_min[:10]:
    yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
    print(f"    {wv1*100:4.1f}  {wi437*100:5.1f}  {wi600*100:5.1f}  {wf144*100:5.1f}  {wf168*100:4.1f}  {r['avg']:.3f}  {r['min']:.3f}  {yby_r}")

# Pareto frontier for k4 grid
pareto_k4 = []
for i, row_i in enumerate(grid_k4):
    dominated = False
    ri = row_i[5]
    for j, row_j in enumerate(grid_k4):
        if i == j:
            continue
        rj = row_j[5]
        if rj["avg"] >= ri["avg"] and rj["min"] >= ri["min"] and (rj["avg"] > ri["avg"] or rj["min"] > ri["min"]):
            dominated = True
            break
    if not dominated:
        pareto_k4.append(row_i)

pareto_k4.sort(key=lambda x: x[5]["avg"], reverse=True)
print(f"\n  Pareto frontier for k4 grid ({len(pareto_k4)} non-dominated configs):")
print(f"       V1%  I437_k4%  I600%  F144%  F168%    AVG     MIN   YbY")
print(f"  {'─'*80}")
for wv1, wi437, wi600, wf144, wf168, r in pareto_k4[:15]:
    yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
    print(f"    {wv1*100:4.1f}  {wi437*100:5.1f}  {wi600*100:5.1f}  {wf144*100:5.1f}  {wf168*100:4.1f}  {r['avg']:.3f}  {r['min']:.3f}  {yby_r}")

# ─── SECTION C: Dual-k strategy (I437_k3 + I437_k4) ─────────────────────────
print("\n" + "═" * 70)
print("SECTION C: Dual-k — I437_k3 + I437_k4 simultaneously")
print("  k3: better 2023 floor; k4: better 2022/2024 profile")
print("  Can we get AVG>1.95 AND MIN>1.190?")
print("═" * 70)

strats_dual_k = {"v1": v1_data, "idio_437_k3": i437_k3, "idio_437_k4": i437_k4,
                 "idio_600": i600_k2, "f144": f144_k2, "f168": f168_k2}

# Sweep: split idio between k3 and k4
print(f"\n  Dual-k sweep: V1=17.5%, I600=20%, F144=45%; split total idio budget between k3/k4")
print(f"  I437_k3%  I437_k4%    AVG     MIN   YbY")
print(f"  {'─'*60}")
dual_k_results = []
total_idio = 0.175  # same total as P78 champion
for frac_k4 in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    wi437_k4 = round(total_idio * frac_k4, 4)
    wi437_k3 = round(total_idio * (1 - frac_k4), 4)
    wts = {"v1": 0.175, "idio_437_k3": wi437_k3, "idio_437_k4": wi437_k4,
           "idio_600": 0.20, "f144": 0.45}
    r = blend_ensemble(strats_dual_k, wts)
    dual_k_results.append((wi437_k3, wi437_k4, r))
    yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
    print(f"  {wi437_k3*100:5.1f}     {wi437_k4*100:5.1f}    {r['avg']:.3f}  {r['min']:.3f}  {yby_r}")

# Broader dual-k grid
print(f"\n  Broader dual-k 2.5% grid:")
dual_k_grid = []
for wi437_k3 in pct_range(0, 1750, 250):      # 0% to 15%
    for wi437_k4 in pct_range(250, 2750, 250): # 2.5% to 25%
        if wi437_k3 + wi437_k4 > 0.30:
            continue
        wf144 = round(1.0 - 0.175 - wi437_k3 - wi437_k4 - 0.20, 4)
        if wf144 < 0.35 or wf144 > 0.575:
            continue
        wts = {"v1": 0.175, "idio_437_k3": wi437_k3, "idio_437_k4": wi437_k4,
               "idio_600": 0.20, "f144": wf144}
        r = blend_ensemble(strats_dual_k, wts)
        dual_k_grid.append((wi437_k3, wi437_k4, wf144, r))

dual_k_grid.sort(key=lambda x: x[3]["avg"], reverse=True)
print(f"  Tested {len(dual_k_grid)} dual-k configurations")
print(f"\n  Top 10 by AVG:")
print(f"  I437_k3%  I437_k4%  F144%    AVG     MIN   YbY")
print(f"  {'─'*65}")
for wk3, wk4, wf144, r in dual_k_grid[:10]:
    yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
    print(f"  {wk3*100:5.1f}     {wk4*100:5.1f}  {wf144*100:5.1f}   {r['avg']:.3f}  {r['min']:.3f}  {yby_r}")

# Dual-k best MIN with AVG > 1.950
dual_k_high = sorted(
    [(wk3, wk4, wf144, r) for wk3, wk4, wf144, r in dual_k_grid if r["avg"] > 1.950],
    key=lambda x: x[3]["min"], reverse=True
)
print(f"\n  Best MIN (dual-k, AVG > 1.950, {len(dual_k_high)} configs):")
print(f"  I437_k3%  I437_k4%  F144%    AVG     MIN   YbY")
print(f"  {'─'*65}")
for wk3, wk4, wf144, r in dual_k_high[:10]:
    yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
    print(f"  {wk3*100:5.1f}     {wk4*100:5.1f}  {wf144*100:5.1f}   {r['avg']:.3f}  {r['min']:.3f}  {yby_r}")

# ─── SECTION D: I437_k4 + I600_k4 ────────────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION D: I437_k4 + I600_k4 combination")
yby_i600_k4 = [round(i600_k4.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]
print(f"  I600 k=2 standalone: AVG={i600_k2['_avg']}, MIN={i600_k2['_min']}")
print(f"    YbY: {[round(i600_k2.get(y, {}).get('sharpe', 0.0), 3) for y in YEARS]}")
print(f"  I600 k=4 standalone: AVG={i600_k4['_avg']}, MIN={i600_k4['_min']}")
print(f"    YbY: {yby_i600_k4}")
print("═" * 70)

strats_k4_both = {"v1": v1_data, "idio_437_k4": i437_k4, "idio_600_k4": i600_k4,
                  "f144": f144_k2, "f168": f168_k2}

# P80 weights with both k4
r_both_k4 = blend_ensemble(strats_k4_both, {"v1": 0.175, "idio_437_k4": 0.175, "idio_600_k4": 0.20, "f144": 0.45})
print(f"\n  P80 weights + I437_k4 + I600_k4: AVG={r_both_k4['avg']}, MIN={r_both_k4['min']}")
print(f"  YbY: {r_both_k4['yby']}")

# ─── SECTION E: Summary and Champions ────────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION E: Phase 81 Summary & Pareto Comparison")
print("═" * 70)

print(f"\n  Prior Pareto points (k3 space):")
print(f"    P78 balanced: AVG=1.919, MIN=1.206  ← best balanced (k3)")
print(f"    P79 AVG-max:  AVG=1.934, MIN=1.098  ← best AVG (k3, bw168)")

print(f"\n  k4 Pareto frontier (from Section B):")
for wv1, wi437, wi600, wf144, wf168, r in pareto_k4[:10]:
    print(f"    V1={wv1*100:.1f}%, I437_k4={wi437*100:.1f}%, I600={wi600*100:.1f}%, F144={wf144*100:.1f}%, F168={wf168*100:.1f}%: AVG={r['avg']}, MIN={r['min']}")

# Save champions
saved_names = []

# Best balanced k4 (AVG+MIN combined score)
best_k4_balanced = None
for wv1, wi437, wi600, wf144, wf168, r in grid_k4:
    if r["avg"] > P78_BALANCED["avg"] and r["min"] > P78_BALANCED["min"]:
        if best_k4_balanced is None or (r["avg"] + r["min"]) > (best_k4_balanced[5]["avg"] + best_k4_balanced[5]["min"]):
            best_k4_balanced = (wv1, wi437, wi600, wf144, wf168, r)

if best_k4_balanced:
    wv1, wi437, wi600, wf144, wf168, r = best_k4_balanced
    yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
    print(f"\n  ★ NEW BALANCED CHAMPION (k4): AVG={r['avg']}, MIN={r['min']}")
    print(f"    V1={wv1*100:.1f}%, I437_k4={wi437*100:.1f}%, I600={wi600*100:.1f}%, F144={wf144*100:.1f}%, F168={wf168*100:.1f}%")
    print(f"    YbY={yby_r}")
    save_champion(
        "ensemble_p81_k4i437_balanced",
        f"Phase 81 balanced champion: V1({wv1*100:.1f}%)+I437_bw168_k4({wi437*100:.1f}%)+I600_bw168_k2({wi600*100:.1f}%)+F144({wf144*100:.1f}%)+F168({wf168*100:.1f}%). AVG={r['avg']}, MIN={r['min']}",
        [
            {"name": "nexus_alpha_v1", "params": V1_PARAMS, "weight": wv1},
            {"name": "idio_momentum_alpha", "params": make_idio_params(437, 168, k=4), "weight": wi437},
            {"name": "idio_momentum_alpha", "params": make_idio_params(600, 168, k=2), "weight": wi600},
            {"name": "funding_momentum_alpha", "params": make_fund_params(144, k=2), "weight": wf144},
            {"name": "funding_momentum_alpha", "params": make_fund_params(168, k=2), "weight": wf168},
        ],
    )
    saved_names.append("k4 balanced champion")

# Save best AVG-max k4
if grid_k4:
    best_avgmax_k4 = grid_k4[0]
    wv1, wi437, wi600, wf144, wf168, r = best_avgmax_k4
    yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
    print(f"\n  k4 AVG-max: AVG={r['avg']}, MIN={r['min']}")
    print(f"    V1={wv1*100:.1f}%, I437_k4={wi437*100:.1f}%, I600={wi600*100:.1f}%, F144={wf144*100:.1f}%, F168={wf168*100:.1f}%")
    print(f"    YbY={yby_r}")
    save_champion(
        "ensemble_p81_k4i437_avgmax",
        f"Phase 81 AVG-max: V1({wv1*100:.1f}%)+I437_bw168_k4({wi437*100:.1f}%)+I600_bw168_k2({wi600*100:.1f}%)+F144({wf144*100:.1f}%)+F168({wf168*100:.1f}%). AVG={r['avg']}, MIN={r['min']}",
        [
            {"name": "nexus_alpha_v1", "params": V1_PARAMS, "weight": wv1},
            {"name": "idio_momentum_alpha", "params": make_idio_params(437, 168, k=4), "weight": wi437},
            {"name": "idio_momentum_alpha", "params": make_idio_params(600, 168, k=2), "weight": wi600},
            {"name": "funding_momentum_alpha", "params": make_fund_params(144, k=2), "weight": wf144},
            {"name": "funding_momentum_alpha", "params": make_fund_params(168, k=2), "weight": wf168},
        ],
    )

# Save best dual-k
if dual_k_high:
    wk3, wk4, wf144, r = dual_k_high[0]
    yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
    print(f"\n  Best dual-k (AVG>1.950): AVG={r['avg']}, MIN={r['min']}")
    print(f"    I437_k3={wk3*100:.1f}%, I437_k4={wk4*100:.1f}%, I600=20%, F144={wf144*100:.1f}%")
    print(f"    YbY={yby_r}")
    if r["avg"] > P78_BALANCED["avg"] and r["min"] > P78_BALANCED["min"]:
        print(f"  ★ Dual-k strictly dominates P78!")
    save_champion(
        "ensemble_p81_dualk_balanced",
        f"Phase 81 dual-k champion: V1(17.5%)+I437_bw168_k3({wk3*100:.1f}%)+I437_bw168_k4({wk4*100:.1f}%)+I600(20%)+F144({wf144*100:.1f}%). AVG={r['avg']}, MIN={r['min']}",
        [
            {"name": "nexus_alpha_v1", "params": V1_PARAMS, "weight": 0.175},
            {"name": "idio_momentum_alpha", "params": make_idio_params(437, 168, k=3), "weight": wk3},
            {"name": "idio_momentum_alpha", "params": make_idio_params(437, 168, k=4), "weight": wk4},
            {"name": "idio_momentum_alpha", "params": make_idio_params(600, 168, k=2), "weight": 0.20},
            {"name": "funding_momentum_alpha", "params": make_fund_params(144, k=2), "weight": wf144},
        ],
    )

if not best_k4_balanced:
    print(f"\n  k4 does not strictly dominate P78 balanced (1.919/1.206).")
    print(f"  However, k4 trades better AVG for worse MIN — different operating point.")

print("\n" + "=" * 70)
print("PHASE 81 COMPLETE")
print("=" * 70)
