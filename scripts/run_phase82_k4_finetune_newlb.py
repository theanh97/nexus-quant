#!/usr/bin/env python3
"""
Phase 82: k=4 Fine Grid + New Lookback Sweep + F144_k4 Test
=============================================================
Phase 81 findings:
  k4 BALANCED: V1=22.5%, I437_k4=22.5%, I600=15%, F144=40% → 2.010/1.245
  k4 BEST-MIN:  V1=22.5%, I437_k4=17.5%, I600=15%, F144=45% → 1.981/1.260
  k4 AVG-MAX:  V1=10%, I437_k4=25%, I600=20%, F144=40%, F168=5% → 2.079/1.015
  Dual-k suboptimal; I600_k4 fails; k3 Pareto fully dominated

Phase 82 agenda:
  A) Fine grid (1.25% steps) around P81 balanced — higher-res Pareto mapping
  B) High-MIN frontier push: V1=25-32.5% region (push MIN above 1.260?)
  C) F144 k=4 standalone + ensemble test (unknown territory)
  D) I437 lookback neighborhood sweep with k=4 (is 437h still optimal?)
  E) Summary, save new champions
"""

import json, math, statistics, subprocess, sys, copy
from pathlib import Path
from itertools import product

OUT_DIR = "artifacts/phase82"
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
    path = f"/tmp/phase82_{run_name}_{year}.json"
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

def pct_range(lo_10ths: int, hi_10ths_exclusive: int, step_10ths: int = 125) -> list:
    """Generate weight fractions. step_10ths=125 → 1.25% steps."""
    return [x / 10000 for x in range(lo_10ths, hi_10ths_exclusive, step_10ths)]

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
print("PHASE 82: k=4 Fine Grid + New Lookback Sweep + F144_k4 Test")
print("=" * 70)

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

P81_BALANCED = {"avg": 2.010, "min": 1.245}
P81_BESTMIN  = {"avg": 1.981, "min": 1.260}
P81_AVGMAX   = {"avg": 2.079, "min": 1.015}

# ─── Reference runs ───────────────────────────────────────────────────────────
print("\n  Caching reference runs...", flush=True)

v1_data   = run_strategy("p82_v1",          "nexus_alpha_v1",      V1_PARAMS)
i437_k4   = run_strategy("p82_i437_k4",    "idio_momentum_alpha", make_idio_params(437, 168, k=4))
i437_k3   = run_strategy("p82_i437_k3",    "idio_momentum_alpha", make_idio_params(437, 168, k=3))
i600_k2   = run_strategy("p82_i600_k2",    "idio_momentum_alpha", make_idio_params(600, 168, k=2))
f144_k2   = run_strategy("p82_f144_k2",    "funding_momentum_alpha", make_fund_params(144, k=2))
f168_k2   = run_strategy("p82_f168_k2",    "funding_momentum_alpha", make_fund_params(168, k=2))

base = {"v1": v1_data, "i437_k4": i437_k4, "i437_k3": i437_k3,
        "i600_k2": i600_k2, "f144_k2": f144_k2, "f168_k2": f168_k2}

# Verify P81 balanced
p81b_verify = blend_ensemble(base, {"v1": 0.225, "i437_k4": 0.225, "i600_k2": 0.15, "f144_k2": 0.40})
print(f"\n  P81 balanced verify: AVG={p81b_verify['avg']}, MIN={p81b_verify['min']}")
print(f"  YbY: {[round(v, 3) if v is not None else None for v in p81b_verify['yby']]}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION A: Fine grid (1.25% steps) around P81 balanced champion
# V1 ~22.5%, I437_k4 ~22.5%, I600 ~15%, F144 residual
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("SECTION A: Fine grid (1.25% steps) around P81 balanced")
print("  P81 balanced: V1=22.5%, I437_k4=22.5%, I600=15%, F144=40%")
print("═" * 70)

# Ranges centered on P81 balanced (±5% around each axis)
v1_range   = pct_range(1750, 2875, 125)    # 17.5% to 28.75%
k4_range   = pct_range(1500, 2875, 125)    # 15% to 28.75%
i600_range = pct_range(1000, 2000, 125)    # 10% to 20%

grid_a = []
count_a = 0
for wv1, wk4, wi600 in product(v1_range, k4_range, i600_range):
    wf144 = round(1.0 - wv1 - wk4 - wi600, 4)
    if not (0.30 <= wf144 <= 0.55):
        continue
    count_a += 1
    r = blend_ensemble(base, {"v1": wv1, "i437_k4": wk4, "i600_k2": wi600, "f144_k2": wf144})
    grid_a.append((wv1, wk4, wi600, wf144, r))

print(f"\n  Tested {count_a} configurations")
grid_a.sort(key=lambda x: x[4]["avg"], reverse=True)

print(f"\n  Top 15 by AVG:")
print(f"  {'V1%':>7} {'I437k4%':>8} {'I600%':>6} {'F144%':>6}  {'AVG':>7} {'MIN':>7}  YbY")
print("  " + "─" * 80)
for wv1, wk4, wi600, wf144, r in grid_a[:15]:
    yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
    print(f"  {wv1*100:>7.2f} {wk4*100:>8.2f} {wi600*100:>6.2f} {wf144*100:>6.2f}  {r['avg']:>7.3f} {r['min']:>7.3f}  {yby_r}")

# Best MIN where AVG >= 2.000
high_avg_a = [(wv1, wk4, wi600, wf144, r) for wv1, wk4, wi600, wf144, r in grid_a if r["avg"] >= 2.000]
best_min_a = sorted(high_avg_a, key=lambda x: x[4]["min"], reverse=True)
print(f"\n  Best MIN where AVG >= 2.000 ({len(high_avg_a)} configs):")
print(f"  {'V1%':>7} {'I437k4%':>8} {'I600%':>6} {'F144%':>6}  {'AVG':>7} {'MIN':>7}  YbY")
print("  " + "─" * 80)
for wv1, wk4, wi600, wf144, r in best_min_a[:10]:
    yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
    print(f"  {wv1*100:>7.2f} {wk4*100:>8.2f} {wi600*100:>6.2f} {wf144*100:>6.2f}  {r['avg']:>7.3f} {r['min']:>7.3f}  {yby_r}")

# Pareto frontier for Section A
pareto_a = []
for i, row_i in enumerate(grid_a):
    ri = row_i[4]
    dominated = any(
        j != i and row_j[4]["avg"] >= ri["avg"] and row_j[4]["min"] >= ri["min"]
        and (row_j[4]["avg"] > ri["avg"] or row_j[4]["min"] > ri["min"])
        for j, row_j in enumerate(grid_a)
    )
    if not dominated:
        pareto_a.append(row_i)

pareto_a.sort(key=lambda x: x[4]["avg"], reverse=True)
print(f"\n  k4 fine-grid Pareto frontier ({len(pareto_a)} non-dominated):")
print(f"  {'V1%':>7} {'I437k4%':>8} {'I600%':>6} {'F144%':>6}  {'AVG':>7} {'MIN':>7}  YbY")
print("  " + "─" * 80)
for wv1, wk4, wi600, wf144, r in pareto_a[:20]:
    yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
    print(f"  {wv1*100:>7.2f} {wk4*100:>8.2f} {wi600*100:>6.2f} {wf144*100:>6.2f}  {r['avg']:>7.3f} {r['min']:>7.3f}  {yby_r}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION B: High-MIN frontier push — V1=25-32.5%, lower F144
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("SECTION B: High-MIN frontier push — V1=25-32.5%")
print("  Goal: push MIN above 1.260 (current k4 frontier ceiling)")
print("═" * 70)

v1_high  = pct_range(2500, 3375, 250)       # 25% to 32.5%, step 2.5%
k4_mid   = pct_range(1000, 2500, 250)       # 10% to 22.5%, step 2.5%
i600_b   = pct_range(1000, 2500, 250)       # 10% to 22.5%, step 2.5%

grid_b = []
count_b = 0
for wv1, wk4, wi600 in product(v1_high, k4_mid, i600_b):
    wf144 = round(1.0 - wv1 - wk4 - wi600, 4)
    if not (0.20 <= wf144 <= 0.50):
        continue
    count_b += 1
    r = blend_ensemble(base, {"v1": wv1, "i437_k4": wk4, "i600_k2": wi600, "f144_k2": wf144})
    grid_b.append((wv1, wk4, wi600, wf144, r))

print(f"\n  Tested {count_b} configurations")
grid_b.sort(key=lambda x: x[4]["min"], reverse=True)

print(f"\n  Top 10 by MIN (all AVG levels):")
print(f"  {'V1%':>7} {'I437k4%':>8} {'I600%':>6} {'F144%':>6}  {'AVG':>7} {'MIN':>7}  YbY")
print("  " + "─" * 80)
for wv1, wk4, wi600, wf144, r in grid_b[:10]:
    yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
    print(f"  {wv1*100:>7.2f} {wk4*100:>8.2f} {wi600*100:>6.2f} {wf144*100:>6.2f}  {r['avg']:>7.3f} {r['min']:>7.3f}  {yby_r}")

# Best MIN where AVG >= 1.900
high_avg_b = [(wv1, wk4, wi600, wf144, r) for wv1, wk4, wi600, wf144, r in grid_b if r["avg"] >= 1.900]
best_min_b = sorted(high_avg_b, key=lambda x: x[4]["min"], reverse=True)
print(f"\n  Best MIN where AVG >= 1.900 ({len(high_avg_b)} configs):")
print(f"  {'V1%':>7} {'I437k4%':>8} {'I600%':>6} {'F144%':>6}  {'AVG':>7} {'MIN':>7}  YbY")
print("  " + "─" * 80)
for wv1, wk4, wi600, wf144, r in best_min_b[:10]:
    yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
    print(f"  {wv1*100:>7.2f} {wk4*100:>8.2f} {wi600*100:>6.2f} {wf144*100:>6.2f}  {r['avg']:>7.3f} {r['min']:>7.3f}  {yby_r}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION C: F144 k=4 standalone + ensemble test
# F144 k=3 failed (MIN=-0.247, 2025 negative). What about k=4?
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("SECTION C: F144 k=4 standalone + ensemble test")
print("  F144 k=3: AVG=1.226, MIN=-0.247. F144 k=2: AVG=1.302, MIN=0.062.")
print("═" * 70)

f144_k4 = run_strategy("p82_f144_k4", "funding_momentum_alpha", make_fund_params(144, k=4))
base["f144_k4"] = f144_k4

f4_s = f144_k4["_avg"]
f4_m = f144_k4["_min"]
print(f"\n  F144 k=4 standalone: AVG={f4_s}, MIN={f4_m}")
print(f"  F144 k=2 reference:  AVG={f144_k2['_avg']}, MIN={f144_k2['_min']}")

# Test in P81 balanced weights
ens_f144k4 = blend_ensemble(base, {"v1": 0.225, "i437_k4": 0.225, "i600_k2": 0.15, "f144_k4": 0.40})
print(f"\n  P81 weights + F144_k4: AVG={ens_f144k4['avg']}, MIN={ens_f144k4['min']}")
print(f"  YbY: {[round(v, 3) for v in ens_f144k4['yby']]}")
print(f"  (vs P81 balanced k2: AVG={P81_BALANCED['avg']}, MIN={P81_BALANCED['min']})")

# Mixed F144 k2+k4 in P81 weights if k4 standalone > 0.5
if f4_s > 0.5:
    print(f"\n  Mixed F144 (k2+k4) sweep at P81 balanced weights:")
    print(f"  {'k4_frac':>8} {'F144_k4%':>9} {'F144_k2%':>9}  {'AVG':>7} {'MIN':>7}  YbY")
    print("  " + "─" * 65)
    for frac_k4 in [0.0, 0.25, 0.50, 0.75, 1.0]:
        w_k4 = 0.40 * frac_k4
        w_k2 = 0.40 * (1 - frac_k4)
        ens = blend_ensemble(base, {
            "v1": 0.225, "i437_k4": 0.225, "i600_k2": 0.15,
            "f144_k4": w_k4, "f144_k2": w_k2
        })
        yby_r = [round(v, 3) for v in ens["yby"]]
        print(f"  {frac_k4:>8.0%} {w_k4*100:>9.1f} {w_k2*100:>9.1f}  {ens['avg']:>7.3f} {ens['min']:>7.3f}  {yby_r}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION D: I437 lookback neighborhood sweep with k=4
# k=3 optimal: 437h. Is 437h still optimal for k=4?
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("SECTION D: I437 lookback neighborhood sweep with k=4")
print("  k=3 optimal: 437h. Testing neighborhood for k=4.")
print("═" * 70)

lb_candidates = [360, 390, 410, 420, 430, 437, 445, 460, 480, 510, 550]
lb_data = {}

print(f"\n  {'LB':>5}  {'AVG':>7} {'MIN':>7}  YbY")
print("  " + "─" * 60)
for lb in lb_candidates:
    if lb == 437:
        lb_data[lb] = i437_k4
        avg_ = i437_k4["_avg"]
        mn_  = i437_k4["_min"]
        yby_ = [round(i437_k4.get(y, {}).get("sharpe", 0), 3) for y in YEARS]
    else:
        d = run_strategy(f"p82_i{lb}_k4", "idio_momentum_alpha", make_idio_params(lb, 168, k=4))
        lb_data[lb] = d
        avg_ = d["_avg"]
        mn_  = d["_min"]
        yby_ = [round(d.get(y, {}).get("sharpe", 0), 3) for y in YEARS]
    flag = " ★" if avg_ == max(lb_data[k]["_avg"] for k in lb_data) else ""
    print(f"  {lb:>5}  {avg_:>7.3f} {mn_:>7.3f}  {yby_}{flag}")

best_lb = max(lb_data.keys(), key=lambda lb: lb_data[lb]["_avg"])
best_lb_avg = lb_data[best_lb]["_avg"]
print(f"\n  Best lb by AVG: {best_lb}h → {best_lb_avg}")
print(f"  437h standalone: AVG={i437_k4['_avg']}")

# If a different lb is significantly better, test in P81 ensemble
if best_lb != 437 and best_lb_avg > i437_k4["_avg"] + 0.02:
    best_idio_k4 = lb_data[best_lb]
    base[f"i{best_lb}_k4"] = best_idio_k4
    ens_best_lb = blend_ensemble(base, {"v1": 0.225, f"i{best_lb}_k4": 0.225, "i600_k2": 0.15, "f144_k2": 0.40})
    print(f"\n  P81 weights + I{best_lb}_k4: AVG={ens_best_lb['avg']}, MIN={ens_best_lb['min']}")
    print(f"  YbY: {[round(v, 3) for v in ens_best_lb['yby']]}")
    print(f"  (vs P81 balanced 437h: AVG={P81_BALANCED['avg']}, MIN={P81_BALANCED['min']})")
else:
    print(f"\n  437h remains optimal for k=4 (improvement < 0.02 AVG). No change.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION E: Phase 82 Summary — new champions
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("SECTION E: Phase 82 Summary")
print("═" * 70)

all_results = grid_a + grid_b

print(f"\n  Prior champions:")
print(f"    P81 balanced: AVG={P81_BALANCED['avg']}, MIN={P81_BALANCED['min']}")
print(f"    P81 best-MIN: AVG={P81_BESTMIN['avg']}, MIN={P81_BESTMIN['min']}")
print(f"    P81 AVG-max:  AVG={P81_AVGMAX['avg']}, MIN={P81_AVGMAX['min']}")

# New balanced: best MIN where AVG > 2.000
new_bal_candidates = [(wv1, wk4, wi600, wf144, r) for wv1, wk4, wi600, wf144, r in all_results if r["avg"] >= 2.000]
new_balanced = max(new_bal_candidates, key=lambda x: x[4]["min"]) if new_bal_candidates else None

# New AVG-max
new_avgmax = max(all_results, key=lambda x: x[4]["avg"]) if all_results else None

# New absolute best MIN (any AVG)
new_bestmin = max(all_results, key=lambda x: x[4]["min"]) if all_results else None

# New best MIN (AVG >= 1.900)
new_bestmin_900 = max(
    [(wv1, wk4, wi600, wf144, r) for wv1, wk4, wi600, wf144, r in all_results if r["avg"] >= 1.900],
    key=lambda x: x[4]["min"], default=None
)

if new_balanced:
    wv1, wk4, wi600, wf144, r = new_balanced
    yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
    flag = ""
    if r["avg"] > P81_BALANCED["avg"] and r["min"] > P81_BALANCED["min"]:
        flag = " → STRICTLY DOMINATES P81 balanced!"
    elif r["min"] > P81_BALANCED["min"]:
        flag = f" → Better MIN (+{r['min']-P81_BALANCED['min']:.3f})"
    elif r["avg"] > P81_BALANCED["avg"]:
        flag = f" → Better AVG (+{r['avg']-P81_BALANCED['avg']:.3f})"
    print(f"\n  ★ NEW BALANCED CHAMPION: AVG={r['avg']}, MIN={r['min']}{flag}")
    print(f"    V1={wv1*100:.2f}%, I437_k4={wk4*100:.2f}%, I600={wi600*100:.2f}%, F144={wf144*100:.2f}%")
    print(f"    YbY={yby_r}")
    save_champion(
        "ensemble_p82_k4i437_balanced",
        f"Phase 82 balanced: V1={wv1*100:.2f}%+I437k4={wk4*100:.2f}%+I600={wi600*100:.2f}%+F144={wf144*100:.2f}%. AVG={r['avg']}, MIN={r['min']}. YbY={yby_r}",
        [
            {"name": "nexus_alpha_v1",       "params": V1_PARAMS,                        "weight": wv1},
            {"name": "idio_momentum_alpha",   "params": make_idio_params(437, 168, k=4), "weight": wk4},
            {"name": "idio_momentum_alpha",   "params": make_idio_params(600, 168, k=2), "weight": wi600},
            {"name": "funding_momentum_alpha","params": make_fund_params(144, k=2),       "weight": wf144},
        ],
    )
else:
    print(f"\n  No new config with AVG >= 2.000 strictly dominates P81 balanced.")

if new_bestmin_900:
    wv1, wk4, wi600, wf144, r = new_bestmin_900
    yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
    if r["min"] > P81_BESTMIN["min"]:
        print(f"\n  ★ NEW BEST-MIN CHAMPION (AVG>=1.900): AVG={r['avg']}, MIN={r['min']}")
        print(f"    V1={wv1*100:.2f}%, I437_k4={wk4*100:.2f}%, I600={wi600*100:.2f}%, F144={wf144*100:.2f}%")
        print(f"    YbY={yby_r}")
        save_champion(
            "ensemble_p82_k4i437_highmin",
            f"Phase 82 high-MIN: V1={wv1*100:.2f}%+I437k4={wk4*100:.2f}%+I600={wi600*100:.2f}%+F144={wf144*100:.2f}%. AVG={r['avg']}, MIN={r['min']}. YbY={yby_r}",
            [
                {"name": "nexus_alpha_v1",       "params": V1_PARAMS,                        "weight": wv1},
                {"name": "idio_momentum_alpha",   "params": make_idio_params(437, 168, k=4), "weight": wk4},
                {"name": "idio_momentum_alpha",   "params": make_idio_params(600, 168, k=2), "weight": wi600},
                {"name": "funding_momentum_alpha","params": make_fund_params(144, k=2),       "weight": wf144},
            ],
        )
    else:
        print(f"\n  Best-MIN (AVG>=1.900): {r['avg']}/{r['min']} — P81 best-MIN (1.981/1.260) not beaten.")

if new_avgmax and new_avgmax[4]["avg"] > P81_AVGMAX["avg"]:
    wv1, wk4, wi600, wf144, r = new_avgmax
    yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
    print(f"\n  ★ NEW AVG-MAX CHAMPION: AVG={r['avg']}, MIN={r['min']}")
    print(f"    V1={wv1*100:.2f}%, I437_k4={wk4*100:.2f}%, I600={wi600*100:.2f}%, F144={wf144*100:.2f}%")
    print(f"    YbY={yby_r}")
else:
    print(f"\n  P81 AVG-max (2.079) not surpassed in Sections A/B.")

print("\n" + "=" * 70)
print("PHASE 82 COMPLETE")
print("=" * 70)
