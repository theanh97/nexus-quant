#!/usr/bin/env python3
"""
Phase 83: I460_k4 Full Grid + Dual-Lookback k4 Exploration
============================================================
Phase 82 discoveries:
  I460_k4 standalone: AVG=1.828 >> I437_k4's 1.623 (best k4 standalone found!)
    2021=2.595, 2022=0.635, 2023=1.038, 2024=2.709, 2025=2.162
    vs I437_k4: 2021=2.040, 2022=1.281, 2023=0.563, 2024=2.398, 2025=1.834
  In P81 weights: I460_k4 → 2.067/1.222 (vs I437_k4 → 2.010/1.245)
  I410_k4: AVG=1.567, 2022=1.613, 2023=1.027 (balanced 2022+2023 profile)

Year bottleneck analysis:
  I437_k4 ensemble: bottleneck = 2023 (0.563 standalone → MIN~1.245)
  I460_k4 ensemble: bottleneck = 2022 (0.635 standalone → MIN~1.222)
  I437+I460 dual: covers both 2022 (I437=1.281) and 2023 (I460=1.038)
  I410_k4: covers both 2022 (1.613) and 2023 (1.027) in a single signal!

Phase 83 agenda:
  A) Reference runs: confirm I460, I410, I437 profiles; fine lb tune (450-470)
  B) I460_k4 full 2.5% grid — map the complete I460 Pareto frontier
  C) Dual-lookback k4: I437_k4 + I460_k4 simultaneously
     - Can we get AVG>2.05 AND MIN>1.240?
  D) I410_k4 in ensemble — this balanced lb might be the new champion
  E) Triple-lb: I437 + I460 + I410 (all k=4) — if dual improves, does triple too?
  F) Summary, save new champions
"""

import json, math, statistics, subprocess, sys, copy
from pathlib import Path
from itertools import product

OUT_DIR = "artifacts/phase83"
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
    path = f"/tmp/phase83_{run_name}_{year}.json"
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

def pct_range(lo_10ths: int, hi_10ths_exclusive: int, step_10ths: int = 250) -> list:
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

def dominates(a, b):
    return a["avg"] > b["avg"] and a["min"] > b["min"]

def pareto_filter(results: list, avg_key="avg", min_key="min") -> list:
    nd = []
    for i, ri in enumerate(results):
        dominated = any(
            j != i and results[j][avg_key] >= ri[avg_key] and results[j][min_key] >= ri[min_key]
            and (results[j][avg_key] > ri[avg_key] or results[j][min_key] > ri[min_key])
            for j in range(len(results))
        )
        if not dominated:
            nd.append(ri)
    return sorted(nd, key=lambda x: x[avg_key], reverse=True)

# ═════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PHASE 83: I460_k4 Full Grid + Dual-Lookback k4 Exploration")
print("=" * 70)

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

P81_BALANCED = {"avg": 2.010, "min": 1.245}
P82_HIGHMIN  = {"avg": 2.000, "min": 1.292}
P82_BESTMIN  = {"avg": 1.901, "min": 1.349}
P81_AVGMAX   = {"avg": 2.079, "min": 1.015}

# ─── Reference runs ───────────────────────────────────────────────────────────
print("\n  Caching reference runs...", flush=True)

v1_data = run_strategy("p83_v1",       "nexus_alpha_v1",      V1_PARAMS)
i437_k4 = run_strategy("p83_i437_k4", "idio_momentum_alpha", make_idio_params(437, 168, k=4))
i460_k4 = run_strategy("p83_i460_k4", "idio_momentum_alpha", make_idio_params(460, 168, k=4))
i410_k4 = run_strategy("p83_i410_k4", "idio_momentum_alpha", make_idio_params(410, 168, k=4))
i600_k2 = run_strategy("p83_i600_k2", "idio_momentum_alpha", make_idio_params(600, 168, k=2))
f144_k2 = run_strategy("p83_f144_k2", "funding_momentum_alpha", make_fund_params(144, k=2))

base = {"v1": v1_data, "i437_k4": i437_k4, "i460_k4": i460_k4,
        "i410_k4": i410_k4, "i600_k2": i600_k2, "f144_k2": f144_k2}

# Quick verifications
p81b_v = blend_ensemble(base, {"v1": 0.225, "i437_k4": 0.225, "i600_k2": 0.15, "f144_k2": 0.40})
p82_v = blend_ensemble(base, {"v1": 0.225, "i460_k4": 0.225, "i600_k2": 0.15, "f144_k2": 0.40})
print(f"\n  P81 balanced (I437) verify: AVG={p81b_v['avg']}, MIN={p81b_v['min']}")
print(f"  I460 at P81 weights:        AVG={p82_v['avg']}, MIN={p82_v['min']}")
print(f"  YbY I437: {[round(v,3) if v else None for v in p81b_v['yby']]}")
print(f"  YbY I460: {[round(v,3) if v else None for v in p82_v['yby']]}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION A: Fine lookback tuning around 460h with k=4
# I460 standalone: AVG=1.828. Are 450h, 455h, 465h, 470h better?
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("SECTION A: Fine lookback tuning around I460_k4")
print("  I460 standalone: AVG=1.828. Fine tune ±15h.")
print("═" * 70)

fine_lbs = [447, 453, 460, 467, 474, 481]
fine_lb_data = {460: i460_k4}  # already cached

print(f"\n  {'LB':>5}  {'AVG':>7} {'MIN':>7}  YbY")
print("  " + "─" * 65)
for lb in fine_lbs:
    if lb == 460:
        d = i460_k4
    else:
        d = run_strategy(f"p83_i{lb}_k4", "idio_momentum_alpha", make_idio_params(lb, 168, k=4))
        fine_lb_data[lb] = d
    avg_ = d["_avg"]
    mn_  = d["_min"]
    yby_ = [round(d.get(y, {}).get("sharpe", 0), 3) for y in YEARS]
    flag = " ★" if avg_ >= 1.828 else ""
    print(f"  {lb:>5}  {avg_:>7.3f} {mn_:>7.3f}  {yby_}{flag}")

# Find best standalone lb for k=4
best_fine_lb = max(fine_lb_data.keys(), key=lambda lb: fine_lb_data[lb]["_avg"])
print(f"\n  Best lb in fine sweep: {best_fine_lb}h → AVG={fine_lb_data[best_fine_lb]['_avg']}")
if best_fine_lb != 460:
    base[f"i{best_fine_lb}_k4"] = fine_lb_data[best_fine_lb]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION B: I460_k4 full 2.5% grid — map Pareto frontier for I460
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("SECTION B: I460_k4 full 2.5% grid — Pareto frontier")
print("  (same grid structure as Phase 81 Section B but with I460)")
print("═" * 70)

v1_vals   = pct_range(1000, 3500, 250)   # 10% to 32.5%
i460_vals = pct_range(750, 3000, 250)    # 7.5% to 27.5%
i600_vals = pct_range(1000, 2500, 250)   # 10% to 22.5%
f168_vals = [0.0, 0.025, 0.05]

grid_b = []
count_b = 0
for wv1, wi460, wi600, wf168 in product(v1_vals, i460_vals, i600_vals, f168_vals):
    wf144 = round(1.0 - wv1 - wi460 - wi600 - wf168, 4)
    if not (0.30 <= wf144 <= 0.55):
        continue
    count_b += 1
    r = blend_ensemble(base, {"v1": wv1, "i460_k4": wi460, "i600_k2": wi600,
                               "f144_k2": wf144, "f168_k2": wf168 if wf168 > 0 else 0.0})
    grid_b.append({
        "v1": wv1, "i460": wi460, "i600": wi600, "f144": wf144, "f168": wf168,
        "avg": r["avg"], "min": r["min"], "yby": r["yby"],
    })

print(f"\n  Tested {count_b} configurations")
grid_b.sort(key=lambda x: x["avg"], reverse=True)

print(f"\n  Top 15 by AVG:")
print(f"  {'V1%':>6} {'I460k4%':>8} {'I600%':>6} {'F144%':>6} {'F168%':>5}  {'AVG':>7} {'MIN':>7}")
print("  " + "─" * 70)
for r in grid_b[:15]:
    print(f"  {r['v1']*100:>6.1f} {r['i460']*100:>8.1f} {r['i600']*100:>6.1f} {r['f144']*100:>6.1f} {r['f168']*100:>5.1f}  {r['avg']:>7.3f} {r['min']:>7.3f}")

# Best MIN where AVG >= 2.000
high_avg_b = [r for r in grid_b if r["avg"] >= 2.000]
best_min_b = sorted(high_avg_b, key=lambda x: x["min"], reverse=True)
print(f"\n  Best MIN where AVG >= 2.000 ({len(high_avg_b)} configs):")
print(f"  {'V1%':>6} {'I460k4%':>8} {'I600%':>6} {'F144%':>6} {'F168%':>5}  {'AVG':>7} {'MIN':>7}  YbY")
print("  " + "─" * 80)
for r in best_min_b[:10]:
    yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
    print(f"  {r['v1']*100:>6.1f} {r['i460']*100:>8.1f} {r['i600']*100:>6.1f} {r['f144']*100:>6.1f} {r['f168']*100:>5.1f}  {r['avg']:>7.3f} {r['min']:>7.3f}  {yby_r}")

# Best MIN where AVG >= 1.900
high_avg_b2 = [r for r in grid_b if r["avg"] >= 1.900]
best_min_b2 = sorted(high_avg_b2, key=lambda x: x["min"], reverse=True)
print(f"\n  Best MIN where AVG >= 1.900 ({len(high_avg_b2)} configs, top 10 by MIN):")
print(f"  {'V1%':>6} {'I460k4%':>8} {'I600%':>6} {'F144%':>6} {'F168%':>5}  {'AVG':>7} {'MIN':>7}  YbY")
print("  " + "─" * 80)
for r in best_min_b2[:10]:
    yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
    print(f"  {r['v1']*100:>6.1f} {r['i460']*100:>8.1f} {r['i600']*100:>6.1f} {r['f144']*100:>6.1f} {r['f168']*100:>5.1f}  {r['avg']:>7.3f} {r['min']:>7.3f}  {yby_r}")

# I460 Pareto frontier
pareto_b = pareto_filter(grid_b)
print(f"\n  I460_k4 Pareto frontier ({len(pareto_b)} non-dominated):")
print(f"  {'V1%':>6} {'I460k4%':>8} {'I600%':>6} {'F144%':>6} {'F168%':>5}  {'AVG':>7} {'MIN':>7}  YbY")
print("  " + "─" * 80)
for r in pareto_b[:20]:
    yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
    print(f"  {r['v1']*100:>6.1f} {r['i460']*100:>8.1f} {r['i600']*100:>6.1f} {r['f144']*100:>6.1f} {r['f168']*100:>5.1f}  {r['avg']:>7.3f} {r['min']:>7.3f}  {yby_r}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION C: Dual-lookback k4: I437_k4 + I460_k4 simultaneously
# I437: better 2022 (1.281 vs 0.635); I460: better 2023 (1.038 vs 0.563)
# Can combining both cover the year bottleneck?
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("SECTION C: Dual-lookback k4 — I437_k4 + I460_k4 simultaneously")
print("  I437: 2022=1.281, 2023=0.563. I460: 2022=0.635, 2023=1.038.")
print("  Goal: cover both years. Can we get AVG>2.05 AND MIN>1.250?")
print("═" * 70)

# Fixed-budget sweep: total idio budget = 35% (sum of i437+i460)
# V1=17.5%, I600=15%, F144=32.5% (fixed), split the 35% between i437 and i460
print(f"\n  Sweep: fix V1=17.5%, total_idio=35% split between I437/I460, I600=15%, F144 residual")
print(f"  {'I437%':>7} {'I460%':>7} {'I600%':>6} {'F144%':>6}  {'AVG':>7} {'MIN':>7}  YbY")
print("  " + "─" * 75)
for wi437 in [0.0, 0.05, 0.10, 0.15, 0.175, 0.20, 0.25, 0.30, 0.35]:
    wi460 = 0.35 - wi437
    wf144 = 1.0 - 0.175 - 0.35 - 0.15
    if wf144 < 0.30:
        continue
    r = blend_ensemble(base, {
        "v1": 0.175, "i437_k4": wi437, "i460_k4": wi460,
        "i600_k2": 0.15, "f144_k2": round(wf144, 4)
    })
    yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
    print(f"  {wi437*100:>7.1f} {wi460*100:>7.1f} {15.0:>6.1f} {wf144*100:>6.1f}  {r['avg']:>7.3f} {r['min']:>7.3f}  {yby_r}")

# Full dual-lb grid: V1, I437, I460, I600, F144
print(f"\n  Full dual-lb 2.5% grid (I437+I460 simultaneously):")

v1_d   = pct_range(1000, 3000, 250)    # 10% to 27.5%
i437_d = pct_range(500, 2000, 250)     # 5% to 17.5%
i460_d = pct_range(500, 2000, 250)     # 5% to 17.5%
i600_d = pct_range(1000, 2250, 250)    # 10% to 20%

grid_c = []
count_c = 0
for wv1, wi437, wi460, wi600 in product(v1_d, i437_d, i460_d, i600_d):
    wf144 = round(1.0 - wv1 - wi437 - wi460 - wi600, 4)
    if not (0.30 <= wf144 <= 0.55):
        continue
    count_c += 1
    r = blend_ensemble(base, {
        "v1": wv1, "i437_k4": wi437, "i460_k4": wi460,
        "i600_k2": wi600, "f144_k2": wf144
    })
    grid_c.append({
        "v1": wv1, "i437": wi437, "i460": wi460, "i600": wi600, "f144": wf144,
        "avg": r["avg"], "min": r["min"], "yby": r["yby"],
    })

print(f"\n  Tested {count_c} dual-lb configurations")
grid_c.sort(key=lambda x: x["avg"], reverse=True)

print(f"\n  Top 10 by AVG:")
print(f"  {'V1%':>6} {'I437%':>6} {'I460%':>6} {'I600%':>6} {'F144%':>6}  {'AVG':>7} {'MIN':>7}")
print("  " + "─" * 68)
for r in grid_c[:10]:
    print(f"  {r['v1']*100:>6.1f} {r['i437']*100:>6.1f} {r['i460']*100:>6.1f} {r['i600']*100:>6.1f} {r['f144']*100:>6.1f}  {r['avg']:>7.3f} {r['min']:>7.3f}")

high_avg_c = [r for r in grid_c if r["avg"] >= 2.000]
best_min_c = sorted(high_avg_c, key=lambda x: x["min"], reverse=True)
print(f"\n  Best MIN where AVG >= 2.000 ({len(high_avg_c)} dual-lb configs):")
print(f"  {'V1%':>6} {'I437%':>6} {'I460%':>6} {'I600%':>6} {'F144%':>6}  {'AVG':>7} {'MIN':>7}  YbY")
print("  " + "─" * 80)
for r in best_min_c[:10]:
    yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
    print(f"  {r['v1']*100:>6.1f} {r['i437']*100:>6.1f} {r['i460']*100:>6.1f} {r['i600']*100:>6.1f} {r['f144']*100:>6.1f}  {r['avg']:>7.3f} {r['min']:>7.3f}  {yby_r}")

# Pareto for dual-lb
pareto_c = pareto_filter(grid_c)
print(f"\n  Dual-lb Pareto ({len(pareto_c)} non-dominated, top 15):")
for r in pareto_c[:15]:
    yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
    print(f"  V1={r['v1']*100:.1f}%/I437={r['i437']*100:.1f}%/I460={r['i460']*100:.1f}%/I600={r['i600']*100:.1f}%/F144={r['f144']*100:.1f}%  AVG={r['avg']}, MIN={r['min']}  {yby_r}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION D: I410_k4 in ensemble
# I410_k4: AVG=1.567, 2022=1.613, 2023=1.027 (balanced 2022+2023!)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("SECTION D: I410_k4 in ensemble")
print("  I410_k4: 2022=1.613, 2023=1.027 — much more balanced than I437 or I460!")
print("═" * 70)

# I410 at P81 balanced weights
r410_p81 = blend_ensemble(base, {"v1": 0.225, "i410_k4": 0.225, "i600_k2": 0.15, "f144_k2": 0.40})
print(f"\n  P81 weights + I410_k4: AVG={r410_p81['avg']}, MIN={r410_p81['min']}")
print(f"  YbY: {[round(v,3) if v else None for v in r410_p81['yby']]}")
print(f"  (vs P81 balanced I437: AVG={P81_BALANCED['avg']}, MIN={P81_BALANCED['min']})")
print(f"  (vs I460 at P81 wts:   AVG={p82_v['avg']}, MIN={p82_v['min']})")

# I410 weight sweep at P81 balanced V1/I600/F144
print(f"\n  I410_k4 weight sweep (V1=22.5%, I600=15% fixed):")
print(f"  {'I410k4%':>9} {'F144%':>7}  {'AVG':>7} {'MIN':>7}  YbY")
print("  " + "─" * 65)
for wi410 in [0.10, 0.15, 0.175, 0.20, 0.225, 0.25, 0.275, 0.30]:
    wf144 = round(1.0 - 0.225 - wi410 - 0.15, 4)
    if wf144 < 0.30 or wf144 > 0.55:
        continue
    r = blend_ensemble(base, {"v1": 0.225, "i410_k4": wi410, "i600_k2": 0.15, "f144_k2": wf144})
    yby_r = [round(v,3) if v else None for v in r["yby"]]
    print(f"  {wi410*100:>9.1f} {wf144*100:>7.1f}  {r['avg']:>7.3f} {r['min']:>7.3f}  {yby_r}")

# I410 full 2.5% grid
print(f"\n  I410_k4 full 2.5% grid:")
v1_d4   = pct_range(1000, 3000, 250)
i410_d4 = pct_range(750, 2750, 250)
i600_d4 = pct_range(1000, 2250, 250)

grid_d = []
count_d = 0
for wv1, wi410, wi600 in product(v1_d4, i410_d4, i600_d4):
    wf144 = round(1.0 - wv1 - wi410 - wi600, 4)
    if not (0.30 <= wf144 <= 0.55):
        continue
    count_d += 1
    r = blend_ensemble(base, {"v1": wv1, "i410_k4": wi410, "i600_k2": wi600, "f144_k2": wf144})
    grid_d.append({
        "v1": wv1, "i410": wi410, "i600": wi600, "f144": wf144,
        "avg": r["avg"], "min": r["min"], "yby": r["yby"],
    })

print(f"\n  Tested {count_d} configurations")
grid_d.sort(key=lambda x: x["avg"], reverse=True)

print(f"\n  Top 10 by AVG:")
print(f"  {'V1%':>6} {'I410k4%':>8} {'I600%':>6} {'F144%':>6}  {'AVG':>7} {'MIN':>7}")
print("  " + "─" * 60)
for r in grid_d[:10]:
    print(f"  {r['v1']*100:>6.1f} {r['i410']*100:>8.1f} {r['i600']*100:>6.1f} {r['f144']*100:>6.1f}  {r['avg']:>7.3f} {r['min']:>7.3f}")

high_avg_d = [r for r in grid_d if r["avg"] >= 2.000]
best_min_d = sorted(high_avg_d, key=lambda x: x["min"], reverse=True)
print(f"\n  Best MIN where AVG >= 2.000 ({len(high_avg_d)} I410 configs):")
print(f"  {'V1%':>6} {'I410k4%':>8} {'I600%':>6} {'F144%':>6}  {'AVG':>7} {'MIN':>7}  YbY")
print("  " + "─" * 75)
for r in best_min_d[:10]:
    yby_r = [round(v,3) if v else None for v in r["yby"]]
    print(f"  {r['v1']*100:>6.1f} {r['i410']*100:>8.1f} {r['i600']*100:>6.1f} {r['f144']*100:>6.1f}  {r['avg']:>7.3f} {r['min']:>7.3f}  {yby_r}")

pareto_d = pareto_filter(grid_d)
print(f"\n  I410_k4 Pareto frontier ({len(pareto_d)} non-dominated, top 15):")
for r in pareto_d[:15]:
    yby_r = [round(v,3) if v else None for v in r["yby"]]
    print(f"  V1={r['v1']*100:.1f}%/I410={r['i410']*100:.1f}%/I600={r['i600']*100:.1f}%/F144={r['f144']*100:.1f}%  AVG={r['avg']}, MIN={r['min']}  {yby_r}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION E: Summary — cross-lb Pareto comparison
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("SECTION E: Phase 83 Summary — cross-lb Pareto comparison")
print("═" * 70)

print(f"\n  Prior champions:")
print(f"    P81 balanced (I437): AVG={P81_BALANCED['avg']}, MIN={P81_BALANCED['min']}")
print(f"    P82 high-MIN (I437): AVG={P82_HIGHMIN['avg']}, MIN={P82_HIGHMIN['min']}")
print(f"    P82 best-MIN (I437): AVG={P82_BESTMIN['avg']}, MIN={P82_BESTMIN['min']}")
print(f"    P81 AVG-max (I437):  AVG={P81_AVGMAX['avg']}, MIN={P81_AVGMAX['min']}")

# Find overall best balanced (AVG >= 2.000, best MIN)
all_grid = (
    [{"lb": "I460", **r} for r in grid_b] +
    [{"lb": "I437+I460", **r} for r in grid_c] +
    [{"lb": "I410", **r} for r in grid_d]
)

new_balanced_cands = sorted(
    [r for r in all_grid if r["avg"] >= 2.000],
    key=lambda x: x["min"], reverse=True
)
new_avgmax_cands = sorted(all_grid, key=lambda x: x["avg"], reverse=True)

if new_balanced_cands:
    best = new_balanced_cands[0]
    yby_r = [round(v, 3) if v is not None else None for v in best["yby"]]
    lb = best.get("lb", "?")
    if best.get("i460") is not None and best.get("i437") is not None:
        cfg_str = f"V1={best['v1']*100:.1f}%/I437={best['i437']*100:.1f}%/I460={best['i460']*100:.1f}%/I600={best['i600']*100:.1f}%/F144={best['f144']*100:.1f}%"
    elif "i460" in best:
        cfg_str = f"V1={best['v1']*100:.1f}%/I460={best['i460']*100:.1f}%/I600={best['i600']*100:.1f}%/F144={best['f144']*100:.1f}%"
    elif "i410" in best:
        cfg_str = f"V1={best['v1']*100:.1f}%/I410={best['i410']*100:.1f}%/I600={best['i600']*100:.1f}%/F144={best['f144']*100:.1f}%"
    else:
        cfg_str = str(best)
    flag = ""
    if best["avg"] > P81_BALANCED["avg"] and best["min"] > P81_BALANCED["min"]:
        flag = " → STRICTLY DOMINATES P81 balanced!"
    elif best["min"] > P82_HIGHMIN["min"]:
        flag = f" → New best MIN! (+{best['min']-P82_HIGHMIN['min']:.3f} vs P82 high-MIN)"
    elif best["min"] > P81_BALANCED["min"]:
        flag = f" → Better MIN than P81 balanced (+{best['min']-P81_BALANCED['min']:.3f})"
    print(f"\n  ★ NEW BALANCED CHAMPION ({lb}): AVG={best['avg']}, MIN={best['min']}{flag}")
    print(f"    {cfg_str}")
    print(f"    YbY={yby_r}")

# Save the best balanced champion
# (For flexibility we handle I460, I410, and dual-lb cases)
def get_strategy_list(r):
    strategies = [{"name": "nexus_alpha_v1", "params": V1_PARAMS, "weight": r["v1"]}]
    if "i437" in r and r.get("i437", 0) > 0:
        strategies.append({"name": "idio_momentum_alpha", "params": make_idio_params(437, 168, k=4), "weight": r["i437"]})
    if "i460" in r and r.get("i460", 0) > 0:
        strategies.append({"name": "idio_momentum_alpha", "params": make_idio_params(460, 168, k=4), "weight": r["i460"]})
    if "i410" in r and r.get("i410", 0) > 0:
        strategies.append({"name": "idio_momentum_alpha", "params": make_idio_params(410, 168, k=4), "weight": r["i410"]})
    strategies.append({"name": "idio_momentum_alpha", "params": make_idio_params(600, 168, k=2), "weight": r["i600"]})
    strategies.append({"name": "funding_momentum_alpha", "params": make_fund_params(144, k=2), "weight": r["f144"]})
    if r.get("f168", 0) > 0:
        strategies.append({"name": "funding_momentum_alpha", "params": make_fund_params(168, k=2), "weight": r["f168"]})
    return strategies

if new_balanced_cands:
    best = new_balanced_cands[0]
    yby_r = [round(v, 3) if v is not None else None for v in best["yby"]]
    save_champion(
        "ensemble_p83_balanced",
        f"Phase 83 balanced: AVG={best['avg']}, MIN={best['min']}. YbY={yby_r}",
        get_strategy_list(best),
    )

# Save best AVG-max that beats P81 AVG-max
if new_avgmax_cands and new_avgmax_cands[0]["avg"] > P81_AVGMAX["avg"]:
    best_am = new_avgmax_cands[0]
    yby_r = [round(v, 3) if v is not None else None for v in best_am["yby"]]
    print(f"\n  ★ NEW AVG-MAX CHAMPION: AVG={best_am['avg']}, MIN={best_am['min']}")
    print(f"    YbY={yby_r}")
    save_champion(
        "ensemble_p83_avgmax",
        f"Phase 83 AVG-max: AVG={best_am['avg']}, MIN={best_am['min']}. YbY={yby_r}",
        get_strategy_list(best_am),
    )
else:
    print(f"\n  P81 AVG-max (2.079) not surpassed in this phase.")

print("\n" + "=" * 70)
print("PHASE 83 COMPLETE")
print("=" * 70)
