#!/usr/bin/env python3
"""
Phase 84: Fine Grid + I474 Dual-lb + I410 Fine + Triple-lb
===========================================================
Phase 83 BREAKTHROUGH:
  Dual-lb I437+I460 gave:
  - Balanced: V1=27.5%/I437=17.5%/I460=15%/I600=10%/F144=30% → 2.019/1.368
    Strictly dominates P81 balanced (2.010/1.245)!
  - AVG-max: V1=10%/I437=17.5%/I460=17.5%/I600=12.5%/F144=42.5% → 2.181/1.180
  I410_k4 single-lb also strong: 2.014/1.342 at V1=20%, I410=25%, I600=17.5%, F144=37.5%
  I474_k4 standalone: 1.880 (best single lb found) — needs full grid test

Phase 84 agenda:
  A) Fine grid (1.25% steps) around P83 dual-lb balanced champion
  B) I437+I474 dual-lb: does I474 beat I460 in ensemble?
  C) I410 single-lb fine grid (1.25% steps)
  D) Triple-lb: I437+I460+I410 simultaneously
  E) Summary, save new champions
"""

import json, math, statistics, subprocess, sys, copy
from pathlib import Path
from itertools import product

OUT_DIR = "artifacts/phase84"
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
    path = f"/tmp/phase84_{run_name}_{year}.json"
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
    """Generate weight fractions. Default step_10ths=125 → 1.25% steps."""
    return [x / 10000 for x in range(lo_10ths, hi_10ths_exclusive, step_10ths)]

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
print("PHASE 84: Fine Grid + I474 Dual-lb + I410 Fine + Triple-lb")
print("=" * 70)

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

P83_BALANCED = {"avg": 2.019, "min": 1.368}
P83_AVGMAX   = {"avg": 2.181, "min": 1.180}

# ─── Reference runs ───────────────────────────────────────────────────────────
print("\n  Caching reference runs...", flush=True)

v1_data = run_strategy("p84_v1",       "nexus_alpha_v1",      V1_PARAMS)
i437_k4 = run_strategy("p84_i437_k4", "idio_momentum_alpha", make_idio_params(437, 168, k=4))
i460_k4 = run_strategy("p84_i460_k4", "idio_momentum_alpha", make_idio_params(460, 168, k=4))
i474_k4 = run_strategy("p84_i474_k4", "idio_momentum_alpha", make_idio_params(474, 168, k=4))
i410_k4 = run_strategy("p84_i410_k4", "idio_momentum_alpha", make_idio_params(410, 168, k=4))
i600_k2 = run_strategy("p84_i600_k2", "idio_momentum_alpha", make_idio_params(600, 168, k=2))
f144_k2 = run_strategy("p84_f144_k2", "funding_momentum_alpha", make_fund_params(144, k=2))

base = {"v1": v1_data, "i437_k4": i437_k4, "i460_k4": i460_k4, "i474_k4": i474_k4,
        "i410_k4": i410_k4, "i600_k2": i600_k2, "f144_k2": f144_k2}

# Verify P83 balanced champion
p83b_v = blend_ensemble(base, {"v1": 0.275, "i437_k4": 0.175, "i460_k4": 0.15, "i600_k2": 0.10, "f144_k2": 0.30})
p83a_v = blend_ensemble(base, {"v1": 0.10, "i437_k4": 0.175, "i460_k4": 0.175, "i600_k2": 0.125, "f144_k2": 0.425})
print(f"\n  P83 balanced verify: AVG={p83b_v['avg']}, MIN={p83b_v['min']}")
print(f"  P83 AVG-max verify:  AVG={p83a_v['avg']}, MIN={p83a_v['min']}")
print(f"  YbY balanced: {[round(v,3) if v else None for v in p83b_v['yby']]}")
print(f"  YbY AVG-max:  {[round(v,3) if v else None for v in p83a_v['yby']]}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION A: Fine grid (1.25% steps) around P83 dual-lb balanced champion
# P83 balanced: V1=27.5%, I437=17.5%, I460=15%, I600=10%, F144=30%
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("SECTION A: Fine grid (1.25% steps) around P83 dual-lb balanced")
print("  P83 balanced: V1=27.5%, I437=17.5%, I460=15%, I600=10%, F144=30%")
print("═" * 70)

v1_a   = pct_range(2250, 3250, 125)    # 22.5% to 32.5%
i437_a = pct_range(1250, 2125, 125)    # 12.5% to 21.25%
i460_a = pct_range(1000, 1875, 125)    # 10% to 18.75%
i600_a = pct_range(750, 1500, 125)     # 7.5% to 15%

grid_a = []
count_a = 0
for wv1, wi437, wi460, wi600 in product(v1_a, i437_a, i460_a, i600_a):
    wf144 = round(1.0 - wv1 - wi437 - wi460 - wi600, 4)
    if not (0.22 <= wf144 <= 0.45):
        continue
    count_a += 1
    r = blend_ensemble(base, {"v1": wv1, "i437_k4": wi437, "i460_k4": wi460,
                               "i600_k2": wi600, "f144_k2": wf144})
    grid_a.append({
        "v1": wv1, "i437": wi437, "i460": wi460, "i600": wi600, "f144": wf144,
        "avg": r["avg"], "min": r["min"], "yby": r["yby"],
    })

print(f"\n  Tested {count_a} configurations")
grid_a.sort(key=lambda x: x["avg"], reverse=True)

print(f"\n  Top 10 by AVG:")
print(f"  {'V1%':>7} {'I437%':>6} {'I460%':>6} {'I600%':>6} {'F144%':>6}  {'AVG':>7} {'MIN':>7}")
print("  " + "─" * 65)
for r in grid_a[:10]:
    print(f"  {r['v1']*100:>7.2f} {r['i437']*100:>6.2f} {r['i460']*100:>6.2f} {r['i600']*100:>6.2f} {r['f144']*100:>6.2f}  {r['avg']:>7.3f} {r['min']:>7.3f}")

# Best MIN where AVG >= 2.000
high_avg_a = [r for r in grid_a if r["avg"] >= 2.000]
best_min_a = sorted(high_avg_a, key=lambda x: x["min"], reverse=True)
print(f"\n  Best MIN where AVG >= 2.000 ({len(high_avg_a)} configs):")
print(f"  {'V1%':>7} {'I437%':>6} {'I460%':>6} {'I600%':>6} {'F144%':>6}  {'AVG':>7} {'MIN':>7}  YbY")
print("  " + "─" * 80)
for r in best_min_a[:10]:
    yby_r = [round(v,3) if v else None for v in r["yby"]]
    print(f"  {r['v1']*100:>7.2f} {r['i437']*100:>6.2f} {r['i460']*100:>6.2f} {r['i600']*100:>6.2f} {r['f144']*100:>6.2f}  {r['avg']:>7.3f} {r['min']:>7.3f}  {yby_r}")

pareto_a = pareto_filter(grid_a)
print(f"\n  Fine-grid Pareto ({len(pareto_a)} non-dominated, top 15):")
for r in pareto_a[:15]:
    yby_r = [round(v,3) if v else None for v in r["yby"]]
    print(f"  V1={r['v1']*100:.2f}%/I437={r['i437']*100:.2f}%/I460={r['i460']*100:.2f}%/I600={r['i600']*100:.2f}%/F144={r['f144']*100:.2f}%  AVG={r['avg']}, MIN={r['min']}  {yby_r}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION B: I437+I474 dual-lb — does I474 beat I460 in ensemble?
# I474 standalone: AVG=1.880 > I460's 1.828. Profiles differ.
# I460: 2022=0.635, 2023=1.038. I474: 2022=0.883, 2023=0.974.
# I474 has better 2022 but slightly worse 2023 than I460.
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("SECTION B: I437+I474 dual-lb — compare to I437+I460")
print("  I474: 2022=0.883, 2023=0.974 vs I460: 2022=0.635, 2023=1.038")
print("═" * 70)

# Quick comparison at P83 balanced weights
r474_p83 = blend_ensemble(base, {"v1": 0.275, "i437_k4": 0.175, "i474_k4": 0.15, "i600_k2": 0.10, "f144_k2": 0.30})
print(f"\n  P83 balanced weights + I474 (instead of I460): AVG={r474_p83['avg']}, MIN={r474_p83['min']}")
print(f"  YbY: {[round(v,3) if v else None for v in r474_p83['yby']]}")
print(f"  (vs P83 balanced I460: AVG={P83_BALANCED['avg']}, MIN={P83_BALANCED['min']})")

r460_p83 = blend_ensemble(base, {"v1": 0.275, "i437_k4": 0.175, "i460_k4": 0.15, "i600_k2": 0.10, "f144_k2": 0.30})
print(f"  P83 balanced weights + I460:                     AVG={r460_p83['avg']}, MIN={r460_p83['min']}")

# Full 2.5% grid with I437+I474
v1_b   = pct_range(1000, 3000, 250)   # 10% to 27.5%
i437_b = pct_range(500, 2250, 250)    # 5% to 20%
i474_b = pct_range(500, 2250, 250)    # 5% to 20%
i600_b = pct_range(1000, 2250, 250)   # 10% to 20%

grid_b = []
count_b = 0
for wv1, wi437, wi474, wi600 in product(v1_b, i437_b, i474_b, i600_b):
    wf144 = round(1.0 - wv1 - wi437 - wi474 - wi600, 4)
    if not (0.25 <= wf144 <= 0.55):
        continue
    count_b += 1
    r = blend_ensemble(base, {"v1": wv1, "i437_k4": wi437, "i474_k4": wi474,
                               "i600_k2": wi600, "f144_k2": wf144})
    grid_b.append({
        "v1": wv1, "i437": wi437, "i474": wi474, "i600": wi600, "f144": wf144,
        "avg": r["avg"], "min": r["min"], "yby": r["yby"],
    })

print(f"\n  Tested {count_b} I437+I474 configurations")
grid_b.sort(key=lambda x: x["avg"], reverse=True)

print(f"\n  Top 10 by AVG:")
print(f"  {'V1%':>6} {'I437%':>6} {'I474%':>6} {'I600%':>6} {'F144%':>6}  {'AVG':>7} {'MIN':>7}")
print("  " + "─" * 62)
for r in grid_b[:10]:
    print(f"  {r['v1']*100:>6.1f} {r['i437']*100:>6.1f} {r['i474']*100:>6.1f} {r['i600']*100:>6.1f} {r['f144']*100:>6.1f}  {r['avg']:>7.3f} {r['min']:>7.3f}")

high_avg_b = [r for r in grid_b if r["avg"] >= 2.000]
best_min_b = sorted(high_avg_b, key=lambda x: x["min"], reverse=True)
print(f"\n  Best MIN where AVG >= 2.000 ({len(high_avg_b)} I437+I474 configs):")
print(f"  {'V1%':>6} {'I437%':>6} {'I474%':>6} {'I600%':>6} {'F144%':>6}  {'AVG':>7} {'MIN':>7}  YbY")
print("  " + "─" * 78)
for r in best_min_b[:10]:
    yby_r = [round(v,3) if v else None for v in r["yby"]]
    print(f"  {r['v1']*100:>6.1f} {r['i437']*100:>6.1f} {r['i474']*100:>6.1f} {r['i600']*100:>6.1f} {r['f144']*100:>6.1f}  {r['avg']:>7.3f} {r['min']:>7.3f}  {yby_r}")

pareto_b = pareto_filter(grid_b)
print(f"\n  I437+I474 Pareto ({len(pareto_b)} non-dominated, top 10):")
for r in pareto_b[:10]:
    yby_r = [round(v,3) if v else None for v in r["yby"]]
    print(f"  V1={r['v1']*100:.1f}%/I437={r['i437']*100:.1f}%/I474={r['i474']*100:.1f}%/I600={r['i600']*100:.1f}%/F144={r['f144']*100:.1f}%  AVG={r['avg']}, MIN={r['min']}  {yby_r}")

# Best balanced from I437+I474 vs I437+I460
b474_best = best_min_b[0] if best_min_b else None
if b474_best:
    print(f"\n  I437+I474 best balanced: AVG={b474_best['avg']}, MIN={b474_best['min']}")
    print(f"  I437+I460 P83 balanced:  AVG={P83_BALANCED['avg']}, MIN={P83_BALANCED['min']}")
    if b474_best["avg"] > P83_BALANCED["avg"] and b474_best["min"] > P83_BALANCED["min"]:
        print(f"  → I437+I474 STRICTLY DOMINATES I437+I460!")
    elif b474_best["min"] > P83_BALANCED["min"]:
        print(f"  → I437+I474 better MIN (+{b474_best['min']-P83_BALANCED['min']:.3f})")
    elif b474_best["avg"] > P83_BALANCED["avg"]:
        print(f"  → I437+I474 better AVG (+{b474_best['avg']-P83_BALANCED['avg']:.3f})")
    else:
        print(f"  → I437+I460 remains better on both metrics.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION C: I410 single-lb fine grid (1.25% steps)
# P83 found: V1=20%, I410=25%, I600=17.5%, F144=37.5% → 2.014/1.342
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("SECTION C: I410 single-lb fine grid (1.25% steps)")
print("  P83 I410 best: V1=20%, I410=25%, I600=17.5%, F144=37.5% → 2.014/1.342")
print("═" * 70)

v1_c   = pct_range(1250, 2875, 125)   # 12.5% to 28.75%
i410_c = pct_range(1875, 3000, 125)   # 18.75% to 30%
i600_c = pct_range(1250, 2250, 125)   # 12.5% to 22.5%

grid_c = []
count_c = 0
for wv1, wi410, wi600 in product(v1_c, i410_c, i600_c):
    wf144 = round(1.0 - wv1 - wi410 - wi600, 4)
    if not (0.25 <= wf144 <= 0.50):
        continue
    count_c += 1
    r = blend_ensemble(base, {"v1": wv1, "i410_k4": wi410, "i600_k2": wi600, "f144_k2": wf144})
    grid_c.append({
        "v1": wv1, "i410": wi410, "i600": wi600, "f144": wf144,
        "avg": r["avg"], "min": r["min"], "yby": r["yby"],
    })

print(f"\n  Tested {count_c} configurations")
grid_c.sort(key=lambda x: x["avg"], reverse=True)

print(f"\n  Top 10 by AVG:")
print(f"  {'V1%':>7} {'I410k4%':>8} {'I600%':>6} {'F144%':>6}  {'AVG':>7} {'MIN':>7}")
print("  " + "─" * 60)
for r in grid_c[:10]:
    print(f"  {r['v1']*100:>7.2f} {r['i410']*100:>8.2f} {r['i600']*100:>6.2f} {r['f144']*100:>6.2f}  {r['avg']:>7.3f} {r['min']:>7.3f}")

high_avg_c = [r for r in grid_c if r["avg"] >= 2.000]
best_min_c = sorted(high_avg_c, key=lambda x: x["min"], reverse=True)
print(f"\n  Best MIN where AVG >= 2.000 ({len(high_avg_c)} I410 configs):")
print(f"  {'V1%':>7} {'I410k4%':>8} {'I600%':>6} {'F144%':>6}  {'AVG':>7} {'MIN':>7}  YbY")
print("  " + "─" * 75)
for r in best_min_c[:10]:
    yby_r = [round(v,3) if v else None for v in r["yby"]]
    print(f"  {r['v1']*100:>7.2f} {r['i410']*100:>8.2f} {r['i600']*100:>6.2f} {r['f144']*100:>6.2f}  {r['avg']:>7.3f} {r['min']:>7.3f}  {yby_r}")

pareto_c = pareto_filter(grid_c)
print(f"\n  I410 fine-grid Pareto ({len(pareto_c)} non-dominated, top 10):")
for r in pareto_c[:10]:
    yby_r = [round(v,3) if v else None for v in r["yby"]]
    print(f"  V1={r['v1']*100:.2f}%/I410={r['i410']*100:.2f}%/I600={r['i600']*100:.2f}%/F144={r['f144']*100:.2f}%  AVG={r['avg']}, MIN={r['min']}  {yby_r}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION D: Triple-lb: I437+I460+I410 simultaneously
# I437: covers 2022. I460: covers 2023/2025. I410: covers both 2022+2023 (balanced).
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("SECTION D: Triple-lb — I437+I460+I410 simultaneously")
print("  I437: 2022=1.281. I460: 2023=1.038. I410: 2022=1.613, 2023=1.027.")
print("═" * 70)

# Fixed-budget sweep first
print(f"\n  Sweep: V1=17.5%, total_idio=35% split 3-ways, I600=15%, F144 residual")
print(f"  {'I437%':>6} {'I460%':>6} {'I410%':>6} {'F144%':>6}  {'AVG':>7} {'MIN':>7}  YbY")
print("  " + "─" * 70)

for wi437, wi460, wi410 in [(0.175, 0.175, 0.0), (0.1, 0.15, 0.1),
                              (0.1, 0.1, 0.15), (0.05, 0.15, 0.15),
                              (0.1, 0.125, 0.125), (0.125, 0.125, 0.1),
                              (0.075, 0.15, 0.125), (0.1, 0.125, 0.1)]:
    total = wi437 + wi460 + wi410
    if total > 0.36:
        continue
    wf144 = round(1.0 - 0.175 - total - 0.15, 4)
    if wf144 < 0.25 or wf144 > 0.50:
        continue
    r = blend_ensemble(base, {
        "v1": 0.175, "i437_k4": wi437, "i460_k4": wi460, "i410_k4": wi410,
        "i600_k2": 0.15, "f144_k2": wf144
    })
    yby_r = [round(v,3) if v else None for v in r["yby"]]
    print(f"  {wi437*100:>6.1f} {wi460*100:>6.1f} {wi410*100:>6.1f} {wf144*100:>6.1f}  {r['avg']:>7.3f} {r['min']:>7.3f}  {yby_r}")

# Broader triple-lb grid
print(f"\n  Triple-lb 2.5% grid:")
v1_d   = pct_range(1000, 2750, 250)  # 10% to 25%
i437_d = pct_range(500, 2000, 250)   # 5% to 17.5%
i460_d = pct_range(500, 2000, 250)   # 5% to 17.5%
i410_d = pct_range(500, 1750, 250)   # 5% to 15%
i600_d = pct_range(1000, 2250, 250)  # 10% to 20%

grid_d = []
count_d = 0
for wv1, wi437, wi460, wi410, wi600 in product(v1_d, i437_d, i460_d, i410_d, i600_d):
    wf144 = round(1.0 - wv1 - wi437 - wi460 - wi410 - wi600, 4)
    if not (0.30 <= wf144 <= 0.55):
        continue
    count_d += 1
    r = blend_ensemble(base, {
        "v1": wv1, "i437_k4": wi437, "i460_k4": wi460, "i410_k4": wi410,
        "i600_k2": wi600, "f144_k2": wf144
    })
    grid_d.append({
        "v1": wv1, "i437": wi437, "i460": wi460, "i410": wi410,
        "i600": wi600, "f144": wf144,
        "avg": r["avg"], "min": r["min"], "yby": r["yby"],
    })

print(f"\n  Tested {count_d} triple-lb configurations")
grid_d.sort(key=lambda x: x["avg"], reverse=True)

high_avg_d = [r for r in grid_d if r["avg"] >= 2.000]
best_min_d = sorted(high_avg_d, key=lambda x: x["min"], reverse=True)
print(f"\n  Top 10 by AVG:")
print(f"  {'V1%':>5} {'I437%':>6} {'I460%':>6} {'I410%':>6} {'I600%':>6} {'F144%':>6}  {'AVG':>7} {'MIN':>7}")
print("  " + "─" * 68)
for r in grid_d[:10]:
    print(f"  {r['v1']*100:>5.1f} {r['i437']*100:>6.1f} {r['i460']*100:>6.1f} {r['i410']*100:>6.1f} {r['i600']*100:>6.1f} {r['f144']*100:>6.1f}  {r['avg']:>7.3f} {r['min']:>7.3f}")

print(f"\n  Best MIN where AVG >= 2.000 ({len(high_avg_d)} triple-lb configs):")
print(f"  {'V1%':>5} {'I437%':>6} {'I460%':>6} {'I410%':>6} {'I600%':>6} {'F144%':>6}  {'AVG':>7} {'MIN':>7}  YbY")
print("  " + "─" * 80)
for r in best_min_d[:10]:
    yby_r = [round(v,3) if v else None for v in r["yby"]]
    print(f"  {r['v1']*100:>5.1f} {r['i437']*100:>6.1f} {r['i460']*100:>6.1f} {r['i410']*100:>6.1f} {r['i600']*100:>6.1f} {r['f144']*100:>6.1f}  {r['avg']:>7.3f} {r['min']:>7.3f}  {yby_r}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION E: Summary — find best new champions
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("SECTION E: Phase 84 Summary")
print("═" * 70)

print(f"\n  Prior champions:")
print(f"    P83 balanced: AVG={P83_BALANCED['avg']}, MIN={P83_BALANCED['min']}")
print(f"    P83 AVG-max:  AVG={P83_AVGMAX['avg']}, MIN={P83_AVGMAX['min']}")

# Collect all candidates
all_balanced_cands = []
for r in grid_a + grid_b + grid_c + grid_d:
    if r["avg"] >= 2.000:
        all_balanced_cands.append(r)

new_balanced = max(all_balanced_cands, key=lambda x: x["min"]) if all_balanced_cands else None
new_avgmax = max(grid_a + grid_b + grid_c + grid_d, key=lambda x: x["avg"]) if grid_a or grid_b or grid_c or grid_d else None

if new_balanced:
    yby_r = [round(v,3) if v else None for v in new_balanced["yby"]]
    flag = ""
    if new_balanced["avg"] > P83_BALANCED["avg"] and new_balanced["min"] > P83_BALANCED["min"]:
        flag = " → STRICTLY DOMINATES P83 balanced!"
    elif new_balanced["min"] > P83_BALANCED["min"]:
        flag = f" → Better MIN (+{new_balanced['min']-P83_BALANCED['min']:.3f})"
    else:
        flag = " → P83 balanced remains champion"
    print(f"\n  ★ NEW BALANCED (AVG>=2.000, best MIN): AVG={new_balanced['avg']}, MIN={new_balanced['min']}{flag}")
    # Build readable config string
    parts = [f"V1={new_balanced['v1']*100:.2f}%"]
    for k in ["i437", "i460", "i474", "i410"]:
        if new_balanced.get(k, 0) > 0:
            parts.append(f"I{k[1:].upper()}={new_balanced[k]*100:.2f}%")
    parts += [f"I600={new_balanced['i600']*100:.2f}%", f"F144={new_balanced['f144']*100:.2f}%"]
    print(f"    {', '.join(parts)}")
    print(f"    YbY={yby_r}")

if new_avgmax and new_avgmax["avg"] > P83_AVGMAX["avg"]:
    yby_r = [round(v,3) if v else None for v in new_avgmax["yby"]]
    print(f"\n  ★ NEW AVG-MAX CHAMPION: AVG={new_avgmax['avg']}, MIN={new_avgmax['min']}")
    print(f"    YbY={yby_r}")
else:
    top_all = max((grid_a + grid_b + grid_c + grid_d), key=lambda x: x["avg"]) if grid_a or grid_b or grid_c or grid_d else None
    if top_all:
        print(f"\n  Best AVG across all sections: {top_all['avg']}/{top_all['min']} (P83 AVG-max 2.181 {'beaten' if top_all['avg'] > P83_AVGMAX['avg'] else 'not beaten'})")

# Save champions
def get_strategy_list_84(r):
    strategies = [{"name": "nexus_alpha_v1", "params": V1_PARAMS, "weight": r["v1"]}]
    for lb, k in [(437, 4), (460, 4), (474, 4), (410, 4)]:
        key = f"i{lb}"
        if r.get(key, 0) > 0:
            strategies.append({"name": "idio_momentum_alpha", "params": make_idio_params(lb, 168, k=4), "weight": r[key]})
    strategies.append({"name": "idio_momentum_alpha", "params": make_idio_params(600, 168, k=2), "weight": r["i600"]})
    strategies.append({"name": "funding_momentum_alpha", "params": make_fund_params(144, k=2), "weight": r["f144"]})
    return strategies

print("\n  Saving champions:")
if new_balanced and (new_balanced["avg"] > P83_BALANCED["avg"] or new_balanced["min"] > P83_BALANCED["min"]):
    yby_r = [round(v,3) if v else None for v in new_balanced["yby"]]
    save_champion(
        "ensemble_p84_balanced",
        f"Phase 84 balanced: AVG={new_balanced['avg']}, MIN={new_balanced['min']}. YbY={yby_r}",
        get_strategy_list_84(new_balanced),
    )
else:
    print("  P83 balanced remains best balanced champion (2.019/1.368).")

if new_avgmax and new_avgmax["avg"] > P83_AVGMAX["avg"]:
    yby_r = [round(v,3) if v else None for v in new_avgmax["yby"]]
    save_champion(
        "ensemble_p84_avgmax",
        f"Phase 84 AVG-max: AVG={new_avgmax['avg']}, MIN={new_avgmax['min']}. YbY={yby_r}",
        get_strategy_list_84(new_avgmax),
    )
else:
    print("  P83 AVG-max (2.181) remains best AVG-max champion.")

print("\n" + "=" * 70)
print("PHASE 84 COMPLETE")
print("=" * 70)
