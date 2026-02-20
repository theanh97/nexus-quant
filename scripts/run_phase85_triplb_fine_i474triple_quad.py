#!/usr/bin/env python3
"""
Phase 85: Triple-lb Fine Grid + I474 Triple-lb + I437+I474 Fine + Quad-lb
==========================================================================
Phase 84 BREAKTHROUGHS:
  1. Triple-lb BALANCED (STRICTLY DOMINATES P83):
       V1=25%, I437_k4=5%, I460_k4=12.5%, I410_k4=15%, I600=10%, F144=32.5%
       AVG=2.040, MIN=1.431, YbY=[3.303, 1.452, 1.431, 2.507, 1.505] — ALL >1.4!
  2. I437+I474 dual-lb beats I437+I460 dual-lb:
       I437+I474 balanced: 2.046/1.377 vs I437+I460: 2.019/1.368
       I437+I474 AVG-max:  2.224/1.177 (V1=10%, I437=20%, I474=20%, I600=10%, F144=40%)

Phase 85 agenda:
  A) Fine 1.25% grid around triple-lb balanced champion (I437+I460+I410)
  B) Triple-lb with I474: I437+I474+I410 (swap I460→I474 in triple-lb)
  C) I437+I474 dual-lb fine grid (1.25% steps) → Pareto map
  D) Quad-lb: I437+I460+I474+I410 — can we push MIN above 1.44?
  E) Summary, save new champions
"""

import json, math, statistics, subprocess, sys, copy
from pathlib import Path
from itertools import product

OUT_DIR = "artifacts/phase85"
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
    path = f"/tmp/phase85_{run_name}_{year}.json"
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
        blended = [
            sum(w * rets[i] for w, rets in pairs)
            for i in range(min_len)
        ]
        mu = statistics.mean(blended)
        sd = statistics.pstdev(blended)
        sharpe = (mu / sd) * math.sqrt(8760) if sd > 0 else 0.0
        year_sharpes[year] = round(sharpe, 4)

    valid_sharpes = [v for v in year_sharpes.values() if v is not None]
    avg = round(sum(valid_sharpes) / len(valid_sharpes), 4) if valid_sharpes else 0.0
    mn = round(min(valid_sharpes), 4) if valid_sharpes else 0.0
    pos = sum(1 for s in valid_sharpes if s > 0)
    yby = [year_sharpes.get(y) for y in YEARS]
    return {"avg": avg, "min": mn, "pos": pos, "yby": yby}

def pct_range(lo_10ths: int, hi_10ths_exclusive: int, step_10ths: int = 125) -> list:
    """Generate weight range from integers representing 10000ths.
    e.g. pct_range(1000, 3000, 250) → [0.10, 0.125, ..., 0.2875]"""
    return [x / 10000 for x in range(lo_10ths, hi_10ths_exclusive, step_10ths)]

def pareto_filter(results: list) -> list:
    """Return non-dominated configs: (avg, min, config_dict)"""
    dominated = set()
    for i, (ai, mi, _) in enumerate(results):
        for j, (aj, mj, _) in enumerate(results):
            if i == j:
                continue
            if aj >= ai and mj >= mi and (aj > ai or mj > mi):
                dominated.add(i)
                break
    return [results[i] for i in range(len(results)) if i not in dominated]

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("PHASE 85: Triple-lb Fine + I474-Triple + Dual-lb Fine + Quad-lb")
print("=" * 70)

# ─── Reference strategy runs (reuse across all sections) ─────────────────────
print("\n" + "═"*70)
print("REFERENCE RUNS")
print("═"*70)

v1_data    = run_strategy("p85_v1",        "nexus_alpha_v1",        V1_PARAMS)
i437_k4    = run_strategy("p85_i437_k4",   "idio_momentum_alpha",   make_idio_params(437, 168, k=4))
i460_k4    = run_strategy("p85_i460_k4",   "idio_momentum_alpha",   make_idio_params(460, 168, k=4))
i474_k4    = run_strategy("p85_i474_k4",   "idio_momentum_alpha",   make_idio_params(474, 168, k=4))
i410_k4    = run_strategy("p85_i410_k4",   "idio_momentum_alpha",   make_idio_params(410, 168, k=4))
i600_k2    = run_strategy("p85_i600_k2",   "idio_momentum_alpha",   make_idio_params(600, 168, k=2))
fund_144   = run_strategy("p85_f144_k2",   "funding_momentum_alpha", make_fund_params(144, k=2))

base = {"v1": v1_data, "i437": i437_k4, "i460": i460_k4, "i474": i474_k4,
        "i410": i410_k4, "i600": i600_k2, "f144": fund_144}

print("\nReference year profiles:")
for name, data in [("V1", v1_data), ("I437_k4", i437_k4), ("I460_k4", i460_k4),
                   ("I474_k4", i474_k4), ("I410_k4", i410_k4), ("I600_k2", i600_k2),
                   ("F144_k2", fund_144)]:
    yby = [round(data.get(y, {}).get("sharpe", 0.0), 3) if data.get(y, {}).get("sharpe") is not None else None for y in YEARS]
    print(f"  {name}: YbY={yby}, AVG={data['_avg']}, MIN={data['_min']}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION A: Fine 1.25% grid around triple-lb balanced champion
# P84 balanced: V1=25%, I437=5%, I460=12.5%, I410=15%, I600=10%, F144=32.5%
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION A: Fine Grid around Triple-lb Balanced (I437+I460+I410)")
print("  P84 champion: V1=25%, I437=5%, I460=12.5%, I410=15%, I600=10%, F144=32.5%")
print("  AVG=2.040, MIN=1.431, ALL years >1.4!")
print("═"*70)

# Fix I437=5%, I600=10%. Sweep V1/I460/I410, F144=remainder.
# Also try I437=0% (pure I460+I410) and I437=2.5%
sec_a_results = []
i437_fixed_vals = pct_range(0, 1250, 250)     # 0%, 2.5%, 5%, 7.5%, 10%
v1_vals_a   = pct_range(1875, 3125, 125)       # 18.75% to 30% (9 vals)
i460_vals_a = pct_range(875,  1875, 125)       # 8.75% to 17.5% (8 vals)
i410_vals_a = pct_range(875,  2125, 125)       # 8.75% to 20% (9 vals)
i600_fixed  = 0.10

total_a = len(i437_fixed_vals) * len(v1_vals_a) * len(i460_vals_a) * len(i410_vals_a)
print(f"\nSection A: {len(i437_fixed_vals)}x{len(v1_vals_a)}x{len(i460_vals_a)}x{len(i410_vals_a)} = ~{total_a} raw configs (filtered by F144>=0.20)")

for wi437 in i437_fixed_vals:
    for wv1 in v1_vals_a:
        for wi460 in i460_vals_a:
            for wi410 in i410_vals_a:
                wf144 = 1.0 - wv1 - wi437 - wi460 - wi410 - i600_fixed
                if wf144 < 0.20 or wf144 > 0.55:
                    continue
                # Round to avoid float drift
                wf144 = round(wf144, 6)
                w = {"v1": wv1, "i437": wi437, "i460": wi460, "i410": wi410,
                     "i600": i600_fixed, "f144": wf144}
                r = blend_ensemble(base, w)
                sec_a_results.append((r["avg"], r["min"], w))

pareto_a = pareto_filter(sec_a_results)
pareto_a.sort(key=lambda x: x[1], reverse=True)

print(f"\nSection A: {len(sec_a_results)} valid configs, {len(pareto_a)} Pareto non-dominated")
print("\nPareto frontier (sorted by MIN desc):")
best_a_avg, best_a_min = 0.0, 0.0
best_a_balanced = None
for avg, mn, w in pareto_a[:20]:
    print(f"  AVG={avg:.4f}, MIN={mn:.4f} | V1={w['v1']*100:.2f}% I437={w['i437']*100:.2f}%"
          f" I460={w['i460']*100:.2f}% I410={w['i410']*100:.2f}% I600={w['i600']*100:.1f}%"
          f" F144={w['f144']*100:.2f}%")
    if avg > best_a_avg:
        best_a_avg = avg
    # Best balanced: AVG>=2.000 and highest MIN
    if avg >= 2.000 and mn > best_a_min:
        best_a_min = mn
        best_a_balanced = (avg, mn, w)

if best_a_balanced:
    avg, mn, w = best_a_balanced
    r_check = blend_ensemble(base, w)
    print(f"\n★ BEST BALANCED (AVG>=2.000): AVG={avg:.4f}, MIN={mn:.4f}")
    print(f"  V1={w['v1']*100:.2f}%, I437={w['i437']*100:.2f}%, I460={w['i460']*100:.2f}%,"
          f" I410={w['i410']*100:.2f}%, I600={w['i600']*100:.1f}%, F144={w['f144']*100:.2f}%")
    print(f"  YbY={[round(v, 3) if v is not None else None for v in r_check['yby']]}")
    if mn > 1.431:
        print(f"  ★★ STRICTLY DOMINATES P84 balanced (MIN {mn:.3f} > 1.431)!")
    if avg > 2.040:
        print(f"  ★★ STRICTLY DOMINATES P84 balanced (AVG {avg:.3f} > 2.040)!")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION B: Triple-lb with I474: I437+I474+I410
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION B: Triple-lb I437+I474+I410 (swap I460→I474 in triple-lb)")
print("  Motivation: I474 has better 2022 (0.883 vs 0.635) and similar 2023 (0.974 vs 1.038)")
print("  I474 ALREADY dominates I460 in dual-lb setting")
print("═"*70)

# Same structure as Section A but I460→I474
sec_b_results = []
i437_b_vals = pct_range(0, 1250, 250)          # 0%, 2.5%, 5%, 7.5%, 10%
v1_vals_b   = pct_range(1875, 3125, 125)        # 18.75% to 30%
i474_vals_b = pct_range(875,  1875, 125)        # 8.75% to 17.5%
i410_vals_b = pct_range(875,  2125, 125)        # 8.75% to 20%

total_b = len(i437_b_vals) * len(v1_vals_b) * len(i474_vals_b) * len(i410_vals_b)
print(f"\nSection B: ~{total_b} raw configs (filtered by F144>=0.20)")

for wi437 in i437_b_vals:
    for wv1 in v1_vals_b:
        for wi474 in i474_vals_b:
            for wi410 in i410_vals_b:
                wf144 = 1.0 - wv1 - wi437 - wi474 - wi410 - i600_fixed
                if wf144 < 0.20 or wf144 > 0.55:
                    continue
                wf144 = round(wf144, 6)
                w = {"v1": wv1, "i437": wi437, "i474": wi474, "i410": wi410,
                     "i600": i600_fixed, "f144": wf144}
                r = blend_ensemble(base, w)
                sec_b_results.append((r["avg"], r["min"], w))

pareto_b = pareto_filter(sec_b_results)
pareto_b.sort(key=lambda x: x[1], reverse=True)

print(f"\nSection B: {len(sec_b_results)} valid configs, {len(pareto_b)} Pareto non-dominated")
print("\nPareto frontier (sorted by MIN desc):")
best_b_min = 0.0
best_b_balanced = None
for avg, mn, w in pareto_b[:20]:
    print(f"  AVG={avg:.4f}, MIN={mn:.4f} | V1={w['v1']*100:.2f}% I437={w['i437']*100:.2f}%"
          f" I474={w['i474']*100:.2f}% I410={w['i410']*100:.2f}% I600={w['i600']*100:.1f}%"
          f" F144={w['f144']*100:.2f}%")
    if avg >= 2.000 and mn > best_b_min:
        best_b_min = mn
        best_b_balanced = (avg, mn, w)

if best_b_balanced:
    avg, mn, w = best_b_balanced
    r_check = blend_ensemble(base, w)
    print(f"\n★ BEST B BALANCED (AVG>=2.000): AVG={avg:.4f}, MIN={mn:.4f}")
    print(f"  V1={w['v1']*100:.2f}%, I437={w['i437']*100:.2f}%, I474={w['i474']*100:.2f}%,"
          f" I410={w['i410']*100:.2f}%, I600={w['i600']*100:.1f}%, F144={w['f144']*100:.2f}%")
    print(f"  YbY={[round(v, 3) if v is not None else None for v in r_check['yby']]}")
    if mn > 1.431:
        print(f"  ★★ STRICTLY DOMINATES P84 balanced triple-lb (MIN {mn:.3f} > 1.431)!")
    if avg > 2.040:
        print(f"  ★★ STRICTLY DOMINATES P84 balanced triple-lb (AVG {avg:.3f} > 2.040)!")

# Compare best B vs best A
print("\nA vs B summary (triple-lb I460 vs I474):")
if best_a_balanced and best_b_balanced:
    aa, am, _ = best_a_balanced
    ba, bm, _ = best_b_balanced
    print(f"  I437+I460+I410 best balanced: AVG={aa:.4f}, MIN={am:.4f}")
    print(f"  I437+I474+I410 best balanced: AVG={ba:.4f}, MIN={bm:.4f}")
    if ba > aa and bm >= am:
        print("  → I474-triple STRICTLY DOMINATES I460-triple! ★")
    elif bm > am and ba >= aa:
        print("  → I474-triple STRICTLY DOMINATES I460-triple (via MIN)! ★")
    elif ba >= aa or bm >= am:
        print("  → I474-triple improves one dimension")
    else:
        print("  → I460-triple remains superior")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION C: I437+I474 dual-lb fine grid (1.25% steps)
# P84 I437+I474 balanced was ~2.046/1.377 — find the full Pareto
# P84 I437+I474 AVG-max: V1=10%, I437=20%, I474=20%, I600=10%, F144=40% → 2.224/1.177
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION C: I437+I474 Dual-lb Fine Grid (1.25% steps)")
print("  P84 I437+I474 balanced: ~2.046/1.377")
print("  P84 I437+I474 AVG-max: 2.224/1.177")
print("═"*70)

# Free: V1, I437, I474. Fixed: I600=10%. F144=remainder.
sec_c_results = []
v1_vals_c   = pct_range(750,  3000, 125)  # 7.5% to 28.75% (17 vals)
i437_vals_c = pct_range(1000, 2875, 125)  # 10% to 28.75% (15 vals)
i474_vals_c = pct_range(875,  2875, 125)  # 8.75% to 28.75% (16 vals)

total_c = len(v1_vals_c) * len(i437_vals_c) * len(i474_vals_c)
print(f"\nSection C: {len(v1_vals_c)}x{len(i437_vals_c)}x{len(i474_vals_c)} = ~{total_c} raw configs (filtered F144>=0.25)")

for wv1 in v1_vals_c:
    for wi437 in i437_vals_c:
        for wi474 in i474_vals_c:
            wf144 = 1.0 - wv1 - wi437 - wi474 - i600_fixed
            if wf144 < 0.25 or wf144 > 0.60:
                continue
            wf144 = round(wf144, 6)
            w = {"v1": wv1, "i437": wi437, "i474": wi474, "i600": i600_fixed, "f144": wf144}
            r = blend_ensemble(base, w)
            sec_c_results.append((r["avg"], r["min"], w))

pareto_c = pareto_filter(sec_c_results)
pareto_c.sort(key=lambda x: x[0], reverse=True)

print(f"\nSection C: {len(sec_c_results)} valid configs, {len(pareto_c)} Pareto non-dominated")
print("\nPareto frontier (sorted by AVG desc):")
best_c_min = 0.0
best_c_balanced = None
best_c_avgmax = None
best_c_avg = 0.0
for avg, mn, w in pareto_c[:25]:
    print(f"  AVG={avg:.4f}, MIN={mn:.4f} | V1={w['v1']*100:.2f}% I437={w['i437']*100:.2f}%"
          f" I474={w['i474']*100:.2f}% I600={w['i600']*100:.1f}% F144={w['f144']*100:.2f}%")
    if avg >= 2.000 and mn > best_c_min:
        best_c_min = mn
        best_c_balanced = (avg, mn, w)
    if avg > best_c_avg:
        best_c_avg = avg
        best_c_avgmax = (avg, mn, w)

if best_c_balanced:
    avg, mn, w = best_c_balanced
    r_check = blend_ensemble(base, w)
    print(f"\n★ BEST C BALANCED (AVG>=2.000): AVG={avg:.4f}, MIN={mn:.4f}")
    print(f"  V1={w['v1']*100:.2f}%, I437={w['i437']*100:.2f}%, I474={w['i474']*100:.2f}%,"
          f" I600={w['i600']*100:.1f}%, F144={w['f144']*100:.2f}%")
    print(f"  YbY={[round(v, 3) if v is not None else None for v in r_check['yby']]}")

if best_c_avgmax:
    avg, mn, w = best_c_avgmax
    r_check = blend_ensemble(base, w)
    print(f"\n★ BEST C AVG-MAX: AVG={avg:.4f}, MIN={mn:.4f}")
    print(f"  V1={w['v1']*100:.2f}%, I437={w['i437']*100:.2f}%, I474={w['i474']*100:.2f}%,"
          f" I600={w['i600']*100:.1f}%, F144={w['f144']*100:.2f}%")
    print(f"  YbY={[round(v, 3) if v is not None else None for v in r_check['yby']]}")
    if avg > 2.224:
        print(f"  ★★ NEW AVG-MAX RECORD! {avg:.3f} > 2.224")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION D: Quad-lb I437+I460+I474+I410 — can we push MIN above 1.44?
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION D: Quad-lb I437+I460+I474+I410 — push MIN >1.44?")
print("  Hypothesis: Adding I474 to triple-lb further plugs 2022 gap")
print("  P84 triple-lb balanced: all years exactly 1.4-1.5 range")
print("═"*70)

# Strategy: start from triple-lb balanced, split some weight off I460 → give to I474
# Also try: replace I460 allocation split as I460/I474
# V1: 22.5%-27.5%, I437:0-7.5%, I460:7.5-12.5%, I474:5-12.5%, I410:12.5-17.5%, I600=10%
sec_d_results = []
v1_vals_d   = [0.225, 0.25, 0.275]
i437_d_vals = [0.0, 0.025, 0.05, 0.075]
i460_d_vals = [0.075, 0.10, 0.125]
i474_d_vals = [0.05, 0.075, 0.10, 0.125]
i410_d_vals = [0.125, 0.15, 0.175]

total_d = len(v1_vals_d)*len(i437_d_vals)*len(i460_d_vals)*len(i474_d_vals)*len(i410_d_vals)
print(f"\nSection D: {total_d} raw configs (filtered by F144>=0.20)")

for wv1 in v1_vals_d:
    for wi437 in i437_d_vals:
        for wi460 in i460_d_vals:
            for wi474 in i474_d_vals:
                for wi410 in i410_d_vals:
                    wf144 = 1.0 - wv1 - wi437 - wi460 - wi474 - wi410 - i600_fixed
                    if wf144 < 0.20 or wf144 > 0.50:
                        continue
                    wf144 = round(wf144, 6)
                    w = {"v1": wv1, "i437": wi437, "i460": wi460, "i474": wi474,
                         "i410": wi410, "i600": i600_fixed, "f144": wf144}
                    r = blend_ensemble(base, w)
                    sec_d_results.append((r["avg"], r["min"], w))

pareto_d = pareto_filter(sec_d_results)
pareto_d.sort(key=lambda x: x[1], reverse=True)

print(f"\nSection D: {len(sec_d_results)} valid configs, {len(pareto_d)} Pareto non-dominated")
print("\nPareto frontier (sorted by MIN desc):")
best_d_min = 0.0
best_d_balanced = None
for avg, mn, w in pareto_d[:20]:
    print(f"  AVG={avg:.4f}, MIN={mn:.4f} | V1={w['v1']*100:.1f}% I437={w['i437']*100:.1f}%"
          f" I460={w['i460']*100:.1f}% I474={w['i474']*100:.1f}%"
          f" I410={w['i410']*100:.1f}% I600={w['i600']*100:.1f}% F144={w['f144']*100:.2f}%")
    if avg >= 2.000 and mn > best_d_min:
        best_d_min = mn
        best_d_balanced = (avg, mn, w)

if best_d_balanced:
    avg, mn, w = best_d_balanced
    r_check = blend_ensemble(base, w)
    print(f"\n★ BEST D BALANCED (AVG>=2.000): AVG={avg:.4f}, MIN={mn:.4f}")
    print(f"  V1={w['v1']*100:.1f}%, I437={w['i437']*100:.1f}%, I460={w['i460']*100:.1f}%,"
          f" I474={w['i474']*100:.1f}%, I410={w['i410']*100:.1f}%, I600={w['i600']*100:.1f}%,"
          f" F144={w['f144']*100:.2f}%")
    print(f"  YbY={[round(v, 3) if v is not None else None for v in r_check['yby']]}")
    if mn > 1.431:
        print(f"  ★★ STRICTLY DOMINATES P84 balanced (MIN {mn:.3f} > 1.431)!")
    if avg > 2.040:
        print(f"  ★★ STRICTLY DOMINATES P84 balanced (AVG {avg:.3f} > 2.040)!")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION E: Summary + Save Champions
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION E: PHASE 85 SUMMARY")
print("═"*70)

# Collect all candidates
all_candidates = []

if best_a_balanced:
    avg, mn, w = best_a_balanced
    all_candidates.append(("A-triple-I460-balanced", avg, mn, w))
if best_b_balanced:
    avg, mn, w = best_b_balanced
    all_candidates.append(("B-triple-I474-balanced", avg, mn, w))
if best_c_balanced:
    avg, mn, w = best_c_balanced
    all_candidates.append(("C-dual-I474-balanced", avg, mn, w))
if best_c_avgmax:
    avg, mn, w = best_c_avgmax
    all_candidates.append(("C-dual-I474-avgmax", avg, mn, w))
if best_d_balanced:
    avg, mn, w = best_d_balanced
    all_candidates.append(("D-quad-balanced", avg, mn, w))

# Also reference P84 champions for comparison
print("\nP84 champions for comparison:")
print("  P84 BALANCED (triple I437+I460+I410): AVG=2.040, MIN=1.431")
print("  P84 AVG-MAX  (dual I437+I474):        AVG=2.224, MIN=1.177")

print("\nP85 best candidates:")
for name, avg, mn, w in all_candidates:
    r_check = blend_ensemble(base, w)
    yby = [round(v, 3) if v is not None else None for v in r_check["yby"]]
    print(f"\n  [{name}] AVG={avg:.4f}, MIN={mn:.4f}, YbY={yby}")
    for k, v in w.items():
        if v > 0:
            print(f"    {k}: {v*100:.2f}%")

# Determine overall champions
# For balanced: highest MIN among AVG>=2.000
# For AVG-max: highest AVG
balanced_cands = [(a, m, w, n) for n, a, m, w in all_candidates if a >= 2.000]
avgmax_cands   = [(a, m, w, n) for n, a, m, w in all_candidates]

new_balanced_champ = None
new_avgmax_champ   = None

if balanced_cands:
    new_balanced_champ = max(balanced_cands, key=lambda x: (x[1], x[0]))
if avgmax_cands:
    new_avgmax_champ = max(avgmax_cands, key=lambda x: (x[0], x[1]))

print("\n" + "─"*60)
print("★★★ PHASE 85 CHAMPIONS ★★★")

if new_balanced_champ:
    a, m, w, n = new_balanced_champ
    r_f = blend_ensemble(base, w)
    yby = [round(v, 3) if v is not None else None for v in r_f["yby"]]
    print(f"\n  BALANCED CHAMPION [{n}]:")
    print(f"    AVG={a:.4f}, MIN={m:.4f}")
    print(f"    YbY={yby}")
    print(f"    Weights: {', '.join(f'{k}={v*100:.2f}%' for k, v in w.items() if v > 0)}")
    if m > 1.431:
        print(f"    ★★ STRICTLY DOMINATES P84 balanced (MIN {m:.3f} > 1.431)!")
    elif a > 2.040:
        print(f"    ★★ STRICTLY DOMINATES P84 balanced (AVG {a:.3f} > 2.040)!")
    else:
        print(f"    ≈ Similar to P84 balanced")

    # Save config
    cfg_balanced = {
        "phase": 85,
        "label": f"p85_balanced_{n}",
        "description": f"Phase 85 balanced champion: {n}",
        "avg_sharpe": a,
        "min_sharpe": m,
        "yby_sharpes": yby,
        "weights": {k: round(v, 6) for k, v in w.items() if v > 0},
        "strategy_params": {
            "nexus_alpha_v1": V1_PARAMS,
            "idio_i437_k4": make_idio_params(437, 168, k=4),
            "idio_i460_k4": make_idio_params(460, 168, k=4),
            "idio_i474_k4": make_idio_params(474, 168, k=4),
            "idio_i410_k4": make_idio_params(410, 168, k=4),
            "idio_i600_k2": make_idio_params(600, 168, k=2),
            "fund_f144_k2": make_fund_params(144, k=2),
        },
    }
    out_path = "configs/ensemble_p85_balanced.json"
    with open(out_path, "w") as f:
        json.dump(cfg_balanced, f, indent=2)
    print(f"    Saved: {out_path}")

if new_avgmax_champ:
    a, m, w, n = new_avgmax_champ
    r_f = blend_ensemble(base, w)
    yby = [round(v, 3) if v is not None else None for v in r_f["yby"]]
    print(f"\n  AVG-MAX CHAMPION [{n}]:")
    print(f"    AVG={a:.4f}, MIN={m:.4f}")
    print(f"    YbY={yby}")
    print(f"    Weights: {', '.join(f'{k}={v*100:.2f}%' for k, v in w.items() if v > 0)}")
    if a > 2.224:
        print(f"    ★★ NEW AVG-MAX RECORD! {a:.3f} > 2.224")

    cfg_avgmax = {
        "phase": 85,
        "label": f"p85_avgmax_{n}",
        "description": f"Phase 85 AVG-max champion: {n}",
        "avg_sharpe": a,
        "min_sharpe": m,
        "yby_sharpes": yby,
        "weights": {k: round(v, 6) for k, v in w.items() if v > 0},
        "strategy_params": {
            "nexus_alpha_v1": V1_PARAMS,
            "idio_i437_k4": make_idio_params(437, 168, k=4),
            "idio_i460_k4": make_idio_params(460, 168, k=4),
            "idio_i474_k4": make_idio_params(474, 168, k=4),
            "idio_i410_k4": make_idio_params(410, 168, k=4),
            "idio_i600_k2": make_idio_params(600, 168, k=2),
            "fund_f144_k2": make_fund_params(144, k=2),
        },
    }
    out_path = "configs/ensemble_p85_avgmax.json"
    with open(out_path, "w") as f:
        json.dump(cfg_avgmax, f, indent=2)
    print(f"    Saved: {out_path}")

print("\n" + "═"*70)
print("PHASE 85 COMPLETE")
print("═"*70)
