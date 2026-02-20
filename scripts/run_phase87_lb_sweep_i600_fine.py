#!/usr/bin/env python3
"""
Phase 87: Idio Lookback Sweep (410-460) + Variable I600 + Beta Window
======================================================================
Phase 86 findings:
  - Balanced converging: V1≈27%, I460≈12%, I410≈20%, I600=10%, F144≈31% → 2.007/1.469
  - High-MIN: V1=30%, I460=11.23%, I410=21.8% → 1.954/1.478 (highest MIN)
  - AVG-max: I437(16.17%)+I474(30%) → 2.268/1.125 (new record, I474=30% optimal)
  - Balanced frontier flattening: P85→P86 MIN gain only +0.001

Phase 87 agenda:
  A) Idio lookback sweep 410-460 (step 5h) — find intermediate lookbacks
     Each new lb tested standalone to find 2022+2023 profile vs I410/I460
  B) Variable I600 weight (5-20%) with I460+I410 balanced
  C) Beta window sweep (bw=120,144,168,192,216,240) for I460 and I410
  D) Best intermediate lb + I460+I410 triple (if any intermediate beats single lb)
  E) Ultra-fine (0.3%) grid around P86 balanced champion
  F) Summary, save champions
"""

import json, math, statistics, subprocess, sys, copy
from pathlib import Path
import numpy as np

OUT_DIR = "artifacts/phase87"
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
        return {"sharpe": 0.0, "error": "insufficient data", "returns_np": None}
    arr = np.array(rets, dtype=np.float64)
    sharpe = float(arr.mean() / arr.std() * np.sqrt(8760)) if arr.std() > 0 else 0.0
    return {"sharpe": round(sharpe, 3), "returns_np": arr}

def make_config(run_name: str, year: str, strategy_name: str, params: dict) -> str:
    cfg = copy.deepcopy(BASE_CONFIG)
    start, end = YEAR_RANGES[year]
    cfg["run_name"] = run_name
    cfg["data"]["start"] = start
    cfg["data"]["end"] = end
    cfg["strategy"] = {"name": strategy_name, "params": params}
    path = f"/tmp/phase87_{run_name}_{year}.json"
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
                year_results[year] = {"error": "run failed", "sharpe": 0.0, "returns_np": None}
                print(f"    {year}: ERROR", flush=True)
                continue
        except subprocess.TimeoutExpired:
            year_results[year] = {"error": "timeout", "sharpe": 0.0, "returns_np": None}
            print(f"    {year}: TIMEOUT", flush=True)
            continue

        runs_dir = Path(OUT_DIR) / "runs"
        if not runs_dir.exists():
            year_results[year] = {"error": "no runs dir", "sharpe": 0.0, "returns_np": None}
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
                print(f"    {year}: Sharpe={m['sharpe']}", flush=True)
                continue
        year_results[year] = {"error": "no result", "sharpe": 0.0, "returns_np": None}
        print(f"    {year}: no result", flush=True)

    sharpes = [year_results[y].get("sharpe", 0.0) for y in YEARS
               if isinstance(year_results.get(y, {}).get("sharpe"), (int, float))]
    avg = round(sum(sharpes) / len(sharpes), 3) if sharpes else 0.0
    mn = round(min(sharpes), 3) if sharpes else 0.0
    yby = [round(year_results.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]
    year_results["_avg"] = avg
    year_results["_min"] = mn
    year_results["_pos"] = sum(1 for s in sharpes if s > 0)
    print(f"  → AVG={avg}, MIN={mn}", flush=True)
    print(f"  → YbY: {yby}", flush=True)
    return year_results

def build_year_matrices(base: dict, strat_names: list) -> dict:
    year_matrices = {}
    for year in YEARS:
        arrays = []
        for name in strat_names:
            arr = base[name][year].get("returns_np")
            arrays.append(arr if arr is not None else np.zeros(1))
        min_len = min(len(a) for a in arrays)
        year_matrices[year] = np.stack([a[:min_len] for a in arrays])
    return year_matrices

def sweep_configs_numpy(weight_configs: list, strat_names: list,
                        year_matrices: dict) -> list:
    if not weight_configs:
        return []
    W = np.array([[wc.get(n, 0.0) for n in strat_names]
                  for wc in weight_configs], dtype=np.float64)
    year_sharpes = {}
    for year in YEARS:
        R = year_matrices[year]
        B = W @ R
        mu = B.mean(axis=1)
        sd = B.std(axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            year_sharpes[year] = np.where(sd > 0, mu / sd * np.sqrt(8760), 0.0)
    yby_matrix = np.stack([year_sharpes[y] for y in YEARS], axis=1)
    avgs = yby_matrix.mean(axis=1)
    mins = yby_matrix.min(axis=1)
    return [(float(avgs[i]), float(mins[i]), wc) for i, wc in enumerate(weight_configs)]

def pareto_filter(results: list) -> list:
    if not results:
        return []
    avgs = np.array([r[0] for r in results])
    mins = np.array([r[1] for r in results])
    dominated = np.zeros(len(results), dtype=bool)
    for i in range(len(results)):
        mask = (avgs >= avgs[i]) & (mins >= mins[i]) & ((avgs > avgs[i]) | (mins > mins[i]))
        mask[i] = False
        if mask.any():
            dominated[i] = True
    return [results[i] for i in range(len(results)) if not dominated[i]]

def yby_for_weights(w: dict, strat_names: list, year_matrices: dict) -> list:
    W = np.array([[w.get(n, 0.0) for n in strat_names]])
    yby = []
    for year in YEARS:
        R = year_matrices[year]
        B = (W @ R)[0]
        s = float(B.mean() / B.std() * np.sqrt(8760)) if B.std() > 0 else 0.0
        yby.append(round(s, 3))
    return yby

def pct_range(lo: int, hi: int, step: int = 62) -> list:
    return [x / 10000 for x in range(lo, hi, step)]

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("PHASE 87: Lookback Sweep 410-460 + Variable I600 + Beta Window Fine")
print("=" * 70, flush=True)

# ─── Reference runs ───────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("REFERENCE RUNS (core signals)")
print("═"*70, flush=True)

v1_data   = run_strategy("p87_v1",      "nexus_alpha_v1",        V1_PARAMS)
i410_k4   = run_strategy("p87_i410_k4", "idio_momentum_alpha",   make_idio_params(410, 168, k=4))
i460_k4   = run_strategy("p87_i460_k4", "idio_momentum_alpha",   make_idio_params(460, 168, k=4))
i474_k4   = run_strategy("p87_i474_k4", "idio_momentum_alpha",   make_idio_params(474, 168, k=4))
i437_k4   = run_strategy("p87_i437_k4", "idio_momentum_alpha",   make_idio_params(437, 168, k=4))
i600_k2   = run_strategy("p87_i600_k2", "idio_momentum_alpha",   make_idio_params(600, 168, k=2))
fund_144  = run_strategy("p87_f144_k2", "funding_momentum_alpha", make_fund_params(144, k=2))

print("\nReference profiles:", flush=True)
for lbl, d in [("V1", v1_data), ("I410_k4", i410_k4), ("I460_k4", i460_k4),
               ("I474_k4", i474_k4), ("I437_k4", i437_k4), ("I600_k2", i600_k2),
               ("F144_k2", fund_144)]:
    print(f"  {lbl}: {[round(d.get(y,{}).get('sharpe',0),3) for y in YEARS]}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION A: Idio lookback sweep 410-460 in steps of 5h
# Find profile for each lb — seeking intermediate that beats I410+I460 pair
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION A: Idio Lookback Sweep 410→460 (step 5h), k=4, bw=168")
print("  Seeking intermediate lb that has I410's 2022 strength + I460's 2023 strength")
print("═"*70, flush=True)

# Lookbacks from 415 to 455 (step 5), excluding 410/437/460 already known
sweep_lbs = list(range(415, 460, 5))  # 415, 420, 425, ..., 455
print(f"Testing {len(sweep_lbs)} intermediate lookbacks: {sweep_lbs}", flush=True)

lb_data = {}
lb_data["410"] = i410_k4
lb_data["437"] = i437_k4
lb_data["460"] = i460_k4
lb_data["474"] = i474_k4

for lb in sweep_lbs:
    key = str(lb)
    lb_data[key] = run_strategy(f"p87_i{lb}_k4", "idio_momentum_alpha",
                                make_idio_params(lb, 168, k=4))

print("\n" + "─"*70, flush=True)
print("LOOKBACK PROFILE COMPARISON (year-by-year Sharpe):", flush=True)
print(f"  {'LB':>6}  {'2021':>6}  {'2022':>6}  {'2023':>6}  {'2024':>6}  {'2025':>6}  {'AVG':>6}  {'MIN':>6}", flush=True)

best_2022 = ("410", 0.0)
best_2023 = ("460", 0.0)
best_avg = ("474", 0.0)

all_lb_keys = ["410", "415", "420", "425", "430", "435", "437", "440", "445", "450", "455", "460", "474"]
for key in all_lb_keys:
    if key not in lb_data:
        continue
    d = lb_data[key]
    yby = [round(d.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]
    print(f"  {key:>6}  {yby[0]:>6.3f}  {yby[1]:>6.3f}  {yby[2]:>6.3f}  {yby[3]:>6.3f}  {yby[4]:>6.3f}  {d['_avg']:>6.3f}  {d['_min']:>6.3f}", flush=True)
    if yby[1] > best_2022[1]:
        best_2022 = (key, yby[1])
    if yby[2] > best_2023[1]:
        best_2023 = (key, yby[2])
    if d["_avg"] > best_avg[1]:
        best_avg = (key, d["_avg"])

print(f"\n  Best 2022: lb={best_2022[0]} ({best_2022[1]:.3f})", flush=True)
print(f"  Best 2023: lb={best_2023[0]} ({best_2023[1]:.3f})", flush=True)
print(f"  Best AVG:  lb={best_avg[0]} ({best_avg[1]:.3f})", flush=True)

# Find the best intermediate lb for ensemble (one that has high 2022 AND 2023)
# Score = 2022_sharpe + 2023_sharpe (want both covered)
print("\n  2022+2023 combined score (seeking best 'bridging' lookback):", flush=True)
best_bridge = ("410", -99.0, [])
for key in all_lb_keys:
    if key not in lb_data:
        continue
    d = lb_data[key]
    y22 = d.get("2022", {}).get("sharpe", 0.0)
    y23 = d.get("2023", {}).get("sharpe", 0.0)
    score = y22 + y23
    yby = [round(d.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]
    print(f"  lb={key:>4}: 2022={y22:.3f}, 2023={y23:.3f}, score={score:.3f}", flush=True)
    if score > best_bridge[1]:
        best_bridge = (key, score, yby)

print(f"\n  ★ BEST BRIDGE lb={best_bridge[0]}: score={best_bridge[1]:.3f}, YbY={best_bridge[2]}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION B: Variable I600 weight with I460+I410 balanced
# P86: I600=10% fixed. Test 5%, 7.5%, 10%, 12.5%, 15%, 17.5%, 20%
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION B: Variable I600 Weight with I460+I410 (fine numpy sweep)")
print("  P86 fixed I600=10%. Testing 5% to 20%")
print("═"*70, flush=True)

# For each I600 level, run fine grid V1/I460/I410
# Use same strategy set as P86 (no new lb needed)
STRATS_B = ["v1", "i460", "i410", "i600", "f144"]
base_b = {"v1": v1_data, "i460": i460_k4, "i410": i410_k4,
          "i600": i600_k2, "f144": fund_144}
ym_b = build_year_matrices(base_b, STRATS_B)

i600_levels = [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20]
best_per_i600 = {}  # i600_level → (avg, min, weights)

print(f"\nSweeping {len(i600_levels)} I600 levels with fine V1/I460/I410 grid...", flush=True)
for wi600 in i600_levels:
    configs = []
    for wv1 in pct_range(1875, 3375, 62):       # 18.75% to 33.75%
        for wi460 in pct_range(625, 1875, 62):   # 6.25% to 18.75%
            for wi410 in pct_range(1250, 2375, 62):  # 12.5% to 23.75%
                wf144 = 1.0 - wv1 - wi460 - wi410 - wi600
                if 0.175 <= wf144 <= 0.55:
                    configs.append({"v1": wv1, "i460": wi460, "i410": wi410,
                                    "i600": wi600, "f144": round(wf144, 8)})
    results = sweep_configs_numpy(configs, STRATS_B, ym_b)
    # Best balanced: highest MIN where AVG>=2.000
    bal_cands = [(a, m, w) for a, m, w in results if a >= 2.000]
    if bal_cands:
        best = max(bal_cands, key=lambda x: (x[1], x[0]))
        best_per_i600[wi600] = best
    else:
        best = max(results, key=lambda x: x[0]) if results else None
        best_per_i600[wi600] = best
    if best:
        print(f"  I600={wi600*100:.1f}%: best balanced AVG={best[0]:.4f}, MIN={best[1]:.4f}"
              f" ({len(configs)} configs)", flush=True)

print("\nI600 sweep summary:", flush=True)
best_b_overall = None
for wi600, result in sorted(best_per_i600.items()):
    if result and result[0] >= 2.000:
        a, m, w = result
        yby = yby_for_weights(w, STRATS_B, ym_b)
        print(f"  I600={wi600*100:.1f}%: AVG={a:.4f}, MIN={m:.4f}, YbY={yby}", flush=True)
        if best_b_overall is None or m > best_b_overall[1]:
            best_b_overall = (a, m, w, wi600)

if best_b_overall:
    a, m, w, wi600 = best_b_overall
    yby = yby_for_weights(w, STRATS_B, ym_b)
    print(f"\n★ BEST B BALANCED: AVG={a:.4f}, MIN={m:.4f} at I600={wi600*100:.1f}%", flush=True)
    print(f"  V1={w['v1']*100:.2f}%, I460={w['i460']*100:.2f}%,"
          f" I410={w['i410']*100:.2f}%, I600={wi600*100:.1f}%, F144={w['f144']*100:.2f}%", flush=True)
    print(f"  YbY={yby}", flush=True)
    if m > 1.4690:
        print(f"  ★★ STRICTLY DOMINATES P86 balanced (MIN {m:.4f} > 1.4690)!", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION C: Beta window sweep for I460 and I410 (bw=120,144,168,192,216,240)
# All prior phases fixed bw=168. Is this optimal?
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION C: Beta Window Sweep for I460 and I410 (bw=120 to 240)")
print("  All prior work fixed bw=168. Testing if different beta window helps.")
print("═"*70, flush=True)

beta_windows = [120, 144, 168, 192, 216, 240]
bw_data = {}

# Test I460 with different beta windows
print("\nI460_k4 with various beta windows:", flush=True)
bw_data["i460_bw168"] = i460_k4  # already have this
for bw in beta_windows:
    if bw == 168:
        continue
    key = f"i460_bw{bw}"
    bw_data[key] = run_strategy(f"p87_i460_bw{bw}_k4", "idio_momentum_alpha",
                                make_idio_params(460, bw, k=4))

print("\n  I460 beta window comparison:", flush=True)
print(f"  {'BW':>5}  {'2022':>6}  {'2023':>6}  {'2024':>6}  {'2025':>6}  {'AVG':>6}  {'MIN':>6}", flush=True)
best_i460_bw = (168, i460_k4["_avg"])
for bw in beta_windows:
    key = f"i460_bw{bw}" if bw != 168 else "i460_bw168"
    if key not in bw_data:
        continue
    d = bw_data[key]
    yby = [round(d.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]
    print(f"  {bw:>5}  {yby[1]:>6.3f}  {yby[2]:>6.3f}  {yby[3]:>6.3f}  {yby[4]:>6.3f}  {d['_avg']:>6.3f}  {d['_min']:>6.3f}", flush=True)
    if d["_avg"] > best_i460_bw[1]:
        best_i460_bw = (bw, d["_avg"])

print(f"\n  Best I460 beta window: bw={best_i460_bw[0]} (AVG={best_i460_bw[1]:.3f})", flush=True)

# Test I410 with different beta windows
print("\nI410_k4 with various beta windows:", flush=True)
bw_data["i410_bw168"] = i410_k4
for bw in beta_windows:
    if bw == 168:
        continue
    key = f"i410_bw{bw}"
    bw_data[key] = run_strategy(f"p87_i410_bw{bw}_k4", "idio_momentum_alpha",
                                make_idio_params(410, bw, k=4))

print("\n  I410 beta window comparison:", flush=True)
print(f"  {'BW':>5}  {'2022':>6}  {'2023':>6}  {'2024':>6}  {'2025':>6}  {'AVG':>6}  {'MIN':>6}", flush=True)
best_i410_bw = (168, i410_k4["_avg"])
for bw in beta_windows:
    key = f"i410_bw{bw}" if bw != 168 else "i410_bw168"
    if key not in bw_data:
        continue
    d = bw_data[key]
    yby = [round(d.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]
    print(f"  {bw:>5}  {yby[1]:>6.3f}  {yby[2]:>6.3f}  {yby[3]:>6.3f}  {yby[4]:>6.3f}  {d['_avg']:>6.3f}  {d['_min']:>6.3f}", flush=True)
    if d["_avg"] > best_i410_bw[1]:
        best_i410_bw = (bw, d["_avg"])

print(f"\n  Best I410 beta window: bw={best_i410_bw[0]} (AVG={best_i410_bw[1]:.3f})", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION D: Best intermediate lb + I460+I410 triple or replacement
# If any lb between 410-460 has combined 2022+2023 score > I410 OR I460, test it
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION D: Best Intermediate LB in Ensemble with I460 and I410")
print("═"*70, flush=True)

# Use top-3 intermediate lbs by 2022+2023 bridge score
bridge_scores = []
for key in all_lb_keys:
    if key not in lb_data or key in ("410", "460", "474", "437"):
        continue
    d = lb_data[key]
    y22 = d.get("2022", {}).get("sharpe", 0.0)
    y23 = d.get("2023", {}).get("sharpe", 0.0)
    bridge_scores.append((y22 + y23, key, d))

bridge_scores.sort(key=lambda x: x[0], reverse=True)
top_bridge_lbs = bridge_scores[:3]  # top 3 intermediate lbs

# Build matrices including all potentially useful strategies
# Key: test replacement of I410 with bridge lb, or adding bridge lb to I460+I410
best_d_result = None

for score, lb_key, lb_d in top_bridge_lbs:
    print(f"\n  Testing lb={lb_key} (bridge score={score:.3f}) as replacement/addition:", flush=True)

    # Build fresh matrices for this lb
    strat_names_d = ["v1", "i460", f"i_lb{lb_key}", "i600", "f144"]
    base_d = {
        "v1": v1_data, "i460": i460_k4,
        f"i_lb{lb_key}": lb_d,
        "i600": i600_k2, "f144": fund_144
    }
    ym_d = build_year_matrices(base_d, strat_names_d)

    # Test: V1 + I460 + bridge_lb as replacement for I410
    configs_d = []
    for wv1 in pct_range(1875, 3375, 125):
        for wi460 in pct_range(625, 1875, 125):
            for wi_lb in pct_range(1250, 2375, 125):
                wf144 = 1.0 - wv1 - wi460 - wi_lb - 0.10
                if 0.20 <= wf144 <= 0.55:
                    configs_d.append({"v1": wv1, "i460": wi460,
                                      f"i_lb{lb_key}": wi_lb, "i600": 0.10,
                                      "f144": round(wf144, 8)})
    results_d = sweep_configs_numpy(configs_d, strat_names_d, ym_d)
    bal_d = [(a, m, w) for a, m, w in results_d if a >= 2.000]
    if bal_d:
        best_d = max(bal_d, key=lambda x: (x[1], x[0]))
        a, m, w = best_d
        yby = yby_for_weights(w, strat_names_d, ym_d)
        print(f"    V1+I460+I{lb_key} (replace I410): AVG={a:.4f}, MIN={m:.4f}, YbY={yby}", flush=True)
        if best_d_result is None or m > best_d_result[1]:
            best_d_result = (a, m, w, f"V1+I460+I{lb_key}")
    else:
        print(f"    V1+I460+I{lb_key}: no config reached AVG>=2.000", flush=True)

if best_d_result:
    a, m, w, desc = best_d_result
    print(f"\n★ BEST D: [{desc}] AVG={a:.4f}, MIN={m:.4f}", flush=True)
    if m > 1.4690:
        print(f"  ★★ STRICTLY DOMINATES P86 balanced!", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION E: Ultra-fine 0.3% grid around P86 balanced champion
# P86: V1=26.82%, I460=11.85%, I410=20.56%, I600=10%, F144=30.77%
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION E: Ultra-fine 0.3% Grid around P86 Balanced Champion")
print("  P86: V1=26.82%/I460=11.85%/I410=20.56%/I600=10%/F144=30.77% → 2.007/1.469")
print("═"*70, flush=True)

STRATS_E = ["v1", "i460", "i410", "i600", "f144"]
base_e = {"v1": v1_data, "i460": i460_k4, "i410": i410_k4,
          "i600": i600_k2, "f144": fund_144}
ym_e = build_year_matrices(base_e, STRATS_E)

# Center: V1=26.82%, I460=11.85%, I410=20.56%, I600=10%, F144=30.77%
# Range: ±3% from center in 0.3% steps (step = 30/10000)
STEP_E = 30  # 0.30%
v1_e   = pct_range(2382, 3182, STEP_E)   # 23.82% to 31.82%  (±3%)
i460_e = pct_range(885,  1685, STEP_E)   # 8.85% to 16.85%
i410_e = pct_range(1756, 2556, STEP_E)   # 17.56% to 25.56%

configs_e = []
for wv1 in v1_e:
    for wi460 in i460_e:
        for wi410 in i410_e:
            wf144 = 1.0 - wv1 - wi460 - wi410 - 0.10
            if 0.20 <= wf144 <= 0.45:
                configs_e.append({"v1": wv1, "i460": wi460, "i410": wi410,
                                   "i600": 0.10, "f144": round(wf144, 8)})

print(f"Section E: {len(configs_e)} valid configs (0.3% step, ±3% from P86 champion)", flush=True)
results_e = sweep_configs_numpy(configs_e, STRATS_E, ym_e)
pareto_e = pareto_filter(results_e)
pareto_e.sort(key=lambda x: x[1], reverse=True)

print(f"  {len(results_e)} configs, {len(pareto_e)} Pareto non-dominated", flush=True)
print("\nPareto frontier (top 15, MIN desc):", flush=True)
best_e_balanced = None
best_e_min = 0.0
for avg, mn, w in pareto_e[:15]:
    print(f"  AVG={avg:.4f}, MIN={mn:.4f} | V1={w['v1']*100:.2f}%"
          f" I460={w['i460']*100:.2f}% I410={w['i410']*100:.2f}% F144={w['f144']*100:.2f}%", flush=True)
    if avg >= 2.000 and mn > best_e_min:
        best_e_min = mn
        best_e_balanced = (avg, mn, w)

if best_e_balanced:
    avg, mn, w = best_e_balanced
    yby = yby_for_weights(w, STRATS_E, ym_e)
    print(f"\n★ BEST E BALANCED: AVG={avg:.4f}, MIN={mn:.4f}", flush=True)
    print(f"  V1={w['v1']*100:.2f}%, I460={w['i460']*100:.2f}%,"
          f" I410={w['i410']*100:.2f}%, I600=10%, F144={w['f144']*100:.2f}%", flush=True)
    print(f"  YbY={yby}", flush=True)
    if mn > 1.4690:
        print(f"  ★★ NEW MIN RECORD! {mn:.4f} > 1.4690", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION F: Summary + Save Champions
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION F: PHASE 87 SUMMARY")
print("═"*70, flush=True)

all_candidates = []
if best_b_overall:
    a, m, w, wi600 = best_b_overall
    all_candidates.append((f"B-I600-{wi600*100:.1f}pct", a, m, w, STRATS_B, ym_b))
if best_e_balanced:
    avg, mn, w = best_e_balanced
    all_candidates.append(("E-ultrafine", avg, mn, w, STRATS_E, ym_e))

print("\nP86 champions for comparison:")
print("  BALANCED: V1=26.82%/I460=11.85%/I410=20.56%/I600=10%/F144=30.77% → 2.007/1.469")
print("  AVG-MAX:  V1=4.98%/I437=16.17%/I474=30%/I600=10%/F144=38.85% → 2.268/1.125")

print("\nP87 best candidates:")
for name, a, m, w, snames, ym in all_candidates:
    yby = yby_for_weights(w, snames, ym)
    wstr = ", ".join(f"{k}={v*100:.2f}%" for k, v in w.items() if v > 1e-6)
    print(f"\n  [{name}] AVG={a:.4f}, MIN={m:.4f}")
    print(f"    {wstr}")
    print(f"    YbY={yby}", flush=True)

# Determine and save balanced champion (highest MIN where AVG>=2.000)
balanced_overall = [(a, m, w, n, sn, ym) for n, a, m, w, sn, ym in all_candidates if a >= 2.000]
if balanced_overall:
    new_champ = max(balanced_overall, key=lambda x: (x[1], x[0]))
    a, m, w, n, sn, ym = new_champ
    yby = yby_for_weights(w, sn, ym)
    print(f"\n★★★ PHASE 87 BALANCED CHAMPION [{n}]: AVG={a:.4f}, MIN={m:.4f}")
    print(f"    YbY={yby}")
    print(f"    Weights: {', '.join(f'{k}={v*100:.2f}%' for k,v in w.items() if v>1e-6)}", flush=True)
    if m > 1.4690:
        print(f"    ★★ STRICTLY DOMINATES P86 balanced (MIN {m:.4f} > 1.4690)!", flush=True)
    cfg = {
        "phase": 87, "label": f"p87_balanced_{n}",
        "avg_sharpe": a, "min_sharpe": m, "yby_sharpes": yby,
        "weights": {k: round(v, 6) for k, v in w.items() if v > 1e-6},
    }
    with open("configs/ensemble_p87_balanced.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"    Saved: configs/ensemble_p87_balanced.json", flush=True)

print("\n" + "═"*70)
print("PHASE 87 COMPLETE")
print("═"*70, flush=True)
