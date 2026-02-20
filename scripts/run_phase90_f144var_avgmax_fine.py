#!/usr/bin/env python3
"""
Phase 90: F144 Variable + AVG-max Fine + I437_bw216 + I600 Lower Bound
=======================================================================
Phase 89 discoveries:
  - I415_bw216 is the KEY balanced signal: 2022=1.737, 2023=1.143, ALL years>1.0
  - P89 balanced: V1=28.76%/I460bw168=16.26%/I415bw216=26.88%/I600=5%/F144=23.10% → 2.001/1.546
  - F144 TREND: 30.77% → 28.34% → 27.5% → 23.10% (each phase lower — bw=216 coverage reduces need)
  - I600 trend: lower is better (5% > 7.5% > 10%) — test I600=2.5%, 3.75%
  - NEW AVG-MAX RECORD: I474_bw216(35%) + I415_bw216(12.5%) → 2.286/1.186

Phase 90 agenda:
  A) Run reference signals + I437_bw216 (new candidate)
  B) F144 variable sweep (10-30%) with P89 combo (I460_bw168 + I415_bw216, I600=5%)
     - Hypothesis: F144=15-20% may further improve MIN
  C) I600 lower sweep (2.5%, 3.75%, 5%) with P89 combo
     - Hypothesis: I600=2.5-3.75% might beat I600=5%
  D) AVG-max fine grid around P89 champion (0.5% step)
     - V1~3.75%, I415bw216~12.5%, I474bw216~35%, I600~10%, F144~38.75%
     - Target: AVG > 2.30!
  E) I437_bw216 standalone profile + in AVG-max combo (I474_bw216 + I437_bw216)
  F) Triple-idio balanced: I460_bw168 + I415_bw216 + I474_bw216 + I600=5%
     - I474_bw216 has 2024=3.085 — could diversify 2024 coverage
  G) Ultra-fine 0.3125% grid around best balanced champion
  H) Pareto summary, champion save
"""

import json, math, statistics, subprocess, sys, copy
from pathlib import Path
import numpy as np

OUT_DIR = "artifacts/phase90"
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
    path = f"/tmp/phase90_{run_name}_{year}.json"
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

def sweep_configs_numpy(weight_configs: list, strat_names: list, year_matrices: dict) -> list:
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

def yby_for_weights(w: dict, strat_names: list, year_matrices: dict) -> list:
    W = np.array([[w.get(n, 0.0) for n in strat_names]])
    yby = []
    for year in YEARS:
        R = year_matrices[year]
        B = (W @ R)[0]
        s = float(B.mean() / B.std() * np.sqrt(8760)) if B.std() > 0 else 0.0
        yby.append(round(s, 3))
    return yby

def pct_range(lo: int, hi: int, step: int):
    return [x / 10000 for x in range(lo, hi + 1, step)]

def best_balanced(results, min_avg=2.000):
    cands = [(a, m, w) for a, m, w in results if a >= min_avg]
    return max(cands, key=lambda x: (x[1], x[0])) if cands else None

def print_result(label, r, strats, ym, p89_min=1.5463):
    if r is None:
        print(f"  [{label}] No balanced config found", flush=True)
        return
    a, m, w = r
    yby = yby_for_weights(w, strats, ym)
    beats = "★★ DOMINATES P89!" if m > p89_min else "> P88" if m > 1.5287 else ""
    print(f"  [{label}] AVG={a:.4f}, MIN={m:.4f} {beats}", flush=True)
    print(f"    YbY={yby}", flush=True)
    wfmt = " ".join(f"{k}={v*100:.2f}%" for k, v in sorted(w.items()) if v > 0.001)
    print(f"    Weights: {wfmt}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION A: Run reference signals + I437_bw216 (new)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("PHASE 90: F144 Variable + AVG-max Fine + I437_bw216 + I600 Lower")
print("=" * 70, flush=True)
print("\nSECTION A: Running reference signals + I437_bw216", flush=True)

v1_data    = run_strategy("p90_v1",           "nexus_alpha_v1",       V1_PARAMS)
i460_bw168 = run_strategy("p90_i460_bw168_k4","idio_momentum_alpha",  make_idio_params(460, 168, k=4))
i415_bw216 = run_strategy("p90_i415_bw216_k4","idio_momentum_alpha",  make_idio_params(415, 216, k=4))
i474_bw216 = run_strategy("p90_i474_bw216_k4","idio_momentum_alpha",  make_idio_params(474, 216, k=4))
i437_bw216 = run_strategy("p90_i437_bw216_k4","idio_momentum_alpha",  make_idio_params(437, 216, k=4))
i600_k2    = run_strategy("p90_i600_k2",      "idio_momentum_alpha",  make_idio_params(600, 168, k=2))
fund_144   = run_strategy("p90_fund144",       "funding_momentum_alpha",make_fund_params(144, k=2))

print("\n" + "─"*60)
print("SECTION A SUMMARY:", flush=True)
print(f"  {'Signal':20s}  {'2021':>6}  {'2022':>6}  {'2023':>6}  {'2024':>6}  {'2025':>6}  {'AVG':>6}  {'MIN':>6}", flush=True)
for lbl, data in [
    ("V1", v1_data),
    ("I460_bw168_k4", i460_bw168), ("I415_bw216_k4", i415_bw216),
    ("I474_bw216_k4", i474_bw216), ("I437_bw216_k4", i437_bw216),
    ("I600_k2", i600_k2), ("F144_k2", fund_144),
]:
    yby = [round(data.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]
    print(f"  {lbl:20s}  {yby[0]:>6.3f}  {yby[1]:>6.3f}  {yby[2]:>6.3f}  {yby[3]:>6.3f}  {yby[4]:>6.3f}  {data['_avg']:>6.3f}  {data['_min']:>6.3f}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION B: F144 variable (10-32%) with P89 combo + I600=5%
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION B: F144 Variable (10-32%) with I460_bw168 + I415_bw216 + I600=5%")
print("  P89 champion: F144=23.10% — trend declining each phase")
print("  Hypothesis: F144=15-20% may improve MIN (less drag from F144's weak 2025=0.062)")
print("═"*70, flush=True)

STRATS_B = ["v1", "i460bw168", "i415bw216", "i600", "f144"]
base_b = {"v1": v1_data, "i460bw168": i460_bw168, "i415bw216": i415_bw216,
          "i600": i600_k2, "f144": fund_144}
ym_b = build_year_matrices(base_b, STRATS_B)

f144_levels = [0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25, 0.275, 0.30, 0.325]
best_per_f144 = {}

for wf144 in f144_levels:
    configs = []
    for wv1 in pct_range(1250, 3750, 125):
        for wi460 in pct_range(500, 2000, 125):
            for wi415 in pct_range(1000, 3500, 125):
                wi600 = 0.05
                w_sum = wv1 + wi460 + wi415 + wi600 + wf144
                if abs(w_sum - 1.0) < 1e-9:
                    configs.append({"v1": wv1, "i460bw168": wi460, "i415bw216": wi415,
                                    "i600": wi600, "f144": wf144})
                elif abs(w_sum - 1.0) <= 0.0001:
                    # Allow very small rounding diff
                    pass
    # Also allow f144 to be "remainder"
    configs = []
    for wv1 in pct_range(1250, 3750, 125):
        for wi460 in pct_range(500, 2000, 125):
            for wi415 in pct_range(1000, 3500, 125):
                wi600 = 0.05
                # Check if sum = 1.0 with f144 as given
                remainder = round(1.0 - wv1 - wi460 - wi415 - wi600, 8)
                if abs(remainder - wf144) < 0.0001:
                    configs.append({"v1": wv1, "i460bw168": wi460, "i415bw216": wi415,
                                    "i600": wi600, "f144": wf144})

    # Use flexible f144 constraint: compute directly
    configs = []
    for wv1 in pct_range(1250, 3750, 125):
        for wi460 in pct_range(500, 2000, 125):
            for wi415 in pct_range(1000, 3500, 125):
                wi600 = 0.05
                # f144 is fixed here — only include if weights sum to 1.0
                wf = wf144
                if abs(wv1 + wi460 + wi415 + wi600 + wf - 1.0) < 1e-9:
                    configs.append({"v1": wv1, "i460bw168": wi460, "i415bw216": wi415,
                                    "i600": wi600, "f144": wf})

    # Actually: better to keep f144 fixed and let the other 3 sum to (1-i600-f144)
    budget = 1.0 - 0.05 - wf144  # budget for v1+i460+i415
    configs = []
    for wv1 in pct_range(1250, 3750, 125):
        for wi460 in pct_range(500, 2000, 125):
            wi415 = budget - wv1 - wi460
            if 0.10 <= wi415 <= 0.35 and wi415 >= 0:
                configs.append({"v1": wv1, "i460bw168": wi460, "i415bw216": round(wi415, 8),
                                "i600": 0.05, "f144": wf144})

    if not configs:
        print(f"  F144={wf144*100:.1f}%: no configs", flush=True)
        continue

    results = sweep_configs_numpy(configs, STRATS_B, ym_b)
    r = best_balanced(results)
    if r:
        a, m, w = r
        yby = yby_for_weights(w, STRATS_B, ym_b)
        best_per_f144[wf144] = (a, m, w)
        print(f"  F144={wf144*100:.1f}%: AVG={a:.4f}, MIN={m:.4f}, YbY={yby}", flush=True)
    else:
        print(f"  F144={wf144*100:.1f}%: no balanced result ({len(configs)} configs)", flush=True)

print("\nF144 sweep summary (best balanced per level):", flush=True)
best_b_overall = None
for wf144, (a, m, w) in sorted(best_per_f144.items()):
    mark = " ★" if m > 1.5463 else ""
    print(f"  F144={wf144*100:.1f}%: AVG={a:.4f}, MIN={m:.4f}{mark}", flush=True)
    if best_b_overall is None or m > best_b_overall[1]:
        best_b_overall = (a, m, w, wf144)

if best_b_overall:
    a, m, w, wf144 = best_b_overall
    yby = yby_for_weights(w, STRATS_B, ym_b)
    print(f"\n  ★ BEST B: F144={wf144*100:.1f}% → AVG={a:.4f}, MIN={m:.4f}", flush=True)
    print(f"    YbY={yby}", flush=True)
    wfmt = " ".join(f"{k}={v*100:.2f}%" for k, v in sorted(w.items()) if v > 0.001)
    print(f"    Weights: {wfmt}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION C: I600 lower sweep (2.5%, 3.75%, 5%) with optimal F144
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION C: I600 Lower Bound (2.5%, 3.75%, 5%) with I460_bw168 + I415_bw216")
print("  Trend: 10%→7.5%→5% each improved. Does 2.5-3.75% go further?")
print("═"*70, flush=True)

# Use optimal F144 from Section B (or P89's 23.1% if B found nothing)
opt_f144 = best_b_overall[3] if best_b_overall else 0.231

i600_low_levels = [0.025, 0.0375, 0.05]
best_c_overall = None

for wi600 in i600_low_levels:
    configs = []
    budget = 1.0 - wi600 - opt_f144
    for wv1 in pct_range(1250, 3750, 125):
        for wi460 in pct_range(500, 2000, 125):
            wi415 = budget - wv1 - wi460
            if 0.10 <= wi415 <= 0.35:
                configs.append({"v1": wv1, "i460bw168": wi460, "i415bw216": round(wi415, 8),
                                "i600": wi600, "f144": opt_f144})

    print(f"\n  I600={wi600*100:.2f}% ({len(configs)} configs, F144={opt_f144*100:.1f}%)", flush=True)
    results = sweep_configs_numpy(configs, STRATS_B, ym_b)
    r = best_balanced(results)
    if r:
        a, m, w = r
        yby = yby_for_weights(w, STRATS_B, ym_b)
        mark = " ★★ DOMINATES P89!" if m > 1.5463 else " > P88" if m > 1.5287 else ""
        print(f"  Best: AVG={a:.4f}, MIN={m:.4f}{mark}", flush=True)
        print(f"  YbY={yby}", flush=True)
        wfmt = " ".join(f"{k}={v*100:.2f}%" for k, v in sorted(w.items()) if v > 0.001)
        print(f"  Weights: {wfmt}", flush=True)
        if best_c_overall is None or m > best_c_overall[1]:
            best_c_overall = (a, m, w, wi600)
    else:
        print(f"  No balanced result", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION D: AVG-max fine grid around P89 champion (0.5% step)
# P89: V1=3.75%, I415bw216=12.5%, I474bw216=35%, I600=10%, F144=38.75% → 2.286/1.186
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION D: AVG-max Fine Grid (0.5% step around P89 AVG-max champion)")
print("  P89: V1=3.75%/I415bw216=12.5%/I474bw216=35%/I600=10%/F144=38.75% → 2.286/1.186")
print("  Target: AVG > 2.30!")
print("═"*70, flush=True)

STRATS_D = ["v1", "i415bw216", "i474bw216", "i600", "f144"]
base_d = {"v1": v1_data, "i415bw216": i415_bw216, "i474bw216": i474_bw216,
          "i600": i600_k2, "f144": fund_144}
ym_d = build_year_matrices(base_d, STRATS_D)

# Fine 0.5% step around P89 center
configs_d = []
for wv1 in pct_range(0, 1000, 50):        # 0 to 10%
    for wi415 in pct_range(500, 2000, 50): # 5 to 20%
        for wi474 in pct_range(2500, 4000, 50):  # 25 to 40%
            for wi600 in [0.075, 0.10, 0.125]:  # 7.5%, 10%, 12.5%
                wf144 = 1.0 - wv1 - wi415 - wi474 - wi600
                if 0.25 <= wf144 <= 0.55:
                    configs_d.append({"v1": wv1, "i415bw216": wi415, "i474bw216": wi474,
                                      "i600": wi600, "f144": round(wf144, 8)})

print(f"\n  AVG-max fine grid: {len(configs_d)} configs (0.5% step)", flush=True)
results_d = sweep_configs_numpy(configs_d, STRATS_D, ym_d)

if results_d:
    best_d_avg = max(results_d, key=lambda x: x[0])
    a_da, m_da, w_da = best_d_avg
    yby_da = yby_for_weights(w_da, STRATS_D, ym_d)
    print(f"  AVG-max best: AVG={a_da:.4f}, MIN={m_da:.4f}", flush=True)
    print(f"  YbY={yby_da}", flush=True)
    wfmt = " ".join(f"{k}={v*100:.2f}%" for k, v in sorted(w_da.items()) if v > 0.001)
    print(f"  Weights: {wfmt}", flush=True)
    if a_da > 2.286:
        print(f"  ★★ NEW AVG-MAX RECORD! (AVG {a_da:.4f} > P89's 2.286)", flush=True)

    # Also find best balanced (AVG>=2.0) in D
    bal_d = [(a, m, w) for a, m, w in results_d if a >= 2.000]
    if bal_d:
        best_d_bal = max(bal_d, key=lambda x: (x[1], x[0]))
        a_db, m_db, w_db = best_d_bal
        yby_db = yby_for_weights(w_db, STRATS_D, ym_d)
        print(f"  Best balanced: AVG={a_db:.4f}, MIN={m_db:.4f}, YbY={yby_db}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION E: I437_bw216 in AVG-max combo (I474_bw216 + I437_bw216)
# I437_bw168: 2022=1.281, 2023=0.563 — with bw=216, 2022 should improve
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION E: I437_bw216 standalone profile + in AVG-max combo")
print("  Testing I437_bw216 replacing I415_bw216 in AVG-max")
print("═"*70, flush=True)

i437_yby = [round(i437_bw216.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]
print(f"\n  I437_bw216 standalone: AVG={i437_bw216['_avg']:.3f}, MIN={i437_bw216['_min']:.3f}", flush=True)
print(f"  YbY={i437_yby}", flush=True)
print(f"  Compare: I415_bw216 YbY=[1.967, 1.737, 1.143, 2.015, 1.338]", flush=True)

# E1: V1 + I437_bw216 + I474_bw216 + I600 + F144
STRATS_E1 = ["v1", "i437bw216", "i474bw216", "i600", "f144"]
base_e1 = {"v1": v1_data, "i437bw216": i437_bw216, "i474bw216": i474_bw216,
            "i600": i600_k2, "f144": fund_144}
ym_e1 = build_year_matrices(base_e1, STRATS_E1)

configs_e1 = []
for wv1 in pct_range(0, 1500, 50):
    for wi437 in pct_range(500, 2500, 50):
        for wi474 in pct_range(2000, 4500, 50):
            for wi600 in [0.075, 0.10]:
                wf144 = 1.0 - wv1 - wi437 - wi474 - wi600
                if 0.25 <= wf144 <= 0.55:
                    configs_e1.append({"v1": wv1, "i437bw216": wi437, "i474bw216": wi474,
                                       "i600": wi600, "f144": round(wf144, 8)})

print(f"\n  E1: I437_bw216 + I474_bw216 AVG-max ({len(configs_e1)} configs)", flush=True)
results_e1 = sweep_configs_numpy(configs_e1, STRATS_E1, ym_e1)
if results_e1:
    best_e1_avg = max(results_e1, key=lambda x: x[0])
    a, m, w = best_e1_avg
    yby = yby_for_weights(w, STRATS_E1, ym_e1)
    print(f"  E1 AVG-max: AVG={a:.4f}, MIN={m:.4f}", flush=True)
    print(f"  YbY={yby}", flush=True)
    wfmt = " ".join(f"{k}={v*100:.2f}%" for k, v in sorted(w.items()) if v > 0.001)
    print(f"  Weights: {wfmt}", flush=True)
    if a > 2.286:
        print(f"  ★★ NEW AVG-MAX RECORD! (AVG {a:.4f} > P89's 2.286)", flush=True)

    # E1 balanced
    bal_e1 = [(a2, m2, w2) for a2, m2, w2 in results_e1 if a2 >= 2.000]
    if bal_e1:
        best_e1_bal = max(bal_e1, key=lambda x: (x[1], x[0]))
        a_b, m_b, w_b = best_e1_bal
        yby_b = yby_for_weights(w_b, STRATS_E1, ym_e1)
        print(f"  E1 balanced: AVG={a_b:.4f}, MIN={m_b:.4f}, YbY={yby_b}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION F: Triple-idio balanced: I460_bw168 + I415_bw216 + I474_bw216
# I474_bw216 has 2024=3.085 — could diversify 2024 coverage with I460
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION F: Triple-idio Balanced: I460_bw168 + I415_bw216 + I474_bw216")
print("  I474_bw216: 2024=3.085, AVG=1.869 — strong 2024 diversifier")
print("  Does adding I474_bw216 help the balanced combo?")
print("═"*70, flush=True)

STRATS_F = ["v1", "i460bw168", "i415bw216", "i474bw216", "i600", "f144"]
base_f = {"v1": v1_data, "i460bw168": i460_bw168, "i415bw216": i415_bw216,
          "i474bw216": i474_bw216, "i600": i600_k2, "f144": fund_144}
ym_f = build_year_matrices(base_f, STRATS_F)

best_f = None
for wi600 in [0.05, 0.075]:
    opt_f = best_b_overall[3] if best_b_overall else 0.231
    configs_f = []
    for wv1 in pct_range(1250, 3375, 125):
        for wi460 in pct_range(500, 1750, 125):
            for wi415 in pct_range(1000, 2750, 125):
                for wi474 in pct_range(250, 1500, 125):
                    wf144 = 1.0 - wv1 - wi460 - wi415 - wi474 - wi600
                    if 0.15 <= wf144 <= 0.40 and wi460 + wi415 + wi474 <= 0.50:
                        configs_f.append({
                            "v1": wv1, "i460bw168": wi460, "i415bw216": wi415,
                            "i474bw216": wi474, "i600": wi600, "f144": round(wf144, 8)
                        })

    print(f"\n  F: Triple-idio I600={wi600*100:.1f}% ({len(configs_f)} configs)", flush=True)
    results_f = sweep_configs_numpy(configs_f, STRATS_F, ym_f)
    r = best_balanced(results_f)
    if r:
        a, m, w = r
        yby = yby_for_weights(w, STRATS_F, ym_f)
        mark = " ★★ DOMINATES P89!" if m > 1.5463 else " > P88" if m > 1.5287 else ""
        print(f"  Best F [{wi600*100:.1f}%]: AVG={a:.4f}, MIN={m:.4f}{mark}", flush=True)
        print(f"    YbY={yby}", flush=True)
        wfmt = " ".join(f"{k}={v*100:.2f}%" for k, v in sorted(w.items()) if v > 0.001)
        print(f"    Weights: {wfmt}", flush=True)
        if best_f is None or m > best_f[1]:
            best_f = (a, m, w, wi600)
    else:
        print(f"  No triple-idio config reached AVG>=2.000", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION G: Ultra-fine 0.3125% grid around best balanced champion
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION G: Ultra-fine 0.3125% Grid around Best Balanced Champion")
print("═"*70, flush=True)

P89_MIN = 1.5463

# Pick best balanced across B and C
balanced_candidates = []
if best_b_overall:
    a, m, w, wf144 = best_b_overall
    balanced_candidates.append((a, m, w, 0.05, wf144, "B", STRATS_B, ym_b))
if best_c_overall:
    a, m, w, wi600 = best_c_overall
    balanced_candidates.append((a, m, w, wi600, opt_f144, "C", STRATS_B, ym_b))

ultra_champ = None
if balanced_candidates:
    best_cand = max(balanced_candidates, key=lambda x: (x[1], x[0]))
    a_bc, m_bc, w_bc, wi600_bc, wf144_bc, lbl_bc, strats_bc, ym_bc = best_cand

    print(f"\n  Starting from: {lbl_bc} AVG={a_bc:.4f}/MIN={m_bc:.4f}", flush=True)
    yby_bc = yby_for_weights(w_bc, strats_bc, ym_bc)
    print(f"  YbY={yby_bc}", flush=True)

    # Ultra-fine ±2% around center
    moveable = [(k, v) for k, v in w_bc.items() if k not in ("i600", "f144")]
    configs_g = []
    for n1, w1c in [(moveable[0][0], moveable[0][1])]:
        for n2, w2c in [(moveable[1][0], moveable[1][1])]:
            lo1, hi1 = max(int(w1c*10000)-200, 500), min(int(w1c*10000)+200, 4000)
            lo2, hi2 = max(int(w2c*10000)-200, 250), min(int(w2c*10000)+200, 2500)
            for wv in pct_range(lo1, hi1, 31):
                for wa in pct_range(lo2, hi2, 31):
                    wb = (1.0 - wv - wa - wi600_bc - wf144_bc)
                    if moveable[0][0] == "v1":
                        # v1, i460bw168, i415bw216
                        wb_key = moveable[2][0] if len(moveable) > 2 else n2
                        if len(moveable) >= 3:
                            wb_actual = 1.0 - wv - wa - wi600_bc - wf144_bc - w_bc.get(moveable[2][0], 0)
                        else:
                            wb_actual = wb
                    else:
                        wb_actual = wb
                    # Simpler: fix the 3rd signal as remainder
                    if len(moveable) == 3:
                        n3 = moveable[2][0]
                        w3 = 1.0 - wv - wa - wi600_bc - wf144_bc
                        if 0.10 <= w3 <= 0.40:
                            configs_g.append({n1: wv, n2: wa, n3: round(w3, 8),
                                              "i600": wi600_bc, "f144": wf144_bc})

    if not configs_g and len(moveable) == 2:
        # Only 2 moveable signals
        n1, w1c = moveable[0]
        n2, w2c = moveable[1]
        lo1, hi1 = max(int(w1c*10000)-200, 500), min(int(w1c*10000)+200, 4000)
        for wv in pct_range(lo1, hi1, 31):
            wa = 1.0 - wv - wi600_bc - wf144_bc
            if 0.10 <= wa <= 0.40:
                configs_g.append({n1: wv, n2: round(wa, 8),
                                  "i600": wi600_bc, "f144": wf144_bc})

    print(f"\n  Ultra-fine grid: {len(configs_g)} configs (0.3125% step)", flush=True)
    if configs_g:
        results_g = sweep_configs_numpy(configs_g, strats_bc, ym_bc)
        bal_g = [(a, m, w) for a, m, w in results_g if a >= 2.000]
        if bal_g:
            best_g = max(bal_g, key=lambda x: (x[1], x[0]))
            a_g, m_g, w_g = best_g
            improved = m_g > m_bc
            yby_g = yby_for_weights(w_g, strats_bc, ym_bc)
            mark = "IMPROVED!" if improved else "converged"
            print(f"  Result: AVG={a_g:.4f}, MIN={m_g:.4f} ({mark})", flush=True)
            print(f"  YbY={yby_g}", flush=True)
            if improved:
                a_bc, m_bc, w_bc = a_g, m_g, w_g
                lbl_bc = lbl_bc + "_ultrafine"
        ultra_champ = (a_bc, m_bc, w_bc, wi600_bc, wf144_bc, lbl_bc, strats_bc, ym_bc)
else:
    print("  No balanced candidates to refine.", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION H: Pareto summary + champion save
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION H: Pareto Summary + Champion Save")
print("═"*70, flush=True)

if ultra_champ:
    a_ch, m_ch, w_ch, wi600_ch, wf144_ch, lbl_ch, strats_ch, ym_ch = ultra_champ
    yby_ch = yby_for_weights(w_ch, strats_ch, ym_ch)
    wfmt = " ".join(f"{k}={v*100:.2f}%" for k, v in sorted(w_ch.items()) if v > 0.001)

    print(f"\n★★★ PHASE 90 BALANCED CHAMPION [{lbl_ch}]:", flush=True)
    print(f"    AVG={a_ch:.4f}, MIN={m_ch:.4f}", flush=True)
    print(f"    YbY={yby_ch}", flush=True)
    print(f"    Weights: {wfmt}", flush=True)

    if m_ch > P89_MIN:
        print(f"    ★★ STRICTLY DOMINATES P89 (MIN {m_ch:.4f} > {P89_MIN:.4f})!", flush=True)

    champion_config = {
        "phase": 90,
        "section": lbl_ch,
        "avg_sharpe": round(a_ch, 4),
        "min_sharpe": round(m_ch, 4),
        "yby_sharpes": {y: round(v, 3) for y, v in zip(YEARS, yby_ch)},
        "weights": {k: round(v, 6) for k, v in w_ch.items() if v > 0.001},
        "signal_config": {
            "strat_names": strats_ch,
            "notes": {
                "V1":            "nexus_alpha_v1 k=2",
                "I460_bw168_k4": "idio_momentum_alpha lb=460 bw=168 k=4",
                "I415_bw216_k4": "idio_momentum_alpha lb=415 bw=216 k=4",
                "I474_bw216_k4": "idio_momentum_alpha lb=474 bw=216 k=4",
                "I437_bw216_k4": "idio_momentum_alpha lb=437 bw=216 k=4",
                "I600_k2":       "idio_momentum_alpha lb=600 bw=168 k=2",
                "F144_k2":       "funding_momentum_alpha lb=144 k=2",
            }
        }
    }
    out_path = "configs/ensemble_p90_balanced.json"
    with open(out_path, "w") as f:
        json.dump(champion_config, f, indent=2)
    print(f"\n    Saved: {out_path}", flush=True)

# Collect best AVG-max from section D/E
if results_d:
    best_d_avgmax = max(results_d, key=lambda x: x[0])
    a_dm, m_dm, w_dm = best_d_avgmax
    if a_dm > 2.286:
        avgmax_config = {
            "phase": 90,
            "section": "D:AVGmax-fine",
            "type": "avg_max",
            "avg_sharpe": round(a_dm, 4),
            "min_sharpe": round(m_dm, 4),
            "weights": {k: round(v, 6) for k, v in w_dm.items() if v > 0.001},
        }
        with open("configs/ensemble_p90_avgmax.json", "w") as f:
            json.dump(avgmax_config, f, indent=2)
        print(f"  ★★ NEW AVG-MAX CONFIG SAVED: AVG={a_dm:.4f}/MIN={m_dm:.4f}", flush=True)

# Print final Pareto
print("\nFINAL PARETO FRONTIER:", flush=True)
all_points = [(2.286, 1.186, "P89-AVG-MAX"), (2.001, 1.546, "P89-BALANCED")]
if ultra_champ:
    all_points.append((a_ch, m_ch, f"P90-{lbl_ch}"))

pareto = []
for i, (a, m, lbl) in enumerate(all_points):
    dominated = any(
        a2 >= a and m2 >= m and (a2 > a or m2 > m)
        for j, (a2, m2, lbl2) in enumerate(all_points) if j != i
    )
    if not dominated:
        pareto.append((a, m, lbl))
pareto.sort(key=lambda x: -x[0])
print(f"  {'AVG':>6}  {'MIN':>6}  Config", flush=True)
for a, m, lbl in pareto:
    print(f"  {a:>6.4f}  {m:>6.4f}  {lbl}", flush=True)

print("\nPHASE 90 COMPLETE", flush=True)
