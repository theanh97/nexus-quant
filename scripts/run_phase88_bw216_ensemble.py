#!/usr/bin/env python3
"""
Phase 88: Beta Window bw=216 Ensemble Integration
==================================================
Phase 87 discoveries:
  - I410 bw=216: 2022=1.928 (vs 1.613 at bw=168) — massive +0.315 improvement!
  - I460 bw=216: standalone AVG=1.881 (vs 1.828), 2024=3.032
  - lb=415 bridge_score=2.759 (2022=1.764, 2023=0.995) > I410 (2.640)
  - I600=7.5% optimal: P87 balanced champion 2.002/1.493
  - I600=5% Pareto: AVG=2.061, MIN=1.487

Phase 88 agenda:
  A) Run reference signals + new bw=216 and lb=415 variants
     - V1, I460_bw168, I410_bw168, I600, F144 (reference)
     - I460_bw216_k4, I410_bw216_k4  (bw upgrade candidates)
     - I415_bw168_k4, I415_bw216_k4  (lb=415 bridge winner variants)
  B) Ensemble blend: I460_bw168 + I410_bw216 (upgrade I410 only)
  C) Ensemble blend: I460_bw216 + I410_bw168 (upgrade I460 only)
  D) Ensemble blend: I460_bw216 + I410_bw216 (both upgraded)
  E) lb=415_bw168 as I410 replacement; lb=415_bw216 variant
  F) Best bw combo + variable I600 (5%, 7.5%, 10%) fine grid
  G) Triple signal: I460_bw168 + I460_bw216 dual beta window
  H) Pareto summary, champion save
"""

import json, math, statistics, subprocess, sys, copy
from pathlib import Path
import numpy as np

OUT_DIR = "artifacts/phase88"
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
    path = f"/tmp/phase88_{run_name}_{year}.json"
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

def pct_range(lo: int, hi: int, step: int):
    return [x / 10000 for x in range(lo, hi + 1, step)]

def blend_section(label: str, strat_names: list, year_matrices: dict,
                  configs: list, p87_min: float = 1.4928) -> tuple:
    """Run numpy blend sweep, print top results, return best balanced."""
    results = sweep_configs_numpy(configs, strat_names, year_matrices)
    bal_cands = [(a, m, w) for a, m, w in results if a >= 2.000]
    if bal_cands:
        best = max(bal_cands, key=lambda x: (x[1], x[0]))
        a, m, w = best
        yby = yby_for_weights(w, strat_names, year_matrices)
        dominates = "★★ STRICTLY DOMINATES P87!" if m > p87_min else ""
        print(f"\n  [{label}] Best balanced: AVG={a:.4f}, MIN={m:.4f} {dominates}", flush=True)
        print(f"  YbY={yby}", flush=True)
        wfmt = " ".join(f"{k}={v*100:.2f}%" for k, v in sorted(w.items()) if v > 0.001)
        print(f"  Weights: {wfmt}", flush=True)
        return best
    else:
        print(f"\n  [{label}] No config reached AVG>=2.000 ({len(results)} configs tested)", flush=True)
        if results:
            best = max(results, key=lambda x: x[0])
            a, m, w = best
            yby = yby_for_weights(w, strat_names, year_matrices)
            print(f"  Best overall: AVG={a:.4f}, MIN={m:.4f}, YbY={yby}", flush=True)
            return best
        return None

# ─────────────────────────────────────────────────────────────────────────────
# SECTION A: Run all reference + new signal variants
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("PHASE 88: Beta Window bw=216 Ensemble Integration")
print("=" * 70, flush=True)
print("\nSECTION A: Running reference signals + bw=216 + lb=415 variants", flush=True)
print("  Reference: V1, I460_bw168, I410_bw168, I600_k2, F144")
print("  New: I460_bw216, I410_bw216, I415_bw168, I415_bw216", flush=True)

# Reference strategies
v1_data  = run_strategy("p88_v1",      "nexus_alpha_v1",       V1_PARAMS)
i460_k4  = run_strategy("p88_i460_bw168_k4", "idio_momentum_alpha", make_idio_params(460, 168, k=4))
i410_k4  = run_strategy("p88_i410_bw168_k4", "idio_momentum_alpha", make_idio_params(410, 168, k=4))
i600_k2  = run_strategy("p88_i600_k2", "idio_momentum_alpha",   make_idio_params(600, 168, k=2))
fund_144 = run_strategy("p88_fund144", "funding_momentum_alpha", make_fund_params(144, k=2))

# New bw=216 variants
i460_bw216 = run_strategy("p88_i460_bw216_k4", "idio_momentum_alpha", make_idio_params(460, 216, k=4))
i410_bw216 = run_strategy("p88_i410_bw216_k4", "idio_momentum_alpha", make_idio_params(410, 216, k=4))

# lb=415 bridge winner
i415_bw168 = run_strategy("p88_i415_bw168_k4", "idio_momentum_alpha", make_idio_params(415, 168, k=4))
i415_bw216 = run_strategy("p88_i415_bw216_k4", "idio_momentum_alpha", make_idio_params(415, 216, k=4))

print("\n" + "─"*60)
print("SECTION A SUMMARY — Standalone Sharpe Profiles:", flush=True)
print(f"  {'Signal':20s}  {'2021':>6}  {'2022':>6}  {'2023':>6}  {'2024':>6}  {'2025':>6}  {'AVG':>6}  {'MIN':>6}", flush=True)
for label, data in [
    ("V1", v1_data),
    ("I460_bw168_k4", i460_k4), ("I460_bw216_k4", i460_bw216),
    ("I410_bw168_k4", i410_k4), ("I410_bw216_k4", i410_bw216),
    ("I415_bw168_k4", i415_bw168), ("I415_bw216_k4", i415_bw216),
    ("I600_k2", i600_k2), ("F144_k2", fund_144),
]:
    yby = [round(data.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]
    print(f"  {label:20s}  {yby[0]:>6.3f}  {yby[1]:>6.3f}  {yby[2]:>6.3f}  {yby[3]:>6.3f}  {yby[4]:>6.3f}  {data['_avg']:>6.3f}  {data['_min']:>6.3f}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION B: I460_bw168 + I410_bw216 — upgrade only I410 beta window
# P87 champion used I460_bw168 + I410_bw168; I410_bw216 has 2022=1.928 (+0.315!)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION B: I460_bw168 + I410_bw216 (upgrade I410 beta window)")
print("  I410 bw=216 has 2022=1.928 vs 1.613 — can ensemble capture this?")
print("  Fixed I600=7.5% (P87 optimal)")
print("═"*70, flush=True)

STRATS_B = ["v1", "i460", "i410bw216", "i600", "f144"]
base_b = {
    "v1": v1_data, "i460": i460_k4,
    "i410bw216": i410_bw216,
    "i600": i600_k2, "f144": fund_144,
}
ym_b = build_year_matrices(base_b, STRATS_B)

configs_b = []
for wv1 in pct_range(1250, 3750, 125):        # 12.5% to 37.5% step 1.25%
    for wi460 in pct_range(500, 2000, 125):    # 5% to 20% step 1.25%
        for wi410 in pct_range(1000, 3000, 125):  # 10% to 30% step 1.25%
            wi600 = 0.075
            wf144 = 1.0 - wv1 - wi460 - wi410 - wi600
            if 0.15 <= wf144 <= 0.55:
                configs_b.append({"v1": wv1, "i460": wi460, "i410bw216": wi410,
                                   "i600": wi600, "f144": round(wf144, 8)})

print(f"  Grid size: {len(configs_b)} configs", flush=True)
best_b = blend_section("B:I410bw216", STRATS_B, ym_b, configs_b)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION C: I460_bw216 + I410_bw168 — upgrade only I460 beta window
# I460 bw=216 has AVG=1.881 (vs 1.828) and 2024=3.032
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION C: I460_bw216 + I410_bw168 (upgrade I460 beta window)")
print("  I460 bw=216 has AVG=1.881 vs 1.828, 2024=3.032 — huge 2024 boost!")
print("  Fixed I600=7.5% (P87 optimal)")
print("═"*70, flush=True)

STRATS_C = ["v1", "i460bw216", "i410", "i600", "f144"]
base_c = {
    "v1": v1_data, "i460bw216": i460_bw216,
    "i410": i410_k4,
    "i600": i600_k2, "f144": fund_144,
}
ym_c = build_year_matrices(base_c, STRATS_C)

configs_c = []
for wv1 in pct_range(1250, 3750, 125):
    for wi460 in pct_range(500, 2000, 125):
        for wi410 in pct_range(1000, 3000, 125):
            wi600 = 0.075
            wf144 = 1.0 - wv1 - wi460 - wi410 - wi600
            if 0.15 <= wf144 <= 0.55:
                configs_c.append({"v1": wv1, "i460bw216": wi460, "i410": wi410,
                                   "i600": wi600, "f144": round(wf144, 8)})

print(f"  Grid size: {len(configs_c)} configs", flush=True)
best_c = blend_section("C:I460bw216", STRATS_C, ym_c, configs_c)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION D: I460_bw216 + I410_bw216 — both signals upgraded to bw=216
# Testing combined bw=216 upgrade
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION D: I460_bw216 + I410_bw216 (both upgraded to bw=216)")
print("  Both idio signals use bw=216 — maximum beta window upgrade")
print("  Fixed I600=7.5%")
print("═"*70, flush=True)

STRATS_D = ["v1", "i460bw216", "i410bw216", "i600", "f144"]
base_d = {
    "v1": v1_data, "i460bw216": i460_bw216,
    "i410bw216": i410_bw216,
    "i600": i600_k2, "f144": fund_144,
}
ym_d = build_year_matrices(base_d, STRATS_D)

configs_d = []
for wv1 in pct_range(1250, 3750, 125):
    for wi460 in pct_range(500, 2000, 125):
        for wi410 in pct_range(1000, 3000, 125):
            wi600 = 0.075
            wf144 = 1.0 - wv1 - wi460 - wi410 - wi600
            if 0.15 <= wf144 <= 0.55:
                configs_d.append({"v1": wv1, "i460bw216": wi460, "i410bw216": wi410,
                                   "i600": wi600, "f144": round(wf144, 8)})

print(f"  Grid size: {len(configs_d)} configs", flush=True)
best_d = blend_section("D:BothBW216", STRATS_D, ym_d, configs_d)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION E: lb=415 as I410 replacement (bw=168 and bw=216)
# lb=415 bridge_score=2.759 vs I410's 2.640 — does it improve ensemble?
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION E: lb=415 as I410 Replacement (bw=168 and bw=216)")
print("  lb=415 bridge 2022+2023=2.759 (vs I410 2.640) — test in ensemble")
print("  Also test I460 + I415 dual-lb (skip I410)")
print("═"*70, flush=True)

# E1: I460_bw168 + I415_bw168 (replace I410 with lb=415 at bw=168)
STRATS_E1 = ["v1", "i460", "i415", "i600", "f144"]
base_e1 = {"v1": v1_data, "i460": i460_k4, "i415": i415_bw168,
            "i600": i600_k2, "f144": fund_144}
ym_e1 = build_year_matrices(base_e1, STRATS_E1)
configs_e1 = []
for wv1 in pct_range(1250, 3750, 125):
    for wi460 in pct_range(500, 2000, 125):
        for wi415 in pct_range(1000, 3000, 125):
            wi600 = 0.075
            wf144 = 1.0 - wv1 - wi460 - wi415 - wi600
            if 0.15 <= wf144 <= 0.55:
                configs_e1.append({"v1": wv1, "i460": wi460, "i415": wi415,
                                    "i600": wi600, "f144": round(wf144, 8)})
print(f"\n  E1: I460_bw168 + I415_bw168 grid ({len(configs_e1)} configs)", flush=True)
best_e1 = blend_section("E1:I415bw168", STRATS_E1, ym_e1, configs_e1)

# E2: I460_bw168 + I415_bw216 (lb=415 with best bw)
STRATS_E2 = ["v1", "i460", "i415bw216", "i600", "f144"]
base_e2 = {"v1": v1_data, "i460": i460_k4, "i415bw216": i415_bw216,
            "i600": i600_k2, "f144": fund_144}
ym_e2 = build_year_matrices(base_e2, STRATS_E2)
configs_e2 = []
for wv1 in pct_range(1250, 3750, 125):
    for wi460 in pct_range(500, 2000, 125):
        for wi415 in pct_range(1000, 3000, 125):
            wi600 = 0.075
            wf144 = 1.0 - wv1 - wi460 - wi415 - wi600
            if 0.15 <= wf144 <= 0.55:
                configs_e2.append({"v1": wv1, "i460": wi460, "i415bw216": wi415,
                                    "i600": wi600, "f144": round(wf144, 8)})
print(f"\n  E2: I460_bw168 + I415_bw216 grid ({len(configs_e2)} configs)", flush=True)
best_e2 = blend_section("E2:I415bw216", STRATS_E2, ym_e2, configs_e2)

# E3: I460_bw216 + I415_bw216 (both bw=216 — fully upgraded)
STRATS_E3 = ["v1", "i460bw216", "i415bw216", "i600", "f144"]
base_e3 = {"v1": v1_data, "i460bw216": i460_bw216, "i415bw216": i415_bw216,
            "i600": i600_k2, "f144": fund_144}
ym_e3 = build_year_matrices(base_e3, STRATS_E3)
configs_e3 = []
for wv1 in pct_range(1250, 3750, 125):
    for wi460 in pct_range(500, 2000, 125):
        for wi415 in pct_range(1000, 3000, 125):
            wi600 = 0.075
            wf144 = 1.0 - wv1 - wi460 - wi415 - wi600
            if 0.15 <= wf144 <= 0.55:
                configs_e3.append({"v1": wv1, "i460bw216": wi460, "i415bw216": wi415,
                                    "i600": wi600, "f144": round(wf144, 8)})
print(f"\n  E3: I460_bw216 + I415_bw216 (both bw=216) grid ({len(configs_e3)} configs)", flush=True)
best_e3 = blend_section("E3:FullBW216lb415", STRATS_E3, ym_e3, configs_e3)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION F: Best bw combo + variable I600 fine grid
# Test I600 = 5%, 7.5%, 10% with the best bw combination from B/C/D
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION F: Best bw Combo + Variable I600 (5%, 7.5%, 10%)")
print("  Test if I600=5% Pareto point is still valid with bw=216 signals")
print("═"*70, flush=True)

# Find which section had the best balanced result
section_bests = [(best_b, "B:I410bw216", STRATS_B, ym_b),
                 (best_c, "C:I460bw216", STRATS_C, ym_c),
                 (best_d, "D:BothBW216", STRATS_D, ym_d),
                 (best_e3, "E3:FullBW216lb415", STRATS_E3, ym_e3)]
best_overall = max(
    [(b, lbl, sn, ym) for b, lbl, sn, ym in section_bests if b is not None],
    key=lambda x: (x[0][1], x[0][0]),
    default=(None, None, None, None)
)

if best_overall[0] is not None:
    best_result, best_label, best_strats, best_ym = best_overall
    print(f"  Best from sections B/C/D: {best_label}", flush=True)
    print(f"  Sweeping I600 = 5%, 7.5%, 10% with {best_label} signal combo", flush=True)

    i600_levels_f = [0.05, 0.075, 0.10]
    best_f_overall = None

    for wi600 in i600_levels_f:
        configs_f = []
        for wv1 in pct_range(1250, 3750, 125):
            for wi_a in pct_range(500, 2000, 125):    # idio signal 1
                for wi_b in pct_range(1000, 3000, 125):  # idio signal 2
                    wf144 = 1.0 - wv1 - wi_a - wi_b - wi600
                    if 0.15 <= wf144 <= 0.55:
                        w = {best_strats[0]: wv1, best_strats[1]: wi_a,
                             best_strats[2]: wi_b, best_strats[3]: wi600,
                             "f144": round(wf144, 8)}
                        configs_f.append(w)

        results_f = sweep_configs_numpy(configs_f, best_strats, best_ym)
        bal_f = [(a, m, w) for a, m, w in results_f if a >= 2.000]
        if bal_f:
            best_fi = max(bal_f, key=lambda x: (x[1], x[0]))
            a, m, w = best_fi
            yby = yby_for_weights(w, best_strats, best_ym)
            print(f"  I600={wi600*100:.1f}%: AVG={a:.4f}, MIN={m:.4f}, YbY={yby}", flush=True)
            if best_f_overall is None or m > best_f_overall[1]:
                best_f_overall = (a, m, w, wi600)
        else:
            print(f"  I600={wi600*100:.1f}%: no config reached AVG>=2.000", flush=True)

    if best_f_overall:
        a, m, w, wi600 = best_f_overall
        yby = yby_for_weights(w, best_strats, best_ym)
        print(f"\n  ★ BEST F: I600={wi600*100:.1f}% → AVG={a:.4f}, MIN={m:.4f}", flush=True)
        print(f"  YbY={yby}", flush=True)
        wfmt = " ".join(f"{k}={v*100:.2f}%" for k, v in sorted(w.items()) if v > 0.001)
        print(f"  Weights: {wfmt}", flush=True)
else:
    print("  No valid balanced results from sections B/C/D; skipping F", flush=True)
    best_f_overall = None

# ─────────────────────────────────────────────────────────────────────────────
# SECTION G: Dual beta window signal (I460_bw168 + I460_bw216)
# Both use lb=460 but different beta windows — complementary coverage
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION G: Dual Beta Window I460 (bw=168 + bw=216)")
print("  Two I460 signals with different bw — does diversification help?")
print("  Also test triple: I460_bw168 + I460_bw216 + I410_bw216")
print("═"*70, flush=True)

# G1: V1 + I460_bw168 + I460_bw216 + I600 + F144
STRATS_G1 = ["v1", "i460bw168", "i460bw216", "i600", "f144"]
base_g1 = {"v1": v1_data, "i460bw168": i460_k4, "i460bw216": i460_bw216,
            "i600": i600_k2, "f144": fund_144}
ym_g1 = build_year_matrices(base_g1, STRATS_G1)
configs_g1 = []
for wv1 in pct_range(1250, 3750, 125):
    for wi460a in pct_range(250, 1500, 125):   # bw168 (smaller weight likely)
        for wi460b in pct_range(500, 2000, 125):  # bw216
            wi600 = 0.075
            wf144 = 1.0 - wv1 - wi460a - wi460b - wi600
            if 0.15 <= wf144 <= 0.55 and wi460a + wi460b <= 0.35:
                configs_g1.append({"v1": wv1, "i460bw168": wi460a, "i460bw216": wi460b,
                                    "i600": wi600, "f144": round(wf144, 8)})
print(f"\n  G1: Dual I460 bw grid ({len(configs_g1)} configs)", flush=True)
best_g1 = blend_section("G1:DualI460bw", STRATS_G1, ym_g1, configs_g1)

# G2: V1 + I460_bw168 + I460_bw216 + I410_bw168 + I600 + F144 (5-signal)
STRATS_G2 = ["v1", "i460bw168", "i460bw216", "i410", "i600", "f144"]
base_g2 = {"v1": v1_data, "i460bw168": i460_k4, "i460bw216": i460_bw216,
            "i410": i410_k4, "i600": i600_k2, "f144": fund_144}
ym_g2 = build_year_matrices(base_g2, STRATS_G2)
configs_g2 = []
for wv1 in pct_range(1250, 3375, 125):
    for wi460a in pct_range(250, 1250, 125):
        for wi460b in pct_range(250, 1500, 125):
            for wi410 in pct_range(1000, 2500, 125):
                wi600 = 0.075
                wf144 = 1.0 - wv1 - wi460a - wi460b - wi410 - wi600
                if 0.15 <= wf144 <= 0.50 and wi460a + wi460b + wi410 <= 0.45:
                    configs_g2.append({
                        "v1": wv1, "i460bw168": wi460a, "i460bw216": wi460b,
                        "i410": wi410, "i600": wi600, "f144": round(wf144, 8)
                    })
print(f"\n  G2: Triple idio (dual-I460 + I410) grid ({len(configs_g2)} configs)", flush=True)
best_g2 = blend_section("G2:DualI460+I410", STRATS_G2, ym_g2, configs_g2)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION H: Fine grid around best champion + Pareto summary + save
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION H: Fine Grid + Pareto Summary + Champion Save")
print("═"*70, flush=True)

# Collect all balanced champions from all sections
all_champions = []
P87_MIN = 1.4928  # current record to beat

section_results = [
    (best_b, "B:I410bw216", STRATS_B, ym_b),
    (best_c, "C:I460bw216", STRATS_C, ym_c),
    (best_d, "D:BothBW216", STRATS_D, ym_d),
    (best_e1, "E1:I415bw168", STRATS_E1, ym_e1),
    (best_e2, "E2:I415bw216", STRATS_E2, ym_e2),
    (best_e3, "E3:FullBW216lb415", STRATS_E3, ym_e3),
    (best_g1, "G1:DualI460bw", STRATS_G1, ym_g1),
    (best_g2, "G2:DualI460+I410", STRATS_G2, ym_g2),
]

print("\nAll section champions vs P87 balanced (2.002/1.493):", flush=True)
print(f"  {'Section':24s}  {'AVG':>6}  {'MIN':>6}  {'Beats P87?'}", flush=True)
best_champion = None
for result, label, strats, ym in section_results:
    if result is None:
        print(f"  {label:24s}  N/A", flush=True)
        continue
    a, m, w = result
    beats = "★★ YES" if m > P87_MIN else ("YES (AVG)" if a > 2.002 else "no")
    print(f"  {label:24s}  {a:>6.4f}  {m:>6.4f}  {beats}", flush=True)
    if a >= 2.000:
        all_champions.append((a, m, w, label, strats, ym))
    if best_champion is None or (m > best_champion[1] and a >= 2.000):
        best_champion = (a, m, w, label, strats, ym)

# Fine grid around best champion (0.625% step)
if best_champion:
    a_best, m_best, w_best, lbl_best, strats_best, ym_best = best_champion
    print(f"\n  Running fine 0.625% grid around best: {lbl_best} AVG={a_best:.4f}/MIN={m_best:.4f}", flush=True)

    # Extract best weights and center ±3% range
    def fine_configs(center_w, strats, i600_fixed, step=63):
        """0.625% step grid centered on best weights."""
        # map weights to strat slot indices (skipping i600 and f144)
        moveable = [(k, v) for k, v in center_w.items()
                    if k not in ("i600", "f144") and "i600" not in k]
        configs = []
        if len(moveable) == 3:
            n1, w1c = moveable[0]
            n2, w2c = moveable[1]
            n3, w3c = moveable[2]
            lo1 = max(int(w1c * 10000) - 300, 500)
            hi1 = min(int(w1c * 10000) + 300, 4000)
            lo2 = max(int(w2c * 10000) - 300, 250)
            hi2 = min(int(w2c * 10000) + 300, 2500)
            lo3 = max(int(w3c * 10000) - 300, 750)
            hi3 = min(int(w3c * 10000) + 300, 3500)
            for wv1 in pct_range(lo1, hi1, step):
                for wa in pct_range(lo2, hi2, step):
                    for wb in pct_range(lo3, hi3, step):
                        wf144 = 1.0 - wv1 - wa - wb - i600_fixed
                        if 0.15 <= wf144 <= 0.55:
                            configs.append({n1: wv1, n2: wa, n3: wb,
                                            "i600": i600_fixed, "f144": round(wf144, 8)})
        return configs

    i600_best = w_best.get("i600", 0.075)
    fine_cfgs = fine_configs(w_best, strats_best, i600_best, step=63)
    if fine_cfgs:
        print(f"  Fine grid: {len(fine_cfgs)} configs", flush=True)
        fine_results = sweep_configs_numpy(fine_cfgs, strats_best, ym_best)
        bal_fine = [(a, m, w) for a, m, w in fine_results if a >= 2.000]
        if bal_fine:
            best_fine = max(bal_fine, key=lambda x: (x[1], x[0]))
            a_f, m_f, w_f = best_fine
            if m_f > m_best:
                print(f"  Fine grid IMPROVED: AVG={a_f:.4f}, MIN={m_f:.4f} (vs {m_best:.4f})", flush=True)
                a_best, m_best, w_best = a_f, m_f, w_f
            else:
                print(f"  Fine grid: AVG={a_f:.4f}, MIN={m_f:.4f} (no improvement)", flush=True)

    yby_best = yby_for_weights(w_best, strats_best, ym_best)
    wfmt = " ".join(f"{k}={v*100:.2f}%" for k, v in sorted(w_best.items()) if v > 0.001)
    print(f"\n★★★ PHASE 88 BALANCED CHAMPION [{lbl_best}]:", flush=True)
    print(f"    AVG={a_best:.4f}, MIN={m_best:.4f}", flush=True)
    print(f"    YbY={yby_best}", flush=True)
    print(f"    Weights: {wfmt}", flush=True)
    if m_best > P87_MIN:
        print(f"    ★★ STRICTLY DOMINATES P87 balanced (MIN {m_best:.4f} > {P87_MIN:.4f})!", flush=True)

    # Save champion config
    champion_config = {
        "phase": 88,
        "section": lbl_best,
        "avg_sharpe": round(a_best, 4),
        "min_sharpe": round(m_best, 4),
        "yby_sharpes": {y: round(v, 3) for y, v in zip(YEARS, yby_best)},
        "weights": {k: round(v, 6) for k, v in w_best.items() if v > 0.001},
        "signal_config": {
            "strat_names": strats_best,
            "notes": {
                "I460_bw168": "idio_momentum_alpha lb=460 bw=168 k=4",
                "I460_bw216": "idio_momentum_alpha lb=460 bw=216 k=4",
                "I410_bw168": "idio_momentum_alpha lb=410 bw=168 k=4",
                "I410_bw216": "idio_momentum_alpha lb=410 bw=216 k=4",
                "I415_bw168": "idio_momentum_alpha lb=415 bw=168 k=4",
                "I415_bw216": "idio_momentum_alpha lb=415 bw=216 k=4",
                "I600_k2":    "idio_momentum_alpha lb=600 bw=168 k=2",
                "F144":       "funding_momentum_alpha lb=144 k=2",
            }
        }
    }
    out_path = "configs/ensemble_p88_balanced.json"
    with open(out_path, "w") as f:
        json.dump(champion_config, f, indent=2)
    print(f"\n    Saved: {out_path}", flush=True)

else:
    print("  No valid balanced champion found.", flush=True)

# Pareto summary across all collected champions
print("\n" + "─"*60)
print("PARETO FRONTIER (Phase 88 candidates vs P87 baseline):", flush=True)
# Add P87 reference points
all_for_pareto = [(2.002, 1.493, {}, "P87-BALANCED"), (2.268, 1.125, {}, "P86-AVG-MAX")]
for a, m, w, label, _, _ in all_champions:
    all_for_pareto.append((a, m, w, label))

# Simple Pareto filter
pareto = []
for i, (a, m, w, lbl) in enumerate(all_for_pareto):
    dominated = any(
        a2 >= a and m2 >= m and (a2 > a or m2 > m)
        for j, (a2, m2, w2, lbl2) in enumerate(all_for_pareto) if j != i
    )
    if not dominated:
        pareto.append((a, m, lbl))

pareto.sort(key=lambda x: -x[0])
print(f"  {'AVG':>6}  {'MIN':>6}  Config", flush=True)
for a, m, lbl in pareto:
    print(f"  {a:>6.4f}  {m:>6.4f}  {lbl}", flush=True)

print("\nPHASE 88 COMPLETE", flush=True)
