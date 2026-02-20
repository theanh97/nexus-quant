#!/usr/bin/env python3
"""
Phase 92: Dual-bw216 (I410+I415) + F144 Floor + AVG-max Extended Push
=======================================================================
Phase 91b discoveries:
  - P91b balanced: V1=27.47%/I460bw168=19.67%/I415bw216=32.47%/F144=20.39% → 2.010/1.576
  - P91b AVG-max: V1=4.25%/I415bw216=6%/I474bw216=47.25%/I600=5%/F144=37.5% → 2.319/1.126
  - I600=0% confirmed optimal for balanced
  - F144=20.39% true floor (ultra-fine); lb=415 global optimal (cliff at lb=420)
  - 2022 and 2023 BOTH at floor (~1.576) — need both years improved simultaneously

Bottleneck analysis (P91b balanced YbY=[2.876, 1.577, 1.576, 2.445, 1.576]):
  - Worst years: 2022=1.577, 2023=1.576 — essentially equal floors
  - I415_bw216 covers both (2022=1.737, 2023=1.143) but maxed at 32.47%
  - I410_bw216 profile: [2.117, 1.928, 1.046, 1.679, 1.408] — 2022=1.928 (STRONGEST!)
  - Hypothesis: I410_bw216 + I415_bw216 together could boost 2022 floor

Phase 92 agenda:
  A) Profile I410_bw216 (fresh run — P88 artifacts deleted)
  B) I410_bw216 + I415_bw216 balanced (no I600): dual-bw216 combo
     Hypothesis: higher 2022 via I410 (1.928) + better 2023 via I415 (1.143) → push MIN above 1.58!
  C) F144 floor fine sweep (14%-22%, step=0.3125%) with P91b architecture
     Hypothesis: F144 might go below 20.39% at ultra-fine resolution
  D) AVG-max extended push (I474=44%-55%) — how far can I474 go?
  E) Ultra-fine 0.3125% around best balanced champion
  F) Pareto summary + champion save
"""

import json, copy, subprocess, sys
from pathlib import Path
import numpy as np

OUT_DIR      = "artifacts/phase92"
P90_OUT_DIR  = "artifacts/phase90"
P91_OUT_DIR  = "artifacts/phase91"
YEARS        = ["2021", "2022", "2023", "2024", "2025"]
SYMBOLS      = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
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
        return {"sharpe": 0.0, "returns_np": None}
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
    path = f"/tmp/phase92_{run_name}_{year}.json"
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
                year_results[year] = {"sharpe": 0.0, "returns_np": None}
                print(f"    {year}: ERROR", flush=True)
                continue
        except subprocess.TimeoutExpired:
            year_results[year] = {"sharpe": 0.0, "returns_np": None}
            print(f"    {year}: TIMEOUT", flush=True)
            continue

        runs_dir = Path(OUT_DIR) / "runs"
        matching = sorted(
            [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith(run_name)],
            key=lambda d: d.stat().st_mtime,
        ) if runs_dir.exists() else []
        if matching:
            rp = matching[-1] / "result.json"
            if rp.exists():
                m = compute_metrics(str(rp))
                year_results[year] = m
                print(f"    {year}: Sharpe={m['sharpe']}", flush=True)
                continue
        year_results[year] = {"sharpe": 0.0, "returns_np": None}
        print(f"    {year}: no result", flush=True)

    sharpes = [year_results[y].get("sharpe", 0.0) for y in YEARS]
    avg = round(sum(sharpes) / len(sharpes), 3)
    mn  = round(min(sharpes), 3)
    yby = [round(year_results.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]
    year_results["_avg"] = avg
    year_results["_min"] = mn
    print(f"  → AVG={avg}, MIN={mn}", flush=True)
    print(f"  → YbY: {yby}", flush=True)
    return year_results

def load_strategy(label: str, out_dir: str) -> dict:
    runs_dir = Path(out_dir) / "runs"
    if not runs_dir.exists():
        return None
    year_results = {}
    for year in YEARS:
        run_name = f"{label}_{year}"
        matching = sorted(
            [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith(run_name)],
            key=lambda d: d.stat().st_mtime,
        )
        if matching:
            rp = matching[-1] / "result.json"
            if rp.exists():
                year_results[year] = compute_metrics(str(rp))
            else:
                year_results[year] = {"sharpe": 0.0, "returns_np": None}
        else:
            year_results[year] = {"sharpe": 0.0, "returns_np": None}
    sharpes = [year_results[y].get("sharpe", 0.0) for y in YEARS]
    year_results["_avg"] = round(sum(sharpes) / len(sharpes), 3)
    year_results["_min"] = round(min(sharpes), 3)
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

def sweep_configs_numpy(weight_configs: list, strat_names: list, year_matrices: dict,
                        chunk_size: int = 40000) -> list:
    """Chunked numpy sweep to avoid OOM."""
    if not weight_configs:
        return []
    all_results = []
    for start in range(0, len(weight_configs), chunk_size):
        chunk = weight_configs[start:start+chunk_size]
        W = np.array([[wc.get(s, 0.0) for s in strat_names] for wc in chunk], dtype=np.float64)
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
        all_results.extend([(float(avgs[i]), float(mins[i]), chunk[i]) for i in range(len(chunk))])
    return all_results

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

P91b_MIN = 1.5761   # champion to beat
P90_MIN  = 1.5614

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("PHASE 92: Dual-bw216 (I410+I415) + F144 Floor + AVG-max Extended")
print("=" * 70, flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION A: Load P90/P91 artifacts + run I410_bw216 fresh
# ─────────────────────────────────────────────────────────────────────────────
print("\nSECTION A: Load reference signals (re-run if missing) + I410_bw216", flush=True)

# Try loading from P90/P91 artifacts first; re-run fresh into P92 if missing
v1_data    = load_strategy("p90_v1",           P90_OUT_DIR)
i460_bw168 = load_strategy("p90_i460_bw168_k4",P90_OUT_DIR)
i415_bw216 = load_strategy("p90_i415_bw216_k4",P90_OUT_DIR)
i474_bw216 = load_strategy("p90_i474_bw216_k4",P90_OUT_DIR)
i600_k2    = load_strategy("p90_i600_k2",      P90_OUT_DIR)
fund_144   = load_strategy("p90_fund144",       P90_OUT_DIR)

# Also try loading from P92 (in case we already ran them)
def _try_load(var, p90_label, p92_label, strat_name, params):
    if var and var.get("_avg", 0) > 0 and all(var.get(y, {}).get("returns_np") is not None for y in YEARS):
        return var
    cached = load_strategy(p92_label, OUT_DIR)
    if cached and cached.get("_avg", 0) > 0 and all(cached.get(y, {}).get("returns_np") is not None for y in YEARS):
        return cached
    print(f"  Re-running {p92_label} (artifacts missing)...", flush=True)
    return run_strategy(p92_label, strat_name, params)

v1_data    = _try_load(v1_data,    "p90_v1",            "p92_v1",            "nexus_alpha_v1",       V1_PARAMS)
i460_bw168 = _try_load(i460_bw168, "p90_i460_bw168_k4", "p92_i460_bw168_k4", "idio_momentum_alpha",  make_idio_params(460, 168, k=4))
i415_bw216 = _try_load(i415_bw216, "p90_i415_bw216_k4", "p92_i415_bw216_k4", "idio_momentum_alpha",  make_idio_params(415, 216, k=4))
i474_bw216 = _try_load(i474_bw216, "p90_i474_bw216_k4", "p92_i474_bw216_k4", "idio_momentum_alpha",  make_idio_params(474, 216, k=4))
i600_k2    = _try_load(i600_k2,    "p90_i600_k2",       "p92_i600_k2",       "idio_momentum_alpha",  make_idio_params(600, 168, k=2))
fund_144   = _try_load(fund_144,   "p90_fund144",        "p92_fund144",        "funding_momentum_alpha", make_fund_params(144, k=2))

for name, data in [("V1", v1_data), ("I460bw168", i460_bw168), ("I415bw216", i415_bw216),
                   ("I474bw216", i474_bw216), ("I600", i600_k2), ("F144", fund_144)]:
    status = f"AVG={data['_avg']:.3f} ✓" if data and data.get("_avg", 0) > 0 else "MISSING!"
    print(f"  {name}: {status}", flush=True)

# Run I410_bw216 fresh (or load if already cached in P92)
i410_bw216 = load_strategy("p92_i410_bw216_k4", OUT_DIR)
if not (i410_bw216 and i410_bw216.get("_avg", 0) > 0 and all(i410_bw216.get(y, {}).get("returns_np") is not None for y in YEARS)):
    i410_bw216 = run_strategy("p92_i410_bw216_k4", "idio_momentum_alpha",
                              make_idio_params(410, 216, k=4))

print("\n" + "─"*60)
print("SIGNAL PROFILE:", flush=True)
print(f"  {'Signal':22s}  {'2021':>6}  {'2022':>6}  {'2023':>6}  {'2024':>6}  {'2025':>6}  {'AVG':>6}  {'MIN':>6}", flush=True)
for lbl, data in [("V1", v1_data), ("I410_bw216_k4", i410_bw216),
                  ("I415_bw216_k4", i415_bw216), ("I460_bw168_k4", i460_bw168),
                  ("I474_bw216_k4", i474_bw216), ("F144_k2", fund_144)]:
    if data is None:
        continue
    yby = [round(data.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]
    print(f"  {lbl:22s}  {yby[0]:>6.3f}  {yby[1]:>6.3f}  {yby[2]:>6.3f}  {yby[3]:>6.3f}  {yby[4]:>6.3f}  {data['_avg']:>6.3f}  {data['_min']:>6.3f}", flush=True)

i410_2022 = i410_bw216.get("2022", {}).get("sharpe", 0.0) if i410_bw216 else 0.0
i410_2023 = i410_bw216.get("2023", {}).get("sharpe", 0.0) if i410_bw216 else 0.0
print(f"\n  I410_bw216 bridge (2022+2023) = {i410_2022+i410_2023:.3f}", flush=True)
print(f"  I415_bw216 bridge (2022+2023) = {1.737+1.143:.3f} (reference)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION B: Dual-bw216 balanced: I410_bw216 + I415_bw216 (no I600)
# Current P91b: V1=27.47%/I460bw168=19.67%/I415bw216=32.47%/F144=20.39% → 2.010/1.576
# Hypothesis: I410(2022=1.928) + I415(2023=1.143) together → push both years above 1.58
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION B: Dual-bw216 Balanced: I410_bw216 + I415_bw216 (no I600)")
print("  I410: 2022=1.928, 2023=1.046 — strong 2022!")
print("  I415: 2022=1.737, 2023=1.143 — strong 2023!")
print("  Together: cover BOTH weak years! Target MIN > 1.58")
print("═"*70, flush=True)

STRATS_B = ["v1", "i460bw168", "i410bw216", "i415bw216", "f144"]
base_b = {"v1": v1_data, "i460bw168": i460_bw168,
          "i410bw216": i410_bw216, "i415bw216": i415_bw216, "f144": fund_144}
ym_b = build_year_matrices(base_b, STRATS_B)

# Sweep: F144=18-23%, sweep V1/I460/I410/I415 with remaining budget
# F144 optimizes around 20%, I410 small-medium weight (5-20%)
configs_b = []
for wf144 in pct_range(1700, 2300, 31):       # 17-23% in 0.31% steps
    for wv1 in pct_range(1500, 3500, 125):     # 15-35%
        for wi460 in pct_range(1000, 2500, 125):  # 10-25%
            for wi410 in pct_range(500, 1500, 125):  # 5-15%
                wi415 = 1.0 - wv1 - wi460 - wi410 - wf144
                if 0.15 <= wi415 <= 0.40:
                    configs_b.append({
                        "v1": wv1, "i460bw168": wi460,
                        "i410bw216": wi410,
                        "i415bw216": round(wi415, 8),
                        "f144": wf144
                    })

print(f"\n  5-signal dual-bw216 sweep: {len(configs_b)} configs", flush=True)
results_b = sweep_configs_numpy(configs_b, STRATS_B, ym_b)
r_b = best_balanced(results_b)
if r_b:
    a_b, m_b, w_b = r_b
    yby_b = yby_for_weights(w_b, STRATS_B, ym_b)
    mark = " ★★ BEATS P91b!" if m_b > P91b_MIN else " > P90" if m_b > P90_MIN else ""
    print(f"  Best: AVG={a_b:.4f}, MIN={m_b:.4f}{mark}", flush=True)
    print(f"  YbY={yby_b}", flush=True)
    wfmt = " ".join(f"{k}={v*100:.2f}%" for k, v in sorted(w_b.items()) if v > 0.001)
    print(f"  Weights: {wfmt}", flush=True)

    # Also find the truly max-MIN config in B
    all_bal_b = [(a, m, w) for a, m, w in results_b if a >= 2.000]
    if all_bal_b:
        best_min_b = max(all_bal_b, key=lambda x: x[1])
        print(f"\n  Max-MIN in B: AVG={best_min_b[0]:.4f}, MIN={best_min_b[1]:.4f}", flush=True)
else:
    print("  No balanced config found in B!", flush=True)
    r_b = None

# ─────────────────────────────────────────────────────────────────────────────
# SECTION C: F144 floor fine sweep (14%-22%, step=0.3125%) with P91b architecture
# P91b optimal: F144=20.39%. Can we go lower and improve MIN?
# Architecture: V1 + I460bw168 + I415bw216 + F144 (no I600)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION C: F144 Floor Fine Sweep (14-22%, step 0.3125%) — 4-signal no I600")
print("  P91b optimal: F144=20.39%. Testing lower F144 with ultra-fine step.")
print("═"*70, flush=True)

STRATS_C = ["v1", "i460bw168", "i415bw216", "f144"]
base_c = {"v1": v1_data, "i460bw168": i460_bw168, "i415bw216": i415_bw216, "f144": fund_144}
ym_c = build_year_matrices(base_c, STRATS_C)

# Sweep F144 from 14% to 22% in 0.3125% steps
best_per_f144_c = {}
f144_levels_c = pct_range(1400, 2200, 31)

for wf144 in f144_levels_c:
    configs_c = []
    for wv1 in pct_range(2000, 3500, 31):
        for wi460 in pct_range(1500, 2500, 31):
            wi415 = 1.0 - wv1 - wi460 - wf144
            if 0.25 <= wi415 <= 0.45:
                configs_c.append({"v1": wv1, "i460bw168": wi460,
                                   "i415bw216": round(wi415, 8), "f144": wf144})
    if not configs_c:
        continue
    results_c = sweep_configs_numpy(configs_c, STRATS_C, ym_c)
    r = best_balanced(results_c)
    if r:
        a, m, w = r
        best_per_f144_c[wf144] = (a, m, w)

print("\n  F144 sweep (best balanced per level):", flush=True)
print(f"  {'F144':>7}  {'AVG':>7}  {'MIN':>7}  {'2022':>7}  {'2023':>7}", flush=True)
best_c_overall = None
for wf144 in sorted(best_per_f144_c.keys()):
    a, m, w = best_per_f144_c[wf144]
    yby = yby_for_weights(w, STRATS_C, ym_c)
    mark = " ★" if m > P91b_MIN else ""
    print(f"  {wf144*100:>7.2f}%  {a:>7.4f}  {m:>7.4f}  {yby[1]:>7.3f}  {yby[2]:>7.3f}{mark}", flush=True)
    if best_c_overall is None or m > best_c_overall[0]:
        best_c_overall = (m, a, w, wf144)

if best_c_overall:
    m_co, a_co, w_co, wf144_co = best_c_overall
    yby_co = yby_for_weights(w_co, STRATS_C, ym_c)
    mark = " ★★ BEATS P91b!" if m_co > P91b_MIN else ""
    print(f"\n  ★ BEST C: F144={wf144_co*100:.2f}% → AVG={a_co:.4f}, MIN={m_co:.4f}{mark}", flush=True)
    print(f"    YbY={yby_co}", flush=True)
    wfmt = " ".join(f"{k}={v*100:.2f}%" for k, v in sorted(w_co.items()) if v > 0.001)
    print(f"    Weights: {wfmt}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION D: AVG-max extended push (I474=44%-55%)
# P91b: I474bw216=47.25%, AVG=2.319. How far can I474 go?
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION D: AVG-max Extended Push (I474=44-55%)")
print("  P91b: I474bw216=47.25%, AVG=2.319. How far can we push I474?")
print("═"*70, flush=True)

STRATS_D = ["v1", "i415bw216", "i474bw216", "i600", "f144"]
base_d = {"v1": v1_data, "i415bw216": i415_bw216, "i474bw216": i474_bw216,
          "i600": i600_k2, "f144": fund_144}
ym_d = build_year_matrices(base_d, STRATS_D)

configs_d = []
for wv1 in pct_range(100, 700, 25):           # 1-7%
    for wi415 in pct_range(400, 1200, 25):     # 4-12%
        for wi474 in pct_range(4400, 5500, 25):  # 44-55%
            for wi600 in [0.05, 0.075]:
                wf144 = 1.0 - wv1 - wi415 - wi474 - wi600
                if 0.25 <= wf144 <= 0.48:
                    configs_d.append({"v1": wv1, "i415bw216": wi415, "i474bw216": wi474,
                                      "i600": wi600, "f144": round(wf144, 8)})

print(f"\n  AVG-max extended grid: {len(configs_d)} configs (I474=44-55%, 0.25% step)", flush=True)
results_d = sweep_configs_numpy(configs_d, STRATS_D, ym_d)

if results_d:
    best_d_avg = max(results_d, key=lambda x: x[0])
    a_da, m_da, w_da = best_d_avg
    yby_da = yby_for_weights(w_da, STRATS_D, ym_d)
    wfmt_da = " ".join(f"{k}={v*100:.2f}%" for k, v in sorted(w_da.items()) if v > 0.001)
    print(f"  AVG-max best: AVG={a_da:.4f}, MIN={m_da:.4f}", flush=True)
    print(f"  YbY={yby_da}", flush=True)
    print(f"  Weights: {wfmt_da}", flush=True)
    if a_da > 2.3192:
        print(f"  ★★ NEW AVG-MAX RECORD! (AVG {a_da:.4f} > P91b's 2.3192)", flush=True)
    elif a_da > 2.3038:
        print(f"  ★ Beats P90 AVG-max (2.3038)", flush=True)

    # Also report per-I474 level trend
    print(f"\n  I474 weight vs best AVG:", flush=True)
    i474_buckets = {}
    for a, m, w in results_d:
        wi474 = round(w.get("i474bw216", 0.0) * 100) / 100  # round to 1%
        if wi474 not in i474_buckets or a > i474_buckets[wi474][0]:
            i474_buckets[wi474] = (a, m, w)
    for wi474 in sorted(i474_buckets.keys()):
        a, m, w = i474_buckets[wi474]
        print(f"    I474={wi474*100:.0f}%: AVG={a:.4f}, MIN={m:.4f}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION E: Ultra-fine 0.3125% around best balanced champion
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION E: Ultra-fine 0.3125% around Best Balanced Champion")
print("═"*70, flush=True)

# Pick best balanced from B and C
candidates_e = []
if r_b:
    a_b, m_b, w_b = r_b
    candidates_e.append((a_b, m_b, w_b, STRATS_B, ym_b, "B_dual"))
if best_c_overall:
    m_co, a_co, w_co, wf144_co = best_c_overall
    candidates_e.append((a_co, m_co, w_co, STRATS_C, ym_c, f"C_f{wf144_co*100:.1f}"))

# Also include P91b champion as baseline
p91b_w = {"v1": 0.2747, "i460bw168": 0.1967, "i415bw216": 0.3247, "f144": 0.2039}
p91b_yby = yby_for_weights(p91b_w, STRATS_C, ym_c)
candidates_e.append((2.0101, 1.5761, p91b_w, STRATS_C, ym_c, "P91b_baseline"))

best_e_champ = None
for a_ec, m_ec, w_ec, strats_ec, ym_ec, lbl_ec in candidates_e:
    print(f"\n  Refining {lbl_ec}: AVG={a_ec:.4f}, MIN={m_ec:.4f}", flush=True)
    yby_ec = yby_for_weights(w_ec, strats_ec, ym_ec)
    print(f"  YbY={yby_ec}", flush=True)

    fixed_keys = {k: v for k, v in w_ec.items() if k == "f144"}
    moveable = [(k, v) for k, v in w_ec.items() if k != "f144"]
    fixed_sum = sum(fixed_keys.values())

    configs_e = []
    if len(moveable) == 3:
        n1, w1c = moveable[0]
        n2, w2c = moveable[1]
        n3 = moveable[2][0]
        lo1 = max(int(w1c*10000) - 250, 1000)
        hi1 = min(int(w1c*10000) + 250, 4000)
        lo2 = max(int(w2c*10000) - 250, 800)
        hi2 = min(int(w2c*10000) + 250, 3000)
        for wv in pct_range(lo1, hi1, 31):
            for wa in pct_range(lo2, hi2, 31):
                wb = 1.0 - wv - wa - fixed_sum
                if 0.15 <= wb <= 0.45:
                    configs_e.append({n1: wv, n2: wa, n3: round(wb, 8), **fixed_keys})
    elif len(moveable) == 4:
        n1, w1c = moveable[0]
        n2, w2c = moveable[1]
        n3, w3c = moveable[2]
        n4 = moveable[3][0]
        lo1 = max(int(w1c*10000) - 200, 1000)
        hi1 = min(int(w1c*10000) + 200, 4000)
        lo2 = max(int(w2c*10000) - 200, 800)
        hi2 = min(int(w2c*10000) + 200, 3000)
        lo3 = max(int(w3c*10000) - 200, 300)
        hi3 = min(int(w3c*10000) + 200, 2000)
        for wv in pct_range(lo1, hi1, 63):   # 0.625% step for 4-way
            for wa in pct_range(lo2, hi2, 63):
                for wb in pct_range(lo3, hi3, 63):
                    wc = 1.0 - wv - wa - wb - fixed_sum
                    if 0.10 <= wc <= 0.40:
                        configs_e.append({n1: wv, n2: wa, n3: wb,
                                          n4: round(wc, 8), **fixed_keys})

    if not configs_e:
        continue

    print(f"  Ultra-fine grid: {len(configs_e)} configs", flush=True)
    results_e = sweep_configs_numpy(configs_e, strats_ec, ym_ec)
    bal_e = [(a, m, w) for a, m, w in results_e if a >= 2.000]
    if bal_e:
        best_e = max(bal_e, key=lambda x: (x[1], x[0]))
        a_e, m_e, w_e = best_e
        improved = m_e > m_ec
        yby_e = yby_for_weights(w_e, strats_ec, ym_ec)
        mark = "★★ IMPROVED!" if improved else "converged"
        print(f"  Result: AVG={a_e:.4f}, MIN={m_e:.4f} ({mark})", flush=True)
        print(f"  YbY={yby_e}", flush=True)
        if improved:
            m_ec, a_ec, w_ec = m_e, a_e, w_e
            lbl_ec = lbl_ec + "_ultrafine"

        if best_e_champ is None or m_ec > best_e_champ[1]:
            best_e_champ = (a_ec, m_ec, w_ec, strats_ec, ym_ec, lbl_ec)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION F: Pareto Summary + Champion Save
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION F: Final Pareto + Champion Save")
print("═"*70, flush=True)

if best_e_champ:
    a_ch, m_ch, w_ch, strats_ch, ym_ch, lbl_ch = best_e_champ
    yby_ch = yby_for_weights(w_ch, strats_ch, ym_ch)
    wfmt = " ".join(f"{k}={v*100:.2f}%" for k, v in sorted(w_ch.items()) if v > 0.001)

    print(f"\n★★★ PHASE 92 BALANCED CHAMPION [{lbl_ch}]:", flush=True)
    print(f"    AVG={a_ch:.4f}, MIN={m_ch:.4f}", flush=True)
    print(f"    YbY={yby_ch}", flush=True)
    print(f"    Weights: {wfmt}", flush=True)
    if m_ch > P91b_MIN:
        print(f"    ★★ STRICTLY DOMINATES P91b (MIN {m_ch:.4f} > {P91b_MIN:.4f})!", flush=True)
    elif m_ch >= P91b_MIN - 0.001:
        print(f"    ≈ P91b champion (MIN {m_ch:.4f} ≈ {P91b_MIN:.4f})", flush=True)

    champion_config = {
        "phase": 92,
        "section": lbl_ch,
        "avg_sharpe": round(a_ch, 4),
        "min_sharpe": round(m_ch, 4),
        "yby_sharpes": {y: round(v, 3) for y, v in zip(YEARS, yby_ch)},
        "weights": {k: round(v, 6) for k, v in w_ch.items() if v > 0.001},
        "signal_config": {
            "strat_names": strats_ch,
            "architecture": "4 or 5 signal ensemble — see weights",
            "notes": {
                "v1":        "nexus_alpha_v1 k=2",
                "i460bw168": "idio_momentum_alpha lb=460 bw=168 k=4",
                "i410bw216": "idio_momentum_alpha lb=410 bw=216 k=4",
                "i415bw216": "idio_momentum_alpha lb=415 bw=216 k=4",
                "i474bw216": "idio_momentum_alpha lb=474 bw=216 k=4",
                "i600":      "idio_momentum_alpha lb=600 bw=168 k=2",
                "f144":      "funding_momentum_alpha lb=144 k=2",
            }
        }
    }
    out_path = "configs/ensemble_p92_balanced.json"
    with open(out_path, "w") as f:
        json.dump(champion_config, f, indent=2)
    print(f"\n    Saved: {out_path}", flush=True)

# Save AVG-max if improved
if results_d:
    best_d_avgmax = max(results_d, key=lambda x: x[0])
    a_dm, m_dm, w_dm = best_d_avgmax
    if a_dm > 2.3038:
        avgmax_config = {
            "phase": 92,
            "section": "D:AVGmax-extended",
            "type": "avg_max",
            "avg_sharpe": round(a_dm, 4),
            "min_sharpe": round(m_dm, 4),
            "weights": {k: round(v, 6) for k, v in w_dm.items() if v > 0.001},
        }
        with open("configs/ensemble_p92_avgmax.json", "w") as f:
            json.dump(avgmax_config, f, indent=2)
        record = "★★ NEW AVG-MAX RECORD!" if a_dm > 2.3192 else "★ Beats P90"
        print(f"  {record} AVG-MAX SAVED: AVG={a_dm:.4f}/MIN={m_dm:.4f}", flush=True)

# Final Pareto
print("\nFINAL PARETO FRONTIER:", flush=True)
all_points = [
    (2.3192, 1.1256, "P91b-AVG-MAX"),
    (2.0101, 1.5761, "P91b-BALANCED"),
]
if best_e_champ:
    all_points.append((a_ch, m_ch, f"P92-{lbl_ch}"))
if results_d:
    a_dm, m_dm, _ = max(results_d, key=lambda x: x[0])
    if a_dm > 2.3192:
        all_points.append((a_dm, m_dm, "P92-D-AVGmax"))

pareto = []
for i, (a, m, lbl) in enumerate(all_points):
    dominated = any(
        a2 >= a and m2 >= m and (a2 > a or m2 > m)
        for j, (a2, m2, _) in enumerate(all_points) if j != i
    )
    if not dominated:
        pareto.append((a, m, lbl))
pareto.sort(key=lambda x: -x[0])
print(f"  {'AVG':>6}  {'MIN':>6}  Config", flush=True)
for a, m, lbl in pareto:
    print(f"  {a:>6.4f}  {m:>6.4f}  {lbl}", flush=True)

print("\nPHASE 92 COMPLETE", flush=True)
