#!/usr/bin/env python3
"""
Phase 91b: Recovery — AVG-max Ultra-Fine (chunked) + Ultra-fine balanced + lb combo test
=========================================================================================
Phase 91 crashed at Section C due to OOM (145K configs × 8760 bars = ~10GB).
This script recovers with:
  - Chunked sweep (50K chunks) to handle large grids
  - Narrowed AVG-max range around P90 champion
  - Ultra-fine 0.3125% around B best (I600=0% champion)
  - Section D: new lb signals (I420/I425/I430) in balanced combo
  - Save P91 balanced + AVG-max champions

Phase 91 findings so far:
  B: I600=0% → AVG=2.0396, MIN=1.5712 ★★ NEW CHAMP!
     Weights: V1=26.25%, I460bw168=20.00%, I415bw216=31.25%, F144=22.50%
  B confirmed: I600 trend ends at 0% (fully remove!)
  A: lb=415 is optimal (2023=1.143 vs I420=0.758) — cliff between 415 and 420!
"""

import json, copy, subprocess, sys
from pathlib import Path
import numpy as np

OUT_DIR_91  = "artifacts/phase91b"
P90_OUT_DIR = "artifacts/phase90"
P91_OUT_DIR = "artifacts/phase91"
YEARS       = ["2021", "2022", "2023", "2024", "2025"]
SYMBOLS     = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
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
    path = f"/tmp/phase91b_{run_name}_{year}.json"
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
                 "--config", config_path, "--out", OUT_DIR_91],
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

        runs_dir = Path(OUT_DIR_91) / "runs"
        if not runs_dir.exists():
            year_results[year] = {"sharpe": 0.0, "returns_np": None}
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
        year_results[year] = {"sharpe": 0.0, "returns_np": None}
        print(f"    {year}: no result", flush=True)

    sharpes = [year_results[y].get("sharpe", 0.0) for y in YEARS]
    avg = round(sum(sharpes) / len(sharpes), 3)
    mn  = round(min(sharpes), 3)
    year_results["_avg"] = avg
    year_results["_min"] = mn
    print(f"  → AVG={avg}, MIN={mn}", flush=True)
    print(f"  → YbY: {[round(year_results.get(y, {}).get('sharpe', 0.0), 3) for y in YEARS]}", flush=True)
    return year_results

def load_strategy(p90_label: str, out_dir: str) -> dict:
    runs_dir = Path(out_dir) / "runs"
    if not runs_dir.exists():
        return None
    year_results = {}
    for year in YEARS:
        run_name = f"{p90_label}_{year}"
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
                        chunk_size: int = 30000) -> list:
    """Memory-safe chunked numpy sweep."""
    if not weight_configs:
        return []
    all_results = []
    n = len(weight_configs)
    for start in range(0, n, chunk_size):
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

P91_MIN = 1.5712  # B: I600=0% champion
P90_MIN = 1.5614
F144_OPT = 0.225

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("PHASE 91b: Recovery — AVG-max Chunked + Ultra-fine Balanced + lb Combo")
print("=" * 70, flush=True)

# Load all reference signals from P90 artifacts
print("\nLoading P90 + P91 artifacts...", flush=True)
v1_data    = load_strategy("p90_v1",           P90_OUT_DIR)
i460_bw168 = load_strategy("p90_i460_bw168_k4",P90_OUT_DIR)
i415_bw216 = load_strategy("p90_i415_bw216_k4",P90_OUT_DIR)
i474_bw216 = load_strategy("p90_i474_bw216_k4",P90_OUT_DIR)
i437_bw216 = load_strategy("p90_i437_bw216_k4",P90_OUT_DIR)
i600_k2    = load_strategy("p90_i600_k2",      P90_OUT_DIR)
fund_144   = load_strategy("p90_fund144",       P90_OUT_DIR)
i420_bw216 = load_strategy("p91_i420_bw216_k4",P91_OUT_DIR)
i425_bw216 = load_strategy("p91_i425_bw216_k4",P91_OUT_DIR)
i430_bw216 = load_strategy("p91_i430_bw216_k4",P91_OUT_DIR)

# Verify loads
for name, data in [("V1", v1_data), ("I460bw168", i460_bw168), ("I415bw216", i415_bw216),
                   ("I474bw216", i474_bw216), ("I600", i600_k2), ("F144", fund_144),
                   ("I420bw216", i420_bw216), ("I425bw216", i425_bw216), ("I430bw216", i430_bw216)]:
    if data is None or data.get("_avg", 0) == 0.0:
        print(f"  WARNING: {name} not loaded!", flush=True)
    else:
        print(f"  {name}: AVG={data['_avg']:.3f} ✓", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION A: Ultra-fine 0.3125% around P91 B champion (I600=0%)
# B best: V1=26.25%, I460bw168=20.00%, I415bw216=31.25%, F144=22.50% → AVG=2.0396, MIN=1.5712
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION A: Ultra-fine 0.3125% around P91 B Champion (I600=0%)")
print("  B best: V1=26.25%/I460bw168=20.00%/I415bw216=31.25%/F144=22.50%")
print("  → AVG=2.0396, MIN=1.5712 (NEW P91 RECORD!)")
print("═"*70, flush=True)

STRATS_A = ["v1", "i460bw168", "i415bw216", "f144"]  # no i600!
base_a = {"v1": v1_data, "i460bw168": i460_bw168, "i415bw216": i415_bw216, "f144": fund_144}
ym_a = build_year_matrices(base_a, STRATS_A)

# Center at B best: v1=26.25%, i460=20%, i415=31.25%, f144=22.5%
# Ultra-fine ±2.5% range, 0.3125% step (31 in 10000)
configs_a = []
for wv1 in pct_range(2375, 2875, 31):        # 23.75% to 28.75%
    for wi460 in pct_range(1750, 2250, 31):   # 17.50% to 22.50%
        for wi415 in pct_range(2875, 3375, 31):  # 28.75% to 33.75%
            wf144 = 1.0 - wv1 - wi460 - wi415  # no i600!
            if 0.18 <= wf144 <= 0.28:
                configs_a.append({"v1": wv1, "i460bw168": wi460, "i415bw216": wi415,
                                   "f144": round(wf144, 8)})

print(f"\n  Ultra-fine grid (no I600): {len(configs_a)} configs (0.3125% step)", flush=True)
results_a = sweep_configs_numpy(configs_a, STRATS_A, ym_a)
bal_a = [(a, m, w) for a, m, w in results_a if a >= 2.000]
if bal_a:
    best_a = max(bal_a, key=lambda x: (x[1], x[0]))
    a_a, m_a, w_a = best_a
    yby_a = yby_for_weights(w_a, STRATS_A, ym_a)
    mark = " ★★ IMPROVED!" if m_a > P91_MIN else " converged" if m_a >= P91_MIN - 0.001 else ""
    print(f"  Result: AVG={a_a:.4f}, MIN={m_a:.4f}{mark}", flush=True)
    print(f"  YbY={yby_a}", flush=True)
    wfmt = " ".join(f"{k}={v*100:.2f}%" for k, v in sorted(w_a.items()) if v > 0.001)
    print(f"  Weights: {wfmt}", flush=True)
else:
    print("  No balanced config found!", flush=True)
    # fallback to B best directly
    a_a, m_a = 2.0396, 1.5712
    w_a = {"v1": 0.2625, "i460bw168": 0.200, "i415bw216": 0.3125, "f144": 0.225}
    yby_a = [2.972, 1.571, 1.578, 2.503, 1.573]
    print(f"  Using B best as champion", flush=True)

ultra_champ_balanced = (a_a, m_a, w_a, STRATS_A, ym_a, "A_ultrafine_noI600")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION B: AVG-max Ultra-Fine (chunked, narrowed range)
# P90: V1=4.5%/I415bw216=10%/I474bw216=40%/I600=7.5%/F144=38% → 2.304/1.205
# Narrowed: V1=3-6%, I415=8.5-11.5%, I474=38-42%, I600={5%,7.5%,10%}
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION B: AVG-max Ultra-Fine (0.25% step, chunked) — Narrowed Range")
print("  P90: V1=4.5%/I415bw216=10%/I474bw216=40%/I600=7.5%/F144=38% → 2.3038/1.2046")
print("  Narrowed: V1=3-6%, I415=8.5-11.5%, I474=38-42%, I600={5%,7.5%,10%}")
print("═"*70, flush=True)

STRATS_B = ["v1", "i415bw216", "i474bw216", "i600", "f144"]
base_b = {"v1": v1_data, "i415bw216": i415_bw216, "i474bw216": i474_bw216,
          "i600": i600_k2, "f144": fund_144}
ym_b = build_year_matrices(base_b, STRATS_B)

configs_b = []
for wv1 in pct_range(300, 600, 25):          # 3% to 6%
    for wi415 in pct_range(850, 1150, 25):    # 8.5% to 11.5%
        for wi474 in pct_range(3800, 4200, 25):  # 38% to 42%
            for wi600 in [0.05, 0.075, 0.10]:
                wf144 = 1.0 - wv1 - wi415 - wi474 - wi600
                if 0.30 <= wf144 <= 0.46:
                    configs_b.append({"v1": wv1, "i415bw216": wi415, "i474bw216": wi474,
                                      "i600": wi600, "f144": round(wf144, 8)})

print(f"\n  AVG-max fine grid: {len(configs_b)} configs (0.25% step, chunked)", flush=True)
results_b = sweep_configs_numpy(configs_b, STRATS_B, ym_b, chunk_size=30000)

if results_b:
    best_b_avg = max(results_b, key=lambda x: x[0])
    a_ba, m_ba, w_ba = best_b_avg
    yby_ba = yby_for_weights(w_ba, STRATS_B, ym_b)
    wfmt_ba = " ".join(f"{k}={v*100:.2f}%" for k, v in sorted(w_ba.items()) if v > 0.001)
    print(f"  AVG-max best: AVG={a_ba:.4f}, MIN={m_ba:.4f}", flush=True)
    print(f"  YbY={yby_ba}", flush=True)
    print(f"  Weights: {wfmt_ba}", flush=True)
    if a_ba > 2.3038:
        print(f"  ★★ NEW AVG-MAX RECORD! (AVG {a_ba:.4f} > P90's 2.3038)", flush=True)
    elif a_ba > 2.286:
        print(f"  ★ Beats P89 AVG-max (2.286)", flush=True)

    # Also wider search: V1=1-8%, I415=6-14%, I474=35-50%
    print(f"\n  Extended AVG-max range: V1=1-8%, I415=6-14%, I474=35-50%...", flush=True)
    configs_b2 = []
    for wv1 in pct_range(100, 800, 25):         # 1% to 8%
        for wi415 in pct_range(600, 1400, 25):   # 6% to 14%
            for wi474 in pct_range(3500, 5000, 25):  # 35% to 50%
                for wi600 in [0.05, 0.075]:
                    wf144 = 1.0 - wv1 - wi415 - wi474 - wi600
                    if 0.25 <= wf144 <= 0.50:
                        configs_b2.append({"v1": wv1, "i415bw216": wi415, "i474bw216": wi474,
                                           "i600": wi600, "f144": round(wf144, 8)})

    print(f"  Extended grid: {len(configs_b2)} configs", flush=True)
    results_b2 = sweep_configs_numpy(configs_b2, STRATS_B, ym_b, chunk_size=30000)
    if results_b2:
        best_b2_avg = max(results_b2, key=lambda x: x[0])
        a_b2, m_b2, w_b2 = best_b2_avg
        yby_b2 = yby_for_weights(w_b2, STRATS_B, ym_b)
        wfmt_b2 = " ".join(f"{k}={v*100:.2f}%" for k, v in sorted(w_b2.items()) if v > 0.001)
        print(f"  Extended best: AVG={a_b2:.4f}, MIN={m_b2:.4f}", flush=True)
        print(f"  YbY={yby_b2}", flush=True)
        print(f"  Weights: {wfmt_b2}", flush=True)
        if a_b2 > a_ba:
            print(f"  ★ Extended range found better AVG! ({a_b2:.4f} vs {a_ba:.4f})", flush=True)
            a_ba, m_ba, w_ba = a_b2, m_b2, w_b2
            if a_ba > 2.3038:
                print(f"  ★★ NEW AVG-MAX RECORD! (AVG {a_ba:.4f} > P90's 2.3038)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION C: New lb signals in balanced combo
# I420: bridge=2.277 (vs I415 bridge=2.880) — too weak alone
# But I420+I415 together? I430: 2022=1.782 — strong 2022!
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION C: New lb Signals in Balanced Combo (no I600)")
print("  Goal: can I420/I430 +I415 improve further?")
print("  I430: 2022=1.782 (strong!), 2023=0.487 (weak)")
print("  Architecture: V1 + I460bw168 + I415bw216 + Inew_bw216 (no I600, no small allocation)")
print("═"*70, flush=True)

best_c_overall = None

for new_lbl, new_data, new_lb in [
    ("I420_bw216", i420_bw216, 420),
    ("I430_bw216", i430_bw216, 430),
]:
    if new_data is None or new_data.get("_min", 0.0) < 0.3:
        print(f"  {new_lbl}: not loaded or too weak, skipping", flush=True)
        continue

    y2022_new = new_data.get("2022", {}).get("sharpe", 0.0)
    y2023_new = new_data.get("2023", {}).get("sharpe", 0.0)
    print(f"\n  Testing {new_lbl} alongside I415 (no I600):", flush=True)
    print(f"    {new_lbl}: 2022={y2022_new:.3f}, 2023={y2023_new:.3f}", flush=True)

    # 5-signal ensemble: V1, I460bw168, I415bw216, Inew, F144
    strats_c = ["v1", "i460bw168", "i415bw216", f"i{new_lb}bw216", "f144"]
    base_c = {
        "v1": v1_data, "i460bw168": i460_bw168, "i415bw216": i415_bw216,
        f"i{new_lb}bw216": new_data, "f144": fund_144
    }
    ym_c = build_year_matrices(base_c, strats_c)

    # Sweep: new lb gets small 2.5-15% weight, rest as in B architecture
    configs_c = []
    for wv1 in pct_range(1250, 3500, 125):
        for wi460 in pct_range(500, 2250, 125):
            for wi415 in pct_range(1500, 3500, 125):
                for wi_new in pct_range(250, 1500, 125):
                    wf144 = 1.0 - wv1 - wi460 - wi415 - wi_new
                    if 0.15 <= wf144 <= 0.30:
                        configs_c.append({
                            "v1": wv1, "i460bw168": wi460, "i415bw216": wi415,
                            f"i{new_lb}bw216": round(wi_new, 8),
                            "f144": round(wf144, 8)
                        })

    print(f"    Balanced sweep: {len(configs_c)} configs", flush=True)
    results_c = sweep_configs_numpy(configs_c, strats_c, ym_c, chunk_size=30000)
    r = best_balanced(results_c)
    if r:
        a, m, w = r
        yby = yby_for_weights(w, strats_c, ym_c)
        mark = " ★★ BEATS P91!" if m > P91_MIN else " > P90" if m > P90_MIN else " > P89" if m > 1.5463 else ""
        print(f"    Best: AVG={a:.4f}, MIN={m:.4f}{mark}", flush=True)
        print(f"    YbY={yby}", flush=True)
        wfmt = " ".join(f"{k}={v*100:.2f}%" for k, v in sorted(w.items()) if v > 0.001)
        print(f"    Weights: {wfmt}", flush=True)
        if best_c_overall is None or m > best_c_overall[0][1]:
            best_c_overall = ((a, m, w), strats_c, ym_c, new_lbl)
    else:
        print(f"    No balanced config found", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION D: Final champion determination + save
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION D: Final Champion + Save")
print("═"*70, flush=True)

# Collect all balanced candidates
a_ch, m_ch, w_ch, strats_ch, ym_ch, lbl_ch = ultra_champ_balanced

# Check if C improved on A
if best_c_overall:
    (a_c, m_c, w_c), strats_c2, ym_c2, lbl_c2 = best_c_overall
    if m_c > m_ch:
        a_ch, m_ch, w_ch, strats_ch, ym_ch, lbl_ch = a_c, m_c, w_c, strats_c2, ym_c2, f"C_{lbl_c2}"
        print(f"  C improved on A! New champion: {lbl_ch}", flush=True)

yby_ch = yby_for_weights(w_ch, strats_ch, ym_ch)
wfmt_final = " ".join(f"{k}={v*100:.2f}%" for k, v in sorted(w_ch.items()) if v > 0.001)

print(f"\n★★★ PHASE 91 BALANCED CHAMPION [{lbl_ch}]:", flush=True)
print(f"    AVG={a_ch:.4f}, MIN={m_ch:.4f}", flush=True)
print(f"    YbY={yby_ch}", flush=True)
print(f"    Weights: {wfmt_final}", flush=True)

if m_ch > P91_MIN:
    print(f"    ★★ IMPROVES ON P91b B (MIN {m_ch:.4f} > {P91_MIN:.4f})!", flush=True)
elif m_ch >= P91_MIN - 0.001:
    print(f"    ≈ P91 B champion (MIN {m_ch:.4f} ≈ {P91_MIN:.4f})", flush=True)
elif m_ch > P90_MIN:
    print(f"    ★ BEATS P90 (MIN {m_ch:.4f} > {P90_MIN:.4f})!", flush=True)

champion_config = {
    "phase": "91b",
    "section": lbl_ch,
    "avg_sharpe": round(a_ch, 4),
    "min_sharpe": round(m_ch, 4),
    "yby_sharpes": {y: round(v, 3) for y, v in zip(YEARS, yby_ch)},
    "weights": {k: round(v, 6) for k, v in w_ch.items() if v > 0.001},
    "signal_config": {
        "strat_names": strats_ch,
        "architecture": "V1 + I460bw168 + I415bw216 (+ optional new lb) + F144 [NO I600!]",
        "notes": {
            "v1":        "nexus_alpha_v1 k=2",
            "i460bw168": "idio_momentum_alpha lb=460 bw=168 k=4",
            "i415bw216": "idio_momentum_alpha lb=415 bw=216 k=4",
            "i420bw216": "idio_momentum_alpha lb=420 bw=216 k=4",
            "i430bw216": "idio_momentum_alpha lb=430 bw=216 k=4",
            "f144":      "funding_momentum_alpha lb=144 k=2",
        }
    }
}
out_path = "configs/ensemble_p91_balanced.json"
with open(out_path, "w") as f:
    json.dump(champion_config, f, indent=2)
print(f"\n    Saved: {out_path}", flush=True)

# Save AVG-max if improved
if results_b and a_ba > 2.3038:
    avgmax_config = {
        "phase": "91b",
        "section": "B:AVGmax-fine",
        "type": "avg_max",
        "avg_sharpe": round(a_ba, 4),
        "min_sharpe": round(m_ba, 4),
        "weights": {k: round(v, 6) for k, v in w_ba.items() if v > 0.001},
    }
    with open("configs/ensemble_p91_avgmax.json", "w") as f:
        json.dump(avgmax_config, f, indent=2)
    print(f"  ★★ NEW AVG-MAX CONFIG SAVED: AVG={a_ba:.4f}/MIN={m_ba:.4f}", flush=True)

# Final Pareto
print("\nFINAL PARETO FRONTIER:", flush=True)
all_points = [
    (2.3038, 1.2046, "P90-AVG-MAX"),
    (2.0211, 1.5614, "P90-BALANCED"),
    (a_ch, m_ch, f"P91-{lbl_ch}"),
]
if results_b and 'a_ba' in dir():
    all_points.append((a_ba, m_ba, "P91-C-AVGmax"))

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

print("\nPHASE 91b COMPLETE", flush=True)
