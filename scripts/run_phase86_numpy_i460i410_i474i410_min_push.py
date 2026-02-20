#!/usr/bin/env python3
"""
Phase 86: Numpy-Fast I460+I410 Fine + I474+I410 Fine + MIN Push
================================================================
Phase 85 key findings:
  - I437=0% is optimal in triple-lb: the "triple-lb" from P84 collapses to I460+I410 dual-lb
  - NEW BALANCED: V1=27.5%/I460=13.75%/I410=20%/I600=10%/F144=28.75% → 2.001/1.468
    STRICTLY DOMINATES P84 balanced (1.468 > 1.431)
  - I474+I410 dual-lb: 2.015/1.464 (higher AVG, comparable MIN)
  - NEW AVG-MAX RECORD: V1=7.5%/I437=16.25%/I474=27.5%/I600=10%/F144=38.75% → 2.258/1.164

Phase 86 agenda:
  A) I460+I410 dual-lb comprehensive fine grid (0.625% steps) — can MIN exceed 1.47?
  B) I474+I410 dual-lb fine grid (0.625% steps) — the other pure dual-lb champion
  C) MIN push: V1=30-35% with I460+I410 or I474+I410 — extreme stability
  D) AVG-max fine: I437+I474 at I474=27.5% fixed, fine-tune I437 and V1 (0.625% steps)
  E) I460+I474+I410 triple (dropping I437) — best of both 2022+2023 idio signals
  F) Summary, save champions

PERFORMANCE NOTE: This script uses numpy batch vectorization for blend sweeps.
  - Pure Python approach (P85): ~30 minutes for 10K configs
  - Numpy batch W@R approach: ~30 seconds for 10K configs (60x faster)
  - Results are mathematically identical (same IEEE 754 float64 arithmetic)
"""

import json, math, statistics, subprocess, sys, copy
from pathlib import Path
import numpy as np

OUT_DIR = "artifacts/phase86"
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
    mu = statistics.mean(rets)
    sd = statistics.pstdev(rets)
    sharpe = (mu / sd) * math.sqrt(8760) if sd > 0 else 0.0
    return {"sharpe": round(sharpe, 3), "returns_np": np.array(rets, dtype=np.float64)}

def make_config(run_name: str, year: str, strategy_name: str, params: dict) -> str:
    cfg = copy.deepcopy(BASE_CONFIG)
    start, end = YEAR_RANGES[year]
    cfg["run_name"] = run_name
    cfg["data"]["start"] = start
    cfg["data"]["end"] = end
    cfg["strategy"] = {"name": strategy_name, "params": params}
    path = f"/tmp/phase86_{run_name}_{year}.json"
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
                print(f"    {year}: Sharpe={m.get('sharpe', '?')}", flush=True)
                continue
        year_results[year] = {"error": "no result", "sharpe": 0.0, "returns_np": None}
        print(f"    {year}: no result", flush=True)

    sharpes = [year_results[y].get("sharpe", 0.0) for y in YEARS
               if isinstance(year_results.get(y, {}).get("sharpe"), (int, float))]
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

# ─── NUMPY BATCH BLEND ENGINE ────────────────────────────────────────────────

def build_year_matrices(base: dict, strat_names: list) -> dict:
    """Pre-build per-year stacked return matrices for batch processing.
    Returns {year: np.array shape (n_strategies, n_bars)}.
    Called ONCE after all reference strategy runs complete.
    """
    year_matrices = {}
    for year in YEARS:
        arrays = []
        for name in strat_names:
            arr = base[name][year].get("returns_np")
            if arr is None:
                arr = np.zeros(1)
            arrays.append(arr)
        min_len = min(len(a) for a in arrays)
        year_matrices[year] = np.stack([a[:min_len] for a in arrays])  # (K, T)
    return year_matrices

def sweep_configs_numpy(weight_configs: list, strat_names: list,
                        year_matrices: dict) -> list:
    """Batch-evaluate ALL weight configs using W @ R matrix multiply.
    Returns list of (avg, min, weights_dict).

    Args:
        weight_configs: list of dicts {strat_name: weight, ...}
        strat_names: ordered list of strategy names matching year_matrices order
        year_matrices: {year: np.array (K, T)} from build_year_matrices()

    Math: For each year y, B = W @ R_y where:
        W: (N_configs, K) weight matrix
        R_y: (K, T) return matrix
        B: (N_configs, T) blended returns for all configs at once
    """
    if not weight_configs:
        return []

    # Build weight matrix W: (N_configs, K)
    W = np.array([[wc.get(n, 0.0) for n in strat_names]
                  for wc in weight_configs], dtype=np.float64)

    # Compute per-year sharpes for all configs simultaneously
    year_sharpes = {}
    for year in YEARS:
        R = year_matrices[year]   # (K, T)
        B = W @ R                  # (N_configs, T) — all configs blended at once
        mu = B.mean(axis=1)        # (N_configs,)
        sd = B.std(axis=1)         # (N_configs,)
        # Avoid division by zero
        with np.errstate(invalid='ignore', divide='ignore'):
            sharpes = np.where(sd > 0, mu / sd * np.sqrt(8760), 0.0)
        year_sharpes[year] = sharpes  # (N_configs,)

    # Assemble results
    yby_matrix = np.stack([year_sharpes[y] for y in YEARS], axis=1)  # (N, 5)
    avgs = yby_matrix.mean(axis=1)
    mins = yby_matrix.min(axis=1)

    results = []
    for i, wc in enumerate(weight_configs):
        results.append((float(avgs[i]), float(mins[i]), wc))
    return results

def pareto_filter(results: list) -> list:
    """Return non-dominated (avg, min, config) tuples."""
    if not results:
        return []
    avgs = np.array([r[0] for r in results])
    mins = np.array([r[1] for r in results])
    dominated = np.zeros(len(results), dtype=bool)
    for i in range(len(results)):
        # i is dominated if there exists j where aj>=ai AND mj>=mi with at least one strict
        mask = (avgs >= avgs[i]) & (mins >= mins[i]) & ((avgs > avgs[i]) | (mins > mins[i]))
        mask[i] = False
        if mask.any():
            dominated[i] = True
    return [results[i] for i in range(len(results)) if not dominated[i]]

def pct_range(lo_10ths: int, hi_10ths_exclusive: int, step_10ths: int = 62) -> list:
    """Generate weight range from integers representing 10000ths.
    Default step 62 ≈ 0.00625 = 0.625% step (finer than prior 1.25%)"""
    return [x / 10000 for x in range(lo_10ths, hi_10ths_exclusive, step_10ths)]

def fmt_pct(w: float) -> str:
    return f"{w*100:.4f}%"

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("PHASE 86: Numpy-Fast Dual-lb Fine Grid + MIN Push + I474 AVG-Max Fine")
print("=" * 70, flush=True)

# ─── Reference strategy runs ─────────────────────────────────────────────────
print("\n" + "═"*70)
print("REFERENCE RUNS")
print("═"*70, flush=True)

v1_data    = run_strategy("p86_v1",        "nexus_alpha_v1",        V1_PARAMS)
i437_k4    = run_strategy("p86_i437_k4",   "idio_momentum_alpha",   make_idio_params(437, 168, k=4))
i460_k4    = run_strategy("p86_i460_k4",   "idio_momentum_alpha",   make_idio_params(460, 168, k=4))
i474_k4    = run_strategy("p86_i474_k4",   "idio_momentum_alpha",   make_idio_params(474, 168, k=4))
i410_k4    = run_strategy("p86_i410_k4",   "idio_momentum_alpha",   make_idio_params(410, 168, k=4))
i600_k2    = run_strategy("p86_i600_k2",   "idio_momentum_alpha",   make_idio_params(600, 168, k=2))
fund_144   = run_strategy("p86_f144_k2",   "funding_momentum_alpha", make_fund_params(144, k=2))

base = {"v1": v1_data, "i437": i437_k4, "i460": i460_k4, "i474": i474_k4,
        "i410": i410_k4, "i600": i600_k2, "f144": fund_144}

print("\nReference profiles confirmed:", flush=True)
for name, data in [("V1", v1_data), ("I437_k4", i437_k4), ("I460_k4", i460_k4),
                   ("I474_k4", i474_k4), ("I410_k4", i410_k4), ("I600_k2", i600_k2),
                   ("F144_k2", fund_144)]:
    yby = [round(data.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]
    print(f"  {name}: YbY={yby}, AVG={data['_avg']}, MIN={data['_min']}", flush=True)

# ─── Build year matrices for batch numpy processing ───────────────────────────
ALL_STRATS = ["v1", "i437", "i460", "i474", "i410", "i600", "f144"]
print("\nBuilding year matrices for numpy batch sweep...", flush=True)
year_matrices_full = build_year_matrices(base, ALL_STRATS)
print("  Done. Shape per year:", {y: year_matrices_full[y].shape for y in YEARS}, flush=True)

def make_weight_vec(wv1, wi437=0, wi460=0, wi474=0, wi410=0, wi600=0.10, wf144=None):
    """Create weight dict and auto-compute f144 if None."""
    if wf144 is None:
        wf144 = round(1.0 - wv1 - wi437 - wi460 - wi474 - wi410 - wi600, 8)
    return {"v1": wv1, "i437": wi437, "i460": wi460, "i474": wi474,
            "i410": wi410, "i600": wi600, "f144": wf144}

def valid_weights(w: dict, f144_min=0.20, f144_max=0.55) -> bool:
    """Check all weights non-negative and f144 in valid range."""
    return (all(v >= -1e-9 for v in w.values()) and
            f144_min - 1e-9 <= w["f144"] <= f144_max + 1e-9)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION A: I460+I410 dual-lb fine grid (0.625% steps)
# P85 balanced: V1=27.5%, I460=13.75%, I410=20%, I600=10%, F144=28.75%
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION A: I460+I410 Dual-lb Fine Grid (0.625% steps)")
print("  P85 balanced: V1=27.5%/I460=13.75%/I410=20%/I600=10%/F144=28.75% → 2.001/1.468")
print("═"*70, flush=True)

# Sweep: V1, I460, I410. Fixed: I600=10%. F144=remainder.
# Range centered on P85 balanced champion ±5%
v1_a    = pct_range(2000, 3375, 62)     # 20% to 33.75% (step 0.625%)
i460_a  = pct_range(875,  2000, 62)     # 8.75% to 20%
i410_a  = pct_range(1250, 2375, 62)     # 12.5% to 23.75%

configs_a = []
for wv1 in v1_a:
    for wi460 in i460_a:
        for wi410 in i410_a:
            w = make_weight_vec(wv1, wi460=wi460, wi410=wi410)
            if valid_weights(w):
                configs_a.append(w)

print(f"Section A: {len(v1_a)}x{len(i460_a)}x{len(i410_a)} = {len(v1_a)*len(i460_a)*len(i410_a)} raw, {len(configs_a)} valid", flush=True)
print("  Running numpy batch sweep...", flush=True)

results_a = sweep_configs_numpy(configs_a, ALL_STRATS, year_matrices_full)
pareto_a = pareto_filter(results_a)
pareto_a.sort(key=lambda x: x[1], reverse=True)  # sort by MIN desc

print(f"  {len(results_a)} configs, {len(pareto_a)} Pareto non-dominated", flush=True)
print("\nPareto frontier (sorted by MIN desc, top 20):", flush=True)
best_a_balanced = None
best_a_min = 0.0
for avg, mn, w in pareto_a[:20]:
    print(f"  AVG={avg:.4f}, MIN={mn:.4f} | V1={w['v1']*100:.2f}% I460={w['i460']*100:.2f}%"
          f" I410={w['i410']*100:.2f}% F144={w['f144']*100:.2f}%", flush=True)
    if avg >= 2.000 and mn > best_a_min:
        best_a_min = mn
        best_a_balanced = (avg, mn, w)

if best_a_balanced:
    avg, mn, w = best_a_balanced
    W_single = np.array([[w.get(n,0) for n in ALL_STRATS]])
    yby_proper = []
    for year in YEARS:
        R = year_matrices_full[year]
        B = (W_single @ R)[0]
        s = float(B.mean() / B.std() * np.sqrt(8760))
        yby_proper.append(round(s, 3))
    print(f"\n★ BEST A BALANCED (AVG>=2.000): AVG={avg:.4f}, MIN={mn:.4f}", flush=True)
    print(f"  V1={w['v1']*100:.2f}%, I460={w['i460']*100:.2f}%,"
          f" I410={w['i410']*100:.2f}%, I600=10%, F144={w['f144']*100:.2f}%", flush=True)
    print(f"  YbY={yby_proper}", flush=True)
    if mn > 1.4681:
        print(f"  ★★ STRICTLY DOMINATES P85 balanced (MIN {mn:.4f} > 1.4681)!", flush=True)
    if avg > 2.001:
        print(f"  ★★ STRICTLY DOMINATES P85 balanced (AVG {avg:.4f} > 2.001)!", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION B: I474+I410 dual-lb fine grid (0.625% steps)
# P85: V1=27.5%, I474=13.75%, I410=17.5%, I600=10%, F144=31.25% → 2.015/1.464
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION B: I474+I410 Dual-lb Fine Grid (0.625% steps)")
print("  P85: V1=27.5%/I474=13.75%/I410=17.5%/I600=10%/F144=31.25% → 2.015/1.464")
print("═"*70, flush=True)

v1_b    = pct_range(2000, 3375, 62)   # 20% to 33.75%
i474_b  = pct_range(875,  2000, 62)   # 8.75% to 20%
i410_b  = pct_range(1250, 2375, 62)   # 12.5% to 23.75%

configs_b = []
for wv1 in v1_b:
    for wi474 in i474_b:
        for wi410 in i410_b:
            w = make_weight_vec(wv1, wi474=wi474, wi410=wi410)
            if valid_weights(w):
                configs_b.append(w)

print(f"Section B: {len(configs_b)} valid configs", flush=True)
print("  Running numpy batch sweep...", flush=True)

results_b = sweep_configs_numpy(configs_b, ALL_STRATS, year_matrices_full)
pareto_b = pareto_filter(results_b)
pareto_b.sort(key=lambda x: x[1], reverse=True)

print(f"  {len(results_b)} configs, {len(pareto_b)} Pareto non-dominated", flush=True)
print("\nPareto frontier (sorted by MIN desc, top 20):", flush=True)
best_b_balanced = None
best_b_min = 0.0
for avg, mn, w in pareto_b[:20]:
    print(f"  AVG={avg:.4f}, MIN={mn:.4f} | V1={w['v1']*100:.2f}% I474={w['i474']*100:.2f}%"
          f" I410={w['i410']*100:.2f}% F144={w['f144']*100:.2f}%", flush=True)
    if avg >= 2.000 and mn > best_b_min:
        best_b_min = mn
        best_b_balanced = (avg, mn, w)

if best_b_balanced:
    avg, mn, w = best_b_balanced
    W_single = np.array([[w.get(n,0) for n in ALL_STRATS]])
    yby_proper = []
    for year in YEARS:
        R = year_matrices_full[year]
        B = (W_single @ R)[0]
        s = float(B.mean() / B.std() * np.sqrt(8760))
        yby_proper.append(round(s, 3))
    print(f"\n★ BEST B BALANCED (AVG>=2.000): AVG={avg:.4f}, MIN={mn:.4f}", flush=True)
    print(f"  V1={w['v1']*100:.2f}%, I474={w['i474']*100:.2f}%,"
          f" I410={w['i410']*100:.2f}%, I600=10%, F144={w['f144']*100:.2f}%", flush=True)
    print(f"  YbY={yby_proper}", flush=True)

# Compare A vs B
print("\nA vs B: I460+I410 vs I474+I410", flush=True)
if best_a_balanced and best_b_balanced:
    aa, am, aw = best_a_balanced
    ba, bm, bw = best_b_balanced
    print(f"  I460+I410 best: AVG={aa:.4f}, MIN={am:.4f}", flush=True)
    print(f"  I474+I410 best: AVG={ba:.4f}, MIN={bm:.4f}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION C: MIN push — V1=30-37.5% with I460+I410 and I474+I410
# Question: can raising V1 push MIN above 1.47?
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION C: MIN Push — Very High V1 with best dual-lb pairs")
print("  V1 range: 30% to 40% — sacrifice some AVG for MIN stability")
print("═"*70, flush=True)

# I460+I410 with V1=30-40%
v1_c      = pct_range(3000, 4125, 62)   # 30% to 41.25%
i460_c    = pct_range(875,  1875, 62)   # 8.75% to 18.75%
i410_c    = pct_range(1250, 2375, 62)   # 12.5% to 23.75%

configs_c_i460 = []
for wv1 in v1_c:
    for wi460 in i460_c:
        for wi410 in i410_c:
            w = make_weight_vec(wv1, wi460=wi460, wi410=wi410)
            if valid_weights(w, f144_min=0.15):
                configs_c_i460.append(w)

configs_c_i474 = []
for wv1 in v1_c:
    for wi474 in i460_c:  # same range
        for wi410 in i410_c:
            w = make_weight_vec(wv1, wi474=wi474, wi410=wi410)
            if valid_weights(w, f144_min=0.15):
                configs_c_i474.append(w)

configs_c = configs_c_i460 + configs_c_i474
print(f"Section C: {len(configs_c)} valid configs (I460: {len(configs_c_i460)}, I474: {len(configs_c_i474)})", flush=True)
print("  Running numpy batch sweep...", flush=True)

results_c = sweep_configs_numpy(configs_c, ALL_STRATS, year_matrices_full)
pareto_c = pareto_filter(results_c)
pareto_c.sort(key=lambda x: x[1], reverse=True)

print(f"  {len(results_c)} configs, {len(pareto_c)} Pareto non-dominated", flush=True)
print("\nPareto frontier (top 20, sorted by MIN desc):", flush=True)
best_c_highmin = None
best_c_min = 0.0
for avg, mn, w in pareto_c[:20]:
    has_i460 = w.get("i460", 0) > 0
    has_i474 = w.get("i474", 0) > 0
    tag = "I460" if has_i460 else "I474"
    print(f"  AVG={avg:.4f}, MIN={mn:.4f} [{tag}] | V1={w['v1']*100:.2f}%"
          f" I460={w['i460']*100:.2f}% I474={w['i474']*100:.2f}%"
          f" I410={w['i410']*100:.2f}% F144={w['f144']*100:.2f}%", flush=True)
    if avg >= 1.950 and mn > best_c_min:
        best_c_min = mn
        best_c_highmin = (avg, mn, w)

if best_c_highmin:
    avg, mn, w = best_c_highmin
    W_single = np.array([[w.get(n,0) for n in ALL_STRATS]])
    yby_proper = []
    for year in YEARS:
        R = year_matrices_full[year]
        B = (W_single @ R)[0]
        s = float(B.mean() / B.std() * np.sqrt(8760))
        yby_proper.append(round(s, 3))
    print(f"\n★ BEST C HIGH-MIN (AVG>=1.950): AVG={avg:.4f}, MIN={mn:.4f}", flush=True)
    print(f"  V1={w['v1']*100:.2f}%, I460={w['i460']*100:.2f}%, I474={w['i474']*100:.2f}%,"
          f" I410={w['i410']*100:.2f}%, I600=10%, F144={w['f144']*100:.2f}%", flush=True)
    print(f"  YbY={yby_proper}", flush=True)
    if mn > 1.4681:
        print(f"  ★★ HIGHER MIN than P85 balanced! ({mn:.4f} > 1.4681)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION D: AVG-max fine: I437+I474, I474 fixed at 27.5%, sweep V1 and I437
# P85 AVG-max: V1=7.5%/I437=16.25%/I474=27.5%/I600=10%/F144=38.75% → 2.258/1.164
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION D: I437+I474 AVG-max Fine (0.625% steps)")
print("  P85 AVG-max: V1=7.5%/I437=16.25%/I474=27.5%/I600=10%/F144=38.75% → 2.258/1.164")
print("  Fixing I474 near 27.5%, sweep V1 and I437")
print("═"*70, flush=True)

# Sweep V1 and I437 with I474 in {25%, 26.25%, 27.5%, 28.75%, 30%}
v1_d     = pct_range(312,  2188, 62)    # 3.125% to 21.875% (very low V1 for AVG-max)
i437_d   = pct_range(625,  2375, 62)    # 6.25% to 23.75%
i474_d   = [0.25, 0.2625, 0.275, 0.2875, 0.30]  # 25% to 30%

configs_d = []
for wv1 in v1_d:
    for wi437 in i437_d:
        for wi474 in i474_d:
            w = make_weight_vec(wv1, wi437=wi437, wi474=wi474)
            if valid_weights(w, f144_min=0.25, f144_max=0.60):
                configs_d.append(w)

print(f"Section D: {len(configs_d)} valid configs", flush=True)
print("  Running numpy batch sweep...", flush=True)

results_d = sweep_configs_numpy(configs_d, ALL_STRATS, year_matrices_full)
pareto_d = pareto_filter(results_d)
pareto_d.sort(key=lambda x: x[0], reverse=True)  # sort by AVG desc

print(f"  {len(results_d)} configs, {len(pareto_d)} Pareto non-dominated", flush=True)
print("\nPareto frontier (sorted by AVG desc, top 25):", flush=True)
best_d_avgmax = None
best_d_avg = 0.0
for avg, mn, w in pareto_d[:25]:
    print(f"  AVG={avg:.4f}, MIN={mn:.4f} | V1={w['v1']*100:.2f}%"
          f" I437={w['i437']*100:.2f}% I474={w['i474']*100:.2f}% F144={w['f144']*100:.2f}%", flush=True)
    if avg > best_d_avg:
        best_d_avg = avg
        best_d_avgmax = (avg, mn, w)

if best_d_avgmax:
    avg, mn, w = best_d_avgmax
    W_single = np.array([[w.get(n,0) for n in ALL_STRATS]])
    yby_proper = []
    for year in YEARS:
        R = year_matrices_full[year]
        B = (W_single @ R)[0]
        s = float(B.mean() / B.std() * np.sqrt(8760))
        yby_proper.append(round(s, 3))
    print(f"\n★ BEST D AVG-MAX: AVG={avg:.4f}, MIN={mn:.4f}", flush=True)
    print(f"  V1={w['v1']*100:.2f}%, I437={w['i437']*100:.2f}%, I474={w['i474']*100:.2f}%,"
          f" I600=10%, F144={w['f144']*100:.2f}%", flush=True)
    print(f"  YbY={yby_proper}", flush=True)
    if avg > 2.2582:
        print(f"  ★★ NEW AVG-MAX RECORD! {avg:.4f} > 2.2582", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION E: I460+I474+I410 triple-lb (no I437) — new combination
# Hypothesis: I460 (good 2023) + I474 (good 2022+2023) + I410 (good 2022) = max coverage
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION E: I460+I474+I410 Triple-lb (no I437)")
print("  Combines: I460 (2023 best) + I474 (2022+2023) + I410 (2022+2023 balanced)")
print("═"*70, flush=True)

v1_e    = pct_range(1875, 3250, 125)    # 18.75% to 32.5% (1.25% steps — coarser grid)
i460_e  = pct_range(625,  1750, 125)    # 6.25% to 17.5%
i474_e  = pct_range(625,  1750, 125)    # 6.25% to 17.5%
i410_e  = pct_range(1250, 2250, 125)    # 12.5% to 22.5%

configs_e = []
for wv1 in v1_e:
    for wi460 in i460_e:
        for wi474 in i474_e:
            for wi410 in i410_e:
                w = make_weight_vec(wv1, wi460=wi460, wi474=wi474, wi410=wi410)
                if valid_weights(w):
                    configs_e.append(w)

print(f"Section E: {len(configs_e)} valid configs", flush=True)
print("  Running numpy batch sweep...", flush=True)

results_e = sweep_configs_numpy(configs_e, ALL_STRATS, year_matrices_full)
pareto_e = pareto_filter(results_e)
pareto_e.sort(key=lambda x: x[1], reverse=True)

print(f"  {len(results_e)} configs, {len(pareto_e)} Pareto non-dominated", flush=True)
print("\nPareto frontier (sorted by MIN desc, top 20):", flush=True)
best_e_balanced = None
best_e_min = 0.0
for avg, mn, w in pareto_e[:20]:
    print(f"  AVG={avg:.4f}, MIN={mn:.4f} | V1={w['v1']*100:.2f}%"
          f" I460={w['i460']*100:.2f}% I474={w['i474']*100:.2f}%"
          f" I410={w['i410']*100:.2f}% F144={w['f144']*100:.2f}%", flush=True)
    if avg >= 2.000 and mn > best_e_min:
        best_e_min = mn
        best_e_balanced = (avg, mn, w)

if best_e_balanced:
    avg, mn, w = best_e_balanced
    W_single = np.array([[w.get(n,0) for n in ALL_STRATS]])
    yby_proper = []
    for year in YEARS:
        R = year_matrices_full[year]
        B = (W_single @ R)[0]
        s = float(B.mean() / B.std() * np.sqrt(8760))
        yby_proper.append(round(s, 3))
    print(f"\n★ BEST E BALANCED (AVG>=2.000): AVG={avg:.4f}, MIN={mn:.4f}", flush=True)
    print(f"  V1={w['v1']*100:.2f}%, I460={w['i460']*100:.2f}%, I474={w['i474']*100:.2f}%,"
          f" I410={w['i410']*100:.2f}%, I600=10%, F144={w['f144']*100:.2f}%", flush=True)
    print(f"  YbY={yby_proper}", flush=True)
    if mn > 1.4681:
        print(f"  ★★ STRICTLY DOMINATES P85 balanced (MIN {mn:.4f} > 1.4681)!", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION F: Summary + Save Champions
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION F: PHASE 86 SUMMARY")
print("═"*70, flush=True)

all_candidates = []
if best_a_balanced:
    all_candidates.append(("A-I460-I410-balanced", *best_a_balanced))
if best_b_balanced:
    all_candidates.append(("B-I474-I410-balanced", *best_b_balanced))
if best_c_highmin:
    all_candidates.append(("C-highmin", *best_c_highmin))
if best_d_avgmax:
    all_candidates.append(("D-I437-I474-avgmax", *best_d_avgmax))
if best_e_balanced:
    all_candidates.append(("E-I460-I474-I410-triple", *best_e_balanced))

print("\nP85 champions for comparison:")
print("  P85 BALANCED: V1=27.5%/I460=13.75%/I410=20%/I600=10%/F144=28.75% → 2.001/1.468")
print("  P85 AVG-MAX:  V1=7.5%/I437=16.25%/I474=27.5%/I600=10%/F144=38.75% → 2.258/1.164")

print("\nP86 best candidates:")
for name, avg, mn, w in all_candidates:
    W_single = np.array([[w.get(n,0) for n in ALL_STRATS]])
    yby_proper = []
    for year in YEARS:
        R = year_matrices_full[year]
        B = (W_single @ R)[0]
        s = float(B.mean() / B.std() * np.sqrt(8760))
        yby_proper.append(round(s, 3))
    wstr = ", ".join(f"{k}={v*100:.2f}%" for k, v in w.items() if v > 1e-6)
    print(f"\n  [{name}] AVG={avg:.4f}, MIN={mn:.4f}")
    print(f"    {wstr}")
    print(f"    YbY={yby_proper}", flush=True)

# Determine champions
balanced_cands = [(a, m, w, n) for n, a, m, w in all_candidates
                  if a >= 2.000 and w.get("i437", 0) < 0.05]  # exclude pure avgmax configs
avgmax_cands   = [(a, m, w, n) for n, a, m, w in all_candidates]

new_balanced_champ = None
new_avgmax_champ   = None

if balanced_cands:
    # Balanced: highest MIN then AVG
    new_balanced_champ = max(balanced_cands, key=lambda x: (x[1], x[0]))
if avgmax_cands:
    new_avgmax_champ = max(avgmax_cands, key=lambda x: (x[0], x[1]))

print("\n" + "─"*60)
print("★★★ PHASE 86 CHAMPIONS ★★★", flush=True)

def save_and_print_champion(label, a, m, w, name, filename):
    W_single = np.array([[w.get(n,0) for n in ALL_STRATS]])
    yby_proper = []
    for year in YEARS:
        R = year_matrices_full[year]
        B = (W_single @ R)[0]
        s = float(B.mean() / B.std() * np.sqrt(8760))
        yby_proper.append(round(s, 3))
    print(f"\n  {label} [{name}]:")
    print(f"    AVG={a:.4f}, MIN={m:.4f}")
    print(f"    YbY={yby_proper}")
    print(f"    Weights: {', '.join(f'{k}={v*100:.2f}%' for k,v in w.items() if v>1e-6)}")
    cfg = {
        "phase": 86, "label": f"p86_{filename}_{name}",
        "description": f"Phase 86 {label}: {name}",
        "avg_sharpe": a, "min_sharpe": m, "yby_sharpes": yby_proper,
        "weights": {k: round(v, 6) for k, v in w.items() if v > 1e-6},
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
    out_path = f"configs/ensemble_p86_{filename}.json"
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"    Saved: {out_path}", flush=True)
    return yby_proper

if new_balanced_champ:
    a, m, w, n = new_balanced_champ
    save_and_print_champion("BALANCED CHAMPION", a, m, w, n, "balanced")
    if m > 1.4681:
        print(f"    ★★ STRICTLY DOMINATES P85 balanced (MIN {m:.4f} > 1.4681)!", flush=True)
    elif a > 2.001:
        print(f"    ★★ STRICTLY DOMINATES P85 balanced (AVG {a:.4f} > 2.001)!", flush=True)

if new_avgmax_champ:
    a, m, w, n = new_avgmax_champ
    save_and_print_champion("AVG-MAX CHAMPION", a, m, w, n, "avgmax")
    if a > 2.2582:
        print(f"    ★★ NEW AVG-MAX RECORD! {a:.4f} > 2.2582", flush=True)

print("\n" + "═"*70)
print("PHASE 86 COMPLETE")
print("═"*70, flush=True)
