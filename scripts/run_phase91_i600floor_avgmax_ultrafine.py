#!/usr/bin/env python3
"""
Phase 91: I600 Floor + AVG-max Ultra-Fine + lb 420-430 Profiles
================================================================
Phase 90 discoveries:
  - P90 balanced: V1=27.36%/I460bw168=17.98%/I415bw216=29.66%/I600=2.5%/F144=22.5% → 2.021/1.561
  - P90 AVG-max: V1=4.5%/I415bw216=10%/I474bw216=40%/I600=7.5%/F144=38% → 2.304/1.205
  - I600 trend: 10%→7.5%→5%→3.75%→2.5% each improved MIN. Does it keep going?
  - F144=22.5% is optimal floor (22.5% max MIN, below 17.5% = no balanced results)
  - I437_bw216: 2023=0.175 (fails, not useful)
  - Triple-idio balanced weaker than dual+I600=2.5%

Phase 91 agenda:
  A) Load P90 reference signals + profile I420/I425/I430 with bw=216
     - Find the 2023 cliff between lb=415 (2023=1.143) and lb=437 (2023=0.175)
     - I420/I425/I430 are NEW — require fresh backtests
  B) I600 ultra-low sweep (0%, 1.25%, 2.5%=reference)
     - Can we remove I600 entirely and reallocate budget?
  C) AVG-max ultra-fine (0.25% step) around P90 champion
     - Target: AVG > 2.31!
     - Extended I474 range up to 50%
  D) If any A signal beats I415, test it in balanced combo
  E) Ultra-fine 0.3125% around best balanced champion
  F) Pareto summary + champion save
"""

import json, copy, statistics, subprocess, sys
from pathlib import Path
import numpy as np

OUT_DIR      = "artifacts/phase91"
P90_OUT_DIR  = "artifacts/phase90"
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
    path = f"/tmp/phase91_{run_name}_{year}.json"
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

    sharpes = [year_results[y].get("sharpe", 0.0) for y in YEARS]
    avg = round(sum(sharpes) / len(sharpes), 3) if sharpes else 0.0
    mn  = round(min(sharpes), 3) if sharpes else 0.0
    yby = [round(year_results.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]
    year_results["_avg"] = avg
    year_results["_min"] = mn
    year_results["_pos"] = sum(1 for s in sharpes if s > 0)
    print(f"  → AVG={avg}, MIN={mn}", flush=True)
    print(f"  → YbY: {yby}", flush=True)
    return year_results

def load_p90_strategy(p90_label: str) -> dict:
    """Load strategy results from Phase 90 artifacts (avoids re-running identical backtests)."""
    runs_dir = Path(P90_OUT_DIR) / "runs"
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
    year_results["_pos"] = sum(1 for s in sharpes if s > 0)
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

# ─────────────────────────────────────────────────────────────────────────────
# SECTION A: Load P90 signals + run new lb profiles (420, 425, 430 bw=216)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("PHASE 91: I600 Floor + AVG-max Ultra-Fine + lb 420-430 Profiles")
print("=" * 70, flush=True)
print("\nSECTION A: Load P90 reference signals + profile lb=420/425/430 bw=216", flush=True)

# Load P90 reference signals (saves 35 backtest runs)
print("  Loading P90 artifacts for known signals...", flush=True)
v1_data    = load_p90_strategy("p90_v1")
i460_bw168 = load_p90_strategy("p90_i460_bw168_k4")
i415_bw216 = load_p90_strategy("p90_i415_bw216_k4")
i474_bw216 = load_p90_strategy("p90_i474_bw216_k4")
i437_bw216 = load_p90_strategy("p90_i437_bw216_k4")
i600_k2    = load_p90_strategy("p90_i600_k2")
fund_144   = load_p90_strategy("p90_fund144")

# Fallback: run fresh if P90 artifacts not found
if v1_data is None or v1_data.get("_avg", 0) == 0.0:
    print("  P90 artifacts not found — running fresh...", flush=True)
    v1_data    = run_strategy("p91_v1",           "nexus_alpha_v1",        V1_PARAMS)
    i460_bw168 = run_strategy("p91_i460_bw168_k4","idio_momentum_alpha",   make_idio_params(460, 168, k=4))
    i415_bw216 = run_strategy("p91_i415_bw216_k4","idio_momentum_alpha",   make_idio_params(415, 216, k=4))
    i474_bw216 = run_strategy("p91_i474_bw216_k4","idio_momentum_alpha",   make_idio_params(474, 216, k=4))
    i437_bw216 = run_strategy("p91_i437_bw216_k4","idio_momentum_alpha",   make_idio_params(437, 216, k=4))
    i600_k2    = run_strategy("p91_i600_k2",       "idio_momentum_alpha",  make_idio_params(600, 168, k=2))
    fund_144   = run_strategy("p91_fund144",        "funding_momentum_alpha",make_fund_params(144, k=2))
else:
    print("  P90 artifacts loaded successfully!", flush=True)

# Run NEW signals: lb=420, 425, 430 with bw=216
# Goal: find where 2023 drops from I415 (1.143) towards I437 (0.175)
i420_bw216 = run_strategy("p91_i420_bw216_k4", "idio_momentum_alpha", make_idio_params(420, 216, k=4))
i425_bw216 = run_strategy("p91_i425_bw216_k4", "idio_momentum_alpha", make_idio_params(425, 216, k=4))
i430_bw216 = run_strategy("p91_i430_bw216_k4", "idio_momentum_alpha", make_idio_params(430, 216, k=4))

print("\n" + "─"*60)
print("SECTION A SUMMARY:", flush=True)
print(f"  {'Signal':22s}  {'2021':>6}  {'2022':>6}  {'2023':>6}  {'2024':>6}  {'2025':>6}  {'AVG':>6}  {'MIN':>6}", flush=True)
for lbl, data in [
    ("V1",             v1_data),
    ("I415_bw216_k4",  i415_bw216),
    ("I420_bw216_k4",  i420_bw216),
    ("I425_bw216_k4",  i425_bw216),
    ("I430_bw216_k4",  i430_bw216),
    ("I437_bw216_k4",  i437_bw216),
    ("I460_bw168_k4",  i460_bw168),
    ("I474_bw216_k4",  i474_bw216),
    ("I600_k2",        i600_k2),
    ("F144_k2",        fund_144),
]:
    if data is None:
        print(f"  {lbl:22s}  [NOT LOADED]", flush=True)
        continue
    yby = [round(data.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]
    print(f"  {lbl:22s}  {yby[0]:>6.3f}  {yby[1]:>6.3f}  {yby[2]:>6.3f}  {yby[3]:>6.3f}  {yby[4]:>6.3f}  {data['_avg']:>6.3f}  {data['_min']:>6.3f}", flush=True)

print("\n  2023 cliff between lb=415 and lb=437 (bw=216):", flush=True)
for lbl, data in [("I415", i415_bw216), ("I420", i420_bw216), ("I425", i425_bw216), ("I430", i430_bw216), ("I437", i437_bw216)]:
    if data is None:
        continue
    y2023 = round(data.get("2023", {}).get("sharpe", 0.0), 3)
    y2022 = round(data.get("2022", {}).get("sharpe", 0.0), 3)
    print(f"    {lbl}: 2022={y2022}, 2023={y2023}", flush=True)

# Identify best new lb signal
best_new_lb = None
best_new_lb_score = 0.0
for lbl, data, lb in [("I420_bw216", i420_bw216, 420), ("I425_bw216", i425_bw216, 425), ("I430_bw216", i430_bw216, 430)]:
    if data is None:
        continue
    # Bridge score: 2022 + 2023 combined (need both > 1.0 for balanced utility)
    y2022 = data.get("2022", {}).get("sharpe", 0.0)
    y2023 = data.get("2023", {}).get("sharpe", 0.0)
    score = y2022 + y2023
    if score > best_new_lb_score:
        best_new_lb_score = score
        best_new_lb = (lbl, data, lb)

if best_new_lb:
    print(f"\n  Best new lb: {best_new_lb[0]} (bridge={best_new_lb_score:.3f})", flush=True)
    print(f"  Compare I415_bw216 bridge={i415_bw216.get('2022',{}).get('sharpe',0)+i415_bw216.get('2023',{}).get('sharpe',0):.3f}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION B: I600 ultra-low sweep (0%, 1.25%, 2.5%=ref)
# P90 balanced: V1=27.36%/I460bw168=17.98%/I415bw216=29.66%/I600=2.5%/F144=22.5% → 2.021/1.561
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION B: I600 Ultra-Low Sweep (0%, 1.25%, 2.5%=ref)")
print("  P90 balanced: I600=2.5% → MIN=1.561. Does trend continue?")
print("  If 0% still improves: can remove I600 entirely!")
print("═"*70, flush=True)

STRATS_B = ["v1", "i460bw168", "i415bw216", "i600", "f144"]
base_b = {"v1": v1_data, "i460bw168": i460_bw168, "i415bw216": i415_bw216,
          "i600": i600_k2, "f144": fund_144}
ym_b = build_year_matrices(base_b, STRATS_B)

P90_MIN = 1.5614
F144_OPT = 0.225  # confirmed floor from P90

best_b_overall = None
for wi600 in [0.0, 0.0125, 0.025]:
    budget = 1.0 - F144_OPT - wi600  # budget for v1 + i460 + i415
    configs_b = []
    for wv1 in pct_range(1250, 3750, 125):
        for wi460 in pct_range(500, 2250, 125):
            wi415 = budget - wv1 - wi460
            if 0.10 <= wi415 <= 0.35:
                configs_b.append({
                    "v1": wv1, "i460bw168": wi460,
                    "i415bw216": round(wi415, 8),
                    "i600": wi600, "f144": F144_OPT
                })

    n_cfg = len(configs_b)
    results_b = sweep_configs_numpy(configs_b, STRATS_B, ym_b)
    r = best_balanced(results_b)

    if wi600 == 0.0:
        label = "I600=0% (NO I600)"
    else:
        label = f"I600={wi600*100:.2f}%"

    print(f"\n  {label} ({n_cfg} configs, F144={F144_OPT*100:.1f}%)", flush=True)
    if r:
        a, m, w = r
        yby = yby_for_weights(w, STRATS_B, ym_b)
        mark = " ★★ NEW CHAMP!" if m > P90_MIN else " DOMINATES P89!" if m > 1.5614 else " > P89" if m > 1.5463 else ""
        print(f"  Best: AVG={a:.4f}, MIN={m:.4f}{mark}", flush=True)
        print(f"  YbY={yby}", flush=True)
        wfmt = " ".join(f"{k}={v*100:.2f}%" for k, v in sorted(w.items()) if v > 0.001)
        print(f"  Weights: {wfmt}", flush=True)
        if best_b_overall is None or m > best_b_overall[1]:
            best_b_overall = (a, m, w, wi600)
    else:
        print(f"  No balanced result", flush=True)

if best_b_overall:
    a, m, w, wi600 = best_b_overall
    yby = yby_for_weights(w, STRATS_B, ym_b)
    print(f"\n  ★ SECTION B BEST: I600={wi600*100:.2f}% → AVG={a:.4f}, MIN={m:.4f}", flush=True)
    print(f"    YbY={yby}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION C: AVG-max ultra-fine (0.25% step) around P90 champion
# P90: V1=4.5%/I415bw216=10%/I474bw216=40%/I600=7.5%/F144=38% → 2.304/1.205
# Target: AVG > 2.31!
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION C: AVG-max Ultra-Fine (0.25% step) around P90 Champion")
print("  P90: V1=4.5%/I415bw216=10%/I474bw216=40%/I600=7.5%/F144=38% → 2.304/1.205")
print("  Target: AVG > 2.31! Extended I474 range to 50%.")
print("═"*70, flush=True)

STRATS_C = ["v1", "i415bw216", "i474bw216", "i600", "f144"]
base_c = {"v1": v1_data, "i415bw216": i415_bw216, "i474bw216": i474_bw216,
          "i600": i600_k2, "f144": fund_144}
ym_c = build_year_matrices(base_c, STRATS_C)

# Ultra-fine 0.25% step around P90 champion
# V1: 2-8%, I415: 6-14%, I474: 35-50%, I600: {5%, 7.5%, 10%}
# F144 = remainder (must be 25-50%)
configs_c = []
for wv1 in pct_range(200, 800, 25):        # 2% to 8%
    for wi415 in pct_range(600, 1400, 25):  # 6% to 14%
        for wi474 in pct_range(3500, 5000, 25):  # 35% to 50%
            for wi600 in [0.05, 0.075, 0.10]:
                wf144 = 1.0 - wv1 - wi415 - wi474 - wi600
                if 0.25 <= wf144 <= 0.50:
                    configs_c.append({
                        "v1": wv1, "i415bw216": wi415, "i474bw216": wi474,
                        "i600": wi600, "f144": round(wf144, 8)
                    })

print(f"\n  AVG-max ultra-fine grid: {len(configs_c)} configs (0.25% step)", flush=True)
results_c = sweep_configs_numpy(configs_c, STRATS_C, ym_c)

if results_c:
    best_c_avg = max(results_c, key=lambda x: x[0])
    a_ca, m_ca, w_ca = best_c_avg
    yby_ca = yby_for_weights(w_ca, STRATS_C, ym_c)
    wfmt_ca = " ".join(f"{k}={v*100:.2f}%" for k, v in sorted(w_ca.items()) if v > 0.001)
    print(f"  AVG-max best: AVG={a_ca:.4f}, MIN={m_ca:.4f}", flush=True)
    print(f"  YbY={yby_ca}", flush=True)
    print(f"  Weights: {wfmt_ca}", flush=True)
    if a_ca > 2.3038:
        print(f"  ★★ NEW AVG-MAX RECORD! (AVG {a_ca:.4f} > P90's 2.3038)", flush=True)
    elif a_ca > 2.286:
        print(f"  ★ Beats P89 AVG-max (2.286)", flush=True)

    # Also report best balanced in C
    bal_c = [(a, m, w) for a, m, w in results_c if a >= 2.000]
    if bal_c:
        best_c_bal = max(bal_c, key=lambda x: (x[1], x[0]))
        a_cb, m_cb, w_cb = best_c_bal
        yby_cb = yby_for_weights(w_cb, STRATS_C, ym_c)
        print(f"  Best balanced in C: AVG={a_cb:.4f}, MIN={m_cb:.4f}, YbY={yby_cb}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION D: Test best new lb signal in balanced combo (if bridge > I415)
# I415 bridge = 2022+2023 = 1.737+1.143 = 2.880
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION D: Best New lb Signal in Balanced Combo")
print("  Testing I420/I425/I430 as replacement for I415_bw216")
print("  I415_bw216 bridge=2.880 (2022=1.737, 2023=1.143) — hard to beat")
print("═"*70, flush=True)

i415_bridge = 1.737 + 1.143  # 2.880
best_d_overall = None

for new_lbl, new_data, new_lb in [
    ("I420_bw216", i420_bw216, 420),
    ("I425_bw216", i425_bw216, 425),
    ("I430_bw216", i430_bw216, 430),
]:
    if new_data is None:
        continue
    y2022 = new_data.get("2022", {}).get("sharpe", 0.0)
    y2023 = new_data.get("2023", {}).get("sharpe", 0.0)
    bridge = y2022 + y2023
    min_yr = new_data.get("_min", 0.0)

    print(f"\n  {new_lbl}: bridge={bridge:.3f} (2022={y2022:.3f}, 2023={y2023:.3f}), MIN={min_yr:.3f}", flush=True)
    if bridge < i415_bridge - 0.05:
        print(f"  → Bridge too weak vs I415 ({i415_bridge:.3f}), skipping balanced test", flush=True)
        continue
    if min_yr < 0.5:
        print(f"  → MIN too low ({min_yr:.3f}), skipping balanced test", flush=True)
        continue

    # Test new lb as replacement for I415
    strats_d = ["v1", "i460bw168", f"i{new_lb}bw216", "i600", "f144"]
    base_d = {"v1": v1_data, "i460bw168": i460_bw168, f"i{new_lb}bw216": new_data,
              "i600": i600_k2, "f144": fund_144}
    ym_d = build_year_matrices(base_d, strats_d)

    configs_d = []
    for wi600 in [0.0125, 0.025]:
        budget = 1.0 - F144_OPT - wi600
        for wv1 in pct_range(1250, 3750, 125):
            for wi460 in pct_range(500, 2250, 125):
                wi_new = budget - wv1 - wi460
                if 0.10 <= wi_new <= 0.35:
                    cfg = {"v1": wv1, "i460bw168": wi460, f"i{new_lb}bw216": round(wi_new, 8),
                           "i600": wi600, "f144": F144_OPT}
                    configs_d.append(cfg)

    print(f"  Balanced sweep: {len(configs_d)} configs", flush=True)
    results_d = sweep_configs_numpy(configs_d, strats_d, ym_d)
    r = best_balanced(results_d)
    if r:
        a, m, w = r
        yby = yby_for_weights(w, strats_d, ym_d)
        mark = " ★★ BEATS P90!" if m > P90_MIN else " > P89" if m > 1.5463 else ""
        print(f"  Best: AVG={a:.4f}, MIN={m:.4f}{mark}", flush=True)
        print(f"  YbY={yby}", flush=True)
        wfmt = " ".join(f"{k}={v*100:.2f}%" for k, v in sorted(w.items()) if v > 0.001)
        print(f"  Weights: {wfmt}", flush=True)
        if best_d_overall is None or m > best_d_overall[0][1]:
            best_d_overall = ((a, m, w), strats_d, ym_d, new_lbl)
    else:
        print(f"  No balanced config (AVG>=2.000) found", flush=True)

# Also test combining new lb WITH I415 (both in ensemble)
print("\n  D-combo: Best new lb combined WITH I415_bw216 in balanced:", flush=True)
best_d_combo = None
for new_lbl, new_data, new_lb in [
    ("I420_bw216", i420_bw216, 420),
    ("I425_bw216", i425_bw216, 425),
    ("I430_bw216", i430_bw216, 430),
]:
    if new_data is None:
        continue
    if new_data.get("_min", 0.0) < 0.5:
        continue

    strats_dc = ["v1", "i460bw168", "i415bw216", f"i{new_lb}bw216", "i600", "f144"]
    base_dc = {
        "v1": v1_data, "i460bw168": i460_bw168, "i415bw216": i415_bw216,
        f"i{new_lb}bw216": new_data, "i600": i600_k2, "f144": fund_144
    }
    ym_dc = build_year_matrices(base_dc, strats_dc)

    configs_dc = []
    for wi600 in [0.0125, 0.025]:
        budget = 1.0 - F144_OPT - wi600
        for wv1 in pct_range(1250, 3500, 125):
            for wi460 in pct_range(500, 2000, 125):
                for wi415 in pct_range(1000, 3000, 125):
                    wi_new = budget - wv1 - wi460 - wi415
                    if 0.025 <= wi_new <= 0.15:
                        cfg = {
                            "v1": wv1, "i460bw168": wi460,
                            "i415bw216": wi415, f"i{new_lb}bw216": round(wi_new, 8),
                            "i600": wi600, "f144": F144_OPT
                        }
                        configs_dc.append(cfg)

    if not configs_dc:
        continue

    results_dc = sweep_configs_numpy(configs_dc, strats_dc, ym_dc)
    r = best_balanced(results_dc)
    if r:
        a, m, w = r
        yby = yby_for_weights(w, strats_dc, ym_dc)
        mark = " ★★ BEATS P90!" if m > P90_MIN else " > P89" if m > 1.5463 else ""
        print(f"  {new_lbl}+I415 combo: AVG={a:.4f}, MIN={m:.4f}{mark}", flush=True)
        print(f"  YbY={yby}", flush=True)
        wfmt = " ".join(f"{k}={v*100:.2f}%" for k, v in sorted(w.items()) if v > 0.001)
        print(f"  Weights: {wfmt}", flush=True)
        if best_d_combo is None or m > best_d_combo[0][1]:
            best_d_combo = ((a, m, w), strats_dc, ym_dc, f"{new_lbl}+I415")
    else:
        print(f"  {new_lbl}+I415: no balanced config found", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION E: Ultra-fine 0.3125% around best balanced champion
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION E: Ultra-fine 0.3125% Grid around Best Balanced Champion")
print("═"*70, flush=True)

# Collect all balanced candidates
balanced_cands = []
if best_b_overall:
    a, m, w, wi600 = best_b_overall
    balanced_cands.append((a, m, w, STRATS_B, ym_b, "B"))
if best_d_overall:
    (a, m, w), strats, ym, lbl = best_d_overall
    balanced_cands.append((a, m, w, strats, ym, f"D-{lbl}"))
if best_d_combo:
    (a, m, w), strats, ym, lbl = best_d_combo
    balanced_cands.append((a, m, w, strats, ym, f"D-combo-{lbl}"))

ultra_champ = None
if balanced_cands:
    best_cand = max(balanced_cands, key=lambda x: (x[1], x[0]))
    a_bc, m_bc, w_bc, strats_bc, ym_bc, lbl_bc = best_cand

    print(f"\n  Starting from: {lbl_bc} AVG={a_bc:.4f}/MIN={m_bc:.4f}", flush=True)
    yby_bc = yby_for_weights(w_bc, strats_bc, ym_bc)
    print(f"  YbY={yby_bc}", flush=True)

    # Fix i600 and f144, sweep the other signals with ±2% range
    fixed_keys = {k: v for k, v in w_bc.items() if k in ("i600", "f144")}
    moveable = [(k, v) for k, v in w_bc.items() if k not in ("i600", "f144")]
    fixed_sum = sum(fixed_keys.values())

    configs_e = []
    # 3-way moveable case (v1, i460bw168, i415bw216)
    if len(moveable) == 3:
        n1, w1c = moveable[0]
        n2, w2c = moveable[1]
        lo1 = max(int(w1c*10000) - 200, 500)
        hi1 = min(int(w1c*10000) + 200, 4000)
        lo2 = max(int(w2c*10000) - 200, 250)
        hi2 = min(int(w2c*10000) + 200, 2500)
        n3 = moveable[2][0]
        for wv in pct_range(lo1, hi1, 31):
            for wa in pct_range(lo2, hi2, 31):
                wb = 1.0 - wv - wa - fixed_sum
                if 0.10 <= wb <= 0.40:
                    configs_e.append({n1: wv, n2: wa, n3: round(wb, 8), **fixed_keys})
    elif len(moveable) == 2:
        n1, w1c = moveable[0]
        n2 = moveable[1][0]
        lo1 = max(int(w1c*10000) - 200, 500)
        hi1 = min(int(w1c*10000) + 200, 4000)
        for wv in pct_range(lo1, hi1, 31):
            wb = 1.0 - wv - fixed_sum
            if 0.10 <= wb <= 0.40:
                configs_e.append({n1: wv, n2: round(wb, 8), **fixed_keys})

    print(f"\n  Ultra-fine grid: {len(configs_e)} configs (0.3125% step)", flush=True)
    if configs_e:
        results_e = sweep_configs_numpy(configs_e, strats_bc, ym_bc)
        bal_e = [(a, m, w) for a, m, w in results_e if a >= 2.000]
        if bal_e:
            best_e = max(bal_e, key=lambda x: (x[1], x[0]))
            a_e, m_e, w_e = best_e
            improved = m_e > m_bc
            yby_e = yby_for_weights(w_e, strats_bc, ym_bc)
            mark = "IMPROVED!" if improved else "converged"
            print(f"  Result: AVG={a_e:.4f}, MIN={m_e:.4f} ({mark})", flush=True)
            print(f"  YbY={yby_e}", flush=True)
            if improved:
                a_bc, m_bc, w_bc = a_e, m_e, w_e
                lbl_bc = lbl_bc + "_ultrafine"

    ultra_champ = (a_bc, m_bc, w_bc, strats_bc, ym_bc, lbl_bc)
else:
    print("  No balanced candidates to refine.", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION F: Pareto summary + champion save
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("SECTION F: Pareto Summary + Champion Save")
print("═"*70, flush=True)

if ultra_champ:
    a_ch, m_ch, w_ch, strats_ch, ym_ch, lbl_ch = ultra_champ
    yby_ch = yby_for_weights(w_ch, strats_ch, ym_ch)
    wfmt = " ".join(f"{k}={v*100:.2f}%" for k, v in sorted(w_ch.items()) if v > 0.001)

    print(f"\n★★★ PHASE 91 BALANCED CHAMPION [{lbl_ch}]:", flush=True)
    print(f"    AVG={a_ch:.4f}, MIN={m_ch:.4f}", flush=True)
    print(f"    YbY={yby_ch}", flush=True)
    print(f"    Weights: {wfmt}", flush=True)

    if m_ch > P90_MIN:
        print(f"    ★★ STRICTLY DOMINATES P90 (MIN {m_ch:.4f} > {P90_MIN:.4f})!", flush=True)
    elif m_ch > 1.5463:
        print(f"    ★ Dominates P89 (MIN {m_ch:.4f} > 1.5463)", flush=True)
    elif a_ch >= 2.000:
        print(f"    Balanced (AVG>=2.000)", flush=True)

    champion_config = {
        "phase": 91,
        "section": lbl_ch,
        "avg_sharpe": round(a_ch, 4),
        "min_sharpe": round(m_ch, 4),
        "yby_sharpes": {y: round(v, 3) for y, v in zip(YEARS, yby_ch)},
        "weights": {k: round(v, 6) for k, v in w_ch.items() if v > 0.001},
        "signal_config": {
            "strat_names": strats_ch,
            "notes": {
                "v1":            "nexus_alpha_v1 k=2",
                "i460bw168":     "idio_momentum_alpha lb=460 bw=168 k=4",
                "i415bw216":     "idio_momentum_alpha lb=415 bw=216 k=4",
                "i420bw216":     "idio_momentum_alpha lb=420 bw=216 k=4",
                "i425bw216":     "idio_momentum_alpha lb=425 bw=216 k=4",
                "i430bw216":     "idio_momentum_alpha lb=430 bw=216 k=4",
                "i600":          "idio_momentum_alpha lb=600 bw=168 k=2",
                "f144":          "funding_momentum_alpha lb=144 k=2",
            }
        }
    }
    out_path = "configs/ensemble_p91_balanced.json"
    with open(out_path, "w") as f:
        json.dump(champion_config, f, indent=2)
    print(f"\n    Saved: {out_path}", flush=True)

# Save AVG-max if improved
if results_c:
    best_c_avgmax = max(results_c, key=lambda x: x[0])
    a_cm, m_cm, w_cm = best_c_avgmax
    if a_cm > 2.3038:
        avgmax_config = {
            "phase": 91,
            "section": "C:AVGmax-ultrafine",
            "type": "avg_max",
            "avg_sharpe": round(a_cm, 4),
            "min_sharpe": round(m_cm, 4),
            "weights": {k: round(v, 6) for k, v in w_cm.items() if v > 0.001},
        }
        with open("configs/ensemble_p91_avgmax.json", "w") as f:
            json.dump(avgmax_config, f, indent=2)
        print(f"  ★★ NEW AVG-MAX CONFIG SAVED: AVG={a_cm:.4f}/MIN={m_cm:.4f}", flush=True)

# Pareto frontier
print("\nFINAL PARETO FRONTIER:", flush=True)
all_points = [
    (2.3038, 1.2046, "P90-AVG-MAX"),
    (2.0211, 1.5614, "P90-BALANCED"),
]
if ultra_champ:
    a_ch, m_ch = ultra_champ[0], ultra_champ[1]
    all_points.append((a_ch, m_ch, f"P91-{ultra_champ[5]}"))
if results_c:
    a_cm, m_cm, _ = max(results_c, key=lambda x: x[0])
    all_points.append((a_cm, m_cm, "P91-C-AVGmax"))

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

print("\nPHASE 91 COMPLETE", flush=True)
