#!/usr/bin/env python3
"""
Phase 73: Fine Grid Around bw168×168 + AVG/MIN Frontier Mapping

Phase 72 findings:
  - AVG-max champion: V1(10)+I437_bw336(20)+I600_bw168(20)+F144(35)+F168(15)
    AVG=1.933, MIN=0.894 (2023 weak at 0.894)
  - Balanced champion: I437_bw168 + I600_bw168 (Phase 71 weights)
    AVG=1.904, MIN=1.151
  - Phase 71 baseline: AVG=1.861, MIN=1.097

Goal: Find the optimal point on the AVG/MIN frontier
  - Ideally: AVG > 1.900 AND MIN > 1.000 simultaneously
  - Use the I437_bw168 + I600_bw168 combination (best MIN combo)
  - Do fine 5% grid search over all weight combinations

Also explore:
  - Three-point idio: I437_bw168 + I600_bw168 + I730_bw168 (annual momentum)
  - Or: I437_bw336 + I600_bw168 with MIN-prioritizing weights
"""

import json
import math
import statistics
import subprocess
import sys
import copy
from pathlib import Path

OUT_DIR = "artifacts/phase73"
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

FUND_144_PARAMS = {
    "k_per_side": 2, "funding_lookback_bars": 144, "direction": "contrarian",
    "vol_lookback_bars": 168, "target_gross_leverage": 0.25, "rebalance_interval_bars": 24,
}

FUND_168_PARAMS = {
    "k_per_side": 2, "funding_lookback_bars": 168, "direction": "contrarian",
    "vol_lookback_bars": 168, "target_gross_leverage": 0.25, "rebalance_interval_bars": 24,
}

def make_idio_params(lookback: int, beta_window: int) -> dict:
    return {
        "k_per_side": 2, "lookback_bars": lookback, "beta_window_bars": beta_window,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 48,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    path = f"/tmp/phase73_{run_name}_{year}.json"
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


def get_all_returns(data: dict) -> list:
    rets = []
    for year in YEARS:
        rets.extend(data.get(year, {}).get("returns", []))
    return rets


def metrics_from_returns(rets: list) -> float:
    if not rets or len(rets) < 10:
        return 0.0
    mu = statistics.mean(rets)
    sd = statistics.pstdev(rets)
    return round((mu / sd) * math.sqrt(8760), 3) if sd > 0 else 0.0


def blend_ensemble(strategies: dict, weights: dict) -> dict:
    year_sharpes = {}
    for year in YEARS:
        pairs = []
        min_len = None
        valid = True
        for name, data in strategies.items():
            w = weights.get(name, 0.0)
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
        year_sharpes[year] = metrics_from_returns(blended)

    valid_s = [s for s in year_sharpes.values() if s is not None]
    return {
        "avg": round(sum(valid_s) / len(valid_s), 3) if valid_s else 0.0,
        "min": round(min(valid_s), 3) if valid_s else 0.0,
        "pos": sum(1 for s in valid_s if s > 0),
        "yby": [year_sharpes.get(y) for y in YEARS],
    }


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
print("PHASE 73: Fine Grid Around bw168×168 + Frontier Mapping")
print("=" * 70)

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# ─── SECTION A: Run idio variants ────────────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION A: Running idio variants")
print("Phase 72: bw168×168 had best MIN (1.151), bw336×168 had best AVG (1.927)")
print("═" * 70)

print("  Running idio_437_bw168...", flush=True)
i437_bw168_data = run_strategy("idio_lb437_bw168", "idio_momentum_alpha", make_idio_params(437, 168))
print("  Running idio_437_bw336...", flush=True)
i437_bw336_data = run_strategy("idio_lb437_bw336", "idio_momentum_alpha", make_idio_params(437, 336))
print("  Running idio_600_bw168...", flush=True)
i600_bw168_data = run_strategy("idio_lb600_bw168", "idio_momentum_alpha", make_idio_params(600, 168))
print("  Running idio_730_bw168...", flush=True)
i730_bw168_data = run_strategy("idio_lb730_bw168", "idio_momentum_alpha", make_idio_params(730, 168))

# ─── SECTION B: Reference strategies ─────────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION B: Running reference strategies")
print("═" * 70)
print("  Running V1 reference...", flush=True)
v1_data = run_strategy("v1_ref", "nexus_alpha_v1", V1_PARAMS)
print("  Running fund_144 reference...", flush=True)
f144_data = run_strategy("fund_144_ref", "funding_momentum_alpha", FUND_144_PARAMS)
print("  Running fund_168 reference...", flush=True)
f168_data = run_strategy("fund_168_ref", "funding_momentum_alpha", FUND_168_PARAMS)

# ─── SECTION C: Fine grid around bw168×168 (best MIN combo) ──────────────────
print("\n" + "═" * 70)
print("SECTION C: Fine 5% grid around I437_bw168 + I600_bw168")
print("Target: AVG > 1.900 AND MIN > 1.000")
print("═" * 70)

# Baseline check
strats_168_168 = {
    "v1": v1_data, "idio_437": i437_bw168_data, "idio_600": i600_bw168_data,
    "f144": f144_data, "f168": f168_data,
}
p72_base = blend_ensemble(strats_168_168, {"v1": 0.15, "idio_437": 0.21, "idio_600": 0.14, "f144": 0.40, "f168": 0.10})
print(f"\n  Phase 72 baseline (bw168×168, Phase 71 weights): AVG={p72_base['avg']}, MIN={p72_base['min']}")
print(f"  YbY: {p72_base['yby']}")

grid_168_168 = []
for w_v1 in range(5, 26, 5):
    for w_i437 in range(10, 36, 5):
        for w_i600 in range(0, 31, 5):
            if w_i600 == 0:
                continue  # need at least some idio_600
            for w_f144 in range(25, 51, 5):
                w_f168 = 100 - w_v1 - w_i437 - w_i600 - w_f144
                if w_f168 < 5 or w_f168 > 25:
                    continue
                total_idio = w_i437 + w_i600
                if total_idio < 20 or total_idio > 45:
                    continue
                wts = {
                    "v1": w_v1/100, "idio_437": w_i437/100, "idio_600": w_i600/100,
                    "f144": w_f144/100, "f168": w_f168/100
                }
                r = blend_ensemble(strats_168_168, wts)
                grid_168_168.append((wts, r))

grid_168_168.sort(key=lambda x: x[1]["avg"], reverse=True)
print(f"\n  Total configurations tested: {len(grid_168_168)}")
print(f"\n  Top 10 by AVG:")
print(f"  {'V1':>4} {'I437':>6} {'I600':>6} {'F144':>6} {'F168':>6} {'AVG':>8} {'MIN':>8} {'YbY'}")
print("  " + "─" * 80)
for wts, r in grid_168_168[:10]:
    w_v1 = int(wts["v1"] * 100)
    w_i437 = int(wts["idio_437"] * 100)
    w_i600 = int(wts["idio_600"] * 100)
    w_f144 = int(wts["f144"] * 100)
    w_f168 = int(wts["f168"] * 100)
    print(f"  {w_v1:>4} {w_i437:>6} {w_i600:>6} {w_f144:>6} {w_f168:>6} {r['avg']:>8.3f} {r['min']:>8.3f}  {r['yby']}")

# Find Pareto-optimal: maximize min given avg > threshold
print(f"\n  Best MIN-first (AVG > 1.900 constraint):")
good_avg = [(wts, r) for wts, r in grid_168_168 if r["avg"] > 1.900]
good_avg.sort(key=lambda x: x[1]["min"], reverse=True)
for wts, r in good_avg[:5]:
    w_v1 = int(wts["v1"] * 100)
    w_i437 = int(wts["idio_437"] * 100)
    w_i600 = int(wts["idio_600"] * 100)
    w_f144 = int(wts["f144"] * 100)
    w_f168 = int(wts["f168"] * 100)
    print(f"    V1={w_v1},I437_bw168={w_i437},I600_bw168={w_i600},F144={w_f144},F168={w_f168}: AVG={r['avg']}, MIN={r['min']}")

best_avg_config = grid_168_168[0]
best_min_config = good_avg[0] if good_avg else grid_168_168[0]

# ─── SECTION D: Mix bw336 and bw168 in fine grid ─────────────────────────────
print("\n" + "═" * 70)
print("SECTION D: Fine 5% grid with I437_bw336 + I600_bw168 (best AVG combo)")
print("═" * 70)

strats_336_168 = {
    "v1": v1_data, "idio_437": i437_bw336_data, "idio_600": i600_bw168_data,
    "f144": f144_data, "f168": f168_data,
}

grid_336_168 = []
for w_v1 in range(5, 26, 5):
    for w_i437 in range(10, 36, 5):
        for w_i600 in range(5, 31, 5):
            for w_f144 in range(25, 51, 5):
                w_f168 = 100 - w_v1 - w_i437 - w_i600 - w_f144
                if w_f168 < 5 or w_f168 > 25:
                    continue
                total_idio = w_i437 + w_i600
                if total_idio < 20 or total_idio > 45:
                    continue
                wts = {
                    "v1": w_v1/100, "idio_437": w_i437/100, "idio_600": w_i600/100,
                    "f144": w_f144/100, "f168": w_f168/100
                }
                r = blend_ensemble(strats_336_168, wts)
                grid_336_168.append((wts, r))

grid_336_168.sort(key=lambda x: x[1]["avg"], reverse=True)
print(f"\n  Top 10 by AVG (I437_bw336 + I600_bw168):")
for wts, r in grid_336_168[:10]:
    w_v1 = int(wts["v1"] * 100)
    w_i437 = int(wts["idio_437"] * 100)
    w_i600 = int(wts["idio_600"] * 100)
    w_f144 = int(wts["f144"] * 100)
    w_f168 = int(wts["f168"] * 100)
    print(f"  V1={w_v1},I437_bw336={w_i437},I600_bw168={w_i600},F144={w_f144},F168={w_f168}: AVG={r['avg']}, MIN={r['min']}")

good_336 = [(wts, r) for wts, r in grid_336_168 if r["avg"] > 1.900]
good_336.sort(key=lambda x: x[1]["min"], reverse=True)
print(f"\n  Best MIN-first (AVG > 1.900, I437_bw336+I600_bw168):")
for wts, r in good_336[:5]:
    w_v1 = int(wts["v1"] * 100)
    w_i437 = int(wts["idio_437"] * 100)
    w_i600 = int(wts["idio_600"] * 100)
    w_f144 = int(wts["f144"] * 100)
    w_f168 = int(wts["f168"] * 100)
    print(f"    V1={w_v1},I437_bw336={w_i437},I600_bw168={w_i600},F144={w_f144},F168={w_f168}: AVG={r['avg']}, MIN={r['min']}")

# ─── SECTION E: Triple-idio experiment ───────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION E: Triple-idio (I437_bw168 + I600_bw168 + I730_bw168)")
print("idio_730 YbY from Phase 70: [1.746, 1.282, 0.173, 0.12, 0.612] AVG=0.787")
print("Note: 730 has high 2021-2022 but weak 2023-2024 — complementary to 437")
print("═" * 70)

strats_triple = {
    "v1": v1_data, "idio_437": i437_bw168_data, "idio_600": i600_bw168_data,
    "idio_730": i730_bw168_data, "f144": f144_data, "f168": f168_data,
}

triple_results = []
for w_v1 in [10, 15]:
    for w_i437 in range(15, 31, 5):
        for w_i600 in range(5, 21, 5):
            for w_i730 in range(5, 16, 5):
                for w_f144 in range(25, 46, 5):
                    w_f168 = 100 - w_v1 - w_i437 - w_i600 - w_i730 - w_f144
                    if w_f168 < 5 or w_f168 > 20:
                        continue
                    total_idio = w_i437 + w_i600 + w_i730
                    if total_idio < 25 or total_idio > 45:
                        continue
                    wts = {
                        "v1": w_v1/100, "idio_437": w_i437/100, "idio_600": w_i600/100,
                        "idio_730": w_i730/100, "f144": w_f144/100, "f168": w_f168/100
                    }
                    r = blend_ensemble(strats_triple, wts)
                    triple_results.append((wts, r))

triple_results.sort(key=lambda x: x[1]["avg"], reverse=True)
print(f"\n  Top 5 triple-idio configs:")
for wts, r in triple_results[:5]:
    w_v1 = int(wts["v1"] * 100)
    w_i437 = int(wts["idio_437"] * 100)
    w_i600 = int(wts["idio_600"] * 100)
    w_i730 = int(wts["idio_730"] * 100)
    w_f144 = int(wts["f144"] * 100)
    w_f168 = int(wts["f168"] * 100)
    print(f"    V1={w_v1},I437={w_i437},I600={w_i600},I730={w_i730},F144={w_f144},F168={w_f168}: AVG={r['avg']}, MIN={r['min']}")

# ─── SECTION F: Grand Summary ─────────────────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION F: Phase 73 Summary + Champion Candidates")
print("═" * 70)

PREV_CHAMP_AVG = 1.933
PREV_CHAMP_MIN = 0.894

all_candidates = []
# From phase 72
all_candidates.append(("Phase 72 AVG-max (bw336×168)", {"avg": 1.933, "min": 0.894, "yby": [3.443, 1.158, 0.894, 2.673, 1.496]},
                       "V1(10)+I437_bw336(20)+I600_bw168(20)+F144(35)+F168(15)"))
all_candidates.append(("Phase 71 balanced", {"avg": 1.861, "min": 1.097, "yby": [3.429, 1.24, 1.097, 2.389, 1.151]},
                       "V1(15)+I437_bw72(21)+I600_bw72(14)+F144(40)+F168(10)"))

# Best from 168×168 grid
if grid_168_168:
    wts_b, r_b = grid_168_168[0]
    all_candidates.append(("Best 168×168 grid (AVG-max)", {"avg": r_b["avg"], "min": r_b["min"], "yby": r_b["yby"]},
                           f"V1({int(wts_b['v1']*100)})+I437_bw168({int(wts_b['idio_437']*100)})+I600_bw168({int(wts_b['idio_600']*100)})+F144({int(wts_b['f144']*100)})+F168({int(wts_b['f168']*100)})"))
    if good_avg:
        wts_m, r_m = good_avg[0]
        all_candidates.append(("Best 168×168 MIN-opt (AVG>1.900)", {"avg": r_m["avg"], "min": r_m["min"], "yby": r_m["yby"]},
                               f"V1({int(wts_m['v1']*100)})+I437_bw168({int(wts_m['idio_437']*100)})+I600_bw168({int(wts_m['idio_600']*100)})+F144({int(wts_m['f144']*100)})+F168({int(wts_m['f168']*100)})"))

# Best from 336×168 grid
if grid_336_168:
    wts_g2, r_g2 = grid_336_168[0]
    all_candidates.append(("Best 336×168 grid (AVG-max)", {"avg": r_g2["avg"], "min": r_g2["min"], "yby": r_g2["yby"]},
                           f"V1({int(wts_g2['v1']*100)})+I437_bw336({int(wts_g2['idio_437']*100)})+I600_bw168({int(wts_g2['idio_600']*100)})+F144({int(wts_g2['f144']*100)})+F168({int(wts_g2['f168']*100)})"))
    if good_336:
        wts_g2m, r_g2m = good_336[0]
        all_candidates.append(("Best 336×168 MIN-opt (AVG>1.900)", {"avg": r_g2m["avg"], "min": r_g2m["min"], "yby": r_g2m["yby"]},
                               f"V1({int(wts_g2m['v1']*100)})+I437_bw336({int(wts_g2m['idio_437']*100)})+I600_bw168({int(wts_g2m['idio_600']*100)})+F144({int(wts_g2m['f144']*100)})+F168({int(wts_g2m['f168']*100)})"))

# Best triple-idio
if triple_results:
    wts_t, r_t = triple_results[0]
    all_candidates.append(("Best triple-idio", {"avg": r_t["avg"], "min": r_t["min"], "yby": r_t["yby"]},
                           f"triple-idio {int(wts_t['v1']*100)}+{int(wts_t['idio_437']*100)}+{int(wts_t['idio_600']*100)}+{int(wts_t['idio_730']*100)}+..."))

all_candidates.sort(key=lambda x: x[1]["avg"], reverse=True)
print(f"\n  {'Configuration':45} {'AVG':>8} {'MIN':>8} {'YbY'}")
print("  " + "─" * 100)
for desc, r, detail in all_candidates:
    print(f"  {desc:45} {r['avg']:>8.3f} {r['min']:>8.3f}  {r['yby']}")
    print(f"    → {detail}")

# Save new champions
best_by_avg = max(all_candidates, key=lambda x: x[1]["avg"])
best_by_min = max([(d, r, det) for d, r, det in all_candidates if r["avg"] > 1.900],
                  key=lambda x: x[1]["min"], default=(None, None, None))

if best_by_avg[1]["avg"] > PREV_CHAMP_AVG:
    print(f"\n  ★ NEW AVG CHAMPION: {best_by_avg[0]}, AVG={best_by_avg[1]['avg']}")

if best_by_min[1] and (best_by_min[1]["min"] > 1.000 and best_by_min[1]["avg"] > 1.900):
    print(f"\n  ★ NEW BALANCED CHAMPION: {best_by_min[0]}")
    print(f"    AVG={best_by_min[1]['avg']}, MIN={best_by_min[1]['min']}")
    # Save balanced champion
    desc_b, r_b, detail_b = best_by_min
    # Choose the right strats and weights
    if "168×168" in desc_b or "168_bw168" in desc_b:
        strats_save = strats_168_168
        wts_save = good_avg[0][0] if good_avg else best_min_config[0]
        bw_str = "bw168x168"
        i437_params = make_idio_params(437, 168)
        i600_params = make_idio_params(600, 168)
    else:
        strats_save = strats_336_168
        wts_save = good_336[0][0] if good_336 else {}
        bw_str = "bw336x168"
        i437_params = make_idio_params(437, 336)
        i600_params = make_idio_params(600, 168)

    config_name = f"ensemble_dual_idio_p73_{bw_str}_balanced"
    comment = (f"Phase 73 balanced champion: {detail_b}."
               f" AVG={r_b['avg']}, MIN={r_b['min']}. Best AVG+MIN combination.")
    sub_strats = [
        {"name": "nexus_alpha_v1", "params": V1_PARAMS, "weight": wts_save.get("v1", 0.15)},
        {"name": "idio_momentum_alpha", "params": i437_params, "weight": wts_save.get("idio_437", 0.20)},
        {"name": "idio_momentum_alpha", "params": i600_params, "weight": wts_save.get("idio_600", 0.15)},
        {"name": "funding_momentum_alpha", "params": FUND_144_PARAMS, "weight": wts_save.get("f144", 0.40)},
        {"name": "funding_momentum_alpha", "params": FUND_168_PARAMS, "weight": wts_save.get("f168", 0.10)},
    ]
    save_champion(config_name, comment, sub_strats)

# Save AVG-max grid config if new best
if grid_168_168 and grid_168_168[0][1]["avg"] > PREV_CHAMP_AVG:
    wts_new, r_new = grid_168_168[0]
    config_name2 = f"ensemble_dual_idio_p73_bw168x168_avgmax"
    comment2 = (f"Phase 73 AVG-max: dual-idio_bw168. "
                f"V1({int(wts_new['v1']*100)})+I437_bw168({int(wts_new['idio_437']*100)})+I600_bw168({int(wts_new['idio_600']*100)})"
                f"+F144({int(wts_new['f144']*100)})+F168({int(wts_new['f168']*100)}). AVG={r_new['avg']}, MIN={r_new['min']}")
    sub_strats2 = [
        {"name": "nexus_alpha_v1", "params": V1_PARAMS, "weight": wts_new["v1"]},
        {"name": "idio_momentum_alpha", "params": make_idio_params(437, 168), "weight": wts_new["idio_437"]},
        {"name": "idio_momentum_alpha", "params": make_idio_params(600, 168), "weight": wts_new["idio_600"]},
        {"name": "funding_momentum_alpha", "params": FUND_144_PARAMS, "weight": wts_new["f144"]},
        {"name": "funding_momentum_alpha", "params": FUND_168_PARAMS, "weight": wts_new["f168"]},
    ]
    save_champion(config_name2, comment2, sub_strats2)

print("\n" + "=" * 70)
print("PHASE 73 COMPLETE")
print("=" * 70)
