#!/usr/bin/env python3
"""
Phase 74: 2.5% Step Fine Grid + Idio Lookback Gap (437-600) Search

Phase 73 balanced champion: V1(15)+I437_bw168(15)+I600_bw168(20)+F144(45)+F168(5)
  AVG=1.909, MIN=1.170, 5/5

Phase 72 AVG-max: V1(10)+I437_bw336(20)+I600_bw168(20)+F144(35)+F168(15)
  AVG=1.933, MIN=0.894

Goals:
  A. Fine 2.5% grid around Phase 73 champion (I437_bw168 + I600_bw168)
     → Push AVG/MIN frontier beyond 1.910/1.170
  B. Idio lookback gap search: lb=[500, 550] with bw=168
     → Is there a better second-idio between 437 and 600?
  C. Best second idio in ensemble: replace I600_bw168 with I500/I550
  D. V1 parameter sensitivity: what if we tweak V1's internal weights?
     (already optimized, probably not worth it — focus on A/B/C)
"""

import json
import math
import statistics
import subprocess
import sys
import copy
from pathlib import Path

OUT_DIR = "artifacts/phase74"
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
    path = f"/tmp/phase74_{run_name}_{year}.json"
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
print("PHASE 74: 2.5% Grid + Idio Lookback Gap Search (437-600h)")
print("=" * 70)

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# Phase 73 champion for reference
P73_CHAMP = {"avg": 1.909, "min": 1.170, "yby": [3.464, 1.175, 1.170, 2.471, 1.267]}
P72_CHAMP_AVG = {"avg": 1.933, "min": 0.894}

# ─── SECTION A: Run idio lookback gap variants ────────────────────────────────
print("\n" + "═" * 70)
print("SECTION A: Idio lookback gap search (437-600h) with bw=168")
print("Known: 437→1.222, 600→1.235 (both with bw=168). What's between?")
print("═" * 70)

# Test lookbacks between 437 and 600, all with bw=168
GAP_LOOKBACKS = [475, 510, 550]
gap_results = {}

for lb in GAP_LOOKBACKS:
    label = f"idio_lb{lb}_bw168"
    data = run_strategy(label, "idio_momentum_alpha", make_idio_params(lb, 168))
    gap_results[lb] = data

# Known results for comparison
print("\n── Idio lookback curve (bw=168) ────────────────────────────────────────")
known = {437: (1.222, [1.612, 0.61, 0.824, 1.447, 1.619]),
         600: (1.235, [1.913, 0.872, 0.417, 1.466, 1.507])}
print(f"{'LB':>6} {'AVG':>8} {'YbY (2021-2025)'}")
print("─" * 65)
for lb in sorted(list(known.keys()) + GAP_LOOKBACKS):
    if lb in known:
        avg, yby = known[lb]
        print(f"{lb:>6}h {avg:>8.3f}  {yby}  (known)")
    else:
        d = gap_results[lb]
        yby = [round(d.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]
        print(f"{lb:>6}h {d['_avg']:>8.3f}  {yby}")

best_gap_lb = max(GAP_LOOKBACKS, key=lambda lb: gap_results[lb]["_avg"])
best_gap_avg = gap_results[best_gap_lb]["_avg"]
print(f"\nBest gap lookback: lb={best_gap_lb}h, AVG={best_gap_avg:.3f}")

# ─── SECTION B: Run reference strategies ─────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION B: Reference strategies")
print("═" * 70)

print("  Running V1 reference...", flush=True)
v1_data = run_strategy("v1_ref", "nexus_alpha_v1", V1_PARAMS)
print("  Running idio_437_bw168...", flush=True)
i437_bw168_data = run_strategy("idio_lb437_bw168", "idio_momentum_alpha", make_idio_params(437, 168))
print("  Running idio_600_bw168...", flush=True)
i600_bw168_data = run_strategy("idio_lb600_bw168", "idio_momentum_alpha", make_idio_params(600, 168))
print("  Running fund_144 reference...", flush=True)
f144_data = run_strategy("fund_144_ref", "funding_momentum_alpha", FUND_144_PARAMS)
print("  Running fund_168 reference...", flush=True)
f168_data = run_strategy("fund_168_ref", "funding_momentum_alpha", FUND_168_PARAMS)

# Phase 73 champion baseline
strats_base = {
    "v1": v1_data, "idio_437": i437_bw168_data, "idio_600": i600_bw168_data,
    "f144": f144_data, "f168": f168_data,
}
p73_baseline = blend_ensemble(strats_base, {"v1": 0.15, "idio_437": 0.15, "idio_600": 0.20, "f144": 0.45, "f168": 0.05})
print(f"\n  Phase 73 champion baseline: AVG={p73_baseline['avg']}, MIN={p73_baseline['min']}, {p73_baseline['pos']}/5")

# ─── SECTION C: Best gap lb in ensemble ──────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION C: Best gap lookback in ensemble")
print("═" * 70)

# Replace I600 with best gap lb
best_gap_data = gap_results[best_gap_lb]
strats_gap = {
    "v1": v1_data, "idio_437": i437_bw168_data, "idio_2nd": best_gap_data,
    "f144": f144_data, "f168": f168_data,
}
gap_ens = blend_ensemble(strats_gap, {"v1": 0.15, "idio_437": 0.15, "idio_2nd": 0.20, "f144": 0.45, "f168": 0.05})
print(f"\n  V1(15)+I437_bw168(15)+I{best_gap_lb}_bw168(20)+F144(45)+F168(5):")
print(f"  AVG={gap_ens['avg']}, MIN={gap_ens['min']}, {gap_ens['pos']}/5, YbY={gap_ens['yby']}")

# Full grid with best gap lb
strats_gap_grid = {"v1": v1_data, "idio_437": i437_bw168_data, "idio_2nd": best_gap_data,
                   "f144": f144_data, "f168": f168_data}
gap_grid = []
for w_v1 in range(5, 26, 5):
    for w_i437 in range(10, 31, 5):
        for w_i2nd in range(10, 31, 5):
            for w_f144 in range(30, 51, 5):
                w_f168 = 100 - w_v1 - w_i437 - w_i2nd - w_f144
                if w_f168 < 0 or w_f168 > 25:
                    continue
                total_idio = w_i437 + w_i2nd
                if total_idio < 20 or total_idio > 45:
                    continue
                wts = {"v1": w_v1/100, "idio_437": w_i437/100, "idio_2nd": w_i2nd/100,
                       "f144": w_f144/100, "f168": w_f168/100}
                r = blend_ensemble(strats_gap_grid, wts)
                gap_grid.append((wts, r))

gap_grid.sort(key=lambda x: x[1]["avg"], reverse=True)
print(f"\n  Top 5 with I{best_gap_lb}_bw168 as second idio:")
for wts, r in gap_grid[:5]:
    w_v1 = int(wts["v1"] * 100)
    w_i437 = int(wts["idio_437"] * 100)
    w_i2nd = int(wts["idio_2nd"] * 100)
    w_f144 = int(wts["f144"] * 100)
    w_f168 = int(wts["f168"] * 100)
    print(f"    V1={w_v1},I437={w_i437},I{best_gap_lb}={w_i2nd},F144={w_f144},F168={w_f168}: AVG={r['avg']}, MIN={r['min']}")

# ─── SECTION D: 2.5% step grid around Phase 73 champion ─────────────────────
print("\n" + "═" * 70)
print("SECTION D: 2.5% step grid around Phase 73 champion")
print("Champion: V1=15, I437_bw168=15, I600_bw168=20, F144=45, F168=5")
print("═" * 70)

grid_25 = []
# 2.5% steps: use multiples of 2.5
# Range: V1=[10-20], I437=[10-22.5], I600=[15-27.5], F144=[37.5-50], F168=[0-12.5]
STEPS = [x * 0.025 for x in range(4, 9)]   # 10, 12.5, 15, 17.5, 20

for w_v1 in [0.10, 0.125, 0.15, 0.175, 0.20]:
    for w_i437 in [0.10, 0.125, 0.15, 0.175, 0.20, 0.225]:
        for w_i600 in [0.125, 0.15, 0.175, 0.20, 0.225, 0.25]:
            for w_f144 in [0.375, 0.40, 0.425, 0.45, 0.475, 0.50]:
                w_f168 = 1.0 - w_v1 - w_i437 - w_i600 - w_f144
                w_f168 = round(w_f168, 4)
                if w_f168 < 0 or w_f168 > 0.15:
                    continue
                total_idio = w_i437 + w_i600
                if total_idio < 0.20 or total_idio > 0.45:
                    continue
                wts = {"v1": w_v1, "idio_437": w_i437, "idio_600": w_i600,
                       "f144": w_f144, "f168": w_f168}
                r = blend_ensemble(strats_base, wts)
                grid_25.append((wts, r))

grid_25.sort(key=lambda x: x[1]["avg"], reverse=True)
print(f"\n  Total 2.5% grid configurations: {len(grid_25)}")
print(f"\n  Top 10 by AVG:")
print(f"  {'V1':>7} {'I437':>7} {'I600':>7} {'F144':>7} {'F168':>7} {'AVG':>8} {'MIN':>8}")
print("  " + "─" * 60)
for wts, r in grid_25[:10]:
    print(f"  {wts['v1']*100:>7.1f} {wts['idio_437']*100:>7.1f} {wts['idio_600']*100:>7.1f} {wts['f144']*100:>7.1f} {wts['f168']*100:>7.1f} {r['avg']:>8.3f} {r['min']:>8.3f}")

# Pareto: AVG > 1.900, sort by MIN
pareto_25 = [(wts, r) for wts, r in grid_25 if r["avg"] > 1.900]
pareto_25.sort(key=lambda x: x[1]["min"], reverse=True)
print(f"\n  Best MIN-first (AVG > 1.900) from 2.5% grid:")
for wts, r in pareto_25[:5]:
    print(f"    V1={wts['v1']*100:.1f},I437_bw168={wts['idio_437']*100:.1f},I600_bw168={wts['idio_600']*100:.1f},F144={wts['f144']*100:.1f},F168={wts['f168']*100:.1f}: AVG={r['avg']}, MIN={r['min']}")
    print(f"      YbY: {r['yby']}")

print(f"\n  AVG/MIN tradeoff frontier:")
last_min = 0
for wts, r in grid_25:
    if r["min"] > last_min:
        last_min = r["min"]
        print(f"    AVG={r['avg']:.3f}, MIN={r['min']:.3f}, YbY={r['yby']}")
        if r["avg"] < 1.850:
            break

# ─── SECTION E: Grand Summary ─────────────────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION E: Phase 74 Summary")
print("═" * 70)

best_25_avg = grid_25[0] if grid_25 else None
best_25_balanced = pareto_25[0] if pareto_25 else None

print(f"\n  Phase 73 champion: AVG=1.909, MIN=1.170")
print(f"    V1(15)+I437_bw168(15)+I600_bw168(20)+F144(45)+F168(5)")

if best_25_avg:
    wts, r = best_25_avg
    print(f"\n  Best 2.5% grid (AVG-max): AVG={r['avg']}, MIN={r['min']}")
    print(f"    V1={wts['v1']*100:.1f},I437_bw168={wts['idio_437']*100:.1f},I600_bw168={wts['idio_600']*100:.1f},F144={wts['f144']*100:.1f},F168={wts['f168']*100:.1f}")
    print(f"    YbY: {r['yby']}")

if best_25_balanced:
    wts, r = best_25_balanced
    print(f"\n  Best 2.5% grid (balanced, AVG>1.900): AVG={r['avg']}, MIN={r['min']}")
    print(f"    V1={wts['v1']*100:.1f},I437_bw168={wts['idio_437']*100:.1f},I600_bw168={wts['idio_600']*100:.1f},F144={wts['f144']*100:.1f},F168={wts['f168']*100:.1f}")
    print(f"    YbY: {r['yby']}")

print(f"\n  Idio lookback gap search (bw=168):")
for lb in GAP_LOOKBACKS:
    d = gap_results[lb]
    yby = [round(d.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]
    print(f"    lb={lb}h: AVG={d['_avg']:.3f}, MIN={d['_min']:.3f}, YbY={yby}")

if gap_grid:
    wts_g, r_g = gap_grid[0]
    print(f"\n  Best I{best_gap_lb} in ensemble: AVG={r_g['avg']}, MIN={r_g['min']}")

# Save best balanced if improved
save_configs = []
if best_25_balanced:
    wts, r = best_25_balanced
    if r["avg"] > P73_CHAMP["avg"] or r["min"] > P73_CHAMP["min"]:
        config_name = f"ensemble_p74_balanced_bw168x168"
        comment = (f"Phase 74 balanced champion: V1({wts['v1']*100:.1f}%)+I437_bw168({wts['idio_437']*100:.1f}%)"
                   f"+I600_bw168({wts['idio_600']*100:.1f}%)+F144({wts['f144']*100:.1f}%)+F168({wts['f168']*100:.1f}%)"
                   f". AVG={r['avg']}, MIN={r['min']}. 2.5% step refinement of Phase 73.")
        sub_strats = [
            {"name": "nexus_alpha_v1", "params": V1_PARAMS, "weight": wts["v1"]},
            {"name": "idio_momentum_alpha", "params": make_idio_params(437, 168), "weight": wts["idio_437"]},
            {"name": "idio_momentum_alpha", "params": make_idio_params(600, 168), "weight": wts["idio_600"]},
            {"name": "funding_momentum_alpha", "params": FUND_144_PARAMS, "weight": wts["f144"]},
            {"name": "funding_momentum_alpha", "params": FUND_168_PARAMS, "weight": wts["f168"]},
        ]
        save_champion(config_name, comment, sub_strats)
        save_configs.append(config_name)

if best_25_avg and (not best_25_balanced or best_25_avg[1]["avg"] > best_25_balanced[1]["avg"] + 0.005):
    wts, r = best_25_avg
    if r["avg"] > 1.933:  # only save if beats phase 72 avg-max
        config_name = f"ensemble_p74_avgmax_bw168x168"
        comment = (f"Phase 74 AVG-max champion. AVG={r['avg']}, MIN={r['min']}.")
        sub_strats = [
            {"name": "nexus_alpha_v1", "params": V1_PARAMS, "weight": wts["v1"]},
            {"name": "idio_momentum_alpha", "params": make_idio_params(437, 168), "weight": wts["idio_437"]},
            {"name": "idio_momentum_alpha", "params": make_idio_params(600, 168), "weight": wts["idio_600"]},
            {"name": "funding_momentum_alpha", "params": FUND_144_PARAMS, "weight": wts["f144"]},
            {"name": "funding_momentum_alpha", "params": FUND_168_PARAMS, "weight": wts["f168"]},
        ]
        save_champion(config_name, comment, sub_strats)
        save_configs.append(config_name)

if save_configs:
    print(f"\n  ★ New champion configs saved: {save_configs}")
else:
    print(f"\n  No significant improvement over Phase 73 champion (AVG=1.909, MIN=1.170)")

print("\n" + "=" * 70)
print("PHASE 74 COMPLETE")
print("=" * 70)
