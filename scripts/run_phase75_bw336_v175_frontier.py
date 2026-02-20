#!/usr/bin/env python3
"""
Phase 75: Mixed bw336×168 with V1=17.5 Region + Single-I600 Approach

Phase 74 balanced champion: V1=17.5, I437_bw168=12.5, I600_bw168=22.5, F144=47.5, F168=0
  AVG=1.902, MIN=1.199
Phase 72 AVG-max: V1=10, I437_bw336=20, I600_bw168=20, F144=35, F168=15
  AVG=1.933, MIN=0.894

Questions:
  1. Can bw336 for idio_437 push AVG higher while keeping MIN near 1.199?
     → V1=17.5, I437_bw336=12.5, I600_bw168=22.5, F144=47.5, F168=0
  2. Does dropping idio_437 entirely (use ONLY I600_bw168) simplify and improve?
     → V1=20, I600_bw168=35, F144=45, F168=0
  3. Full 2.5% grid for bw336×168 in the V1=15-20 region
  4. Can we find AVG>1.910 AND MIN>1.170 simultaneously?
"""

import json
import math
import statistics
import subprocess
import sys
import copy
from pathlib import Path

OUT_DIR = "artifacts/phase75"
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
    path = f"/tmp/phase75_{run_name}_{year}.json"
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
print("PHASE 75: Mixed bw336×168, V1=17.5 Region + Single-I600")
print("=" * 70)

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

P74_BALANCED = {"avg": 1.902, "min": 1.199}
P73_BALANCED = {"avg": 1.909, "min": 1.170}
P72_AVG_MAX  = {"avg": 1.933, "min": 0.894}

# ─── SECTION A: Run idio variants ────────────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION A: Running idio variants")
print("═" * 70)

print("  Running idio_437_bw168...", flush=True)
i437_bw168 = run_strategy("idio_lb437_bw168", "idio_momentum_alpha", make_idio_params(437, 168))
print("  Running idio_437_bw336...", flush=True)
i437_bw336 = run_strategy("idio_lb437_bw336", "idio_momentum_alpha", make_idio_params(437, 336))
print("  Running idio_600_bw168...", flush=True)
i600_bw168 = run_strategy("idio_lb600_bw168", "idio_momentum_alpha", make_idio_params(600, 168))

# ─── SECTION B: Reference strategies ─────────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION B: Reference strategies")
print("═" * 70)
print("  Running V1 reference...", flush=True)
v1_data = run_strategy("v1_ref", "nexus_alpha_v1", V1_PARAMS)
print("  Running fund_144 reference...", flush=True)
f144_data = run_strategy("fund_144_ref", "funding_momentum_alpha", FUND_144_PARAMS)
print("  Running fund_168 reference...", flush=True)
f168_data = run_strategy("fund_168_ref", "funding_momentum_alpha", FUND_168_PARAMS)

# ─── SECTION C: bw336×168 with V1=17.5 region ────────────────────────────────
print("\n" + "═" * 70)
print("SECTION C: 2.5% grid for bw336×168 in V1=15-20 region")
print("Goal: Does bw336 for I437 push AVG while keeping MIN near 1.199?")
print("═" * 70)

strats_336_168 = {
    "v1": v1_data, "idio_437": i437_bw336, "idio_600": i600_bw168,
    "f144": f144_data, "f168": f168_data,
}

# Phase 74 balanced analog with bw336
p74_bw336 = blend_ensemble(strats_336_168,
    {"v1": 0.175, "idio_437": 0.125, "idio_600": 0.225, "f144": 0.475, "f168": 0.0})
print(f"\n  Phase 74 analog (bw336×168, same weights): AVG={p74_bw336['avg']}, MIN={p74_bw336['min']}")
print(f"  YbY: {p74_bw336['yby']}")

# Full 2.5% grid for bw336×168
grid_336_168 = []
for w_v1 in [0.125, 0.15, 0.175, 0.20]:
    for w_i437 in [0.10, 0.125, 0.15, 0.175, 0.20]:
        for w_i600 in [0.15, 0.175, 0.20, 0.225, 0.25]:
            for w_f144 in [0.375, 0.40, 0.425, 0.45, 0.475, 0.50]:
                for w_f168 in [0.0, 0.025, 0.05, 0.075, 0.10]:
                    total = w_v1 + w_i437 + w_i600 + w_f144 + w_f168
                    if abs(total - 1.0) > 0.001:
                        continue
                    total_idio = w_i437 + w_i600
                    if total_idio < 0.25 or total_idio > 0.45:
                        continue
                    wts = {"v1": w_v1, "idio_437": w_i437, "idio_600": w_i600,
                           "f144": w_f144, "f168": w_f168}
                    r = blend_ensemble(strats_336_168, wts)
                    grid_336_168.append((wts, r))

grid_336_168.sort(key=lambda x: x[1]["avg"], reverse=True)
print(f"\n  Top 10 by AVG (bw336×168):")
print(f"  {'V1':>7} {'I437':>7} {'I600':>7} {'F144':>7} {'F168':>7} {'AVG':>8} {'MIN':>8}")
print("  " + "─" * 60)
for wts, r in grid_336_168[:10]:
    print(f"  {wts['v1']*100:>7.1f} {wts['idio_437']*100:>7.1f} {wts['idio_600']*100:>7.1f} {wts['f144']*100:>7.1f} {wts['f168']*100:>7.1f} {r['avg']:>8.3f} {r['min']:>8.3f}")

# Best MIN-first for AVG > 1.900
pareto_336 = [(wts, r) for wts, r in grid_336_168 if r["avg"] > 1.900]
pareto_336.sort(key=lambda x: x[1]["min"], reverse=True)
print(f"\n  Best MIN-first (AVG > 1.900, bw336×168):")
for wts, r in pareto_336[:5]:
    print(f"    V1={wts['v1']*100:.1f},I437_bw336={wts['idio_437']*100:.1f},I600_bw168={wts['idio_600']*100:.1f},F144={wts['f144']*100:.1f},F168={wts['f168']*100:.1f}: AVG={r['avg']}, MIN={r['min']}")
    print(f"      YbY: {r['yby']}")

# ─── SECTION D: Single-idio approach (drop I437 entirely) ────────────────────
print("\n" + "═" * 70)
print("SECTION D: Single-I600_bw168 approach")
print("Simplify: use ONLY I600_bw168 as the idio signal")
print("I600_bw168 standalone: AVG=1.235, YbY=[1.913, 0.872, 0.417, 1.466, 1.507]")
print("═" * 70)

strats_single_600 = {
    "v1": v1_data, "idio_600": i600_bw168, "f144": f144_data, "f168": f168_data,
}

single_600_grid = []
for w_v1 in [0.10, 0.125, 0.15, 0.175, 0.20, 0.225]:
    for w_i600 in [0.20, 0.225, 0.25, 0.275, 0.30, 0.325, 0.35]:
        for w_f144 in [0.375, 0.40, 0.425, 0.45, 0.475, 0.50]:
            for w_f168 in [0.0, 0.025, 0.05, 0.075, 0.10]:
                total = w_v1 + w_i600 + w_f144 + w_f168
                if abs(total - 1.0) > 0.001:
                    continue
                wts = {"v1": w_v1, "idio_600": w_i600, "f144": w_f144, "f168": w_f168}
                r = blend_ensemble(strats_single_600, wts)
                single_600_grid.append((wts, r))

single_600_grid.sort(key=lambda x: x[1]["avg"], reverse=True)
print(f"\n  Top 10 single-I600_bw168 configurations:")
print(f"  {'V1':>7} {'I600':>7} {'F144':>7} {'F168':>7} {'AVG':>8} {'MIN':>8}")
print("  " + "─" * 55)
for wts, r in single_600_grid[:10]:
    print(f"  {wts['v1']*100:>7.1f} {wts['idio_600']*100:>7.1f} {wts['f144']*100:>7.1f} {wts.get('f168', 0)*100:>7.1f} {r['avg']:>8.3f} {r['min']:>8.3f}  {r['yby']}")

pareto_single = [(wts, r) for wts, r in single_600_grid if r["avg"] > 1.900]
pareto_single.sort(key=lambda x: x[1]["min"], reverse=True)
print(f"\n  Best single-I600 (AVG > 1.900) by MIN:")
for wts, r in pareto_single[:5]:
    print(f"    V1={wts['v1']*100:.1f},I600_bw168={wts['idio_600']*100:.1f},F144={wts['f144']*100:.1f},F168={wts.get('f168',0)*100:.1f}: AVG={r['avg']}, MIN={r['min']}")
    print(f"      YbY: {r['yby']}")

# ─── SECTION E: Grand Summary + Frontier ──────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION E: Phase 75 Summary — AVG/MIN Frontier")
print("═" * 70)

all_results = []
# Previous champions
all_results.append(("Phase 72 AVG-max", P72_AVG_MAX, "V1(10)+I437_bw336(20)+I600_bw168(20)+F144(35)+F168(15)"))
all_results.append(("Phase 73 balanced", P73_BALANCED, "V1(15)+I437_bw168(15)+I600_bw168(20)+F144(45)+F168(5)"))
all_results.append(("Phase 74 balanced", P74_BALANCED, "V1(17.5)+I437_bw168(12.5)+I600_bw168(22.5)+F144(47.5)+F168(0)"))

# New from this phase
if grid_336_168:
    wts_g, r_g = grid_336_168[0]
    all_results.append(("Phase 75 bw336×168 AVG-max", {"avg": r_g["avg"], "min": r_g["min"]},
                       f"V1({wts_g['v1']*100:.1f})+I437_bw336({wts_g['idio_437']*100:.1f})+I600_bw168({wts_g['idio_600']*100:.1f})+F144({wts_g['f144']*100:.1f})+F168({wts_g['f168']*100:.1f})"))
    if pareto_336:
        wts_p, r_p = pareto_336[0]
        all_results.append(("Phase 75 bw336×168 balanced", {"avg": r_p["avg"], "min": r_p["min"]},
                           f"V1({wts_p['v1']*100:.1f})+I437_bw336({wts_p['idio_437']*100:.1f})+I600_bw168({wts_p['idio_600']*100:.1f})+F144({wts_p['f144']*100:.1f})+F168({wts_p['f168']*100:.1f})"))

if single_600_grid:
    wts_s, r_s = single_600_grid[0]
    all_results.append(("Phase 75 single-I600 AVG-max", {"avg": r_s["avg"], "min": r_s["min"]},
                       f"V1({wts_s['v1']*100:.1f})+I600_bw168({wts_s['idio_600']*100:.1f})+F144({wts_s['f144']*100:.1f})+F168({wts_s.get('f168',0)*100:.1f})"))
    if pareto_single:
        wts_ps, r_ps = pareto_single[0]
        all_results.append(("Phase 75 single-I600 balanced", {"avg": r_ps["avg"], "min": r_ps["min"]},
                           f"V1({wts_ps['v1']*100:.1f})+I600_bw168({wts_ps['idio_600']*100:.1f})+F144({wts_ps['f144']*100:.1f})+F168({wts_ps.get('f168',0)*100:.1f})"))

all_results.sort(key=lambda x: x[1]["avg"], reverse=True)
print(f"\n  {'Configuration':40} {'AVG':>8} {'MIN':>8}")
print("  " + "─" * 55)
for desc, r, detail in all_results:
    print(f"  {desc:40} {r['avg']:>8.3f} {r['min']:>8.3f}")
    print(f"    → {detail}")

# Save new champions
save_count = 0
# Check if bw336 balanced beats Phase 74
if pareto_336 and (pareto_336[0][1]["avg"] > P74_BALANCED["avg"] or pareto_336[0][1]["min"] > P74_BALANCED["min"]):
    wts, r = pareto_336[0]
    config_name = "ensemble_p75_bw336x168_balanced"
    comment = (f"Phase 75 bw336×168 balanced champion. "
               f"V1({wts['v1']*100:.1f})+I437_bw336({wts['idio_437']*100:.1f})+I600_bw168({wts['idio_600']*100:.1f})+F144({wts['f144']*100:.1f})+F168({wts['f168']*100:.1f}). "
               f"AVG={r['avg']}, MIN={r['min']}. YbY={r['yby']}")
    sub_strats = [
        {"name": "nexus_alpha_v1", "params": V1_PARAMS, "weight": wts["v1"]},
        {"name": "idio_momentum_alpha", "params": make_idio_params(437, 336), "weight": wts["idio_437"]},
        {"name": "idio_momentum_alpha", "params": make_idio_params(600, 168), "weight": wts["idio_600"]},
        {"name": "funding_momentum_alpha", "params": FUND_144_PARAMS, "weight": wts["f144"]},
        {"name": "funding_momentum_alpha", "params": FUND_168_PARAMS, "weight": wts["f168"]},
    ]
    save_champion(config_name, comment, sub_strats)
    save_count += 1

# Check if bw336 AVG-max beats Phase 72
if grid_336_168 and grid_336_168[0][1]["avg"] > P72_AVG_MAX["avg"]:
    wts, r = grid_336_168[0]
    config_name = "ensemble_p75_bw336x168_avgmax"
    comment = (f"Phase 75 bw336×168 AVG-max. AVG={r['avg']}, MIN={r['min']}")
    sub_strats = [
        {"name": "nexus_alpha_v1", "params": V1_PARAMS, "weight": wts["v1"]},
        {"name": "idio_momentum_alpha", "params": make_idio_params(437, 336), "weight": wts["idio_437"]},
        {"name": "idio_momentum_alpha", "params": make_idio_params(600, 168), "weight": wts["idio_600"]},
        {"name": "funding_momentum_alpha", "params": FUND_144_PARAMS, "weight": wts["f144"]},
        {"name": "funding_momentum_alpha", "params": FUND_168_PARAMS, "weight": wts["f168"]},
    ]
    save_champion(config_name, comment, sub_strats)
    save_count += 1

print(f"\n  Saved {save_count} new champion configs.")
print(f"\n  Current best overall:")
print(f"    AVG-max:  AVG={P72_AVG_MAX['avg']}, MIN={P72_AVG_MAX['min']} (Phase 72)")
print(f"    Balanced: AVG={P74_BALANCED['avg']}, MIN={P74_BALANCED['min']} (Phase 74)")

print("\n" + "=" * 70)
print("PHASE 75 COMPLETE")
print("=" * 70)
