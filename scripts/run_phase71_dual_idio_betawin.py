#!/usr/bin/env python3
"""
Phase 71: Dual-Idio Blend + Beta Window Sensitivity

Phase 68 champion: V1(15%)+Idio(30%)+F144(40%)+F168(15%) AVG=1.844, MIN=0.996

Key insight from Phase 70:
  - idio_437: AVG=1.200, 2022=0.505 (WEAK in 2022)
  - idio_600: AVG=1.060, 2022=1.366 (MUCH BETTER in 2022)
  - Current ensemble 2022 floor ≈ 0.996, driven by weak F144+F168+Idio in 2022
  - If we combine idio_437 (good for 2023-2024-2025) with idio_600 (good for 2022),
    we could improve 2022 floor while preserving other year performance

Goals:
  A. IdioMomentum beta_window sensitivity: beta_window=[24, 48, 72, 168, 336]
     Does the OLS estimation window matter? 72h is the current default.
  B. Dual-idio ensemble blending: split idio weight between 437 and 600
     Key question: does idio_437+idio_600 mix outperform just idio_437?
  C. Idio with short+long lookback combo in ensemble grid
  D. If best combo > 1.844, save as new champion config
"""

import json
import math
import statistics
import subprocess
import sys
import copy
from pathlib import Path

OUT_DIR = "artifacts/phase71"
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

# ── Frozen params ─────────────────────────────────────────────────────────────
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

IDIO_437_PARAMS = {
    "k_per_side": 2, "lookback_bars": 437, "beta_window_bars": 72,
    "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 48,
}

IDIO_600_PARAMS = {
    "k_per_side": 2, "lookback_bars": 600, "beta_window_bars": 72,
    "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 48,
}

FUND_144_PARAMS = {
    "k_per_side": 2, "funding_lookback_bars": 144, "direction": "contrarian",
    "vol_lookback_bars": 168, "target_gross_leverage": 0.25, "rebalance_interval_bars": 24,
}

FUND_168_PARAMS = {
    "k_per_side": 2, "funding_lookback_bars": 168, "direction": "contrarian",
    "vol_lookback_bars": 168, "target_gross_leverage": 0.25, "rebalance_interval_bars": 24,
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
    path = f"/tmp/phase71_{run_name}_{year}.json"
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


def pearson_corr(x: list, y: list) -> float:
    n = min(len(x), len(y))
    if n < 10:
        return float("nan")
    x, y = x[:n], y[:n]
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((x[i] - mx) * (y[i] - my) for i in range(n)) / n
    sx = math.sqrt(sum((v - mx) ** 2 for v in x) / n)
    sy = math.sqrt(sum((v - my) ** 2 for v in y) / n)
    if sx < 1e-10 or sy < 1e-10:
        return float("nan")
    return round(cov / (sx * sy), 4)


# ═════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PHASE 71: Dual-Idio Blend + Beta Window Sensitivity")
print("=" * 70)

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# ─── SECTION A: Beta window sensitivity for Idio_437 ─────────────────────────
print("\n" + "═" * 70)
print("SECTION A: Idio beta_window sensitivity (lookback=437, vary beta_window)")
print("Current: beta_window=72h. Testing: 24, 48, 72, 168, 336")
print("═" * 70)

BETA_WINDOWS = [24, 48, 72, 168, 336]
beta_win_results = {}

for bw in BETA_WINDOWS:
    label = f"idio_lb437_bw{bw}"
    params = {
        "k_per_side": 2, "lookback_bars": 437, "beta_window_bars": bw,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.30,
        "rebalance_interval_bars": 48,
    }
    data = run_strategy(label, "idio_momentum_alpha", params)
    beta_win_results[bw] = data

print("\n── Beta window sensitivity summary ─────────────────────────────────────")
print(f"{'BetaWin':>10} {'AVG':>8} {'MIN':>8} {'Pos':>6}")
print("─" * 38)
for bw in BETA_WINDOWS:
    d = beta_win_results[bw]
    marker = " ← BEST" if bw == max(BETA_WINDOWS, key=lambda b: beta_win_results[b]["_avg"]) else ""
    print(f"{bw:>10}h {d['_avg']:>8.3f} {d['_min']:>8.3f} {d['_pos']:>6}/5{marker}")

best_bw = max(BETA_WINDOWS, key=lambda bw: beta_win_results[bw]["_avg"])
best_bw_avg = beta_win_results[best_bw]["_avg"]
print(f"\nBest beta_window: {best_bw}h, AVG={best_bw_avg:.3f}")

# ─── SECTION B: Dual-idio standalone (idio_437 + idio_600 blend) ─────────────
print("\n" + "═" * 70)
print("SECTION B: Dual-Idio standalone blend (idio_437 + idio_600)")
print("idio_437 2022=0.505, idio_600 2022=1.366 — dual may give better floor")
print("═" * 70)

print("  Running idio_437 reference...", flush=True)
idio_437_data = run_strategy("idio_437_ref", "idio_momentum_alpha", IDIO_437_PARAMS)
print("  Running idio_600 reference...", flush=True)
idio_600_data = run_strategy("idio_600_ref", "idio_momentum_alpha", IDIO_600_PARAMS)

# Test standalone blend ratios
print("\n  Testing dual-idio blend (standalone vs baseline):")
dual_strats = {"idio_437": idio_437_data, "idio_600": idio_600_data}
dual_splits = [(1.0, 0.0), (0.8, 0.2), (0.6, 0.4), (0.5, 0.5), (0.4, 0.6), (0.2, 0.8), (0.0, 1.0)]
best_dual_split = None
best_dual_avg = 0.0

for w437, w600 in dual_splits:
    r = blend_ensemble(dual_strats, {"idio_437": w437, "idio_600": w600})
    print(f"    Idio_437({int(w437*100)})+Idio_600({int(w600*100)}): AVG={r['avg']}, MIN={r['min']}, {r['pos']}/5, YbY={r['yby']}")
    if r["avg"] > best_dual_avg:
        best_dual_avg = r["avg"]
        best_dual_split = (w437, w600)

print(f"\n  Best dual split: 437×{best_dual_split[0]:.0%} + 600×{best_dual_split[1]:.0%}, AVG={best_dual_avg:.3f}")

# ─── SECTION C: Best idio config in ensemble ─────────────────────────────────
print("\n" + "═" * 70)
print("SECTION C: Best idio configuration in ensemble")
print("═" * 70)

print("  Running V1 reference...", flush=True)
v1_data = run_strategy("v1_ref", "nexus_alpha_v1", V1_PARAMS)
print("  Running fund_144 reference...", flush=True)
f144_data = run_strategy("fund_144_ref", "funding_momentum_alpha", FUND_144_PARAMS)
print("  Running fund_168 reference...", flush=True)
f168_data = run_strategy("fund_168_ref", "funding_momentum_alpha", FUND_168_PARAMS)

# Also run best beta_window if different from 72
best_bw_data = beta_win_results[best_bw]
if best_bw != 72:
    print(f"  Best beta_window ({best_bw}h) differs from default 72h!", flush=True)
    print(f"  YbY for best bw={best_bw}: {[round(best_bw_data.get(y, {}).get('sharpe', 0.0), 3) for y in YEARS]}")

# Phase 68 baseline
base_strats = {"v1": v1_data, "idio": idio_437_data, "f144": f144_data, "f168": f168_data}
p68_champ = blend_ensemble(base_strats, {"v1": 0.15, "idio": 0.30, "f144": 0.40, "f168": 0.15})
print(f"\n  Phase 68 champion (baseline): AVG={p68_champ['avg']}, MIN={p68_champ['min']}, {p68_champ['pos']}/5")
print(f"  YbY: {p68_champ['yby']}")

print(f"\n── C1: Dual-idio in champion grid ─────────────────────────────────────")
strats_dual = {"v1": v1_data, "idio_437": idio_437_data, "idio_600": idio_600_data,
               "f144": f144_data, "f168": f168_data}

# Test dual idio splits in the ensemble
best_c1 = None
best_c1_avg = 0.0
experiments = []
for w_v1 in [10, 15, 20]:
    for total_idio in [25, 30, 35]:
        for frac_437 in [1.0, 0.8, 0.6, 0.5]:
            frac_600 = 1.0 - frac_437
            w_idio_437 = round(total_idio * frac_437 / 100, 4)
            w_idio_600 = round(total_idio * frac_600 / 100, 4)
            for w_f144 in [30, 35, 40]:
                w_f168 = 1.0 - w_v1/100 - total_idio/100 - w_f144/100
                if not (0.05 <= w_f168 <= 0.25):
                    continue
                w_f168 = round(w_f168, 4)
                wts = {"v1": w_v1/100, "idio_437": w_idio_437, "idio_600": w_idio_600,
                       "f144": w_f144/100, "f168": w_f168}
                r = blend_ensemble(strats_dual, wts)
                experiments.append((wts, r))
                if r["avg"] > best_c1_avg:
                    best_c1_avg = r["avg"]
                    best_c1 = (wts, r)

# Show top 5
experiments.sort(key=lambda x: x[1]["avg"], reverse=True)
print(f"  Top 5 dual-idio configurations:")
for wts, r in experiments[:5]:
    w_v1 = int(wts["v1"] * 100)
    w_i437 = int(wts["idio_437"] * 100)
    w_i600 = int(wts["idio_600"] * 100)
    w_f144 = int(wts["f144"] * 100)
    w_f168 = int(wts["f168"] * 100)
    print(f"    V1={w_v1},I437={w_i437},I600={w_i600},F144={w_f144},F168={w_f168}: AVG={r['avg']}, MIN={r['min']}, {r['pos']}/5")

print(f"\n  Best dual-idio ensemble: AVG={best_c1[1]['avg']}, MIN={best_c1[1]['min']}, {best_c1[1]['pos']}/5")
print(f"  YbY: {best_c1[1]['yby']}")

# C2: Best beta_window in ensemble
if best_bw != 72 and best_bw_avg > 1.0:
    print(f"\n── C2: Best beta_window={best_bw} in champion ensemble ─────────────────")
    strats_best_bw = {"v1": v1_data, "idio": best_bw_data, "f144": f144_data, "f168": f168_data}
    r_bw = blend_ensemble(strats_best_bw, {"v1": 0.15, "idio": 0.30, "f144": 0.40, "f168": 0.15})
    print(f"  V1(15)+Idio_bw{best_bw}(30)+F144(40)+F168(15): AVG={r_bw['avg']}, MIN={r_bw['min']}, {r_bw['pos']}/5")
    print(f"  YbY: {r_bw['yby']}")

# ─── SECTION D: Broader grid around best dual-idio ───────────────────────────
print("\n" + "═" * 70)
print("SECTION D: Fine-tune best dual-idio configuration")
print("═" * 70)

if best_c1:
    wts, r = best_c1
    print(f"  Starting from: AVG={r['avg']}, YbY={r['yby']}")
    # Try smaller refinements around best config
    best_wts = wts
    w_v1_base = round(best_wts["v1"] * 100)
    w_i437_base = round(best_wts["idio_437"] * 100)
    w_i600_base = round(best_wts["idio_600"] * 100)
    w_f144_base = round(best_wts["f144"] * 100)
    w_f168_base = round(best_wts["f168"] * 100)
    total_idio_base = w_i437_base + w_i600_base
    print(f"  Base: V1={w_v1_base}, I437={w_i437_base}, I600={w_i600_base}, F144={w_f144_base}, F168={w_f168_base}")

    fine_results = []
    for adj_v1 in [-5, 0, 5]:
        for adj_idio_total in [-5, 0, 5]:
            for adj_f144 in [-5, 0, 5]:
                w_v1 = w_v1_base + adj_v1
                total_idio = total_idio_base + adj_idio_total
                w_f144 = w_f144_base + adj_f144
                w_f168 = 100 - w_v1 - total_idio - w_f144
                if w_v1 < 5 or total_idio < 15 or w_f144 < 20 or w_f168 < 5:
                    continue
                if w_v1 + total_idio + w_f144 + w_f168 != 100:
                    continue
                # Keep same idio split ratio
                frac_437 = w_i437_base / (total_idio_base or 1)
                w_i437 = round(total_idio * frac_437)
                w_i600 = total_idio - w_i437
                if w_i600 < 0:
                    continue
                wts_fine = {
                    "v1": w_v1/100, "idio_437": w_i437/100, "idio_600": w_i600/100,
                    "f144": w_f144/100, "f168": w_f168/100
                }
                r = blend_ensemble(strats_dual, wts_fine)
                fine_results.append((wts_fine, r))

    fine_results.sort(key=lambda x: x[1]["avg"], reverse=True)
    print(f"\n  Fine-tuning top 5:")
    for wts_f, r_f in fine_results[:5]:
        w_v1 = int(wts_f["v1"] * 100)
        w_i437 = int(wts_f["idio_437"] * 100)
        w_i600 = int(wts_f["idio_600"] * 100)
        w_f144 = int(wts_f["f144"] * 100)
        w_f168 = int(wts_f["f168"] * 100)
        print(f"    V1={w_v1},I437={w_i437},I600={w_i600},F144={w_f144},F168={w_f168}: AVG={r_f['avg']}, MIN={r_f['min']}")

# ─── SECTION E: Grand summary ─────────────────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION E: Phase 71 Summary")
print("═" * 70)

print(f"\nPhase 68 champion (baseline): V1(15)+Idio_437(30)+F144(40)+F168(15)")
print(f"  AVG={p68_champ['avg']}, MIN={p68_champ['min']}, {p68_champ['pos']}/5")
print(f"  YbY: {p68_champ['yby']}")

print(f"\nBeta window sensitivity (lb=437):")
for bw in BETA_WINDOWS:
    d = beta_win_results[bw]
    yby = [round(d.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]
    marker = " ← BEST" if bw == best_bw else ""
    print(f"  bw={bw:4}h: AVG={d['_avg']:.3f}, MIN={d['_min']:.3f}, YbY={yby}{marker}")

print(f"\nDual-idio blend (standalone):")
print(f"  Best split: 437×{best_dual_split[0]:.0%} + 600×{best_dual_split[1]:.0%}, AVG={best_dual_avg:.3f}")

print(f"\nBest dual-idio ensemble:")
if best_c1:
    wts, r = best_c1
    print(f"  V1={int(wts['v1']*100)},I437={int(wts['idio_437']*100)},I600={int(wts['idio_600']*100)},F144={int(wts['f144']*100)},F168={int(wts['f168']*100)}")
    print(f"  AVG={r['avg']}, MIN={r['min']}, {r['pos']}/5, YbY={r['yby']}")
    if r["avg"] > p68_champ["avg"]:
        print(f"  ★ NEW CHAMPION! AVG {p68_champ['avg']} → {r['avg']}")
    else:
        print(f"  No improvement over champion (AVG={p68_champ['avg']})")

# Save champion config if new best
if best_c1 and best_c1[1]["avg"] > 1.844:
    wts, r = best_c1
    config_name = f"ensemble_dual_idio_p71"
    comment = (f"Phase 71 champion: dual-idio V1({int(wts['v1']*100)}%)+Idio_437({int(wts['idio_437']*100)}%)"
               f"+Idio_600({int(wts['idio_600']*100)}%)+F144({int(wts['f144']*100)}%)+F168({int(wts['f168']*100)}%)"
               f" AVG={r['avg']}, MIN={r['min']}")
    sub_strats = [
        {"name": "nexus_alpha_v1", "params": V1_PARAMS, "weight": wts["v1"]},
        {"name": "idio_momentum_alpha", "params": IDIO_437_PARAMS, "weight": wts["idio_437"]},
        {"name": "idio_momentum_alpha", "params": IDIO_600_PARAMS, "weight": wts["idio_600"]},
        {"name": "funding_momentum_alpha", "params": FUND_144_PARAMS, "weight": wts["f144"]},
        {"name": "funding_momentum_alpha", "params": FUND_168_PARAMS, "weight": wts["f168"]},
    ]
    cfg = {
        "_comment": comment,
        "run_name": config_name,
        **{k: v for k, v in BASE_CONFIG.items()},
        "_strategies": sub_strats,
    }
    with open(f"configs/{config_name}.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\n  Saved new champion: configs/{config_name}.json")

print("\n" + "=" * 70)
print("PHASE 71 COMPLETE")
print("=" * 70)
