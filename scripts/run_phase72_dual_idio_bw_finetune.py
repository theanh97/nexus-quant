#!/usr/bin/env python3
"""
Phase 72: Dual-Idio + Beta Window Combination + Fine Grid Search

Phase 71 champion: V1(15%)+Idio_437(21%)+Idio_600(14%)+F144(40%)+F168(10%)
  AVG=1.861, MIN=1.097, 5/5 positive

Phase 71 also found:
  - beta_window=336 in champion ensemble: AVG=1.880 (BEST AVG but MIN=0.995)
  - beta_window=168 standalone: AVG=1.222 (better MIN standalone than 336)
  - Dual-idio standalone: 437×60%+600×40% = AVG=1.226

Hypotheses to test:
  A. Dual-idio with beta_window=336: Idio_437_bw336 + Idio_600_bw336
     Does using longer beta window improve BOTH idio signals?
  B. Mixed beta windows: Idio_437_bw336 + Idio_600_bw72
     437 with longer window (more stable) + 600 with default window
  C. Idio_437_bw168 + Idio_600_bw72 (168h for 437, default for 600)
  D. Fine grid search (5% steps) around Phase 71 champion
     V1=[10-25%], I437=[15-30%], I600=[0-20%], F144=[30-45%], F168=[5-20%]
  E. beta_window=168 specifically: does it combine well with idio_600?
"""

import json
import math
import statistics
import subprocess
import sys
import copy
from pathlib import Path

OUT_DIR = "artifacts/phase72"
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
    path = f"/tmp/phase72_{run_name}_{year}.json"
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
print("PHASE 72: Dual-Idio + Beta Window Combination + Fine Grid")
print("=" * 70)

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# ─── SECTION A: Run idio variants ────────────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION A: Running idio variants for ensemble blending")
print("Phase 71 champion used: Idio_437_bw72 + Idio_600_bw72")
print("Testing: bw=[72, 168, 336] × lb=[437, 600]")
print("═" * 70)

idio_variants = {}
for lb, bw in [(437, 72), (437, 168), (437, 336), (600, 72), (600, 168), (600, 336)]:
    label = f"idio_lb{lb}_bw{bw}"
    data = run_strategy(label, "idio_momentum_alpha", make_idio_params(lb, bw))
    idio_variants[(lb, bw)] = data

print("\n── Idio variants summary ────────────────────────────────────────────────")
print(f"{'Variant':25} {'AVG':>8} {'MIN':>8} {'YbY (2021-2025)':>50}")
print("─" * 90)
for (lb, bw), d in sorted(idio_variants.items()):
    yby = [round(d.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]
    print(f"  idio_lb{lb}_bw{bw:<6}       {d['_avg']:>8.3f} {d['_min']:>8.3f}  {str(yby)}")

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

# ─── SECTION C: Phase 71 champion as baseline ─────────────────────────────────
print("\n" + "═" * 70)
print("SECTION C: Phase 71 champion as baseline")
print("═" * 70)

p71_strats = {
    "v1": v1_data,
    "idio_437": idio_variants[(437, 72)],
    "idio_600": idio_variants[(600, 72)],
    "f144": f144_data,
    "f168": f168_data,
}
p71_champ = blend_ensemble(p71_strats, {"v1": 0.15, "idio_437": 0.21, "idio_600": 0.14, "f144": 0.40, "f168": 0.10})
print(f"\n  Phase 71 champion (baseline): AVG={p71_champ['avg']}, MIN={p71_champ['min']}, {p71_champ['pos']}/5")
print(f"  YbY: {p71_champ['yby']}")

# ─── SECTION D: Beta-window dual-idio combinations ───────────────────────────
print("\n" + "═" * 70)
print("SECTION D: Best beta-window dual-idio combinations")
print("═" * 70)

# Test 9 combinations of (437_bw) × (600_bw)
combo_results = []
for bw437 in [72, 168, 336]:
    for bw600 in [72, 168, 336]:
        strats = {
            "v1": v1_data,
            "idio_437": idio_variants[(437, bw437)],
            "idio_600": idio_variants[(600, bw600)],
            "f144": f144_data,
            "f168": f168_data,
        }
        # Use Phase 71 champion weights as baseline
        r = blend_ensemble(strats, {"v1": 0.15, "idio_437": 0.21, "idio_600": 0.14, "f144": 0.40, "f168": 0.10})
        combo_results.append((bw437, bw600, r))

combo_results.sort(key=lambda x: x[2]["avg"], reverse=True)
print(f"\n  All bw437×bw600 combinations (Phase 71 weights: V1=15,I437=21,I600=14,F144=40,F168=10):")
print(f"  {'bw437':>8} {'bw600':>8} {'AVG':>8} {'MIN':>8} {'YbY'}")
print("  " + "─" * 80)
for bw437, bw600, r in combo_results:
    marker = " ←" if bw437 == combo_results[0][0] and bw600 == combo_results[0][1] else ""
    print(f"  {bw437:>8} {bw600:>8} {r['avg']:>8.3f} {r['min']:>8.3f}  {r['yby']}{marker}")

best_bw437, best_bw600, best_combo_r = combo_results[0]
print(f"\n  Best bw combo: 437_bw{best_bw437} + 600_bw{best_bw600}: AVG={best_combo_r['avg']}, MIN={best_combo_r['min']}")

# ─── SECTION E: Fine grid search around best configuration ───────────────────
print("\n" + "═" * 70)
print("SECTION E: Fine grid search (5% steps) around best configuration")
print("═" * 70)

best_strats_for_grid = {
    "v1": v1_data,
    "idio_437": idio_variants[(437, best_bw437)],
    "idio_600": idio_variants[(600, best_bw600)],
    "f144": f144_data,
    "f168": f168_data,
}

grid_results = []
for w_v1 in range(5, 26, 5):
    for w_i437 in range(10, 36, 5):
        for w_i600 in range(0, 26, 5):
            for w_f144 in range(25, 51, 5):
                w_f168 = 100 - w_v1 - w_i437 - w_i600 - w_f144
                if w_f168 < 5 or w_f168 > 25:
                    continue
                if w_i437 + w_i600 < 15 or w_i437 + w_i600 > 40:
                    continue
                wts = {
                    "v1": w_v1/100, "idio_437": w_i437/100, "idio_600": w_i600/100,
                    "f144": w_f144/100, "f168": w_f168/100
                }
                r = blend_ensemble(best_strats_for_grid, wts)
                grid_results.append((wts, r))

grid_results.sort(key=lambda x: x[1]["avg"], reverse=True)
print(f"\n  Top 10 configurations (5% step grid):")
print(f"  {'V1':>4} {'I437':>6} {'I600':>6} {'F144':>6} {'F168':>6} {'AVG':>8} {'MIN':>8}")
print("  " + "─" * 55)
for wts, r in grid_results[:10]:
    w_v1 = int(wts["v1"] * 100)
    w_i437 = int(wts["idio_437"] * 100)
    w_i600 = int(wts["idio_600"] * 100)
    w_f144 = int(wts["f144"] * 100)
    w_f168 = int(wts["f168"] * 100)
    print(f"  {w_v1:>4} {w_i437:>6} {w_i600:>6} {w_f144:>6} {w_f168:>6} {r['avg']:>8.3f} {r['min']:>8.3f}")

best_grid_wts, best_grid_r = grid_results[0]
print(f"\n  Best grid config: AVG={best_grid_r['avg']}, MIN={best_grid_r['min']}, {best_grid_r['pos']}/5")
print(f"  YbY: {best_grid_r['yby']}")

# Compare grid best with best beta_window single idio
print(f"\n── Comparison with single-idio bw=336 ensemble ────────────────────────")
strats_single_bw336 = {
    "v1": v1_data,
    "idio": idio_variants[(437, 336)],
    "f144": f144_data,
    "f168": f168_data,
}
single_bw336 = blend_ensemble(strats_single_bw336, {"v1": 0.15, "idio": 0.30, "f144": 0.40, "f168": 0.15})
print(f"  Single-idio bw=336: AVG={single_bw336['avg']}, MIN={single_bw336['min']}, {single_bw336['pos']}/5, YbY={single_bw336['yby']}")

# Also test full 5% grid for single bw=336
grid_single = []
for w_v1 in range(5, 26, 5):
    for w_idio in range(20, 41, 5):
        for w_f144 in range(25, 51, 5):
            w_f168 = 100 - w_v1 - w_idio - w_f144
            if w_f168 < 5 or w_f168 > 25:
                continue
            wts_s = {"v1": w_v1/100, "idio": w_idio/100, "f144": w_f144/100, "f168": w_f168/100}
            r_s = blend_ensemble(strats_single_bw336, wts_s)
            grid_single.append((wts_s, r_s))

grid_single.sort(key=lambda x: x[1]["avg"], reverse=True)
print(f"\n  Best single-idio_bw336 grid configs:")
for wts_s, r_s in grid_single[:5]:
    w_v1 = int(wts_s["v1"] * 100)
    w_idio = int(wts_s["idio"] * 100)
    w_f144 = int(wts_s["f144"] * 100)
    w_f168 = int(wts_s["f168"] * 100)
    print(f"    V1={w_v1},Idio_bw336={w_idio},F144={w_f144},F168={w_f168}: AVG={r_s['avg']}, MIN={r_s['min']}")

# ─── SECTION F: Grand Summary + Champion Candidates ────────────────────────────
print("\n" + "═" * 70)
print("SECTION F: Phase 72 Summary + Champion Candidates")
print("═" * 70)

CURRENT_CHAMP_AVG = 1.861
CURRENT_CHAMP_MIN = 1.097

candidates = []
# Phase 71 champion
candidates.append(("Phase 71 (dual-idio_bw72)", {"avg": 1.861, "min": 1.097, "yby": p71_champ["yby"]},
                   "V1(15)+I437_bw72(21)+I600_bw72(14)+F144(40)+F168(10)"))
# Best dual-idio bw combo
candidates.append((f"Best bw combo (437_bw{best_bw437}+600_bw{best_bw600})",
                   {"avg": best_combo_r["avg"], "min": best_combo_r["min"], "yby": best_combo_r["yby"]},
                   f"V1(15)+I437_bw{best_bw437}(21)+I600_bw{best_bw600}(14)+F144(40)+F168(10)"))
# Best grid
wts_g, r_g = grid_results[0]
candidates.append(("Best 5% grid (dual-idio)", {"avg": r_g["avg"], "min": r_g["min"], "yby": r_g["yby"]},
                   f"V1({int(wts_g['v1']*100)})+I437_bw{best_bw437}({int(wts_g['idio_437']*100)})+I600_bw{best_bw600}({int(wts_g['idio_600']*100)})+F144({int(wts_g['f144']*100)})+F168({int(wts_g['f168']*100)})"))
# Best single bw336
wts_s_best, r_s_best = grid_single[0]
candidates.append(("Best single-idio_bw336 grid", {"avg": r_s_best["avg"], "min": r_s_best["min"], "yby": r_s_best["yby"]},
                   f"V1({int(wts_s_best['v1']*100)})+Idio_bw336({int(wts_s_best['idio']*100)})+F144({int(wts_s_best['f144']*100)})+F168({int(wts_s_best['f168']*100)})"))

print(f"\n  {'Configuration':45} {'AVG':>8} {'MIN':>8} {'YbY'}")
print("  " + "─" * 100)
for desc, r, detail in sorted(candidates, key=lambda x: x[1]["avg"], reverse=True):
    print(f"  {desc:45} {r['avg']:>8.3f} {r['min']:>8.3f}  {r['yby']}")
    print(f"    → {detail}")

# Save new champion if better than Phase 71
best_candidate = max(candidates, key=lambda x: x[1]["avg"])
desc, r, detail = best_candidate
if r["avg"] > CURRENT_CHAMP_AVG or (r["avg"] == CURRENT_CHAMP_AVG and r["min"] > CURRENT_CHAMP_MIN):
    print(f"\n  ★ NEW CHAMPION: {desc}")
    print(f"    AVG={r['avg']}, MIN={r['min']}")

    # Save both AVG-max and MIN-max configs
    # Save AVG champion
    if best_bw437 != 72 or best_bw600 != 72:
        # New bw combo champion
        bw437_k, bw600_k = best_bw437, best_bw600
        wts_champ = wts_g if r["avg"] == r_g["avg"] else {"v1": 0.15, "idio_437": 0.21, "idio_600": 0.14, "f144": 0.40, "f168": 0.10}
        config_name = f"ensemble_dual_idio_p72_bw{best_bw437}x{best_bw600}"
        comment = (f"Phase 72 champion: dual-idio {detail}."
                   f" AVG={r['avg']}, MIN={r['min']}. Best in class beating P71=1.861")
        sub_strats = [
            {"name": "nexus_alpha_v1", "params": V1_PARAMS, "weight": wts_g["v1"]},
            {"name": "idio_momentum_alpha", "params": make_idio_params(437, best_bw437), "weight": wts_g["idio_437"]},
            {"name": "idio_momentum_alpha", "params": make_idio_params(600, best_bw600), "weight": wts_g["idio_600"]},
            {"name": "funding_momentum_alpha", "params": FUND_144_PARAMS, "weight": wts_g["f144"]},
            {"name": "funding_momentum_alpha", "params": FUND_168_PARAMS, "weight": wts_g["f168"]},
        ]
        save_champion(config_name, comment, sub_strats)

    # Save single-idio bw336 if it has higher AVG
    if r_s_best["avg"] > r["avg"]:
        config_name_s = f"ensemble_single_idio_bw336_p72"
        comment_s = (f"Phase 72 single-idio_bw336 champion: {detail}."
                     f" AVG={r_s_best['avg']}, MIN={r_s_best['min']}.")
        sub_strats_s = [
            {"name": "nexus_alpha_v1", "params": V1_PARAMS, "weight": wts_s_best["v1"]},
            {"name": "idio_momentum_alpha", "params": make_idio_params(437, 336), "weight": wts_s_best["idio"]},
            {"name": "funding_momentum_alpha", "params": FUND_144_PARAMS, "weight": wts_s_best["f144"]},
            {"name": "funding_momentum_alpha", "params": FUND_168_PARAMS, "weight": wts_s_best["f168"]},
        ]
        save_champion(config_name_s, comment_s, sub_strats_s)
else:
    print(f"\n  No improvement over Phase 71 champion (AVG={CURRENT_CHAMP_AVG})")

print("\n" + "=" * 70)
print("PHASE 72 COMPLETE")
print("=" * 70)
