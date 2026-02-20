#!/usr/bin/env python3
"""
Phase 80: F144 k=3 + I437 k=4 + Fine-Grid Confirmation

Phase 79 established the Pareto frontier in the (V1, I437_k3, I600_k2, F144) signal space:
  Best balanced:  P78 → V1=17.5%, I437_k3=17.5%, I600=20%, F144=45% → 1.919/1.206
  Best AVG-max:   P79 → V1=10%, I437_k3=25%, I600=17.5%, F144=42.5%, F168=5% → 1.934/1.098
  "No config achieves AVG≥1.920 AND MIN≥1.207 simultaneously in current signal space"

Phase 80 breaks new ground:
  A. k=3 for F144 (funding signal with k_per_side=3): NEVER TESTED
     - F144 currently uses k=2 (top/bottom 2 by 144h cumulative funding)
     - k=3 means 6 active positions (3L/3S) vs current 4 (2L/2S)
     - Analogy: k=3 for I437 improved both AVG AND MIN simultaneously
     - F144 is the DOMINANT signal (45% weight) — even small improvement matters
  B. k=4 for I437: extension beyond k=3
     - k=4: 4L/4S = 8 active positions out of 10 symbols
     - May over-dilute signal (diminishing returns vs k=3 breakthrough)
  C. Combine F144_k3 with P78 balanced champion
     - Does F144_k3 + I437_k3 ensemble improve on P78 (1.919/1.206)?
  D. Fine 1.25% step grid around P78 champion
     - Confirm 1.919/1.206 is truly optimal at finer resolution
"""

import json
import math
import statistics
import subprocess
import sys
import copy
from pathlib import Path

OUT_DIR = "artifacts/phase80"
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

FUND_144_K2 = make_fund_params(144, k=2)
FUND_144_K3 = make_fund_params(144, k=3)
FUND_168_K2 = make_fund_params(168, k=2)


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
    path = f"/tmp/phase80_{run_name}_{year}.json"
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


def blend_ensemble(strategies: dict, weights: dict) -> dict:
    year_sharpes = {}
    for year in YEARS:
        pairs = []
        min_len = None
        valid = True
        for name, data in strategies.items():
            w = weights.get(name, 0.0)
            if w == 0.0:
                continue
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
        mu = statistics.mean(blended)
        sd = statistics.pstdev(blended)
        s = round((mu / sd) * math.sqrt(8760), 3) if sd > 0 else 0.0
        year_sharpes[year] = s

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


def pct_range(lo_10ths: int, hi_10ths_exclusive: int, step_10ths: int = 125) -> list:
    """Generate weight fractions. step_10ths=125 → 1.25% steps."""
    return [x / 10000 for x in range(lo_10ths, hi_10ths_exclusive, step_10ths)]


# ═════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PHASE 80: F144 k=3 + I437 k=4 + Fine-Grid Confirmation")
print("=" * 70)

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

P78_BALANCED = {"avg": 1.919, "min": 1.206}
P79_AVGMAX   = {"avg": 1.934, "min": 1.098}

# ─── Reference runs ───────────────────────────────────────────────────────────
print("\n  Running reference strategies...", flush=True)

v1_data = run_strategy("v1_ref", "nexus_alpha_v1", V1_PARAMS)
i437_k3 = run_strategy("idio_lb437_bw168_k3", "idio_momentum_alpha", make_idio_params(437, 168, k=3))
i600_k2 = run_strategy("idio_lb600_bw168", "idio_momentum_alpha", make_idio_params(600, 168, k=2))
f144_k2 = run_strategy("fund_144_k2", "funding_momentum_alpha", FUND_144_K2)
f168_k2 = run_strategy("fund_168_k2", "funding_momentum_alpha", FUND_168_K2)

# ─── SECTION A: F144 k=3 standalone ──────────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION A: F144 k=3 — NEVER TESTED before Phase 80")
print("  Current F144 k=2: more concentrated (2L/2S out of 10 symbols)")
print("  New F144 k=3: more diversified (3L/3S out of 10 symbols)")
print("═" * 70)

f144_k3 = run_strategy("fund_144_k3", "funding_momentum_alpha", FUND_144_K3)

yby_f144_k2 = [round(f144_k2.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]
yby_f144_k3 = [round(f144_k3.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]
print(f"\n  F144 k=2: AVG={f144_k2['_avg']}, MIN={f144_k2['_min']}")
print(f"    YbY: {yby_f144_k2}")
print(f"  F144 k=3: AVG={f144_k3['_avg']}, MIN={f144_k3['_min']}")
print(f"    YbY: {yby_f144_k3}")

# Delta analysis year by year
delta_yby = [round(yby_f144_k3[i] - yby_f144_k2[i], 3) for i in range(5)]
print(f"  Delta (k3-k2): {delta_yby}")

# ─── SECTION B: I437 k=4 standalone ──────────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION B: I437 k=4 — extension beyond k=3")
print("  k=4: 4L/4S = 8 active positions out of 10 symbols")
print("═" * 70)

i437_k4 = run_strategy("idio_lb437_bw168_k4", "idio_momentum_alpha", make_idio_params(437, 168, k=4))

yby_i437_k2 = [round(f144_k2.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]  # wrong, fix:
yby_i437_k2 = [round(i437_k3.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]  # using k3 as ref
yby_i437_k4 = [round(i437_k4.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]
print(f"\n  I437 k=3: AVG={i437_k3['_avg']}, MIN={i437_k3['_min']}")
print(f"    YbY: {yby_i437_k2}")
print(f"  I437 k=4: AVG={i437_k4['_avg']}, MIN={i437_k4['_min']}")
print(f"    YbY: {yby_i437_k4}")

# ─── SECTION C: F144_k3 in ensemble ──────────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION C: F144_k3 in ensemble combinations")
print("  Testing F144_k3 replacing F144_k2 in P78 champion")
print("═" * 70)

# P78 champion with F144_k3
strats_f144k3 = {"v1": v1_data, "idio_437_k3": i437_k3, "idio_600": i600_k2,
                 "f144_k3": f144_k3, "f168": f168_k2}

r_p78_f144k3 = blend_ensemble(strats_f144k3, {"v1": 0.175, "idio_437_k3": 0.175, "idio_600": 0.20, "f144_k3": 0.45})
print(f"\n  P78 weights + F144_k3: AVG={r_p78_f144k3['avg']}, MIN={r_p78_f144k3['min']}")
print(f"  YbY: {r_p78_f144k3['yby']}")

# AVG-max config with F144_k3
strats_avgmax_f144k3 = {"v1": v1_data, "idio_437_k3": i437_k3, "idio_600": i600_k2,
                         "f144_k3": f144_k3, "f168": f168_k2}
r_avgmax_f144k3 = blend_ensemble(strats_avgmax_f144k3, {"v1": 0.10, "idio_437_k3": 0.25, "idio_600": 0.175, "f144_k3": 0.425, "f168": 0.05})
print(f"  P79 AVG-max + F144_k3: AVG={r_avgmax_f144k3['avg']}, MIN={r_avgmax_f144k3['min']}")
print(f"  YbY: {r_avgmax_f144k3['yby']}")

# Grid: F144_k3 weight sweep in P78-style config (V1=17.5, I437_k3=17.5, I600=20)
print(f"\n  F144_k3 weight sweep at V1=17.5%, I437_k3=17.5%, I600=20% fixed:")
print(f"  F144_k3%    AVG     MIN   YbY")
print(f"  {'─'*55}")
f144k3_sweep = []
for wf144 in [0.35, 0.375, 0.40, 0.425, 0.45, 0.475, 0.50]:
    wrest = round(1.0 - 0.175 - 0.175 - 0.20 - wf144, 4)
    if abs(wrest) > 0.001 and wrest != 0.0:
        # small F168 added if needed
        if wrest > 0.05:
            continue
    wts = {"v1": 0.175, "idio_437_k3": 0.175, "idio_600": 0.20, "f144_k3": wf144}
    r = blend_ensemble(strats_f144k3, wts)
    f144k3_sweep.append((wf144, r))
    yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
    print(f"  {wf144*100:5.1f}%    {r['avg']:.3f}  {r['min']:.3f}  {yby_r}")

# Mini-grid: F144_k3 combinations
print(f"\n  2.5% mini-grid: V1 and F144_k3 sweep:")
print(f"       V1%  I437_k3%  I600%  F144_k3%    AVG     MIN")
print(f"  {'─'*60}")
f144k3_grid = []
for wv1 in [0.125, 0.150, 0.175, 0.20, 0.225]:
    for wi437 in [0.150, 0.175, 0.20]:
        for wi600 in [0.175, 0.20, 0.225]:
            wf144 = round(1.0 - wv1 - wi437 - wi600, 4)
            if wf144 < 0.35 or wf144 > 0.575:
                continue
            r = blend_ensemble(strats_f144k3, {"v1": wv1, "idio_437_k3": wi437, "idio_600": wi600, "f144_k3": wf144})
            f144k3_grid.append((wv1, wi437, wi600, wf144, r))

f144k3_grid.sort(key=lambda x: x[4]["avg"], reverse=True)
for wv1, wi437, wi600, wf144, r in f144k3_grid[:10]:
    print(f"    {wv1*100:4.1f}  {wi437*100:5.1f}  {wi600*100:5.1f}  {wf144*100:5.1f}     {r['avg']:.3f}  {r['min']:.3f}")

# Best MIN with AVG > 1.900 (F144_k3 ensemble)
f144k3_high_avg = sorted(
    [(x, r) for *x, r in [(wv1, wi437, wi600, wf144, r) for wv1, wi437, wi600, wf144, r in f144k3_grid]
     if r["avg"] > 1.900],
    key=lambda x: x[1]["min"], reverse=True
)
if f144k3_high_avg:
    print(f"\n  Best MIN (F144_k3, AVG > 1.900):")
    for (wv1, wi437, wi600, wf144), r in f144k3_high_avg[:5]:
        yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
        print(f"    V1={wv1*100:.1f}%, I437_k3={wi437*100:.1f}%, I600={wi600*100:.1f}%, F144_k3={wf144*100:.1f}%: AVG={r['avg']}, MIN={r['min']}  {yby_r}")

# ─── SECTION D: I437 k=4 in ensemble ─────────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION D: I437 k=4 in ensemble (P78 weights)")
print("═" * 70)

strats_k4 = {"v1": v1_data, "idio_437_k4": i437_k4, "idio_600": i600_k2,
             "f144": f144_k2, "f168": f168_k2}
r_p78_k4 = blend_ensemble(strats_k4, {"v1": 0.175, "idio_437_k4": 0.175, "idio_600": 0.20, "f144": 0.45})
print(f"\n  P78 weights + I437_k4: AVG={r_p78_k4['avg']}, MIN={r_p78_k4['min']}")
print(f"  YbY: {r_p78_k4['yby']}")

r_p73_k4 = blend_ensemble(strats_k4, {"v1": 0.15, "idio_437_k4": 0.15, "idio_600": 0.20, "f144": 0.45, "f168": 0.05})
print(f"  P73 weights + I437_k4: AVG={r_p73_k4['avg']}, MIN={r_p73_k4['min']}")
print(f"  YbY: {r_p73_k4['yby']}")

# Combined: F144_k3 + I437_k4
strats_both_k3k4 = {"v1": v1_data, "idio_437_k4": i437_k4, "idio_600": i600_k2,
                    "f144_k3": f144_k3, "f168": f168_k2}
r_both_k3k4 = blend_ensemble(strats_both_k3k4, {"v1": 0.175, "idio_437_k4": 0.175, "idio_600": 0.20, "f144_k3": 0.45})
print(f"\n  P78 weights + I437_k4 + F144_k3: AVG={r_both_k3k4['avg']}, MIN={r_both_k3k4['min']}")
print(f"  YbY: {r_both_k3k4['yby']}")

# ─── SECTION E: Fine 1.25% grid around P78 champion ─────────────────────────
print("\n" + "═" * 70)
print("SECTION E: Fine 1.25% grid around P78 champion")
print("  Center: V1=17.5%, I437_k3=17.5%, I600=20%, F144=45%, F168=0%")
print("  Range: ±3.75% around each weight (3 steps each side)")
print("═" * 70)

strats_e = {"v1": v1_data, "idio_437_k3": i437_k3, "idio_600": i600_k2,
            "f144": f144_k2, "f168": f168_k2}

# Fine grid: 1.25% steps
v1_fine   = pct_range(1250, 2250, 125)    # 12.5% to 21.25%
i437_fine = pct_range(1250, 2250, 125)    # 12.5% to 21.25%
i600_fine = pct_range(1500, 2500, 125)    # 15% to 23.75%

fine_grid = []
total_fine = 0
for wv1 in v1_fine:
    for wi437 in i437_fine:
        for wi600 in i600_fine:
            wf144 = round(1.0 - wv1 - wi437 - wi600, 4)
            if wf144 < 0.37 or wf144 > 0.55:
                continue
            wts = {"v1": wv1, "idio_437_k3": wi437, "idio_600": wi600, "f144": wf144}
            r = blend_ensemble(strats_e, wts)
            fine_grid.append((wv1, wi437, wi600, wf144, r))
            total_fine += 1

print(f"\n  Tested {total_fine} fine-grid configurations")
fine_grid.sort(key=lambda x: x[4]["avg"], reverse=True)

print(f"\n  Top 15 by AVG (1.25% grid):")
print(f"       V1%  I437_k3%  I600%  F144%    AVG     MIN")
print(f"  {'─'*55}")
for wv1, wi437, wi600, wf144, r in fine_grid[:15]:
    print(f"    {wv1*100:5.2f}  {wi437*100:5.2f}  {wi600*100:5.2f}  {wf144*100:5.2f}  {r['avg']:.3f}  {r['min']:.3f}")

# Best MIN with AVG > 1.910
fine_high_avg = sorted(
    [(wv1, wi437, wi600, wf144, r) for wv1, wi437, wi600, wf144, r in fine_grid if r["avg"] > 1.910],
    key=lambda x: x[4]["min"], reverse=True
)
if fine_high_avg:
    print(f"\n  Best MIN where AVG > 1.910 ({len(fine_high_avg)} configs):")
    print(f"       V1%  I437_k3%  I600%  F144%    AVG     MIN   YbY")
    print(f"  {'─'*75}")
    for wv1, wi437, wi600, wf144, r in fine_high_avg[:10]:
        yby_r = [round(v, 3) if v is not None else None for v in r["yby"]]
        flag = " ★" if r["avg"] > P78_BALANCED["avg"] and r["min"] > P78_BALANCED["min"] else ""
        print(f"    {wv1*100:5.2f}  {wi437*100:5.2f}  {wi600*100:5.2f}  {wf144*100:5.2f}  {r['avg']:.3f}  {r['min']:.3f}  {yby_r}{flag}")

# ─── SECTION F: Phase 80 Summary ──────────────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION F: Phase 80 Summary")
print("═" * 70)

print(f"\n  Champion landscape entering Phase 80:")
print(f"    Phase 78 balanced: AVG=1.919, MIN=1.206")
print(f"    Phase 79 AVG-max:  AVG=1.934, MIN=1.098")

print(f"\n  Section A (F144 k=3 standalone):")
print(f"    F144 k=2: AVG={f144_k2['_avg']}, MIN={f144_k2['_min']}")
print(f"    F144 k=3: AVG={f144_k3['_avg']}, MIN={f144_k3['_min']}")
delta_avg = round(f144_k3['_avg'] - f144_k2['_avg'], 3)
print(f"    Delta: {delta_avg:+.3f} AVG, {round(f144_k3['_min'] - f144_k2['_min'], 3):+.3f} MIN")

print(f"\n  Section B (I437 k=4 standalone):")
print(f"    I437 k=3: AVG={i437_k3['_avg']}, MIN={i437_k3['_min']}")
print(f"    I437 k=4: AVG={i437_k4['_avg']}, MIN={i437_k4['_min']}")

print(f"\n  Section C (F144_k3 in ensemble):")
print(f"    P78 + F144_k3: AVG={r_p78_f144k3['avg']}, MIN={r_p78_f144k3['min']}")
if f144k3_grid:
    best_f144k3 = f144k3_grid[0]
    print(f"    Best F144_k3 grid: V1={best_f144k3[0]*100:.1f}%, I437_k3={best_f144k3[1]*100:.1f}%, I600={best_f144k3[2]*100:.1f}%, F144_k3={best_f144k3[3]*100:.1f}%: AVG={best_f144k3[4]['avg']}, MIN={best_f144k3[4]['min']}")

print(f"\n  Section D (I437 k=4 in ensemble):")
print(f"    P78 + I437_k4: AVG={r_p78_k4['avg']}, MIN={r_p78_k4['min']}")
print(f"    P78 + I437_k4 + F144_k3: AVG={r_both_k3k4['avg']}, MIN={r_both_k3k4['min']}")

print(f"\n  Section E (1.25% fine grid):")
if fine_grid:
    top_fine = fine_grid[0]
    print(f"    Best AVG: V1={top_fine[0]*100:.2f}%, I437_k3={top_fine[1]*100:.2f}%, I600={top_fine[2]*100:.2f}%, F144={top_fine[3]*100:.2f}%: AVG={top_fine[4]['avg']}, MIN={top_fine[4]['min']}")
    if fine_high_avg:
        top_min_fine = fine_high_avg[0]
        print(f"    Best MIN (AVG>1.910): V1={top_min_fine[0]*100:.2f}%, I437_k3={top_min_fine[1]*100:.2f}%, I600={top_min_fine[2]*100:.2f}%, F144={top_min_fine[3]*100:.2f}%: AVG={top_min_fine[4]['avg']}, MIN={top_min_fine[4]['min']}")

# Save new champions
saved = []

# Check F144_k3 combinations
best_f144k3_combo = None
if f144k3_grid:
    for wv1, wi437, wi600, wf144, r in f144k3_grid:
        if r["avg"] > P78_BALANCED["avg"] and r["min"] > P78_BALANCED["min"]:
            if best_f144k3_combo is None or (r["avg"] + r["min"] > best_f144k3_combo[4]["avg"] + best_f144k3_combo[4]["min"]):
                best_f144k3_combo = (wv1, wi437, wi600, wf144, r)

if best_f144k3_combo:
    wv1, wi437, wi600, wf144, r = best_f144k3_combo
    print(f"\n  ★ NEW CHAMPION (F144_k3): AVG={r['avg']}, MIN={r['min']}")
    print(f"    V1={wv1*100:.1f}%, I437_k3={wi437*100:.1f}%, I600={wi600*100:.1f}%, F144_k3={wf144*100:.1f}%")
    save_champion(
        "ensemble_p80_f144k3_balanced",
        f"Phase 80 balanced champion: V1({wv1*100:.1f}%)+I437_bw168_k3({wi437*100:.1f}%)+I600_bw168_k2({wi600*100:.1f}%)+F144_k3({wf144*100:.1f}%). AVG={r['avg']}, MIN={r['min']}",
        [
            {"name": "nexus_alpha_v1", "params": V1_PARAMS, "weight": wv1},
            {"name": "idio_momentum_alpha", "params": make_idio_params(437, 168, k=3), "weight": wi437},
            {"name": "idio_momentum_alpha", "params": make_idio_params(600, 168, k=2), "weight": wi600},
            {"name": "funding_momentum_alpha", "params": FUND_144_K3, "weight": wf144},
        ],
    )
    saved.append("F144_k3 champion")

# Check fine grid
if fine_high_avg:
    wv1, wi437, wi600, wf144, r = fine_high_avg[0]
    if r["avg"] > P78_BALANCED["avg"] and r["min"] > P78_BALANCED["min"]:
        print(f"\n  ★ NEW CHAMPION (fine grid): AVG={r['avg']}, MIN={r['min']}")
        save_champion(
            "ensemble_p80_finegrid_balanced",
            f"Phase 80 fine-grid champion: V1({wv1*100:.2f}%)+I437_bw168_k3({wi437*100:.2f}%)+I600({wi600*100:.2f}%)+F144({wf144*100:.2f}%). AVG={r['avg']}, MIN={r['min']}",
            [
                {"name": "nexus_alpha_v1", "params": V1_PARAMS, "weight": wv1},
                {"name": "idio_momentum_alpha", "params": make_idio_params(437, 168, k=3), "weight": wi437},
                {"name": "idio_momentum_alpha", "params": make_idio_params(600, 168, k=2), "weight": wi600},
                {"name": "funding_momentum_alpha", "params": make_fund_params(144, k=2), "weight": wf144},
            ],
        )
        saved.append("fine grid champion")

if not saved:
    print(f"\n  Phase 78 balanced champion (1.919/1.206) remains optimal.")
    print(f"  No improvement found through k=3 funding, k=4 idio, or finer grid.")

print("\n" + "=" * 70)
print("PHASE 80 COMPLETE")
print("=" * 70)
