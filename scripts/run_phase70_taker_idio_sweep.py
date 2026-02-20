#!/usr/bin/env python3
"""
Phase 70: TakerBuyAlpha sweep + IdioMomentum lookback sweep

Phase 68 champion: V1(15%)+Idio(30%)+F144(40%)+F168(15%) AVG=1.844, MIN=0.996

Motivation:
  - funding_vol_alpha failed in Phase 69 (avg=-0.004 across all lb/dir)
  - 144h confirmed as global peak for funding contrarian lookback
  - Need NEW orthogonal signal sources to push ensemble beyond 1.844
  - Volume-flow (TakerBuyAlpha) and idio lookback diversity are most promising

Goals:
  A. TakerBuyAlpha standalone sweep: lb=[24, 48, 96, 168]
     Hypothesis: taker buy ratio is orthogonal to funding (market structure)
     and to idio (BTC-beta-removed price momentum)
  B. IdioMomentum lookback sweep: lb=[168, 288, 437, 600, 730]
     Does 437 remain optimal? Longer lookback may capture annual momentum effects
  C. Ensemble grid: best new signal + Phase 68 champion components
"""

import json
import math
import statistics
import subprocess
import sys
import copy
from pathlib import Path

OUT_DIR = "artifacts/phase70"
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
    path = f"/tmp/phase70_{run_name}_{year}.json"
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
    year_results["_avg"] = avg
    year_results["_min"] = mn
    year_results["_pos"] = pos
    yby = [round(year_results.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]
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
print("PHASE 70: TakerBuyAlpha sweep + IdioMomentum lookback sweep")
print("=" * 70)

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# ─── SECTION A: TakerBuyAlpha standalone sweep ───────────────────────────────
print("\n" + "═" * 70)
print("SECTION A: TakerBuyAlpha standalone — lookback sweep")
print("Signal: cross-sectional ranking by taker buy ratio (volume-flow based)")
print("═" * 70)

TAKER_LOOKBACKS = [24, 48, 96, 168]
taker_results = {}

for lb in TAKER_LOOKBACKS:
    label = f"taker_lb{lb}"
    params = {
        "k_per_side": 2, "ratio_lookback_bars": lb,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.30,
        "rebalance_interval_bars": 24,
    }
    data = run_strategy(label, "taker_buy_alpha", params)
    taker_results[lb] = data

print("\n── TakerBuyAlpha sweep summary ─────────────────────────────────────────")
print(f"{'LB':>6} {'AVG':>8} {'MIN':>8} {'Pos':>6}")
print("─" * 35)
for lb in TAKER_LOOKBACKS:
    d = taker_results[lb]
    print(f"{lb:>6}h {d['_avg']:>8.3f} {d['_min']:>8.3f} {d['_pos']:>6}/5")

best_taker_lb = max(TAKER_LOOKBACKS, key=lambda lb: taker_results[lb]["_avg"])
best_taker_data = taker_results[best_taker_lb]
best_taker_avg = best_taker_data["_avg"]
print(f"\nBest TakerBuy: lb={best_taker_lb}h, AVG={best_taker_avg:.3f}")

# ─── SECTION B: IdioMomentum lookback sweep ──────────────────────────────────
print("\n" + "═" * 70)
print("SECTION B: IdioMomentum lookback sweep")
print("Does 437h remain optimal? Testing range 168-730h")
print("═" * 70)

IDIO_LOOKBACKS = [168, 288, 437, 600, 730]
idio_results = {}

for lb in IDIO_LOOKBACKS:
    label = f"idio_lb{lb}"
    params = {
        "k_per_side": 2, "lookback_bars": lb, "beta_window_bars": 72,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.30,
        "rebalance_interval_bars": 48,
    }
    data = run_strategy(label, "idio_momentum_alpha", params)
    idio_results[lb] = data

print("\n── IdioMomentum lookback sweep summary ─────────────────────────────────")
print(f"{'LB':>6} {'AVG':>8} {'MIN':>8} {'Pos':>6} {'YbY':>40}")
print("─" * 65)
for lb in IDIO_LOOKBACKS:
    d = idio_results[lb]
    yby = [round(d.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]
    marker = " ← BEST" if lb == max(IDIO_LOOKBACKS, key=lambda l: idio_results[l]["_avg"]) else ""
    print(f"{lb:>6}h {d['_avg']:>8.3f} {d['_min']:>8.3f} {d['_pos']:>6}/5  {str(yby):<40}{marker}")

best_idio_lb = max(IDIO_LOOKBACKS, key=lambda lb: idio_results[lb]["_avg"])
best_idio_data = idio_results[best_idio_lb]
best_idio_avg = best_idio_data["_avg"]
print(f"\nBest Idio lookback: lb={best_idio_lb}h, AVG={best_idio_avg:.3f}")

# ─── SECTION C: Reference strategies for blending ────────────────────────────
print("\n" + "═" * 70)
print("SECTION C: Reference strategies for ensemble blending")
print("═" * 70)

print("  Running V1 reference...", flush=True)
v1_data = run_strategy("v1_ref", "nexus_alpha_v1", V1_PARAMS)
print("  Running idio_437 reference...", flush=True)
idio_437_data = run_strategy("idio_437_ref", "idio_momentum_alpha", IDIO_437_PARAMS)
print("  Running fund_144 reference...", flush=True)
f144_data = run_strategy("fund_144_ref", "funding_momentum_alpha", FUND_144_PARAMS)
print("  Running fund_168 reference...", flush=True)
f168_data = run_strategy("fund_168_ref", "funding_momentum_alpha", FUND_168_PARAMS)

# ─── SECTION D: Ensemble integration ─────────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION D: Ensemble integration")
print("Phase 68 champion: V1(15)+Idio(30)+F144(40)+F168(15) AVG=1.844")
print("═" * 70)

base_strats = {
    "v1": v1_data,
    "idio": idio_437_data,
    "f144": f144_data,
    "f168": f168_data,
}

p68_champ = blend_ensemble(base_strats, {"v1": 0.15, "idio": 0.30, "f144": 0.40, "f168": 0.15})
print(f"\n  Phase 68 champion (baseline): AVG={p68_champ['avg']}, MIN={p68_champ['min']}, {p68_champ['pos']}/5")

# D1: Replace idio_437 with best new idio lookback (if better)
if best_idio_lb != 437:
    print(f"\n── D1: Idio_{best_idio_lb} replacing Idio_437 ─────────────────────────────")
    strats_new_idio = dict(base_strats)
    strats_new_idio["idio"] = best_idio_data
    r = blend_ensemble(strats_new_idio, {"v1": 0.15, "idio": 0.30, "f144": 0.40, "f168": 0.15})
    print(f"  V1(15)+Idio_{best_idio_lb}(30)+F144(40)+F168(15): AVG={r['avg']}, MIN={r['min']}, {r['pos']}/5")

    # Also test dual-idio: 437 + best_new
    strats_dual_idio = dict(base_strats)
    strats_dual_idio[f"idio_{best_idio_lb}"] = best_idio_data
    r2 = blend_ensemble(strats_dual_idio, {"v1": 0.15, "idio": 0.20, f"idio_{best_idio_lb}": 0.15, "f144": 0.35, "f168": 0.15})
    print(f"  V1(15)+Idio_437(20)+Idio_{best_idio_lb}(15)+F144(35)+F168(15): AVG={r2['avg']}, MIN={r2['min']}, {r2['pos']}/5")

# D2: Add TakerBuyAlpha if competitive
TAKER_THRESHOLD = 0.5
if best_taker_avg > TAKER_THRESHOLD:
    print(f"\n── D2: TakerBuyAlpha ensemble integration (AVG={best_taker_avg:.3f}) ─────────")
    strats_with_taker = dict(base_strats)
    strats_with_taker["taker"] = best_taker_data

    taker_experiments = [
        ("v1=15,idio=25,f144=35,f168=15,taker=10", {"v1": 0.15, "idio": 0.25, "f144": 0.35, "f168": 0.15, "taker": 0.10}),
        ("v1=15,idio=20,f144=35,f168=15,taker=15", {"v1": 0.15, "idio": 0.20, "f144": 0.35, "f168": 0.15, "taker": 0.15}),
        ("v1=10,idio=30,f144=35,f168=10,taker=15", {"v1": 0.10, "idio": 0.30, "f144": 0.35, "f168": 0.10, "taker": 0.15}),
        ("v1=15,idio=25,f144=30,f168=10,taker=20", {"v1": 0.15, "idio": 0.25, "f144": 0.30, "f168": 0.10, "taker": 0.20}),
    ]

    for desc, wts in taker_experiments:
        r = blend_ensemble(strats_with_taker, wts)
        print(f"    {desc}: AVG={r['avg']}, MIN={r['min']}, {r['pos']}/5")

    if best_idio_lb != 437:
        print(f"\n  Also test taker + new idio_{best_idio_lb}:")
        strats_new_idio_taker = dict(strats_with_taker)
        strats_new_idio_taker["idio"] = best_idio_data
        r = blend_ensemble(strats_new_idio_taker, {"v1": 0.15, "idio": 0.25, "f144": 0.30, "f168": 0.15, "taker": 0.15})
        print(f"    V1(15)+Idio_{best_idio_lb}(25)+F144(30)+F168(15)+Taker(15): AVG={r['avg']}, MIN={r['min']}, {r['pos']}/5")
else:
    print(f"\n── D2: TakerBuyAlpha not competitive (AVG={best_taker_avg:.3f} < {TAKER_THRESHOLD}) ──")

# D3: Full 5-way grid if both new signals are competitive
if best_taker_avg > TAKER_THRESHOLD and best_idio_lb != 437:
    print(f"\n── D3: Full 5-way grid (V1+NewIdio+F144+F168+Taker) ──────────────────")
    strats_full = dict(base_strats)
    strats_full["idio"] = best_idio_data  # replace with best idio
    strats_full["taker"] = best_taker_data

    best_full = None
    best_full_avg = 0.0
    for w_v1 in [10, 15]:
        for w_idio in [20, 25, 30]:
            for w_f144 in [25, 30, 35]:
                for w_f168 in [10, 15]:
                    for w_taker in [10, 15]:
                        total = w_v1 + w_idio + w_f144 + w_f168 + w_taker
                        if total != 100:
                            continue
                        wts = {"v1": w_v1/100, "idio": w_idio/100, "f144": w_f144/100, "f168": w_f168/100, "taker": w_taker/100}
                        r = blend_ensemble(strats_full, wts)
                        if r["avg"] > best_full_avg:
                            best_full_avg = r["avg"]
                            best_full = (wts, r)

    if best_full:
        wts, r = best_full
        print(f"  Best 5-way: V1={int(wts['v1']*100)}/Idio_{best_idio_lb}={int(wts['idio']*100)}/F144={int(wts['f144']*100)}/F168={int(wts['f168']*100)}/Taker={int(wts['taker']*100)}")
        print(f"  AVG={r['avg']}, MIN={r['min']}, {r['pos']}/5, YbY={r['yby']}")

# ─── SECTION E: Correlation Analysis ─────────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION E: Correlation Analysis")
print("═" * 70)

all_returns = {
    "v1":   get_all_returns(v1_data),
    "idio": get_all_returns(idio_437_data),
    "f144": get_all_returns(f144_data),
    "f168": get_all_returns(f168_data),
    "taker_best": get_all_returns(best_taker_data),
}
if best_idio_lb != 437:
    all_returns[f"idio_{best_idio_lb}"] = get_all_returns(best_idio_data)

keys = list(all_returns.keys())
print(f"\n{'':18}", end="")
for k in keys:
    print(f"{k:>16}", end="")
print()
for k1 in keys:
    print(f"{k1:18}", end="")
    for k2 in keys:
        c = pearson_corr(all_returns[k1], all_returns[k2])
        print(f"{c:>16.4f}", end="")
    print()

# ─── SECTION F: Grand summary ─────────────────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION F: Phase 70 Summary")
print("═" * 70)

print(f"\nPhase 68 champion (baseline): V1(15)+Idio(30)+F144(40)+F168(15)")
print(f"  AVG={p68_champ['avg']}, MIN={p68_champ['min']}, {p68_champ['pos']}/5")

print(f"\nTakerBuyAlpha sweep:")
for lb in TAKER_LOOKBACKS:
    d = taker_results[lb]
    yby = [round(d.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]
    print(f"  lb={lb:4}h: AVG={d['_avg']:.3f}, MIN={d['_min']:.3f}, YbY={yby}")

print(f"\nIdioMomentum lookback sweep:")
for lb in IDIO_LOOKBACKS:
    d = idio_results[lb]
    yby = [round(d.get(y, {}).get("sharpe", 0.0), 3) for y in YEARS]
    marker = " ← BEST" if lb == best_idio_lb else ""
    print(f"  lb={lb:4}h: AVG={d['_avg']:.3f}, MIN={d['_min']:.3f}, YbY={yby}{marker}")

print("\n" + "=" * 70)
print("PHASE 70 COMPLETE")
print("=" * 70)
