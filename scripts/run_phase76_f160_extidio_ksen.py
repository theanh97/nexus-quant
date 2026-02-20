#!/usr/bin/env python3
"""
Phase 76: F160 Funding + Extended Idio Lookback + k_per_side Sensitivity

Current champion landscape:
  AVG-max:  Phase 75 → AVG=1.934, MIN=0.972  (V1+I437_bw336+I600_bw168+F144+F168)
  Balanced: Phase 74 → AVG=1.902, MIN=1.199  (V1+I437_bw168+I600_bw168+F144 only)
  Sweet spot: Phase 73 → AVG=1.909, MIN=1.170 (V1+I437_bw168+I600_bw168+F144+F168)

Untested directions:
  A. F160 as funding signal: lb=160h (n=20) had AVG=1.252, 2nd best after F144
     - Replace F168 with F160 (F168 2025=0.571 vs F160 2025=?)
     - Or add F160 alongside F144 (like dual funding)
  B. Extended idio lookbacks (800, 1000, 1200h) with bw=168
     - lb=730_bw168 AVG=0.942 from Phase 73 — declines beyond 600h generally
     - But worth testing 800h+ to confirm no secondary peak
  C. k_per_side sensitivity: k=1, 2, 3 for idio strategies
     - k=2 is current default (top 2 longs, bottom 2 shorts out of 10)
     - k=3: more diversified, may improve MIN; k=1: more concentrated
  D. F144 + F160 dual funding (replacing F144+F168)
     - F160 might complement F144 better than F168
"""

import json
import math
import statistics
import subprocess
import sys
import copy
from pathlib import Path

OUT_DIR = "artifacts/phase76"
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

FUND_160_PARAMS = {
    "k_per_side": 2, "funding_lookback_bars": 160, "direction": "contrarian",
    "vol_lookback_bars": 168, "target_gross_leverage": 0.25, "rebalance_interval_bars": 24,
}

FUND_168_PARAMS = {
    "k_per_side": 2, "funding_lookback_bars": 168, "direction": "contrarian",
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
    path = f"/tmp/phase76_{run_name}_{year}.json"
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
print("PHASE 76: F160 + Extended Idio + k_per_side Sensitivity")
print("=" * 70)

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

P74_BALANCED = {"avg": 1.902, "min": 1.199}
P73_BALANCED = {"avg": 1.909, "min": 1.170}
P75_AVG_MAX  = {"avg": 1.934, "min": 0.972}

# ─── SECTION A: Run new signals ───────────────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION A: Running new signal variants")
print("═" * 70)

print("  Running F160 (funding lb=160h)...", flush=True)
f160_data = run_strategy("fund_160_ref", "funding_momentum_alpha", FUND_160_PARAMS)

print("  Running idio_800_bw168 (extended lookback)...", flush=True)
i800_bw168 = run_strategy("idio_lb800_bw168", "idio_momentum_alpha", make_idio_params(800, 168))

print("  Running idio_437_k1 (k_per_side=1)...", flush=True)
i437_k1 = run_strategy("idio_lb437_bw168_k1", "idio_momentum_alpha", make_idio_params(437, 168, k=1))

print("  Running idio_437_k3 (k_per_side=3)...", flush=True)
i437_k3 = run_strategy("idio_lb437_bw168_k3", "idio_momentum_alpha", make_idio_params(437, 168, k=3))

# ─── SECTION B: Reference strategies ─────────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION B: Reference strategies")
print("═" * 70)
print("  Running V1 reference...", flush=True)
v1_data = run_strategy("v1_ref", "nexus_alpha_v1", V1_PARAMS)
print("  Running idio_437_bw168...", flush=True)
i437_bw168 = run_strategy("idio_lb437_bw168", "idio_momentum_alpha", make_idio_params(437, 168))
print("  Running idio_600_bw168...", flush=True)
i600_bw168 = run_strategy("idio_lb600_bw168", "idio_momentum_alpha", make_idio_params(600, 168))
print("  Running fund_144 reference...", flush=True)
f144_data = run_strategy("fund_144_ref", "funding_momentum_alpha", FUND_144_PARAMS)
print("  Running fund_168 reference...", flush=True)
f168_data = run_strategy("fund_168_ref", "funding_momentum_alpha", FUND_168_PARAMS)

# Phase 73 baseline
strats_base = {
    "v1": v1_data, "idio_437": i437_bw168, "idio_600": i600_bw168,
    "f144": f144_data, "f168": f168_data,
}
p73_baseline = blend_ensemble(strats_base, {"v1": 0.15, "idio_437": 0.15, "idio_600": 0.20, "f144": 0.45, "f168": 0.05})
p74_baseline = blend_ensemble(strats_base, {"v1": 0.175, "idio_437": 0.125, "idio_600": 0.225, "f144": 0.475, "f168": 0.00})
print(f"\n  Phase 73 champion baseline: AVG={p73_baseline['avg']}, MIN={p73_baseline['min']}")
print(f"  Phase 74 champion baseline: AVG={p74_baseline['avg']}, MIN={p74_baseline['min']}")

# ─── SECTION C: F160 in ensemble ─────────────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION C: F160 in ensemble")
print(f"  F160 standalone: AVG={f160_data['_avg']}, MIN={f160_data['_min']}")
print(f"  YbY: {[round(f160_data.get(y, {}).get('sharpe', 0.0), 3) for y in YEARS]}")
print("═" * 70)

# Replace F168 with F160 in Phase 73 champion
strats_f160 = {"v1": v1_data, "idio_437": i437_bw168, "idio_600": i600_bw168,
               "f144": f144_data, "f160": f160_data}
r_f160_replace = blend_ensemble(strats_f160, {"v1": 0.15, "idio_437": 0.15, "idio_600": 0.20, "f144": 0.45, "f160": 0.05})
print(f"\n  V1(15)+I437_bw168(15)+I600_bw168(20)+F144(45)+F160(5): AVG={r_f160_replace['avg']}, MIN={r_f160_replace['min']}")
print(f"  YbY: {r_f160_replace['yby']}")

# Replace F144 with F160 + F144 dual funding
strats_dual_f = {"v1": v1_data, "idio_437": i437_bw168, "idio_600": i600_bw168,
                 "f144": f144_data, "f160": f160_data}
r_dual_f = blend_ensemble(strats_dual_f, {"v1": 0.15, "idio_437": 0.15, "idio_600": 0.20, "f144": 0.25, "f160": 0.25})
print(f"\n  V1(15)+I437(15)+I600(20)+F144(25)+F160(25): AVG={r_dual_f['avg']}, MIN={r_dual_f['min']}")

# Grid over F144/F160 split
print(f"\n  F144/F160 split grid (Phase 73 non-funding weights fixed):")
f_split_results = []
for f_total in [40, 45, 50]:
    for frac_144 in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        w_f144 = f_total * frac_144 / 100
        w_f160 = f_total * (1 - frac_144) / 100
        wts_f = {"v1": 0.15, "idio_437": 0.15, "idio_600": 0.20, "f144": w_f144, "f160": w_f160}
        total_so_far = sum(wts_f.values())
        # Adjust nothing else changes
        if abs(total_so_far - 1.0) > 0.001:
            # Need to fit within 1.0 total
            remainder = 1.0 - 0.15 - 0.15 - 0.20 - f_total / 100
            continue
        r_f = blend_ensemble(strats_f160, wts_f)
        f_split_results.append((w_f144, w_f160, r_f))

f_split_results.sort(key=lambda x: x[2]["avg"], reverse=True)
for w_f144, w_f160, r in f_split_results[:5]:
    print(f"    F144={w_f144*100:.0f}%,F160={w_f160*100:.0f}%: AVG={r['avg']}, MIN={r['min']}")

# ─── SECTION D: Extended idio lookback ───────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION D: Extended idio lookback (800h) analysis")
print(f"  I800_bw168 standalone: AVG={i800_bw168['_avg']}, MIN={i800_bw168['_min']}")
print(f"  YbY: {[round(i800_bw168.get(y, {}).get('sharpe', 0.0), 3) for y in YEARS]}")
print("═" * 70)

# Extended idio in ensemble
strats_800 = dict(strats_base)
strats_800["idio_800"] = i800_bw168
r_800_replace = blend_ensemble(strats_800, {"v1": 0.15, "idio_437": 0.15, "idio_800": 0.20, "f144": 0.45, "f168": 0.05})
print(f"\n  Replace I600 with I800: AVG={r_800_replace['avg']}, MIN={r_800_replace['min']}")
r_800_add = blend_ensemble(strats_800, {"v1": 0.15, "idio_437": 0.10, "idio_600": 0.10, "idio_800": 0.10, "f144": 0.45, "f168": 0.10})
print(f"  Triple-idio (437+600+800): AVG={r_800_add['avg']}, MIN={r_800_add['min']}")

# ─── SECTION E: k_per_side sensitivity ───────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION E: k_per_side sensitivity for idio signals")
print(f"  I437_bw168_k1: AVG={i437_k1['_avg']}, MIN={i437_k1['_min']}")
print(f"    YbY: {[round(i437_k1.get(y, {}).get('sharpe', 0.0), 3) for y in YEARS]}")
print(f"  I437_bw168_k2 (current): AVG={i437_bw168['_avg']}, MIN={i437_bw168['_min']}")
print(f"    YbY: {[round(i437_bw168.get(y, {}).get('sharpe', 0.0), 3) for y in YEARS]}")
print(f"  I437_bw168_k3: AVG={i437_k3['_avg']}, MIN={i437_k3['_min']}")
print(f"    YbY: {[round(i437_k3.get(y, {}).get('sharpe', 0.0), 3) for y in YEARS]}")
print("═" * 70)

# k=3 for idio_437 in ensemble
strats_k3 = {"v1": v1_data, "idio_437_k3": i437_k3, "idio_600": i600_bw168,
             "f144": f144_data, "f168": f168_data}
r_k3_p73 = blend_ensemble(strats_k3, {"v1": 0.15, "idio_437_k3": 0.15, "idio_600": 0.20, "f144": 0.45, "f168": 0.05})
print(f"\n  k=3 for idio_437 (Phase 73 weights): AVG={r_k3_p73['avg']}, MIN={r_k3_p73['min']}")
r_k3_p74 = blend_ensemble(strats_k3, {"v1": 0.175, "idio_437_k3": 0.125, "idio_600": 0.225, "f144": 0.475})
print(f"  k=3 for idio_437 (Phase 74 weights): AVG={r_k3_p74['avg']}, MIN={r_k3_p74['min']}")

# k=3 for idio_600 too
i600_k3 = run_strategy("idio_lb600_bw168_k3", "idio_momentum_alpha", make_idio_params(600, 168, k=3))
print(f"\n  I600_bw168_k3: AVG={i600_k3['_avg']}, MIN={i600_k3['_min']}")
strats_k3_both = {"v1": v1_data, "idio_437_k3": i437_k3, "idio_600_k3": i600_k3,
                  "f144": f144_data, "f168": f168_data}
r_k3_both = blend_ensemble(strats_k3_both, {"v1": 0.15, "idio_437_k3": 0.15, "idio_600_k3": 0.20, "f144": 0.45, "f168": 0.05})
print(f"  k=3 for BOTH idio (Phase 73 weights): AVG={r_k3_both['avg']}, MIN={r_k3_both['min']}")

# ─── SECTION F: Grand Summary ─────────────────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION F: Phase 76 Summary")
print("═" * 70)

print(f"\n  Current champion landscape:")
print(f"    AVG-max:  Phase 75 → AVG=1.934, MIN=0.972")
print(f"    Balanced: Phase 74 → AVG=1.902, MIN=1.199")
print(f"    Sweet spot: Phase 73 → AVG=1.909, MIN=1.170")

print(f"\n  F160 standalone: AVG={f160_data['_avg']:.3f}, MIN={f160_data['_min']:.3f}")
print(f"  I800_bw168 standalone: AVG={i800_bw168['_avg']:.3f}, MIN={i800_bw168['_min']:.3f}")
print(f"  k_per_side=1: AVG={i437_k1['_avg']:.3f}, k=2: {i437_bw168['_avg']:.3f}, k=3: {i437_k3['_avg']:.3f}")

print(f"\n  Ensemble tests:")
print(f"    F160 replacing F168 (Ph73 weights): AVG={r_f160_replace['avg']}, MIN={r_f160_replace['min']}")
print(f"    I800 replacing I600 (Ph73 weights): AVG={r_800_replace['avg']}, MIN={r_800_replace['min']}")
print(f"    k=3 for I437 (Ph73 weights): AVG={r_k3_p73['avg']}, MIN={r_k3_p73['min']}")
print(f"    k=3 for both (Ph73 weights): AVG={r_k3_both['avg']}, MIN={r_k3_both['min']}")

# Save any new champion
new_champs = []
for desc, r, strats_d, wts_d, detail in [
    ("F160 replacing F168", r_f160_replace, strats_f160,
     {"v1": 0.15, "idio_437": 0.15, "idio_600": 0.20, "f144": 0.45, "f160": 0.05},
     "V1(15)+I437_bw168(15)+I600_bw168(20)+F144(45)+F160(5)"),
    ("k=3 both (Ph73)", r_k3_both, strats_k3_both,
     {"v1": 0.15, "idio_437_k3": 0.15, "idio_600_k3": 0.20, "f144": 0.45, "f168": 0.05},
     "V1(15)+I437_bw168_k3(15)+I600_bw168_k3(20)+F144(45)+F168(5)"),
]:
    if r["avg"] > P73_BALANCED["avg"] and r["min"] > P74_BALANCED["min"]:
        print(f"\n  ★ NEW CHAMPION: {desc}! AVG={r['avg']}, MIN={r['min']}")
    elif r["avg"] > P75_AVG_MAX["avg"]:
        print(f"\n  ★ NEW AVG-MAX: {desc}! AVG={r['avg']}")

print("\n" + "=" * 70)
print("PHASE 76 COMPLETE")
print("=" * 70)
