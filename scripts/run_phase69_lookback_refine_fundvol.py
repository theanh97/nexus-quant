#!/usr/bin/env python3
"""
Phase 69: Canonical Lookback Refinement + Funding Volatility Signal

Phase 68 champion: V1(15%)+Idio(30%)+F144(40%)+F168(15%) AVG=1.844, MIN=0.996
Phase 68 balanced: V1(20%)+SR(10%)+Idio(20%)+F144(30%)+F168(20%) AVG=1.826, MIN=1.112

Observations:
  - funding_momentum_alpha samples at 8h intervals: n_samples = lookback // 8
  - Phase 67 tested: 96, 108, 120, 132, 144, 156, 168h (every 12h = 1.5 steps)
  - Missing canonical 8h multiples: 104, 112, 128, 136, 152, 160h
  - fund_144 peak may not be the global maximum — need full 8h-step curve
  - funding_vol_alpha = std dev of funding rates (new orthogonal signal)

Goals:
  A. Fill in missing canonical lookbacks (104, 112, 128, 136, 152, 160h)
     → Complete the n_samples vs AVG curve to find true peak
  B. FundingVolAlphaStrategy standalone: contrarian vs momentum, lb=[96,144,168]
     → Test if funding volatility is a useful orthogonal signal
  C. Best new funding lookback in ensemble vs Phase 68 champion
  D. Fund_vol ensemble integration (if competitive)
     → V1+Idio+F_best+FundVol vs current champion
"""

import json
import math
import statistics
import subprocess
import sys
import copy
from pathlib import Path

OUT_DIR = "artifacts/phase69"
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

SR_437_PARAMS = {
    "k_per_side": 2, "lookback_bars": 437,
    "vol_lookback_bars": 168, "target_gross_leverage": 0.35, "rebalance_interval_bars": 48,
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
    path = f"/tmp/phase69_{run_name}_{year}.json"
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
    print(f"  → AVG={avg}, MIN={mn}, {pos}/5 positive", flush=True)
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
    """Fast blend using pre-extracted lists to minimize per-element overhead."""
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
        blended = [
            sum(w * arr[i] for w, arr in pairs)
            for i in range(min_len)
        ]
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


def save_champion_config(config_name: str, comment: str, weights: dict,
                          extra_strats: dict = None) -> None:
    """Save a champion config. extra_strats: {key: (sname, sparams)} for non-standard strats."""
    strat_cfgs = {
        "v1": ("nexus_alpha_v1", V1_PARAMS),
        "sr": ("sharpe_ratio_alpha", SR_437_PARAMS),
        "idio": ("idio_momentum_alpha", IDIO_437_PARAMS),
        "f144": ("funding_momentum_alpha", FUND_144_PARAMS),
        "f168": ("funding_momentum_alpha", FUND_168_PARAMS),
    }
    if extra_strats:
        strat_cfgs.update(extra_strats)

    sub_strats = []
    for key, wt in weights.items():
        if wt > 0 and key in strat_cfgs:
            sname, sparams = strat_cfgs[key]
            sub_strats.append({"name": sname, "params": sparams, "weight": wt})

    cfg = {
        "_comment": comment,
        "run_name": config_name,
        "seed": 42,
        **{k: v for k, v in BASE_CONFIG.items() if k not in ("seed",)},
        "_strategies": sub_strats,
    }
    path = f"configs/{config_name}.json"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  Saved config: {path}", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PHASE 69: Canonical Lookback Refine + Funding Volatility Alpha")
print("=" * 70)

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# ─── SECTION A: Fill missing canonical 8h-step lookbacks ─────────────────────
print("\n" + "═" * 70)
print("SECTION A: Missing canonical lookbacks (8h steps)")
print("Phase 67 tested: 96, 108, 120, 132, 144, 156, 168")
print("Missing: 104, 112, 128, 136, 152, 160 (canonical 8h multiples)")
print("═" * 70)

# Phase 67 known results for reference
KNOWN_LOOKBACKS = {
    96: 1.031, 108: 0.996, 120: 0.855, 132: 0.917,
    144: 1.302, 156: 1.184, 168: 1.197,
}

# Missing canonical 8h multiples between 96 and 168
MISSING_LOOKBACKS = [104, 112, 128, 136, 152, 160]

fund_lb_results = dict(KNOWN_LOOKBACKS)  # start with known data (AVG only)
fund_lb_data = {}  # store full data for best ones

for lb in MISSING_LOOKBACKS:
    n_s = lb // 8
    label = f"fund_mom_lb{lb}"
    params = {
        "k_per_side": 2, "funding_lookback_bars": lb, "direction": "contrarian",
        "vol_lookback_bars": 168, "target_gross_leverage": 0.25, "rebalance_interval_bars": 24,
    }
    data = run_strategy(label, "funding_momentum_alpha", params)
    fund_lb_results[lb] = data["_avg"]
    fund_lb_data[lb] = data

print("\n── Full lookback sweep (8h steps, n_samples = lb//8) ──────────────────")
print(f"{'LB':>6} {'n_samples':>10} {'AVG Sharpe':>12}")
print("─" * 32)
for lb in sorted(fund_lb_results.keys()):
    n_s = lb // 8
    avg = fund_lb_results[lb]
    marker = " ← PEAK" if avg == max(fund_lb_results.values()) else ""
    print(f"{lb:>6}h {n_s:>10} {avg:>12.3f}{marker}")

best_lb = max(fund_lb_results.keys(), key=lambda lb: fund_lb_results[lb])
best_lb_avg = fund_lb_results[best_lb]
print(f"\nBest lookback: {best_lb}h (n_samples={best_lb//8}), AVG={best_lb_avg:.3f}")

# Run the best new lookback in full if it's not 144 (already known)
if best_lb not in [144, 168]:
    print(f"\n→ Best new lookback is {best_lb}h — running standalone for full data", flush=True)
    best_fund_params = {
        "k_per_side": 2, "funding_lookback_bars": best_lb, "direction": "contrarian",
        "vol_lookback_bars": 168, "target_gross_leverage": 0.25, "rebalance_interval_bars": 24,
    }
    best_fund_data = fund_lb_data[best_lb]
    yby = [best_fund_data.get(y, {}).get("sharpe", "?") for y in YEARS]
    print(f"  Year-by-year: {yby}")
else:
    print(f"\n→ Best lookback is {best_lb}h (already known from Phase 67/68)")
    best_fund_data = None

# Also re-run f144 and f168 for ensemble blending in later sections
print("\n  Running fund_144 reference (for ensemble)...", flush=True)
f144_data = run_strategy("fund_144_ref", "funding_momentum_alpha", FUND_144_PARAMS)
print("  Running fund_168 reference (for ensemble)...", flush=True)
f168_data = run_strategy("fund_168_ref", "funding_momentum_alpha", FUND_168_PARAMS)

# ─── SECTION B: FundingVolAlpha standalone ───────────────────────────────────
print("\n" + "═" * 70)
print("SECTION B: FundingVolAlpha standalone (new signal)")
print("Hypothesis: std dev of funding rates → identifies coins post-squeeze")
print("═" * 70)

# Contrarian direction: LONG high-vol-funding (they've been squeezed, now clean)
# Momentum direction: SHORT high-vol-funding (still unstable)
FUNDVOL_LOOKBACKS = [96, 144, 168]
fundvol_results = {}

for direction in ["contrarian", "momentum"]:
    print(f"\n  Direction: {direction}")
    for lb in FUNDVOL_LOOKBACKS:
        label = f"fundvol_{direction}_lb{lb}"
        params = {
            "k_per_side": 2, "funding_lookback_bars": lb, "direction": direction,
            "vol_lookback_bars": 168, "target_gross_leverage": 0.25, "rebalance_interval_bars": 24,
        }
        data = run_strategy(label, "funding_vol_alpha", params)
        fundvol_results[label] = data

print("\n── FundingVolAlpha Results Summary ─────────────────────────────────────")
print(f"{'Label':40} {'AVG':>7} {'MIN':>7} {'Pos':>5}")
print("─" * 60)
for label, data in sorted(fundvol_results.items(), key=lambda x: x[1]["_avg"], reverse=True):
    print(f"{label:40} {data['_avg']:>7.3f} {data['_min']:>7.3f} {data['_pos']:>5}/5")

best_fundvol_label = max(fundvol_results.keys(), key=lambda k: fundvol_results[k]["_avg"])
best_fundvol_data = fundvol_results[best_fundvol_label]
best_fundvol_avg = best_fundvol_data["_avg"]
best_fundvol_direction = best_fundvol_label.split("_")[1]
best_fundvol_lb = int(best_fundvol_label.split("lb")[1])
print(f"\nBest fund_vol: {best_fundvol_label}, AVG={best_fundvol_avg:.3f}")

# ─── SECTION C: Run reference strategies for ensemble ────────────────────────
print("\n" + "═" * 70)
print("SECTION C: Reference strategies for ensemble blending")
print("═" * 70)

print("  Running V1 reference...", flush=True)
v1_data = run_strategy("v1_ref", "nexus_alpha_v1", V1_PARAMS)
print("  Running idio_437 reference...", flush=True)
idio_data = run_strategy("idio_437_ref", "idio_momentum_alpha", IDIO_437_PARAMS)

# Also run best new lb if different from 144
if best_lb != 144 and best_lb != 168 and best_fund_data is not None:
    print(f"  Running fund_{best_lb} for ensemble...", flush=True)
    best_new_fund_params = {
        "k_per_side": 2, "funding_lookback_bars": best_lb, "direction": "contrarian",
        "vol_lookback_bars": 168, "target_gross_leverage": 0.25, "rebalance_interval_bars": 24,
    }
    best_new_fund_data = fund_lb_data[best_lb]
else:
    best_new_fund_data = f144_data if best_lb == 144 else f168_data

# ─── SECTION D: Ensemble integration ─────────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION D: Ensemble integration")
print("Phase 68 champion: V1(15)+Idio(30)+F144(40)+F168(15) AVG=1.844")
print("Phase 68 balanced: V1(20)+SR(10)+Idio(20)+F144(30)+F168(20) AVG=1.826")
print("═" * 70)

# Build ensemble strategies dict with available data
ensemble_strats = {
    "v1": v1_data,
    "idio": idio_data,
    "f144": f144_data,
    "f168": f168_data,
}

# Phase 68 champion as baseline
p68_champ = blend_ensemble(ensemble_strats, {"v1": 0.15, "idio": 0.30, "f144": 0.40, "f168": 0.15})
p68_bal   = blend_ensemble(ensemble_strats, {"v1": 0.20, "idio": 0.20, "f144": 0.30, "f168": 0.20})
print(f"\n  Phase 68 champion (baseline): AVG={p68_champ['avg']}, MIN={p68_champ['min']}, {p68_champ['pos']}/5")
print(f"  Phase 68 balanced (baseline): AVG={p68_bal['avg']}, MIN={p68_bal['min']}, {p68_bal['pos']}/5")

# D1: If there's a new best lookback, try swapping fund_144 for it
if best_lb not in [144, 168] and best_new_fund_data is not None:
    print(f"\n── D1: Swap fund_144 → fund_{best_lb} (new peak lb) ─────────────────────")
    strats_with_new = dict(ensemble_strats)
    strats_with_new[f"f{best_lb}"] = best_new_fund_data
    new_champ = blend_ensemble(strats_with_new, {"v1": 0.15, "idio": 0.30, f"f{best_lb}": 0.40, "f168": 0.15})
    print(f"  V1(15)+Idio(30)+F{best_lb}(40)+F168(15): AVG={new_champ['avg']}, MIN={new_champ['min']}, {new_champ['pos']}/5")
    # Also try adding it alongside f144
    strats_triple = dict(ensemble_strats)
    strats_triple[f"f{best_lb}"] = best_new_fund_data
    triple = blend_ensemble(strats_triple, {"v1": 0.15, "idio": 0.25, "f144": 0.25, f"f{best_lb}": 0.20, "f168": 0.15})
    print(f"  V1(15)+Idio(25)+F144(25)+F{best_lb}(20)+F168(15): AVG={triple['avg']}, MIN={triple['min']}, {triple['pos']}/5")

# D2: FundingVolAlpha ensemble integration (if competitive standalone)
FUNDVOL_USEFUL_THRESHOLD = 0.5  # If AVG > 0.5, worth integrating
if best_fundvol_avg > FUNDVOL_USEFUL_THRESHOLD:
    print(f"\n── D2: Fund_vol ensemble integration (AVG={best_fundvol_avg:.3f} > threshold) ─")

    # Get best fundvol data
    strats_with_fv = dict(ensemble_strats)
    strats_with_fv["fvol"] = best_fundvol_data

    # Test grids: replace part of fund_144 weight with fund_vol
    print(f"\n  Adding {best_fundvol_label} to ensemble")
    fv_experiments = [
        # Replace some F144 weight
        ("v1=15,idio=25,f144=30,f168=15,fvol=15", {"v1": 0.15, "idio": 0.25, "f144": 0.30, "f168": 0.15, "fvol": 0.15}),
        ("v1=15,idio=30,f144=30,f168=10,fvol=15", {"v1": 0.15, "idio": 0.30, "f144": 0.30, "f168": 0.10, "fvol": 0.15}),
        ("v1=15,idio=25,f144=25,f168=15,fvol=20", {"v1": 0.15, "idio": 0.25, "f144": 0.25, "f168": 0.15, "fvol": 0.20}),
        ("v1=20,idio=20,f144=25,f168=15,fvol=20", {"v1": 0.20, "idio": 0.20, "f144": 0.25, "f168": 0.15, "fvol": 0.20}),
        # Pure funding vol focus
        ("v1=15,idio=25,f144=20,f168=15,fvol=25", {"v1": 0.15, "idio": 0.25, "f144": 0.20, "f168": 0.15, "fvol": 0.25}),
    ]

    fv_results = []
    for desc, wts in fv_experiments:
        r = blend_ensemble(strats_with_fv, wts)
        fv_results.append((desc, wts, r))
        print(f"    {desc}: AVG={r['avg']}, MIN={r['min']}, {r['pos']}/5")

    best_fv = max(fv_results, key=lambda x: x[2]["avg"])
    print(f"\n  Best fund_vol ensemble: {best_fv[0]}")
    print(f"  AVG={best_fv[2]['avg']}, MIN={best_fv[2]['min']}, {best_fv[2]['pos']}/5")
    print(f"  YbY: {best_fv[2]['yby']}")

    # Compare with baseline
    if best_fv[2]["avg"] > p68_champ["avg"]:
        print(f"\n  ★ NEW CHAMPION with fund_vol! AVG {p68_champ['avg']} → {best_fv[2]['avg']}")
    else:
        print(f"\n  fund_vol doesn't beat champion ({p68_champ['avg']})")

else:
    print(f"\n── D2: fund_vol not competitive (AVG={best_fundvol_avg:.3f} < {FUNDVOL_USEFUL_THRESHOLD}) ──")
    print("  Skipping fund_vol ensemble experiments.")

# ─── SECTION E: Correlation analysis ─────────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION E: Correlation Analysis")
print("═" * 70)

all_returns = {
    "v1":   get_all_returns(v1_data),
    "idio": get_all_returns(idio_data),
    "f144": get_all_returns(f144_data),
    "f168": get_all_returns(f168_data),
}

# Add best new lookback if different
if best_lb not in [144, 168] and best_new_fund_data is not None:
    all_returns[f"f{best_lb}"] = get_all_returns(best_new_fund_data)

# Add best fundvol if competitive
if best_fundvol_avg > FUNDVOL_USEFUL_THRESHOLD:
    all_returns["fvol"] = get_all_returns(best_fundvol_data)

keys = list(all_returns.keys())
print(f"\n{'':12}", end="")
for k in keys:
    print(f"{k:>10}", end="")
print()
for k1 in keys:
    print(f"{k1:12}", end="")
    for k2 in keys:
        c = pearson_corr(all_returns[k1], all_returns[k2])
        print(f"{c:>10.4f}", end="")
    print()

# ─── SECTION F: Grand summary ─────────────────────────────────────────────────
print("\n" + "═" * 70)
print("SECTION F: Phase 69 Summary")
print("═" * 70)
print(f"\nPhase 68 champion (baseline): V1(15)+Idio(30)+F144(40)+F168(15)")
print(f"  AVG={p68_champ['avg']}, MIN={p68_champ['min']}, {p68_champ['pos']}/5")
print(f"\nFull lookback sweep summary (n_samples = lb//8):")
for lb in sorted(fund_lb_results.keys()):
    n_s = lb // 8
    avg = fund_lb_results[lb]
    marker = " ← BEST" if lb == best_lb else ""
    print(f"  lb={lb:4}h (n={n_s:2}): AVG={avg:.3f}{marker}")
print(f"\nFundingVolAlpha standalone:")
for label, data in sorted(fundvol_results.items(), key=lambda x: x[1]["_avg"], reverse=True):
    yby = [data.get(y, {}).get("sharpe", "?") for y in YEARS]
    print(f"  {label}: AVG={data['_avg']:.3f}, MIN={data['_min']:.3f}, YbY={yby}")

print("\n" + "=" * 70)
print("PHASE 69 COMPLETE")
print("=" * 70)
