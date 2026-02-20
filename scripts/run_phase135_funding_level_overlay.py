#!/usr/bin/env python3
"""
Phase 135: Funding Rate Level Overlay
=======================================
Hypothesis: Average funding rate across all coins is a market-level
sentiment/leverage indicator:
- Very high funding (everyone long) → over-leveraged, cascade risk → reduce
- Very negative funding (everyone short) → crowded short → reduce
- Normal funding → full exposure

Different from F144: F144 is cross-sectional (long low-funding, short high-funding).
This overlay uses the LEVEL (average) of funding as a conditioning variable.

Also test: funding rate dispersion — when dispersion is high, more differentiation
between coins = more cross-sectional alpha opportunity.
"""
import json
import os
import signal as _signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from nexus_quant.backtest.engine import BacktestConfig, BacktestEngine
from nexus_quant.backtest.costs import cost_model_from_config
from nexus_quant.data.providers.registry import make_provider
from nexus_quant.strategies.registry import make_strategy

_partial = {}
def _on_timeout(signum, frame):
    print("\n⏰ TIMEOUT — saving partial...")
    _save(_partial, partial=True)
    sys.exit(0)
_signal.signal(_signal.SIGALRM, _on_timeout)
_signal.alarm(600)

PROD_CFG = json.loads((ROOT / "configs" / "production_p91b_champion.json").read_text())
SYMBOLS = PROD_CFG["data"]["symbols"]
ENSEMBLE_WEIGHTS = PROD_CFG["ensemble"]["weights"]
SIGNAL_DEFS = PROD_CFG["ensemble"]["signals"]
SIG_KEYS = list(SIGNAL_DEFS.keys())

YEAR_RANGES = {
    "2021": ("2021-02-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
}
YEARS = sorted(YEAR_RANGES.keys())

OUT_DIR = ROOT / "artifacts" / "phase135"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sharpe(rets: np.ndarray) -> float:
    if len(rets) < 100:
        return 0.0
    mu = float(np.mean(rets)) * 8760
    sd = float(np.std(rets)) * np.sqrt(8760)
    return round(mu / sd, 4) if sd > 1e-12 else 0.0


def obj_func(yearly_sharpes: list) -> float:
    arr = np.array(yearly_sharpes)
    return round(float(np.mean(arr) - 0.5 * np.std(arr)), 4)


def _save(data: dict, partial: bool = False) -> None:
    data["partial"] = partial
    data["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path = OUT_DIR / "funding_level_overlay_report.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → Saved: {path}")


def compute_rolling_avg_funding(dataset, symbols, window: int = 72) -> tuple:
    """Compute rolling average and std of funding rates across symbols.
    Returns (avg_funding, std_funding) arrays aligned to bars 1..N.
    Uses last_funding_rate_before() for each bar's timestamp."""
    n = len(dataset.timeline)
    # Build per-bar funding rate for each symbol using last known funding
    sym_funding = {}
    for sym in symbols:
        rates = np.zeros(n)
        for i in range(n):
            ts = dataset.timeline[i]
            rates[i] = dataset.last_funding_rate_before(sym, ts)
        sym_funding[sym] = rates

    # Rolling average across symbols and time
    avg_fund = np.full(n - 1, np.nan)
    std_fund = np.full(n - 1, np.nan)

    for i in range(1, n):
        idx = i - 1  # align to returns (which are bar 1..N)
        if i >= window:
            # Rolling window of avg cross-symbol funding
            window_avgs = []
            for j in range(i - window, i):
                cross_sym = [sym_funding[s][j] for s in symbols]
                window_avgs.append(float(np.mean(cross_sym)))
            avg_fund[idx] = float(np.mean(window_avgs))
            std_fund[idx] = float(np.std([sym_funding[s][j] for s in symbols for j in range(i - window, i)]))
        elif i > 0:
            cross_sym = [sym_funding[s][i] for s in symbols]
            avg_fund[idx] = float(np.mean(cross_sym))
            std_fund[idx] = float(np.std(cross_sym))

    return avg_fund, std_fund


def main():
    global _partial
    t0 = time.time()
    print("=" * 70)
    print("Phase 135: Funding Rate Level Overlay")
    print("=" * 70)

    # 1. Load data
    print("\n[1/4] Loading data + funding rates...")
    sig_returns = {sk: {} for sk in SIG_KEYS}
    funding_avg = {}
    funding_std = {}

    for year, (start, end) in YEAR_RANGES.items():
        print(f"  {year}:", end="", flush=True)
        cfg_data = {
            "provider": "binance_rest_v1", "symbols": SYMBOLS,
            "start": start, "end": end, "bar_interval": "1h",
            "cache_dir": ".cache/binance_rest",
        }
        provider = make_provider(cfg_data, seed=42)
        dataset = provider.load()

        # Signal returns
        for sk in SIG_KEYS:
            sig_def = SIGNAL_DEFS[sk]
            cost_model = cost_model_from_config({"fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0})
            bt_cfg = BacktestConfig(costs=cost_model)
            strat = make_strategy({"name": sig_def["strategy"], "params": sig_def["params"]})
            engine = BacktestEngine(bt_cfg)
            result = engine.run(dataset, strat)
            sig_returns[sk][year] = np.array(result.returns, dtype=np.float64)

        # Funding rates
        favg, fstd = compute_rolling_avg_funding(dataset, SYMBOLS, window=72)
        funding_avg[year] = favg
        funding_std[year] = fstd

        # Stats
        valid_avg = favg[~np.isnan(favg)]
        valid_std = fstd[~np.isnan(fstd)]
        if len(valid_avg) > 0:
            ann_rate = float(np.mean(valid_avg)) * 3 * 365 * 100  # annualized %
            print(f" avg_fund={float(np.mean(valid_avg)):.6f} ({ann_rate:+.1f}%/yr) "
                  f"std={float(np.mean(valid_std)):.6f}", flush=True)
        else:
            print(f" no valid funding data", flush=True)

    _partial = {"phase": 135}

    # 2. Baseline
    print("\n[2/4] Computing baseline...")
    baseline_sharpes = {}
    for year in YEARS:
        min_len = min(len(sig_returns[sk][year]) for sk in SIG_KEYS)
        ens = np.zeros(int(min_len))
        for sk in SIG_KEYS:
            ens += ENSEMBLE_WEIGHTS[sk] * sig_returns[sk][year][:int(min_len)]
        baseline_sharpes[year] = sharpe(ens)
    baseline_obj = obj_func(list(baseline_sharpes.values()))
    print(f"  BASELINE: OBJ={baseline_obj:.4f}")

    # 3. Regime analysis: split by funding level
    print("\n[3/4] Regime analysis: ensemble Sharpe by funding regime...")

    for year in YEARS:
        min_len = min(len(sig_returns[sk][year]) for sk in SIG_KEYS)
        min_len = min(min_len, len(funding_avg[year]))
        favg = funding_avg[year][:min_len]
        valid_mask = ~np.isnan(favg)

        # Compute ensemble
        ens = np.zeros(min_len)
        for sk in SIG_KEYS:
            ens += ENSEMBLE_WEIGHTS[sk] * sig_returns[sk][year][:min_len]

        # Split by funding level: high (>median), low (<median), extreme high (>p90), extreme negative (<p10)
        valid = favg[valid_mask]
        if len(valid) < 100:
            print(f"  {year}: insufficient valid funding data")
            continue

        med = float(np.median(valid))
        p10 = float(np.percentile(valid, 10))
        p90 = float(np.percentile(valid, 90))

        high_mask = valid_mask & (favg >= med)
        low_mask = valid_mask & (favg < med)
        extreme_high = valid_mask & (favg >= p90)
        extreme_low = valid_mask & (favg <= p10)
        normal = valid_mask & (favg > p10) & (favg < p90)

        s_high = sharpe(ens[high_mask])
        s_low = sharpe(ens[low_mask])
        s_ext_high = sharpe(ens[extreme_high])
        s_ext_low = sharpe(ens[extreme_low])
        s_normal = sharpe(ens[normal])

        print(f"  {year}: high_f={s_high:+.2f}  low_f={s_low:+.2f}  "
              f"ext_hi={s_ext_high:+.2f}  ext_lo={s_ext_low:+.2f}  "
              f"normal={s_normal:+.2f}  "
              f"med={med:.6f}")

    # Also check funding dispersion
    print("\n  Funding dispersion regime...")
    for year in YEARS:
        min_len = min(len(sig_returns[sk][year]) for sk in SIG_KEYS)
        min_len = min(min_len, len(funding_std[year]))
        fstd = funding_std[year][:min_len]
        valid_mask = ~np.isnan(fstd)

        ens = np.zeros(min_len)
        for sk in SIG_KEYS:
            ens += ENSEMBLE_WEIGHTS[sk] * sig_returns[sk][year][:min_len]

        valid = fstd[valid_mask]
        if len(valid) < 100:
            continue

        med_std = float(np.median(valid))
        high_disp = valid_mask & (fstd >= med_std)
        low_disp = valid_mask & (fstd < med_std)

        s_high = sharpe(ens[high_disp])
        s_low = sharpe(ens[low_disp])

        tag = "✓" if s_high > s_low else "✗"
        print(f"  {year}: high_disp={s_high:+.2f}  low_disp={s_low:+.2f}  "
              f"Δ={s_high - s_low:+.2f} {tag}")

    # 4. Test overlays
    print("\n[4/4] Testing funding level overlays...")

    overlay_configs = {
        "baseline": {"mode": "none"},
        "reduce_ext_high_50pct": {"mode": "reduce_extreme", "high_pct": 90, "low_pct": 10, "scale": 0.5},
        "reduce_ext_high_flat": {"mode": "reduce_extreme", "high_pct": 90, "low_pct": 10, "scale": 0.0},
        "reduce_ext_high_95th": {"mode": "reduce_extreme", "high_pct": 95, "low_pct": 5, "scale": 0.5},
        "reduce_high_funding": {"mode": "reduce_high", "pct": 75, "scale": 0.5},
        "boost_low_funding": {"mode": "boost_low", "pct": 25, "scale": 1.3},
        "tilt_extreme": {"mode": "tilt_extreme", "high_pct": 90, "low_pct": 10,
                         "boost_f144": 0.20, "reduce_scale": 0.7},
    }

    overlay_results = {}
    for oname, ocfg in overlay_configs.items():
        yearly_sharpes = []
        yearly_detail = {}

        for year in YEARS:
            min_len = min(len(sig_returns[sk][year]) for sk in SIG_KEYS)
            min_len = min(min_len, len(funding_avg[year]))
            favg = funding_avg[year][:min_len]
            valid = favg[~np.isnan(favg)]

            if ocfg["mode"] == "none":
                ens = np.zeros(min_len)
                for sk in SIG_KEYS:
                    ens += ENSEMBLE_WEIGHTS[sk] * sig_returns[sk][year][:min_len]
            elif ocfg["mode"] == "reduce_extreme":
                high_thresh = float(np.percentile(valid, ocfg["high_pct"]))
                low_thresh = float(np.percentile(valid, ocfg["low_pct"]))
                ens = np.zeros(min_len)
                for i in range(min_len):
                    ret = 0.0
                    for sk in SIG_KEYS:
                        ret += ENSEMBLE_WEIGHTS[sk] * sig_returns[sk][year][i]
                    if not np.isnan(favg[i]) and (favg[i] >= high_thresh or favg[i] <= low_thresh):
                        ret *= ocfg["scale"]
                    ens[i] = ret
            elif ocfg["mode"] == "reduce_high":
                thresh = float(np.percentile(valid, ocfg["pct"]))
                ens = np.zeros(min_len)
                for i in range(min_len):
                    ret = 0.0
                    for sk in SIG_KEYS:
                        ret += ENSEMBLE_WEIGHTS[sk] * sig_returns[sk][year][i]
                    if not np.isnan(favg[i]) and favg[i] >= thresh:
                        ret *= ocfg["scale"]
                    ens[i] = ret
            elif ocfg["mode"] == "boost_low":
                thresh = float(np.percentile(valid, ocfg["pct"]))
                ens = np.zeros(min_len)
                for i in range(min_len):
                    ret = 0.0
                    for sk in SIG_KEYS:
                        ret += ENSEMBLE_WEIGHTS[sk] * sig_returns[sk][year][i]
                    if not np.isnan(favg[i]) and favg[i] <= thresh:
                        ret *= ocfg["scale"]
                    ens[i] = ret
            elif ocfg["mode"] == "tilt_extreme":
                high_thresh = float(np.percentile(valid, ocfg["high_pct"]))
                low_thresh = float(np.percentile(valid, ocfg["low_pct"]))
                ens = np.zeros(min_len)
                for i in range(min_len):
                    if not np.isnan(favg[i]) and (favg[i] >= high_thresh or favg[i] <= low_thresh):
                        # Tilt toward F144 + reduce
                        w = dict(ENSEMBLE_WEIGHTS)
                        boost = ocfg["boost_f144"]
                        reduce_each = boost / max(1, len(w) - 1)
                        for sk in SIG_KEYS:
                            if sk == "f144":
                                w[sk] = min(0.60, w[sk] + boost)
                            else:
                                w[sk] = max(0.05, w[sk] - reduce_each)
                        total = sum(w.values())
                        ret = 0.0
                        for sk in SIG_KEYS:
                            ret += (w[sk] / total) * sig_returns[sk][year][i] * ocfg["reduce_scale"]
                        ens[i] = ret
                    else:
                        for sk in SIG_KEYS:
                            ens[i] += ENSEMBLE_WEIGHTS[sk] * sig_returns[sk][year][i]

            s = sharpe(ens)
            yearly_sharpes.append(s)
            yearly_detail[year] = s

        avg = round(float(np.mean(yearly_sharpes)), 4)
        mn = round(float(np.min(yearly_sharpes)), 4)
        obj = obj_func(yearly_sharpes)
        overlay_results[oname] = {
            "yearly": yearly_detail,
            "avg_sharpe": avg,
            "min_sharpe": mn,
            "obj": obj,
        }

    print(f"\n  {'Variant':28s} {'OBJ':>8s} {'AVG':>8s} {'MIN':>8s} {'ΔOBJ':>8s}")
    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for oname in sorted(overlay_results, key=lambda k: overlay_results[k]["obj"], reverse=True):
        r = overlay_results[oname]
        delta = r["obj"] - baseline_obj
        tag = " ✓" if delta > 0.05 else ""
        print(f"  {oname:28s} {r['obj']:8.4f} {r['avg_sharpe']:8.3f} "
              f"{r['min_sharpe']:8.3f} {delta:+8.4f}{tag}")

    best_name = max(overlay_results, key=lambda k: overlay_results[k]["obj"])
    best = overlay_results[best_name]
    improvement = best["obj"] - baseline_obj

    if improvement < 0.03:
        verdict = "NO IMPROVEMENT — funding level overlay does not help"
    elif improvement < 0.10:
        verdict = f"MARGINAL — {best_name} +{improvement:.3f} OBJ, needs WF"
    else:
        verdict = f"POTENTIAL — {best_name} adds +{improvement:.3f} OBJ"

    elapsed = time.time() - t0
    _partial = {
        "phase": 135,
        "description": "Funding Rate Level Overlay",
        "elapsed_seconds": round(elapsed, 1),
        "overlays": overlay_results,
        "best": {"name": best_name, **best},
        "improvement": round(improvement, 4),
        "verdict": verdict,
    }
    _save(_partial, partial=False)

    print(f"\n  VERDICT: {verdict}")
    print(f"\nPhase 135 complete in {elapsed:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
