#!/usr/bin/env python3
"""
Phase 111: Positioning & Volume Tilt on P91b
=============================================
Test 1: Binance OI/LS as tilt → BLOCKED (API only has ~30 days history)
Test 2: Volume momentum as proxy tilt (available for all years)
Test 3: Walk-forward validation of any promising volume tilt

Volume momentum hypothesis: when aggregate volume spikes (crowded),
contrarian tilt reduces leverage → avoids crowded trades reversing.
"""

import copy, json, os, sys, time
from datetime import datetime

PROJ = "/Users/qtmobile/Desktop/Nexus - Quant Trading "
sys.path.insert(0, PROJ)

import numpy as np

from nexus_quant.data.providers.registry import make_provider
from nexus_quant.strategies.registry import make_strategy
from nexus_quant.backtest.engine import BacktestConfig, BacktestEngine
from nexus_quant.backtest.costs import cost_model_from_config

OUT_DIR = os.path.join(PROJ, "artifacts", "phase111")
os.makedirs(OUT_DIR, exist_ok=True)

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
]

YEARS = ["2021", "2022", "2023", "2024", "2025"]
YEAR_RANGES = {
    "2021": ("2021-01-01", "2022-01-01"),
    "2022": ("2022-01-01", "2023-01-01"),
    "2023": ("2023-01-01", "2024-01-01"),
    "2024": ("2024-01-01", "2025-01-01"),
    "2025": ("2025-01-01", "2026-01-01"),
}
OOS_RANGE = ("2026-01-01", "2026-02-20")

P91B_WEIGHTS = {"v1": 0.2747, "i460bw168": 0.1967, "i415bw216": 0.3247, "f144": 0.2039}
SIG_KEYS = sorted(P91B_WEIGHTS.keys())

SIGNALS = {
    "v1": {"name": "nexus_alpha_v1", "params": {
        "k_per_side": 2, "w_carry": 0.35, "w_mom": 0.45, "w_mean_reversion": 0.20,
        "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 72,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.35, "rebalance_interval_bars": 60}},
    "i460bw168": {"name": "idio_momentum_alpha", "params": {
        "k_per_side": 4, "lookback_bars": 460, "beta_window_bars": 168,
        "target_gross_leverage": 0.30, "rebalance_interval_bars": 48}},
    "i415bw216": {"name": "idio_momentum_alpha", "params": {
        "k_per_side": 4, "lookback_bars": 415, "beta_window_bars": 216,
        "target_gross_leverage": 0.30, "rebalance_interval_bars": 48}},
    "f144": {"name": "funding_momentum_alpha", "params": {
        "k_per_side": 2, "funding_lookback_bars": 144, "direction": "contrarian",
        "target_gross_leverage": 0.25, "rebalance_interval_bars": 24}},
}

TILT_RATIOS = [round(r * 0.1, 1) for r in range(11)]  # 0.0 to 1.0

WF_WINDOWS = [
    {"train": ["2021", "2022", "2023"], "test": ["2024"]},
    {"train": ["2022", "2023", "2024"], "test": ["2025"]},
    {"train": ["2021", "2022", "2023", "2024"], "test": ["2025"]},
    {"train": ["2021", "2022", "2023", "2024", "2025"], "test": ["2026_oos"]},
]


def log(msg):
    print(f"[P111] {msg}", flush=True)


def compute_sharpe(returns, bars_per_year=8760):
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size < 50:
        return 0.0
    std = float(np.std(arr))
    if std <= 0:
        return 0.0
    return float(np.mean(arr) / std * np.sqrt(bars_per_year))


def get_dataset(year_key, cache):
    """Fetch dataset for a year, with caching."""
    if year_key == "2026_oos":
        start, end = OOS_RANGE
    else:
        start, end = YEAR_RANGES[year_key]
    cache_key = f"{start}_{end}"
    if cache_key not in cache:
        provider = make_provider({
            "provider": "binance_rest_v1",
            "symbols": SYMBOLS,
            "bar_interval": "1h",
            "start": start,
            "end": end,
        }, seed=42)
        cache[cache_key] = provider.load()
    return cache[cache_key]


def run_p91b_returns(dataset):
    """Run P91b ensemble, return hourly returns array."""
    cost_model = cost_model_from_config({"fee_rate": 0.0005, "slippage_rate": 0.0003})
    all_returns = []
    for sig_key in SIG_KEYS:
        cfg = SIGNALS[sig_key]
        strat = make_strategy({"name": cfg["name"], "params": cfg["params"]})
        engine = BacktestEngine(BacktestConfig(costs=cost_model))
        result = engine.run(dataset, strat)
        rets = np.diff(result.equity_curve) / result.equity_curve[:-1]
        all_returns.append(rets)
    min_len = min(len(r) for r in all_returns)
    all_returns = [r[:min_len] for r in all_returns]
    weights = np.array([P91B_WEIGHTS[k] for k in SIG_KEYS])
    ensemble_rets = np.zeros(min_len)
    for i, k in enumerate(SIG_KEYS):
        ensemble_rets += weights[i] * all_returns[i]
    return ensemble_rets


def compute_volume_z(dataset, lookback):
    """Compute aggregate volume momentum z-score from dataset."""
    # Sum perp_volume across all symbols
    total_vol = None
    if dataset.perp_volume:
        for sym in SYMBOLS:
            vols = np.array(dataset.perp_volume.get(sym, []), dtype=np.float64)
            if total_vol is None:
                total_vol = vols.copy()
            else:
                min_l = min(len(total_vol), len(vols))
                total_vol = total_vol[:min_l] + vols[:min_l]

    if total_vol is None or len(total_vol) < lookback + 50:
        return None

    # Log volume momentum
    log_vol = np.log(np.maximum(total_vol, 1.0))
    mom = np.zeros(len(log_vol))
    mom[lookback:] = log_vol[lookback:] - log_vol[:-lookback]

    # Rolling z-score
    z_scores = np.zeros(len(mom))
    for i in range(lookback * 2, len(mom)):
        window = mom[max(0, i - lookback):i + 1]
        mu = np.mean(window)
        sigma = np.std(window)
        if sigma > 0:
            z_scores[i] = (mom[i] - mu) / sigma

    return z_scores


def apply_tilt(hourly_rets, z_scores, tilt_ratio):
    """Contrarian tilt: when z > 0 (high volume), reduce to tilt_ratio."""
    min_len = min(len(hourly_rets), len(z_scores))
    tilted = hourly_rets[:min_len].copy()
    mask = z_scores[:min_len] > 0
    tilted[mask] *= tilt_ratio
    return tilted


def main():
    t0 = time.time()

    # ── Step 1: Check Binance positioning data availability ─────────────
    log("Step 1: Checking Binance positioning API...")
    import urllib.request
    url = "https://fapi.binance.com/futures/data/openInterestHist?symbol=BTCUSDT&period=1h&limit=500"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "NexusQuant/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            oi_data = json.loads(resp.read().decode())
        if oi_data:
            ts_list = sorted(int(d["timestamp"]) for d in oi_data)
            d0 = datetime.fromtimestamp(ts_list[0] / 1000).strftime("%Y-%m-%d")
            d1 = datetime.fromtimestamp(ts_list[-1] / 1000).strftime("%Y-%m-%d")
            days = (ts_list[-1] - ts_list[0]) / (86400 * 1000)
            log(f"  BTC OI: {len(oi_data)} points, range {d0} to {d1} ({days:.0f} days)")
            log(f"  LIMITATION: Only ~30 days of hourly data available")
            log(f"  BLOCKED: Cannot backtest positioning tilt over 5yr cycle")
        else:
            log(f"  No OI data returned")
    except Exception as e:
        log(f"  OI API error: {e}")

    # ── Step 2: Fetch all data and compute baselines ────────────────────
    log("\nStep 2: Computing P91b baselines...")
    ds_cache = {}
    yearly_rets = {}
    yearly_vol_z = {}

    for yr in YEARS + ["2026_oos"]:
        dataset = get_dataset(yr, ds_cache)
        rets = run_p91b_returns(dataset)
        yearly_rets[yr] = rets

        # Check if positioning data exists
        has_oi = dataset.open_interest is not None and any(
            sum(1 for v in vals if v > 0) > 100
            for vals in dataset.open_interest.values()
        ) if dataset.open_interest else False
        log(f"  {yr}: {len(rets)} bars, OI_data={'YES' if has_oi else 'NO'}")

    # Baseline
    baseline = {}
    for yr in YEARS + ["2026_oos"]:
        baseline[yr] = round(compute_sharpe(yearly_rets[yr]), 4)
    is_sharpes = [baseline[yr] for yr in YEARS]
    base_avg = np.mean(is_sharpes)
    base_min = np.min(is_sharpes)
    base_obj = (base_avg + base_min) / 2

    log(f"\n  Baseline: AVG={base_avg:.4f}, MIN={base_min:.4f}, OBJ={base_obj:.4f}")
    log(f"  OOS: {baseline['2026_oos']}")
    for yr in YEARS:
        log(f"    {yr}: {baseline[yr]}")

    # ── Step 3: Volume momentum tilt (IS grid search) ───────────────────
    log("\n============================================================")
    log("Step 3: Volume Momentum Tilt (IS Grid Search)")
    log("============================================================")

    lookbacks = [24, 48, 72, 168, 336]
    tilt_results = {}

    for lb in lookbacks:
        sig_name = f"vol_mom_z_{lb}"
        log(f"\n  Signal: {sig_name}")

        best_ratio = 1.0
        best_obj = base_obj

        for ratio in TILT_RATIOS:
            yr_sharpes = []
            for yr in YEARS:
                dataset = get_dataset(yr, ds_cache)
                z_scores = compute_volume_z(dataset, lb)
                if z_scores is None:
                    yr_sharpes.append(compute_sharpe(yearly_rets[yr]))
                    continue
                tilted = apply_tilt(yearly_rets[yr], z_scores, ratio)
                yr_sharpes.append(compute_sharpe(tilted))

            avg_s = np.mean(yr_sharpes)
            min_s = np.min(yr_sharpes)
            obj = (avg_s + min_s) / 2

            if obj > best_obj:
                best_obj = obj
                best_ratio = ratio

        if best_ratio == 1.0:
            log(f"    Best ratio=1.0 (no tilt) → NO VALUE")
            tilt_results[sig_name] = {"best_ratio": 1.0, "delta_obj": 0.0, "verdict": "NO_VALUE"}
        else:
            # Compute full metrics at best ratio
            yr_sharpes = {}
            for yr in YEARS:
                dataset = get_dataset(yr, ds_cache)
                z_scores = compute_volume_z(dataset, lb)
                tilted = apply_tilt(yearly_rets[yr], z_scores, best_ratio)
                yr_sharpes[yr] = round(compute_sharpe(tilted), 4)

            avg_s = np.mean(list(yr_sharpes.values()))
            min_s = np.min(list(yr_sharpes.values()))
            obj = (avg_s + min_s) / 2
            delta_obj = obj - base_obj

            # OOS test
            dataset_oos = get_dataset("2026_oos", ds_cache)
            z_oos = compute_volume_z(dataset_oos, lb)
            if z_oos is not None:
                tilted_oos = apply_tilt(yearly_rets["2026_oos"], z_oos, best_ratio)
                oos_s = round(compute_sharpe(tilted_oos), 4)
            else:
                oos_s = baseline["2026_oos"]

            delta_oos = oos_s - baseline["2026_oos"]

            log(f"    Best ratio={best_ratio}")
            log(f"    IS: AVG={avg_s:.4f}, MIN={min_s:.4f}, OBJ={obj:.4f} (ΔOBJ={delta_obj:+.4f})")
            log(f"    OOS: {oos_s} (Δ={delta_oos:+.4f})")
            for yr, s in yr_sharpes.items():
                log(f"      {yr}: {s} (baseline={baseline[yr]}, Δ={s - baseline[yr]:+.4f})")

            tilt_results[sig_name] = {
                "best_ratio": best_ratio,
                "avg": round(avg_s, 4),
                "min": round(min_s, 4),
                "obj": round(obj, 4),
                "delta_obj": round(delta_obj, 4),
                "oos": oos_s,
                "delta_oos": round(delta_oos, 4),
                "yearly": yr_sharpes,
                "verdict": "PROMISING" if delta_obj > 0.05 else "MARGINAL" if delta_obj > 0 else "NO_VALUE",
            }

    # ── Step 4: Walk-forward validation of promising signals ────────────
    wf_results = {}
    promising = [k for k, v in tilt_results.items() if v.get("delta_obj", 0) > 0.02]

    if promising:
        log("\n============================================================")
        log("Step 4: Walk-Forward Validation")
        log("============================================================")

        for sig_name in promising:
            lb = int(sig_name.split("_")[-1])
            log(f"\n  Signal: {sig_name}")

            wf_deltas = []
            wf_ratios = []

            for wf in WF_WINDOWS:
                train_yrs = wf["train"]
                test_yr = wf["test"][0]

                # Find best ratio on train
                best_ratio = 1.0
                best_obj = -999

                for ratio in TILT_RATIOS:
                    yr_sharpes = []
                    for yr in train_yrs:
                        dataset = get_dataset(yr, ds_cache)
                        z_scores = compute_volume_z(dataset, lb)
                        if z_scores is None:
                            yr_sharpes.append(compute_sharpe(yearly_rets[yr]))
                            continue
                        tilted = apply_tilt(yearly_rets[yr], z_scores, ratio)
                        yr_sharpes.append(compute_sharpe(tilted))

                    avg_s = np.mean(yr_sharpes)
                    min_s = np.min(yr_sharpes)
                    obj = (avg_s + min_s) / 2
                    if obj > best_obj:
                        best_obj = obj
                        best_ratio = ratio

                # Test on OOS
                dataset_test = get_dataset(test_yr, ds_cache)
                z_test = compute_volume_z(dataset_test, lb)
                if z_test is not None:
                    tilted_test = apply_tilt(yearly_rets[test_yr], z_test, best_ratio)
                    test_sharpe = compute_sharpe(tilted_test)
                else:
                    test_sharpe = compute_sharpe(yearly_rets[test_yr])

                delta = test_sharpe - baseline.get(test_yr, 0)
                wf_deltas.append(delta)
                wf_ratios.append(best_ratio)

                log(f"    Train={'+'.join(train_yrs)} → r={best_ratio}, Test: {test_yr}: Δ{delta:+.3f}")

            n_positive = sum(1 for d in wf_deltas if d > 0)
            avg_delta = np.mean(wf_deltas)
            ratio_std = np.std(wf_ratios)

            validated = n_positive >= 2 and avg_delta > 0
            log(f"  → {n_positive}/4 positive, avg Δ={avg_delta:+.4f}, ratio_std={ratio_std:.2f}")
            log(f"  → {'VALIDATED' if validated else 'FAILED'}")

            wf_results[sig_name] = {
                "n_positive": n_positive,
                "avg_delta": round(avg_delta, 4),
                "ratio_std": round(ratio_std, 2),
                "deltas": [round(d, 4) for d in wf_deltas],
                "ratios": wf_ratios,
                "validated": validated,
            }
    else:
        log("\nStep 4: No promising signals to validate via walk-forward.")

    # ── Summary ─────────────────────────────────────────────────────────
    log("\n============================================================")
    log("PHASE 111 SUMMARY")
    log("============================================================")
    log(f"Positioning (OI/LS): BLOCKED — Binance free API only has ~30 days")
    log(f"Volume momentum tilts tested: {len(tilt_results)}")

    for name, res in tilt_results.items():
        if res["best_ratio"] == 1.0:
            log(f"  {name}: NO VALUE (best=1.0)")
        else:
            log(f"  {name}: r={res['best_ratio']}, ΔOBJ={res['delta_obj']:+.4f}, ΔOOS={res.get('delta_oos', 0):+.4f} → {res['verdict']}")

    if wf_results:
        log(f"\nWalk-forward results:")
        for name, res in wf_results.items():
            log(f"  {name}: {res['n_positive']}/4 positive, avg Δ={res['avg_delta']:+.4f} → {'VALIDATED' if res['validated'] else 'FAILED'}")

    validated_signals = [k for k, v in wf_results.items() if v.get("validated", False)]
    if validated_signals:
        log(f"\nVALIDATED: {validated_signals}")
    else:
        log(f"\nNo volume tilt validated via walk-forward.")
        log(f"P91b champion remains UNCHANGED.")

    # ── Save report ─────────────────────────────────────────────────────
    elapsed = time.time() - t0
    report = {
        "phase": 111,
        "description": "Positioning & volume tilt on P91b",
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "positioning_blocked": True,
        "positioning_reason": "Binance free API only has ~30 days of hourly OI/LS data",
        "baseline": {
            "yearly": baseline,
            "is_avg": round(base_avg, 4),
            "is_min": round(base_min, 4),
            "is_obj": round(base_obj, 4),
        },
        "volume_tilt_results": tilt_results,
        "walk_forward_results": wf_results,
        "validated_signals": validated_signals,
        "conclusion": "VALIDATED" if validated_signals else "NO_IMPROVEMENT",
    }

    def _default(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    report_path = os.path.join(OUT_DIR, "phase111_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=_default)

    log(f"\nPhase 111 COMPLETE in {elapsed:.1f}s → {report_path}")


if __name__ == "__main__":
    main()
