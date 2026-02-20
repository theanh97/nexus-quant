#!/usr/bin/env python3
"""
Phase 114: Taker Buy Ratio + Remaining Dataset Signals as Tilts
================================================================
Volume momentum tilt worked (Phase 111-113). Now test other data
already in the dataset at hourly frequency:

1. Taker buy ratio: taker_buy_volume / perp_volume (aggregated across symbols)
   High ratio → aggressive buying → contrarian reduce
2. Volume concentration: HHI of volume across symbols
   High concentration → crowded few names → reduce
3. Funding rate dispersion: std of funding rates across symbols
   High dispersion → divergent bets → reduce

All use same walk-forward validation framework.
"""

import json, os, sys, time
from datetime import datetime

PROJ = "/Users/qtmobile/Desktop/Nexus - Quant Trading "
sys.path.insert(0, PROJ)

import numpy as np

from nexus_quant.data.providers.registry import make_provider
from nexus_quant.strategies.registry import make_strategy
from nexus_quant.backtest.engine import BacktestConfig, BacktestEngine
from nexus_quant.backtest.costs import cost_model_from_config

OUT_DIR = os.path.join(PROJ, "artifacts", "phase114")
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

# Apply vol tilt from Phase 113 to get tilted baseline
VOL_TILT_LB = 168
VOL_TILT_RATIO = 0.65


def log(msg):
    print(f"[P114] {msg}", flush=True)


def compute_sharpe(returns, bars_per_year=8760):
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size < 50: return 0.0
    std = float(np.std(arr))
    if std <= 0: return 0.0
    return float(np.mean(arr) / std * np.sqrt(bars_per_year))


def get_dataset(year_key, cache):
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


def compute_volume_z(dataset, lookback=VOL_TILT_LB):
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
    log_vol = np.log(np.maximum(total_vol, 1.0))
    mom = np.zeros(len(log_vol))
    mom[lookback:] = log_vol[lookback:] - log_vol[:-lookback]
    z_scores = np.zeros(len(mom))
    for i in range(lookback * 2, len(mom)):
        window = mom[max(0, i - lookback):i + 1]
        mu, sigma = np.mean(window), np.std(window)
        if sigma > 0:
            z_scores[i] = (mom[i] - mu) / sigma
    return z_scores


def apply_vol_tilt(rets, z_scores, ratio=VOL_TILT_RATIO):
    min_len = min(len(rets), len(z_scores))
    tilted = rets[:min_len].copy()
    mask = z_scores[:min_len] > 0
    tilted[mask] *= ratio
    return tilted


# ── New signal builders ──────────────────────────────────────────────────
def build_taker_ratio_z(dataset, lookback):
    """Taker buy ratio: sum(taker_buy_vol) / sum(perp_vol) across all symbols."""
    if not dataset.taker_buy_volume or not dataset.perp_volume:
        return None
    total_taker = None
    total_vol = None
    for sym in SYMBOLS:
        tv = np.array(dataset.taker_buy_volume.get(sym, []), dtype=np.float64)
        pv = np.array(dataset.perp_volume.get(sym, []), dtype=np.float64)
        if total_taker is None:
            total_taker = tv.copy()
            total_vol = pv.copy()
        else:
            min_l = min(len(total_taker), len(tv), len(total_vol), len(pv))
            total_taker = total_taker[:min_l] + tv[:min_l]
            total_vol = total_vol[:min_l] + pv[:min_l]

    if total_taker is None or len(total_taker) < lookback + 50:
        return None

    ratio = np.divide(total_taker, np.maximum(total_vol, 1.0))
    z_scores = np.zeros(len(ratio))
    for i in range(lookback * 2, len(ratio)):
        window = ratio[max(0, i - lookback):i + 1]
        mu, sigma = np.mean(window), np.std(window)
        if sigma > 0:
            z_scores[i] = (ratio[i] - mu) / sigma
    return z_scores


def build_vol_concentration_z(dataset, lookback):
    """Volume HHI: sum of (vol_i / total_vol)^2 across symbols."""
    if not dataset.perp_volume:
        return None
    sym_vols = []
    for sym in SYMBOLS:
        sv = np.array(dataset.perp_volume.get(sym, []), dtype=np.float64)
        sym_vols.append(sv)
    min_len = min(len(v) for v in sym_vols)
    sym_vols = [v[:min_len] for v in sym_vols]
    vols = np.array(sym_vols)  # (n_symbols, n_bars)
    total = np.sum(vols, axis=0)
    total = np.maximum(total, 1.0)
    shares = vols / total  # (n_symbols, n_bars)
    hhi = np.sum(shares ** 2, axis=0)  # (n_bars,)

    if len(hhi) < lookback + 50:
        return None

    z_scores = np.zeros(len(hhi))
    for i in range(lookback * 2, len(hhi)):
        window = hhi[max(0, i - lookback):i + 1]
        mu, sigma = np.mean(window), np.std(window)
        if sigma > 0:
            z_scores[i] = (hhi[i] - mu) / sigma
    return z_scores


def build_funding_dispersion_z(dataset, lookback):
    """Funding rate cross-sectional dispersion (std across symbols at each time)."""
    if not dataset.funding:
        return None

    timeline = dataset.timeline
    n = len(timeline)
    funding_matrix = np.zeros((len(SYMBOLS), n))
    for si, sym in enumerate(SYMBOLS):
        for ti, t in enumerate(timeline):
            funding_matrix[si, ti] = dataset.last_funding_rate_before(sym, t)

    # Cross-sectional std at each bar
    disp = np.std(funding_matrix, axis=0)

    if len(disp) < lookback + 50:
        return None

    z_scores = np.zeros(len(disp))
    for i in range(lookback * 2, len(disp)):
        window = disp[max(0, i - lookback):i + 1]
        mu, sigma = np.mean(window), np.std(window)
        if sigma > 0:
            z_scores[i] = (disp[i] - mu) / sigma
    return z_scores


def apply_tilt(rets, z_scores, ratio):
    min_len = min(len(rets), len(z_scores))
    tilted = rets[:min_len].copy()
    mask = z_scores[:min_len] > 0
    tilted[mask] *= ratio
    return tilted


def main():
    t0 = time.time()
    log("Phase 114: Taker Ratio + Dataset Signal Tilts")
    log("=" * 60)

    # Fetch data and compute P91b+voltilt baseline
    ds_cache = {}
    yearly_raw_rets = {}
    yearly_tilted_rets = {}  # P91b + vol tilt (current champion)

    log("Fetching data and computing champion baseline...")
    for yr in YEARS + ["2026_oos"]:
        dataset = get_dataset(yr, ds_cache)
        raw = run_p91b_returns(dataset)
        yearly_raw_rets[yr] = raw
        vz = compute_volume_z(dataset)
        if vz is not None:
            tilted = apply_vol_tilt(raw, vz)
        else:
            tilted = raw
        yearly_tilted_rets[yr] = tilted

    # Baseline (with vol tilt)
    baseline = {}
    for yr in YEARS + ["2026_oos"]:
        baseline[yr] = round(compute_sharpe(yearly_tilted_rets[yr]), 4)
    base_is = [baseline[yr] for yr in YEARS]
    base_avg = np.mean(base_is)
    base_min = np.min(base_is)
    base_obj = (base_avg + base_min) / 2
    log(f"Champion baseline (P91b+voltilt): AVG={base_avg:.4f}, MIN={base_min:.4f}, OBJ={base_obj:.4f}, OOS={baseline['2026_oos']}")

    # ── Test each new signal ────────────────────────────────────────────
    signal_builders = {
        "taker_ratio": build_taker_ratio_z,
        "vol_concentration": build_vol_concentration_z,
        "funding_dispersion": build_funding_dispersion_z,
    }

    lookbacks = [72, 168, 336]
    all_results = {}

    for sig_name, builder in signal_builders.items():
        log(f"\n{'='*60}")
        log(f"Signal: {sig_name}")
        log(f"{'='*60}")

        for lb in lookbacks:
            variant_name = f"{sig_name}_{lb}"
            log(f"\n  {variant_name}:")

            # Check data availability
            ds0 = get_dataset("2021", ds_cache)
            z0 = builder(ds0, lb)
            if z0 is None:
                log(f"    DATA UNAVAILABLE — skipping")
                all_results[variant_name] = {"verdict": "NO_DATA"}
                continue

            # IS grid search
            best_ratio = 1.0
            best_obj = base_obj

            for ratio in TILT_RATIOS:
                yr_sharpes = []
                for yr in YEARS:
                    ds = get_dataset(yr, ds_cache)
                    z = builder(ds, lb)
                    if z is not None:
                        # Apply new tilt ON TOP of existing vol-tilted returns
                        tilted = apply_tilt(yearly_tilted_rets[yr], z, ratio)
                    else:
                        tilted = yearly_tilted_rets[yr]
                    yr_sharpes.append(compute_sharpe(tilted))

                avg_s = np.mean(yr_sharpes)
                min_s = np.min(yr_sharpes)
                obj = (avg_s + min_s) / 2
                if obj > best_obj:
                    best_obj = obj
                    best_ratio = ratio

            if best_ratio == 1.0:
                log(f"    Best ratio=1.0 (no tilt) → NO VALUE")
                all_results[variant_name] = {"best_ratio": 1.0, "verdict": "NO_VALUE"}
                continue

            # Compute metrics at best ratio
            yr_sharpes = {}
            for yr in YEARS:
                ds = get_dataset(yr, ds_cache)
                z = builder(ds, lb)
                tilted = apply_tilt(yearly_tilted_rets[yr], z, best_ratio)
                yr_sharpes[yr] = round(compute_sharpe(tilted), 4)

            avg_s = np.mean(list(yr_sharpes.values()))
            min_s = np.min(list(yr_sharpes.values()))
            obj = (avg_s + min_s) / 2
            delta_obj = obj - base_obj

            # OOS
            ds_oos = get_dataset("2026_oos", ds_cache)
            z_oos = builder(ds_oos, lb)
            if z_oos is not None:
                tilted_oos = apply_tilt(yearly_tilted_rets["2026_oos"], z_oos, best_ratio)
                oos_s = round(compute_sharpe(tilted_oos), 4)
            else:
                oos_s = baseline["2026_oos"]

            log(f"    Best r={best_ratio}, ΔOBJ={delta_obj:+.4f}, OOS={oos_s}")
            for yr, s in yr_sharpes.items():
                log(f"      {yr}: {s} (Δ{s - baseline[yr]:+.4f})")

            is_promising = delta_obj > 0.02
            all_results[variant_name] = {
                "best_ratio": best_ratio,
                "avg": round(avg_s, 4),
                "min": round(min_s, 4),
                "obj": round(obj, 4),
                "delta_obj": round(delta_obj, 4),
                "oos": oos_s,
                "verdict": "PROMISING" if is_promising else "MARGINAL" if delta_obj > 0 else "NO_VALUE",
            }

            # Walk-forward if promising
            if is_promising:
                log(f"    → Walk-forward validation...")
                wf_deltas = []
                wf_ratios = []

                for wf in WF_WINDOWS:
                    train_yrs = wf["train"]
                    test_yr = wf["test"][0]

                    # Find best ratio on train
                    wf_best_r = 1.0
                    wf_best_obj = -999
                    for ratio in TILT_RATIOS:
                        yr_s = []
                        for yr in train_yrs:
                            ds = get_dataset(yr, ds_cache)
                            z = builder(ds, lb)
                            if z is not None:
                                tilted = apply_tilt(yearly_tilted_rets[yr], z, ratio)
                            else:
                                tilted = yearly_tilted_rets[yr]
                            yr_s.append(compute_sharpe(tilted))
                        avg_t = np.mean(yr_s)
                        min_t = np.min(yr_s)
                        obj_t = (avg_t + min_t) / 2
                        if obj_t > wf_best_obj:
                            wf_best_obj = obj_t
                            wf_best_r = ratio

                    # Test OOS
                    ds_test = get_dataset(test_yr, ds_cache)
                    z_test = builder(ds_test, lb)
                    if z_test is not None:
                        tilted_test = apply_tilt(yearly_tilted_rets[test_yr], z_test, wf_best_r)
                        test_s = compute_sharpe(tilted_test)
                    else:
                        test_s = compute_sharpe(yearly_tilted_rets[test_yr])

                    delta = test_s - baseline.get(test_yr, 0)
                    wf_deltas.append(delta)
                    wf_ratios.append(wf_best_r)
                    log(f"      Train={'+'.join(train_yrs)} → r={wf_best_r}, Test {test_yr}: Δ{delta:+.3f}")

                n_pos = sum(1 for d in wf_deltas if d > 0)
                avg_d = np.mean(wf_deltas)
                validated = n_pos >= 2 and avg_d > 0
                log(f"    → {n_pos}/4 positive, avg Δ={avg_d:+.4f} → {'VALIDATED' if validated else 'FAILED'}")
                all_results[variant_name]["wf_n_positive"] = n_pos
                all_results[variant_name]["wf_avg_delta"] = round(avg_d, 4)
                all_results[variant_name]["wf_validated"] = validated

    # ── Summary ─────────────────────────────────────────────────────────
    log(f"\n{'='*60}")
    log("PHASE 114 SUMMARY")
    log(f"{'='*60}")
    log(f"Champion baseline: AVG={base_avg:.4f}, MIN={base_min:.4f}, OBJ={base_obj:.4f}")

    for name, res in all_results.items():
        v = res.get("verdict", "?")
        if v == "NO_DATA":
            log(f"  {name}: NO DATA")
        elif v == "NO_VALUE":
            log(f"  {name}: NO VALUE")
        else:
            wf = f", WF={res.get('wf_n_positive', '?')}/4={'VALIDATED' if res.get('wf_validated') else 'FAILED'}" if "wf_n_positive" in res else ""
            log(f"  {name}: r={res['best_ratio']}, ΔOBJ={res.get('delta_obj', 0):+.4f}{wf}")

    validated = [k for k, v in all_results.items() if v.get("wf_validated", False)]
    if validated:
        log(f"\nVALIDATED: {validated}")
    else:
        log(f"\nNo additional tilts validated. P91b+vol_tilt remains champion.")

    # ── Save report ─────────────────────────────────────────────────────
    elapsed = time.time() - t0

    def _default(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        raise TypeError(f"Not JSON serializable: {type(obj)}")

    report = {
        "phase": 114,
        "description": "Taker ratio + dataset signals as tilts on champion",
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "baseline_obj": round(base_obj, 4),
        "results": all_results,
        "validated": validated,
    }

    report_path = os.path.join(OUT_DIR, "phase114_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=_default)

    log(f"\nPhase 114 COMPLETE in {elapsed:.1f}s → {report_path}")


if __name__ == "__main__":
    main()
