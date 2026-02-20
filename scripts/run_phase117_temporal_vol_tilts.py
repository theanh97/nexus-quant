#!/usr/bin/env python3
"""
Phase 117: Temporal & Volatility Regime Tilts
==============================================
Last untested overlay directions using data already in dataset:

1. Day-of-week tilt: reduce on historically weak days
2. Realized vol regime: reduce when recent vol is elevated (z > 0)
3. Return autocorrelation: reduce when autocorr is high (trend exhaustion)
4. Drawdown tilt: reduce when in drawdown (momentum slowdown)

All tested on top of current champion (P91b + vol_tilt_168).
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

OUT_DIR = os.path.join(PROJ, "artifacts", "phase117")
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

TILT_RATIOS = [round(r * 0.1, 1) for r in range(11)]

WF_WINDOWS = [
    {"train": ["2021", "2022", "2023"], "test": ["2024"]},
    {"train": ["2022", "2023", "2024"], "test": ["2025"]},
    {"train": ["2021", "2022", "2023", "2024"], "test": ["2025"]},
    {"train": ["2021", "2022", "2023", "2024", "2025"], "test": ["2026_oos"]},
]

VOL_TILT_LB = 168
VOL_TILT_RATIO = 0.65


def log(msg):
    print(f"[P117] {msg}", flush=True)


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
            "provider": "binance_rest_v1", "symbols": SYMBOLS,
            "bar_interval": "1h", "start": start, "end": end,
        }, seed=42)
        cache[cache_key] = provider.load()
    return cache[cache_key]


def run_champion_returns(year_key, ds_cache):
    """Run P91b + vol_tilt (current champion)."""
    dataset = get_dataset(year_key, ds_cache)
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
    ens = np.zeros(min_len)
    for i, k in enumerate(SIG_KEYS):
        ens += weights[i] * all_returns[i]

    # Vol tilt
    total_vol = None
    if dataset.perp_volume:
        for sym in SYMBOLS:
            vols = np.array(dataset.perp_volume.get(sym, []), dtype=np.float64)
            if total_vol is None: total_vol = vols.copy()
            else:
                ml = min(len(total_vol), len(vols))
                total_vol = total_vol[:ml] + vols[:ml]

    if total_vol is not None and len(total_vol) >= VOL_TILT_LB + 50:
        log_vol = np.log(np.maximum(total_vol, 1.0))
        mom = np.zeros(len(log_vol))
        mom[VOL_TILT_LB:] = log_vol[VOL_TILT_LB:] - log_vol[:-VOL_TILT_LB]
        z = np.zeros(len(mom))
        for i in range(VOL_TILT_LB * 2, len(mom)):
            w = mom[max(0, i - VOL_TILT_LB):i + 1]
            mu, sigma = np.mean(w), np.std(w)
            if sigma > 0: z[i] = (mom[i] - mu) / sigma
        ml = min(len(ens), len(z))
        ens = ens[:ml]
        ens[z[:ml] > 0] *= VOL_TILT_RATIO

    return ens


# ── Signal builders (from champion returns) ─────────────────────────────
def build_realized_vol_z(rets, lookback):
    """Rolling realized vol z-score of returns."""
    n = len(rets)
    z_scores = np.zeros(n)
    for i in range(lookback * 2, n):
        window = rets[max(0, i - lookback):i + 1]
        vol = np.std(window)
        # z-score of current vol vs longer history
        hist_vols = []
        for j in range(lookback * 2, i + 1):
            hist_vols.append(np.std(rets[max(0, j - lookback):j + 1]))
        if len(hist_vols) > 10:
            mu, sigma = np.mean(hist_vols), np.std(hist_vols)
            if sigma > 0:
                z_scores[i] = (vol - mu) / sigma
    return z_scores


def build_realized_vol_z_fast(rets, lookback):
    """Fast rolling realized vol z-score using numpy."""
    n = len(rets)
    if n < lookback * 2 + 50:
        return np.zeros(n)

    # Rolling std
    rolling_vol = np.zeros(n)
    for i in range(lookback, n):
        rolling_vol[i] = np.std(rets[i - lookback:i + 1])

    # z-score of rolling vol
    z_scores = np.zeros(n)
    for i in range(lookback * 2, n):
        window = rolling_vol[max(lookback, i - lookback):i + 1]
        mu, sigma = np.mean(window), np.std(window)
        if sigma > 0:
            z_scores[i] = (rolling_vol[i] - mu) / sigma

    return z_scores


def build_autocorr_z(rets, lookback, lag=1):
    """Rolling autocorrelation z-score."""
    n = len(rets)
    z_scores = np.zeros(n)
    for i in range(lookback * 2, n):
        window = rets[i - lookback:i + 1]
        if len(window) > lag + 10:
            ac = np.corrcoef(window[lag:], window[:-lag])[0, 1]
            if not np.isnan(ac):
                # z-score vs history
                hist_acs = []
                for j in range(lookback * 2, i + 1):
                    w = rets[j - lookback:j + 1]
                    if len(w) > lag + 10:
                        c = np.corrcoef(w[lag:], w[:-lag])[0, 1]
                        if not np.isnan(c):
                            hist_acs.append(c)
                if len(hist_acs) > 10:
                    mu, sigma = np.mean(hist_acs), np.std(hist_acs)
                    if sigma > 0:
                        z_scores[i] = (ac - mu) / sigma
    return z_scores


def build_drawdown_indicator(rets, lookback):
    """Rolling drawdown z-score: are we in drawdown?"""
    n = len(rets)
    equity = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.maximum(peak, 1e-10)

    z_scores = np.zeros(n)
    for i in range(lookback * 2, n):
        window = dd[max(0, i - lookback):i + 1]
        mu, sigma = np.mean(window), np.std(window)
        if sigma > 0:
            z_scores[i] = (dd[i] - mu) / sigma  # negative = deeper drawdown
    # We want to tilt when drawdown is deep (z < 0)
    # Invert: positive = drawdown, to match tilt convention (z > 0 → reduce)
    return -z_scores


def apply_tilt(rets, z_scores, ratio):
    ml = min(len(rets), len(z_scores))
    tilted = rets[:ml].copy()
    tilted[z_scores[:ml] > 0] *= ratio
    return tilted


def main():
    t0 = time.time()
    log("Phase 117: Temporal & Vol Regime Tilts")
    log("=" * 60)

    ds_cache = {}
    yearly_rets = {}

    log("Computing champion returns...")
    for yr in YEARS + ["2026_oos"]:
        yearly_rets[yr] = run_champion_returns(yr, ds_cache)

    baseline = {}
    for yr in YEARS + ["2026_oos"]:
        baseline[yr] = round(compute_sharpe(yearly_rets[yr]), 4)
    base_is = [baseline[yr] for yr in YEARS]
    base_avg = np.mean(base_is)
    base_min = np.min(base_is)
    base_obj = (base_avg + base_min) / 2
    log(f"Baseline: AVG={base_avg:.4f}, MIN={base_min:.4f}, OBJ={base_obj:.4f}, OOS={baseline['2026_oos']}")

    # ── Test signals ────────────────────────────────────────────────────
    all_results = {}

    signal_specs = [
        ("rvol_z_72", lambda rets: build_realized_vol_z_fast(rets, 72)),
        ("rvol_z_168", lambda rets: build_realized_vol_z_fast(rets, 168)),
        ("rvol_z_336", lambda rets: build_realized_vol_z_fast(rets, 336)),
        ("dd_z_72", lambda rets: build_drawdown_indicator(rets, 72)),
        ("dd_z_168", lambda rets: build_drawdown_indicator(rets, 168)),
        ("dd_z_336", lambda rets: build_drawdown_indicator(rets, 336)),
    ]

    for sig_name, builder in signal_specs:
        log(f"\n  Signal: {sig_name}")

        best_ratio = 1.0
        best_obj = base_obj

        for ratio in TILT_RATIOS:
            yr_sharpes = []
            for yr in YEARS:
                z = builder(yearly_rets[yr])
                tilted = apply_tilt(yearly_rets[yr], z, ratio)
                yr_sharpes.append(compute_sharpe(tilted))

            avg_s = np.mean(yr_sharpes)
            min_s = np.min(yr_sharpes)
            obj = (avg_s + min_s) / 2
            if obj > best_obj:
                best_obj = obj
                best_ratio = ratio

        if best_ratio == 1.0:
            log(f"    NO VALUE")
            all_results[sig_name] = {"verdict": "NO_VALUE"}
            continue

        # Full metrics
        yr_sharpes = {}
        for yr in YEARS:
            z = builder(yearly_rets[yr])
            tilted = apply_tilt(yearly_rets[yr], z, best_ratio)
            yr_sharpes[yr] = round(compute_sharpe(tilted), 4)

        avg_s = np.mean(list(yr_sharpes.values()))
        min_s = np.min(list(yr_sharpes.values()))
        obj = (avg_s + min_s) / 2
        delta = obj - base_obj

        z_oos = builder(yearly_rets["2026_oos"])
        tilted_oos = apply_tilt(yearly_rets["2026_oos"], z_oos, best_ratio)
        oos_s = round(compute_sharpe(tilted_oos), 4)

        log(f"    r={best_ratio}, ΔOBJ={delta:+.4f}, OOS={oos_s}")

        is_promising = delta > 0.02
        all_results[sig_name] = {
            "best_ratio": best_ratio, "delta_obj": round(delta, 4),
            "oos": oos_s, "verdict": "PROMISING" if is_promising else "MARGINAL",
        }

        if is_promising:
            log(f"    → Walk-forward validation...")
            wf_deltas = []
            for wf in WF_WINDOWS:
                train_yrs, test_yr = wf["train"], wf["test"][0]

                wf_best_r, wf_best_obj = 1.0, -999
                for ratio in TILT_RATIOS:
                    yr_s = []
                    for yr in train_yrs:
                        z = builder(yearly_rets[yr])
                        tilted = apply_tilt(yearly_rets[yr], z, ratio)
                        yr_s.append(compute_sharpe(tilted))
                    obj_t = (np.mean(yr_s) + np.min(yr_s)) / 2
                    if obj_t > wf_best_obj:
                        wf_best_obj, wf_best_r = obj_t, ratio

                z_test = builder(yearly_rets[test_yr])
                tilted_test = apply_tilt(yearly_rets[test_yr], z_test, wf_best_r)
                test_s = compute_sharpe(tilted_test)
                delta_wf = test_s - baseline.get(test_yr, 0)
                wf_deltas.append(delta_wf)
                log(f"      Train={'+'.join(train_yrs)} → r={wf_best_r}, Test {test_yr}: Δ{delta_wf:+.3f}")

            n_pos = sum(1 for d in wf_deltas if d > 0)
            avg_d = np.mean(wf_deltas)
            validated = n_pos >= 2 and avg_d > 0
            log(f"    → {n_pos}/4 positive, avg Δ={avg_d:+.4f} → {'VALIDATED' if validated else 'FAILED'}")
            all_results[sig_name]["wf_validated"] = validated

    # ── Summary ─────────────────────────────────────────────────────────
    log(f"\n{'='*60}")
    log("PHASE 117 SUMMARY")
    log(f"{'='*60}")

    for name, res in all_results.items():
        v = res.get("verdict", "?")
        if v == "NO_VALUE":
            log(f"  {name}: NO VALUE")
        else:
            wf = f", WF={'VALIDATED' if res.get('wf_validated') else 'FAILED'}" if "wf_validated" in res else ""
            log(f"  {name}: r={res.get('best_ratio', '?')}, ΔOBJ={res.get('delta_obj', 0):+.4f}{wf}")

    validated = [k for k, v in all_results.items() if v.get("wf_validated")]
    if validated:
        log(f"\nVALIDATED: {validated}")
    else:
        log(f"\nNo additional tilts validated. Champion FINAL.")

    # Save
    elapsed = time.time() - t0
    def _default(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        return str(obj)

    report = {
        "phase": 117, "description": "Temporal & vol regime tilts",
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "baseline_obj": round(base_obj, 4),
        "results": all_results, "validated": validated,
    }

    report_path = os.path.join(OUT_DIR, "phase117_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=_default)

    log(f"\nPhase 117 COMPLETE in {elapsed:.1f}s → {report_path}")


if __name__ == "__main__":
    main()
