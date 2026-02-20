#!/usr/bin/env python3
"""
Phase 112: Full Integration Test — P91b + vol_mom_z_168 Tilt
=============================================================
Phase 111 validated vol_mom_z_168 @ r=0.7 as a leverage overlay:
- 3/4 WF positive, avg Δ=+0.066, ratio perfectly stable

This phase:
1. Full year-by-year IS metrics (Sharpe, MDD, CAGR, Sortino, Calmar)
2. OOS 2026 metrics
3. Comprehensive comparison: baseline vs tilted
4. Combined with composite_30 (also validated in P108)
5. Final champion determination
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

OUT_DIR = os.path.join(PROJ, "artifacts", "phase112")
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

VOL_TILT_LOOKBACK = 168
VOL_TILT_RATIO = 0.7


def log(msg):
    print(f"[P112] {msg}", flush=True)


def compute_sharpe(returns, bars_per_year=8760):
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size < 50:
        return 0.0
    std = float(np.std(arr))
    if std <= 0:
        return 0.0
    return float(np.mean(arr) / std * np.sqrt(bars_per_year))


def compute_sortino(returns, bars_per_year=8760):
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size < 50:
        return 0.0
    neg = arr[arr < 0]
    if len(neg) == 0:
        return 999.0
    downside_std = float(np.std(neg))
    if downside_std <= 0:
        return 0.0
    return float(np.mean(arr) / downside_std * np.sqrt(bars_per_year))


def compute_mdd(returns):
    arr = np.asarray(returns, dtype=np.float64)
    equity = np.cumprod(1 + arr)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(np.min(dd))


def compute_cagr(returns, bars_per_year=8760):
    arr = np.asarray(returns, dtype=np.float64)
    total = float(np.prod(1 + arr))
    years = len(arr) / bars_per_year
    if years <= 0 or total <= 0:
        return 0.0
    return float(total ** (1 / years) - 1)


def compute_calmar(returns, bars_per_year=8760):
    cagr = compute_cagr(returns, bars_per_year)
    mdd = compute_mdd(returns)
    if mdd >= 0:
        return 0.0
    return cagr / abs(mdd)


def compute_full_metrics(returns, bars_per_year=8760):
    return {
        "sharpe": round(compute_sharpe(returns, bars_per_year), 4),
        "sortino": round(compute_sortino(returns, bars_per_year), 4),
        "mdd": round(compute_mdd(returns) * 100, 2),  # as percentage
        "cagr": round(compute_cagr(returns, bars_per_year) * 100, 2),  # as percentage
        "calmar": round(compute_calmar(returns, bars_per_year), 4),
        "n_bars": len(returns),
    }


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


def compute_volume_z(dataset, lookback=VOL_TILT_LOOKBACK):
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
        mu = np.mean(window)
        sigma = np.std(window)
        if sigma > 0:
            z_scores[i] = (mom[i] - mu) / sigma
    return z_scores


def apply_vol_tilt(hourly_rets, z_scores, ratio=VOL_TILT_RATIO):
    min_len = min(len(hourly_rets), len(z_scores))
    tilted = hourly_rets[:min_len].copy()
    mask = z_scores[:min_len] > 0
    tilted[mask] *= ratio
    pct_tilted = mask.sum() / len(mask) * 100
    return tilted, pct_tilted


def fetch_onchain_composite(lookback=30):
    """Fetch blockchain.com composite signal (from Phase 108 validated)."""
    import urllib.request
    charts = ["hash-rate", "n-transactions", "n-unique-addresses", "mempool-size"]
    all_data = {}
    for chart in charts:
        url = f"https://api.blockchain.info/charts/{chart}?timespan=6years&format=json&rollingAverage=24hours"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NexusQuant/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
            values = {d["x"]: d["y"] for d in data.get("values", [])}
            all_data[chart] = values
            log(f"  {chart}: {len(values)} points")
        except Exception as e:
            log(f"  {chart}: ERROR {e}")
    if not all_data:
        return None

    # Compute composite momentum z-score
    all_ts = set()
    for v in all_data.values():
        all_ts.update(v.keys())
    all_ts = sorted(all_ts)

    # For each chart: momentum then z-score
    chart_z = {}
    for chart, values in all_data.items():
        ts_vals = [(t, values[t]) for t in all_ts if t in values and values[t] > 0]
        if len(ts_vals) < lookback + 50:
            continue
        ts_arr = [t for t, _ in ts_vals]
        val_arr = np.log([v for _, v in ts_vals])
        mom = np.zeros(len(val_arr))
        mom[lookback:] = val_arr[lookback:] - val_arr[:-lookback]

        z = np.zeros(len(mom))
        for i in range(lookback * 2, len(mom)):
            window = mom[max(0, i - lookback):i + 1]
            mu = np.mean(window)
            sigma = np.std(window)
            if sigma > 0:
                z[i] = (mom[i] - mu) / sigma
        chart_z[chart] = dict(zip(ts_arr, z))

    if not chart_z:
        return None

    # Average z-scores across charts (daily)
    composite = {}
    for ts in all_ts:
        vals = [chart_z[c].get(ts, 0) for c in chart_z if ts in chart_z[c]]
        if vals:
            composite[ts] = np.mean(vals)
    return composite


def apply_onchain_tilt(hourly_rets, composite_daily, year_start_epoch, tilt_ratio=0.8):
    """Apply composite_30 tilt: when composite z > 0 (overheated), reduce to tilt_ratio."""
    tilted = hourly_rets.copy()
    n = len(tilted)
    day_ts = sorted(composite_daily.keys())
    if not day_ts:
        return tilted, 0.0

    day_idx = 0
    for i in range(n):
        hour_epoch = year_start_epoch + i * 3600
        day_epoch = hour_epoch // 86400 * 86400

        while day_idx < len(day_ts) - 1 and day_ts[day_idx + 1] <= day_epoch:
            day_idx += 1

        if day_ts[day_idx] <= day_epoch:
            z = composite_daily[day_ts[day_idx]]
            if z > 0:
                tilted[i] *= tilt_ratio

    pct = sum(1 for i in range(n) if tilted[i] != hourly_rets[i]) / n * 100
    return tilted, pct


def main():
    t0 = time.time()
    log("Phase 112: Full Integration Test")
    log("=" * 60)

    # Fetch datasets and compute baselines
    ds_cache = {}
    yearly_rets = {}

    log("\nFetching data and computing P91b baselines...")
    for yr in YEARS + ["2026_oos"]:
        dataset = get_dataset(yr, ds_cache)
        rets = run_p91b_returns(dataset)
        yearly_rets[yr] = rets

    # ── Baseline metrics ────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("BASELINE (P91b without tilt)")
    log("=" * 60)

    baseline_metrics = {}
    for yr in YEARS + ["2026_oos"]:
        m = compute_full_metrics(yearly_rets[yr])
        baseline_metrics[yr] = m
        log(f"  {yr}: Sharpe={m['sharpe']}, MDD={m['mdd']}%, CAGR={m['cagr']}%, Sortino={m['sortino']}, Calmar={m['calmar']}")

    is_sharpes = [baseline_metrics[yr]["sharpe"] for yr in YEARS]
    base_avg = np.mean(is_sharpes)
    base_min = np.min(is_sharpes)
    base_obj = (base_avg + base_min) / 2
    log(f"\n  IS: AVG={base_avg:.4f}, MIN={base_min:.4f}, OBJ={base_obj:.4f}")

    # ── Vol tilt metrics ────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("VARIANT A: P91b + vol_mom_z_168 @ r=0.7")
    log("=" * 60)

    voltilt_metrics = {}
    for yr in YEARS + ["2026_oos"]:
        dataset = get_dataset(yr, ds_cache)
        z_scores = compute_volume_z(dataset)
        if z_scores is not None:
            tilted, pct = apply_vol_tilt(yearly_rets[yr], z_scores)
        else:
            tilted = yearly_rets[yr]
            pct = 0
        m = compute_full_metrics(tilted)
        voltilt_metrics[yr] = m
        delta_s = m["sharpe"] - baseline_metrics[yr]["sharpe"]
        log(f"  {yr}: Sharpe={m['sharpe']} (Δ{delta_s:+.4f}), MDD={m['mdd']}%, CAGR={m['cagr']}%, tilt_active={pct:.1f}%")

    vt_sharpes = [voltilt_metrics[yr]["sharpe"] for yr in YEARS]
    vt_avg = np.mean(vt_sharpes)
    vt_min = np.min(vt_sharpes)
    vt_obj = (vt_avg + vt_min) / 2
    log(f"\n  IS: AVG={vt_avg:.4f}, MIN={vt_min:.4f}, OBJ={vt_obj:.4f}")
    log(f"  vs baseline: ΔAVG={vt_avg - base_avg:+.4f}, ΔMIN={vt_min - base_min:+.4f}, ΔOBJ={vt_obj - base_obj:+.4f}")

    # ── On-chain composite_30 tilt ──────────────────────────────────────
    log("\n" + "=" * 60)
    log("VARIANT B: P91b + composite_30 @ r=0.8")
    log("=" * 60)

    log("Fetching on-chain data...")
    composite = fetch_onchain_composite(lookback=30)

    if composite:
        oc_metrics = {}
        for yr in YEARS + ["2026_oos"]:
            if yr == "2026_oos":
                yr_start = int(datetime(2026, 1, 1).timestamp())
            else:
                yr_start = int(datetime(int(yr), 1, 1).timestamp())
            tilted, pct = apply_onchain_tilt(yearly_rets[yr], composite, yr_start, tilt_ratio=0.8)
            m = compute_full_metrics(tilted)
            oc_metrics[yr] = m
            delta_s = m["sharpe"] - baseline_metrics[yr]["sharpe"]
            log(f"  {yr}: Sharpe={m['sharpe']} (Δ{delta_s:+.4f}), MDD={m['mdd']}%, tilt_active={pct:.1f}%")

        oc_sharpes = [oc_metrics[yr]["sharpe"] for yr in YEARS]
        oc_avg = np.mean(oc_sharpes)
        oc_min = np.min(oc_sharpes)
        oc_obj = (oc_avg + oc_min) / 2
        log(f"\n  IS: AVG={oc_avg:.4f}, MIN={oc_min:.4f}, OBJ={oc_obj:.4f}")
        log(f"  vs baseline: ΔAVG={oc_avg - base_avg:+.4f}, ΔMIN={oc_min - base_min:+.4f}, ΔOBJ={oc_obj - base_obj:+.4f}")
    else:
        log("  On-chain data unavailable, skipping variant B")
        oc_metrics = None

    # ── Combined: vol tilt + on-chain tilt ──────────────────────────────
    log("\n" + "=" * 60)
    log("VARIANT C: P91b + vol_mom_z_168 + composite_30 (double tilt)")
    log("=" * 60)

    if composite:
        combo_metrics = {}
        for yr in YEARS + ["2026_oos"]:
            # First apply vol tilt
            dataset = get_dataset(yr, ds_cache)
            z_scores = compute_volume_z(dataset)
            if z_scores is not None:
                step1, _ = apply_vol_tilt(yearly_rets[yr], z_scores)
            else:
                step1 = yearly_rets[yr]

            # Then apply on-chain tilt
            if yr == "2026_oos":
                yr_start = int(datetime(2026, 1, 1).timestamp())
            else:
                yr_start = int(datetime(int(yr), 1, 1).timestamp())
            tilted, pct = apply_onchain_tilt(step1, composite, yr_start, tilt_ratio=0.8)
            m = compute_full_metrics(tilted)
            combo_metrics[yr] = m
            delta_s = m["sharpe"] - baseline_metrics[yr]["sharpe"]
            log(f"  {yr}: Sharpe={m['sharpe']} (Δ{delta_s:+.4f}), MDD={m['mdd']}%")

        cb_sharpes = [combo_metrics[yr]["sharpe"] for yr in YEARS]
        cb_avg = np.mean(cb_sharpes)
        cb_min = np.min(cb_sharpes)
        cb_obj = (cb_avg + cb_min) / 2
        log(f"\n  IS: AVG={cb_avg:.4f}, MIN={cb_min:.4f}, OBJ={cb_obj:.4f}")
        log(f"  vs baseline: ΔAVG={cb_avg - base_avg:+.4f}, ΔMIN={cb_min - base_min:+.4f}, ΔOBJ={cb_obj - base_obj:+.4f}")
    else:
        combo_metrics = None

    # ── Final comparison ────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("FINAL COMPARISON")
    log("=" * 60)

    variants = [
        ("Baseline (P91b)", base_avg, base_min, base_obj, baseline_metrics.get("2026_oos", {}).get("sharpe", 0)),
        ("A: +vol_tilt_168", vt_avg, vt_min, vt_obj, voltilt_metrics.get("2026_oos", {}).get("sharpe", 0)),
    ]
    if oc_metrics:
        variants.append(("B: +composite_30", oc_avg, oc_min, oc_obj, oc_metrics.get("2026_oos", {}).get("sharpe", 0)))
    if combo_metrics:
        variants.append(("C: +both tilts", cb_avg, cb_min, cb_obj, combo_metrics.get("2026_oos", {}).get("sharpe", 0)))

    log(f"  {'Variant':<22} {'AVG':>7} {'MIN':>7} {'OBJ':>7} {'OOS':>7}")
    log(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for name, avg, mn, obj, oos in variants:
        log(f"  {name:<22} {avg:>7.4f} {mn:>7.4f} {obj:>7.4f} {oos:>7.4f}")

    # Determine champion
    best = max(variants, key=lambda x: x[3])  # by OBJ
    log(f"\n  CHAMPION: {best[0]} (OBJ={best[3]:.4f})")

    if best[0] == "Baseline (P91b)":
        log("  NO IMPROVEMENT — P91b remains unchanged champion")
    else:
        log(f"  IMPROVEMENT over baseline: ΔOBJ={best[3] - base_obj:+.4f}")

    # ── Save report ─────────────────────────────────────────────────────
    elapsed = time.time() - t0

    def _default(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    report = {
        "phase": 112,
        "description": "Full integration test — P91b + overlays",
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "baseline": {
            "yearly": baseline_metrics,
            "is_avg": round(base_avg, 4),
            "is_min": round(base_min, 4),
            "is_obj": round(base_obj, 4),
        },
        "variant_a_vol_tilt": {
            "config": {"lookback": VOL_TILT_LOOKBACK, "ratio": VOL_TILT_RATIO},
            "yearly": voltilt_metrics,
            "is_avg": round(vt_avg, 4),
            "is_min": round(vt_min, 4),
            "is_obj": round(vt_obj, 4),
        },
        "variant_b_onchain": {
            "yearly": oc_metrics,
            "is_avg": round(oc_avg, 4) if oc_metrics else None,
            "is_min": round(oc_min, 4) if oc_metrics else None,
            "is_obj": round(oc_obj, 4) if oc_metrics else None,
        } if oc_metrics else None,
        "variant_c_combined": {
            "yearly": combo_metrics,
            "is_avg": round(cb_avg, 4) if combo_metrics else None,
            "is_min": round(cb_min, 4) if combo_metrics else None,
            "is_obj": round(cb_obj, 4) if combo_metrics else None,
        } if combo_metrics else None,
        "champion": best[0],
        "champion_obj": round(best[3], 4),
    }

    report_path = os.path.join(OUT_DIR, "phase112_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=_default)

    log(f"\nPhase 112 COMPLETE in {elapsed:.1f}s → {report_path}")


if __name__ == "__main__":
    main()
