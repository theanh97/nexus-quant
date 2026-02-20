#!/usr/bin/env python3
"""
Phase 106: On-Chain Signal — Blockchain.com Data
==================================================
Build and test signals from on-chain BTC metrics:
  - Hashrate momentum (rising hashrate = miner confidence = bullish)
  - Transaction count momentum (rising activity = demand)
  - Mempool size (congestion as volatility/activity proxy)
  - Composite on-chain score

All from blockchain.com free API (daily, years of history).
Test as:
  A) Standalone signal (daily Sharpe on 2021-2025 + 2026 OOS)
  B) Filter for P91b ensemble (only trade when on-chain is positive)
  C) Weight tilter (increase weight when on-chain is bullish)

CRITICAL: On-chain data is DAILY. Our strategy is HOURLY.
We use on-chain as a daily filter/overlay, not as the primary signal.
"""

import copy, json, math, os, sys, time
from pathlib import Path
from datetime import datetime, timedelta

PROJ = "/Users/qtmobile/Desktop/Nexus - Quant Trading "
sys.path.insert(0, PROJ)

import numpy as np

from nexus_quant.data.providers.registry import make_provider
from nexus_quant.strategies.registry import make_strategy
from nexus_quant.backtest.engine import BacktestConfig, BacktestEngine
from nexus_quant.backtest.costs import cost_model_from_config

OUT_DIR = os.path.join(PROJ, "artifacts", "phase106")
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


def log(msg):
    print(f"[P106] {msg}", flush=True)


def compute_sharpe(returns, bars_per_year=8760):
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size < 50:
        return 0.0
    std = float(np.std(arr))
    if std <= 0:
        return 0.0
    return float(np.mean(arr) / std * np.sqrt(bars_per_year))


# ════════════════════════════════════
# On-chain data fetching
# ════════════════════════════════════
def fetch_blockchain_chart(chart_name, timespan="5years"):
    """Fetch daily chart data from blockchain.com API."""
    import urllib.request
    url = f"https://api.blockchain.info/charts/{chart_name}?timespan={timespan}&format=json&rollingAverage=24hours"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            values = data.get("values", [])
            return [(v["x"], v["y"]) for v in values]  # (unix_ts, value)
    except Exception as e:
        log(f"  ERROR fetching {chart_name}: {e}")
        return []


def resample_to_daily(ts_data):
    """Convert (unix_ts, value) pairs to daily dict {date_str: value}."""
    daily = {}
    for ts, val in ts_data:
        dt = datetime.utcfromtimestamp(ts)
        date_str = dt.strftime("%Y-%m-%d")
        daily[date_str] = val
    return daily


def compute_z_score(series, lookback=30):
    """Compute rolling z-score for each day."""
    dates = sorted(series.keys())
    z_scores = {}
    for i, date in enumerate(dates):
        if i < lookback:
            continue
        window = [series[dates[j]] for j in range(i - lookback, i)]
        mean = sum(window) / len(window)
        std = (sum((v - mean) ** 2 for v in window) / len(window)) ** 0.5
        if std > 0:
            z_scores[date] = (series[date] - mean) / std
        else:
            z_scores[date] = 0.0
    return z_scores


def compute_momentum(series, lookback=14):
    """Compute momentum (% change over lookback days)."""
    dates = sorted(series.keys())
    momentum = {}
    for i, date in enumerate(dates):
        if i < lookback:
            continue
        prev_date = dates[i - lookback]
        if series[prev_date] > 0:
            momentum[date] = (series[date] / series[prev_date]) - 1.0
        else:
            momentum[date] = 0.0
    return momentum


def run_signal(sig_cfg, start, end):
    data_cfg = {
        "provider": "binance_rest_v1", "symbols": SYMBOLS,
        "start": start, "end": end, "bar_interval": "1h",
        "cache_dir": ".cache/binance_rest",
    }
    costs_cfg = {"fee_rate": 0.0005, "slippage_rate": 0.0003}
    exec_cfg = {"style": "taker", "slippage_bps": 3.0}
    provider = make_provider(data_cfg, seed=42)
    dataset = provider.load()
    strategy = make_strategy({"name": sig_cfg["name"], "params": copy.deepcopy(sig_cfg["params"])})
    cost_model = cost_model_from_config(costs_cfg, execution_cfg=exec_cfg)
    engine = BacktestEngine(BacktestConfig(costs=cost_model))
    result = engine.run(dataset=dataset, strategy=strategy, seed=42)
    return result.returns


def blend(sig_rets, weights):
    keys = sorted(weights.keys())
    n = min(len(sig_rets.get(k, [])) for k in keys)
    if n == 0:
        return []
    R = np.zeros((len(keys), n), dtype=np.float64)
    W = np.array([weights[k] for k in keys], dtype=np.float64)
    for i, k in enumerate(keys):
        R[i, :] = sig_rets[k][:n]
    return (W @ R).tolist()


def date_to_hour_index(date_str, year_start):
    """Convert date string to approximate hour index within a year."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    year_dt = datetime.strptime(year_start, "%Y-%m-%d")
    delta = dt - year_dt
    return int(delta.total_seconds() / 3600)


if __name__ == "__main__":
    t0 = time.time()
    report = {"phase": 106}

    # ════════════════════════════════════
    # SECTION A: Fetch on-chain data
    # ════════════════════════════════════
    log("=" * 60)
    log("SECTION A: Fetching on-chain data from blockchain.com")
    log("=" * 60)

    metrics = {
        "hash_rate": "hash-rate",
        "n_transactions": "n-transactions",
        "mempool_size": "mempool-size",
        "n_unique_addresses": "n-unique-addresses",
    }

    onchain_daily = {}
    for name, chart_id in metrics.items():
        log(f"  Fetching {name} ({chart_id})...")
        raw = fetch_blockchain_chart(chart_id, timespan="5years")
        daily = resample_to_daily(raw)
        onchain_daily[name] = daily
        log(f"    Got {len(daily)} daily points ({min(daily.keys()) if daily else '?'} to {max(daily.keys()) if daily else '?'})")

    report["onchain_data"] = {
        name: {"n_points": len(daily), "start": min(daily.keys()) if daily else None, "end": max(daily.keys()) if daily else None}
        for name, daily in onchain_daily.items()
    }

    # ════════════════════════════════════
    # SECTION B: Compute on-chain signals
    # ════════════════════════════════════
    log("\n" + "=" * 60)
    log("SECTION B: Computing on-chain signals")
    log("=" * 60)

    # Signal variants with different lookbacks
    ONCHAIN_VARIANTS = [
        {"name": "hash_mom_14", "metric": "hash_rate", "type": "momentum", "lookback": 14},
        {"name": "hash_mom_30", "metric": "hash_rate", "type": "momentum", "lookback": 30},
        {"name": "tx_mom_14", "metric": "n_transactions", "type": "momentum", "lookback": 14},
        {"name": "tx_mom_30", "metric": "n_transactions", "type": "momentum", "lookback": 30},
        {"name": "addr_mom_14", "metric": "n_unique_addresses", "type": "momentum", "lookback": 14},
        {"name": "mempool_z_30", "metric": "mempool_size", "type": "zscore", "lookback": 30},
    ]

    onchain_signals = {}
    for var in ONCHAIN_VARIANTS:
        daily_data = onchain_daily.get(var["metric"], {})
        if not daily_data:
            log(f"  {var['name']}: no data")
            continue
        if var["type"] == "momentum":
            signal = compute_momentum(daily_data, var["lookback"])
        elif var["type"] == "zscore":
            signal = compute_z_score(daily_data, var["lookback"])
        else:
            continue
        onchain_signals[var["name"]] = signal
        log(f"  {var['name']}: {len(signal)} daily signal points")

    # Composite: average z-score of hashrate + tx + addresses momentum
    composite_signals = {}
    for lookback in [14, 30]:
        name = f"composite_{lookback}d"
        h_mom = compute_momentum(onchain_daily.get("hash_rate", {}), lookback)
        t_mom = compute_momentum(onchain_daily.get("n_transactions", {}), lookback)
        a_mom = compute_momentum(onchain_daily.get("n_unique_addresses", {}), lookback)

        # Normalize each to z-score
        all_dates = sorted(set(h_mom.keys()) & set(t_mom.keys()) & set(a_mom.keys()))

        composite = {}
        for date in all_dates:
            score = (h_mom[date] + t_mom[date] + a_mom[date]) / 3.0
            composite[date] = score

        # Now compute z-score of composite
        composite_z = compute_z_score(composite, lookback)
        onchain_signals[name] = composite_z
        log(f"  {name}: {len(composite_z)} daily composite z-score points")

    # ════════════════════════════════════
    # SECTION C: Test as filter for P91b
    # ════════════════════════════════════
    log("\n" + "=" * 60)
    log("SECTION C: On-chain as filter for P91b ensemble")
    log("=" * 60)

    # Precompute P91b returns per year
    log("  Precomputing P91b returns per year...")
    p91b_rets_by_year = {}
    for year in YEARS:
        start, end = YEAR_RANGES[year]
        sig_rets = {}
        for sig_key in SIG_KEYS:
            try:
                sig_rets[sig_key] = run_signal(SIGNALS[sig_key], start, end)
            except Exception as exc:
                sig_rets[sig_key] = []
        p91b_rets_by_year[year] = blend(sig_rets, P91B_WEIGHTS)
        log(f"    {year}: {len(p91b_rets_by_year[year])} hourly returns")

    # Baseline (no filter)
    baseline_sharpes = {}
    for year in YEARS:
        baseline_sharpes[year] = round(compute_sharpe(p91b_rets_by_year[year]), 4)
    baseline_avg = round(sum(baseline_sharpes.values()) / len(baseline_sharpes), 4)
    baseline_min = round(min(baseline_sharpes.values()), 4)
    log(f"  Baseline P91b: {baseline_sharpes} → AVG={baseline_avg}, MIN={baseline_min}")

    # Filter: only keep hourly returns when on-chain signal is positive
    # If on-chain signal is negative, set returns to 0 (flat, no trading)
    filter_results = {}
    for sig_name, signal_daily in onchain_signals.items():
        yearly_sharpes = {}
        for year in YEARS:
            year_start = YEAR_RANGES[year][0]
            hourly_rets = p91b_rets_by_year[year]
            if not hourly_rets:
                yearly_sharpes[year] = 0.0
                continue

            # Map daily signal to hourly: each day covers 24 hours
            filtered_rets = []
            for h_idx, ret in enumerate(hourly_rets):
                day_offset = h_idx // 24
                date_dt = datetime.strptime(year_start, "%Y-%m-%d") + timedelta(days=day_offset)
                date_str = date_dt.strftime("%Y-%m-%d")
                sig_val = signal_daily.get(date_str, 0.0)

                # Filter: trade only when signal > 0 (positive on-chain momentum)
                if sig_val > 0:
                    filtered_rets.append(ret)
                else:
                    filtered_rets.append(0.0)  # flat

            yearly_sharpes[year] = round(compute_sharpe(filtered_rets), 4)

        vals = list(yearly_sharpes.values())
        avg_s = round(sum(vals) / len(vals), 4)
        min_s = round(min(vals), 4)
        delta_avg = round(avg_s - baseline_avg, 4)
        delta_min = round(min_s - baseline_min, 4)

        # What % of time is filter active?
        active_pct = {}
        for year in YEARS:
            year_start = YEAR_RANGES[year][0]
            n_bars = len(p91b_rets_by_year[year])
            active = 0
            for h_idx in range(n_bars):
                day_offset = h_idx // 24
                date_dt = datetime.strptime(year_start, "%Y-%m-%d") + timedelta(days=day_offset)
                date_str = date_dt.strftime("%Y-%m-%d")
                if signal_daily.get(date_str, 0.0) > 0:
                    active += 1
            active_pct[year] = round(active / n_bars * 100, 1) if n_bars > 0 else 0

        filter_results[sig_name] = {
            "yearly": yearly_sharpes,
            "avg": avg_s, "min": min_s,
            "delta_avg": delta_avg, "delta_min": delta_min,
            "active_pct": active_pct,
        }
        log(f"  {sig_name}: AVG={avg_s:.3f} (Δ{delta_avg:+.3f}), MIN={min_s:.3f} (Δ{delta_min:+.3f}), "
            f"active ~{sum(active_pct.values())/len(active_pct):.0f}%")

    report["filter_results"] = filter_results
    report["baseline"] = {"yearly": baseline_sharpes, "avg": baseline_avg, "min": baseline_min}

    # ════════════════════════════════════
    # SECTION D: On-chain as weight tilter
    # ════════════════════════════════════
    log("\n" + "=" * 60)
    log("SECTION D: On-chain as leverage tilter")
    log("=" * 60)

    # Instead of binary filter, scale leverage:
    # When on-chain > 0: use full leverage (1.0x)
    # When on-chain < 0: reduce to 0.5x leverage
    tilt_results = {}
    for sig_name in ["composite_14d", "composite_30d", "hash_mom_14", "tx_mom_14"]:
        signal_daily = onchain_signals.get(sig_name, {})
        if not signal_daily:
            continue

        yearly_sharpes = {}
        for year in YEARS:
            year_start = YEAR_RANGES[year][0]
            hourly_rets = p91b_rets_by_year[year]
            if not hourly_rets:
                yearly_sharpes[year] = 0.0
                continue

            tilted_rets = []
            for h_idx, ret in enumerate(hourly_rets):
                day_offset = h_idx // 24
                date_dt = datetime.strptime(year_start, "%Y-%m-%d") + timedelta(days=day_offset)
                date_str = date_dt.strftime("%Y-%m-%d")
                sig_val = signal_daily.get(date_str, 0.0)

                # Tilt: 1.0x when positive, 0.5x when negative
                if sig_val > 0:
                    tilted_rets.append(ret)
                else:
                    tilted_rets.append(ret * 0.5)

            yearly_sharpes[year] = round(compute_sharpe(tilted_rets), 4)

        vals = list(yearly_sharpes.values())
        avg_s = round(sum(vals) / len(vals), 4)
        min_s = round(min(vals), 4)
        delta_avg = round(avg_s - baseline_avg, 4)
        delta_min = round(min_s - baseline_min, 4)

        tilt_results[sig_name] = {
            "yearly": yearly_sharpes,
            "avg": avg_s, "min": min_s,
            "delta_avg": delta_avg, "delta_min": delta_min,
        }
        log(f"  {sig_name} tilt: AVG={avg_s:.3f} (Δ{delta_avg:+.3f}), MIN={min_s:.3f} (Δ{delta_min:+.3f})")

    report["tilt_results"] = tilt_results

    # ════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════
    elapsed = round(time.time() - t0, 1)
    report["elapsed_seconds"] = elapsed

    out_path = os.path.join(OUT_DIR, "phase106_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    log("\n" + "=" * 60)
    log(f"Phase 106 COMPLETE in {elapsed}s → {out_path}")
    log("=" * 60)

    # Find best filter
    best_filter = max(filter_results, key=lambda k: filter_results[k]["min"]) if filter_results else "none"
    best_f_data = filter_results.get(best_filter, {})
    log(f"\nBest filter: {best_filter}")
    log(f"  AVG={best_f_data.get('avg', 0):.3f} (Δ{best_f_data.get('delta_avg', 0):+.3f}), "
        f"MIN={best_f_data.get('min', 0):.3f} (Δ{best_f_data.get('delta_min', 0):+.3f})")

    best_tilt = max(tilt_results, key=lambda k: tilt_results[k]["min"]) if tilt_results else "none"
    best_t_data = tilt_results.get(best_tilt, {})
    log(f"\nBest tilt: {best_tilt}")
    log(f"  AVG={best_t_data.get('avg', 0):.3f} (Δ{best_t_data.get('delta_avg', 0):+.3f}), "
        f"MIN={best_t_data.get('min', 0):.3f} (Δ{best_t_data.get('delta_min', 0):+.3f})")

    if best_f_data.get("delta_min", 0) > 0 or best_t_data.get("delta_min", 0) > 0:
        log("\nVERDICT: On-chain signal shows POTENTIAL — further optimization warranted")
    else:
        log("\nVERDICT: On-chain signal does NOT improve P91b MIN — daily resolution too coarse")
