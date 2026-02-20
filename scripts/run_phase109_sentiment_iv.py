#!/usr/bin/env python3
"""
Phase 109: Sentiment + Implied Volatility Signals
===================================================
Two remaining feasible alternative data sources:

  A) Fear & Greed Index (alternative.me — try corrected URL)
     - Contrarian: buy at extreme fear, sell at extreme greed
     - Daily since 2018

  B) Deribit DVOL (BTC implied volatility index)
     - Free from Deribit API (no authentication needed for public data)
     - IV momentum/mean-reversion as volatility timing signal

  C) CoinGlass (free tier) — aggregated funding, OI, liquidations
     - May provide multi-exchange data we don't have

Test each as leverage tilt overlay on P91b, with walk-forward validation.
"""

import copy, json, os, sys, time
from datetime import datetime, timedelta

PROJ = "/Users/qtmobile/Desktop/Nexus - Quant Trading "
sys.path.insert(0, PROJ)

import numpy as np

from nexus_quant.data.providers.registry import make_provider
from nexus_quant.strategies.registry import make_strategy
from nexus_quant.backtest.engine import BacktestConfig, BacktestEngine
from nexus_quant.backtest.costs import cost_model_from_config

OUT_DIR = os.path.join(PROJ, "artifacts", "phase109")
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
    print(f"[P109] {msg}", flush=True)


def compute_sharpe(returns, bars_per_year=8760):
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size < 50:
        return 0.0
    std = float(np.std(arr))
    if std <= 0:
        return 0.0
    return float(np.mean(arr) / std * np.sqrt(bars_per_year))


def fetch_url(url, timeout=15):
    import urllib.request
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"error": str(e)}


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


def apply_tilt(hourly_rets, signal_daily, year_start, tilt_ratio):
    tilted = []
    for h_idx, ret in enumerate(hourly_rets):
        day_offset = h_idx // 24
        date_dt = datetime.strptime(year_start, "%Y-%m-%d") + timedelta(days=day_offset)
        date_str = date_dt.strftime("%Y-%m-%d")
        sig_val = signal_daily.get(date_str, None)
        if sig_val is not None and sig_val > 0:
            tilted.append(ret)
        elif sig_val is not None:
            tilted.append(ret * tilt_ratio)
        else:
            tilted.append(ret)  # no data → no tilt
    return tilted


if __name__ == "__main__":
    t0 = time.time()
    report = {"phase": 109}

    # ════════════════════════════════════
    # SECTION A: Fear & Greed Index
    # ════════════════════════════════════
    log("=" * 60)
    log("SECTION A: Fear & Greed Index")
    log("=" * 60)

    # Try multiple URL variations
    fng_urls = [
        "https://api.alternative.me/fng/?limit=2000&format=json",
        "https://api.alternative.me/fng/?limit=2000",
        "https://api.alternative.me/v2/fng/?limit=2000",
    ]

    fng_daily = {}  # date_str -> value (0-100)
    fng_raw = None
    for url in fng_urls:
        log(f"  Trying: {url}")
        result = fetch_url(url)
        if "error" not in result and "data" in result:
            fng_raw = result["data"]
            log(f"    Got {len(fng_raw)} data points")
            for entry in fng_raw:
                ts = int(entry.get("timestamp", 0))
                val = int(entry.get("value", 50))
                if ts > 0:
                    date_str = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
                    fng_daily[date_str] = val
            break
        else:
            log(f"    Failed: {result.get('error', 'no data key')}")

    if fng_daily:
        log(f"  F&G data: {len(fng_daily)} days ({min(fng_daily.keys())} to {max(fng_daily.keys())})")
        report["fng_available"] = True
        report["fng_range"] = {"start": min(fng_daily.keys()), "end": max(fng_daily.keys()), "n_days": len(fng_daily)}

        # Create contrarian signals from F&G:
        # - Buy signal when F&G < 25 (extreme fear)
        # - Sell signal when F&G > 75 (extreme greed)
        # - Neutral when 25-75
        fng_signals = {}

        # Signal 1: Contrarian — positive when fear, negative when greed
        contrarian = {}
        for date, val in fng_daily.items():
            contrarian[date] = 50 - val  # fear → positive, greed → negative
        fng_signals["fng_contrarian"] = contrarian

        # Signal 2: Extreme fear only (buy the dip)
        extreme_fear = {}
        for date, val in fng_daily.items():
            extreme_fear[date] = 1.0 if val < 25 else (-1.0 if val > 75 else 0.0)
        fng_signals["fng_extreme"] = extreme_fear

        # Signal 3: F&G momentum (rising fear → improving, rising greed → deteriorating)
        dates_sorted = sorted(fng_daily.keys())
        fng_mom_14 = {}
        for i, date in enumerate(dates_sorted):
            if i < 14:
                continue
            prev_date = dates_sorted[i - 14]
            delta = fng_daily[date] - fng_daily[prev_date]
            fng_mom_14[date] = -delta  # contrarian: falling F&G (more fear) = positive
        fng_signals["fng_mom_14"] = fng_mom_14

    else:
        log("  F&G data NOT available")
        report["fng_available"] = False
        fng_signals = {}

    # ════════════════════════════════════
    # SECTION B: Deribit DVOL (BTC IV)
    # ════════════════════════════════════
    log("\n" + "=" * 60)
    log("SECTION B: Deribit DVOL (BTC Implied Volatility)")
    log("=" * 60)

    # Deribit public API for DVOL
    dvol_urls = [
        "https://www.deribit.com/api/v2/public/get_volatility_index_data?currency=BTC&resolution=86400&start_timestamp=1609459200000&end_timestamp=1740009600000",
        "https://deribit.com/api/v2/public/get_volatility_index_data?currency=BTC&resolution=86400&start_timestamp=1609459200000&end_timestamp=1740009600000",
    ]

    dvol_daily = {}
    for url in dvol_urls:
        log(f"  Trying Deribit DVOL: {url[:80]}...")
        result = fetch_url(url, timeout=20)
        if "error" not in result and "result" in result:
            data = result["result"].get("data", [])
            log(f"    Got {len(data)} data points")
            for entry in data:
                ts_ms = entry[0]
                close_iv = entry[4]  # [ts, open, high, low, close]
                date_str = datetime.utcfromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d")
                dvol_daily[date_str] = close_iv
            break
        else:
            err = result.get("error", "unknown")
            log(f"    Failed: {err}")

    if dvol_daily:
        log(f"  DVOL data: {len(dvol_daily)} days ({min(dvol_daily.keys())} to {max(dvol_daily.keys())})")
        report["dvol_available"] = True

        # IV signals:
        # - High IV → reduce exposure (volatility is expensive)
        # - Low IV → increase exposure (cheap entry)
        # - IV mean-reversion: when IV is high relative to rolling mean, expect it to fall

        dvol_signals = {}
        dates_sorted = sorted(dvol_daily.keys())

        # Signal: IV z-score (negative z = low vol environment = bullish for our strategy)
        for lookback in [14, 30]:
            z_scores = {}
            for i, date in enumerate(dates_sorted):
                if i < lookback:
                    continue
                window = [dvol_daily[dates_sorted[j]] for j in range(i - lookback, i)]
                mean = sum(window) / len(window)
                std = (sum((v - mean) ** 2 for v in window) / len(window)) ** 0.5
                if std > 0:
                    z_scores[date] = -(dvol_daily[date] - mean) / std  # negative: high IV → negative signal
                else:
                    z_scores[date] = 0.0
            dvol_signals[f"dvol_negz_{lookback}"] = z_scores
            log(f"    dvol_negz_{lookback}: {len(z_scores)} points")

        # IV momentum (falling IV → positive for strategy)
        for lookback in [7, 14]:
            iv_mom = {}
            for i, date in enumerate(dates_sorted):
                if i < lookback:
                    continue
                prev = dates_sorted[i - lookback]
                if dvol_daily[prev] > 0:
                    iv_mom[date] = -(dvol_daily[date] / dvol_daily[prev] - 1.0)  # falling IV → positive
                else:
                    iv_mom[date] = 0.0
            dvol_signals[f"dvol_mom_{lookback}"] = iv_mom
            log(f"    dvol_mom_{lookback}: {len(iv_mom)} points")
    else:
        log("  DVOL data NOT available")
        report["dvol_available"] = False
        dvol_signals = {}

    # ════════════════════════════════════
    # SECTION C: Test all signals as P91b tilt overlay
    # ════════════════════════════════════
    log("\n" + "=" * 60)
    log("SECTION C: Tilt overlay tests")
    log("=" * 60)

    # Combine all signals
    all_signals = {}
    all_signals.update(fng_signals)
    all_signals.update(dvol_signals)

    if not all_signals:
        log("  No alternative data signals available — skipping tilt tests")
        report["tilt_results"] = {}
        report["verdict"] = "NO ALTERNATIVE DATA ACCESSIBLE — APIs unavailable or data insufficient"
    else:
        # Precompute P91b returns
        log("  Precomputing P91b returns...")
        p91b_rets = {}
        for year in YEARS:
            start, end = YEAR_RANGES[year]
            sig_rets = {}
            for sig_key in SIG_KEYS:
                try:
                    sig_rets[sig_key] = run_signal(SIGNALS[sig_key], start, end)
                except:
                    sig_rets[sig_key] = []
            p91b_rets[year] = blend(sig_rets, P91B_WEIGHTS)

        # OOS
        sig_rets_oos = {}
        for sig_key in SIG_KEYS:
            try:
                sig_rets_oos[sig_key] = run_signal(SIGNALS[sig_key], OOS_RANGE[0], OOS_RANGE[1])
            except:
                sig_rets_oos[sig_key] = []
        p91b_rets["2026_oos"] = blend(sig_rets_oos, P91B_WEIGHTS)

        # Baseline
        baseline = {y: round(compute_sharpe(p91b_rets[y]), 4) for y in YEARS}
        baseline["2026_oos"] = round(compute_sharpe(p91b_rets["2026_oos"]), 4)
        baseline_avg = round(sum(baseline[y] for y in YEARS) / len(YEARS), 4)
        baseline_min = round(min(baseline[y] for y in YEARS), 4)
        log(f"  Baseline: AVG={baseline_avg}, MIN={baseline_min}, OOS={baseline['2026_oos']}")

        # Test each signal at multiple tilt ratios
        TILT_RATIOS = [0.3, 0.5, 0.7, 0.8]
        tilt_results = {}

        for sig_name, signal_daily in all_signals.items():
            if len(signal_daily) < 100:
                log(f"  {sig_name}: too few data points ({len(signal_daily)}), skipping")
                continue

            best_obj = -999
            best_ratio = 1.0
            best_data = None

            for ratio in TILT_RATIOS:
                yearly_sharpes = {}
                for year in YEARS:
                    start = YEAR_RANGES[year][0]
                    tilted = apply_tilt(p91b_rets[year], signal_daily, start, ratio)
                    yearly_sharpes[year] = round(compute_sharpe(tilted), 4)

                # OOS
                oos_tilted = apply_tilt(p91b_rets["2026_oos"], signal_daily, OOS_RANGE[0], ratio)
                oos_s = round(compute_sharpe(oos_tilted), 4)

                vals = list(yearly_sharpes.values())
                avg_s = round(sum(vals) / len(vals), 4)
                min_s = round(min(vals), 4)
                obj = round((avg_s + min_s) / 2, 4)

                if obj > best_obj:
                    best_obj = obj
                    best_ratio = ratio
                    best_data = {
                        "yearly": yearly_sharpes, "avg": avg_s, "min": min_s,
                        "obj": obj, "oos": oos_s,
                    }

            delta_avg = round(best_data["avg"] - baseline_avg, 4)
            delta_min = round(best_data["min"] - baseline_min, 4)
            delta_oos = round(best_data["oos"] - baseline["2026_oos"], 4)

            tilt_results[sig_name] = {
                "best_ratio": best_ratio,
                **best_data,
                "delta_avg": delta_avg, "delta_min": delta_min, "delta_oos": delta_oos,
            }
            log(f"  {sig_name} @ r={best_ratio}: AVG={best_data['avg']:.3f} (Δ{delta_avg:+.3f}), "
                f"MIN={best_data['min']:.3f} (Δ{delta_min:+.3f}), OOS={best_data['oos']:.3f} (Δ{delta_oos:+.3f})")

        report["tilt_results"] = tilt_results

        # Verdict
        best_variant = max(tilt_results.items(), key=lambda x: x[1]["obj"]) if tilt_results else (None, None)
        if best_variant[1] and best_variant[1]["obj"] > (baseline_avg + baseline_min) / 2:
            verdict = f"POTENTIAL: {best_variant[0]} at r={best_variant[1]['best_ratio']}, OBJ={best_variant[1]['obj']}"
            verdict += " — needs walk-forward validation"
        else:
            verdict = "NO IMPROVEMENT from sentiment/IV signals"

        report["verdict"] = verdict
        log(f"\n  VERDICT: {verdict}")

    # ════════════════════════════════════
    # SAVE
    # ════════════════════════════════════
    elapsed = round(time.time() - t0, 1)
    report["elapsed_seconds"] = elapsed

    out_path = os.path.join(OUT_DIR, "phase109_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    log(f"\nPhase 109 COMPLETE in {elapsed}s → {out_path}")
