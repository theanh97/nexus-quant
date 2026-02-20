#!/usr/bin/env python3
"""
Phase 115: Cross-Asset Regime Tilt
===================================
Test if TradFi regime indicators can improve the crypto champion.

Sources (free APIs):
1. Yahoo Finance: ^GSPC (S&P500), GC=F (Gold), DX-Y.NYB (DXY)
2. All daily data, years of history

Signals:
- sp500_mom_N: S&P500 momentum z-score
- gold_mom_N: Gold momentum z-score
- dxy_mom_N: Dollar index momentum z-score (inverse: strong dollar = risk-off)
- risk_composite: average z-score of sp500_mom + gold_mom - dxy_mom

Tilt: when risk signal is negative (risk-off), reduce leverage.
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

OUT_DIR = os.path.join(PROJ, "artifacts", "phase115")
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
    print(f"[P115] {msg}", flush=True)


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
            if total_vol is None: total_vol = vols.copy()
            else:
                ml = min(len(total_vol), len(vols))
                total_vol = total_vol[:ml] + vols[:ml]
    if total_vol is None or len(total_vol) < lookback + 50: return None
    log_vol = np.log(np.maximum(total_vol, 1.0))
    mom = np.zeros(len(log_vol))
    mom[lookback:] = log_vol[lookback:] - log_vol[:-lookback]
    z_scores = np.zeros(len(mom))
    for i in range(lookback * 2, len(mom)):
        w = mom[max(0, i - lookback):i + 1]
        mu, sigma = np.mean(w), np.std(w)
        if sigma > 0: z_scores[i] = (mom[i] - mu) / sigma
    return z_scores


def apply_vol_tilt(rets, z_scores, ratio=VOL_TILT_RATIO):
    ml = min(len(rets), len(z_scores))
    tilted = rets[:ml].copy()
    tilted[z_scores[:ml] > 0] *= ratio
    return tilted


# ── Fetch TradFi data via Yahoo Finance ──────────────────────────────────
def fetch_yahoo_daily(symbol, start_date="2020-06-01", end_date="2026-02-21"):
    """Fetch daily OHLC from Yahoo Finance v8 API."""
    import urllib.request
    from datetime import datetime as dt

    start_ts = int(dt.strptime(start_date, "%Y-%m-%d").timestamp())
    end_ts = int(dt.strptime(end_date, "%Y-%m-%d").timestamp())

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={start_ts}&period2={end_ts}&interval=1d"
    headers = {"User-Agent": "NexusQuant/1.0"}

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())

        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        closes = result["indicators"]["quote"][0]["close"]

        # Build {epoch_day: close_price}
        daily = {}
        for ts, c in zip(timestamps, closes):
            if c is not None and c > 0:
                day_epoch = (ts // 86400) * 86400
                daily[day_epoch] = float(c)

        return daily
    except Exception as e:
        log(f"  Yahoo API error for {symbol}: {e}")
        return {}


def build_momentum_z(daily_prices, lookback_days):
    """Build daily momentum z-score from price series."""
    ts_list = sorted(daily_prices.keys())
    if len(ts_list) < lookback_days + 50:
        return {}

    prices = [daily_prices[t] for t in ts_list]
    log_p = np.log(prices)
    mom = np.zeros(len(log_p))
    mom[lookback_days:] = log_p[lookback_days:] - log_p[:-lookback_days]

    z_scores = {}
    for i in range(lookback_days * 2, len(mom)):
        window = mom[max(0, i - lookback_days):i + 1]
        mu, sigma = np.mean(window), np.std(window)
        if sigma > 0:
            z_scores[ts_list[i]] = (mom[i] - mu) / sigma

    return z_scores


def apply_daily_tilt(hourly_rets, daily_signal, year_start_epoch, tilt_ratio, direction="risk_on"):
    """Apply daily signal tilt on hourly returns.

    direction='risk_on': when signal < 0 (risk-off), reduce leverage
    direction='risk_off': when signal > 0, reduce leverage
    """
    tilted = hourly_rets.copy()
    n = len(tilted)
    day_ts = sorted(daily_signal.keys())
    if not day_ts:
        return tilted

    day_idx = 0
    count_tilted = 0
    for i in range(n):
        hour_epoch = year_start_epoch + i * 3600
        day_epoch = (hour_epoch // 86400) * 86400

        while day_idx < len(day_ts) - 1 and day_ts[day_idx + 1] <= day_epoch:
            day_idx += 1

        if day_ts[day_idx] <= day_epoch:
            z = daily_signal[day_ts[day_idx]]
            if direction == "risk_on" and z < 0:
                tilted[i] *= tilt_ratio
                count_tilted += 1
            elif direction == "risk_off" and z > 0:
                tilted[i] *= tilt_ratio
                count_tilted += 1

    return tilted


def get_year_start(yr):
    if yr == "2026_oos":
        return int(datetime(2026, 1, 1).timestamp())
    return int(datetime(int(yr), 1, 1).timestamp())


def main():
    t0 = time.time()
    log("Phase 115: Cross-Asset Regime Tilt")
    log("=" * 60)

    # ── Step 1: Fetch TradFi data ───────────────────────────────────────
    log("Fetching TradFi data from Yahoo Finance...")

    tradfi_data = {}
    for ticker, name in [("^GSPC", "sp500"), ("GC=F", "gold"), ("DX-Y.NYB", "dxy")]:
        prices = fetch_yahoo_daily(ticker)
        tradfi_data[name] = prices
        if prices:
            ts_sorted = sorted(prices.keys())
            d0 = datetime.fromtimestamp(ts_sorted[0]).strftime("%Y-%m-%d")
            d1 = datetime.fromtimestamp(ts_sorted[-1]).strftime("%Y-%m-%d")
            log(f"  {name} ({ticker}): {len(prices)} days ({d0} to {d1})")
        else:
            log(f"  {name} ({ticker}): FAILED")

    if not any(tradfi_data.values()):
        log("All TradFi data fetches failed. Cannot proceed.")
        return

    # ── Step 2: Compute P91b+voltilt baseline ───────────────────────────
    log("\nComputing champion baseline...")
    ds_cache = {}
    yearly_rets = {}  # champion (with vol tilt)

    for yr in YEARS + ["2026_oos"]:
        dataset = get_dataset(yr, ds_cache)
        raw = run_p91b_returns(dataset)
        vz = compute_volume_z(dataset)
        if vz is not None:
            tilted = apply_vol_tilt(raw, vz)
        else:
            tilted = raw
        yearly_rets[yr] = tilted

    baseline = {}
    for yr in YEARS + ["2026_oos"]:
        baseline[yr] = round(compute_sharpe(yearly_rets[yr]), 4)
    base_is = [baseline[yr] for yr in YEARS]
    base_avg = np.mean(base_is)
    base_min = np.min(base_is)
    base_obj = (base_avg + base_min) / 2
    log(f"Baseline: AVG={base_avg:.4f}, MIN={base_min:.4f}, OBJ={base_obj:.4f}, OOS={baseline['2026_oos']}")

    # ── Step 3: Test cross-asset tilts ──────────────────────────────────
    log("\n" + "=" * 60)
    log("Cross-Asset Tilt Tests")
    log("=" * 60)

    lookback_days = [14, 30, 60]
    all_results = {}

    # Individual assets
    for asset_name, direction in [("sp500", "risk_on"), ("gold", "risk_on"), ("dxy", "risk_off")]:
        prices = tradfi_data.get(asset_name, {})
        if not prices:
            log(f"\n  {asset_name}: NO DATA")
            continue

        for lb in lookback_days:
            variant = f"{asset_name}_mom_{lb}"
            z_signal = build_momentum_z(prices, lb)
            if not z_signal:
                log(f"\n  {variant}: insufficient data for z-score")
                continue

            log(f"\n  {variant} (direction={direction}):")

            best_ratio = 1.0
            best_obj = base_obj

            for ratio in TILT_RATIOS:
                yr_sharpes = []
                for yr in YEARS:
                    yr_start = get_year_start(yr)
                    tilted = apply_daily_tilt(yearly_rets[yr], z_signal, yr_start, ratio, direction)
                    yr_sharpes.append(compute_sharpe(tilted))

                avg_s = np.mean(yr_sharpes)
                min_s = np.min(yr_sharpes)
                obj = (avg_s + min_s) / 2
                if obj > best_obj:
                    best_obj = obj
                    best_ratio = ratio

            if best_ratio == 1.0:
                log(f"    NO VALUE")
                all_results[variant] = {"verdict": "NO_VALUE"}
            else:
                # Compute full metrics
                yr_sharpes = {}
                for yr in YEARS:
                    yr_start = get_year_start(yr)
                    tilted = apply_daily_tilt(yearly_rets[yr], z_signal, yr_start, best_ratio, direction)
                    yr_sharpes[yr] = round(compute_sharpe(tilted), 4)

                avg_s = np.mean(list(yr_sharpes.values()))
                min_s = np.min(list(yr_sharpes.values()))
                obj = (avg_s + min_s) / 2
                delta = obj - base_obj

                # OOS
                yr_start_oos = get_year_start("2026_oos")
                tilted_oos = apply_daily_tilt(yearly_rets["2026_oos"], z_signal, yr_start_oos, best_ratio, direction)
                oos_s = round(compute_sharpe(tilted_oos), 4)

                log(f"    r={best_ratio}, ΔOBJ={delta:+.4f}, OOS={oos_s}")

                is_promising = delta > 0.02
                all_results[variant] = {
                    "best_ratio": best_ratio, "delta_obj": round(delta, 4),
                    "oos": oos_s, "verdict": "PROMISING" if is_promising else "MARGINAL",
                }

                # WF if promising
                if is_promising:
                    log(f"    → Walk-forward validation...")
                    wf_deltas = []
                    for wf in WF_WINDOWS:
                        train_yrs = wf["train"]
                        test_yr = wf["test"][0]

                        wf_best_r = 1.0
                        wf_best_obj = -999
                        for ratio in TILT_RATIOS:
                            yr_s = []
                            for yr in train_yrs:
                                tilted = apply_daily_tilt(yearly_rets[yr], z_signal, get_year_start(yr), ratio, direction)
                                yr_s.append(compute_sharpe(tilted))
                            obj_t = (np.mean(yr_s) + np.min(yr_s)) / 2
                            if obj_t > wf_best_obj:
                                wf_best_obj = obj_t
                                wf_best_r = ratio

                        tilted_test = apply_daily_tilt(yearly_rets[test_yr], z_signal, get_year_start(test_yr), wf_best_r, direction)
                        test_s = compute_sharpe(tilted_test)
                        delta_wf = test_s - baseline.get(test_yr, 0)
                        wf_deltas.append(delta_wf)
                        log(f"      Train={'+'.join(train_yrs)} → r={wf_best_r}, Test {test_yr}: Δ{delta_wf:+.3f}")

                    n_pos = sum(1 for d in wf_deltas if d > 0)
                    avg_d = np.mean(wf_deltas)
                    validated = n_pos >= 2 and avg_d > 0
                    log(f"    → {n_pos}/4 positive, avg Δ={avg_d:+.4f} → {'VALIDATED' if validated else 'FAILED'}")
                    all_results[variant]["wf_validated"] = validated

    # ── Composite risk signal ───────────────────────────────────────────
    log(f"\n{'='*60}")
    log("Composite Risk Signal")
    log(f"{'='*60}")

    for lb in lookback_days:
        variant = f"risk_composite_{lb}"
        log(f"\n  {variant}:")

        # Build composite: sp500_z + gold_z - dxy_z (risk-on composite)
        sp_z = build_momentum_z(tradfi_data.get("sp500", {}), lb)
        gold_z = build_momentum_z(tradfi_data.get("gold", {}), lb)
        dxy_z = build_momentum_z(tradfi_data.get("dxy", {}), lb)

        if not sp_z and not gold_z:
            log(f"    Insufficient data")
            continue

        # Merge z-scores by day
        all_days = set()
        for z in [sp_z, gold_z, dxy_z]:
            all_days.update(z.keys())

        composite = {}
        for day in sorted(all_days):
            vals = []
            if day in sp_z: vals.append(sp_z[day])
            if day in gold_z: vals.append(gold_z[day])
            if day in dxy_z: vals.append(-dxy_z[day])  # inverse: strong dollar = risk off
            if vals:
                composite[day] = np.mean(vals)

        if not composite:
            log(f"    No composite data")
            continue

        best_ratio = 1.0
        best_obj = base_obj

        for ratio in TILT_RATIOS:
            yr_sharpes = []
            for yr in YEARS:
                tilted = apply_daily_tilt(yearly_rets[yr], composite, get_year_start(yr), ratio, "risk_on")
                yr_sharpes.append(compute_sharpe(tilted))

            obj = (np.mean(yr_sharpes) + np.min(yr_sharpes)) / 2
            if obj > best_obj:
                best_obj = obj
                best_ratio = ratio

        if best_ratio == 1.0:
            log(f"    NO VALUE")
            all_results[variant] = {"verdict": "NO_VALUE"}
        else:
            yr_sharpes = {}
            for yr in YEARS:
                tilted = apply_daily_tilt(yearly_rets[yr], composite, get_year_start(yr), best_ratio, "risk_on")
                yr_sharpes[yr] = round(compute_sharpe(tilted), 4)
            avg_s = np.mean(list(yr_sharpes.values()))
            min_s = np.min(list(yr_sharpes.values()))
            obj = (avg_s + min_s) / 2
            delta = obj - base_obj

            tilted_oos = apply_daily_tilt(yearly_rets["2026_oos"], composite, get_year_start("2026_oos"), best_ratio, "risk_on")
            oos_s = round(compute_sharpe(tilted_oos), 4)

            log(f"    r={best_ratio}, ΔOBJ={delta:+.4f}, OOS={oos_s}")
            all_results[variant] = {
                "best_ratio": best_ratio, "delta_obj": round(delta, 4),
                "oos": oos_s, "verdict": "PROMISING" if delta > 0.02 else "MARGINAL",
            }

    # ── Summary ─────────────────────────────────────────────────────────
    log(f"\n{'='*60}")
    log("PHASE 115 SUMMARY")
    log(f"{'='*60}")
    log(f"Champion baseline: AVG={base_avg:.4f}, MIN={base_min:.4f}, OBJ={base_obj:.4f}")

    for name, res in all_results.items():
        v = res.get("verdict", "?")
        if v == "NO_VALUE":
            log(f"  {name}: NO VALUE")
        else:
            wf = f", WF={'VALIDATED' if res.get('wf_validated') else 'FAILED'}" if "wf_validated" in res else ""
            log(f"  {name}: r={res.get('best_ratio', '?')}, ΔOBJ={res.get('delta_obj', 0):+.4f}{wf}")

    validated = [k for k, v in all_results.items() if v.get("wf_validated", False)]
    if validated:
        log(f"\nVALIDATED: {validated}")
    else:
        log(f"\nNo cross-asset tilts validated. Champion unchanged.")

    # ── Save ────────────────────────────────────────────────────────────
    elapsed = time.time() - t0

    def _default(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    report = {
        "phase": 115, "description": "Cross-asset regime tilt",
        "timestamp": datetime.now().isoformat(), "elapsed_seconds": round(elapsed, 1),
        "tradfi_data": {k: len(v) for k, v in tradfi_data.items()},
        "baseline_obj": round(base_obj, 4),
        "results": all_results, "validated": validated,
    }

    report_path = os.path.join(OUT_DIR, "phase115_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=_default)

    log(f"\nPhase 115 COMPLETE in {elapsed:.1f}s → {report_path}")


if __name__ == "__main__":
    main()
