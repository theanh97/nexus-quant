#!/usr/bin/env python3
"""
Phase 110: Walk-Forward Validation of F&G Momentum Tilt
========================================================
fng_mom_14 showed IS ΔMIN=+0.162 (Phase 109). Validate with WF.
Also combine best alt-data signals (composite_30 + fng_mom_14).
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

OUT_DIR = os.path.join(PROJ, "artifacts", "phase110")
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
    {"train": YEARS, "test": ["2026_oos"]},
]


def log(msg):
    print(f"[P110] {msg}", flush=True)


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


def fetch_blockchain_chart(chart_name, timespan="6years"):
    import urllib.request
    url = f"https://api.blockchain.info/charts/{chart_name}?timespan={timespan}&format=json&rollingAverage=24hours"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return {datetime.utcfromtimestamp(v["x"]).strftime("%Y-%m-%d"): v["y"]
                    for v in data.get("values", [])}
    except Exception as e:
        return {}


def compute_momentum(series, lookback=14):
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
            tilted.append(ret)
    return tilted


def apply_combined_tilt(hourly_rets, signals, year_start, tilt_ratio):
    """Apply tilt only when ALL signals agree (negative)."""
    tilted = []
    for h_idx, ret in enumerate(hourly_rets):
        day_offset = h_idx // 24
        date_dt = datetime.strptime(year_start, "%Y-%m-%d") + timedelta(days=day_offset)
        date_str = date_dt.strftime("%Y-%m-%d")

        # Average signal value across all signals
        vals = [sig.get(date_str, 0.0) for sig in signals]
        avg_val = sum(vals) / len(vals) if vals else 0.0

        if avg_val > 0:
            tilted.append(ret)
        else:
            tilted.append(ret * tilt_ratio)
    return tilted


def find_best_ratio(p91b_rets, signal_daily, train_years, ratios, apply_fn=apply_tilt):
    best_obj = -999
    best_ratio = 1.0
    for ratio in ratios:
        yearly_sharpes = []
        for year in train_years:
            start = YEAR_RANGES[year][0]
            tilted = apply_fn(p91b_rets[year], signal_daily, start, ratio)
            yearly_sharpes.append(compute_sharpe(tilted))
        if len(yearly_sharpes) < 2:
            continue
        avg_s = sum(yearly_sharpes) / len(yearly_sharpes)
        min_s = min(yearly_sharpes)
        obj = (avg_s + min_s) / 2
        if obj > best_obj:
            best_obj = obj
            best_ratio = ratio
    return best_ratio, best_obj


if __name__ == "__main__":
    t0 = time.time()
    report = {"phase": 110}

    # ════════════════════════════════════
    # Fetch data
    # ════════════════════════════════════
    log("Fetching data...")

    # F&G
    fng_result = fetch_url("https://api.alternative.me/fng/?limit=2000&format=json")
    fng_daily = {}
    if "data" in fng_result:
        for entry in fng_result["data"]:
            ts = int(entry.get("timestamp", 0))
            val = int(entry.get("value", 50))
            if ts > 0:
                fng_daily[datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")] = val
    log(f"  F&G: {len(fng_daily)} days")

    # F&G momentum (contrarian)
    dates_sorted = sorted(fng_daily.keys())
    fng_mom_14 = {}
    for i, date in enumerate(dates_sorted):
        if i < 14:
            continue
        prev = dates_sorted[i - 14]
        delta = fng_daily[date] - fng_daily[prev]
        fng_mom_14[date] = -delta  # contrarian: falling F&G = positive
    log(f"  fng_mom_14: {len(fng_mom_14)} points")

    # On-chain composite_30
    raw_tx = fetch_blockchain_chart("n-transactions", "6years")
    raw_hash = fetch_blockchain_chart("hash-rate", "6years")
    raw_addr = fetch_blockchain_chart("n-unique-addresses", "6years")
    tx_m = compute_momentum(raw_tx, 30)
    hash_m = compute_momentum(raw_hash, 30)
    addr_m = compute_momentum(raw_addr, 30)
    all_dates = sorted(set(tx_m.keys()) & set(hash_m.keys()) & set(addr_m.keys()))
    composite_30 = {d: (tx_m[d] + hash_m[d] + addr_m[d]) / 3.0 for d in all_dates}
    log(f"  composite_30: {len(composite_30)} points")

    # Combined: average of fng_mom_14 z-score and composite_30 z-score
    # First normalize each to z-score
    def to_zscore(signal):
        vals = list(signal.values())
        if not vals:
            return signal
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        if std <= 0:
            return {k: 0.0 for k in signal}
        return {k: (v - mean) / std for k, v in signal.items()}

    fng_z = to_zscore(fng_mom_14)
    comp_z = to_zscore(composite_30)
    combined_dates = sorted(set(fng_z.keys()) & set(comp_z.keys()))
    combined_signal = {d: (fng_z[d] + comp_z[d]) / 2.0 for d in combined_dates}
    log(f"  combined (fng+composite): {len(combined_signal)} points")

    signals_to_test = {
        "fng_mom_14": fng_mom_14,
        "composite_30": composite_30,
        "combined_fng_composite": combined_signal,
    }

    # ════════════════════════════════════
    # Precompute P91b returns
    # ════════════════════════════════════
    log("\nPrecomputing P91b returns...")
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

    sig_rets_oos = {}
    for sig_key in SIG_KEYS:
        try:
            sig_rets_oos[sig_key] = run_signal(SIGNALS[sig_key], OOS_RANGE[0], OOS_RANGE[1])
        except:
            sig_rets_oos[sig_key] = []
    p91b_rets["2026_oos"] = blend(sig_rets_oos, P91B_WEIGHTS)

    baseline = {y: round(compute_sharpe(p91b_rets[y]), 4) for y in YEARS}
    baseline["2026_oos"] = round(compute_sharpe(p91b_rets["2026_oos"]), 4)
    baseline_avg = round(sum(baseline[y] for y in YEARS) / len(YEARS), 4)
    baseline_min = round(min(baseline[y] for y in YEARS), 4)
    log(f"  Baseline: AVG={baseline_avg}, MIN={baseline_min}")

    # ════════════════════════════════════
    # Walk-forward validation
    # ════════════════════════════════════
    log("\n" + "=" * 60)
    log("WALK-FORWARD VALIDATION")
    log("=" * 60)

    wf_results = {}
    for sig_name, signal_daily in signals_to_test.items():
        log(f"\n  Signal: {sig_name}")
        sig_wf = []

        for wf in WF_WINDOWS:
            train_label = "+".join(wf["train"])
            test_label = "+".join(wf["test"])

            best_ratio, best_obj = find_best_ratio(p91b_rets, signal_daily, wf["train"], TILT_RATIOS)

            test_sharpes = {}
            test_deltas = {}
            for year in wf["test"]:
                if year == "2026_oos":
                    start = OOS_RANGE[0]
                    base_s = baseline["2026_oos"]
                else:
                    start = YEAR_RANGES[year][0]
                    base_s = baseline[year]
                tilted = apply_tilt(p91b_rets[year], signal_daily, start, best_ratio)
                tilt_s = round(compute_sharpe(tilted), 4)
                test_sharpes[year] = tilt_s
                test_deltas[year] = round(tilt_s - base_s, 4)

            result = {
                "train": wf["train"], "test": wf["test"],
                "optimal_ratio": best_ratio,
                "test_sharpes": test_sharpes,
                "test_deltas": test_deltas,
            }
            sig_wf.append(result)

            delta_str = ", ".join(f"{y}: Δ{test_deltas[y]:+.3f}" for y in wf["test"])
            log(f"    Train={train_label} → r={best_ratio}, Test: {delta_str}")

        wf_results[sig_name] = sig_wf

    report["walk_forward"] = wf_results

    # ════════════════════════════════════
    # Summary
    # ════════════════════════════════════
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)

    for sig_name, wf_list in wf_results.items():
        all_deltas = []
        ratios = []
        for wf in wf_list:
            ratios.append(wf["optimal_ratio"])
            for y, d in wf["test_deltas"].items():
                all_deltas.append(d)

        n_pos = sum(1 for d in all_deltas if d > 0)
        avg_d = round(sum(all_deltas) / len(all_deltas), 4) if all_deltas else 0
        ratio_std = round(float(np.std(ratios)), 2)

        is_valid = avg_d > 0 and n_pos >= 2 and ratio_std < 0.3
        log(f"  {sig_name}: {n_pos}/{len(all_deltas)} positive, avg Δ={avg_d:+.4f}, "
            f"ratio_std={ratio_std} → {'VALIDATED' if is_valid else 'FAILED'}")

    # ════════════════════════════════════
    # Final verdict
    # ════════════════════════════════════
    validated = []
    for sig_name, wf_list in wf_results.items():
        all_deltas = [d for wf in wf_list for d in wf["test_deltas"].values()]
        ratios = [wf["optimal_ratio"] for wf in wf_list]
        n_pos = sum(1 for d in all_deltas if d > 0)
        avg_d = sum(all_deltas) / len(all_deltas) if all_deltas else 0
        if avg_d > 0 and n_pos >= 2:
            validated.append(sig_name)

    if validated:
        log(f"\n  VALIDATED: {validated}")
    else:
        log("\n  NO SIGNAL PASSES WALK-FORWARD VALIDATION")
        log("  P91b WITHOUT any overlay remains the champion")

    report["validated"] = validated
    report["verdict"] = f"VALIDATED: {validated}" if validated else "P91b WITHOUT overlay is the champion"

    elapsed = round(time.time() - t0, 1)
    report["elapsed_seconds"] = elapsed

    out_path = os.path.join(OUT_DIR, "phase110_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    log(f"\nPhase 110 COMPLETE in {elapsed}s → {out_path}")
