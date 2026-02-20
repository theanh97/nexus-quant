#!/usr/bin/env python3
"""
Phase 108: Walk-Forward Validation of On-Chain Tilt
=====================================================
Phase 107 found composite_14 at ratio=0.4 dramatically improves P91b.
This is TOO GOOD to deploy without walk-forward validation.

Walk-forward tests:
  A) Train 2021-2023 → optimal ratio → test 2024
  B) Train 2022-2024 → optimal ratio → test 2025
  C) Train 2021-2024 → optimal ratio → test 2025
  D) Train 2021-2025 → optimal ratio → test 2026 OOS

For each window, find optimal tilt ratio using balanced (AVG+MIN)/2
on train years only, then measure OOS performance.

If OOS consistently > baseline → validated.
If OOS degrades → IS overfit, keep P91b without tilt.

Also test: is the tilt ratio stable across windows?
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

OUT_DIR = os.path.join(PROJ, "artifacts", "phase108")
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
ONCHAIN_SIGNALS_TO_TEST = ["composite_14", "composite_30", "tx_mom_14", "tx_mom_21"]

WF_WINDOWS = [
    {"train": ["2021", "2022", "2023"], "test": ["2024"]},
    {"train": ["2022", "2023", "2024"], "test": ["2025"]},
    {"train": ["2021", "2022", "2023", "2024"], "test": ["2025"]},
    {"train": ["2021", "2022", "2023", "2024", "2025"], "test": ["2026_oos"]},
]


def log(msg):
    print(f"[P108] {msg}", flush=True)


def compute_sharpe(returns, bars_per_year=8760):
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size < 50:
        return 0.0
    std = float(np.std(arr))
    if std <= 0:
        return 0.0
    return float(np.mean(arr) / std * np.sqrt(bars_per_year))


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
        log(f"  ERROR fetching {chart_name}: {e}")
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
        sig_val = signal_daily.get(date_str, 0.0)
        if sig_val > 0:
            tilted.append(ret)
        else:
            tilted.append(ret * tilt_ratio)
    return tilted


def find_best_ratio(p91b_rets, signal_daily, train_years, ratios):
    """Find best tilt ratio on training years using balanced (AVG+MIN)/2."""
    best_obj = -999
    best_ratio = 1.0
    for ratio in ratios:
        yearly_sharpes = []
        for year in train_years:
            start = YEAR_RANGES[year][0]
            tilted = apply_tilt(p91b_rets[year], signal_daily, start, ratio)
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
    report = {"phase": 108}

    # ════════════════════════════════════
    # Fetch on-chain data
    # ════════════════════════════════════
    log("Fetching on-chain data...")
    raw_tx = fetch_blockchain_chart("n-transactions", "6years")
    raw_hash = fetch_blockchain_chart("hash-rate", "6years")
    raw_addr = fetch_blockchain_chart("n-unique-addresses", "6years")
    log(f"  tx: {len(raw_tx)}, hash: {len(raw_hash)}, addr: {len(raw_addr)} days")

    # Build signals
    onchain_signals = {}
    for lb in [14, 21, 30]:
        tx_m = compute_momentum(raw_tx, lb)
        hash_m = compute_momentum(raw_hash, lb)
        addr_m = compute_momentum(raw_addr, lb)
        onchain_signals[f"tx_mom_{lb}"] = tx_m

        all_dates = sorted(set(tx_m.keys()) & set(hash_m.keys()) & set(addr_m.keys()))
        composite = {d: (tx_m[d] + hash_m[d] + addr_m[d]) / 3.0 for d in all_dates}
        onchain_signals[f"composite_{lb}"] = composite

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
        log(f"  {year}: {len(p91b_rets[year])} bars")

    # OOS
    sig_rets_oos = {}
    for sig_key in SIG_KEYS:
        try:
            sig_rets_oos[sig_key] = run_signal(SIGNALS[sig_key], OOS_RANGE[0], OOS_RANGE[1])
        except:
            sig_rets_oos[sig_key] = []
    p91b_rets["2026_oos"] = blend(sig_rets_oos, P91B_WEIGHTS)
    log(f"  2026_oos: {len(p91b_rets['2026_oos'])} bars")

    # Baseline
    baseline = {y: round(compute_sharpe(p91b_rets[y]), 4) for y in YEARS}
    baseline["2026_oos"] = round(compute_sharpe(p91b_rets["2026_oos"]), 4)
    baseline_avg = round(sum(baseline[y] for y in YEARS) / len(YEARS), 4)
    baseline_min = round(min(baseline[y] for y in YEARS), 4)
    log(f"\nBaseline: {baseline} AVG={baseline_avg} MIN={baseline_min}")

    # ════════════════════════════════════
    # Walk-forward validation
    # ════════════════════════════════════
    log("\n" + "=" * 60)
    log("WALK-FORWARD VALIDATION")
    log("=" * 60)

    wf_results = {}
    for sig_name in ONCHAIN_SIGNALS_TO_TEST:
        signal_daily = onchain_signals.get(sig_name, {})
        if not signal_daily:
            continue

        log(f"\n  Signal: {sig_name}")
        sig_wf = []

        for wf in WF_WINDOWS:
            train_label = "+".join(wf["train"])
            test_label = "+".join(wf["test"])

            # Find best ratio on train
            best_ratio, best_obj = find_best_ratio(p91b_rets, signal_daily, wf["train"], TILT_RATIOS)

            # Compute train Sharpes
            train_sharpes = {}
            for year in wf["train"]:
                tilted = apply_tilt(p91b_rets[year], signal_daily, YEAR_RANGES[year][0], best_ratio)
                train_sharpes[year] = round(compute_sharpe(tilted), 4)

            # Compute test Sharpes
            test_sharpes = {}
            test_baseline = {}
            for year in wf["test"]:
                if year == "2026_oos":
                    start = OOS_RANGE[0]
                    test_baseline[year] = baseline["2026_oos"]
                else:
                    start = YEAR_RANGES[year][0]
                    test_baseline[year] = baseline[year]
                tilted = apply_tilt(p91b_rets[year], signal_daily, start, best_ratio)
                test_sharpes[year] = round(compute_sharpe(tilted), 4)

            result = {
                "train_years": wf["train"],
                "test_years": wf["test"],
                "optimal_ratio": best_ratio,
                "train_obj": round(best_obj, 4),
                "train_sharpes": train_sharpes,
                "test_sharpes": test_sharpes,
                "test_baseline": test_baseline,
                "test_delta": {y: round(test_sharpes[y] - test_baseline[y], 4) for y in wf["test"]},
            }
            sig_wf.append(result)

            test_delta_str = ", ".join(f"{y}: Δ{result['test_delta'][y]:+.3f}" for y in wf["test"])
            log(f"    Train={train_label} → ratio={best_ratio}, "
                f"Test={test_label}: {test_delta_str}")

        wf_results[sig_name] = sig_wf

    report["walk_forward"] = wf_results

    # ════════════════════════════════════
    # Ratio stability across windows
    # ════════════════════════════════════
    log("\n" + "=" * 60)
    log("RATIO STABILITY")
    log("=" * 60)

    ratio_stability = {}
    for sig_name, wf_list in wf_results.items():
        ratios = [wf["optimal_ratio"] for wf in wf_list]
        ratio_stability[sig_name] = {
            "ratios": ratios,
            "mean": round(sum(ratios) / len(ratios), 2),
            "std": round(float(np.std(ratios)), 2),
            "min": min(ratios),
            "max": max(ratios),
        }
        log(f"  {sig_name}: ratios={ratios}, mean={ratio_stability[sig_name]['mean']}, "
            f"std={ratio_stability[sig_name]['std']}")

    report["ratio_stability"] = ratio_stability

    # ════════════════════════════════════
    # OOS summary
    # ════════════════════════════════════
    log("\n" + "=" * 60)
    log("OOS PERFORMANCE SUMMARY")
    log("=" * 60)

    oos_summary = {}
    for sig_name, wf_list in wf_results.items():
        # Collect all test deltas
        all_test_deltas = []
        for wf in wf_list:
            for y, delta in wf["test_delta"].items():
                all_test_deltas.append(delta)

        n_positive = sum(1 for d in all_test_deltas if d > 0)
        n_total = len(all_test_deltas)
        avg_delta = round(sum(all_test_deltas) / n_total, 4) if n_total > 0 else 0

        oos_summary[sig_name] = {
            "n_positive_oos": n_positive,
            "n_total_oos": n_total,
            "avg_oos_delta": avg_delta,
            "all_deltas": all_test_deltas,
        }
        log(f"  {sig_name}: {n_positive}/{n_total} OOS positive, avg Δ={avg_delta:+.4f}")

    report["oos_summary"] = oos_summary

    # ════════════════════════════════════
    # VERDICT
    # ════════════════════════════════════
    log("\n" + "=" * 60)
    log("FINAL VERDICT")
    log("=" * 60)

    # Criteria for validation:
    # 1. Average OOS delta > 0
    # 2. At least 50% of OOS tests positive
    # 3. Ratio is reasonably stable (std < 0.3)
    validated_signals = []
    for sig_name in ONCHAIN_SIGNALS_TO_TEST:
        if sig_name not in oos_summary:
            continue
        s = oos_summary[sig_name]
        r = ratio_stability.get(sig_name, {})

        avg_delta = s["avg_oos_delta"]
        pct_positive = s["n_positive_oos"] / s["n_total_oos"] * 100 if s["n_total_oos"] > 0 else 0
        ratio_std = r.get("std", 1.0)

        is_valid = avg_delta > 0 and pct_positive >= 50 and ratio_std < 0.3

        log(f"  {sig_name}: avg_Δ={avg_delta:+.4f}, "
            f"{pct_positive:.0f}% positive, ratio_std={ratio_std:.2f} → "
            f"{'VALIDATED' if is_valid else 'FAILED'}")

        if is_valid:
            validated_signals.append(sig_name)

    if validated_signals:
        verdict = f"ON-CHAIN TILT VALIDATED: {', '.join(validated_signals)}"
        log(f"\n  {verdict}")
        log("  → NEW CHAMPION: P91b + on-chain tilt")

        # Recommend best validated signal
        best_validated = max(validated_signals, key=lambda s: oos_summary[s]["avg_oos_delta"])
        best_ratio_mean = ratio_stability[best_validated]["mean"]
        log(f"  → Best: {best_validated} at mean ratio={best_ratio_mean}")
    else:
        verdict = "ON-CHAIN TILT NOT VALIDATED — OOS does not consistently improve"
        log(f"\n  {verdict}")
        log("  → P91b without tilt remains champion")

    report["verdict"] = verdict
    report["validated_signals"] = validated_signals

    # ════════════════════════════════════
    # SAVE
    # ════════════════════════════════════
    elapsed = round(time.time() - t0, 1)
    report["elapsed_seconds"] = elapsed

    out_path = os.path.join(OUT_DIR, "phase108_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    log(f"\nPhase 108 COMPLETE in {elapsed}s → {out_path}")
