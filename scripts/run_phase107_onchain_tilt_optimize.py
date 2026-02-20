#!/usr/bin/env python3
"""
Phase 107: On-Chain Tilt Optimization + OOS Validation
========================================================
Phase 106 found tx_mom_14 as leverage tilt improves MIN by +0.107.
This phase:
  A) Grid search over tilt ratios (0.0x to 0.9x in negative regime)
  B) Test multiple on-chain metrics and lookbacks
  C) OOS 2026 validation
  D) Combined tilt (tx + hash + composite)
  E) Critical question: is the MIN improvement robust to tilt ratio?
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

OUT_DIR = os.path.join(PROJ, "artifacts", "phase107")
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
    print(f"[P107] {msg}", flush=True)


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
            values = data.get("values", [])
            return {datetime.utcfromtimestamp(v["x"]).strftime("%Y-%m-%d"): v["y"] for v in values}
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
    """Apply leverage tilt: 1.0x when signal > 0, tilt_ratio when signal <= 0."""
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


if __name__ == "__main__":
    t0 = time.time()
    report = {"phase": 107}

    # ════════════════════════════════════
    # Fetch on-chain data
    # ════════════════════════════════════
    log("Fetching on-chain data...")
    raw_tx = fetch_blockchain_chart("n-transactions", "6years")
    raw_hash = fetch_blockchain_chart("hash-rate", "6years")
    raw_addr = fetch_blockchain_chart("n-unique-addresses", "6years")
    log(f"  tx: {len(raw_tx)} days, hash: {len(raw_hash)} days, addr: {len(raw_addr)} days")

    # Compute signals with various lookbacks
    onchain_signals = {}
    for lb in [7, 14, 21, 30]:
        onchain_signals[f"tx_mom_{lb}"] = compute_momentum(raw_tx, lb)
        onchain_signals[f"hash_mom_{lb}"] = compute_momentum(raw_hash, lb)
        onchain_signals[f"addr_mom_{lb}"] = compute_momentum(raw_addr, lb)

    # Composite: average of tx + hash + addr momentum
    for lb in [7, 14, 21, 30]:
        tx_m = onchain_signals[f"tx_mom_{lb}"]
        hash_m = onchain_signals[f"hash_mom_{lb}"]
        addr_m = onchain_signals[f"addr_mom_{lb}"]
        all_dates = sorted(set(tx_m.keys()) & set(hash_m.keys()) & set(addr_m.keys()))
        composite = {d: (tx_m[d] + hash_m[d] + addr_m[d]) / 3.0 for d in all_dates}
        onchain_signals[f"composite_{lb}"] = composite

    log(f"  Generated {len(onchain_signals)} signal variants")

    # ════════════════════════════════════
    # Precompute P91b returns
    # ════════════════════════════════════
    log("\nPrecomputing P91b returns per year + OOS...")
    p91b_rets = {}
    for year in YEARS + ["2026_oos"]:
        if year == "2026_oos":
            start, end = OOS_RANGE
        else:
            start, end = YEAR_RANGES[year]
        sig_rets = {}
        for sig_key in SIG_KEYS:
            try:
                sig_rets[sig_key] = run_signal(SIGNALS[sig_key], start, end)
            except:
                sig_rets[sig_key] = []
        p91b_rets[year] = blend(sig_rets, P91B_WEIGHTS)
        log(f"  {year}: {len(p91b_rets[year])} bars")

    # Baseline
    baseline_sharpes = {y: round(compute_sharpe(p91b_rets[y]), 4) for y in YEARS}
    baseline_oos = round(compute_sharpe(p91b_rets["2026_oos"]), 4)
    baseline_avg = round(sum(baseline_sharpes.values()) / len(baseline_sharpes), 4)
    baseline_min = round(min(baseline_sharpes.values()), 4)
    log(f"\nBaseline: {baseline_sharpes} AVG={baseline_avg} MIN={baseline_min} OOS={baseline_oos}")

    # ════════════════════════════════════
    # SECTION A: Grid search over tilt ratios
    # ════════════════════════════════════
    log("\n" + "=" * 60)
    log("SECTION A: Tilt ratio grid search")
    log("=" * 60)

    TILT_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    TOP_SIGNALS = ["tx_mom_14", "tx_mom_7", "tx_mom_21", "hash_mom_14", "composite_14", "composite_30"]

    grid_results = {}
    for sig_name in TOP_SIGNALS:
        signal_daily = onchain_signals.get(sig_name, {})
        if not signal_daily:
            continue

        best_obj = -999
        best_ratio = 1.0
        ratio_results = {}

        for ratio in TILT_RATIOS:
            yearly_sharpes = {}
            for year in YEARS:
                tilted = apply_tilt(p91b_rets[year], signal_daily, YEAR_RANGES[year][0], ratio)
                yearly_sharpes[year] = round(compute_sharpe(tilted), 4)

            vals = list(yearly_sharpes.values())
            avg_s = round(sum(vals) / len(vals), 4)
            min_s = round(min(vals), 4)
            obj = round((avg_s + min_s) / 2, 4)

            # OOS
            oos_tilted = apply_tilt(p91b_rets["2026_oos"], signal_daily, OOS_RANGE[0], ratio)
            oos_s = round(compute_sharpe(oos_tilted), 4)

            ratio_results[str(ratio)] = {
                "yearly": yearly_sharpes, "avg": avg_s, "min": min_s,
                "obj": obj, "oos": oos_s,
            }
            if obj > best_obj:
                best_obj = obj
                best_ratio = ratio

        grid_results[sig_name] = {
            "ratios": ratio_results,
            "best_ratio": best_ratio,
            "best_obj": best_obj,
        }

        # Show key ratios
        log(f"\n  {sig_name}:")
        for ratio in [0.0, 0.3, 0.5, 0.7, 1.0]:
            r = ratio_results[str(ratio)]
            delta_avg = round(r["avg"] - baseline_avg, 3)
            delta_min = round(r["min"] - baseline_min, 3)
            log(f"    ratio={ratio}: AVG={r['avg']:.3f} (Δ{delta_avg:+.3f}), "
                f"MIN={r['min']:.3f} (Δ{delta_min:+.3f}), OOS={r['oos']:.3f}")
        log(f"    BEST ratio={best_ratio}, OBJ={best_obj:.4f}")

    report["grid_search"] = grid_results

    # ════════════════════════════════════
    # SECTION B: Best variants comparison
    # ════════════════════════════════════
    log("\n" + "=" * 60)
    log("SECTION B: Best variant comparison")
    log("=" * 60)

    comparison = []
    comparison.append({
        "name": "baseline_p91b",
        "tilt": "none",
        "avg": baseline_avg, "min": baseline_min,
        "obj": round((baseline_avg + baseline_min) / 2, 4),
        "oos": baseline_oos,
    })

    for sig_name, data in grid_results.items():
        best_r = data["best_ratio"]
        best_data = data["ratios"][str(best_r)]
        comparison.append({
            "name": f"{sig_name}_r{best_r}",
            "tilt": best_r,
            "avg": best_data["avg"], "min": best_data["min"],
            "obj": best_data["obj"], "oos": best_data["oos"],
        })

    comparison.sort(key=lambda x: x["obj"], reverse=True)
    report["comparison"] = comparison

    log(f"\n{'Name':>25} | {'AVG':>7} | {'MIN':>7} | {'OBJ':>7} | {'OOS':>7}")
    log(f"{'-'*25}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
    for c in comparison:
        log(f"{c['name']:>25} | {c['avg']:>7.3f} | {c['min']:>7.3f} | {c['obj']:>7.3f} | {c['oos']:>7.3f}")

    # ════════════════════════════════════
    # SECTION C: Per-year analysis of best tilt
    # ════════════════════════════════════
    log("\n" + "=" * 60)
    log("SECTION C: Year-by-year analysis of best tilt")
    log("=" * 60)

    # Find overall best
    best_overall = max(comparison, key=lambda x: x["obj"])
    log(f"  Overall best: {best_overall['name']} (OBJ={best_overall['obj']})")

    if best_overall["name"] != "baseline_p91b":
        sig_parts = best_overall["name"].rsplit("_r", 1)
        sig_name = sig_parts[0]
        ratio = float(sig_parts[1])
        signal_daily = onchain_signals[sig_name]

        log(f"\n  Year-by-year for {sig_name} at ratio={ratio}:")
        for year in YEARS:
            base_s = baseline_sharpes[year]
            tilted = apply_tilt(p91b_rets[year], signal_daily, YEAR_RANGES[year][0], ratio)
            tilt_s = round(compute_sharpe(tilted), 4)

            # Active days %
            n_hours = len(p91b_rets[year])
            active = sum(1 for h_idx in range(n_hours)
                        if signal_daily.get(
                            (datetime.strptime(YEAR_RANGES[year][0], "%Y-%m-%d") + timedelta(days=h_idx//24)).strftime("%Y-%m-%d"),
                            0.0
                        ) > 0)
            active_pct = round(active / n_hours * 100, 1) if n_hours > 0 else 0

            delta = round(tilt_s - base_s, 4)
            marker = "↑" if delta > 0 else "↓" if delta < 0 else "="
            log(f"    {year}: Base={base_s:.3f} → Tilt={tilt_s:.3f} (Δ{delta:+.3f} {marker}), Active={active_pct}%")

    # ════════════════════════════════════
    # SECTION D: Robustness check — does ANY tilt beat baseline OBJ?
    # ════════════════════════════════════
    log("\n" + "=" * 60)
    log("SECTION D: Robustness — tilt vs baseline balanced objective")
    log("=" * 60)

    baseline_obj = round((baseline_avg + baseline_min) / 2, 4)
    log(f"  Baseline OBJ: {baseline_obj}")

    any_beats = False
    for sig_name, data in grid_results.items():
        for ratio_str, r in data["ratios"].items():
            if r["obj"] > baseline_obj and ratio_str != "1.0":
                log(f"    {sig_name} ratio={ratio_str}: OBJ={r['obj']} > {baseline_obj} (Δ{r['obj']-baseline_obj:+.4f})")
                any_beats = True

    if not any_beats:
        log("  NO tilt variant beats baseline balanced objective")
        log("  The MIN improvement comes at too much AVG cost")

    report["robustness"] = {"baseline_obj": baseline_obj, "any_tilt_beats_obj": any_beats}

    # ════════════════════════════════════
    # VERDICT
    # ════════════════════════════════════
    log("\n" + "=" * 60)
    log("VERDICT")
    log("=" * 60)

    if any_beats:
        verdict = "ON-CHAIN TILT IMPROVES BALANCED OBJECTIVE"
        log(f"  {verdict}")
        log(f"  Best: {best_overall['name']}")
    else:
        # Check if MIN improves even if OBJ doesn't
        best_min_variant = max(comparison, key=lambda x: x["min"])
        if best_min_variant["min"] > baseline_min and best_min_variant["name"] != "baseline_p91b":
            verdict = "ON-CHAIN TILT IMPROVES MIN BUT NOT BALANCED OBJ — use only if MIN is priority"
            log(f"  {verdict}")
            log(f"  Best MIN: {best_min_variant['name']} MIN={best_min_variant['min']:.3f} (Δ{best_min_variant['min']-baseline_min:+.3f})")
            log(f"  But AVG cost: {best_min_variant['avg']:.3f} vs {baseline_avg:.3f}")
        else:
            verdict = "ON-CHAIN TILT DOES NOT IMPROVE P91b — daily resolution too coarse"
            log(f"  {verdict}")
            log("  P91b without tilt remains the champion")

    report["verdict"] = verdict

    # ════════════════════════════════════
    # SAVE
    # ════════════════════════════════════
    elapsed = round(time.time() - t0, 1)
    report["elapsed_seconds"] = elapsed

    out_path = os.path.join(OUT_DIR, "phase107_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    log(f"\nPhase 107 COMPLETE in {elapsed}s → {out_path}")
