#!/usr/bin/python3
"""
Phase 97: Universe Expansion Test for P91b Champion
===================================================
Test whether expanding from 10 to 20 coins improves risk-adjusted returns
for the fixed P91b 4-signal blend.
"""

import copy
import json
import os
import sys
import time

PROJ = "/Users/qtmobile/Desktop/Nexus - Quant Trading "
sys.path.insert(0, PROJ)

import numpy as np

from nexus_quant.data.providers.registry import make_provider
from nexus_quant.strategies.registry import make_strategy
from nexus_quant.backtest.engine import BacktestConfig, BacktestEngine
from nexus_quant.backtest.costs import cost_model_from_config

OUT_DIR = os.path.join(PROJ, "artifacts", "phase97")
os.makedirs(OUT_DIR, exist_ok=True)

ORIGINAL_10 = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
]

EXPANDED_20 = ORIGINAL_10 + [
    "MATICUSDT", "ATOMUSDT", "NEARUSDT", "APTUSDT", "ARBUSDT",
    "OPUSDT", "FILUSDT", "LTCUSDT", "UNIUSDT", "AAVEUSDT",
]

YEARS = ["2021", "2022", "2023", "2024", "2025"]
YEAR_RANGES = {
    "2021": ("2021-01-01", "2022-01-01"),
    "2022": ("2022-01-01", "2023-01-01"),
    "2023": ("2023-01-01", "2024-01-01"),
    "2024": ("2024-01-01", "2025-01-01"),
    "2025": ("2025-01-01", "2026-01-01"),
}
OOS_2026 = ("2026-01-01", "2026-02-20")

P91B_SIGNAL_CONFIGS = {
    "v1": {
        "name": "nexus_alpha_v1",
        "params": {
            "k_per_side": 2,
            "w_carry": 0.35,
            "w_mom": 0.45,
            "w_mean_reversion": 0.20,
            "momentum_lookback_bars": 336,
            "mean_reversion_lookback_bars": 72,
            "vol_lookback_bars": 168,
            "target_gross_leverage": 0.35,
            "rebalance_interval_bars": 60,
        },
    },
    "i460bw168": {
        "name": "idio_momentum_alpha",
        "params": {
            "k_per_side": 4,
            "lookback_bars": 460,
            "beta_window_bars": 168,
            "target_gross_leverage": 0.30,
            "rebalance_interval_bars": 48,
        },
    },
    "i415bw216": {
        "name": "idio_momentum_alpha",
        "params": {
            "k_per_side": 4,
            "lookback_bars": 415,
            "beta_window_bars": 216,
            "target_gross_leverage": 0.30,
            "rebalance_interval_bars": 48,
        },
    },
    "f144": {
        "name": "funding_momentum_alpha",
        "params": {
            "k_per_side": 2,
            "funding_lookback_bars": 144,
            "direction": "contrarian",
            "target_gross_leverage": 0.25,
            "rebalance_interval_bars": 24,
        },
    },
}

P91B_WEIGHTS = {
    "v1": 0.2747,
    "i460bw168": 0.1967,
    "i415bw216": 0.3247,
    "f144": 0.2039,
}


def log(msg, end="\n"):
    print(f"[P97] {msg}", end=end, flush=True)


def compute_sharpe(returns):
    if returns is None:
        return 0.0
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size < 50:
        return 0.0
    std = float(np.std(arr))
    if std <= 0:
        return 0.0
    return float(np.mean(arr) / std * np.sqrt(8760))


def blend_returns(signal_returns, weights):
    keys = sorted(weights.keys())
    n = min(len(signal_returns.get(k, [])) for k in keys)
    if n == 0:
        return []
    R = np.zeros((len(keys), n), dtype=np.float64)
    W = np.array([weights[k] for k in keys], dtype=np.float64)
    for i, k in enumerate(keys):
        R[i, :] = signal_returns[k][:n]
    return (W @ R).tolist()


def run_signal(signal_cfg, symbols, start, end):
    data_cfg = {
        "provider": "binance_rest_v1",
        "symbols": symbols,
        "start": start,
        "end": end,
        "bar_interval": "1h",
        "cache_dir": ".cache/binance_rest",
    }
    costs_cfg = {"fee_rate": 0.0005, "slippage_rate": 0.0003}
    exec_cfg = {"style": "taker", "slippage_bps": 3.0}

    provider = make_provider(data_cfg, seed=42)
    dataset = provider.load()
    strategy = make_strategy({"name": signal_cfg["name"], "params": copy.deepcopy(signal_cfg["params"])})
    cost_model = cost_model_from_config(costs_cfg, execution_cfg=exec_cfg)
    engine = BacktestEngine(BacktestConfig(costs=cost_model))
    result = engine.run(dataset=dataset, strategy=strategy, seed=42)
    return result.returns


def make_scaled_20_configs():
    scaled = copy.deepcopy(P91B_SIGNAL_CONFIGS)
    scaled["v1"]["params"]["k_per_side"] = 4
    scaled["i460bw168"]["params"]["k_per_side"] = 6
    scaled["i415bw216"]["params"]["k_per_side"] = 6
    scaled["f144"]["params"]["k_per_side"] = 3
    return scaled


def evaluate_profile(profile_name, symbols, signal_cfgs):
    out = {
        "profile": profile_name,
        "n_symbols": len(symbols),
        "symbols": symbols,
        "yearly_sharpe": {},
        "signal_yearly_sharpe": {},
        "bars_per_year": {},
    }

    for year in YEARS:
        start, end = YEAR_RANGES[year]
        log(f"  {profile_name} / {year}")

        sig_rets = {}
        sig_sharpes = {}
        sig_keys = ["v1", "i460bw168", "i415bw216", "f144"]

        for sig_key in sig_keys:
            log(f"    running {sig_key}...")
            try:
                rets = run_signal(signal_cfgs[sig_key], symbols, start, end)
            except Exception as exc:
                log(f"    {sig_key} ERROR: {exc}")
                rets = []
            sig_rets[sig_key] = rets
            sig_sharpes[sig_key] = round(compute_sharpe(rets), 4)

        blended = blend_returns(sig_rets, P91B_WEIGHTS)
        blend_sharpe = round(compute_sharpe(blended), 4)

        out["yearly_sharpe"][year] = blend_sharpe
        out["signal_yearly_sharpe"][year] = sig_sharpes
        out["bars_per_year"][year] = min(len(sig_rets[k]) for k in sig_keys) if sig_rets else 0
        log(f"    blended Sharpe={blend_sharpe:.4f}")

    yearly_vals = [out["yearly_sharpe"][y] for y in YEARS]
    out["avg_2021_2025"] = round(sum(yearly_vals) / len(yearly_vals), 4)
    out["min_2021_2025"] = round(min(yearly_vals), 4)

    log(f"  {profile_name} / 2026 OOS")
    sig_oos = {}
    sig_oos_sharpes = {}
    for sig_key in ["v1", "i460bw168", "i415bw216", "f144"]:
        log(f"    running {sig_key} OOS...")
        try:
            rets = run_signal(signal_cfgs[sig_key], symbols, OOS_2026[0], OOS_2026[1])
        except Exception as exc:
            log(f"    {sig_key} OOS ERROR: {exc}")
            rets = []
        sig_oos[sig_key] = rets
        sig_oos_sharpes[sig_key] = round(compute_sharpe(rets), 4)

    oos_blended = blend_returns(sig_oos, P91B_WEIGHTS)
    out["oos_2026"] = round(compute_sharpe(oos_blended), 4)
    out["signal_oos_sharpe"] = sig_oos_sharpes
    out["oos_bars"] = min(len(sig_oos[k]) for k in sig_oos) if sig_oos else 0
    return out


def build_delta(base_run, test_run):
    return {
        "avg_2021_2025_delta": round(test_run["avg_2021_2025"] - base_run["avg_2021_2025"], 4),
        "min_2021_2025_delta": round(test_run["min_2021_2025"] - base_run["min_2021_2025"], 4),
        "oos_2026_delta": round(test_run["oos_2026"] - base_run["oos_2026"], 4),
        "yearly_delta": {
            y: round(test_run["yearly_sharpe"][y] - base_run["yearly_sharpe"][y], 4)
            for y in YEARS
        },
    }


def main():
    t0 = time.time()

    scaled_20 = make_scaled_20_configs()

    report = {
        "phase": 97,
        "universes": {
            "ORIGINAL_10": ORIGINAL_10,
            "EXPANDED_20": EXPANDED_20,
        },
        "weights": P91B_WEIGHTS,
        "signal_configs": {
            "base": P91B_SIGNAL_CONFIGS,
            "expanded_20_scaled_k": scaled_20,
        },
        "runs": {},
        "comparisons": {},
    }

    log("=" * 60)
    log("Phase 97: Universe expansion test for P91b champion")
    log("=" * 60)

    run_original = evaluate_profile("original_10_base_k", ORIGINAL_10, P91B_SIGNAL_CONFIGS)
    report["runs"]["original_10_base_k"] = run_original

    run_expanded_base = evaluate_profile("expanded_20_base_k", EXPANDED_20, P91B_SIGNAL_CONFIGS)
    report["runs"]["expanded_20_base_k"] = run_expanded_base

    run_expanded_scaled = evaluate_profile("expanded_20_scaled_k", EXPANDED_20, scaled_20)
    report["runs"]["expanded_20_scaled_k"] = run_expanded_scaled

    report["comparisons"]["expanded_20_base_vs_original_10_base"] = build_delta(run_original, run_expanded_base)
    report["comparisons"]["expanded_20_scaled_vs_original_10_base"] = build_delta(run_original, run_expanded_scaled)
    report["comparisons"]["expanded_20_scaled_vs_expanded_20_base"] = build_delta(run_expanded_base, run_expanded_scaled)

    elapsed = time.time() - t0
    report["elapsed_seconds"] = round(elapsed, 1)

    out_path = os.path.join(OUT_DIR, "phase97_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    log("=" * 60)
    log(f"Phase 97 complete in {elapsed:.1f}s")
    log(f"Report: {out_path}")
    log("=" * 60)

    log("Summary:")
    log(f"  {'Profile':>22} | {'AVG 21-25':>10} | {'MIN 21-25':>10} | {'OOS 2026':>8}")
    log(f"  {'-'*22}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
    for profile in ["original_10_base_k", "expanded_20_base_k", "expanded_20_scaled_k"]:
        r = report["runs"][profile]
        log(
            f"  {profile:>22} | {r['avg_2021_2025']:>10.4f} | "
            f"{r['min_2021_2025']:>10.4f} | {r['oos_2026']:>8.4f}"
        )


if __name__ == "__main__":
    main()
