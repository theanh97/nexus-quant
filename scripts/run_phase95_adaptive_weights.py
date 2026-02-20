#!/usr/bin/env python3
"""
Phase 95: Adaptive Ensemble Weighting
======================================
Instead of static P91b weights, test:
  A) Rolling walk-forward weight optimization (trailing 6mo → next 1mo)
  B) Inverse-volatility weighting (risk parity across 4 signals)
  C) Minimum-correlation weighting (maximize diversification)
  D) Signal-strength gating (reduce weight when signal is flat)

Compare ALL to static P91b across 2021-2025.
"""

import copy, json, os, sys, time
from pathlib import Path

PROJ = "/Users/qtmobile/Desktop/Nexus - Quant Trading "
sys.path.insert(0, PROJ)

import numpy as np

from nexus_quant.data.providers.registry import make_provider
from nexus_quant.strategies.registry import make_strategy
from nexus_quant.backtest.engine import BacktestConfig, BacktestEngine
from nexus_quant.backtest.costs import cost_model_from_config

OUT_DIR = os.path.join(PROJ, "artifacts", "phase95")
os.makedirs(OUT_DIR, exist_ok=True)

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
]

# Full 5-year period (we need continuous returns for adaptive methods)
FULL_RANGE = ("2021-01-01", "2026-01-01")
YEAR_BOUNDARIES = [0, 8760, 17520, 26280, 35040, 43800]  # Approx hourly boundaries
YEAR_LABELS = ["2021", "2022", "2023", "2024", "2025"]
OOS_RANGE = ("2026-01-01", "2026-02-20")

CHAMPION_SIGNALS = {
    "v1": {"name": "nexus_alpha_v1", "params": {
        "k_per_side": 2, "w_carry": 0.35, "w_mom": 0.45, "w_confirm": 0.0,
        "w_mean_reversion": 0.20, "w_vol_momentum": 0.0, "w_funding_trend": 0.0,
        "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 72,
        "vol_lookback_bars": 168, "target_portfolio_vol": 0.0, "use_min_variance": False,
        "target_gross_leverage": 0.35, "min_gross_leverage": 0.05,
        "max_gross_leverage": 0.65, "rebalance_interval_bars": 60, "strict_agreement": False,
    }},
    "i460bw168": {"name": "idio_momentum_alpha", "params": {
        "k_per_side": 4, "lookback_bars": 460, "beta_window_bars": 168,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 48,
    }},
    "i415bw216": {"name": "idio_momentum_alpha", "params": {
        "k_per_side": 4, "lookback_bars": 415, "beta_window_bars": 216,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 48,
    }},
    "f144": {"name": "funding_momentum_alpha", "params": {
        "k_per_side": 2, "funding_lookback_bars": 144, "direction": "contrarian",
        "vol_lookback_bars": 168, "target_gross_leverage": 0.25, "rebalance_interval_bars": 24,
    }},
}
P91B_WEIGHTS = {"v1": 0.2747, "i460bw168": 0.1967, "i415bw216": 0.3247, "f144": 0.2039}
SIG_KEYS = ["v1", "i460bw168", "i415bw216", "f144"]


def log(msg, end="\n"):
    print(f"[P95] {msg}", end=end, flush=True)


def run_signal_full(sig_cfg, start, end):
    """Run a signal over a date range, return hourly returns."""
    data_cfg = {
        "provider": "binance_rest_v1", "symbols": SYMBOLS,
        "start": start, "end": end, "bar_interval": "1h",
        "cache_dir": ".cache/binance_rest",
    }
    costs_cfg = {"fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0}
    exec_cfg = {"style": "taker", "slippage_bps": 3.0}
    provider = make_provider(data_cfg, seed=42)
    dataset = provider.load()
    strategy = make_strategy({"name": sig_cfg["name"], "params": copy.deepcopy(sig_cfg["params"])})
    cost_model = cost_model_from_config(costs_cfg, execution_cfg=exec_cfg)
    engine = BacktestEngine(BacktestConfig(costs=cost_model))
    result = engine.run(dataset=dataset, strategy=strategy, seed=42)
    return result.returns


def compute_sharpe(arr):
    if len(arr) < 50:
        return 0.0
    std = float(np.std(arr))
    return float(np.mean(arr) / std * np.sqrt(8760)) if std > 0 else 0.0


def year_sharpes(blended_arr, year_bounds, year_labels):
    """Compute Sharpe per year from a continuous return stream."""
    results = {}
    for i, label in enumerate(year_labels):
        start_idx = year_bounds[i]
        end_idx = year_bounds[i + 1] if i + 1 < len(year_bounds) else len(blended_arr)
        end_idx = min(end_idx, len(blended_arr))
        if start_idx >= len(blended_arr):
            results[label] = 0.0
            continue
        chunk = blended_arr[start_idx:end_idx]
        results[label] = round(compute_sharpe(chunk), 4)
    return results


# ═══════════════════════════════════════════
#  ADAPTIVE WEIGHTING METHODS
# ═══════════════════════════════════════════

def static_blend(sig_returns, weights):
    """Static P91b weights — baseline."""
    n = min(len(sig_returns[k]) for k in SIG_KEYS)
    R = np.zeros((len(SIG_KEYS), n))
    W = np.array([weights[k] for k in SIG_KEYS])
    for i, k in enumerate(SIG_KEYS):
        R[i, :n] = sig_returns[k][:n]
    return W @ R


def inverse_vol_blend(sig_returns, halflife=720):
    """Risk-parity: weight inversely to trailing realized vol."""
    n = min(len(sig_returns[k]) for k in SIG_KEYS)
    R = np.zeros((len(SIG_KEYS), n))
    for i, k in enumerate(SIG_KEYS):
        R[i, :n] = sig_returns[k][:n]

    blended = np.zeros(n)
    # Warm-up period: use equal weights for first halflife bars
    ew = np.ones(len(SIG_KEYS)) / len(SIG_KEYS)
    for t in range(n):
        if t < halflife:
            w = ew
        else:
            # Trailing vol over halflife window
            window = R[:, max(0, t - halflife):t]
            vols = np.std(window, axis=1)
            vols = np.maximum(vols, 1e-10)
            inv_vol = 1.0 / vols
            w = inv_vol / inv_vol.sum()
        blended[t] = w @ R[:, t]
    return blended


def rolling_sharpe_blend(sig_returns, lookback=4320, rebal_period=720):
    """Walk-forward: optimize weights on trailing lookback, apply for rebal_period."""
    n = min(len(sig_returns[k]) for k in SIG_KEYS)
    R = np.zeros((len(SIG_KEYS), n))
    for i, k in enumerate(SIG_KEYS):
        R[i, :n] = sig_returns[k][:n]

    blended = np.zeros(n)
    current_w = np.ones(len(SIG_KEYS)) / len(SIG_KEYS)

    for t in range(n):
        blended[t] = current_w @ R[:, t]

        # Rebalance every rebal_period bars
        if t > 0 and t % rebal_period == 0 and t >= lookback:
            window = R[:, t - lookback:t]
            # Find weights that maximize trailing Sharpe
            best_w = current_w.copy()
            best_sharpe = -999
            # Simple grid search over weights
            for w0 in np.arange(0.10, 0.45, 0.05):
                for w1 in np.arange(0.05, 0.35, 0.05):
                    for w2 in np.arange(0.10, 0.45, 0.05):
                        w3 = 1.0 - w0 - w1 - w2
                        if w3 < 0.05 or w3 > 0.35:
                            continue
                        w_arr = np.array([w0, w1, w2, w3])
                        trail = w_arr @ window
                        std = np.std(trail)
                        if std > 0:
                            s = np.mean(trail) / std * np.sqrt(8760)
                        else:
                            s = 0.0
                        if s > best_sharpe:
                            best_sharpe = s
                            best_w = w_arr.copy()
            current_w = best_w

    return blended


def min_corr_blend(sig_returns, halflife=2160):
    """Minimize average pairwise correlation — maximize diversification."""
    n = min(len(sig_returns[k]) for k in SIG_KEYS)
    R = np.zeros((len(SIG_KEYS), n))
    for i, k in enumerate(SIG_KEYS):
        R[i, :n] = sig_returns[k][:n]

    blended = np.zeros(n)
    current_w = np.ones(len(SIG_KEYS)) / len(SIG_KEYS)

    for t in range(n):
        blended[t] = current_w @ R[:, t]

        # Rebalance every 720 bars
        if t > 0 and t % 720 == 0 and t >= halflife:
            window = R[:, t - halflife:t]
            C = np.corrcoef(window)
            if np.any(np.isnan(C)):
                continue
            # Weight inversely to average correlation with others
            avg_corr = np.zeros(len(SIG_KEYS))
            for i in range(len(SIG_KEYS)):
                others = [C[i, j] for j in range(len(SIG_KEYS)) if j != i]
                avg_corr[i] = np.mean(others)
            # Lower correlation = higher weight
            inv_corr = 1.0 / (avg_corr + 1.01)  # Shift to avoid div by 0
            w = inv_corr / inv_corr.sum()
            # Clip to reasonable range
            w = np.clip(w, 0.05, 0.45)
            w = w / w.sum()
            current_w = w

    return blended


def signal_gate_blend(sig_returns, gate_window=720, gate_threshold=0.0):
    """Gate: reduce weight of signals with negative trailing Sharpe."""
    n = min(len(sig_returns[k]) for k in SIG_KEYS)
    R = np.zeros((len(SIG_KEYS), n))
    for i, k in enumerate(SIG_KEYS):
        R[i, :n] = sig_returns[k][:n]

    base_w = np.array([P91B_WEIGHTS[k] for k in SIG_KEYS])
    blended = np.zeros(n)

    for t in range(n):
        if t < gate_window:
            w = base_w
        else:
            window = R[:, t - gate_window:t]
            sharpes = np.zeros(len(SIG_KEYS))
            for i in range(len(SIG_KEYS)):
                std = np.std(window[i])
                if std > 0:
                    sharpes[i] = np.mean(window[i]) / std * np.sqrt(8760)
            # Scale weights: if trailing Sharpe < threshold, halve the weight
            scale = np.where(sharpes > gate_threshold, 1.0, 0.5)
            w = base_w * scale
            w = w / w.sum()
        blended[t] = w @ R[:, t]

    return blended


# ═══════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════

if __name__ == "__main__":
    t0 = time.time()
    report = {"phase": 95, "methods": {}, "oos_results": {}}

    # ══════════════════════════════════════
    # SECTION A: Load all signal returns (per year, to match Phase 93/94 structure)
    # ══════════════════════════════════════
    log("=" * 60)
    log("SECTION A: Loading signal returns per year")
    log("=" * 60)

    sig_returns_by_year = {yr: {} for yr in YEAR_LABELS}
    year_ranges = {
        "2021": ("2021-01-01", "2022-01-01"),
        "2022": ("2022-01-01", "2023-01-01"),
        "2023": ("2023-01-01", "2024-01-01"),
        "2024": ("2024-01-01", "2025-01-01"),
        "2025": ("2025-01-01", "2026-01-01"),
    }

    all_sig_rets = {}  # {sig: [concatenated returns across all years]}
    for sig_name, sig_cfg in CHAMPION_SIGNALS.items():
        log(f"  Loading {sig_name}...")
        combined = []
        for yr in YEAR_LABELS:
            start, end = year_ranges[yr]
            rets = run_signal_full(sig_cfg, start, end)
            sig_returns_by_year[yr][sig_name] = np.array(rets)
            combined.extend(rets)
        all_sig_rets[sig_name] = combined
        log(f"    Total bars: {len(combined)}")

    # ══════════════════════════════════════
    # SECTION B: Compute year boundaries from actual data
    # ══════════════════════════════════════
    n_total = min(len(all_sig_rets[k]) for k in SIG_KEYS)
    log(f"\n  Total aligned bars: {n_total}")

    # Use actual per-year lengths for year boundaries
    actual_bounds = [0]
    for yr in YEAR_LABELS:
        yr_len = len(sig_returns_by_year[yr][SIG_KEYS[0]])
        actual_bounds.append(actual_bounds[-1] + yr_len)
    log(f"  Year boundaries: {actual_bounds}")

    # Truncate all to n_total
    for k in SIG_KEYS:
        all_sig_rets[k] = all_sig_rets[k][:n_total]

    # ══════════════════════════════════════
    # SECTION C: Run all methods
    # ══════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("SECTION C: Testing adaptive weighting methods")
    log("=" * 60)

    methods = {
        "static_p91b": lambda sr: static_blend(sr, P91B_WEIGHTS),
        "static_equal": lambda sr: static_blend(sr, {k: 0.25 for k in SIG_KEYS}),
        "inv_vol_720": lambda sr: inverse_vol_blend(sr, halflife=720),
        "inv_vol_2160": lambda sr: inverse_vol_blend(sr, halflife=2160),
        "roll_sharpe_6m": lambda sr: rolling_sharpe_blend(sr, lookback=4320, rebal_period=720),
        "roll_sharpe_3m": lambda sr: rolling_sharpe_blend(sr, lookback=2160, rebal_period=720),
        "min_corr": lambda sr: min_corr_blend(sr, halflife=2160),
        "gate_0": lambda sr: signal_gate_blend(sr, gate_window=720, gate_threshold=0.0),
        "gate_0.3": lambda sr: signal_gate_blend(sr, gate_window=720, gate_threshold=0.3),
    }

    for method_name, method_fn in methods.items():
        log(f"\n  Method: {method_name}")
        blended = method_fn(all_sig_rets)
        yby = year_sharpes(blended, actual_bounds, YEAR_LABELS)
        vals = [yby[yr] for yr in YEAR_LABELS]
        avg_s = sum(vals) / len(vals)
        min_s = min(vals)

        report["methods"][method_name] = {
            "yby": yby,
            "avg": round(avg_s, 4),
            "min_s": round(min_s, 4),
        }
        log(f"    YbY: {vals}")
        log(f"    AVG={avg_s:.4f} MIN={min_s:.4f}")

    # ══════════════════════════════════════
    # SECTION D: 2026 OOS
    # ══════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("SECTION D: 2026 OOS validation")
    log("=" * 60)

    oos_rets = {}
    for sig_name, sig_cfg in CHAMPION_SIGNALS.items():
        try:
            oos_rets[sig_name] = run_signal_full(sig_cfg, "2026-01-01", "2026-02-20")
        except Exception as e:
            log(f"  {sig_name} OOS error: {e}")
            oos_rets[sig_name] = []

    # OOS for static P91b
    oos_n = min(len(oos_rets[k]) for k in SIG_KEYS)
    R_oos = np.zeros((len(SIG_KEYS), oos_n))
    W_p91b = np.array([P91B_WEIGHTS[k] for k in SIG_KEYS])
    for i, k in enumerate(SIG_KEYS):
        R_oos[i, :oos_n] = oos_rets[k][:oos_n]
    oos_static = W_p91b @ R_oos
    oos_static_sharpe = compute_sharpe(oos_static)
    log(f"  P91b static OOS: {oos_static_sharpe:.4f}")
    report["oos_results"]["static_p91b"] = round(oos_static_sharpe, 4)

    # For adaptive methods that need history, we use the full IS+OOS concatenated
    # But the key insight: adaptive methods use trailing windows, so OOS weights
    # come from the end of IS data. We just apply the last computed weights.
    # For a fair test, use equal weights for OOS (since adaptive can't look ahead)
    oos_equal = np.ones(len(SIG_KEYS)) / len(SIG_KEYS) @ R_oos
    log(f"  Equal weight OOS: {compute_sharpe(oos_equal):.4f}")
    report["oos_results"]["equal_weight"] = round(compute_sharpe(oos_equal), 4)

    # ══════════════════════════════════════
    # SAVE
    # ══════════════════════════════════════
    elapsed = time.time() - t0
    report["elapsed_seconds"] = round(elapsed, 1)

    report_path = os.path.join(OUT_DIR, "phase95_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    log(f"\n{'=' * 60}")
    log(f"Phase 95 COMPLETE in {elapsed:.0f}s")
    log(f"Report: {report_path}")
    log("=" * 60)

    # Summary table
    log("\n  METHOD COMPARISON:")
    log(f"  {'Method':>20} | {'2021':>7} | {'2022':>7} | {'2023':>7} | {'2024':>7} | {'2025':>7} | {'AVG':>7} | {'MIN':>7}")
    log(f"  {'-'*20}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
    for method_name in methods:
        m = report["methods"][method_name]
        vals = [m["yby"][yr] for yr in YEAR_LABELS]
        marker = " ★" if m["min_s"] > 1.576 else ""
        log(f"  {method_name:>20} | {vals[0]:>7.4f} | {vals[1]:>7.4f} | {vals[2]:>7.4f} | {vals[3]:>7.4f} | {vals[4]:>7.4f} | {m['avg']:>7.4f} | {m['min_s']:>7.4f}{marker}")
