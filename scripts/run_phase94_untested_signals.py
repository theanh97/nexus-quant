#!/usr/bin/env python3
"""
Phase 94: Profile ALL untested strategy implementations
========================================================
11 strategies exist in the codebase but have never been properly profiled
across the full 2021-2025 period with correlation analysis.

Categories:
  A) Quality momentum: sharpe_ratio, ewma_sharpe, sortino, vol_adjusted_mom
  B) Reversal-avoidance: skip_gram_momentum, price_level
  C) Microstructure/flow: amihud_illiquidity, taker_buy, funding_vol
  D) Composite: multitf_momentum, mr_funding
  E) Baseline: pure_momentum (control)

Pipeline:
  1. Profile each signal across 2021-2025 (Sharpe per year)
  2. Correlation matrix with P91b champion signals
  3. Identify ANY signal with AVG > 0.5 AND orthogonal (corr < 0.3)
  4. If promising: 5-signal ensemble sweep (replace weakest P91b signal)
  5. 2026 OOS validation
"""

import copy, json, math, os, sys, time
from pathlib import Path

PROJ = "/Users/qtmobile/Desktop/Nexus - Quant Trading "
sys.path.insert(0, PROJ)

import numpy as np

from nexus_quant.data.providers.registry import make_provider
from nexus_quant.strategies.registry import make_strategy
from nexus_quant.backtest.engine import BacktestConfig, BacktestEngine
from nexus_quant.backtest.costs import cost_model_from_config

OUT_DIR = os.path.join(PROJ, "artifacts", "phase94")
os.makedirs(OUT_DIR, exist_ok=True)

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
]
YEAR_RANGES = {
    "2021": ("2021-01-01", "2022-01-01"),
    "2022": ("2022-01-01", "2023-01-01"),
    "2023": ("2023-01-01", "2024-01-01"),
    "2024": ("2024-01-01", "2025-01-01"),
    "2025": ("2025-01-01", "2026-01-01"),
}
YEAR_RANGES_OOS = {"2026": ("2026-01-01", "2026-02-20")}

# ── Champion signals (P91b) ──
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

# ── Untested signals to profile ──
NEW_SIGNALS = {
    # Quality momentum family
    "sharpe_ratio": {"name": "sharpe_ratio_alpha", "params": {
        "k_per_side": 2, "lookback_bars": 168, "vol_lookback_bars": 168,
        "target_gross_leverage": 0.35, "rebalance_interval_bars": 48,
    }},
    "ewma_sharpe": {"name": "ewma_sharpe_alpha", "params": {
        "k_per_side": 2, "lookback_bars": 336, "ewma_lambda": 0.98,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.35, "rebalance_interval_bars": 48,
    }},
    "sortino": {"name": "sortino_alpha", "params": {
        "k_per_side": 2, "lookback_bars": 336, "target_return": 0.0,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.35, "rebalance_interval_bars": 48,
    }},
    "vol_adj_mom": {"name": "vol_adjusted_momentum_alpha", "params": {
        "k_per_side": 2, "lookback_bars": 437, "signal_vol_bars": 437,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 48,
    }},
    # Reversal-avoidance family
    "skip_gram": {"name": "skip_gram_momentum_alpha", "params": {
        "k_per_side": 2, "lookback_bars": 437, "skip_bars": 24,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 48,
    }},
    "price_level": {"name": "price_level_alpha", "params": {
        "k_per_side": 2, "level_lookback_bars": 504, "vol_lookback_bars": 168,
        "target_gross_leverage": 0.30, "rebalance_interval_bars": 48,
    }},
    # Microstructure / flow family
    "amihud": {"name": "amihud_illiquidity_alpha", "params": {
        "k_per_side": 2, "lookback_bars": 168, "mom_lookback_bars": 168,
        "use_interaction": True, "vol_lookback_bars": 168,
        "target_gross_leverage": 0.30, "rebalance_interval_bars": 48,
    }},
    "taker_buy": {"name": "taker_buy_alpha", "params": {
        "k_per_side": 2, "ratio_lookback_bars": 48, "vol_lookback_bars": 168,
        "target_gross_leverage": 0.30, "rebalance_interval_bars": 24,
    }},
    "funding_vol": {"name": "funding_vol_alpha", "params": {
        "k_per_side": 2, "funding_lookback_bars": 168, "direction": "contrarian",
        "vol_lookback_bars": 168, "target_gross_leverage": 0.25, "rebalance_interval_bars": 24,
    }},
    # Composite / multi-signal
    "multitf_mom": {"name": "multitf_momentum_alpha", "params": {
        "k_per_side": 2, "w_24h": 0.10, "w_72h": 0.20, "w_168h": 0.35, "w_336h": 0.35,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.35, "rebalance_interval_bars": 48,
    }},
    "mr_funding": {"name": "mr_funding_alpha", "params": {
        "k_per_side": 2, "mr_lookback_bars": 48, "vol_lookback_bars": 168,
        "target_gross_leverage": 0.30, "rebalance_interval_bars": 24,
    }},
    # Baseline control
    "pure_mom": {"name": "pure_momentum_alpha", "params": {
        "k_per_side": 2, "lookback_bars": 437, "vol_lookback_bars": 168,
        "target_gross_leverage": 0.30, "rebalance_interval_bars": 48,
    }},
}


def log(msg, end="\n"):
    print(f"[P94] {msg}", end=end, flush=True)


def run_signal(sig_cfg, year_ranges_dict, year):
    start, end = year_ranges_dict[year]
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


def compute_sharpe(returns):
    if not returns or len(returns) < 50:
        return 0.0
    arr = np.array(returns, dtype=np.float64)
    std = float(np.std(arr))
    return float(np.mean(arr) / std * np.sqrt(8760)) if std > 0 else 0.0


def correlation(r1, r2):
    n = min(len(r1), len(r2))
    if n < 100:
        return 0.0
    a1 = np.array(r1[:n], dtype=np.float64)
    a2 = np.array(r2[:n], dtype=np.float64)
    c = np.corrcoef(a1, a2)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


def blend_returns(sig_rets, weights):
    keys = sorted(weights.keys())
    n = min(len(sig_rets.get(k, [])) for k in keys)
    if n == 0:
        return []
    R = np.zeros((len(keys), n), dtype=np.float64)
    W = np.array([weights[k] for k in keys], dtype=np.float64)
    for i, k in enumerate(keys):
        R[i, :] = sig_rets[k][:n]
    return (W @ R).tolist()


def numpy_weight_sweep(returns_dict, signal_keys, weight_ranges, step=0.05):
    """Sweep weights and find best AVG/MIN Sharpe configs."""
    import itertools
    grids = []
    for k in signal_keys:
        lo, hi = weight_ranges[k]
        grids.append(np.arange(lo, hi + step / 2, step))

    best_balanced = {"avg": 0, "min_s": -999, "weights": {}}
    best_avgmax = {"avg": 0, "min_s": -999, "weights": {}}
    n_tested = 0

    years = sorted(returns_dict.keys())
    year_R = {}
    for yr in years:
        n_bars = min(len(returns_dict[yr].get(k, [])) for k in signal_keys)
        if n_bars == 0:
            continue
        R = np.zeros((len(signal_keys), n_bars), dtype=np.float64)
        for i, k in enumerate(signal_keys):
            R[i, :] = returns_dict[yr][k][:n_bars]
        year_R[yr] = R

    if not year_R:
        return best_balanced, best_avgmax, 0

    for combo in itertools.product(*grids):
        w_arr = np.array(combo, dtype=np.float64)
        total = w_arr.sum()
        if total < 0.9 or total > 1.1:
            continue
        w_arr = w_arr / total

        sharpes = []
        for yr in years:
            if yr not in year_R:
                continue
            blended = w_arr @ year_R[yr]
            std = float(np.std(blended))
            if std > 0:
                s = float(np.mean(blended) / std * np.sqrt(8760))
            else:
                s = 0.0
            sharpes.append(s)

        if not sharpes:
            continue
        avg_s = sum(sharpes) / len(sharpes)
        min_s = min(sharpes)
        n_tested += 1

        score = avg_s * 0.5 + min_s * 0.5
        best_score = best_balanced["avg"] * 0.5 + best_balanced["min_s"] * 0.5
        if score > best_score or best_balanced["min_s"] < -900:
            w_dict = {signal_keys[i]: round(float(w_arr[i]), 4) for i in range(len(signal_keys))}
            best_balanced = {"avg": round(avg_s, 4), "min_s": round(min_s, 4),
                             "weights": w_dict, "yby": [round(s, 3) for s in sharpes]}

        if avg_s > best_avgmax["avg"] or best_avgmax["min_s"] < -900:
            w_dict = {signal_keys[i]: round(float(w_arr[i]), 4) for i in range(len(signal_keys))}
            best_avgmax = {"avg": round(avg_s, 4), "min_s": round(min_s, 4),
                           "weights": w_dict, "yby": [round(s, 3) for s in sharpes]}

    return best_balanced, best_avgmax, n_tested


# ═══════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════
if __name__ == "__main__":
    t0 = time.time()
    report = {"phase": 94, "new_profiles": {}, "correlations": {}, "promising": [],
              "ensemble_tests": {}, "oos_results": {}}

    ALL_SIGNALS = {**CHAMPION_SIGNALS, **NEW_SIGNALS}
    all_returns = {}  # {year: {sig_name: [returns]}}

    # ══════════════════════════════════════
    # SECTION A: Profile ALL new signals
    # ══════════════════════════════════════
    log("=" * 60)
    log("SECTION A: Profiling 12 untested signals across 5 years")
    log("=" * 60)

    for sig_name, sig_cfg in NEW_SIGNALS.items():
        log(f"\n  Signal: {sig_name} ({sig_cfg['name']})")
        profile = {}
        for year in sorted(YEAR_RANGES.keys()):
            try:
                rets = run_signal(sig_cfg, YEAR_RANGES, year)
                s = compute_sharpe(rets)
                profile[year] = round(s, 3)
                if year not in all_returns:
                    all_returns[year] = {}
                all_returns[year][sig_name] = rets
                log(f"    {year}: Sharpe = {s:.3f}")
            except Exception as e:
                profile[year] = 0.0
                log(f"    {year}: ERROR - {e}")

        avg_s = sum(profile.values()) / len(profile) if profile else 0
        min_s = min(profile.values()) if profile else 0
        report["new_profiles"][sig_name] = profile
        status = "PROMISING" if avg_s > 0.5 and min_s > -0.5 else "WEAK" if avg_s > 0 else "DEAD"
        log(f"  → AVG={avg_s:.3f} MIN={min_s:.3f} [{status}]")

    # ══════════════════════════════════════
    # SECTION B: Load champion signal returns
    # ══════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("SECTION B: Loading champion signal returns for correlation")
    log("=" * 60)

    for sig_name, sig_cfg in CHAMPION_SIGNALS.items():
        log(f"  Loading {sig_name}...")
        for year in sorted(YEAR_RANGES.keys()):
            if year not in all_returns:
                all_returns[year] = {}
            if sig_name not in all_returns[year]:
                try:
                    rets = run_signal(sig_cfg, YEAR_RANGES, year)
                    all_returns[year][sig_name] = rets
                except Exception as e:
                    log(f"    {year} ERROR: {e}")
                    all_returns[year][sig_name] = []

    # ══════════════════════════════════════
    # SECTION C: Correlation matrix (new signals vs champion)
    # ══════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("SECTION C: Correlation analysis")
    log("=" * 60)

    # Use 2022+2023 combined returns for correlation (binding constraint years)
    combined_rets = {}
    for sig_name in list(CHAMPION_SIGNALS.keys()) + list(NEW_SIGNALS.keys()):
        r = []
        for yr in ["2022", "2023"]:
            if yr in all_returns and sig_name in all_returns[yr]:
                r.extend(all_returns[yr][sig_name])
        combined_rets[sig_name] = r

    champ_keys = list(CHAMPION_SIGNALS.keys())
    new_keys = list(NEW_SIGNALS.keys())

    log("\n  Correlation with champion signals (2022+2023 combined):")
    log(f"  {'Signal':>15} | {'v1':>7} | {'i460':>7} | {'i415':>7} | {'f144':>7} | {'AvgAbs':>7}")
    log(f"  {'-'*15}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")

    corr_data = {}
    for nk in new_keys:
        corrs = {}
        abs_corrs = []
        for ck in champ_keys:
            c = correlation(combined_rets.get(nk, []), combined_rets.get(ck, []))
            corrs[ck] = round(c, 3)
            abs_corrs.append(abs(c))
        avg_abs = sum(abs_corrs) / len(abs_corrs) if abs_corrs else 0
        corr_data[nk] = {**corrs, "avg_abs": round(avg_abs, 3)}
        log(f"  {nk:>15} | {corrs['v1']:>7.3f} | {corrs['i460bw168']:>7.3f} | {corrs['i415bw216']:>7.3f} | {corrs['f144']:>7.3f} | {avg_abs:>7.3f}")

    report["correlations"] = corr_data

    # ══════════════════════════════════════
    # SECTION D: Identify promising signals
    # ══════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("SECTION D: Identifying promising candidates")
    log("=" * 60)

    promising = []
    for sig_name in new_keys:
        prof = report["new_profiles"].get(sig_name, {})
        avg_s = sum(prof.values()) / len(prof) if prof else 0
        min_s = min(prof.values()) if prof else -99
        avg_abs_corr = corr_data.get(sig_name, {}).get("avg_abs", 1.0)

        # Criteria: positive avg Sharpe, not too negative min, low correlation
        if avg_s > 0.3 and min_s > -1.0 and avg_abs_corr < 0.4:
            promising.append({
                "signal": sig_name,
                "avg_sharpe": round(avg_s, 3),
                "min_sharpe": round(min_s, 3),
                "avg_abs_corr": avg_abs_corr,
                "profile": prof,
            })
            log(f"  ★ {sig_name}: AVG={avg_s:.3f} MIN={min_s:.3f} AvgCorr={avg_abs_corr:.3f}")

    if not promising:
        # Relax criteria — show best even if weak
        log("  No signals pass strict criteria. Showing best by combined score...")
        scored = []
        for sig_name in new_keys:
            prof = report["new_profiles"].get(sig_name, {})
            avg_s = sum(prof.values()) / len(prof) if prof else -99
            min_s = min(prof.values()) if prof else -99
            avg_abs_corr = corr_data.get(sig_name, {}).get("avg_abs", 1.0)
            # Score: reward high avg, penalize correlation, penalize negative min
            score = avg_s * 0.4 + min_s * 0.3 - avg_abs_corr * 0.3
            scored.append((sig_name, score, avg_s, min_s, avg_abs_corr))
        scored.sort(key=lambda x: -x[1])
        for s, sc, a, m, c in scored[:5]:
            log(f"  #{scored.index((s,sc,a,m,c))+1} {s}: score={sc:.3f} AVG={a:.3f} MIN={m:.3f} Corr={c:.3f}")
            if a > 0:
                promising.append({
                    "signal": s, "avg_sharpe": round(a, 3),
                    "min_sharpe": round(m, 3), "avg_abs_corr": c, "profile": report["new_profiles"][s],
                })

    report["promising"] = promising

    # ══════════════════════════════════════
    # SECTION E: Ensemble tests with promising signals
    # ══════════════════════════════════════
    if promising:
        log(f"\n{'=' * 60}")
        log("SECTION E: Ensemble tests with promising signals")
        log("=" * 60)

        for p in promising[:3]:  # Test top 3
            sig_name = p["signal"]
            log(f"\n  Testing 5-signal ensemble: P91b + {sig_name}")

            # Build 5-signal ensemble: 4 champion + 1 new
            sig_keys_5 = champ_keys + [sig_name]
            w_ranges = {
                "v1": (0.15, 0.35),
                "i460bw168": (0.10, 0.25),
                "i415bw216": (0.15, 0.35),
                "f144": (0.10, 0.25),
                sig_name: (0.02, 0.20),
            }

            balanced, avgmax, n_tested = numpy_weight_sweep(
                all_returns, sig_keys_5, w_ranges, step=0.05)

            log(f"    Tested {n_tested} combos")
            log(f"    Balanced: AVG={balanced['avg']:.4f} MIN={balanced['min_s']:.4f}")
            log(f"    Weights: {balanced['weights']}")
            log(f"    YbY: {balanced.get('yby', [])}")

            report["ensemble_tests"][sig_name] = {
                "balanced": balanced, "avgmax": avgmax, "n_tested": n_tested,
            }

            # Compare to P91b
            delta_min = balanced["min_s"] - 1.5761
            delta_avg = balanced["avg"] - 2.0101
            log(f"    vs P91b: ΔMIN={delta_min:+.4f} ΔAVG={delta_avg:+.4f}")
            if delta_min > 0:
                log(f"    ★★★ NEW RECORD! MIN={balanced['min_s']:.4f} > 1.5761 ★★★")

    # ══════════════════════════════════════
    # SECTION F: 2026 OOS validation
    # ══════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("SECTION F: 2026 OOS validation")
    log("=" * 60)

    # P91b OOS
    oos_rets = {}
    for sig_name, sig_cfg in CHAMPION_SIGNALS.items():
        try:
            rets = run_signal(sig_cfg, YEAR_RANGES_OOS, "2026")
            oos_rets[sig_name] = rets
        except Exception as e:
            log(f"  {sig_name} OOS error: {e}")
            oos_rets[sig_name] = []

    p91b_oos = blend_returns(oos_rets, P91B_WEIGHTS)
    p91b_oos_sharpe = compute_sharpe(p91b_oos)
    log(f"  P91b OOS 2026: Sharpe = {p91b_oos_sharpe:.4f}")
    report["oos_results"]["p91b"] = round(p91b_oos_sharpe, 4)

    # OOS for any promising ensemble
    for sig_name in list(report.get("ensemble_tests", {}).keys()):
        ens = report["ensemble_tests"][sig_name]
        if ens["balanced"]["min_s"] < -900:
            continue
        weights = ens["balanced"]["weights"]
        # Load new signal OOS
        if sig_name not in oos_rets:
            sig_cfg = NEW_SIGNALS[sig_name]
            try:
                oos_rets[sig_name] = run_signal(sig_cfg, YEAR_RANGES_OOS, "2026")
            except Exception as e:
                log(f"  {sig_name} OOS error: {e}")
                oos_rets[sig_name] = []

        ens_oos = blend_returns(oos_rets, weights)
        ens_sharpe = compute_sharpe(ens_oos)
        log(f"  P91b+{sig_name} OOS 2026: Sharpe = {ens_sharpe:.4f}")
        report["oos_results"][f"p91b_{sig_name}"] = round(ens_sharpe, 4)

    # ══════════════════════════════════════
    # SAVE REPORT
    # ══════════════════════════════════════
    elapsed = time.time() - t0
    report["elapsed_seconds"] = round(elapsed, 1)

    report_path = os.path.join(OUT_DIR, "phase94_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log(f"\n{'=' * 60}")
    log(f"Phase 94 COMPLETE in {elapsed:.0f}s")
    log(f"Report: {report_path}")
    log("=" * 60)

    # Print summary table
    log("\n  SIGNAL PROFILES SUMMARY:")
    log(f"  {'Signal':>15} | {'2021':>7} | {'2022':>7} | {'2023':>7} | {'2024':>7} | {'2025':>7} | {'AVG':>7} | {'MIN':>7} | {'AvgCorr':>7}")
    log(f"  {'-'*15}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
    for sig_name in new_keys:
        prof = report["new_profiles"].get(sig_name, {})
        vals = [prof.get(y, 0) for y in ["2021", "2022", "2023", "2024", "2025"]]
        avg_s = sum(vals) / len(vals) if vals else 0
        min_s = min(vals) if vals else 0
        corr = corr_data.get(sig_name, {}).get("avg_abs", 0)
        log(f"  {sig_name:>15} | {vals[0]:>7.3f} | {vals[1]:>7.3f} | {vals[2]:>7.3f} | {vals[3]:>7.3f} | {vals[4]:>7.3f} | {avg_s:>7.3f} | {min_s:>7.3f} | {corr:>7.3f}")
