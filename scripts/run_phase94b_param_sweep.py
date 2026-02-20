#!/usr/bin/env python3
"""
Phase 94b: Parameter sweep for top signals from Phase 94
========================================================
Focus on signals with either:
  - Low correlation + some profit (sharpe_ratio: corr=0.292, 2022=1.90)
  - High profit + moderate corr (vol_adj_mom, skip_gram, multitf_mom)

Sweep lookback params to find variants that:
  1. Have positive Sharpe in ALL 5 years (or at least >-0.2)
  2. Have low correlation with champion (<0.35)
  3. If found: test 5-signal ensemble

Also try: sharpe_ratio with k=4 (more positions = more diversified signal)
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

# Champion signals for correlation reference
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

# ── Parameter sweep definitions ──
SWEEP_CONFIGS = {
    # Sharpe Ratio: vary lookback and k_per_side
    "sr_lb168_k2": {"name": "sharpe_ratio_alpha", "params": {
        "k_per_side": 2, "lookback_bars": 168, "vol_lookback_bars": 168,
        "target_gross_leverage": 0.35, "rebalance_interval_bars": 48,
    }},
    "sr_lb336_k2": {"name": "sharpe_ratio_alpha", "params": {
        "k_per_side": 2, "lookback_bars": 336, "vol_lookback_bars": 168,
        "target_gross_leverage": 0.35, "rebalance_interval_bars": 48,
    }},
    "sr_lb504_k2": {"name": "sharpe_ratio_alpha", "params": {
        "k_per_side": 2, "lookback_bars": 504, "vol_lookback_bars": 168,
        "target_gross_leverage": 0.35, "rebalance_interval_bars": 48,
    }},
    "sr_lb168_k4": {"name": "sharpe_ratio_alpha", "params": {
        "k_per_side": 4, "lookback_bars": 168, "vol_lookback_bars": 168,
        "target_gross_leverage": 0.30, "rebalance_interval_bars": 48,
    }},
    "sr_lb336_k4": {"name": "sharpe_ratio_alpha", "params": {
        "k_per_side": 4, "lookback_bars": 336, "vol_lookback_bars": 168,
        "target_gross_leverage": 0.30, "rebalance_interval_bars": 48,
    }},
    "sr_lb504_k4": {"name": "sharpe_ratio_alpha", "params": {
        "k_per_side": 4, "lookback_bars": 504, "vol_lookback_bars": 168,
        "target_gross_leverage": 0.30, "rebalance_interval_bars": 48,
    }},
    # Vol-adjusted momentum: vary lookback
    "vam_lb168": {"name": "vol_adjusted_momentum_alpha", "params": {
        "k_per_side": 2, "lookback_bars": 168, "signal_vol_bars": 168,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 48,
    }},
    "vam_lb336": {"name": "vol_adjusted_momentum_alpha", "params": {
        "k_per_side": 2, "lookback_bars": 336, "signal_vol_bars": 336,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 48,
    }},
    "vam_lb504": {"name": "vol_adjusted_momentum_alpha", "params": {
        "k_per_side": 2, "lookback_bars": 504, "signal_vol_bars": 504,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 48,
    }},
    # Skip-gram: vary lookback and skip
    "sg_lb336_sk24": {"name": "skip_gram_momentum_alpha", "params": {
        "k_per_side": 2, "lookback_bars": 336, "skip_bars": 24,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 48,
    }},
    "sg_lb437_sk48": {"name": "skip_gram_momentum_alpha", "params": {
        "k_per_side": 2, "lookback_bars": 437, "skip_bars": 48,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 48,
    }},
    "sg_lb336_sk48": {"name": "skip_gram_momentum_alpha", "params": {
        "k_per_side": 2, "lookback_bars": 336, "skip_bars": 48,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 48,
    }},
    # Multi-TF: vary horizon weights (more short-term emphasis)
    "mtf_short": {"name": "multitf_momentum_alpha", "params": {
        "k_per_side": 2, "w_24h": 0.30, "w_72h": 0.30, "w_168h": 0.20, "w_336h": 0.20,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.35, "rebalance_interval_bars": 48,
    }},
    "mtf_mid": {"name": "multitf_momentum_alpha", "params": {
        "k_per_side": 2, "w_24h": 0.05, "w_72h": 0.35, "w_168h": 0.35, "w_336h": 0.25,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.35, "rebalance_interval_bars": 48,
    }},
    # Sortino: vary lookback
    "sort_lb168": {"name": "sortino_alpha", "params": {
        "k_per_side": 2, "lookback_bars": 168, "target_return": 0.0,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.35, "rebalance_interval_bars": 48,
    }},
    "sort_lb504": {"name": "sortino_alpha", "params": {
        "k_per_side": 2, "lookback_bars": 504, "target_return": 0.0,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.35, "rebalance_interval_bars": 48,
    }},
    "sort_lb336_k4": {"name": "sortino_alpha", "params": {
        "k_per_side": 4, "lookback_bars": 336, "target_return": 0.0,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 48,
    }},
}


def log(msg, end="\n"):
    print(f"[P94b] {msg}", end=end, flush=True)


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


def numpy_weight_sweep_5sig(returns_dict, signal_keys, weight_ranges, step=0.025):
    """Fine-grained sweep for 5-signal ensemble."""
    import itertools
    grids = []
    for k in signal_keys:
        lo, hi = weight_ranges[k]
        grids.append(np.arange(lo, hi + step / 2, step))

    best_balanced = {"avg": 0, "min_s": -999, "weights": {}}
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
        return best_balanced, 0

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

    return best_balanced, n_tested


if __name__ == "__main__":
    t0 = time.time()
    report = {"phase": "94b", "sweep_profiles": {}, "sweep_corr": {},
              "best_candidates": [], "ensemble_tests": {}, "oos_results": {}}

    all_returns = {}  # {year: {sig_name: [returns]}}

    # ══════════════════════════════════════
    # SECTION A: Profile all sweep variants
    # ══════════════════════════════════════
    log("=" * 60)
    log(f"SECTION A: Profiling {len(SWEEP_CONFIGS)} parameter variants across 5 years")
    log("=" * 60)

    for sig_name, sig_cfg in SWEEP_CONFIGS.items():
        log(f"\n  {sig_name} ({sig_cfg['name']})")
        profile = {}
        for year in sorted(YEAR_RANGES.keys()):
            try:
                rets = run_signal(sig_cfg, YEAR_RANGES, year)
                s = compute_sharpe(rets)
                profile[year] = round(s, 3)
                if year not in all_returns:
                    all_returns[year] = {}
                all_returns[year][sig_name] = rets
                log(f"    {year}: {s:.3f}", end="")
            except Exception as e:
                profile[year] = 0.0
                log(f"    {year}: ERR", end="")

        avg_s = sum(profile.values()) / len(profile) if profile else 0
        min_s = min(profile.values()) if profile else 0
        report["sweep_profiles"][sig_name] = profile
        log(f"  → AVG={avg_s:.3f} MIN={min_s:.3f}")

    # ══════════════════════════════════════
    # SECTION B: Load champion returns + correlation
    # ══════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("SECTION B: Champion returns + correlation analysis")
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
                    all_returns[year][sig_name] = []

    # Combined 2022+2023 returns for correlation (binding years)
    combined = {}
    all_keys = list(CHAMPION_SIGNALS.keys()) + list(SWEEP_CONFIGS.keys())
    for k in all_keys:
        r = []
        for yr in ["2022", "2023"]:
            if yr in all_returns and k in all_returns[yr]:
                r.extend(all_returns[yr][k])
        combined[k] = r

    champ_keys = list(CHAMPION_SIGNALS.keys())
    sweep_keys = list(SWEEP_CONFIGS.keys())

    log("\n  Correlation with champion (2022+2023):")
    log(f"  {'Variant':>18} | {'v1':>6} | {'i460':>6} | {'i415':>6} | {'f144':>6} | {'Avg':>6} | {'AVG_S':>6} | {'MIN_S':>6}")
    log(f"  {'-'*18}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")

    candidates = []
    for sk in sweep_keys:
        prof = report["sweep_profiles"].get(sk, {})
        avg_s = sum(prof.values()) / len(prof) if prof else 0
        min_s = min(prof.values()) if prof else -99

        corrs = {}
        abs_corrs = []
        for ck in champ_keys:
            c = correlation(combined.get(sk, []), combined.get(ck, []))
            corrs[ck] = round(c, 3)
            abs_corrs.append(abs(c))
        avg_corr = sum(abs_corrs) / len(abs_corrs) if abs_corrs else 0
        report["sweep_corr"][sk] = {**corrs, "avg_abs": round(avg_corr, 3)}

        log(f"  {sk:>18} | {corrs['v1']:>6.3f} | {corrs['i460bw168']:>6.3f} | {corrs['i415bw216']:>6.3f} | {corrs['f144']:>6.3f} | {avg_corr:>6.3f} | {avg_s:>6.3f} | {min_s:>6.3f}")

        # Score: reward high avg, low corr, penalty for negative min
        score = avg_s * 0.35 + min_s * 0.35 - avg_corr * 0.30
        candidates.append((sk, score, avg_s, min_s, avg_corr))

    candidates.sort(key=lambda x: -x[1])

    log(f"\n  TOP 5 candidates by composite score:")
    for i, (name, score, avg, mn, corr) in enumerate(candidates[:5]):
        log(f"  #{i+1} {name}: score={score:.3f} AVG={avg:.3f} MIN={mn:.3f} Corr={corr:.3f}")

    report["best_candidates"] = [
        {"name": c[0], "score": round(c[1], 3), "avg": round(c[2], 3),
         "min": round(c[3], 3), "corr": round(c[4], 3)}
        for c in candidates[:5]
    ]

    # ══════════════════════════════════════
    # SECTION C: 5-signal ensemble with top candidates
    # ══════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("SECTION C: 5-signal ensemble tests with top 3 candidates")
    log("=" * 60)

    for name, score, avg_s, min_s, avg_corr in candidates[:3]:
        if avg_s < 0.2:
            log(f"  Skipping {name} (AVG too low: {avg_s:.3f})")
            continue

        log(f"\n  Testing P91b + {name}")
        sig_keys_5 = champ_keys + [name]
        w_ranges = {
            "v1": (0.15, 0.35),
            "i460bw168": (0.10, 0.25),
            "i415bw216": (0.15, 0.35),
            "f144": (0.10, 0.25),
            name: (0.02, 0.20),
        }

        balanced, n_tested = numpy_weight_sweep_5sig(
            all_returns, sig_keys_5, w_ranges, step=0.025)

        log(f"    Tested {n_tested} combos")
        log(f"    Balanced: AVG={balanced['avg']:.4f} MIN={balanced['min_s']:.4f}")
        log(f"    Weights: {balanced['weights']}")
        log(f"    YbY: {balanced.get('yby', [])}")

        delta_min = balanced["min_s"] - 1.5761
        delta_avg = balanced["avg"] - 2.0101
        log(f"    vs P91b: ΔMIN={delta_min:+.4f} ΔAVG={delta_avg:+.4f}")

        report["ensemble_tests"][name] = {
            "balanced": balanced, "n_tested": n_tested,
            "delta_min": round(delta_min, 4), "delta_avg": round(delta_avg, 4),
        }

        if delta_min > 0:
            log(f"    ★★★ BEATS P91b! MIN={balanced['min_s']:.4f} > 1.5761 ★★★")

    # ══════════════════════════════════════
    # SECTION D: 2026 OOS for best ensemble
    # ══════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("SECTION D: 2026 OOS validation")
    log("=" * 60)

    oos_rets = {}
    for sig_name, sig_cfg in CHAMPION_SIGNALS.items():
        try:
            oos_rets[sig_name] = run_signal(sig_cfg, YEAR_RANGES_OOS, "2026")
        except Exception as e:
            log(f"  {sig_name} OOS error: {e}")
            oos_rets[sig_name] = []

    p91b_oos = blend_returns(oos_rets, P91B_WEIGHTS)
    p91b_oos_sharpe = compute_sharpe(p91b_oos)
    log(f"  P91b OOS 2026: {p91b_oos_sharpe:.4f}")
    report["oos_results"]["p91b"] = round(p91b_oos_sharpe, 4)

    for name in list(report.get("ensemble_tests", {}).keys()):
        ens = report["ensemble_tests"][name]
        weights = ens["balanced"]["weights"]
        if name not in oos_rets:
            try:
                oos_rets[name] = run_signal(SWEEP_CONFIGS[name], YEAR_RANGES_OOS, "2026")
            except Exception as e:
                log(f"  {name} OOS error: {e}")
                oos_rets[name] = []

        ens_oos = blend_returns(oos_rets, weights)
        ens_sharpe = compute_sharpe(ens_oos)
        log(f"  P91b+{name} OOS 2026: {ens_sharpe:.4f}")
        report["oos_results"][f"p91b_{name}"] = round(ens_sharpe, 4)

    # ══════════════════════════════════════
    # SAVE
    # ══════════════════════════════════════
    elapsed = time.time() - t0
    report["elapsed_seconds"] = round(elapsed, 1)

    report_path = os.path.join(OUT_DIR, "phase94b_sweep_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    log(f"\n{'=' * 60}")
    log(f"Phase 94b COMPLETE in {elapsed:.0f}s")
    log(f"Report: {report_path}")
    log("=" * 60)

    # Summary table
    log("\n  SWEEP SUMMARY:")
    log(f"  {'Variant':>18} | {'2021':>6} | {'2022':>6} | {'2023':>6} | {'2024':>6} | {'2025':>6} | {'AVG':>6} | {'MIN':>6} | {'Corr':>6}")
    log(f"  {'-'*18}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")
    for sk in sweep_keys:
        prof = report["sweep_profiles"].get(sk, {})
        vals = [prof.get(y, 0) for y in ["2021", "2022", "2023", "2024", "2025"]]
        avg_s = sum(vals) / len(vals)
        min_s = min(vals)
        corr = report["sweep_corr"].get(sk, {}).get("avg_abs", 0)
        log(f"  {sk:>18} | {vals[0]:>6.3f} | {vals[1]:>6.3f} | {vals[2]:>6.3f} | {vals[3]:>6.3f} | {vals[4]:>6.3f} | {avg_s:>6.3f} | {min_s:>6.3f} | {corr:>6.3f}")
