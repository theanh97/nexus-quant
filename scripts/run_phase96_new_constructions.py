#!/usr/bin/env python3
"""
Phase 96: Genuinely New Signal Constructions
=============================================
3 NEW signal types that use existing data in fundamentally different ways:
  A) Pair Spread MR — statistical arbitrage on cointegrated pairs
  B) Volume-Regime Momentum — volume anomaly × momentum interaction
  C) Momentum Breakout — binary threshold activation

Each tested with multiple parameter variants. If promising → ensemble with P91b.
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

OUT_DIR = os.path.join(PROJ, "artifacts", "phase96")
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

# ── New signal variants to test ──
NEW_SIGNALS = {
    # Pair Spread Mean-Reversion variants
    "pair_lb72_z1": {"name": "pair_spread_alpha", "params": {
        "k_per_side": 2, "lookback_bars": 72, "entry_z": 1.0,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 24,
    }},
    "pair_lb168_z1": {"name": "pair_spread_alpha", "params": {
        "k_per_side": 2, "lookback_bars": 168, "entry_z": 1.0,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 24,
    }},
    "pair_lb168_z1.5": {"name": "pair_spread_alpha", "params": {
        "k_per_side": 2, "lookback_bars": 168, "entry_z": 1.5,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 24,
    }},
    "pair_lb336_z1": {"name": "pair_spread_alpha", "params": {
        "k_per_side": 2, "lookback_bars": 336, "entry_z": 1.0,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 24,
    }},
    "pair_lb336_z1.5": {"name": "pair_spread_alpha", "params": {
        "k_per_side": 2, "lookback_bars": 336, "entry_z": 1.5,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 24,
    }},
    "pair_lb168_z1_k4": {"name": "pair_spread_alpha", "params": {
        "k_per_side": 4, "lookback_bars": 168, "entry_z": 1.0,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 24,
    }},
    # Volume-Regime Momentum variants
    "vrm_168_336_0.5": {"name": "vol_regime_mom_alpha", "params": {
        "k_per_side": 2, "mom_lookback_bars": 168, "vol_avg_bars": 336,
        "vol_threshold": 0.5, "vol_lookback_bars": 168,
        "target_gross_leverage": 0.30, "rebalance_interval_bars": 48,
    }},
    "vrm_336_504_0.5": {"name": "vol_regime_mom_alpha", "params": {
        "k_per_side": 2, "mom_lookback_bars": 336, "vol_avg_bars": 504,
        "vol_threshold": 0.5, "vol_lookback_bars": 168,
        "target_gross_leverage": 0.30, "rebalance_interval_bars": 48,
    }},
    "vrm_168_336_1.0": {"name": "vol_regime_mom_alpha", "params": {
        "k_per_side": 2, "mom_lookback_bars": 168, "vol_avg_bars": 336,
        "vol_threshold": 1.0, "vol_lookback_bars": 168,
        "target_gross_leverage": 0.30, "rebalance_interval_bars": 48,
    }},
    "vrm_336_504_1.0": {"name": "vol_regime_mom_alpha", "params": {
        "k_per_side": 2, "mom_lookback_bars": 336, "vol_avg_bars": 504,
        "vol_threshold": 1.0, "vol_lookback_bars": 168,
        "target_gross_leverage": 0.30, "rebalance_interval_bars": 48,
    }},
    "vrm_168_336_0_k4": {"name": "vol_regime_mom_alpha", "params": {
        "k_per_side": 4, "mom_lookback_bars": 168, "vol_avg_bars": 336,
        "vol_threshold": 0.0, "vol_lookback_bars": 168,
        "target_gross_leverage": 0.30, "rebalance_interval_bars": 48,
    }},
    # Momentum Breakout variants
    "brk_336_5pct": {"name": "momentum_breakout_alpha", "params": {
        "k_per_side": 2, "lookback_bars": 336, "threshold": 0.05,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 48,
    }},
    "brk_336_10pct": {"name": "momentum_breakout_alpha", "params": {
        "k_per_side": 2, "lookback_bars": 336, "threshold": 0.10,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 48,
    }},
    "brk_168_5pct": {"name": "momentum_breakout_alpha", "params": {
        "k_per_side": 2, "lookback_bars": 168, "threshold": 0.05,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 48,
    }},
    "brk_504_5pct": {"name": "momentum_breakout_alpha", "params": {
        "k_per_side": 2, "lookback_bars": 504, "threshold": 0.05,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 48,
    }},
    "brk_336_5pct_k4": {"name": "momentum_breakout_alpha", "params": {
        "k_per_side": 4, "lookback_bars": 336, "threshold": 0.05,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 48,
    }},
    "brk_336_3pct": {"name": "momentum_breakout_alpha", "params": {
        "k_per_side": 2, "lookback_bars": 336, "threshold": 0.03,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 48,
    }},
}


def log(msg, end="\n"):
    print(f"[P96] {msg}", end=end, flush=True)


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


def numpy_weight_sweep(returns_dict, signal_keys, weight_ranges, step=0.025):
    import itertools
    grids = [np.arange(lo, hi + step / 2, step) for lo, hi in
             [weight_ranges[k] for k in signal_keys]]
    best = {"avg": 0, "min_s": -999, "weights": {}}
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
        return best, 0
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
            s = float(np.mean(blended) / std * np.sqrt(8760)) if std > 0 else 0.0
            sharpes.append(s)
        if not sharpes:
            continue
        avg_s = sum(sharpes) / len(sharpes)
        min_s = min(sharpes)
        n_tested += 1
        score = avg_s * 0.5 + min_s * 0.5
        best_score = best["avg"] * 0.5 + best["min_s"] * 0.5
        if score > best_score or best["min_s"] < -900:
            w_dict = {signal_keys[i]: round(float(w_arr[i]), 4) for i in range(len(signal_keys))}
            best = {"avg": round(avg_s, 4), "min_s": round(min_s, 4),
                    "weights": w_dict, "yby": [round(s, 3) for s in sharpes]}
    return best, n_tested


if __name__ == "__main__":
    t0 = time.time()
    report = {"phase": 96, "profiles": {}, "correlations": {}, "promising": [],
              "ensemble_tests": {}, "oos_results": {}}

    all_returns = {}
    champ_keys = list(CHAMPION_SIGNALS.keys())
    new_keys = list(NEW_SIGNALS.keys())

    # ═══ SECTION A: Profile all new signals ═══
    log("=" * 60)
    log(f"SECTION A: Profiling {len(NEW_SIGNALS)} new signal variants")
    log("=" * 60)

    for sig_name, sig_cfg in NEW_SIGNALS.items():
        log(f"\n  {sig_name}")
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
                log(f"    {year}: ERR({e})", end="")
        avg_s = sum(profile.values()) / len(profile) if profile else 0
        min_s = min(profile.values()) if profile else 0
        report["profiles"][sig_name] = profile
        tag = "★" if avg_s > 0.5 and min_s > -0.3 else "~" if avg_s > 0 else "✗"
        log(f"  → AVG={avg_s:.3f} MIN={min_s:.3f} [{tag}]")

    # ═══ SECTION B: Champion returns + correlation ═══
    log(f"\n{'=' * 60}")
    log("SECTION B: Champion returns + correlation")
    log("=" * 60)

    for sig_name, sig_cfg in CHAMPION_SIGNALS.items():
        log(f"  Loading {sig_name}...")
        for year in sorted(YEAR_RANGES.keys()):
            if year not in all_returns:
                all_returns[year] = {}
            if sig_name not in all_returns[year]:
                try:
                    all_returns[year][sig_name] = run_signal(sig_cfg, YEAR_RANGES, year)
                except:
                    all_returns[year][sig_name] = []

    combined = {}
    for k in champ_keys + new_keys:
        r = []
        for yr in ["2022", "2023"]:
            if yr in all_returns and k in all_returns[yr]:
                r.extend(all_returns[yr][k])
        combined[k] = r

    log(f"\n  {'Signal':>20} | {'v1':>6} | {'i460':>6} | {'i415':>6} | {'f144':>6} | {'Avg':>6} | {'AVG_S':>6} | {'MIN_S':>6}")
    log(f"  {'-'*20}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")

    candidates = []
    for nk in new_keys:
        prof = report["profiles"].get(nk, {})
        avg_s = sum(prof.values()) / len(prof) if prof else 0
        min_s = min(prof.values()) if prof else -99
        corrs = {}
        abs_corrs = []
        for ck in champ_keys:
            c = correlation(combined.get(nk, []), combined.get(ck, []))
            corrs[ck] = round(c, 3)
            abs_corrs.append(abs(c))
        avg_corr = sum(abs_corrs) / len(abs_corrs) if abs_corrs else 0
        report["correlations"][nk] = {**corrs, "avg_abs": round(avg_corr, 3)}
        log(f"  {nk:>20} | {corrs.get('v1',0):>6.3f} | {corrs.get('i460bw168',0):>6.3f} | {corrs.get('i415bw216',0):>6.3f} | {corrs.get('f144',0):>6.3f} | {avg_corr:>6.3f} | {avg_s:>6.3f} | {min_s:>6.3f}")
        score = avg_s * 0.35 + min_s * 0.35 - avg_corr * 0.30
        candidates.append((nk, score, avg_s, min_s, avg_corr))

    candidates.sort(key=lambda x: -x[1])
    log(f"\n  TOP 5:")
    for i, (n, sc, a, m, c) in enumerate(candidates[:5]):
        log(f"  #{i+1} {n}: score={sc:.3f} AVG={a:.3f} MIN={m:.3f} Corr={c:.3f}")
        report["promising"].append({"name": n, "score": round(sc, 3),
                                     "avg": round(a, 3), "min": round(m, 3), "corr": round(c, 3)})

    # ═══ SECTION C: Ensemble with top candidates ═══
    log(f"\n{'=' * 60}")
    log("SECTION C: 5-signal ensemble with top 3 candidates")
    log("=" * 60)

    for name, score, avg_s, min_s, avg_corr in candidates[:3]:
        if avg_s < 0.1:
            log(f"  Skipping {name} (AVG too low)")
            continue
        log(f"\n  P91b + {name}")
        sig_keys_5 = champ_keys + [name]
        w_ranges = {
            "v1": (0.15, 0.35),
            "i460bw168": (0.10, 0.25),
            "i415bw216": (0.15, 0.35),
            "f144": (0.10, 0.25),
            name: (0.02, 0.20),
        }
        balanced, n_tested = numpy_weight_sweep(all_returns, sig_keys_5, w_ranges, step=0.025)
        log(f"    Tested {n_tested} | AVG={balanced['avg']:.4f} MIN={balanced['min_s']:.4f}")
        log(f"    Weights: {balanced['weights']}")
        log(f"    YbY: {balanced.get('yby', [])}")
        delta = balanced["min_s"] - 1.5761
        log(f"    vs P91b: ΔMIN={delta:+.4f}")
        report["ensemble_tests"][name] = {"balanced": balanced, "n_tested": n_tested, "delta_min": round(delta, 4)}
        if delta > 0:
            log(f"    ★★★ BEATS P91b! ★★★")

    # ═══ SECTION D: 2026 OOS ═══
    log(f"\n{'=' * 60}")
    log("SECTION D: 2026 OOS")
    log("=" * 60)

    oos_rets = {}
    for sig_name, sig_cfg in CHAMPION_SIGNALS.items():
        try:
            oos_rets[sig_name] = run_signal(sig_cfg, YEAR_RANGES_OOS, "2026")
        except Exception as e:
            oos_rets[sig_name] = []

    p91b_oos = blend_returns(oos_rets, P91B_WEIGHTS)
    p91b_sharpe = compute_sharpe(p91b_oos)
    log(f"  P91b: {p91b_sharpe:.4f}")
    report["oos_results"]["p91b"] = round(p91b_sharpe, 4)

    for name in list(report.get("ensemble_tests", {}).keys()):
        ens = report["ensemble_tests"][name]
        weights = ens["balanced"]["weights"]
        if name not in oos_rets:
            try:
                oos_rets[name] = run_signal(NEW_SIGNALS[name], YEAR_RANGES_OOS, "2026")
            except:
                oos_rets[name] = []
        ens_oos = blend_returns(oos_rets, weights)
        ens_sharpe = compute_sharpe(ens_oos)
        log(f"  P91b+{name}: {ens_sharpe:.4f}")
        report["oos_results"][f"p91b_{name}"] = round(ens_sharpe, 4)

    elapsed = time.time() - t0
    report["elapsed_seconds"] = round(elapsed, 1)
    rp = os.path.join(OUT_DIR, "phase96_report.json")
    with open(rp, "w") as f:
        json.dump(report, f, indent=2)

    log(f"\n{'=' * 60}")
    log(f"Phase 96 COMPLETE in {elapsed:.0f}s → {rp}")
    log("=" * 60)

    log(f"\n  {'Signal':>20} | {'2021':>6} | {'2022':>6} | {'2023':>6} | {'2024':>6} | {'2025':>6} | {'AVG':>6} | {'MIN':>6} | {'Corr':>6}")
    log(f"  {'-'*20}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")
    for sk in new_keys:
        p = report["profiles"].get(sk, {})
        v = [p.get(y, 0) for y in ["2021", "2022", "2023", "2024", "2025"]]
        a = sum(v) / len(v)
        m = min(v)
        c = report["correlations"].get(sk, {}).get("avg_abs", 0)
        log(f"  {sk:>20} | {v[0]:>6.3f} | {v[1]:>6.3f} | {v[2]:>6.3f} | {v[3]:>6.3f} | {v[4]:>6.3f} | {a:>6.3f} | {m:>6.3f} | {c:>6.3f}")
