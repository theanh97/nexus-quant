#!/usr/bin/env python3
"""
Phase 104: Regime-Conditional Signal Attribution + Quarterly Analysis
=====================================================================
Deep-dive into WHEN each signal contributes:
  A) Quarterly Sharpe breakdown (20 quarters, 2021Q1-2025Q4)
  B) Regime classification (BTC return: bull/bear/chop per quarter)
  C) Signal contribution per regime: which signals carry which regimes?
  D) Rolling 3-month signal-level Sharpe → identify regime transitions
  E) "Best of" analysis: if we could pick the best signal per quarter
"""

import copy, json, os, sys, time

PROJ = "/Users/qtmobile/Desktop/Nexus - Quant Trading "
sys.path.insert(0, PROJ)

import numpy as np

from nexus_quant.data.providers.registry import make_provider
from nexus_quant.strategies.registry import make_strategy
from nexus_quant.backtest.engine import BacktestConfig, BacktestEngine
from nexus_quant.backtest.costs import cost_model_from_config

OUT_DIR = os.path.join(PROJ, "artifacts", "phase104")
os.makedirs(OUT_DIR, exist_ok=True)

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
]

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

# Quarters: 2021Q1 through 2025Q4
QUARTERS = []
for year in range(2021, 2026):
    for q in range(1, 5):
        start_month = (q - 1) * 3 + 1
        end_month = q * 3 + 1
        end_year = year
        if end_month > 12:
            end_month = 1
            end_year = year + 1
        start = f"{year}-{start_month:02d}-01"
        end = f"{end_year}-{end_month:02d}-01"
        QUARTERS.append({
            "label": f"{year}Q{q}",
            "start": start,
            "end": end,
            "year": year,
            "q": q,
        })


def log(msg):
    print(f"[P104] {msg}", flush=True)


def compute_sharpe(returns, bars_per_year=8760):
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size < 50:
        return 0.0
    std = float(np.std(arr))
    if std <= 0:
        return 0.0
    return float(np.mean(arr) / std * np.sqrt(bars_per_year))


def compute_cum_return(returns):
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.prod(1.0 + arr) - 1.0)


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
    return result.returns, dataset


def get_btc_return(dataset, start_idx=0, end_idx=None):
    """Get BTC price return from dataset for regime classification."""
    try:
        btc_idx = dataset.symbols.index("BTCUSDT")
        closes = dataset.close[start_idx:end_idx, btc_idx]
        if len(closes) < 2:
            return 0.0
        return float(closes[-1] / closes[0] - 1.0)
    except Exception:
        return 0.0


def classify_regime(btc_quarterly_return):
    """Classify BTC quarterly return into regime."""
    if btc_quarterly_return > 0.20:
        return "strong_bull"
    elif btc_quarterly_return > 0.05:
        return "mild_bull"
    elif btc_quarterly_return > -0.05:
        return "chop"
    elif btc_quarterly_return > -0.20:
        return "mild_bear"
    else:
        return "strong_bear"


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


if __name__ == "__main__":
    t0 = time.time()
    report = {"phase": 104}

    # ════════════════════════════════════
    # SECTION A: Quarterly signal returns
    # ════════════════════════════════════
    log("=" * 60)
    log("SECTION A: Quarterly signal computation (20 quarters)")
    log("=" * 60)

    quarterly_data = {}
    for q_info in QUARTERS:
        label = q_info["label"]
        log(f"  {label} ({q_info['start']} to {q_info['end']})")

        sig_rets = {}
        btc_return = None
        for sig_key in SIG_KEYS:
            try:
                rets, ds = run_signal(SIGNALS[sig_key], q_info["start"], q_info["end"])
                if btc_return is None:
                    btc_return = get_btc_return(ds)
            except Exception as exc:
                log(f"    {sig_key} ERROR: {exc}")
                rets = []
            sig_rets[sig_key] = rets

        # Blend
        blended = blend(sig_rets, P91B_WEIGHTS)

        # Compute metrics
        sig_sharpes = {k: round(compute_sharpe(sig_rets[k]), 4) for k in SIG_KEYS}
        sig_cum_rets = {k: round(compute_cum_return(sig_rets[k]) * 100, 2) for k in SIG_KEYS}
        blend_sharpe = round(compute_sharpe(blended), 4)
        blend_cum_ret = round(compute_cum_return(blended) * 100, 2)
        btc_ret_pct = round((btc_return or 0) * 100, 2)
        regime = classify_regime(btc_return or 0)

        quarterly_data[label] = {
            "start": q_info["start"],
            "end": q_info["end"],
            "btc_return_pct": btc_ret_pct,
            "regime": regime,
            "signal_sharpes": sig_sharpes,
            "signal_cum_returns_pct": sig_cum_rets,
            "blend_sharpe": blend_sharpe,
            "blend_cum_return_pct": blend_cum_ret,
        }
        log(f"    BTC={btc_ret_pct:+.1f}% ({regime}), Blend Sharpe={blend_sharpe:.3f}, "
            f"Signals: {sig_sharpes}")

    report["quarterly"] = quarterly_data

    # ════════════════════════════════════
    # SECTION B: Regime-conditional analysis
    # ════════════════════════════════════
    log("\n" + "=" * 60)
    log("SECTION B: Performance by market regime")
    log("=" * 60)

    regime_buckets = {}
    for label, data in quarterly_data.items():
        regime = data["regime"]
        if regime not in regime_buckets:
            regime_buckets[regime] = []
        regime_buckets[regime].append(data)

    regime_analysis = {}
    for regime, quarters in sorted(regime_buckets.items()):
        n_q = len(quarters)
        # Average signal Sharpe per regime
        avg_sig_sharpes = {}
        for sig_key in SIG_KEYS:
            vals = [q["signal_sharpes"][sig_key] for q in quarters]
            avg_sig_sharpes[sig_key] = round(sum(vals) / len(vals), 4)

        avg_blend = round(sum(q["blend_sharpe"] for q in quarters) / n_q, 4)
        avg_btc = round(sum(q["btc_return_pct"] for q in quarters) / n_q, 2)

        # Which signal is best in this regime?
        best_sig = max(avg_sig_sharpes, key=lambda k: avg_sig_sharpes[k])

        regime_analysis[regime] = {
            "n_quarters": n_q,
            "avg_btc_return_pct": avg_btc,
            "avg_blend_sharpe": avg_blend,
            "avg_signal_sharpes": avg_sig_sharpes,
            "best_signal": best_sig,
            "quarters": [q["blend_sharpe"] for q in quarters],
        }
        log(f"\n  {regime} ({n_q} quarters, avg BTC={avg_btc:+.1f}%)")
        log(f"    Blend: {avg_blend:.3f}")
        log(f"    Signals: {avg_sig_sharpes}")
        log(f"    Best signal: {best_sig}")

    report["regime_analysis"] = regime_analysis

    # ════════════════════════════════════
    # SECTION C: Signal contribution attribution
    # ════════════════════════════════════
    log("\n" + "=" * 60)
    log("SECTION C: Signal contribution (weighted Sharpe contribution)")
    log("=" * 60)

    # For each quarter, compute each signal's weighted contribution
    # Contribution_i = w_i * signal_i_cum_return
    contribution_totals = {k: 0.0 for k in SIG_KEYS}
    n_quarters_counted = 0
    for label, data in quarterly_data.items():
        total_blend = data["blend_cum_return_pct"]
        if abs(total_blend) < 0.001:
            continue
        n_quarters_counted += 1
        for sig_key in SIG_KEYS:
            contrib = P91B_WEIGHTS[sig_key] * data["signal_cum_returns_pct"][sig_key]
            contribution_totals[sig_key] += contrib

    # Normalize to show % attribution
    total_contrib = sum(abs(v) for v in contribution_totals.values())
    contribution_pct = {}
    if total_contrib > 0:
        contribution_pct = {k: round(v / total_contrib * 100, 1) for k, v in contribution_totals.items()}

    report["signal_contribution"] = {
        "weighted_cum_return_contribution": {k: round(v, 2) for k, v in contribution_totals.items()},
        "attribution_pct": contribution_pct,
        "n_quarters": n_quarters_counted,
    }
    log(f"  Weighted return contribution over {n_quarters_counted} quarters:")
    for k in SIG_KEYS:
        log(f"    {k}: {contribution_totals[k]:+.2f}% total ({contribution_pct.get(k, 0):.1f}% attribution)")

    # ════════════════════════════════════
    # SECTION D: Signal consistency analysis
    # ════════════════════════════════════
    log("\n" + "=" * 60)
    log("SECTION D: Signal consistency across quarters")
    log("=" * 60)

    signal_consistency = {}
    for sig_key in SIG_KEYS:
        sharpes = [quarterly_data[label]["signal_sharpes"][sig_key] for label in quarterly_data]
        positive_q = sum(1 for s in sharpes if s > 0)
        negative_q = sum(1 for s in sharpes if s <= 0)
        avg_s = round(sum(sharpes) / len(sharpes), 4) if sharpes else 0
        std_s = round(float(np.std(sharpes)), 4) if sharpes else 0
        best_q = max(sharpes) if sharpes else 0
        worst_q = min(sharpes) if sharpes else 0

        signal_consistency[sig_key] = {
            "avg_quarterly_sharpe": avg_s,
            "std_quarterly_sharpe": std_s,
            "positive_quarters": positive_q,
            "negative_quarters": negative_q,
            "hit_rate_pct": round(positive_q / len(sharpes) * 100, 1) if sharpes else 0,
            "best_quarter_sharpe": round(best_q, 4),
            "worst_quarter_sharpe": round(worst_q, 4),
            "information_ratio": round(avg_s / std_s, 4) if std_s > 0 else 0,
        }
        log(f"  {sig_key}: avg={avg_s:.3f} ± {std_s:.3f}, hit={positive_q}/{len(sharpes)} "
            f"({signal_consistency[sig_key]['hit_rate_pct']:.0f}%), "
            f"best={best_q:.3f}, worst={worst_q:.3f}")

    # Blend consistency
    blend_sharpes = [quarterly_data[label]["blend_sharpe"] for label in quarterly_data]
    blend_positive = sum(1 for s in blend_sharpes if s > 0)
    signal_consistency["blend_p91b"] = {
        "avg_quarterly_sharpe": round(sum(blend_sharpes) / len(blend_sharpes), 4),
        "std_quarterly_sharpe": round(float(np.std(blend_sharpes)), 4),
        "positive_quarters": blend_positive,
        "negative_quarters": len(blend_sharpes) - blend_positive,
        "hit_rate_pct": round(blend_positive / len(blend_sharpes) * 100, 1),
        "best_quarter_sharpe": round(max(blend_sharpes), 4),
        "worst_quarter_sharpe": round(min(blend_sharpes), 4),
    }
    log(f"\n  BLEND: avg={signal_consistency['blend_p91b']['avg_quarterly_sharpe']:.3f}, "
        f"hit={blend_positive}/{len(blend_sharpes)} "
        f"({signal_consistency['blend_p91b']['hit_rate_pct']:.0f}%), "
        f"worst={signal_consistency['blend_p91b']['worst_quarter_sharpe']:.3f}")

    report["signal_consistency"] = signal_consistency

    # ════════════════════════════════════
    # SECTION E: "Best of" analysis
    # ════════════════════════════════════
    log("\n" + "=" * 60)
    log("SECTION E: 'Best-of' signal per quarter")
    log("=" * 60)

    best_of_rets = []
    best_of_winners = {k: 0 for k in SIG_KEYS}
    for label in quarterly_data:
        sharpes = quarterly_data[label]["signal_sharpes"]
        best_sig = max(sharpes, key=lambda k: sharpes[k])
        best_of_winners[best_sig] += 1
        best_of_rets.append(quarterly_data[label]["signal_cum_returns_pct"][best_sig])

    best_of_avg_ret = round(sum(best_of_rets) / len(best_of_rets), 2) if best_of_rets else 0

    report["best_of_analysis"] = {
        "signal_win_counts": best_of_winners,
        "avg_best_quarterly_return_pct": best_of_avg_ret,
        "note": "Hypothetical: if oracle chose best signal each quarter",
    }
    log(f"  Signal win counts (best per quarter): {best_of_winners}")
    log(f"  Avg best quarterly return: {best_of_avg_ret:.2f}%")

    # ════════════════════════════════════
    # SUMMARY TABLE
    # ════════════════════════════════════
    log("\n" + "=" * 60)
    log("QUARTERLY SUMMARY TABLE")
    log("=" * 60)

    header = f"{'Quarter':>8} | {'BTC%':>7} | {'Regime':>12} | {'Blend':>7} | {'v1':>7} | {'i415':>7} | {'i460':>7} | {'f144':>7}"
    log(header)
    log("-" * len(header))
    for label in sorted(quarterly_data.keys()):
        d = quarterly_data[label]
        s = d["signal_sharpes"]
        log(f"{label:>8} | {d['btc_return_pct']:>+7.1f} | {d['regime']:>12} | "
            f"{d['blend_sharpe']:>7.3f} | {s['v1']:>7.3f} | {s['i415bw216']:>7.3f} | "
            f"{s['i460bw168']:>7.3f} | {s['f144']:>7.3f}")

    # ════════════════════════════════════
    # SAVE
    # ════════════════════════════════════
    elapsed = round(time.time() - t0, 1)
    report["elapsed_seconds"] = elapsed

    out_path = os.path.join(OUT_DIR, "phase104_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    log("\n" + "=" * 60)
    log(f"Phase 104 COMPLETE in {elapsed}s → {out_path}")
    log("=" * 60)

    # Key insights
    log("\nKEY INSIGHTS:")
    # Which signal is most consistent?
    best_consistency = max(SIG_KEYS, key=lambda k: signal_consistency[k]["hit_rate_pct"])
    log(f"  Most consistent: {best_consistency} ({signal_consistency[best_consistency]['hit_rate_pct']:.0f}% positive quarters)")

    # Which signal is best in bear markets?
    bear_regimes = [r for r in regime_analysis if "bear" in r]
    if bear_regimes:
        bear_data = [regime_analysis[r] for r in bear_regimes]
        bear_best = {}
        for sig_key in SIG_KEYS:
            bear_best[sig_key] = sum(d["avg_signal_sharpes"][sig_key] for d in bear_data) / len(bear_data)
        best_bear_sig = max(bear_best, key=lambda k: bear_best[k])
        log(f"  Best in bear markets: {best_bear_sig} (avg Sharpe={bear_best[best_bear_sig]:.3f})")

    # Which signal is most volatile across regimes?
    for sig_key in SIG_KEYS:
        regime_sharpes = [regime_analysis[r]["avg_signal_sharpes"][sig_key] for r in regime_analysis]
        spread = max(regime_sharpes) - min(regime_sharpes) if regime_sharpes else 0
        log(f"  {sig_key} regime spread: {spread:.3f}")
