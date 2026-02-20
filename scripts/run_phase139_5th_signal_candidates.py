#!/usr/bin/env python3
"""
Phase 139: 5th Signal Candidates — Volume Reversal, Vol Breakout, Lead-Lag
===========================================================================
The overlay research (Phases 127-138) is exhausted. Vol overlay is in production.
Now test orthogonal alpha signals to add as a 5th signal to the P91b ensemble.

Candidates:
  1. Volume Reversal Alpha — capitulation/climax detection via volume × return
  2. Volatility Breakout Alpha — trade breakouts after vol compression
  3. Lead-Lag Alpha — BTC leads altcoin moves

Tests:
  1. Standalone Sharpe per year (each candidate)
  2. Correlation with existing 4-signal ensemble returns
  3. 5-signal blend optimization (add candidate to P91b)
  4. Walk-forward validation (LOYO) of best blend
"""
import json
import os
import signal as _signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from nexus_quant.backtest.engine import BacktestConfig, BacktestEngine
from nexus_quant.backtest.costs import cost_model_from_config
from nexus_quant.data.providers.registry import make_provider
from nexus_quant.strategies.registry import make_strategy

_partial = {}
def _on_timeout(signum, frame):
    print("\n⏰ TIMEOUT — saving partial...")
    _save(_partial, partial=True)
    sys.exit(0)
_signal.signal(_signal.SIGALRM, _on_timeout)
_signal.alarm(900)

PROD_CFG = json.loads((ROOT / "configs" / "production_p91b_champion.json").read_text())
SYMBOLS = PROD_CFG["data"]["symbols"]
ENSEMBLE_WEIGHTS = PROD_CFG["ensemble"]["weights"]
SIGNAL_DEFS = PROD_CFG["ensemble"]["signals"]
SIG_KEYS = list(SIGNAL_DEFS.keys())

YEAR_RANGES = {
    "2021": ("2021-02-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
}
YEARS = sorted(YEAR_RANGES.keys())

# Candidate strategies with parameter variants
CANDIDATES = {
    "vol_rev_24_k2": {
        "strategy": "volume_reversal_alpha",
        "params": {
            "k_per_side": 2, "volume_lookback_bars": 168,
            "return_lookback_bars": 24, "vol_lookback_bars": 168,
            "target_gross_leverage": 0.30, "rebalance_interval_bars": 24,
        },
    },
    "vol_rev_48_k2": {
        "strategy": "volume_reversal_alpha",
        "params": {
            "k_per_side": 2, "volume_lookback_bars": 168,
            "return_lookback_bars": 48, "vol_lookback_bars": 168,
            "target_gross_leverage": 0.30, "rebalance_interval_bars": 24,
        },
    },
    "vol_rev_24_k3": {
        "strategy": "volume_reversal_alpha",
        "params": {
            "k_per_side": 3, "volume_lookback_bars": 168,
            "return_lookback_bars": 24, "vol_lookback_bars": 168,
            "target_gross_leverage": 0.30, "rebalance_interval_bars": 24,
        },
    },
    "vb_24_168_k2": {
        "strategy": "vol_breakout_alpha",
        "params": {
            "k_per_side": 2, "vol_short_bars": 24,
            "vol_long_bars": 168, "compression_threshold": 0.7,
            "return_lookback_bars": 12, "vol_lookback_bars": 168,
            "target_gross_leverage": 0.30, "rebalance_interval_bars": 24,
        },
    },
    "vb_48_168_k2": {
        "strategy": "vol_breakout_alpha",
        "params": {
            "k_per_side": 2, "vol_short_bars": 48,
            "vol_long_bars": 168, "compression_threshold": 0.7,
            "return_lookback_bars": 12, "vol_lookback_bars": 168,
            "target_gross_leverage": 0.30, "rebalance_interval_bars": 24,
        },
    },
    "vb_24_336_k2": {
        "strategy": "vol_breakout_alpha",
        "params": {
            "k_per_side": 2, "vol_short_bars": 24,
            "vol_long_bars": 336, "compression_threshold": 0.7,
            "return_lookback_bars": 12, "vol_lookback_bars": 168,
            "target_gross_leverage": 0.30, "rebalance_interval_bars": 24,
        },
    },
    "ll_12_k2": {
        "strategy": "lead_lag_alpha",
        "params": {
            "k_per_side": 2, "btc_lookback_bars": 12,
            "beta_lookback_bars": 168, "vol_lookback_bars": 168,
            "target_gross_leverage": 0.30, "rebalance_interval_bars": 24,
            "btc_symbol": "BTCUSDT",
        },
    },
    "ll_24_k2": {
        "strategy": "lead_lag_alpha",
        "params": {
            "k_per_side": 2, "btc_lookback_bars": 24,
            "beta_lookback_bars": 168, "vol_lookback_bars": 168,
            "target_gross_leverage": 0.30, "rebalance_interval_bars": 24,
            "btc_symbol": "BTCUSDT",
        },
    },
    "ll_8_k2": {
        "strategy": "lead_lag_alpha",
        "params": {
            "k_per_side": 2, "btc_lookback_bars": 8,
            "beta_lookback_bars": 168, "vol_lookback_bars": 168,
            "target_gross_leverage": 0.30, "rebalance_interval_bars": 24,
            "btc_symbol": "BTCUSDT",
        },
    },
}

OUT_DIR = ROOT / "artifacts" / "phase139"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sharpe(rets: np.ndarray) -> float:
    if len(rets) < 100:
        return 0.0
    mu = float(np.mean(rets)) * 8760
    sd = float(np.std(rets)) * np.sqrt(8760)
    return round(mu / sd, 4) if sd > 1e-12 else 0.0


def obj_func(yearly_sharpes: list) -> float:
    arr = np.array(yearly_sharpes)
    return round(float(np.mean(arr) - 0.5 * np.std(arr)), 4)


def _save(data: dict, partial: bool = False) -> None:
    data["partial"] = partial
    data["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path = OUT_DIR / "5th_signal_candidates_report.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → Saved: {path}")


def main():
    global _partial
    t0 = time.time()
    print("=" * 70)
    print("Phase 139: 5th Signal Candidates — VolRev, VolBreakout, LeadLag")
    print("=" * 70)

    # 1. Load data + run existing 4 signals
    print("\n[1/4] Loading data + running existing signals...")
    sig_returns = {sk: {} for sk in SIG_KEYS}
    cand_returns = {ck: {} for ck in CANDIDATES}
    datasets = {}

    for year, (start, end) in YEAR_RANGES.items():
        print(f"  {year}:", end="", flush=True)
        cfg_data = {
            "provider": "binance_rest_v1", "symbols": SYMBOLS,
            "start": start, "end": end, "bar_interval": "1h",
            "cache_dir": ".cache/binance_rest",
        }
        provider = make_provider(cfg_data, seed=42)
        dataset = provider.load()
        datasets[year] = dataset

        # Existing signals
        for sk in SIG_KEYS:
            sig_def = SIGNAL_DEFS[sk]
            cost_model = cost_model_from_config({"fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0})
            bt_cfg = BacktestConfig(costs=cost_model)
            strat = make_strategy({"name": sig_def["strategy"], "params": sig_def["params"]})
            engine = BacktestEngine(bt_cfg)
            result = engine.run(dataset, strat)
            sig_returns[sk][year] = np.array(result.returns, dtype=np.float64)

        # Candidate signals
        for ck, cdef in CANDIDATES.items():
            cost_model = cost_model_from_config({"fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0})
            bt_cfg = BacktestConfig(costs=cost_model)
            strat = make_strategy({"name": cdef["strategy"], "params": cdef["params"]})
            engine = BacktestEngine(bt_cfg)
            result = engine.run(dataset, strat)
            cand_returns[ck][year] = np.array(result.returns, dtype=np.float64)

        print(f" OK ({len(SIG_KEYS)}+{len(CANDIDATES)} signals)", flush=True)

    _partial = {"phase": 139}

    # 2. Standalone Sharpe per year for each candidate
    print("\n[2/4] Standalone Sharpe per candidate...")
    standalone = {}
    for ck in CANDIDATES:
        yearly = {}
        for year in YEARS:
            s = sharpe(cand_returns[ck][year])
            yearly[year] = s
        ys = list(yearly.values())
        avg = round(float(np.mean(ys)), 4)
        mn = round(float(np.min(ys)), 4)
        ob = obj_func(ys)
        standalone[ck] = {
            "yearly": yearly, "avg_sharpe": avg, "min_sharpe": mn, "obj": ob,
        }

    # Also compute ensemble baseline
    ens_rets = {}
    for year in YEARS:
        min_len = min(len(sig_returns[sk][year]) for sk in SIG_KEYS)
        ens = np.zeros(min_len)
        for sk in SIG_KEYS:
            ens += ENSEMBLE_WEIGHTS[sk] * sig_returns[sk][year][:min_len]
        ens_rets[year] = ens

    baseline_yearly = [sharpe(ens_rets[y]) for y in YEARS]
    baseline_obj = obj_func(baseline_yearly)

    print(f"\n  {'Candidate':18s} {'OBJ':>8s} {'AVG':>8s} {'MIN':>8s} {'2021':>6s} {'2022':>6s} {'2023':>6s} {'2024':>6s} {'2025':>6s}")
    print(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    for ck in sorted(standalone, key=lambda k: standalone[k]["obj"], reverse=True):
        r = standalone[ck]
        print(f"  {ck:18s} {r['obj']:8.4f} {r['avg_sharpe']:8.3f} {r['min_sharpe']:8.3f} "
              f"{r['yearly']['2021']:6.2f} {r['yearly']['2022']:6.2f} {r['yearly']['2023']:6.2f} "
              f"{r['yearly']['2024']:6.2f} {r['yearly']['2025']:6.2f}")

    # 3. Correlation with existing ensemble
    print("\n[3/4] Correlation with P91b ensemble returns...")
    correlations = {}
    for ck in CANDIDATES:
        corrs = []
        for year in YEARS:
            e = ens_rets[year]
            c = cand_returns[ck][year]
            min_len = min(len(e), len(c))
            if min_len > 100:
                r = float(np.corrcoef(e[:min_len], c[:min_len])[0, 1])
                corrs.append(r)
        avg_corr = round(float(np.mean(corrs)), 4) if corrs else 0.0
        correlations[ck] = {"yearly": {YEARS[i]: round(corrs[i], 4) for i in range(len(corrs))}, "avg": avg_corr}

    print(f"\n  {'Candidate':18s} {'Avg Corr':>10s} {'2021':>7s} {'2022':>7s} {'2023':>7s} {'2024':>7s} {'2025':>7s}")
    print(f"  {'-'*18} {'-'*10} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for ck in sorted(correlations, key=lambda k: correlations[k]["avg"]):
        r = correlations[ck]
        vals = [r["yearly"].get(y, 0) for y in YEARS]
        print(f"  {ck:18s} {r['avg']:10.4f} {vals[0]:7.3f} {vals[1]:7.3f} {vals[2]:7.3f} {vals[3]:7.3f} {vals[4]:7.3f}")

    # 4. Test as 5th signal in blend (grid search weight allocation)
    print("\n[4/4] Testing as 5th signal in P91b blend...")

    # Only test top candidates (positive OBJ + low correlation)
    # Score: standalone_obj × (1 - abs(correlation))
    candidate_scores = {}
    for ck in CANDIDATES:
        s_obj = standalone[ck]["obj"]
        corr = abs(correlations[ck]["avg"])
        score = s_obj * (1 - corr)
        candidate_scores[ck] = score

    top_candidates = sorted(candidate_scores, key=lambda k: candidate_scores[k], reverse=True)[:4]
    print(f"  Top candidates for blending: {top_candidates}")

    blend_results = {}
    for ck in top_candidates:
        # Test weight allocations: 5%, 10%, 15%, 20% to new signal
        # Proportionally reduce existing weights
        for new_w in [0.05, 0.10, 0.15, 0.20]:
            scale = 1.0 - new_w  # remaining weight for original 4 signals
            yearly_sharpes = []

            for year in YEARS:
                min_len = min(len(sig_returns[sk][year]) for sk in SIG_KEYS)
                min_len = min(min_len, len(cand_returns[ck][year]))
                ens5 = np.zeros(min_len)

                # Original signals (scaled down)
                for sk in SIG_KEYS:
                    ens5 += (scale * ENSEMBLE_WEIGHTS[sk]) * sig_returns[sk][year][:min_len]

                # New signal
                ens5 += new_w * cand_returns[ck][year][:min_len]

                yearly_sharpes.append(sharpe(ens5))

            avg = round(float(np.mean(yearly_sharpes)), 4)
            mn = round(float(np.min(yearly_sharpes)), 4)
            ob = obj_func(yearly_sharpes)
            delta = round(ob - baseline_obj, 4)

            key = f"{ck}_w{int(new_w*100):02d}"
            blend_results[key] = {
                "candidate": ck, "new_weight": new_w,
                "yearly": {YEARS[i]: yearly_sharpes[i] for i in range(len(YEARS))},
                "avg_sharpe": avg, "min_sharpe": mn, "obj": ob, "delta_obj": delta,
            }

    # Sort and display
    print(f"\n  {'Blend':28s} {'OBJ':>8s} {'AVG':>8s} {'MIN':>8s} {'ΔOBJ':>8s}")
    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'P91b_baseline':28s} {baseline_obj:8.4f} {float(np.mean(baseline_yearly)):8.3f} "
          f"{float(np.min(baseline_yearly)):8.3f} {0.0:+8.4f}")
    for key in sorted(blend_results, key=lambda k: blend_results[k]["obj"], reverse=True)[:15]:
        r = blend_results[key]
        tag = " ✓" if r["delta_obj"] > 0.03 else ""
        print(f"  {key:28s} {r['obj']:8.4f} {r['avg_sharpe']:8.3f} "
              f"{r['min_sharpe']:8.3f} {r['delta_obj']:+8.4f}{tag}")

    # Find best blend
    best_blend_key = max(blend_results, key=lambda k: blend_results[k]["obj"])
    best_blend = blend_results[best_blend_key]

    # Determine verdict
    if best_blend["delta_obj"] > 0.05:
        verdict = f"PROMISING — {best_blend_key} adds +{best_blend['delta_obj']:.3f} OBJ, needs LOYO validation"
    elif best_blend["delta_obj"] > 0.02:
        verdict = f"MARGINAL — {best_blend_key} adds +{best_blend['delta_obj']:.3f} OBJ"
    else:
        verdict = "NO IMPROVEMENT — none of the candidates improve the ensemble"

    # Per-category best
    category_best = {}
    for cat_prefix, cat_name in [("vol_rev", "Volume Reversal"), ("vb", "Vol Breakout"), ("ll", "Lead-Lag")]:
        cat_keys = [k for k in blend_results if blend_results[k]["candidate"].startswith(cat_prefix)]
        if cat_keys:
            best_k = max(cat_keys, key=lambda k: blend_results[k]["obj"])
            category_best[cat_name] = {
                "best_blend": best_k,
                "obj": blend_results[best_k]["obj"],
                "delta_obj": blend_results[best_k]["delta_obj"],
                "standalone_obj": standalone[blend_results[best_k]["candidate"]]["obj"],
                "avg_corr_with_ensemble": correlations[blend_results[best_k]["candidate"]]["avg"],
            }

    elapsed = time.time() - t0
    _partial = {
        "phase": 139,
        "description": "5th Signal Candidates — Volume Reversal, Vol Breakout, Lead-Lag",
        "elapsed_seconds": round(elapsed, 1),
        "baseline": {
            "yearly": {YEARS[i]: baseline_yearly[i] for i in range(len(YEARS))},
            "obj": baseline_obj,
        },
        "standalone": standalone,
        "correlations": correlations,
        "blend_results": blend_results,
        "best_blend": {"key": best_blend_key, **best_blend},
        "category_best": category_best,
        "verdict": verdict,
    }
    _save(_partial, partial=False)

    print(f"\n  Category Summary:")
    for cat, info in category_best.items():
        print(f"    {cat}: best={info['best_blend']} OBJ={info['obj']:.4f} "
              f"Δ={info['delta_obj']:+.4f} standalone_obj={info['standalone_obj']:.4f} "
              f"corr={info['avg_corr_with_ensemble']:.3f}")

    print(f"\n  VERDICT: {verdict}")
    print(f"\nPhase 139 complete in {elapsed:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
