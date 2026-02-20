#!/usr/bin/env python3
"""
Phase 136: Anti-Fragile Sizing / Drawdown Recovery Alpha
==========================================================
Key insight from Phases 130-135: The ensemble is ANTI-FRAGILE.
It generates MORE alpha during market stress (high vol, extreme funding,
inverted vol curve). The only validated overlay (vol, Phase 128) works
through RISK reduction, not alpha timing.

New hypothesis: Instead of reducing during drawdowns (which removes
alpha), test:
1. BOOST exposure during drawdown RECOVERY (after max DD starts narrowing)
2. INVERSE drawdown sizing: higher drawdown → INCREASE allocation
3. Running Sharpe filter: reduce when own recent Sharpe is < 0

Also test: momentum breadth as a complementary signal.
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
_signal.alarm(600)

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

OUT_DIR = ROOT / "artifacts" / "phase136"
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
    path = OUT_DIR / "antifragile_sizing_report.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → Saved: {path}")


def compute_running_drawdown(cum_rets: np.ndarray) -> np.ndarray:
    """Compute running drawdown from peak. Returns array of DD values (negative)."""
    n = len(cum_rets)
    dd = np.zeros(n)
    peak = cum_rets[0]
    for i in range(n):
        if cum_rets[i] > peak:
            peak = cum_rets[i]
        dd[i] = (cum_rets[i] / peak - 1.0) if peak > 0 else 0.0
    return dd


def compute_running_sharpe(rets: np.ndarray, window: int = 720) -> np.ndarray:
    """Compute rolling Sharpe ratio (annualized from hourly)."""
    n = len(rets)
    rs = np.full(n, np.nan)
    for i in range(window, n):
        w = rets[i - window:i]
        mu = float(np.mean(w)) * 8760
        sd = float(np.std(w)) * np.sqrt(8760)
        rs[i] = mu / sd if sd > 1e-12 else 0.0
    if window < n:
        rs[:window] = rs[window]
    return rs


def compute_momentum_breadth(sym_rets: dict, lookback: int = 168) -> np.ndarray:
    """Compute momentum breadth: fraction of symbols with positive lookback return.
    Range: 0 (all falling) to 1 (all rising)."""
    syms = list(sym_rets.keys())
    n = min(len(sym_rets[s]) for s in syms)
    breadth = np.full(n, np.nan)

    for i in range(lookback, n):
        n_positive = 0
        for s in syms:
            cum_ret = float(np.sum(sym_rets[s][i - lookback:i]))
            if cum_ret > 0:
                n_positive += 1
        breadth[i] = n_positive / len(syms)

    if lookback < n:
        breadth[:lookback] = 0.5  # neutral default
    return breadth


def main():
    global _partial
    t0 = time.time()
    print("=" * 70)
    print("Phase 136: Anti-Fragile Sizing / Drawdown Recovery Alpha")
    print("=" * 70)

    # 1. Load data
    print("\n[1/4] Loading data...")
    sig_returns = {sk: {} for sk in SIG_KEYS}
    sym_returns = {}

    for year, (start, end) in YEAR_RANGES.items():
        print(f"  {year}:", end="", flush=True)
        cfg_data = {
            "provider": "binance_rest_v1", "symbols": SYMBOLS,
            "start": start, "end": end, "bar_interval": "1h",
            "cache_dir": ".cache/binance_rest",
        }
        provider = make_provider(cfg_data, seed=42)
        dataset = provider.load()

        sym_returns[year] = {}
        for sym in SYMBOLS:
            rets = []
            for i in range(1, len(dataset.timeline)):
                c0 = dataset.close(sym, i - 1)
                c1 = dataset.close(sym, i)
                rets.append((c1 / c0 - 1.0) if c0 > 0 else 0.0)
            sym_returns[year][sym] = np.array(rets, dtype=np.float64)

        for sk in SIG_KEYS:
            sig_def = SIGNAL_DEFS[sk]
            cost_model = cost_model_from_config({"fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0})
            bt_cfg = BacktestConfig(costs=cost_model)
            strat = make_strategy({"name": sig_def["strategy"], "params": sig_def["params"]})
            engine = BacktestEngine(bt_cfg)
            result = engine.run(dataset, strat)
            sig_returns[sk][year] = np.array(result.returns, dtype=np.float64)

        print(f" OK", flush=True)

    _partial = {"phase": 136}

    # 2. Compute ensemble returns + running metrics
    print("\n[2/4] Computing ensemble returns + drawdown/Sharpe metrics...")
    ens_returns = {}
    ens_dd = {}
    ens_rs = {}
    mom_breadth = {}

    for year in YEARS:
        min_len = min(len(sig_returns[sk][year]) for sk in SIG_KEYS)
        ens = np.zeros(int(min_len))
        for sk in SIG_KEYS:
            ens += ENSEMBLE_WEIGHTS[sk] * sig_returns[sk][year][:int(min_len)]
        ens_returns[year] = ens

        # Cumulative equity for drawdown
        cum = np.cumprod(1.0 + ens)
        dd = compute_running_drawdown(cum)
        ens_dd[year] = dd

        rs = compute_running_sharpe(ens, window=720)
        ens_rs[year] = rs

        mb = compute_momentum_breadth(sym_returns[year], lookback=168)[:int(min_len)]
        mom_breadth[year] = mb

        print(f"  {year}: avg_dd={float(np.mean(dd))*100:.1f}%  "
              f"max_dd={float(np.min(dd))*100:.1f}%  "
              f"avg_rs={float(np.nanmean(rs)):.2f}  "
              f"avg_breadth={float(np.nanmean(mb)):.2f}")

    # 3. Regime analysis
    print("\n[3/4] Regime analysis...")

    for year in YEARS:
        ens = ens_returns[year]
        dd = ens_dd[year]
        rs = ens_rs[year]

        # Split by drawdown level
        deep_dd = dd < -0.03  # >3% drawdown
        normal_dd = dd >= -0.03

        s_deep = sharpe(ens[deep_dd]) if np.sum(deep_dd) > 100 else 0.0
        s_normal = sharpe(ens[normal_dd]) if np.sum(normal_dd) > 100 else 0.0

        # Split by running Sharpe
        valid_rs = ~np.isnan(rs)
        pos_rs = valid_rs & (rs > 0)
        neg_rs = valid_rs & (rs <= 0)

        s_pos = sharpe(ens[pos_rs]) if np.sum(pos_rs) > 100 else 0.0
        s_neg = sharpe(ens[neg_rs]) if np.sum(neg_rs) > 100 else 0.0

        # DD recovery: was in DD last period, now recovering
        recovering = np.zeros(len(dd), dtype=bool)
        for i in range(1, len(dd)):
            recovering[i] = (dd[i - 1] < -0.02) and (dd[i] > dd[i - 1])

        s_recov = sharpe(ens[recovering]) if np.sum(recovering) > 100 else 0.0

        print(f"  {year}: deep_dd={s_deep:+.2f}  normal={s_normal:+.2f}  "
              f"pos_rs={s_pos:+.2f}  neg_rs={s_neg:+.2f}  "
              f"recovering={s_recov:+.2f}")

    # 4. Test sizing overlays
    print("\n[4/4] Testing anti-fragile sizing variants...")

    overlay_configs = {
        "baseline": {"mode": "none"},
        # Drawdown-based
        "boost_deep_dd_130": {"mode": "dd_boost", "dd_thresh": -0.03, "scale": 1.3},
        "boost_deep_dd_150": {"mode": "dd_boost", "dd_thresh": -0.03, "scale": 1.5},
        "boost_recovery": {"mode": "recovery", "dd_thresh": -0.02, "scale": 1.3},
        # Running Sharpe filter
        "reduce_neg_sharpe_50": {"mode": "rs_filter", "threshold": 0, "scale": 0.5},
        "reduce_neg_sharpe_70": {"mode": "rs_filter", "threshold": 0, "scale": 0.7},
        "reduce_low_sharpe_50": {"mode": "rs_filter", "threshold": 0.5, "scale": 0.5},
        # Momentum breadth
        "reduce_extreme_breadth": {"mode": "breadth", "low": 0.2, "high": 0.8, "scale": 0.5},
        "boost_mixed_breadth": {"mode": "breadth_inv", "low": 0.3, "high": 0.7, "scale": 1.3},
    }

    overlay_results = {}
    for oname, ocfg in overlay_configs.items():
        yearly_sharpes = []
        yearly_detail = {}

        for year in YEARS:
            ens = ens_returns[year].copy()
            n = len(ens)

            if ocfg["mode"] == "none":
                pass  # unchanged
            elif ocfg["mode"] == "dd_boost":
                dd = ens_dd[year]
                for i in range(n):
                    if dd[i] < ocfg["dd_thresh"]:
                        ens[i] *= ocfg["scale"]
            elif ocfg["mode"] == "recovery":
                dd = ens_dd[year]
                for i in range(1, n):
                    if dd[i - 1] < ocfg["dd_thresh"] and dd[i] > dd[i - 1]:
                        ens[i] *= ocfg["scale"]
            elif ocfg["mode"] == "rs_filter":
                rs = ens_rs[year]
                for i in range(n):
                    if not np.isnan(rs[i]) and rs[i] < ocfg["threshold"]:
                        ens[i] *= ocfg["scale"]
            elif ocfg["mode"] == "breadth":
                mb = mom_breadth[year]
                for i in range(min(n, len(mb))):
                    if not np.isnan(mb[i]):
                        if mb[i] < ocfg["low"] or mb[i] > ocfg["high"]:
                            ens[i] *= ocfg["scale"]
            elif ocfg["mode"] == "breadth_inv":
                mb = mom_breadth[year]
                for i in range(min(n, len(mb))):
                    if not np.isnan(mb[i]):
                        if ocfg["low"] <= mb[i] <= ocfg["high"]:
                            ens[i] *= ocfg["scale"]

            s = sharpe(ens)
            yearly_sharpes.append(s)
            yearly_detail[year] = s

        avg = round(float(np.mean(yearly_sharpes)), 4)
        mn = round(float(np.min(yearly_sharpes)), 4)
        obj = obj_func(yearly_sharpes)
        overlay_results[oname] = {
            "yearly": yearly_detail,
            "avg_sharpe": avg,
            "min_sharpe": mn,
            "obj": obj,
        }

    baseline_obj = overlay_results["baseline"]["obj"]
    print(f"\n  {'Variant':28s} {'OBJ':>8s} {'AVG':>8s} {'MIN':>8s} {'ΔOBJ':>8s}")
    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for oname in sorted(overlay_results, key=lambda k: overlay_results[k]["obj"], reverse=True):
        r = overlay_results[oname]
        delta = r["obj"] - baseline_obj
        tag = " ✓" if delta > 0.05 else ""
        print(f"  {oname:28s} {r['obj']:8.4f} {r['avg_sharpe']:8.3f} "
              f"{r['min_sharpe']:8.3f} {delta:+8.4f}{tag}")

    best_name = max(overlay_results, key=lambda k: overlay_results[k]["obj"])
    best = overlay_results[best_name]
    improvement = best["obj"] - baseline_obj

    if improvement < 0.03:
        verdict = "NO IMPROVEMENT — anti-fragile sizing does not help"
    elif improvement < 0.10:
        verdict = f"MARGINAL — {best_name} +{improvement:.3f} OBJ"
    else:
        verdict = f"POTENTIAL — {best_name} adds +{improvement:.3f} OBJ"

    elapsed = time.time() - t0
    _partial = {
        "phase": 136,
        "description": "Anti-Fragile Sizing / Drawdown Recovery Alpha",
        "elapsed_seconds": round(elapsed, 1),
        "overlays": overlay_results,
        "best": {"name": best_name, **best},
        "improvement": round(improvement, 4),
        "verdict": verdict,
    }
    _save(_partial, partial=False)

    print(f"\n  VERDICT: {verdict}")
    print(f"\nPhase 136 complete in {elapsed:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
