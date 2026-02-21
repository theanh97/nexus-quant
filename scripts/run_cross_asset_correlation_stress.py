#!/usr/bin/env python3
"""
Multi-Asset Correlation Stress Dynamics — R39
===============================================

R20: VRP-Skew correlation drops during stress (0.01 vs 0.04) — ideal diversification.
R31: BTC+ETH multi-asset gives avg=3.174 (+0.989 vs BTC-only).

Questions:
  1. How correlated are BTC and ETH VRP returns?
  2. Does BTC-ETH correlation change during stress?
  3. Can correlation-based rebalancing improve the ensemble?
  4. What is the correlation between BTC IV and ETH IV regimes?
  5. Cross-asset tail risk: does one asset's stress spill to the other?

Analysis:
  A. BTC-ETH return correlation (full sample, stress, calm)
  B. BTC-ETH IV regime correlation (do both assets enter high-IV together?)
  C. BTC-ETH VRP spread correlation
  D. Cross-asset stress transmission (conditional drawdowns)
  E. Correlation-aware rebalancing: reduce allocation when correlation spikes
"""
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nexus_quant.backtest.costs import ExecutionCostModel, FeeModel, ImpactModel
from nexus_quant.projects.crypto_options.options_engine import compute_metrics
from nexus_quant.projects.crypto_options.providers.deribit_rest import DeribitRestProvider
from nexus_quant.projects.crypto_options.strategies.variance_premium import VariancePremiumStrategy
from nexus_quant.projects.crypto_options.strategies.skew_trade_v2 import SkewTradeV2Strategy

fees = FeeModel(maker_fee_rate=0.0003, taker_fee_rate=0.0005)
impact = ImpactModel(model="sqrt", coef_bps=2.0)
COSTS = ExecutionCostModel(fee=fees, impact=impact)

YEARS = [2021, 2022, 2023, 2024, 2025]
SEED = 42
BARS_PER_YEAR = 365
W_VRP = 0.40
W_SKEW = 0.60

BTC_VRP = {"base_leverage": 1.5, "exit_z_threshold": -3.0, "vrp_lookback": 30, "rebalance_freq": 5, "min_bars": 30}
BTC_SKEW = {"skew_lookback": 60, "z_entry": 2.0, "z_exit": 0.0, "target_leverage": 1.0, "rebalance_freq": 5, "min_bars": 60}
ETH_VRP = {"base_leverage": 1.5, "exit_z_threshold": -3.0, "vrp_lookback": 60, "rebalance_freq": 5, "min_bars": 60}
ETH_SKEW = {"skew_lookback": 90, "z_entry": 2.0, "z_exit": 0.0, "target_leverage": 1.0, "rebalance_freq": 5, "min_bars": 90}


def pearson_corr(x, y):
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n < 5:
        return None
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / n
    sx = (sum((xi - mx) ** 2 for xi in x) / n) ** 0.5
    sy = (sum((yi - my) ** 2 for yi in y) / n) ** 0.5
    if sx < 1e-10 or sy < 1e-10:
        return None
    return cov / (sx * sy)


def ewma_corr(x, y, halflife=20):
    """Compute EWMA correlation time series."""
    alpha = 1 - math.exp(-math.log(2) / halflife)
    n = len(x)
    if n < 5:
        return []

    mx = x[0]
    my = y[0]
    vx = 0.0
    vy = 0.0
    cov = 0.0
    corrs = []

    for i in range(n):
        mx = alpha * x[i] + (1 - alpha) * mx
        my = alpha * y[i] + (1 - alpha) * my
        dx = x[i] - mx
        dy = y[i] - my
        vx = alpha * dx ** 2 + (1 - alpha) * vx
        vy = alpha * dy ** 2 + (1 - alpha) * vy
        cov = alpha * dx * dy + (1 - alpha) * cov

        sx = max(vx, 1e-10) ** 0.5
        sy = max(vy, 1e-10) ** 0.5
        c = cov / (sx * sy) if sx > 1e-6 and sy > 1e-6 else 0.0
        c = max(-1.0, min(1.0, c))
        corrs.append(c)

    return corrs


def iv_percentile(iv_series, idx, lookback=180):
    if idx < lookback or not iv_series:
        return None
    start = max(0, idx - lookback)
    window = [v for v in iv_series[start:idx] if v is not None]
    if len(window) < 10:
        return None
    current = iv_series[idx] if idx < len(iv_series) and iv_series[idx] is not None else None
    if current is None:
        return None
    return sum(1 for v in window if v < current) / len(window)


def iv_sizing_scale(pct):
    if pct < 0.25:
        return 0.50
    elif pct > 0.75:
        return 1.70
    return 1.0


def collect_returns_and_features():
    """Collect daily returns and features for BTC and ETH across all years."""
    btc_rets = []
    eth_rets = []
    btc_ivs = []
    eth_ivs = []
    btc_skews = []
    eth_skews = []
    btc_ensemble_rets = []
    eth_ensemble_rets = []

    for yr in YEARS:
        cfg = {
            "symbols": ["BTC", "ETH"],
            "start": f"{yr}-01-01",
            "end": f"{yr}-12-31",
            "bar_interval": "1d",
            "use_synthetic_iv": True,
        }
        provider = DeribitRestProvider(cfg, seed=SEED)
        dataset = provider.load()
        n = len(dataset.timeline)

        # Strategy instances
        strats = {
            "BTC": {
                "vrp": VariancePremiumStrategy(params=BTC_VRP),
                "skew": SkewTradeV2Strategy(params=BTC_SKEW),
            },
            "ETH": {
                "vrp": VariancePremiumStrategy(params=ETH_VRP),
                "skew": SkewTradeV2Strategy(params=ETH_SKEW),
            },
        }

        dt = 1.0 / BARS_PER_YEAR
        vrp_w = {"BTC": 0.0, "ETH": 0.0}
        skew_w = {"BTC": 0.0, "ETH": 0.0}

        for idx in range(1, n):
            for sym in ["BTC", "ETH"]:
                # Price return
                closes = dataset.perp_close.get(sym, [])
                if idx < len(closes) and closes[idx - 1] > 0 and closes[idx] > 0:
                    price_ret = math.log(closes[idx] / closes[idx - 1])
                else:
                    price_ret = 0.0

                if sym == "BTC":
                    btc_rets.append(price_ret)
                else:
                    eth_rets.append(price_ret)

                # IV
                iv_series = dataset.features.get("iv_atm", {}).get(sym, [])
                iv_val = iv_series[idx] if idx < len(iv_series) else None
                if sym == "BTC":
                    btc_ivs.append(iv_val)
                else:
                    eth_ivs.append(iv_val)

                # Skew
                skew_series = dataset.features.get("skew_25d", {}).get(sym, [])
                skew_val = skew_series[idx] if idx < len(skew_series) else None
                if sym == "BTC":
                    btc_skews.append(skew_val)
                else:
                    eth_skews.append(skew_val)

                # Ensemble return per asset
                w_v = vrp_w.get(sym, 0.0)
                vpnl = 0.0
                if abs(w_v) > 1e-10:
                    rv_bar = abs(price_ret) * math.sqrt(BARS_PER_YEAR)
                    iv = iv_series[idx - 1] if iv_series and idx - 1 < len(iv_series) else None
                    if iv and iv > 0:
                        vrp = 0.5 * (iv ** 2 - rv_bar ** 2) * dt
                        vpnl = (-w_v) * vrp

                w_s = skew_w.get(sym, 0.0)
                spnl = 0.0
                if abs(w_s) > 1e-10:
                    sks = dataset.features.get("skew_25d", {}).get(sym, [])
                    if idx < len(sks) and idx - 1 < len(sks):
                        sn, sp = sks[idx], sks[idx - 1]
                        if sn is not None and sp is not None:
                            ds = float(sn) - float(sp)
                            iv_s = iv_series[idx - 1] if iv_series and idx - 1 < len(iv_series) else 0.70
                            if iv_s and iv_s > 0:
                                spnl = w_s * ds * iv_s * math.sqrt(dt) * 2.5

                ens_ret = W_VRP * vpnl + W_SKEW * spnl
                if sym == "BTC":
                    btc_ensemble_rets.append(ens_ret)
                else:
                    eth_ensemble_rets.append(ens_ret)

            # Rebalance
            for sym in ["BTC", "ETH"]:
                if strats[sym]["vrp"].should_rebalance(dataset, idx):
                    tv = strats[sym]["vrp"].target_weights(dataset, idx, {sym: vrp_w.get(sym, 0.0)})
                    vrp_w[sym] = tv.get(sym, 0.0)
                if strats[sym]["skew"].should_rebalance(dataset, idx):
                    ts = strats[sym]["skew"].target_weights(dataset, idx, {sym: skew_w.get(sym, 0.0)})
                    skew_w[sym] = ts.get(sym, 0.0)
                iv_s = dataset.features.get("iv_atm", {}).get(sym, [])
                pct = iv_percentile(iv_s, idx)
                if pct is not None:
                    sc = iv_sizing_scale(pct)
                    vrp_w[sym] *= sc
                    skew_w[sym] *= sc

    return {
        "btc_rets": btc_rets, "eth_rets": eth_rets,
        "btc_ivs": btc_ivs, "eth_ivs": eth_ivs,
        "btc_skews": btc_skews, "eth_skews": eth_skews,
        "btc_ensemble_rets": btc_ensemble_rets,
        "eth_ensemble_rets": eth_ensemble_rets,
    }


def run_corr_aware_ensemble(corr_threshold, reduce_factor, halflife=20):
    """Run multi-asset ensemble that reduces allocation when BTC-ETH corr > threshold."""
    sharpes = []
    yearly_detail = {}

    for yr in YEARS:
        cfg = {
            "symbols": ["BTC", "ETH"],
            "start": f"{yr}-01-01", "end": f"{yr}-12-31",
            "bar_interval": "1d", "use_synthetic_iv": True,
        }
        provider = DeribitRestProvider(cfg, seed=SEED)
        dataset = provider.load()
        n = len(dataset.timeline)

        strats = {
            "BTC": {
                "vrp": VariancePremiumStrategy(params=BTC_VRP),
                "skew": SkewTradeV2Strategy(params=BTC_SKEW),
            },
            "ETH": {
                "vrp": VariancePremiumStrategy(params=ETH_VRP),
                "skew": SkewTradeV2Strategy(params=ETH_SKEW),
            },
        }

        dt = 1.0 / BARS_PER_YEAR
        equity = 1.0
        vrp_w = {"BTC": 0.0, "ETH": 0.0}
        skew_w = {"BTC": 0.0, "ETH": 0.0}
        equity_curve = [1.0]
        returns_list = []

        # EWMA correlation state
        btc_ret_buf = []
        eth_ret_buf = []
        corr_ewma = 0.0
        alpha = 1 - math.exp(-math.log(2) / halflife)
        mx_btc = 0.0
        mx_eth = 0.0
        vx_btc = 0.001
        vx_eth = 0.001
        cov_be = 0.0

        for idx in range(1, n):
            prev_equity = equity

            # Compute daily returns for correlation tracking
            btc_closes = dataset.perp_close.get("BTC", [])
            eth_closes = dataset.perp_close.get("ETH", [])
            btc_ret = 0.0
            eth_ret = 0.0
            if idx < len(btc_closes) and btc_closes[idx - 1] > 0:
                btc_ret = math.log(btc_closes[idx] / btc_closes[idx - 1])
            if idx < len(eth_closes) and eth_closes[idx - 1] > 0:
                eth_ret = math.log(eth_closes[idx] / eth_closes[idx - 1])

            # Update EWMA correlation
            mx_btc = alpha * btc_ret + (1 - alpha) * mx_btc
            mx_eth = alpha * eth_ret + (1 - alpha) * mx_eth
            dx = btc_ret - mx_btc
            dy = eth_ret - mx_eth
            vx_btc = alpha * dx ** 2 + (1 - alpha) * vx_btc
            vx_eth = alpha * dy ** 2 + (1 - alpha) * vx_eth
            cov_be = alpha * dx * dy + (1 - alpha) * cov_be
            sx = max(vx_btc, 1e-10) ** 0.5
            sy = max(vx_eth, 1e-10) ** 0.5
            corr_ewma = cov_be / (sx * sy) if sx > 1e-6 and sy > 1e-6 else 0.0
            corr_ewma = max(-1.0, min(1.0, corr_ewma))

            # Dynamic allocation based on correlation
            w_btc = 0.50
            w_eth = 0.50
            if idx > 60 and corr_ewma > corr_threshold:
                # High correlation → reduce total exposure
                scale = 1.0 - (corr_ewma - corr_threshold) * reduce_factor
                scale = max(0.30, min(1.0, scale))
                w_btc *= scale
                w_eth *= scale

            asset_weights = {"BTC": w_btc, "ETH": w_eth}

            # P&L
            total_pnl = 0.0
            for sym in ["BTC", "ETH"]:
                aw = asset_weights[sym]
                w_v = vrp_w.get(sym, 0.0)
                vpnl = 0.0
                if abs(w_v) > 1e-10:
                    closes = dataset.perp_close.get(sym, [])
                    if idx < len(closes) and closes[idx - 1] > 0 and closes[idx] > 0:
                        lr = math.log(closes[idx] / closes[idx - 1])
                        rv = abs(lr) * math.sqrt(BARS_PER_YEAR)
                    else:
                        rv = 0.0
                    ivs = dataset.features.get("iv_atm", {}).get(sym, [])
                    iv = ivs[idx - 1] if ivs and idx - 1 < len(ivs) else None
                    if iv and iv > 0:
                        vpnl = (-w_v) * 0.5 * (iv ** 2 - rv ** 2) * dt

                w_s = skew_w.get(sym, 0.0)
                spnl = 0.0
                if abs(w_s) > 1e-10:
                    sks = dataset.features.get("skew_25d", {}).get(sym, [])
                    if idx < len(sks) and idx - 1 < len(sks):
                        sn, sp = sks[idx], sks[idx - 1]
                        if sn is not None and sp is not None:
                            ds = float(sn) - float(sp)
                            ivs = dataset.features.get("iv_atm", {}).get(sym, [])
                            iv_s = ivs[idx - 1] if ivs and idx - 1 < len(ivs) else 0.70
                            if iv_s and iv_s > 0:
                                spnl = w_s * ds * iv_s * math.sqrt(dt) * 2.5

                total_pnl += aw * (W_VRP * vpnl + W_SKEW * spnl)

            equity += equity * total_pnl

            # Rebalance
            for sym in ["BTC", "ETH"]:
                if strats[sym]["vrp"].should_rebalance(dataset, idx):
                    tv = strats[sym]["vrp"].target_weights(dataset, idx, {sym: vrp_w.get(sym, 0.0)})
                    vrp_w[sym] = tv.get(sym, 0.0)
                if strats[sym]["skew"].should_rebalance(dataset, idx):
                    ts = strats[sym]["skew"].target_weights(dataset, idx, {sym: skew_w.get(sym, 0.0)})
                    skew_w[sym] = ts.get(sym, 0.0)
                ivs = dataset.features.get("iv_atm", {}).get(sym, [])
                pct = iv_percentile(ivs, idx)
                if pct is not None:
                    sc = iv_sizing_scale(pct)
                    vrp_w[sym] *= sc
                    skew_w[sym] *= sc

            equity_curve.append(equity)
            bar_ret = (equity / prev_equity) - 1.0 if prev_equity > 0 else 0.0
            returns_list.append(bar_ret)

        m = compute_metrics(equity_curve, returns_list, BARS_PER_YEAR)
        sharpes.append(m["sharpe"])
        yearly_detail[str(yr)] = round(m["sharpe"], 3)

    avg = sum(sharpes) / len(sharpes)
    mn = min(sharpes)
    return {"avg_sharpe": round(avg, 3), "min_sharpe": round(mn, 3), "yearly": yearly_detail}


def main():
    print("=" * 70)
    print("MULTI-ASSET CORRELATION STRESS DYNAMICS — R39")
    print("=" * 70)
    print()

    # ── Collect data ──────────────────────────────────────────────────────
    print("Collecting returns and features...")
    data = collect_returns_and_features()
    n = len(data["btc_rets"])
    print(f"  Total data points: {n}")
    print()

    results = {}

    # ── A. Price Return Correlation ───────────────────────────────────────
    print("--- A. BTC-ETH PRICE RETURN CORRELATION ---")
    full_corr = pearson_corr(data["btc_rets"], data["eth_rets"])
    print(f"  Full sample: r = {full_corr:.4f}")

    # Split into quantiles by BTC return magnitude
    abs_btc = sorted(enumerate(data["btc_rets"]), key=lambda x: abs(x[1]))
    n_q = n // 4
    calm_idx = [i for i, _ in abs_btc[:n_q]]
    normal_idx = [i for i, _ in abs_btc[n_q:3*n_q]]
    stress_idx = [i for i, _ in abs_btc[3*n_q:]]

    calm_corr = pearson_corr([data["btc_rets"][i] for i in calm_idx],
                              [data["eth_rets"][i] for i in calm_idx])
    normal_corr = pearson_corr([data["btc_rets"][i] for i in normal_idx],
                                [data["eth_rets"][i] for i in normal_idx])
    stress_corr = pearson_corr([data["btc_rets"][i] for i in stress_idx],
                                [data["eth_rets"][i] for i in stress_idx])

    print(f"  Calm (Q1 abs BTC ret):    r = {calm_corr:.4f}" if calm_corr else "  Calm: N/A")
    print(f"  Normal (Q2-Q3):            r = {normal_corr:.4f}" if normal_corr else "  Normal: N/A")
    print(f"  Stress (Q4 abs BTC ret):  r = {stress_corr:.4f}" if stress_corr else "  Stress: N/A")

    results["price_return_corr"] = {
        "full": round(full_corr, 4) if full_corr else None,
        "calm": round(calm_corr, 4) if calm_corr else None,
        "normal": round(normal_corr, 4) if normal_corr else None,
        "stress": round(stress_corr, 4) if stress_corr else None,
    }
    print()

    # ── B. Ensemble Return Correlation ────────────────────────────────────
    print("--- B. BTC-ETH ENSEMBLE RETURN CORRELATION ---")
    ens_corr = pearson_corr(data["btc_ensemble_rets"], data["eth_ensemble_rets"])
    print(f"  Full sample: r = {ens_corr:.4f}" if ens_corr else "  N/A")

    abs_btc_ens = sorted(enumerate(data["btc_ensemble_rets"]), key=lambda x: abs(x[1]))
    calm_ens_idx = [i for i, _ in abs_btc_ens[:n_q]]
    stress_ens_idx = [i for i, _ in abs_btc_ens[3*n_q:]]

    calm_ens_corr = pearson_corr([data["btc_ensemble_rets"][i] for i in calm_ens_idx],
                                  [data["eth_ensemble_rets"][i] for i in calm_ens_idx])
    stress_ens_corr = pearson_corr([data["btc_ensemble_rets"][i] for i in stress_ens_idx],
                                    [data["eth_ensemble_rets"][i] for i in stress_ens_idx])

    print(f"  Calm (Q1):   r = {calm_ens_corr:.4f}" if calm_ens_corr else "  Calm: N/A")
    print(f"  Stress (Q4): r = {stress_ens_corr:.4f}" if stress_ens_corr else "  Stress: N/A")

    results["ensemble_return_corr"] = {
        "full": round(ens_corr, 4) if ens_corr else None,
        "calm": round(calm_ens_corr, 4) if calm_ens_corr else None,
        "stress": round(stress_ens_corr, 4) if stress_ens_corr else None,
    }
    print()

    # ── C. IV Regime Correlation ──────────────────────────────────────────
    print("--- C. BTC-ETH IV REGIME CORRELATION ---")
    btc_ivs_clean = [v for v in data["btc_ivs"] if v is not None]
    eth_ivs_clean = [v for v in data["eth_ivs"] if v is not None]
    min_len = min(len(btc_ivs_clean), len(eth_ivs_clean))
    if min_len > 5:
        iv_corr = pearson_corr(btc_ivs_clean[:min_len], eth_ivs_clean[:min_len])
        print(f"  IV level correlation: r = {iv_corr:.4f}" if iv_corr else "  N/A")

        # IV change correlation
        btc_iv_chg = [btc_ivs_clean[i] - btc_ivs_clean[i - 1] for i in range(1, min_len)]
        eth_iv_chg = [eth_ivs_clean[i] - eth_ivs_clean[i - 1] for i in range(1, min_len)]
        iv_chg_corr = pearson_corr(btc_iv_chg, eth_iv_chg)
        print(f"  IV change correlation: r = {iv_chg_corr:.4f}" if iv_chg_corr else "  N/A")

        results["iv_regime_corr"] = {
            "iv_level": round(iv_corr, 4) if iv_corr else None,
            "iv_change": round(iv_chg_corr, 4) if iv_chg_corr else None,
        }
    print()

    # ── D. EWMA Correlation Time Series ───────────────────────────────────
    print("--- D. EWMA CORRELATION DYNAMICS ---")
    ewma_corrs = ewma_corr(data["btc_rets"], data["eth_rets"], halflife=20)
    if ewma_corrs:
        avg_corr = sum(ewma_corrs[60:]) / len(ewma_corrs[60:])
        max_corr = max(ewma_corrs[60:])
        min_corr = min(ewma_corrs[60:])
        q25 = sorted(ewma_corrs[60:])[len(ewma_corrs[60:]) // 4]
        q75 = sorted(ewma_corrs[60:])[3 * len(ewma_corrs[60:]) // 4]
        print(f"  EWMA(hl=20) avg: {avg_corr:.4f}")
        print(f"  Range: [{min_corr:.4f}, {max_corr:.4f}]")
        print(f"  IQR: [{q25:.4f}, {q75:.4f}]")
        print(f"  High corr (>0.80) fraction: {sum(1 for c in ewma_corrs[60:] if c > 0.80) / len(ewma_corrs[60:]):.1%}")
        print(f"  Low corr (<0.20) fraction:  {sum(1 for c in ewma_corrs[60:] if c < 0.20) / len(ewma_corrs[60:]):.1%}")

        results["ewma_corr"] = {
            "avg": round(avg_corr, 4),
            "max": round(max_corr, 4),
            "min": round(min_corr, 4),
            "q25": round(q25, 4),
            "q75": round(q75, 4),
        }
    print()

    # ── E. Correlation-Aware Rebalancing ──────────────────────────────────
    print("--- E. CORRELATION-AWARE REBALANCING ---")
    print("  (Reduce exposure when BTC-ETH corr > threshold)")

    baseline = run_corr_aware_ensemble(corr_threshold=2.0, reduce_factor=0.0)  # no reduction
    print(f"  Baseline (no corr adjustment): avg={baseline['avg_sharpe']:.3f} min={baseline['min_sharpe']:.3f}")

    corr_configs = [
        (0.50, 0.50, "corr>0.50, reduce 50%"),
        (0.50, 1.00, "corr>0.50, reduce 100%"),
        (0.70, 0.50, "corr>0.70, reduce 50%"),
        (0.70, 1.00, "corr>0.70, reduce 100%"),
        (0.80, 0.50, "corr>0.80, reduce 50%"),
        (0.80, 1.00, "corr>0.80, reduce 100%"),
    ]

    corr_results = []
    for thresh, factor, label in corr_configs:
        r = run_corr_aware_ensemble(thresh, factor)
        d = r["avg_sharpe"] - baseline["avg_sharpe"]
        corr_results.append({"label": label, "threshold": thresh, "factor": factor, **r, "delta": round(d, 3)})
        print(f"  {label:30s} avg={r['avg_sharpe']:.3f} min={r['min_sharpe']:.3f} Δ={d:+.3f}")

    results["corr_aware"] = corr_results
    print()

    # ── SUMMARY ───────────────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY — R39: Multi-Asset Correlation Stress Dynamics")
    print("=" * 70)
    print()

    print("  CORRELATION STRUCTURE:")
    print(f"    Price returns:      full={results.get('price_return_corr', {}).get('full', 'N/A')}")
    print(f"                        calm={results.get('price_return_corr', {}).get('calm', 'N/A')}")
    print(f"                        stress={results.get('price_return_corr', {}).get('stress', 'N/A')}")
    print(f"    Ensemble returns:   full={results.get('ensemble_return_corr', {}).get('full', 'N/A')}")
    print(f"                        stress={results.get('ensemble_return_corr', {}).get('stress', 'N/A')}")
    print()

    if results.get("ewma_corr"):
        ec = results["ewma_corr"]
        print(f"  EWMA DYNAMICS:")
        print(f"    Average: {ec['avg']:.4f}  Range: [{ec['min']:.4f}, {ec['max']:.4f}]")
    print()

    # Verdict on correlation-aware rebalancing
    best_corr = max(corr_results, key=lambda x: x["avg_sharpe"]) if corr_results else None
    if best_corr:
        d = best_corr["avg_sharpe"] - baseline["avg_sharpe"]
        print(f"  CORRELATION-AWARE REBALANCING:")
        print(f"    Best: {best_corr['label']} (Δ={d:+.3f})")

    print()
    print("=" * 70)
    if best_corr and best_corr["delta"] > 0.05:
        print("VERDICT: Correlation-aware rebalancing HELPS")
    elif best_corr and best_corr["delta"] > -0.03:
        print("VERDICT: Correlation-aware rebalancing is MARGINAL — not worth complexity")
    else:
        print("VERDICT: Correlation-aware rebalancing HURTS — carry is always-on")
    print("=" * 70)

    # Save
    out = ROOT / "artifacts" / "crypto_options" / "cross_asset_correlation_stress.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "research_id": "R39",
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
