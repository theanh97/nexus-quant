"""
Phase 146: Regime-Adaptive Parameters
=======================================
Use vol_regime to detect market stress and adapt:
  - Low vol regime (< 0.3): Higher vol target, more tilt
  - Normal regime (0.3-0.7): Standard params
  - High vol regime (> 0.7): Lower vol target, less tilt, tighter DD

Also test: trend-strength filter — only tilt when average TSMOM is strong.
"""
from __future__ import annotations

import json, logging, math, sys, hashlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("cta_p146")

from nexus_quant.projects.commodity_cta.providers.yahoo_futures import YahooFuturesProvider
from nexus_quant.projects.commodity_cta.costs.futures_fees import commodity_futures_cost_model
from nexus_quant.strategies.base import Strategy, Weights
from nexus_quant.data.schema import MarketDataset
from nexus_quant.backtest.engine import BacktestEngine, BacktestConfig


SECTOR_MAP = {
    "CL=F": "commodity", "NG=F": "commodity", "GC=F": "commodity",
    "SI=F": "commodity", "HG=F": "commodity", "ZW=F": "commodity",
    "ZC=F": "commodity", "ZS=F": "commodity",
    "EURUSD=X": "fx", "GBPUSD=X": "fx", "AUDUSD=X": "fx", "JPY=X": "fx",
    "TLT": "bond", "IEF": "bond",
}


class RegimeAdaptiveStrategy(Strategy):
    """Sector RP + Regime-Adaptive Tilt + Trend Filter + DD."""

    def __init__(self, name="regime_adaptive", params=None):
        p = params or {}
        super().__init__(name, p)

        # Sector weights
        self.w_commodity = float(p.get("w_commodity", 0.50))
        self.w_fx = float(p.get("w_fx", 0.20))
        self.w_bond = float(p.get("w_bond", 0.30))

        # Base momentum tilt
        self.base_pct = float(p.get("base_pct", 0.90))
        self.tilt_pct = float(p.get("tilt_pct", 0.10))
        self.signal_scale = float(p.get("signal_scale", 1.0))
        self.xsect_weight = float(p.get("xsect_weight", 0.05))

        # Vol target regimes
        self.vt_low = float(p.get("vt_low", 0.10))    # Vol target in calm markets
        self.vt_mid = float(p.get("vt_mid", 0.08))     # Vol target normally
        self.vt_high = float(p.get("vt_high", 0.05))   # Vol target in stress

        # Regime thresholds
        self.regime_low = float(p.get("regime_low", 0.30))
        self.regime_high = float(p.get("regime_high", 0.70))

        # Trend-strength filter
        self.trend_filter = bool(p.get("trend_filter", False))
        self.trend_min_strength = float(p.get("trend_min_strength", 0.3))

        # Bond flight-to-safety boost
        self.bond_boost_in_stress = float(p.get("bond_boost_in_stress", 0.0))

        # DD params
        self.dd_threshold = float(p.get("dd_threshold", 0.05))
        self.dd_max = float(p.get("dd_max", 0.15))
        self.dd_min_alloc = float(p.get("dd_min_alloc", 0.10))

        self.warmup = int(p.get("warmup", 270))
        self.rebalance_freq = int(p.get("rebalance_freq", 42))
        self._last_reb = -1

    def should_rebalance(self, dataset, idx):
        if idx < self.warmup: return False
        if self._last_reb < 0 or (idx - self._last_reb) >= self.rebalance_freq:
            self._last_reb = idx; return True
        return False

    def target_weights(self, dataset, idx, current):
        if idx < self.warmup: return {}

        syms = dataset.symbols
        rv = dataset.features.get("rv_20d", {})
        tsmom_126 = dataset.features.get("tsmom_126d", {})
        tsmom_252 = dataset.features.get("tsmom_252d", {})
        vol_regime = dataset.features.get("vol_regime", {})

        # Compute average vol regime
        avg_vr = _avg_vol_regime(vol_regime, syms, idx)

        # Determine regime-dependent vol target
        if avg_vr < self.regime_low:
            vt = self.vt_low
        elif avg_vr > self.regime_high:
            vt = self.vt_high
        else:
            vt = self.vt_mid

        # Sector budgets — optionally boost bonds in stress
        w_commodity = self.w_commodity
        w_fx = self.w_fx
        w_bond = self.w_bond
        if self.bond_boost_in_stress > 0 and avg_vr > self.regime_high:
            boost = self.bond_boost_in_stress
            w_bond += boost
            w_commodity -= boost * 0.7
            w_fx -= boost * 0.3

        sector_budgets = {"commodity": w_commodity, "fx": w_fx, "bond": w_bond}

        # 1. Sector-balanced inverse-vol weights
        port_vols = {}; rp_weights = {}
        for sector, budget in sector_budgets.items():
            sector_syms = [s for s in syms if SECTOR_MAP.get(s, "commodity") == sector]
            if not sector_syms: continue
            inv_vols = {}
            for sym in sector_syms:
                arr = rv.get(sym, [])
                v = arr[idx] if idx < len(arr) else 0.2
                if not (v == v) or v <= 0.01: v = 0.2
                inv_vols[sym] = 1.0 / v; port_vols[sym] = v
            total_iv = sum(inv_vols.values())
            if total_iv < 1e-10: continue
            for sym in sector_syms:
                rp_weights[sym] = budget * (inv_vols[sym] / total_iv)

        if not rp_weights: return {}

        # 2. TSMOM signals
        tsmom_signals = {}
        for sym in syms:
            s126 = _safe_get(tsmom_126, sym, idx)
            s252 = _safe_get(tsmom_252, sym, idx)
            combo = 0.5 * _clip(s126 / self.signal_scale) + 0.5 * _clip(s252 / self.signal_scale)
            tsmom_signals[sym] = combo

        # 3. Trend-strength filter — only apply tilt if avg trend is strong enough
        effective_tilt = self.tilt_pct
        if self.trend_filter:
            avg_strength = sum(abs(tsmom_signals.get(s, 0)) for s in syms) / len(syms) if syms else 0
            if avg_strength < self.trend_min_strength:
                effective_tilt = 0.0  # Flat RP when trends are weak

        # 4. XSect rank
        xsect_tilt = {}
        if self.xsect_weight > 1e-10:
            ranked = sorted(syms, key=lambda s: tsmom_signals.get(s, 0))
            n = len(ranked)
            for rank_i, sym in enumerate(ranked):
                xsect_tilt[sym] = 2.0 * rank_i / max(n - 1, 1) - 1.0

        # 5. Apply tilts
        weights = {}
        for sym in syms:
            ts_factor = self.base_pct + effective_tilt * tsmom_signals.get(sym, 0)
            xs_factor = self.xsect_weight * xsect_tilt.get(sym, 0)
            factor = max(0.0, ts_factor + xs_factor)
            weights[sym] = rp_weights.get(sym, 0) * factor

        # 6. Vol targeting
        port_vol_sq = sum((weights.get(s, 0) * port_vols.get(s, 0.2)) ** 2 for s in syms)
        port_vol = math.sqrt(port_vol_sq) if port_vol_sq > 0 else 0.01
        if port_vol > 0.01:
            scale = vt / port_vol
            weights = {s: w * min(scale, 3.0) for s, w in weights.items()}

        # 7. DD deleveraging
        dd_factor = _dd_factor(dataset, idx, self.dd_threshold, self.dd_max, self.dd_min_alloc)
        if dd_factor < 1.0:
            weights = {s: w * dd_factor for s, w in weights.items()}

        # 8. Vol ceiling
        if avg_vr > 0.8:
            weights = {s: w * max(0.5, 1.0 - (avg_vr - 0.8)) for s, w in weights.items()}

        return {s: w for s, w in weights.items() if w > 1e-10}


def _safe_get(fd, sym, idx):
    arr = fd.get(sym, [])
    v = arr[idx] if idx < len(arr) else 0
    return v if (v == v) else 0

def _clip(v):
    return max(-1.0, min(1.0, v))

def _avg_vol_regime(vol_regime, syms, idx):
    total = 0
    for sym in syms:
        arr = vol_regime.get(sym, [])
        if idx < len(arr):
            v = arr[idx]
            if v == v: total += v
    return total / len(syms) if syms else 0

def _dd_factor(dataset, idx, dd_threshold, dd_max, dd_min_alloc):
    lookback = min(60, idx)
    if lookback < 5: return 1.0
    cum_ret = 0; peak_ret = 0
    for i in range(idx - lookback, idx + 1):
        bar_ret = 0; cnt = 0
        for sym in dataset.symbols:
            p_prev = dataset.close(sym, max(0, i - 1))
            p_curr = dataset.close(sym, i)
            if p_prev > 0 and p_curr > 0:
                bar_ret += (p_curr / p_prev - 1); cnt += 1
        if cnt > 0: cum_ret += bar_ret / cnt
        if cum_ret > peak_ret: peak_ret = cum_ret
    dd = peak_ret - cum_ret
    if dd <= dd_threshold: return 1.0
    elif dd >= dd_max: return dd_min_alloc
    else:
        progress = (dd - dd_threshold) / (dd_max - dd_threshold)
        return 1.0 - progress * (1.0 - dd_min_alloc)


def compute_metrics(equity_curve, n_per_year=252):
    rets = [equity_curve[i] / equity_curve[i - 1] - 1 for i in range(1, len(equity_curve)) if equity_curve[i - 1] > 0]
    if not rets: return {"sharpe": 0, "cagr": 0, "max_drawdown": 1, "n_bars": 0}
    n = len(rets); mn = sum(rets) / n; var = sum((r - mn) ** 2 for r in rets) / n
    std = math.sqrt(var) if var > 0 else 1e-10
    sharpe = (mn / std) * math.sqrt(n_per_year)
    final = equity_curve[-1]; cagr = final ** (n_per_year / n) - 1 if final > 0 else 0
    peak = 1; mdd = 0
    for e in equity_curve:
        if e > peak: peak = e
        dd = (peak - e) / peak if peak > 0 else 0
        if dd > mdd: mdd = dd
    return {"sharpe": sharpe, "cagr": cagr, "max_drawdown": mdd, "n_bars": n}


def slice_dataset(dataset, start_date, end_date):
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    indices = [i for i, ts in enumerate(dataset.timeline) if start_ts <= ts <= end_ts]
    if not indices: raise ValueError(f"No bars in [{start_date}, {end_date}]")
    i0, i1 = indices[0], indices[-1] + 1
    return MarketDataset(
        provider=dataset.provider, timeline=dataset.timeline[i0:i1], symbols=dataset.symbols,
        perp_close={s: dataset.perp_close[s][i0:i1] for s in dataset.symbols},
        spot_close=None, funding={}, fingerprint=hashlib.md5(f"{start_date}:{end_date}".encode()).hexdigest()[:8],
        market_type=dataset.market_type,
        features={fn: {s: arr[i0:i1] for s, arr in fd.items()} for fn, fd in dataset.features.items()},
        meta={**dataset.meta, "wf_start": start_date, "wf_end": end_date},
    )


def run_test(dataset, params, cost_model, periods):
    results = {}
    for plabel, start, end in periods:
        try:
            ds_s = slice_dataset(dataset, start, end)
            strat = RegimeAdaptiveStrategy(params=params)
            result = BacktestEngine(BacktestConfig(costs=cost_model)).run(ds_s, strat, seed=42)
            results[plabel] = compute_metrics(result.equity_curve)
        except Exception as e:
            logger.warning(f"  Error: {e}")
    return results


def main():
    logger.info("=" * 80)
    logger.info("Phase 146: Regime-Adaptive Parameters")
    logger.info("=" * 80)

    DIVERSIFIED_14 = ["CL=F", "NG=F", "GC=F", "SI=F", "HG=F", "ZW=F", "ZC=F", "ZS=F",
                       "EURUSD=X", "GBPUSD=X", "AUDUSD=X", "JPY=X", "TLT", "IEF"]

    cfg = {"symbols": DIVERSIFIED_14, "start": "2005-01-01", "end": "2026-02-20",
           "cache_dir": "data/cache/yahoo_futures", "min_valid_bars": 300, "request_delay": 0.3}
    dataset = YahooFuturesProvider(cfg, seed=42).load()

    cost_model = commodity_futures_cost_model(slippage_bps=5.0, commission_bps=1.0, spread_bps=1.0)

    periods = [
        ("FULL", "2007-01-01", "2026-02-20"),
        ("IS", "2007-01-01", "2015-12-31"),
        ("OOS1", "2016-01-01", "2020-12-31"),
        ("OOS2", "2021-01-01", "2026-02-20"),
    ]

    base = {
        "w_commodity": 0.50, "w_fx": 0.20, "w_bond": 0.30,
        "base_pct": 0.90, "tilt_pct": 0.10, "signal_scale": 1.0,
        "xsect_weight": 0.05,
        "vt_low": 0.08, "vt_mid": 0.08, "vt_high": 0.08,  # Fixed (baseline)
        "trend_filter": False, "bond_boost_in_stress": 0.0,
        "dd_threshold": 0.05, "dd_max": 0.15, "dd_min_alloc": 0.10,
        "rebalance_freq": 42,
    }

    configs = [
        # Baseline (P145 champion)
        ("P145_baseline", base),

        # Regime-adaptive vol targeting
        ("RAVT_10_08_05", {**base, "vt_low": 0.10, "vt_mid": 0.08, "vt_high": 0.05}),
        ("RAVT_12_08_04", {**base, "vt_low": 0.12, "vt_mid": 0.08, "vt_high": 0.04}),
        ("RAVT_10_08_06", {**base, "vt_low": 0.10, "vt_mid": 0.08, "vt_high": 0.06}),
        ("RAVT_10_10_06", {**base, "vt_low": 0.10, "vt_mid": 0.10, "vt_high": 0.06}),

        # Trend strength filter
        ("TFILT_0.2", {**base, "trend_filter": True, "trend_min_strength": 0.2}),
        ("TFILT_0.3", {**base, "trend_filter": True, "trend_min_strength": 0.3}),
        ("TFILT_0.4", {**base, "trend_filter": True, "trend_min_strength": 0.4}),

        # Bond flight-to-safety boost
        ("BBOOST_0.10", {**base, "bond_boost_in_stress": 0.10}),
        ("BBOOST_0.15", {**base, "bond_boost_in_stress": 0.15}),
        ("BBOOST_0.20", {**base, "bond_boost_in_stress": 0.20}),

        # Combinations
        ("RAVT+TFILT", {**base, "vt_low": 0.10, "vt_mid": 0.08, "vt_high": 0.05,
                        "trend_filter": True, "trend_min_strength": 0.3}),
        ("RAVT+BBOOST", {**base, "vt_low": 0.10, "vt_mid": 0.08, "vt_high": 0.05,
                         "bond_boost_in_stress": 0.15}),
        ("ALL_regime", {**base, "vt_low": 0.10, "vt_mid": 0.08, "vt_high": 0.05,
                        "trend_filter": True, "trend_min_strength": 0.3,
                        "bond_boost_in_stress": 0.15}),

        # Higher tilt regime-adaptive
        ("HT_RAVT", {**base, "tilt_pct": 0.15, "base_pct": 0.85,
                     "vt_low": 0.10, "vt_mid": 0.08, "vt_high": 0.05}),

        # Different regime thresholds
        ("RAVT_thresh_25_65", {**base, "vt_low": 0.10, "vt_mid": 0.08, "vt_high": 0.05,
                               "regime_low": 0.25, "regime_high": 0.65}),
        ("RAVT_thresh_35_75", {**base, "vt_low": 0.10, "vt_mid": 0.08, "vt_high": 0.05,
                               "regime_low": 0.35, "regime_high": 0.75}),
    ]

    all_results = {}
    best_obj = -99; best_label = ""

    logger.info(f"\n  {'Config':25s}  {'FULL':>8s}  {'IS':>8s}  {'OOS1':>8s}  {'OOS2':>8s}  {'OOS_MIN':>8s}  {'MDD':>6s}  {'OBJ':>8s}")
    logger.info("  " + "-" * 105)

    for label, params in configs:
        r = run_test(dataset, params, cost_model, periods)
        all_results[label] = r

        s_full = r.get("FULL", {}).get("sharpe", 0)
        s_oos1 = r.get("OOS1", {}).get("sharpe", 0)
        s_oos2 = r.get("OOS2", {}).get("sharpe", 0)
        mdd = r.get("FULL", {}).get("max_drawdown", 1)
        oos_min = min(s_oos1, s_oos2)
        obj = 0.6 * oos_min + 0.4 * s_full
        star = ""
        if obj > best_obj:
            best_obj = obj; best_label = label; star = " ***"
        logger.info(f"  {label:25s}  {s_full:+8.3f}  {r.get('IS',{}).get('sharpe',0):+8.3f}  {s_oos1:+8.3f}  {s_oos2:+8.3f}  {oos_min:+8.3f}  {mdd*100:5.1f}%  {obj:+8.3f}{star}")

    logger.info(f"\n  BEST: {best_label} → OBJ={best_obj:+.3f}")

    best_r = all_results[best_label]
    for p in ["FULL", "IS", "OOS1", "OOS2"]:
        m = best_r.get(p, {})
        logger.info(f"    {p:6s}  Sharpe={m.get('sharpe',0):+.3f}  CAGR={m.get('cagr',0)*100:+.1f}%  MDD={m.get('max_drawdown',1)*100:.1f}%")

    logger.info(f"\n  vs P145: FULL=+0.614  OOS1=+0.803  OOS2=+0.749  OOS_MIN=+0.749")

    output = {
        "phase": "146", "title": "Regime-Adaptive Parameters",
        "results": {k: {p: dict(m) for p, m in r.items()} for k, r in all_results.items()},
        "best_config": best_label, "best_obj": best_obj,
    }
    out_dir = ROOT / "artifacts" / "phase146"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "regime_adaptive_report.json", "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\n  Saved: artifacts/phase146/regime_adaptive_report.json")


if __name__ == "__main__":
    main()
