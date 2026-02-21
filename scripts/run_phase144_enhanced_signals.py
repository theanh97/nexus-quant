"""
Phase 144: Enhanced Signals — Cross-Sectional Mom + Multi-TF TSMOM + Value Overlay
====================================================================================
Build on Phase 143 champion (OOS_MIN=+0.659) by adding:
  1. Cross-sectional momentum rank tilt (overweight leaders, underweight laggards)
  2. Multi-timeframe TSMOM (add 63d signal to 126d/252d)
  3. Value/mean-reversion overlay (reduce overbought, increase oversold)
  4. Adaptive vol targeting (higher in calm, lower in stress)
"""
from __future__ import annotations

import json, logging, math, sys, hashlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("cta_p144")

from nexus_quant.projects.commodity_cta.providers.yahoo_futures import YahooFuturesProvider
from nexus_quant.projects.commodity_cta.costs.futures_fees import commodity_futures_cost_model
from nexus_quant.strategies.base import Strategy, Weights
from nexus_quant.data.schema import MarketDataset
from nexus_quant.backtest.engine import BacktestEngine, BacktestConfig


class RPMomDDV2Strategy(Strategy):
    """Risk-Parity + Enhanced Signals + Drawdown Deleveraging (v2)."""

    def __init__(self, name="rp_mom_dd_v2", params=None):
        p = params or {}
        super().__init__(name, p)

        # Tilt blend
        self.base_pct = float(p.get("base_pct", 0.90))
        self.tilt_pct = float(p.get("tilt_pct", 0.10))
        self.signal_scale = float(p.get("signal_scale", 1.0))

        # Multi-TF weights (63d, 126d, 252d)
        self.w_63 = float(p.get("w_63", 0.0))
        self.w_126 = float(p.get("w_126", 0.5))
        self.w_252 = float(p.get("w_252", 0.5))

        # Cross-sectional momentum
        self.xsect_weight = float(p.get("xsect_weight", 0.0))  # 0 = off

        # Value overlay
        self.value_weight = float(p.get("value_weight", 0.0))  # 0 = off
        self.value_zscore_clip = float(p.get("value_zscore_clip", 2.0))

        # Vol target (adaptive)
        self.vol_target = float(p.get("vol_target", 0.08))
        self.adaptive_vol = bool(p.get("adaptive_vol", False))

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
        tsmom_63 = dataset.features.get("tsmom_63d", {})
        tsmom_126 = dataset.features.get("tsmom_126d", {})
        tsmom_252 = dataset.features.get("tsmom_252d", {})
        zscore_60 = dataset.features.get("zscore_60d", {})
        vol_regime = dataset.features.get("vol_regime", {})

        # 1. Risk-parity weights (inverse-vol)
        inv_vols = {}; port_vols = {}
        for sym in syms:
            arr = rv.get(sym, [])
            v = arr[idx] if idx < len(arr) else 0.2
            if not (v == v) or v <= 0.01: v = 0.2
            inv_vols[sym] = 1.0 / v; port_vols[sym] = v
        total_iv = sum(inv_vols.values())
        if total_iv < 1e-10: return {}
        rp_weights = {s: inv_vols[s] / total_iv for s in syms}

        # 2. Multi-timeframe TSMOM signal per instrument
        tsmom_signals = {}
        for sym in syms:
            s63 = _safe_get(tsmom_63, sym, idx)
            s126 = _safe_get(tsmom_126, sym, idx)
            s252 = _safe_get(tsmom_252, sym, idx)
            # Weighted combo
            tw = self.w_63 + self.w_126 + self.w_252
            if tw < 1e-10: tw = 1.0
            combo = (self.w_63 * _clip(s63 / self.signal_scale) +
                     self.w_126 * _clip(s126 / self.signal_scale) +
                     self.w_252 * _clip(s252 / self.signal_scale)) / tw
            tsmom_signals[sym] = combo

        # 3. Cross-sectional momentum rank (rank-based tilt)
        xsect_tilt = {s: 0.0 for s in syms}
        if self.xsect_weight > 0:
            # Rank by 126d TSMOM
            ranked = sorted(syms, key=lambda s: tsmom_signals.get(s, 0))
            n = len(ranked)
            for rank_i, sym in enumerate(ranked):
                # Linear rank score: -1 (worst) to +1 (best)
                xsect_tilt[sym] = 2.0 * rank_i / max(n - 1, 1) - 1.0

        # 4. Value overlay (mean-reversion z-score)
        value_tilt = {s: 0.0 for s in syms}
        if self.value_weight > 0:
            for sym in syms:
                z = _safe_get(zscore_60, sym, idx)
                # Contrarian: overbought (z>0) → reduce, oversold (z<0) → increase
                value_tilt[sym] = -_clip(z / self.value_zscore_clip)

        # 5. Combine signals → allocation factor
        weights = {}
        for sym in syms:
            ts_factor = self.base_pct + self.tilt_pct * tsmom_signals.get(sym, 0)
            xs_factor = self.xsect_weight * xsect_tilt.get(sym, 0)
            val_factor = self.value_weight * value_tilt.get(sym, 0)
            factor = max(0.0, ts_factor + xs_factor + val_factor)
            weights[sym] = rp_weights[sym] * factor

        # 6. Vol targeting (adaptive or fixed)
        vt = self.vol_target
        if self.adaptive_vol:
            avg_vr = _avg_vol_regime(vol_regime, syms, idx)
            if avg_vr < 0.3:
                vt = self.vol_target * 1.3  # Calm regime: more leverage
            elif avg_vr > 0.7:
                vt = self.vol_target * 0.7  # Stressed regime: less leverage

        port_vol_sq = sum((weights.get(s, 0) * port_vols.get(s, 0.2)) ** 2 for s in syms)
        port_vol = math.sqrt(port_vol_sq) if port_vol_sq > 0 else 0.01
        if port_vol > 0.01:
            scale = vt / port_vol
            weights = {s: w * min(scale, 3.0) for s, w in weights.items()}

        # 7. DD deleveraging
        dd_factor = self._dd_factor(dataset, idx)
        if dd_factor < 1.0:
            weights = {s: w * dd_factor for s, w in weights.items()}

        # 8. Vol regime ceiling
        avg_vr = _avg_vol_regime(vol_regime, syms, idx)
        if avg_vr > 0.8:
            weights = {s: w * max(0.5, 1.0 - (avg_vr - 0.8)) for s, w in weights.items()}

        return {s: w for s, w in weights.items() if w > 1e-10}

    def _dd_factor(self, dataset, idx):
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
        if dd <= self.dd_threshold: return 1.0
        elif dd >= self.dd_max: return self.dd_min_alloc
        else:
            progress = (dd - self.dd_threshold) / (self.dd_max - self.dd_threshold)
            return 1.0 - progress * (1.0 - self.dd_min_alloc)


def _safe_get(feature_dict, sym, idx):
    arr = feature_dict.get(sym, [])
    v = arr[idx] if idx < len(arr) else 0
    return v if (v == v) else 0

def _clip(v, lo=-1, hi=1):
    return max(lo, min(hi, v))

def _avg_vol_regime(vol_regime, syms, idx):
    total = 0
    for sym in syms:
        arr = vol_regime.get(sym, [])
        if idx < len(arr):
            v = arr[idx]
            if v == v: total += v
    return total / len(syms) if syms else 0


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
            strat = RPMomDDV2Strategy(params=params)
            result = BacktestEngine(BacktestConfig(costs=cost_model)).run(ds_s, strat, seed=42)
            results[plabel] = compute_metrics(result.equity_curve)
        except Exception as e:
            logger.warning(f"  Error: {e}")
            import traceback; traceback.print_exc()
    return results


def main():
    logger.info("=" * 80)
    logger.info("Phase 144: Enhanced Signals — XSect Mom + Multi-TF + Value + Adaptive Vol")
    logger.info("=" * 80)

    DIVERSIFIED_14 = ["CL=F", "NG=F", "GC=F", "SI=F", "HG=F", "ZW=F", "ZC=F", "ZS=F",
                       "EURUSD=X", "GBPUSD=X", "AUDUSD=X", "JPY=X", "TLT", "IEF"]

    cfg = {"symbols": DIVERSIFIED_14, "start": "2005-01-01", "end": "2026-02-20",
           "cache_dir": "data/cache/yahoo_futures", "min_valid_bars": 300, "request_delay": 0.3}
    dataset = YahooFuturesProvider(cfg, seed=42).load()
    logger.info(f"  Loaded: {len(dataset.symbols)} symbols x {dataset.meta['n_bars']} bars")

    cost_model = commodity_futures_cost_model(slippage_bps=5.0, commission_bps=1.0, spread_bps=1.0)

    periods = [
        ("FULL", "2007-01-01", "2026-02-20"),
        ("IS", "2007-01-01", "2015-12-31"),
        ("OOS1", "2016-01-01", "2020-12-31"),
        ("OOS2", "2021-01-01", "2026-02-20"),
    ]

    # Baseline = Phase 143 champion
    baseline = {
        "base_pct": 0.90, "tilt_pct": 0.10, "signal_scale": 1.0,
        "w_63": 0.0, "w_126": 0.5, "w_252": 0.5,
        "xsect_weight": 0.0, "value_weight": 0.0, "adaptive_vol": False,
        "vol_target": 0.08, "dd_threshold": 0.05, "dd_max": 0.15,
        "dd_min_alloc": 0.10, "rebalance_freq": 42,
    }

    configs = [
        # Baseline
        ("P143_baseline", baseline),

        # 144a: Add 63d TSMOM (multi-timeframe)
        ("MTF_333", {**baseline, "w_63": 0.33, "w_126": 0.33, "w_252": 0.34}),
        ("MTF_253", {**baseline, "w_63": 0.20, "w_126": 0.50, "w_252": 0.30}),
        ("MTF_154", {**baseline, "w_63": 0.10, "w_126": 0.50, "w_252": 0.40}),

        # 144b: Cross-sectional momentum
        ("XS_0.05", {**baseline, "xsect_weight": 0.05}),
        ("XS_0.10", {**baseline, "xsect_weight": 0.10}),
        ("XS_0.15", {**baseline, "xsect_weight": 0.15}),
        ("XS_0.20", {**baseline, "xsect_weight": 0.20}),

        # 144c: Value overlay
        ("VAL_0.03", {**baseline, "value_weight": 0.03}),
        ("VAL_0.05", {**baseline, "value_weight": 0.05}),
        ("VAL_0.10", {**baseline, "value_weight": 0.10}),

        # 144d: Adaptive vol targeting
        ("AVOL", {**baseline, "adaptive_vol": True}),

        # 144e: Combinations of best from each
        ("MTF+XS_0.10", {**baseline, "w_63": 0.20, "w_126": 0.50, "w_252": 0.30, "xsect_weight": 0.10}),
        ("MTF+XS+VAL", {**baseline, "w_63": 0.20, "w_126": 0.50, "w_252": 0.30, "xsect_weight": 0.10, "value_weight": 0.05}),
        ("MTF+XS+AVOL", {**baseline, "w_63": 0.20, "w_126": 0.50, "w_252": 0.30, "xsect_weight": 0.10, "adaptive_vol": True}),
        ("ALL_signals", {**baseline, "w_63": 0.20, "w_126": 0.50, "w_252": 0.30, "xsect_weight": 0.10, "value_weight": 0.05, "adaptive_vol": True}),

        # 144f: Higher tilt with enhanced signals
        ("HT_MTF+XS", {**baseline, "tilt_pct": 0.20, "base_pct": 0.80, "w_63": 0.20, "w_126": 0.50, "w_252": 0.30, "xsect_weight": 0.10}),
        ("HT_ALL", {**baseline, "tilt_pct": 0.20, "base_pct": 0.80, "w_63": 0.20, "w_126": 0.50, "w_252": 0.30, "xsect_weight": 0.10, "value_weight": 0.05, "adaptive_vol": True}),
    ]

    # Run all
    all_results = {}
    best_obj = -99; best_label = ""

    logger.info(f"\n  {'Config':25s}  {'FULL':>8s}  {'IS':>8s}  {'OOS1':>8s}  {'OOS2':>8s}  {'OOS_MIN':>8s}  {'OBJ':>8s}")
    logger.info("  " + "-" * 90)

    for label, params in configs:
        r = run_test(dataset, params, cost_model, periods)
        all_results[label] = r

        s_full = r.get("FULL", {}).get("sharpe", 0)
        s_oos1 = r.get("OOS1", {}).get("sharpe", 0)
        s_oos2 = r.get("OOS2", {}).get("sharpe", 0)
        oos_min = min(s_oos1, s_oos2)
        obj = 0.6 * oos_min + 0.4 * s_full
        star = ""
        if obj > best_obj:
            best_obj = obj; best_label = label; star = " ***"
        logger.info(f"  {label:25s}  {s_full:+8.3f}  {r.get('IS',{}).get('sharpe',0):+8.3f}  {s_oos1:+8.3f}  {s_oos2:+8.3f}  {oos_min:+8.3f}  {obj:+8.3f}{star}")

    logger.info(f"\n  BEST: {best_label} → OBJ={best_obj:+.3f}")

    # Show best config details
    best_r = all_results[best_label]
    logger.info(f"\n  BEST results:")
    for p in ["FULL", "IS", "OOS1", "OOS2"]:
        m = best_r.get(p, {})
        logger.info(f"    {p:6s}  Sharpe={m.get('sharpe',0):+.3f}  CAGR={m.get('cagr',0)*100:+.1f}%  MDD={m.get('max_drawdown',1)*100:.1f}%")

    # Save
    output = {
        "phase": "144", "title": "Enhanced Signals",
        "results": {k: {p: dict(m) for p, m in r.items()} for k, r in all_results.items()},
        "best_config": best_label, "best_obj": best_obj,
        "best_params": dict(configs[[l for l, _ in configs].index(best_label)][1]),
    }
    out_dir = ROOT / "artifacts" / "phase144"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "enhanced_signals_report.json", "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\n  Saved: artifacts/phase144/enhanced_signals_report.json")

    return output


if __name__ == "__main__":
    main()
