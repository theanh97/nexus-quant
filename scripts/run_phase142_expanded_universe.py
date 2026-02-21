"""
Phase 142: Expanded Universe CTA — 19 instruments across 4 asset classes
=========================================================================
Phase 141 found: 8 commodities with Yahoo data → Sharpe 0.6 max.
Root cause: too concentrated. Pro CTAs use 50-100+ instruments.

Fix: Expand to ALL cached instruments:
  - Commodities (11): CL, NG, GC, SI, HG, ZW, ZC, ZS, PL, KC, SB, CT, BZ
  - FX (4): EURUSD, GBPUSD, AUDUSD, JPY
  - Bonds (2): TLT, IEF

Strategy: Risk-Parity + TSMOM Momentum Tilt + Drawdown Deleveraging
(Best from Phase 141d, now with 19 instruments for diversification)
"""
from __future__ import annotations

import json, logging, math, sys, hashlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("cta_p142")

from nexus_quant.projects.commodity_cta.providers.yahoo_futures import YahooFuturesProvider
from nexus_quant.projects.commodity_cta.costs.futures_fees import commodity_futures_cost_model
from nexus_quant.strategies.base import Strategy, Weights
from nexus_quant.data.schema import MarketDataset
from nexus_quant.backtest.engine import BacktestEngine, BacktestConfig


class RPMomDDStrategy(Strategy):
    """Risk-Parity + Momentum Tilt + Drawdown Deleveraging."""
    def __init__(self, name="rp_mom_dd", params=None):
        p = params or {}
        super().__init__(name, p)
        self.base_pct = float(p.get("base_pct", 0.80))
        self.tilt_pct = float(p.get("tilt_pct", 0.20))
        self.signal_scale = float(p.get("signal_scale", 2.0))
        self.vol_target = float(p.get("vol_target", 0.15))
        self.dd_threshold = float(p.get("dd_threshold", 0.10))
        self.dd_max = float(p.get("dd_max", 0.25))
        self.dd_min_alloc = float(p.get("dd_min_alloc", 0.20))
        self.warmup = int(p.get("warmup", 270))
        self.rebalance_freq = int(p.get("rebalance_freq", 21))
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

        # Risk-parity weights
        inv_vols = {}
        port_vols = {}
        for sym in syms:
            arr = rv.get(sym, [])
            v = arr[idx] if idx < len(arr) else 0.2
            if not (v == v) or v <= 0.01: v = 0.2
            inv_vols[sym] = 1.0 / v
            port_vols[sym] = v
        total_iv = sum(inv_vols.values())
        rp_weights = {s: inv_vols[s] / total_iv for s in syms}

        # Momentum tilt (6m/12m)
        weights = {}
        for sym in syms:
            arr126 = tsmom_126.get(sym, [])
            arr252 = tsmom_252.get(sym, [])
            s126 = arr126[idx] if idx < len(arr126) else 0
            s252 = arr252[idx] if idx < len(arr252) else 0
            if not (s126 == s126): s126 = 0
            if not (s252 == s252): s252 = 0
            combo = 0.5 * max(-1, min(1, s126 / self.signal_scale)) + 0.5 * max(-1, min(1, s252 / self.signal_scale))
            factor = max(0.0, self.base_pct + self.tilt_pct * combo)
            weights[sym] = rp_weights[sym] * factor

        # Scale to vol target
        port_vol_sq = sum((weights.get(s, 0) * port_vols.get(s, 0.2)) ** 2 for s in syms)
        port_vol = math.sqrt(port_vol_sq) if port_vol_sq > 0 else 0.01
        if port_vol > 0.01:
            scale = self.vol_target / port_vol
            weights = {s: w * min(scale, 3.0) for s, w in weights.items()}

        # Drawdown deleveraging
        dd_factor = self._dd_factor(dataset, idx)
        if dd_factor < 1.0:
            weights = {s: w * dd_factor for s, w in weights.items()}

        # Vol ceiling
        avg_vr = 0
        for sym in syms:
            arr = vol_regime.get(sym, [])
            if idx < len(arr):
                v = arr[idx]
                if v == v: avg_vr += v
        avg_vr /= len(syms) if syms else 1
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


def run_test(label, dataset, params, cost_model, periods):
    results = {}
    for plabel, start, end in periods:
        try:
            ds_s = slice_dataset(dataset, start, end)
            strat = RPMomDDStrategy(params=params)
            result = BacktestEngine(BacktestConfig(costs=cost_model)).run(ds_s, strat, seed=42)
            m = compute_metrics(result.equity_curve)
            results[plabel] = m
            logger.info(f"  [{label:30s}] {plabel:6s}  Sharpe={m['sharpe']:+6.3f}  CAGR={m['cagr']*100:+6.2f}%  MDD={m['max_drawdown']*100:5.1f}%")
        except Exception as e:
            logger.warning(f"  [{label}] {plabel}: {e}")
            import traceback; traceback.print_exc()
    return results


def main():
    logger.info("=" * 80)
    logger.info("Phase 142: Expanded Universe CTA — 19 instruments, 4 asset classes")
    logger.info("=" * 80)

    # All cached instruments
    ALL_19 = [
        # Commodities (13)
        "CL=F", "NG=F", "GC=F", "SI=F", "HG=F", "PL=F",  # Energy + Metals
        "ZW=F", "ZC=F", "ZS=F",                             # Grains
        "KC=F", "SB=F", "CT=F", "BZ=F",                     # Softs + Brent
        # FX (4)
        "EURUSD=X", "GBPUSD=X", "AUDUSD=X", "JPY=X",
        # Bonds (2)
        "TLT", "IEF",
    ]

    COMMODITY_8 = ["CL=F", "NG=F", "GC=F", "SI=F", "HG=F", "ZW=F", "ZC=F", "ZS=F"]
    DIVERSIFIED_14 = ["CL=F", "NG=F", "GC=F", "SI=F", "HG=F", "ZW=F", "ZC=F", "ZS=F",
                       "EURUSD=X", "GBPUSD=X", "AUDUSD=X", "JPY=X", "TLT", "IEF"]

    cost_model = commodity_futures_cost_model(slippage_bps=5.0, commission_bps=1.0, spread_bps=1.0)

    periods = [
        ("FULL", "2007-01-01", "2026-02-20"),
        ("IS", "2007-01-01", "2015-12-31"),
        ("OOS1", "2016-01-01", "2020-12-31"),
        ("OOS2", "2021-01-01", "2026-02-20"),
    ]

    # Load datasets
    universes = [
        ("8-comm", COMMODITY_8),
        ("14-div", DIVERSIFIED_14),
        ("19-all", ALL_19),
    ]

    datasets = {}
    for name, syms in universes:
        logger.info(f"\n  Loading {name} ({len(syms)} instruments)...")
        cfg = {"symbols": syms, "start": "2005-01-01", "end": "2026-02-20",
               "cache_dir": "data/cache/yahoo_futures", "min_valid_bars": 300, "request_delay": 0.3}
        try:
            ds = YahooFuturesProvider(cfg, seed=42).load()
            datasets[name] = ds
            logger.info(f"  ✓ {name}: {len(ds.symbols)} symbols × {ds.meta['n_bars']} bars")
        except Exception as e:
            logger.warning(f"  ✗ {name}: {e}")

    # Strategy configs to test
    param_configs = [
        ("RP+Mom(12%)", {"vol_target": 0.12, "base_pct": 0.80, "tilt_pct": 0.20, "dd_threshold": 0.10, "dd_max": 0.25}),
        ("RP+Mom(15%)", {"vol_target": 0.15, "base_pct": 0.80, "tilt_pct": 0.20, "dd_threshold": 0.10, "dd_max": 0.25}),
        ("RP+Mom(20%)", {"vol_target": 0.20, "base_pct": 0.80, "tilt_pct": 0.20, "dd_threshold": 0.10, "dd_max": 0.25}),
        ("RP+Mom(15%,90/10)", {"vol_target": 0.15, "base_pct": 0.90, "tilt_pct": 0.10, "dd_threshold": 0.10, "dd_max": 0.25}),
        ("RP-only(15%)", {"vol_target": 0.15, "base_pct": 1.00, "tilt_pct": 0.00, "dd_threshold": 0.10, "dd_max": 0.25}),
    ]

    # Run all combinations
    all_results = {}
    for univ_name, ds in datasets.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"  UNIVERSE: {univ_name} ({len(ds.symbols)} instruments)")
        logger.info(f"{'='*80}")

        for param_name, params in param_configs:
            label = f"{univ_name}/{param_name}"
            r = run_test(label, ds, params, cost_model, periods)
            all_results[label] = r

    # Final comparison
    logger.info(f"\n{'='*80}")
    logger.info(f"  FINAL COMPARISON — All Universe × Strategy Combinations")
    logger.info(f"{'='*80}")
    logger.info(f"\n  {'Config':35s}  {'FULL':>8s}  {'IS':>8s}  {'OOS1':>8s}  {'OOS2':>8s}  {'OOS_MIN':>8s}  {'MDD':>6s}  {'CAGR':>7s}")
    logger.info("  " + "-" * 100)

    best_oos_min = -99; best_config = "none"
    best_full = -99; best_full_config = "none"

    for label, r in sorted(all_results.items()):
        def s(p): return r.get(p, {}).get("sharpe", float("nan"))
        mdd = r.get("FULL", {}).get("max_drawdown", 1)
        cagr = r.get("FULL", {}).get("cagr", 0)
        oos_min = min(s("OOS1"), s("OOS2")) if s("OOS1") == s("OOS1") and s("OOS2") == s("OOS2") else float("nan")
        def fmt(v): return f"{v:+8.3f}" if v == v else "     N/A"
        star = ""
        if oos_min == oos_min and oos_min > best_oos_min:
            best_oos_min = oos_min; best_config = label; star = " <-BEST"
        if s("FULL") == s("FULL") and s("FULL") > best_full:
            best_full = s("FULL"); best_full_config = label
        logger.info(f'  {label:35s}  {fmt(s("FULL"))}  {fmt(s("IS"))}  {fmt(s("OOS1"))}  {fmt(s("OOS2"))}  {fmt(oos_min)}  {mdd*100:5.1f}%  {cagr*100:+5.1f}%{star}')

    logger.info(f"\n  BEST OOS_MIN:  {best_config}  → {best_oos_min:+.3f}")
    logger.info(f"  BEST FULL:     {best_full_config}  → {best_full:+.3f}")
    logger.info(f"  TARGET:        Sharpe > 0.8")
    if best_oos_min > 0.8:
        logger.info(f"  STATUS:        TARGET MET!")
    elif best_oos_min > 0.6:
        logger.info(f"  STATUS:        Close ({best_oos_min:+.3f}/0.8 = {best_oos_min/0.8*100:.0f}%)")
    else:
        logger.info(f"  STATUS:        Below target ({best_oos_min:+.3f}/0.8)")

    # Save
    output = {
        "phase": "142", "title": "Expanded Universe CTA",
        "results": {k: {p: dict(m) for p, m in r.items()} for k, r in all_results.items()},
        "best_config": best_config, "best_oos_min": best_oos_min,
        "best_full_config": best_full_config, "best_full": best_full,
    }
    out_dir = ROOT / "artifacts" / "phase142"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "expanded_universe_report.json", "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\n  Saved: artifacts/phase142/expanded_universe_report.json")

    return output


if __name__ == "__main__":
    main()
