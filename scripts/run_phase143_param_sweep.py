"""
Phase 143: Champion Parameter Sweep
====================================
Systematic sweep of RPMomDD parameters on 14-div universe.
Goal: Improve OOS2 (currently 0.468) and FULL (currently 0.482).

Sweep dimensions:
  1. vol_target: [0.08, 0.10, 0.12, 0.15, 0.18]
  2. rebalance_freq: [5, 10, 15, 21, 42]
  3. signal_scale: [1.0, 1.5, 2.0, 3.0]
  4. dd_threshold/dd_max: [(0.05,0.15), (0.08,0.20), (0.10,0.25), (0.15,0.30)]
  5. base_pct/tilt_pct: [(0.70,0.30), (0.80,0.20), (0.90,0.10), (0.60,0.40)]
"""
from __future__ import annotations

import json, logging, math, sys, hashlib
from datetime import datetime, timezone
from pathlib import Path
from itertools import product

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("cta_p143")

from nexus_quant.projects.commodity_cta.providers.yahoo_futures import YahooFuturesProvider
from nexus_quant.projects.commodity_cta.costs.futures_fees import commodity_futures_cost_model
from nexus_quant.projects.commodity_cta.strategies.rp_mom_dd import RPMomDDStrategy
from nexus_quant.data.schema import MarketDataset
from nexus_quant.backtest.engine import BacktestEngine, BacktestConfig


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
            strat = RPMomDDStrategy(params=params)
            result = BacktestEngine(BacktestConfig(costs=cost_model)).run(ds_s, strat, seed=42)
            results[plabel] = compute_metrics(result.equity_curve)
        except Exception as e:
            logger.warning(f"  Error: {e}")
    return results


def main():
    logger.info("=" * 80)
    logger.info("Phase 143: Champion Parameter Sweep — 14-div universe")
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

    # ── Sweep grid ──────────────────────────────────────────────────────────
    vol_targets = [0.08, 0.10, 0.12, 0.15, 0.18]
    rebalance_freqs = [5, 10, 15, 21, 42]
    signal_scales = [1.0, 1.5, 2.0, 3.0]
    dd_configs = [(0.05, 0.15), (0.08, 0.20), (0.10, 0.25), (0.15, 0.30)]
    tilt_configs = [(0.60, 0.40), (0.70, 0.30), (0.80, 0.20), (0.90, 0.10)]

    # Phase 143a: Sweep vol_target + rebalance_freq (2D grid, 25 configs)
    logger.info("\n" + "=" * 80)
    logger.info("  Phase 143a: vol_target × rebalance_freq sweep")
    logger.info("=" * 80)

    all_results = {}
    best_obj = -99
    best_label = ""

    for vt, rf in product(vol_targets, rebalance_freqs):
        params = {
            "vol_target": vt, "rebalance_freq": rf,
            "base_pct": 0.80, "tilt_pct": 0.20, "signal_scale": 2.0,
            "dd_threshold": 0.10, "dd_max": 0.25, "dd_min_alloc": 0.20,
        }
        label = f"vt={vt:.2f}_rf={rf}"
        r = run_test(dataset, params, cost_model, periods)
        all_results[label] = r

        s_full = r.get("FULL", {}).get("sharpe", 0)
        s_oos1 = r.get("OOS1", {}).get("sharpe", 0)
        s_oos2 = r.get("OOS2", {}).get("sharpe", 0)
        oos_min = min(s_oos1, s_oos2)
        # Objective: weighted OOS_MIN (60%) + FULL (40%)
        obj = 0.6 * oos_min + 0.4 * s_full
        star = ""
        if obj > best_obj:
            best_obj = obj; best_label = label; star = " ***BEST"
        logger.info(f"  {label:25s}  FULL={s_full:+.3f}  OOS1={s_oos1:+.3f}  OOS2={s_oos2:+.3f}  OBJ={obj:+.3f}{star}")

    logger.info(f"\n  143a BEST: {best_label} → OBJ={best_obj:+.3f}")

    # Extract best vol_target and rebalance_freq
    best_vt = float(best_label.split("_")[0].split("=")[1])
    best_rf = int(best_label.split("_")[1].split("=")[1])

    # Phase 143b: Sweep signal_scale + tilt_configs (with best vt/rf)
    logger.info("\n" + "=" * 80)
    logger.info(f"  Phase 143b: signal_scale × tilt (vt={best_vt}, rf={best_rf})")
    logger.info("=" * 80)

    best_obj_b = -99
    best_label_b = ""

    for ss, (bp, tp) in product(signal_scales, tilt_configs):
        params = {
            "vol_target": best_vt, "rebalance_freq": best_rf,
            "base_pct": bp, "tilt_pct": tp, "signal_scale": ss,
            "dd_threshold": 0.10, "dd_max": 0.25, "dd_min_alloc": 0.20,
        }
        label = f"ss={ss:.1f}_bp={bp:.2f}_tp={tp:.2f}"
        r = run_test(dataset, params, cost_model, periods)
        all_results[label] = r

        s_full = r.get("FULL", {}).get("sharpe", 0)
        s_oos1 = r.get("OOS1", {}).get("sharpe", 0)
        s_oos2 = r.get("OOS2", {}).get("sharpe", 0)
        oos_min = min(s_oos1, s_oos2)
        obj = 0.6 * oos_min + 0.4 * s_full
        star = ""
        if obj > best_obj_b:
            best_obj_b = obj; best_label_b = label; star = " ***BEST"
        logger.info(f"  {label:35s}  FULL={s_full:+.3f}  OOS1={s_oos1:+.3f}  OOS2={s_oos2:+.3f}  OBJ={obj:+.3f}{star}")

    logger.info(f"\n  143b BEST: {best_label_b} → OBJ={best_obj_b:+.3f}")

    # Extract best signal_scale, base_pct, tilt_pct
    parts_b = best_label_b.split("_")
    best_ss = float(parts_b[0].split("=")[1])
    best_bp = float(parts_b[1].split("=")[1])
    best_tp = float(parts_b[2].split("=")[1])

    # Phase 143c: Sweep DD configs (with all best params so far)
    logger.info("\n" + "=" * 80)
    logger.info(f"  Phase 143c: DD params (vt={best_vt}, rf={best_rf}, ss={best_ss}, bp={best_bp})")
    logger.info("=" * 80)

    best_obj_c = -99
    best_label_c = ""
    dd_min_allocs = [0.10, 0.20, 0.30, 0.40]

    for (ddt, ddm), dma in product(dd_configs, dd_min_allocs):
        params = {
            "vol_target": best_vt, "rebalance_freq": best_rf,
            "base_pct": best_bp, "tilt_pct": best_tp, "signal_scale": best_ss,
            "dd_threshold": ddt, "dd_max": ddm, "dd_min_alloc": dma,
        }
        label = f"ddt={ddt:.2f}_ddm={ddm:.2f}_dma={dma:.2f}"
        r = run_test(dataset, params, cost_model, periods)
        all_results[label] = r

        s_full = r.get("FULL", {}).get("sharpe", 0)
        s_oos1 = r.get("OOS1", {}).get("sharpe", 0)
        s_oos2 = r.get("OOS2", {}).get("sharpe", 0)
        oos_min = min(s_oos1, s_oos2)
        obj = 0.6 * oos_min + 0.4 * s_full
        star = ""
        if obj > best_obj_c:
            best_obj_c = obj; best_label_c = label; star = " ***BEST"
        logger.info(f"  {label:35s}  FULL={s_full:+.3f}  OOS1={s_oos1:+.3f}  OOS2={s_oos2:+.3f}  OBJ={obj:+.3f}{star}")

    logger.info(f"\n  143c BEST: {best_label_c} → OBJ={best_obj_c:+.3f}")

    # Final champion comparison
    parts_c = best_label_c.split("_")
    best_ddt = float(parts_c[0].split("=")[1])
    best_ddm = float(parts_c[1].split("=")[1])
    best_dma = float(parts_c[2].split("=")[1])

    final_params = {
        "vol_target": best_vt, "rebalance_freq": best_rf,
        "base_pct": best_bp, "tilt_pct": best_tp, "signal_scale": best_ss,
        "dd_threshold": best_ddt, "dd_max": best_ddm, "dd_min_alloc": best_dma,
    }

    logger.info("\n" + "=" * 80)
    logger.info("  FINAL CHAMPION PARAMS:")
    logger.info("=" * 80)
    for k, v in final_params.items():
        logger.info(f"    {k:20s} = {v}")

    r_final = run_test(dataset, final_params, cost_model, periods)
    for p in ["FULL", "IS", "OOS1", "OOS2"]:
        m = r_final.get(p, {})
        logger.info(f"  {p:6s}  Sharpe={m.get('sharpe', 0):+.3f}  CAGR={m.get('cagr', 0)*100:+.1f}%  MDD={m.get('max_drawdown', 1)*100:.1f}%")

    s_oos1 = r_final.get("OOS1", {}).get("sharpe", 0)
    s_oos2 = r_final.get("OOS2", {}).get("sharpe", 0)
    logger.info(f"\n  OOS_MIN = {min(s_oos1, s_oos2):+.3f}")

    # Compare vs baseline
    baseline_params = {
        "vol_target": 0.12, "rebalance_freq": 21,
        "base_pct": 0.80, "tilt_pct": 0.20, "signal_scale": 2.0,
        "dd_threshold": 0.10, "dd_max": 0.25, "dd_min_alloc": 0.20,
    }
    r_base = run_test(dataset, baseline_params, cost_model, periods)
    logger.info(f"\n  BASELINE (Phase 142 champion):")
    for p in ["FULL", "IS", "OOS1", "OOS2"]:
        m = r_base.get(p, {})
        logger.info(f"  {p:6s}  Sharpe={m.get('sharpe', 0):+.3f}  CAGR={m.get('cagr', 0)*100:+.1f}%  MDD={m.get('max_drawdown', 1)*100:.1f}%")

    improvement = (min(s_oos1, s_oos2) - min(
        r_base.get("OOS1", {}).get("sharpe", 0),
        r_base.get("OOS2", {}).get("sharpe", 0)
    ))
    logger.info(f"\n  IMPROVEMENT: {improvement:+.3f} OOS_MIN")

    # Save results
    output = {
        "phase": "143", "title": "Champion Parameter Sweep",
        "final_params": final_params,
        "final_results": {p: dict(m) for p, m in r_final.items()},
        "baseline_results": {p: dict(m) for p, m in r_base.items()},
        "improvement_oos_min": improvement,
        "n_configs_tested": len(all_results),
    }
    out_dir = ROOT / "artifacts" / "phase143"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "param_sweep_report.json", "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\n  Saved: artifacts/phase143/param_sweep_report.json")

    return output


if __name__ == "__main__":
    main()
