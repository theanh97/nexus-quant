"""
Phase 145b: Fine-tune sector budget allocation around best configs.
"""
from __future__ import annotations

import json, logging, math, sys, hashlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("cta_p145b")

# Re-use SectorRPStrategy from phase 145
sys.path.insert(0, str(ROOT / "scripts"))
from run_phase145_sector_rp import (SectorRPStrategy, compute_metrics, slice_dataset,
                                      _safe_get, _clip, SECTORS)

from nexus_quant.projects.commodity_cta.providers.yahoo_futures import YahooFuturesProvider
from nexus_quant.projects.commodity_cta.costs.futures_fees import commodity_futures_cost_model
from nexus_quant.data.schema import MarketDataset
from nexus_quant.backtest.engine import BacktestEngine, BacktestConfig


def run_test(dataset, params, cost_model, periods):
    results = {}
    for plabel, start, end in periods:
        try:
            ds_s = slice_dataset(dataset, start, end)
            strat = SectorRPStrategy(params=params)
            result = BacktestEngine(BacktestConfig(costs=cost_model)).run(ds_s, strat, seed=42)
            results[plabel] = compute_metrics(result.equity_curve)
        except Exception as e:
            logger.warning(f"  Error: {e}")
    return results


def main():
    logger.info("=" * 80)
    logger.info("Phase 145b: Fine-tune sector budget allocation")
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
        "base_pct": 0.90, "tilt_pct": 0.10, "signal_scale": 1.0,
        "xsect_weight": 0.05, "vol_target": 0.08,
        "dd_threshold": 0.05, "dd_max": 0.15, "dd_min_alloc": 0.10,
        "rebalance_freq": 42,
    }

    # Fine grid around 40/30/30 and 40/40/20
    configs = []
    for wc in [0.35, 0.40, 0.45, 0.50]:
        for wf in [0.20, 0.25, 0.30, 0.35, 0.40]:
            wb = round(1.0 - wc - wf, 2)
            if wb < 0.10 or wb > 0.45:
                continue
            label = f"c{int(wc*100)}_f{int(wf*100)}_b{int(wb*100)}"
            configs.append((label, {**base, "w_commodity": wc, "w_fx": wf, "w_bond": wb}))

    # Also test with XSect variations on best
    for xs in [0.0, 0.03, 0.05, 0.08, 0.10]:
        configs.append((f"c40_f30_b30_xs{int(xs*100)}", {**base, "w_commodity": 0.40, "w_fx": 0.30, "w_bond": 0.30, "xsect_weight": xs}))

    # Run
    all_results = {}
    best_obj = -99; best_label = ""

    logger.info(f"\n  {'Config':30s}  {'FULL':>8s}  {'IS':>8s}  {'OOS1':>8s}  {'OOS2':>8s}  {'OOS_MIN':>8s}  {'OBJ':>8s}")
    logger.info("  " + "-" * 100)

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
        logger.info(f"  {label:30s}  {s_full:+8.3f}  {r.get('IS',{}).get('sharpe',0):+8.3f}  {s_oos1:+8.3f}  {s_oos2:+8.3f}  {oos_min:+8.3f}  {obj:+8.3f}{star}")

    logger.info(f"\n  BEST: {best_label} → OBJ={best_obj:+.3f}")

    best_r = all_results[best_label]
    for p in ["FULL", "IS", "OOS1", "OOS2"]:
        m = best_r.get(p, {})
        logger.info(f"    {p:6s}  Sharpe={m.get('sharpe',0):+.3f}  CAGR={m.get('cagr',0)*100:+.1f}%  MDD={m.get('max_drawdown',1)*100:.1f}%")

    # Save
    output = {
        "phase": "145b", "title": "Sector Budget Fine-tune",
        "results": {k: {p: dict(m) for p, m in r.items()} for k, r in all_results.items()},
        "best_config": best_label, "best_obj": best_obj,
    }
    out_dir = ROOT / "artifacts" / "phase145"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "sector_finetune_report.json", "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\n  Saved: artifacts/phase145/sector_finetune_report.json")


if __name__ == "__main__":
    main()
