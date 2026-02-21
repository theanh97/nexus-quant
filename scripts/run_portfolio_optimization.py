#!/usr/bin/env python3
"""
NEXUS Cross-Project Portfolio Optimization
============================================

Runs full portfolio optimization with updated strategy profiles:
  - crypto_perps: P91b champion, WF avg Sharpe 2.006 (production)
  - crypto_options: Mixed-freq VRP+Skew ensemble, WF avg Sharpe 2.723 (validated)

Outputs:
  1. Optimal static allocation (max Sharpe)
  2. Efficient frontier across correlation assumptions
  3. Dynamic vol-regime allocation
  4. Cross-project signal aggregator status
  5. Full report saved to artifacts/portfolio/
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexus_quant.portfolio.optimizer import PortfolioOptimizer, StrategyProfile
from nexus_quant.portfolio.aggregator import NexusSignalAggregator


def run_optimization():
    """Run full portfolio optimization."""
    print("=" * 70)
    print("  NEXUS CROSS-PROJECT PORTFOLIO OPTIMIZATION")
    print("=" * 70)
    print()

    opt = PortfolioOptimizer()

    # ── Strategy Profiles ──
    print("  STRATEGY PROFILES:")
    for s in opt.strategies:
        yr_str = " ".join(f"{yr}:{sh:.2f}" for yr, sh in sorted(s.year_sharpe.items()))
        print(f"    {s.name:25s} avg={s.avg_sharpe:.3f} min={s.min_sharpe:.3f} "
              f"vol={s.annual_vol_pct:.1f}% [{s.status}]")
        print(f"      years: {yr_str}")
    print()

    # ── Grid Search at Multiple Correlations ──
    print("  OPTIMAL ALLOCATION BY CORRELATION ASSUMPTION:")
    print(f"  {'Corr':>6} {'Perps%':>8} {'Opts%':>8} {'AvgSh':>8} {'MinSh':>8} "
          f"{'Vol%':>8} {'Ret%':>8} {'DivBen':>8}")
    print("  " + "-" * 64)

    frontier = opt.efficient_frontier(
        step=0.05,
        correlations=[0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
    )

    best_overall = None
    for corr, results in sorted(frontier.items()):
        best = results[0]
        perps_w = best.weights.get("crypto_perps", 0)
        opts_w = best.weights.get("crypto_options", 0)
        print(f"  {corr:>6.2f} {perps_w:>7.0%} {opts_w:>7.0%} "
              f"{best.avg_sharpe:>8.3f} {best.min_sharpe:>8.3f} "
              f"{best.annual_vol_pct:>8.1f} {best.avg_return_pct:>8.1f} "
              f"{best.diversification_benefit:>+8.3f}")
        if best_overall is None or best.avg_sharpe > best_overall.avg_sharpe:
            best_overall = best
    print()

    # ── Detailed Results at Base Correlation ──
    base_corr = 0.20
    print(f"  DETAILED ALLOCATION (correlation={base_corr:.2f}):")
    results = opt.optimize(step=0.05, correlation_override=base_corr)

    # Top 5
    for r in results[:5]:
        perps_w = r.weights.get("crypto_perps", 0)
        opts_w = r.weights.get("crypto_options", 0)
        yr_str = " ".join(f"{yr}:{sh:.2f}" for yr, sh in sorted(r.year_sharpe.items()))
        print(f"    {perps_w:3.0%}/{opts_w:3.0%}  avg={r.avg_sharpe:.3f} min={r.min_sharpe:.3f} "
              f"vol={r.annual_vol_pct:.1f}% ret={r.avg_return_pct:.1f}% "
              f"div={r.diversification_benefit:+.3f}")
        print(f"           {yr_str}")
    print()

    # ── Special Allocations ──
    max_sharpe = results[0]
    max_min_sharpe = max(results, key=lambda r: r.min_sharpe)
    min_vol = min(results, key=lambda r: r.annual_vol_pct)

    print("  KEY PORTFOLIOS:")
    for label, r in [("Max Sharpe", max_sharpe), ("Max Min-Sharpe", max_min_sharpe), ("Min Vol", min_vol)]:
        perps_w = r.weights.get("crypto_perps", 0)
        opts_w = r.weights.get("crypto_options", 0)
        print(f"    {label:20s}: {perps_w:.0%}/{opts_w:.0%}  "
              f"avg={r.avg_sharpe:.3f} min={r.min_sharpe:.3f} vol={r.annual_vol_pct:.1f}%")
    print()

    # ── Dynamic Allocation ──
    print("  DYNAMIC VOL-REGIME ALLOCATION:")
    dyn = opt.dynamic_allocation()
    dyn_sharpes = []
    for yr in sorted(dyn.keys()):
        r = dyn[yr]
        perps_w = r.weights.get("crypto_perps", 0)
        opts_w = r.weights.get("crypto_options", 0)
        print(f"    {yr}: {perps_w:.0%}/{opts_w:.0%}  Sharpe={r.avg_sharpe:.3f}")
        dyn_sharpes.append(r.avg_sharpe)
    dyn_avg = sum(dyn_sharpes) / len(dyn_sharpes)
    dyn_min = min(dyn_sharpes)
    print(f"    Dynamic avg={dyn_avg:.3f} min={dyn_min:.3f}")
    print()

    # ── Comparison Summary ──
    print("  COMPARISON SUMMARY:")
    comparisons = [
        ("crypto_perps only (100/0)", 1.0, 0.0),
        ("crypto_options only (0/100)", 0.0, 1.0),
        ("Equal weight (50/50)", 0.50, 0.50),
        ("Optimal", max_sharpe.weights.get("crypto_perps", 0),
         max_sharpe.weights.get("crypto_options", 0)),
    ]
    for label, wp, wo in comparisons:
        weights = {"crypto_perps": wp, "crypto_options": wo}
        avg_ret, port_vol, yr_sharpe = opt.portfolio_stats(weights, correlation_override=base_corr)
        avg_sh = sum(yr_sharpe.values()) / len(yr_sharpe)
        min_sh = min(yr_sharpe.values())
        print(f"    {label:30s}  avg={avg_sh:.3f} min={min_sh:.3f} "
              f"vol={port_vol:.1f}% ret={avg_ret:.1f}%")
    print()

    return opt, results, frontier


def run_aggregator_status():
    """Show cross-project signal aggregator status."""
    print("=" * 70)
    print("  CROSS-PROJECT SIGNAL AGGREGATOR")
    print("=" * 70)
    print()

    agg = NexusSignalAggregator()
    print(f"  Portfolio weights: {agg.portfolio_weights}")
    print()

    # Read latest signals
    for project in agg.portfolio_weights:
        sig = agg.read_latest_signal(project)
        if sig is None:
            print(f"  {project}: NO SIGNAL AVAILABLE")
        else:
            stale = "(STALE)" if sig.age_seconds > 86400 else ""
            print(f"  {project}: confidence={sig.confidence} "
                  f"age={sig.age_seconds:,}s {stale}")
            for sym, w in sorted(sig.target_weights.items()):
                print(f"    {sym:12s} {w:+.4f}")
    print()


def main():
    t0 = time.time()

    # 1. Portfolio optimization
    opt, results, frontier = run_optimization()

    # 2. Signal aggregator status
    run_aggregator_status()

    # 3. Full optimizer report
    print("=" * 70)
    print("  FULL OPTIMIZER REPORT")
    print("=" * 70)
    print(opt.report(correlation=0.20))
    print()

    # 4. Save results
    os.makedirs("artifacts/portfolio", exist_ok=True)
    output = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "strategies": [
            {
                "name": s.name,
                "market": s.market,
                "avg_sharpe": round(s.avg_sharpe, 4),
                "min_sharpe": round(s.min_sharpe, 4),
                "annual_vol_pct": s.annual_vol_pct,
                "year_sharpe": s.year_sharpe,
                "status": s.status,
            }
            for s in opt.strategies
        ],
        "optimal_allocation": {
            "weights": results[0].weights,
            "avg_sharpe": results[0].avg_sharpe,
            "min_sharpe": results[0].min_sharpe,
            "annual_vol_pct": results[0].annual_vol_pct,
            "avg_return_pct": results[0].avg_return_pct,
            "diversification_benefit": results[0].diversification_benefit,
            "year_sharpe": results[0].year_sharpe,
        },
        "efficient_frontier": {
            str(corr): {
                "weights": res[0].weights,
                "avg_sharpe": res[0].avg_sharpe,
                "min_sharpe": res[0].min_sharpe,
            }
            for corr, res in frontier.items()
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out_path = "artifacts/portfolio/optimization_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved to {out_path}")
    print(f"  Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
