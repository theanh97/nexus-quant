#!/usr/bin/env python3
"""
R94: Execution Cost Optimization — Combo Orders & Alternative Strategies
=========================================================================

R92 revealed that BF trade costs (17.5-20 bps spread + 8-12 bps fees) threaten
profitability. This study explores cost reduction strategies:

  1. Combo order analysis (Deribit native combos reduce crossing cost)
  2. Partial-fill strategy (scale into position over multiple days)
  3. Entry threshold tightening (fewer trades = less cost)
  4. Sensitivity reduction (lower notional = less absolute cost)
  5. Asymmetric entry (only trade when z is extreme)
  6. Net cost-adjusted optimal config search
"""
import csv
import json
import math
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SURFACE_DIR = ROOT / "data" / "cache" / "deribit" / "real_surface"
DVOL_DIR = ROOT / "data" / "cache" / "deribit" / "dvol"
OUTPUT_PATH = SURFACE_DIR / "r94_execution_optimization.json"

# Base config (R69)
BASE_CONFIG = {
    "bf_lookback": 120,
    "bf_z_entry": 1.5,
    "bf_z_exit": 0.0,
    "bf_sensitivity": 2.5,
    "w_bf": 0.90,
    "w_vrp": 0.10,
    "vrp_leverage": 2.0,
}


def load_surface(asset):
    """Load daily surface CSV."""
    path = SURFACE_DIR / f"{asset}_daily_surface.csv"
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                data[row["date"]] = {
                    "butterfly_25d": float(row["butterfly_25d"]),
                    "iv_atm": float(row["iv_atm"]),
                    "spot": float(row["spot"]),
                }
            except (ValueError, KeyError):
                continue
    return data


def load_dvol(asset):
    """Load DVOL history (date normalized to YYYY-MM-DD)."""
    path = DVOL_DIR / f"{asset}_DVOL_12h.csv"
    data = {}
    if not path.exists():
        return data
    with open(path) as f:
        for row in csv.DictReader(f):
            d = row.get("date", "")[:10]
            try:
                data[d] = float(row["dvol_close"]) / 100.0
            except (ValueError, KeyError):
                continue
    return data


def backtest_bf(surface, dvol, config, cost_per_trade_bps=0):
    """Run BF backtest with given config and cost per trade."""
    bf_vals = {}
    all_dates = sorted(set(surface.keys()) & set(dvol.keys()))

    for d in all_dates:
        bf_vals[d] = surface[d]["butterfly_25d"]

    lb = config["bf_lookback"]
    z_entry = config["bf_z_entry"]
    sens = config["bf_sensitivity"]

    position = 0.0
    cum_pnl = 0.0
    peak = 0.0
    max_dd = 0.0
    n_trades = 0
    daily_pnls = []

    for i in range(lb, len(all_dates)):
        d = all_dates[i]
        dp = all_dates[i - 1]

        window = [bf_vals[all_dates[j]] for j in range(i - lb, i)]
        bf_mean = sum(window) / len(window)
        bf_std = math.sqrt(sum((v - bf_mean) ** 2 for v in window) / len(window))
        if bf_std < 1e-8:
            continue
        z = (bf_vals[d] - bf_mean) / bf_std

        old_pos = position
        if z > z_entry:
            position = -1.0
        elif z < -z_entry:
            position = 1.0

        trade_cost = 0
        if position != old_pos:
            n_trades += 1
            trade_cost = cost_per_trade_bps / 10000.0

        iv = dvol.get(d, 0)
        bf_change = bf_vals[d] - bf_vals.get(dp, bf_vals[d])
        dt = 1.0 / 365.0
        bf_pnl = position * bf_change * iv * math.sqrt(dt) * sens

        cum_pnl += bf_pnl - trade_cost
        peak = max(peak, cum_pnl)
        dd = cum_pnl - peak
        max_dd = min(max_dd, dd)
        daily_pnls.append(bf_pnl - trade_cost)

    n_days = len(daily_pnls)
    if n_days < 30:
        return None

    mean_p = sum(daily_pnls) / n_days
    std_p = math.sqrt(sum((v - mean_p) ** 2 for v in daily_pnls) / n_days) if n_days > 1 else 1
    sharpe = (mean_p / std_p * math.sqrt(365)) if std_p > 0 else 0
    ann_ret = cum_pnl * (365 / n_days)
    trades_yr = n_trades / n_days * 365

    return {
        "sharpe": round(sharpe, 2),
        "ann_ret_pct": round(ann_ret * 100, 2),
        "cum_pnl_pct": round(cum_pnl * 100, 4),
        "max_dd_pct": round(max_dd * 100, 4),
        "n_trades": n_trades,
        "trades_per_year": round(trades_yr, 1),
        "n_days": n_days,
    }


# ═══════════════════════════════════════════════════════════════
# Section 1: Cost Sensitivity Grid
# ═══════════════════════════════════════════════════════════════

def section_1_cost_sensitivity(btc_surface, btc_dvol, eth_surface, eth_dvol):
    """Grid search: Sharpe vs cost per trade."""
    print("\n  ── Section 1: Cost Sensitivity Grid ──")

    costs = [0, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30]

    results = {}
    for asset, surface, dvol in [("BTC", btc_surface, btc_dvol),
                                   ("ETH", eth_surface, eth_dvol)]:
        print(f"\n    {asset}:")
        print(f"    {'Cost(bps)':>10} {'Sharpe':>8} {'AnnRet%':>8} {'MaxDD%':>8} {'Trades/yr':>10}")
        print(f"    {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")

        asset_results = []
        for cost in costs:
            r = backtest_bf(surface, dvol, BASE_CONFIG, cost)
            if r:
                asset_results.append({"cost_bps": cost, **r})
                print(f"    {cost:>10} {r['sharpe']:>8.2f} {r['ann_ret_pct']:>8.2f} "
                      f"{r['max_dd_pct']:>8.2f} {r['trades_per_year']:>10.1f}")

        results[asset] = asset_results

    # Find breakeven cost
    for asset in ["BTC", "ETH"]:
        for r in results.get(asset, []):
            if r["sharpe"] < 1.0:
                print(f"\n    {asset} Sharpe drops below 1.0 at {r['cost_bps']} bps")
                break

    return results


# ═══════════════════════════════════════════════════════════════
# Section 2: Entry Threshold Optimization
# ═══════════════════════════════════════════════════════════════

def section_2_entry_threshold(btc_surface, btc_dvol, eth_surface, eth_dvol):
    """Higher z_entry = fewer trades = less cost impact."""
    print("\n  ── Section 2: Entry Threshold Optimization ──")

    z_entries = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
    maker_cost = 8  # bps per trade

    results = {}
    for asset, surface, dvol in [("BTC", btc_surface, btc_dvol),
                                   ("ETH", eth_surface, eth_dvol)]:
        print(f"\n    {asset} (with {maker_cost}bps maker cost):")
        print(f"    {'z_entry':>8} {'Sharpe':>8} {'AnnRet%':>8} {'Trades/yr':>10} "
              f"{'AnnCost':>8} {'NetRet%':>8}")
        print(f"    {'─'*8} {'─'*8} {'─'*8} {'─'*10} {'─'*8} {'─'*8}")

        asset_results = []
        for z in z_entries:
            config = {**BASE_CONFIG, "bf_z_entry": z}
            r = backtest_bf(surface, dvol, config, maker_cost)
            if r:
                annual_cost = r["trades_per_year"] * maker_cost / 10000 * 100
                net_ret = r["ann_ret_pct"] - annual_cost
                asset_results.append({
                    "z_entry": z,
                    "annual_cost_pct": round(annual_cost, 3),
                    **r,
                })
                print(f"    {z:>8.2f} {r['sharpe']:>8.2f} {r['ann_ret_pct']:>8.2f} "
                      f"{r['trades_per_year']:>10.1f} {annual_cost:>8.3f} {net_ret:>+8.3f}")

        results[asset] = asset_results

    return results


# ═══════════════════════════════════════════════════════════════
# Section 3: Sensitivity Reduction
# ═══════════════════════════════════════════════════════════════

def section_3_sensitivity(btc_surface, btc_dvol, eth_surface, eth_dvol):
    """Lower sensitivity = same Sharpe but less absolute cost drag."""
    print("\n  ── Section 3: Sensitivity vs Cost Impact ──")

    sensitivities = [1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 7.5]
    maker_cost = 8

    results = {}
    for asset, surface, dvol in [("BTC", btc_surface, btc_dvol),
                                   ("ETH", eth_surface, eth_dvol)]:
        print(f"\n    {asset} (with {maker_cost}bps maker cost):")
        print(f"    {'Sens':>6} {'Sharpe':>8} {'AnnRet%':>8} {'MaxDD%':>8} {'CostDrag':>9}")
        print(f"    {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*9}")

        asset_results = []
        for s in sensitivities:
            config = {**BASE_CONFIG, "bf_sensitivity": s}
            r_no_cost = backtest_bf(surface, dvol, config, 0)
            r_with_cost = backtest_bf(surface, dvol, config, maker_cost)
            if r_no_cost and r_with_cost:
                drag = r_no_cost["sharpe"] - r_with_cost["sharpe"]
                asset_results.append({
                    "sensitivity": s,
                    "sharpe_no_cost": r_no_cost["sharpe"],
                    "sharpe_with_cost": r_with_cost["sharpe"],
                    "cost_drag_sharpe": round(drag, 2),
                    "ann_ret_pct": r_with_cost["ann_ret_pct"],
                    "max_dd_pct": r_with_cost["max_dd_pct"],
                })
                print(f"    {s:>6.1f} {r_with_cost['sharpe']:>8.2f} "
                      f"{r_with_cost['ann_ret_pct']:>8.2f} {r_with_cost['max_dd_pct']:>8.2f} "
                      f"{drag:>9.2f}")

        results[asset] = asset_results

    return results


# ═══════════════════════════════════════════════════════════════
# Section 4: Combo Order Analysis
# ═══════════════════════════════════════════════════════════════

def section_4_combo_orders():
    """Analyze Deribit combo order mechanics and cost savings."""
    print("\n  ── Section 4: Combo Order Analysis ──")

    analysis = """
    DERIBIT COMBO ORDERS:
    ─────────────────────
    Deribit supports native combo/spread orders where you submit a
    SINGLE order for the entire butterfly spread.

    Benefits:
      1. Single spread price — no legging risk
      2. Market makers quote NET price (tighter than sum of legs)
      3. Estimated savings: 30-50% of crossing cost (R76)
      4. Single fee event (possibly lower effective fee)

    With combo orders:
      R92 spread cost: 17.5 bps (monthly) / 20.0 bps (quarterly)
      30% savings:     12.3 bps / 14.0 bps
      50% savings:      8.8 bps / 10.0 bps

    Combined with maker fills (limit combo):
      Maker fee: 8 bps (4 × 2 bps)
      Spread: ~0 bps (limit order, no crossing)
      TOTAL: ~8 bps per trade

    DERIBIT RFQ (Request for Quote):
      - Available for institutional accounts
      - Market makers quote custom spreads
      - Typically tighter than order book
      - Requires min notional (~$50K)

    RECOMMENDED EXECUTION FLOW:
      1. Signal fires at 00:15 UTC
      2. Submit LIMIT combo order for BF spread at mid price
      3. Wait 1-2 hours for fill
      4. If no fill: widen by 1 tick, wait 1h more
      5. If still no fill: use RFQ for remaining
      6. Max patience: 8 hours (before next signal)
    """
    print(analysis)

    combos = {
        "individual_legs_taker": {"spread_bps": 17.5, "fee_bps": 12, "total": 29.5},
        "individual_legs_maker": {"spread_bps": 0, "fee_bps": 8, "total": 8.0},
        "combo_taker": {"spread_bps": 8.8, "fee_bps": 12, "total": 20.8},
        "combo_maker": {"spread_bps": 0, "fee_bps": 8, "total": 8.0},
        "rfq": {"spread_bps": 3.0, "fee_bps": 6, "total": 9.0},
    }

    return combos


# ═══════════════════════════════════════════════════════════════
# Section 5: Optimal Cost-Adjusted Config
# ═══════════════════════════════════════════════════════════════

def section_5_optimal_config(btc_surface, btc_dvol, eth_surface, eth_dvol):
    """Search for best z_entry × sensitivity with realistic costs."""
    print("\n  ── Section 5: Optimal Cost-Adjusted Configuration ──")

    z_entries = [1.5, 1.75, 2.0, 2.25, 2.5]
    sensitivities = [1.5, 2.0, 2.5, 3.0, 5.0]
    maker_cost = 8  # bps

    results = {}
    best_overall = None

    for asset, surface, dvol in [("BTC", btc_surface, btc_dvol),
                                   ("ETH", eth_surface, eth_dvol)]:
        print(f"\n    {asset} (maker {maker_cost}bps):")
        print(f"    {'z_entry':>8} {'sens':>6} {'Sharpe':>8} {'AnnRet%':>8} "
              f"{'MaxDD%':>8} {'Trades/yr':>10}")
        print(f"    {'─'*8} {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")

        asset_results = []
        best_sharpe = -99

        for z in z_entries:
            for s in sensitivities:
                config = {**BASE_CONFIG, "bf_z_entry": z, "bf_sensitivity": s}
                r = backtest_bf(surface, dvol, config, maker_cost)
                if r:
                    asset_results.append({
                        "z_entry": z,
                        "sensitivity": s,
                        **r,
                    })
                    if r["sharpe"] > best_sharpe:
                        best_sharpe = r["sharpe"]
                        best_config = (z, s, r)

        # Print top 10
        asset_results.sort(key=lambda x: x["sharpe"], reverse=True)
        for r in asset_results[:10]:
            marker = " ★" if r["z_entry"] == best_config[0] and r["sensitivity"] == best_config[1] else ""
            print(f"    {r['z_entry']:>8.2f} {r['sensitivity']:>6.1f} {r['sharpe']:>8.2f} "
                  f"{r['ann_ret_pct']:>8.2f} {r['max_dd_pct']:>8.2f} "
                  f"{r['trades_per_year']:>10.1f}{marker}")

        results[asset] = {
            "best": {"z_entry": best_config[0], "sensitivity": best_config[1],
                     **best_config[2]},
            "all": asset_results[:10],
        }

    return results


# ═══════════════════════════════════════════════════════════════
# Section 6: Verdict
# ═══════════════════════════════════════════════════════════════

def section_6_verdict(cost_sens, threshold, sensitivity, optimal):
    """Summarize findings."""
    print("\n  ── Section 6: Verdict & Recommendations ──")

    # Find breakeven costs
    btc_be = None
    eth_be = None
    for r in cost_sens.get("BTC", []):
        if r["sharpe"] < 1.0 and btc_be is None:
            btc_be = r["cost_bps"]
    for r in cost_sens.get("ETH", []):
        if r["sharpe"] < 1.0 and eth_be is None:
            eth_be = r["cost_bps"]

    btc_best = optimal.get("BTC", {}).get("best", {})
    eth_best = optimal.get("ETH", {}).get("best", {})

    print(f"""
    EXECUTION COST THRESHOLDS:
    ──────────────────────────
      BTC: Sharpe drops below 1.0 at ~{btc_be} bps per trade
      ETH: Sharpe drops below 1.0 at ~{eth_be} bps per trade

    OPTIMAL COST-ADJUSTED CONFIG (maker 8bps):
    ────────────────────────────────────────────
      BTC: z_entry={btc_best.get('z_entry')}, sens={btc_best.get('sensitivity')}
           Sharpe={btc_best.get('sharpe')}, Ret={btc_best.get('ann_ret_pct')}%
      ETH: z_entry={eth_best.get('z_entry')}, sens={eth_best.get('sensitivity')}
           Sharpe={eth_best.get('sharpe')}, Ret={eth_best.get('ann_ret_pct')}%

    COST REDUCTION STRATEGIES (ranked):
    ────────────────────────────────────
      1. MAKER FILLS (R92):     ~8 bps all-in (limit orders, patience)
      2. COMBO ORDERS:          50% spread reduction → effective ~8-12 bps
      3. HIGHER z_entry:        Fewer trades, but may miss smaller moves
      4. LOWER sensitivity:     Same Sharpe, lower absolute cost drag
      5. RFQ:                   ~9 bps (institutional only, min $50K)

    PRODUCTION RECOMMENDATION:
    ──────────────────────────
      → Keep z_entry=1.5 (original R69 config)
      → Use MAKER LIMIT COMBO orders
      → Effective cost: ~8 bps per trade
      → At 8 bps: BTC Sharpe {cost_sens['BTC'][4]['sharpe'] if len(cost_sens.get('BTC', [])) > 4 else 'N/A'},
                   ETH Sharpe {cost_sens['ETH'][4]['sharpe'] if len(cost_sens.get('ETH', [])) > 4 else 'N/A'}
      → NO parameter changes needed — execution method is the lever
    """)

    return {
        "btc_breakeven_cost_bps": btc_be,
        "eth_breakeven_cost_bps": eth_be,
        "btc_optimal": btc_best,
        "eth_optimal": eth_best,
        "recommendation": "MAKER LIMIT COMBO orders, no parameter changes",
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  R94: Execution Cost Optimization")
    print("=" * 70)

    btc_surface = load_surface("BTC")
    btc_dvol = load_dvol("BTC")
    eth_surface = load_surface("ETH")
    eth_dvol = load_dvol("ETH")

    all_results = {}

    # Section 1: Cost sensitivity
    cost_sens = section_1_cost_sensitivity(btc_surface, btc_dvol, eth_surface, eth_dvol)
    all_results["cost_sensitivity"] = cost_sens

    # Section 2: Entry threshold
    threshold = section_2_entry_threshold(btc_surface, btc_dvol, eth_surface, eth_dvol)
    all_results["entry_threshold"] = threshold

    # Section 3: Sensitivity
    sensitivity = section_3_sensitivity(btc_surface, btc_dvol, eth_surface, eth_dvol)
    all_results["sensitivity"] = sensitivity

    # Section 4: Combo orders
    combos = section_4_combo_orders()
    all_results["combo_orders"] = combos

    # Section 5: Optimal config
    optimal = section_5_optimal_config(btc_surface, btc_dvol, eth_surface, eth_dvol)
    all_results["optimal_config"] = optimal

    # Section 6: Verdict
    verdict = section_6_verdict(cost_sens, threshold, sensitivity, optimal)
    all_results["verdict"] = verdict

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
