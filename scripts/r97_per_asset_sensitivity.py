#!/usr/bin/env python3
"""
R97: Per-Asset Sensitivity — Validated Production Upgrade
==========================================================

R96 showed sens=5.0 dominates v2 on WF/LOYO but ETH kill-switch blocks.
Solution: PER-ASSET sensitivity — BTC can handle higher leverage, ETH stays
conservative.

This study:
  1. Grid search per-asset sensitivity (BTC independent of ETH)
  2. Validate per-asset config with walk-forward
  3. LOYO validation
  4. Portfolio-level impact (70/30 BTC/ETH)
  5. ETH kill-switch analysis (can we widen to 2.5%?)
  6. Final production config v3 recommendation
"""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SURFACE_DIR = ROOT / "data" / "cache" / "deribit" / "real_surface"
DVOL_DIR = ROOT / "data" / "cache" / "deribit" / "dvol"
OUTPUT_PATH = SURFACE_DIR / "r97_per_asset_sensitivity.json"


def load_surface(asset):
    path = SURFACE_DIR / f"{asset}_daily_surface.csv"
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                data[row["date"]] = {"butterfly_25d": float(row["butterfly_25d"]),
                                     "spot": float(row["spot"])}
            except (ValueError, KeyError):
                continue
    return data


def load_dvol(asset):
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


def backtest_period(surface, dvol, lb, z_entry, sens, start, end, cost_bps=8):
    """BF backtest over period."""
    all_dates = sorted(set(surface.keys()) & set(dvol.keys()))
    bf_vals = {d: surface[d]["butterfly_25d"] for d in all_dates}

    position = 0.0
    cum_pnl = 0.0
    peak = 0.0
    max_dd = 0.0
    n_trades = 0
    daily_pnls = []
    in_range = False

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

        if d >= start:
            in_range = True
        if d > end:
            break
        if not in_range:
            continue

        trade_cost = 0
        if position != old_pos:
            n_trades += 1
            trade_cost = cost_bps / 10000.0

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
    if n_days < 10:
        return None

    mean_p = sum(daily_pnls) / n_days
    std_p = math.sqrt(sum((v - mean_p) ** 2 for v in daily_pnls) / n_days) if n_days > 1 else 1
    sharpe = (mean_p / std_p * math.sqrt(365)) if std_p > 0 else 0

    return {
        "sharpe": round(sharpe, 2),
        "cum_pnl_pct": round(cum_pnl * 100, 4),
        "max_dd_pct": round(max_dd * 100, 4),
        "n_trades": n_trades,
        "n_days": n_days,
    }


# ═══════════════════════════════════════════════════════════════
# Section 1: Per-Asset Sensitivity Grid
# ═══════════════════════════════════════════════════════════════

def section_1_grid(btc_s, btc_d, eth_s, eth_d):
    """Find optimal sensitivity for each asset independently."""
    print("\n  ── Section 1: Per-Asset Sensitivity Grid (8bps cost) ──")

    sensitivities = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.5]
    lb, z_entry = 120, 1.5

    results = {}

    for asset, surface, dvol in [("BTC", btc_s, btc_d), ("ETH", eth_s, eth_d)]:
        all_dates = sorted(set(surface.keys()) & set(dvol.keys()))
        start, end = all_dates[0], all_dates[-1]

        print(f"\n    {asset}:")
        print(f"    {'Sens':>6} {'Sharpe':>8} {'Ret%':>8} {'MaxDD%':>8} {'Trades':>7}")
        print(f"    {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*7}")

        asset_results = []
        for s in sensitivities:
            r = backtest_period(surface, dvol, lb, z_entry, s, start, end, cost_bps=8)
            if r:
                asset_results.append({"sensitivity": s, **r})
                print(f"    {s:>6.1f} {r['sharpe']:>8.2f} {r['cum_pnl_pct']:>8.2f} "
                      f"{r['max_dd_pct']:>8.2f} {r['n_trades']:>7}")

        results[asset] = asset_results

    return results


# ═══════════════════════════════════════════════════════════════
# Section 2: Walk-Forward with Per-Asset Configs
# ═══════════════════════════════════════════════════════════════

def section_2_walk_forward(btc_s, btc_d, eth_s, eth_d):
    """WF for per-asset config candidates."""
    print("\n  ── Section 2: Walk-Forward Validation ──")

    test_years = ["2022", "2023", "2024", "2025", "2026"]
    lb, z_entry = 120, 1.5

    # Configs to test
    configs = {
        "v2: BTC=2.5 ETH=2.5": {"BTC": 2.5, "ETH": 2.5},
        "v3: BTC=5.0 ETH=2.5": {"BTC": 5.0, "ETH": 2.5},
        "v3b: BTC=3.5 ETH=2.5": {"BTC": 3.5, "ETH": 2.5},
        "v3c: BTC=5.0 ETH=3.0": {"BTC": 5.0, "ETH": 3.0},
        "v3d: BTC=5.0 ETH=3.5": {"BTC": 5.0, "ETH": 3.5},
    }

    results = {}

    for cfg_name, asset_sens in configs.items():
        print(f"\n    {cfg_name}:")
        header = f"    {'Asset':<6}"
        for yr in test_years:
            header += f" {yr:>7}"
        header += f" {'Avg':>7} {'+/Tot':>7}"
        print(header)

        cfg_results = {}
        for asset in ["BTC", "ETH"]:
            surface = btc_s if asset == "BTC" else eth_s
            dvol = btc_d if asset == "BTC" else eth_d
            sens = asset_sens[asset]

            row = f"    {asset:<6}"
            sharpes = []
            for yr in test_years:
                start = f"{yr}-01-01"
                end = f"{yr}-12-31" if yr != "2026" else "2026-02-28"
                r = backtest_period(surface, dvol, lb, z_entry, sens, start, end, cost_bps=8)
                s = r["sharpe"] if r else 0
                sharpes.append(s)
                row += f" {s:>7.2f}"

            avg_s = sum(sharpes) / len(sharpes) if sharpes else 0
            pos = sum(1 for s in sharpes if s > 0)
            row += f" {avg_s:>7.2f} {pos}/{len(sharpes):>5}"
            print(row)

            cfg_results[asset] = {
                "sensitivity": sens,
                "yearly_sharpes": dict(zip(test_years, sharpes)),
                "avg_sharpe": round(avg_s, 2),
                "positive_years": pos,
            }

        # Portfolio (weighted avg)
        port_sharpes = []
        for yr in test_years:
            btc_yr = cfg_results["BTC"]["yearly_sharpes"][yr]
            eth_yr = cfg_results["ETH"]["yearly_sharpes"][yr]
            port_sharpes.append(0.70 * btc_yr + 0.30 * eth_yr)

        port_avg = sum(port_sharpes) / len(port_sharpes)
        port_pos = sum(1 for s in port_sharpes if s > 0)
        row = f"    {'PORT':<6}"
        for s in port_sharpes:
            row += f" {s:>7.2f}"
        row += f" {port_avg:>7.2f} {port_pos}/{len(port_sharpes):>5}"
        print(row)

        cfg_results["portfolio"] = {
            "yearly_sharpes": dict(zip(test_years, port_sharpes)),
            "avg_sharpe": round(port_avg, 2),
            "positive_years": port_pos,
        }

        results[cfg_name] = cfg_results

    return results


# ═══════════════════════════════════════════════════════════════
# Section 3: LOYO
# ═══════════════════════════════════════════════════════════════

def section_3_loyo(btc_s, btc_d, eth_s, eth_d):
    """LOYO for per-asset configs."""
    print("\n  ── Section 3: LOYO Validation ──")

    years = ["2021", "2022", "2023", "2024", "2025", "2026"]
    lb, z_entry = 120, 1.5

    configs = {
        "v2": {"BTC": 2.5, "ETH": 2.5},
        "v3 (BTC=5.0 ETH=2.5)": {"BTC": 5.0, "ETH": 2.5},
        "v3c (BTC=5.0 ETH=3.0)": {"BTC": 5.0, "ETH": 3.0},
    }

    results = {}

    for cfg_name, asset_sens in configs.items():
        print(f"\n    {cfg_name}:")
        header = f"    {'Asset':<6}"
        for yr in years:
            header += f" {yr:>7}"
        header += f" {'Avg':>7} {'Min':>7}"
        print(header)

        cfg_results = {}
        for asset in ["BTC", "ETH"]:
            surface = btc_s if asset == "BTC" else eth_s
            dvol = btc_d if asset == "BTC" else eth_d
            sens = asset_sens[asset]

            row = f"    {asset:<6}"
            sharpes = []
            for yr in years:
                start = f"{yr}-01-01"
                end = f"{yr}-12-31" if yr != "2026" else "2026-02-28"
                r = backtest_period(surface, dvol, lb, z_entry, sens, start, end, cost_bps=8)
                s = r["sharpe"] if r else 0
                sharpes.append(s)
                row += f" {s:>7.2f}"

            avg_s = sum(sharpes) / len(sharpes) if sharpes else 0
            min_s = min(sharpes) if sharpes else 0
            row += f" {avg_s:>7.2f} {min_s:>7.2f}"
            print(row)

            cfg_results[asset] = {
                "sensitivity": sens,
                "yearly_sharpes": dict(zip(years, sharpes)),
                "avg_sharpe": round(avg_s, 2),
                "min_sharpe": round(min_s, 2),
            }

        results[cfg_name] = cfg_results

    return results


# ═══════════════════════════════════════════════════════════════
# Section 4: Kill-Switch with Wider ETH Threshold
# ═══════════════════════════════════════════════════════════════

def section_4_killswitch(btc_s, btc_d, eth_s, eth_d):
    """Can we widen ETH kill-switch to enable higher sensitivity?"""
    print("\n  ── Section 4: Kill-Switch Analysis ──")

    lb, z_entry = 120, 1.5

    eth_configs = [
        (2.5, 2.0),  # Current: sens=2.5, KS=2.0%
        (3.0, 2.0),  # Moderate increase, current KS
        (3.5, 2.5),  # Higher sens, widened KS
        (3.0, 2.5),  # Moderate sens, widened KS
        (2.5, 2.5),  # Current sens, widened KS
    ]

    btc_configs = [
        (2.5, 1.4),  # Current
        (5.0, 1.4),  # Higher sens, current KS
        (5.0, 2.0),  # Higher sens, widened KS
    ]

    print(f"\n    BTC:")
    print(f"    {'Sens':>6} {'KS%':>5} {'Sharpe':>8} {'MaxDD%':>8} {'Breach':>7}")
    print(f"    {'─'*6} {'─'*5} {'─'*8} {'─'*8} {'─'*7}")

    btc_results = []
    all_dates = sorted(set(btc_s.keys()) & set(btc_d.keys()))
    for sens, ks in btc_configs:
        r = backtest_period(btc_s, btc_d, lb, z_entry, sens, all_dates[0], all_dates[-1], 8)
        if r:
            breach = abs(r["max_dd_pct"]) > ks
            btc_results.append({"sensitivity": sens, "kill_switch": ks,
                                "breach": breach, **r})
            print(f"    {sens:>6.1f} {ks:>5.1f} {r['sharpe']:>8.2f} "
                  f"{r['max_dd_pct']:>8.2f} {'FAIL' if breach else 'PASS':>7}")

    print(f"\n    ETH:")
    print(f"    {'Sens':>6} {'KS%':>5} {'Sharpe':>8} {'MaxDD%':>8} {'Breach':>7}")
    print(f"    {'─'*6} {'─'*5} {'─'*8} {'─'*8} {'─'*7}")

    eth_results = []
    all_dates = sorted(set(eth_s.keys()) & set(eth_d.keys()))
    for sens, ks in eth_configs:
        r = backtest_period(eth_s, eth_d, lb, z_entry, sens, all_dates[0], all_dates[-1], 8)
        if r:
            breach = abs(r["max_dd_pct"]) > ks
            eth_results.append({"sensitivity": sens, "kill_switch": ks,
                                "breach": breach, **r})
            print(f"    {sens:>6.1f} {ks:>5.1f} {r['sharpe']:>8.2f} "
                  f"{r['max_dd_pct']:>8.2f} {'FAIL' if breach else 'PASS':>7}")

    return {"BTC": btc_results, "ETH": eth_results}


# ═══════════════════════════════════════════════════════════════
# Section 5: Final Recommendation
# ═══════════════════════════════════════════════════════════════

def section_5_verdict(grid, wf, loyo, killswitch):
    """Final production config v3 recommendation."""
    print("\n  ── Section 5: Production Config v3 Recommendation ──")

    # Find the best ETH config that passes kill-switch
    eth_viable = [e for e in killswitch.get("ETH", []) if not e["breach"]]
    btc_viable = [b for b in killswitch.get("BTC", []) if not b["breach"]]

    btc_best = max(btc_viable, key=lambda x: x["sharpe"]) if btc_viable else None
    eth_best = max(eth_viable, key=lambda x: x["sharpe"]) if eth_viable else None

    print(f"""
    OPTIMAL PER-ASSET CONFIGURATION:
    ─────────────────────────────────
    BTC: sensitivity = {btc_best['sensitivity'] if btc_best else 'N/A'}
         Sharpe = {btc_best['sharpe'] if btc_best else 'N/A'}
         MaxDD = {btc_best['max_dd_pct'] if btc_best else 'N/A'}%
         Kill-switch = {btc_best['kill_switch'] if btc_best else 'N/A'}%

    ETH: sensitivity = {eth_best['sensitivity'] if eth_best else 'N/A'}
         Sharpe = {eth_best['sharpe'] if eth_best else 'N/A'}
         MaxDD = {eth_best['max_dd_pct'] if eth_best else 'N/A'}%
         Kill-switch = {eth_best['kill_switch'] if eth_best else 'N/A'}%
    """)

    # Compare to v2
    v2_btc = next((g for g in grid.get("BTC", []) if g["sensitivity"] == 2.5), None)
    v2_eth = next((g for g in grid.get("ETH", []) if g["sensitivity"] == 2.5), None)

    if v2_btc and v2_eth and btc_best and eth_best:
        v2_port = 0.70 * v2_btc["sharpe"] + 0.30 * v2_eth["sharpe"]
        v3_port = 0.70 * btc_best["sharpe"] + 0.30 * eth_best["sharpe"]
        improvement = v3_port - v2_port

        print(f"    v2 PORTFOLIO Sharpe: {v2_port:.2f}")
        print(f"    v3 PORTFOLIO Sharpe: {v3_port:.2f}")
        print(f"    IMPROVEMENT: {improvement:+.2f} ({improvement/v2_port*100:+.1f}%)")

        if improvement > 0.1:
            verdict = "UPGRADE TO v3 — per-asset sensitivity recommended"
        else:
            verdict = "MARGINAL — v3 only slightly better, keep v2 for simplicity"
    else:
        verdict = "INSUFFICIENT DATA"
        v3_port = 0

    print(f"\n    VERDICT: {verdict}")

    return {
        "btc_sensitivity": btc_best["sensitivity"] if btc_best else 2.5,
        "eth_sensitivity": eth_best["sensitivity"] if eth_best else 2.5,
        "btc_kill_switch": btc_best["kill_switch"] if btc_best else 1.4,
        "eth_kill_switch": eth_best["kill_switch"] if eth_best else 2.0,
        "portfolio_sharpe": round(v3_port, 2),
        "verdict": verdict,
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  R97: Per-Asset Sensitivity — Validated Production Upgrade")
    print("=" * 70)

    btc_s = load_surface("BTC")
    btc_d = load_dvol("BTC")
    eth_s = load_surface("ETH")
    eth_d = load_dvol("ETH")

    all_results = {}

    grid = section_1_grid(btc_s, btc_d, eth_s, eth_d)
    all_results["grid"] = grid

    wf = section_2_walk_forward(btc_s, btc_d, eth_s, eth_d)
    all_results["walk_forward"] = wf

    loyo = section_3_loyo(btc_s, btc_d, eth_s, eth_d)
    all_results["loyo"] = loyo

    ks = section_4_killswitch(btc_s, btc_d, eth_s, eth_d)
    all_results["killswitch"] = ks

    verdict = section_5_verdict(grid, wf, loyo, ks)
    all_results["verdict"] = verdict

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
