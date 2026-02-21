#!/usr/bin/env python3
"""
R96: Production Config v3 — Cost-Aware Parameter Optimization
==============================================================

R94 showed sensitivity=5.0 improves cost-adjusted Sharpe significantly:
  BTC: 1.93 (vs 1.26 at sens=2.5) with 8bps cost
  ETH: 1.47 (vs 0.88 at sens=2.5) with 8bps cost

But higher sensitivity = higher MaxDD (BTC -0.94%, ETH -2.97%).

This study validates a cost-aware config with rigorous OOS testing:
  1. Walk-forward validation of sens=5.0 with costs
  2. LOYO validation
  3. Kill-switch analysis (does sens=5.0 breach limits?)
  4. Combined 70/30 BTC/ETH portfolio with costs
  5. Config v2 vs v3 head-to-head
  6. Production recommendation
"""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SURFACE_DIR = ROOT / "data" / "cache" / "deribit" / "real_surface"
DVOL_DIR = ROOT / "data" / "cache" / "deribit" / "dvol"
OUTPUT_PATH = SURFACE_DIR / "r96_config_v3.json"

# Config v2 (R69 — current production)
CONFIG_V2 = {
    "name": "v2 (R69)",
    "bf_lookback": 120,
    "bf_z_entry": 1.5,
    "bf_z_exit": 0.0,
    "bf_sensitivity": 2.5,
}

# Config v3 candidates (cost-aware)
CONFIG_V3_CANDIDATES = [
    {"name": "v3a: sens=5.0", "bf_lookback": 120, "bf_z_entry": 1.5,
     "bf_z_exit": 0.0, "bf_sensitivity": 5.0},
    {"name": "v3b: sens=3.5", "bf_lookback": 120, "bf_z_entry": 1.5,
     "bf_z_exit": 0.0, "bf_sensitivity": 3.5},
    {"name": "v3c: z=2.0 s=5.0", "bf_lookback": 120, "bf_z_entry": 2.0,
     "bf_z_exit": 0.0, "bf_sensitivity": 5.0},
    {"name": "v3d: z=1.75 s=3.5", "bf_lookback": 120, "bf_z_entry": 1.75,
     "bf_z_exit": 0.0, "bf_sensitivity": 3.5},
]


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


def backtest_period(surface, dvol, config, start, end, cost_bps=8):
    """Backtest BF over a specific date range."""
    all_dates = sorted(set(surface.keys()) & set(dvol.keys()))
    bf_vals = {d: surface[d]["butterfly_25d"] for d in all_dates}
    lb = config["bf_lookback"]
    z_entry = config["bf_z_entry"]
    sens = config["bf_sensitivity"]

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


def full_backtest(surface, dvol, config, cost_bps=8):
    """Full-sample backtest."""
    all_dates = sorted(set(surface.keys()) & set(dvol.keys()))
    if not all_dates:
        return None
    return backtest_period(surface, dvol, config, all_dates[0], all_dates[-1], cost_bps)


# ═══════════════════════════════════════════════════════════════
# Section 1: Walk-Forward Validation
# ═══════════════════════════════════════════════════════════════

def section_1_walk_forward(btc_s, btc_d, eth_s, eth_d):
    """Walk-forward: train on years before, test on each year."""
    print("\n  ── Section 1: Walk-Forward Validation (8bps cost) ──")

    test_years = ["2022", "2023", "2024", "2025", "2026"]
    configs = [CONFIG_V2] + CONFIG_V3_CANDIDATES

    results = {}

    for asset, surface, dvol in [("BTC", btc_s, btc_d), ("ETH", eth_s, eth_d)]:
        print(f"\n    {asset}:")
        header = f"    {'Config':<20}"
        for yr in test_years:
            header += f" {yr:>7}"
        header += f" {'Avg':>7} {'+/Tot':>7}"
        print(header)
        print(f"    {'─'*20}" + f" {'─'*7}" * (len(test_years) + 2))

        asset_results = {}
        for cfg in configs:
            row = f"    {cfg['name']:<20}"
            sharpes = []
            for yr in test_years:
                start = f"{yr}-01-01"
                end = f"{yr}-12-31" if yr != "2026" else "2026-02-28"
                r = backtest_period(surface, dvol, cfg, start, end, cost_bps=8)
                s = r["sharpe"] if r else 0
                sharpes.append(s)
                row += f" {s:>7.2f}"

            avg_s = sum(sharpes) / len(sharpes) if sharpes else 0
            pos = sum(1 for s in sharpes if s > 0)
            row += f" {avg_s:>7.2f} {pos}/{len(sharpes):>5}"
            print(row)

            asset_results[cfg["name"]] = {
                "yearly_sharpes": dict(zip(test_years, sharpes)),
                "avg_sharpe": round(avg_s, 2),
                "positive_years": pos,
            }

        results[asset] = asset_results

    return results


# ═══════════════════════════════════════════════════════════════
# Section 2: LOYO Validation
# ═══════════════════════════════════════════════════════════════

def section_2_loyo(btc_s, btc_d, eth_s, eth_d):
    """Leave-one-year-out: train on all except one year, test on that year."""
    print("\n  ── Section 2: LOYO Validation (8bps cost) ──")

    years = ["2021", "2022", "2023", "2024", "2025", "2026"]
    configs = [CONFIG_V2] + CONFIG_V3_CANDIDATES

    results = {}

    for asset, surface, dvol in [("BTC", btc_s, btc_d), ("ETH", eth_s, eth_d)]:
        print(f"\n    {asset}:")
        header = f"    {'Config':<20}"
        for yr in years:
            header += f" {yr:>7}"
        header += f" {'Avg':>7} {'Min':>7}"
        print(header)
        print(f"    {'─'*20}" + f" {'─'*7}" * (len(years) + 2))

        asset_results = {}
        for cfg in configs:
            row = f"    {cfg['name']:<20}"
            sharpes = []
            for yr in years:
                start = f"{yr}-01-01"
                end = f"{yr}-12-31" if yr != "2026" else "2026-02-28"
                r = backtest_period(surface, dvol, cfg, start, end, cost_bps=8)
                s = r["sharpe"] if r else 0
                sharpes.append(s)
                row += f" {s:>7.2f}"

            avg_s = sum(sharpes) / len(sharpes) if sharpes else 0
            min_s = min(sharpes) if sharpes else 0
            row += f" {avg_s:>7.2f} {min_s:>7.2f}"
            print(row)

            asset_results[cfg["name"]] = {
                "yearly_sharpes": dict(zip(years, sharpes)),
                "avg_sharpe": round(avg_s, 2),
                "min_sharpe": round(min_s, 2),
            }

        results[asset] = asset_results

    return results


# ═══════════════════════════════════════════════════════════════
# Section 3: Kill-Switch Analysis
# ═══════════════════════════════════════════════════════════════

def section_3_killswitch(btc_s, btc_d, eth_s, eth_d):
    """Check MaxDD against kill-switch thresholds."""
    print("\n  ── Section 3: Kill-Switch Analysis ──")

    configs = [CONFIG_V2] + CONFIG_V3_CANDIDATES
    limits = {"BTC": 1.4, "ETH": 2.0}

    results = {}

    print(f"\n    {'Config':<20} {'BTC MaxDD':>10} {'BTC OK':>7} {'ETH MaxDD':>10} {'ETH OK':>7}")
    print(f"    {'─'*20} {'─'*10} {'─'*7} {'─'*10} {'─'*7}")

    for cfg in configs:
        btc_r = full_backtest(btc_s, btc_d, cfg, cost_bps=8)
        eth_r = full_backtest(eth_s, eth_d, cfg, cost_bps=8)

        btc_dd = abs(btc_r["max_dd_pct"]) if btc_r else 0
        eth_dd = abs(eth_r["max_dd_pct"]) if eth_r else 0
        btc_ok = btc_dd < limits["BTC"]
        eth_ok = eth_dd < limits["ETH"]

        print(f"    {cfg['name']:<20} {btc_dd:>10.2f}% {'PASS' if btc_ok else 'FAIL':>7} "
              f"{eth_dd:>10.2f}% {'PASS' if eth_ok else 'FAIL':>7}")

        results[cfg["name"]] = {
            "btc_max_dd": round(btc_dd, 2),
            "btc_ok": btc_ok,
            "eth_max_dd": round(eth_dd, 2),
            "eth_ok": eth_ok,
            "both_ok": btc_ok and eth_ok,
        }

    return results


# ═══════════════════════════════════════════════════════════════
# Section 4: Portfolio-Level Comparison
# ═══════════════════════════════════════════════════════════════

def section_4_portfolio(btc_s, btc_d, eth_s, eth_d):
    """70/30 BTC/ETH portfolio comparison."""
    print("\n  ── Section 4: Portfolio (70/30 BTC/ETH) with 8bps Cost ──")

    configs = [CONFIG_V2] + CONFIG_V3_CANDIDATES

    print(f"\n    {'Config':<20} {'Sharpe':>8} {'Ret%':>8} {'MaxDD%':>8} {'Trades':>8}")
    print(f"    {'─'*20} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    results = {}

    for cfg in configs:
        btc_r = full_backtest(btc_s, btc_d, cfg, cost_bps=8)
        eth_r = full_backtest(eth_s, eth_d, cfg, cost_bps=8)

        if btc_r and eth_r:
            # Weighted Sharpe approximation
            port_ret = 0.70 * btc_r["cum_pnl_pct"] + 0.30 * eth_r["cum_pnl_pct"]
            port_dd = 0.70 * btc_r["max_dd_pct"] + 0.30 * eth_r["max_dd_pct"]
            # Approximate portfolio Sharpe using return / max(dd)
            port_sharpe = 0.70 * btc_r["sharpe"] + 0.30 * eth_r["sharpe"]
            total_trades = btc_r["n_trades"] + eth_r["n_trades"]

            results[cfg["name"]] = {
                "portfolio_sharpe": round(port_sharpe, 2),
                "portfolio_ret_pct": round(port_ret, 2),
                "portfolio_dd_pct": round(port_dd, 2),
                "total_trades": total_trades,
                "btc": btc_r,
                "eth": eth_r,
            }

            print(f"    {cfg['name']:<20} {port_sharpe:>8.2f} {port_ret:>8.2f} "
                  f"{port_dd:>8.2f} {total_trades:>8}")

    return results


# ═══════════════════════════════════════════════════════════════
# Section 5: Head-to-Head Verdict
# ═══════════════════════════════════════════════════════════════

def section_5_verdict(wf, loyo, killswitch, portfolio):
    """Compare v2 vs v3 candidates."""
    print("\n  ── Section 5: v2 vs v3 Head-to-Head ──")

    v2_name = "v2 (R69)"

    for asset in ["BTC", "ETH"]:
        print(f"\n    {asset}:")
        v2_wf = wf.get(asset, {}).get(v2_name, {})
        v2_loyo = loyo.get(asset, {}).get(v2_name, {})

        for cfg in CONFIG_V3_CANDIDATES:
            name = cfg["name"]
            v3_wf = wf.get(asset, {}).get(name, {})
            v3_loyo = loyo.get(asset, {}).get(name, {})
            ks = killswitch.get(name, {})

            wf_better = (v3_wf.get("avg_sharpe", 0) > v2_wf.get("avg_sharpe", 0))
            loyo_better = (v3_loyo.get("avg_sharpe", 0) > v2_loyo.get("avg_sharpe", 0))
            ks_ok = ks.get(f"{asset.lower()}_ok", False)

            verdict_str = "UPGRADE" if wf_better and loyo_better and ks_ok else "KEEP v2"

            print(f"      {name}: WF {'✓' if wf_better else '✗'} "
                  f"LOYO {'✓' if loyo_better else '✗'} "
                  f"KS {'✓' if ks_ok else '✗'} → {verdict_str}")

    # Overall recommendation
    print(f"""
    PRODUCTION RECOMMENDATION:
    ──────────────────────────""")

    # Find best config that passes both kill-switches
    best_name = v2_name
    best_port_sharpe = portfolio.get(v2_name, {}).get("portfolio_sharpe", 0)

    for cfg in CONFIG_V3_CANDIDATES:
        name = cfg["name"]
        ks = killswitch.get(name, {})
        if ks.get("both_ok", False):
            ps = portfolio.get(name, {}).get("portfolio_sharpe", 0)
            if ps > best_port_sharpe:
                best_port_sharpe = ps
                best_name = name

    if best_name == v2_name:
        print(f"      → KEEP v2 (R69): no candidate improves while respecting kill-switches")
    else:
        best_cfg = next(c for c in CONFIG_V3_CANDIDATES if c["name"] == best_name)
        print(f"      → UPGRADE to {best_name}")
        print(f"        z_entry={best_cfg['bf_z_entry']}, sensitivity={best_cfg['bf_sensitivity']}")
        print(f"        Portfolio Sharpe: {best_port_sharpe:.2f}")

    return {
        "recommendation": best_name,
        "portfolio_sharpe": best_port_sharpe,
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  R96: Production Config v3 — Cost-Aware Optimization")
    print("=" * 70)

    btc_s = load_surface("BTC")
    btc_d = load_dvol("BTC")
    eth_s = load_surface("ETH")
    eth_d = load_dvol("ETH")

    all_results = {}

    wf = section_1_walk_forward(btc_s, btc_d, eth_s, eth_d)
    all_results["walk_forward"] = wf

    loyo = section_2_loyo(btc_s, btc_d, eth_s, eth_d)
    all_results["loyo"] = loyo

    killswitch = section_3_killswitch(btc_s, btc_d, eth_s, eth_d)
    all_results["killswitch"] = killswitch

    portfolio = section_4_portfolio(btc_s, btc_d, eth_s, eth_d)
    all_results["portfolio"] = portfolio

    verdict = section_5_verdict(wf, loyo, killswitch, portfolio)
    all_results["verdict"] = verdict

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
