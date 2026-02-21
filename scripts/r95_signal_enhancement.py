#!/usr/bin/env python3
"""
R95: BF Signal Enhancement — Can We Improve Entry Timing?
==========================================================

R94 showed execution costs eat ~30-50% of BF returns. Instead of just
reducing costs, can we improve the SIGNAL to generate more return per trade?

Tests whether combining BF with other surface features improves timing:
  1. Skew confirmation (trade BF only when skew confirms)
  2. Term spread filter (trade only in specific term structure regime)
  3. IV level filter (trade only when IV is above threshold)
  4. BF rate-of-change (trend confirmation before entry)
  5. Multi-feature z-score (composite signal)
  6. Asymmetric thresholds (different entry for long vs short)

IMPORTANT: R68/R73/R74 already showed multi-feature ensembles HURT BF.
This study tests FILTERS (don't take the trade) not ensembles (blend signals).
"""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SURFACE_DIR = ROOT / "data" / "cache" / "deribit" / "real_surface"
DVOL_DIR = ROOT / "data" / "cache" / "deribit" / "dvol"
OUTPUT_PATH = SURFACE_DIR / "r95_signal_enhancement.json"

BASE_CONFIG = {
    "bf_lookback": 120,
    "bf_z_entry": 1.5,
    "bf_z_exit": 0.0,
    "bf_sensitivity": 2.5,
}


def load_surface(asset):
    path = SURFACE_DIR / f"{asset}_daily_surface.csv"
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                data[row["date"]] = {
                    "butterfly_25d": float(row["butterfly_25d"]),
                    "iv_atm": float(row["iv_atm"]),
                    "skew_25d": float(row["skew_25d"]),
                    "term_spread": float(row["term_spread"]),
                    "spot": float(row["spot"]),
                }
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


def rolling_stats(values, lookback):
    """Compute rolling mean and std."""
    if len(values) < lookback:
        return None, None
    window = values[-lookback:]
    mean = sum(window) / len(window)
    std = math.sqrt(sum((v - mean) ** 2 for v in window) / len(window))
    return mean, std


def backtest_with_filter(surface, dvol, config, filter_fn=None, cost_bps=8):
    """BF backtest with optional entry filter.
    filter_fn(date_idx, all_dates, surface, dvol, state) → True (allow trade) or False (skip).
    """
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
    n_filtered = 0
    daily_pnls = []

    # State for filters
    state = {"bf_history": [], "skew_history": [], "ts_history": [], "iv_history": []}

    for i in range(lb, len(all_dates)):
        d = all_dates[i]
        dp = all_dates[i - 1]

        # Compute BF z-score
        window = [bf_vals[all_dates[j]] for j in range(i - lb, i)]
        bf_mean = sum(window) / len(window)
        bf_std = math.sqrt(sum((v - bf_mean) ** 2 for v in window) / len(window))
        if bf_std < 1e-8:
            continue

        z = (bf_vals[d] - bf_mean) / bf_std

        # Update state
        state["bf_history"].append(bf_vals[d])
        state["skew_history"].append(surface[d].get("skew_25d", 0))
        state["ts_history"].append(surface[d].get("term_spread", 0))
        state["iv_history"].append(dvol.get(d, 0))
        state["z"] = z
        state["position"] = position

        # Base signal
        old_pos = position
        signal_z = z  # raw z triggers

        if signal_z > z_entry:
            new_pos = -1.0
        elif signal_z < -z_entry:
            new_pos = 1.0
        else:
            new_pos = position

        # Apply filter if trade would happen
        trade_cost = 0
        if new_pos != old_pos:
            if filter_fn and not filter_fn(i, all_dates, surface, dvol, state):
                n_filtered += 1
                new_pos = old_pos  # don't take the trade
            else:
                n_trades += 1
                trade_cost = cost_bps / 10000.0

        position = new_pos

        # Daily P&L
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
        "n_filtered": n_filtered,
        "trades_per_year": round(trades_yr, 1),
        "n_days": n_days,
    }


# ═══════════════════════════════════════════════════════════════
# Filter definitions
# ═══════════════════════════════════════════════════════════════

def make_skew_filter(lb=60, threshold=0.5):
    """Only trade when skew z-score confirms BF direction."""
    def filter_fn(i, all_dates, surface, dvol, state):
        skew_hist = state["skew_history"]
        if len(skew_hist) < lb:
            return True
        window = skew_hist[-lb:]
        mean = sum(window) / len(window)
        std = math.sqrt(sum((v - mean) ** 2 for v in window) / len(window))
        if std < 1e-8:
            return True
        skew_z = (skew_hist[-1] - mean) / std
        bf_z = state["z"]
        # Skew confirms: when BF is high (short), skew should also be elevated
        # When BF is low (long), skew should be depressed
        if bf_z > 0 and skew_z > threshold:
            return True
        if bf_z < 0 and skew_z < -threshold:
            return True
        return False
    return filter_fn


def make_iv_filter(min_iv=0.40):
    """Only trade when IV is above threshold (higher IV = more carry)."""
    def filter_fn(i, all_dates, surface, dvol, state):
        iv = state["iv_history"][-1] if state["iv_history"] else 0
        return iv >= min_iv
    return filter_fn


def make_ts_filter(lb=60, threshold=0.5):
    """Only trade when term structure is in specific regime."""
    def filter_fn(i, all_dates, surface, dvol, state):
        ts_hist = state["ts_history"]
        if len(ts_hist) < lb:
            return True
        window = ts_hist[-lb:]
        mean = sum(window) / len(window)
        std = math.sqrt(sum((v - mean) ** 2 for v in window) / len(window))
        if std < 1e-8:
            return True
        ts_z = (ts_hist[-1] - mean) / std
        # Only trade when term structure is NOT extreme
        return abs(ts_z) < threshold
    return filter_fn


def make_bf_roc_filter(days=5, threshold=0.0):
    """Only trade when BF is moving in the right direction (momentum confirmation)."""
    def filter_fn(i, all_dates, surface, dvol, state):
        bf_hist = state["bf_history"]
        if len(bf_hist) < days + 1:
            return True
        roc = bf_hist[-1] - bf_hist[-1 - days]
        bf_z = state["z"]
        # Short BF: want BF to be rising (momentum continues)
        # Long BF: want BF to be falling
        if bf_z > 0 and roc > threshold:
            return True
        if bf_z < 0 and roc < -threshold:
            return True
        return False
    return filter_fn


def make_iv_roc_filter(days=10, threshold=0.0):
    """Only trade when IV is rising (more carry opportunity)."""
    def filter_fn(i, all_dates, surface, dvol, state):
        iv_hist = state["iv_history"]
        if len(iv_hist) < days + 1:
            return True
        roc = iv_hist[-1] - iv_hist[-1 - days]
        return roc > threshold
    return filter_fn


def make_combo_filter(iv_min=0.40, skew_lb=60, skew_thresh=0.3):
    """Combined: IV > threshold AND skew confirms."""
    def filter_fn(i, all_dates, surface, dvol, state):
        # IV check
        iv = state["iv_history"][-1] if state["iv_history"] else 0
        if iv < iv_min:
            return False
        # Skew check
        skew_hist = state["skew_history"]
        if len(skew_hist) < skew_lb:
            return True
        window = skew_hist[-skew_lb:]
        mean = sum(window) / len(window)
        std = math.sqrt(sum((v - mean) ** 2 for v in window) / len(window))
        if std < 1e-8:
            return True
        skew_z = (skew_hist[-1] - mean) / std
        bf_z = state["z"]
        if bf_z > 0 and skew_z > skew_thresh:
            return True
        if bf_z < 0 and skew_z < -skew_thresh:
            return True
        return False
    return filter_fn


# ═══════════════════════════════════════════════════════════════
# Main Analysis
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  R95: BF Signal Enhancement — Can We Improve Entry Timing?")
    print("=" * 70)

    all_results = {}

    for asset in ["BTC", "ETH"]:
        surface = load_surface(asset)
        dvol = load_dvol(asset)

        print(f"\n  ═══ {asset} ═══")

        # Baseline (no filter, with 8bps cost)
        baseline = backtest_with_filter(surface, dvol, BASE_CONFIG, None, cost_bps=8)
        print(f"\n    BASELINE (no filter, 8bps cost):")
        print(f"      Sharpe={baseline['sharpe']}, Ret={baseline['ann_ret_pct']}%, "
              f"Trades/yr={baseline['trades_per_year']}, Filtered=0")

        # Test filters
        filters = {
            "skew_confirm_0.3": make_skew_filter(60, 0.3),
            "skew_confirm_0.5": make_skew_filter(60, 0.5),
            "skew_confirm_1.0": make_skew_filter(60, 1.0),
            "iv_min_0.35": make_iv_filter(0.35),
            "iv_min_0.40": make_iv_filter(0.40),
            "iv_min_0.50": make_iv_filter(0.50),
            "ts_neutral_1.0": make_ts_filter(60, 1.0),
            "ts_neutral_1.5": make_ts_filter(60, 1.5),
            "bf_roc_5d": make_bf_roc_filter(5, 0.0),
            "bf_roc_10d": make_bf_roc_filter(10, 0.0),
            "iv_roc_10d": make_iv_roc_filter(10, 0.0),
            "combo_iv40_skew0.3": make_combo_filter(0.40, 60, 0.3),
        }

        print(f"\n    {'Filter':<25} {'Sharpe':>8} {'Ret%':>8} {'Trades/yr':>10} "
              f"{'Filtered':>9} {'Δ Sharpe':>9}")
        print(f"    {'─'*25} {'─'*8} {'─'*8} {'─'*10} {'─'*9} {'─'*9}")

        filter_results = {}
        for name, fn in filters.items():
            r = backtest_with_filter(surface, dvol, BASE_CONFIG, fn, cost_bps=8)
            if r:
                delta = r["sharpe"] - baseline["sharpe"]
                filter_results[name] = {**r, "delta_sharpe": round(delta, 2)}
                marker = " ★" if delta > 0.1 else " ✗" if delta < -0.1 else ""
                print(f"    {name:<25} {r['sharpe']:>8.2f} {r['ann_ret_pct']:>8.2f} "
                      f"{r['trades_per_year']:>10.1f} {r['n_filtered']:>9} "
                      f"{delta:>+9.2f}{marker}")

        # Count improvements
        improved = sum(1 for v in filter_results.values() if v["delta_sharpe"] > 0)
        total = len(filter_results)

        all_results[asset] = {
            "baseline": baseline,
            "filters": filter_results,
            "improved_count": improved,
            "total_tested": total,
        }

        print(f"\n    SUMMARY: {improved}/{total} filters improve Sharpe")

    # Verdict
    print(f"\n  ═══ VERDICT ═══")
    btc_improved = all_results.get("BTC", {}).get("improved_count", 0)
    eth_improved = all_results.get("ETH", {}).get("improved_count", 0)
    btc_total = all_results.get("BTC", {}).get("total_tested", 0)
    eth_total = all_results.get("ETH", {}).get("total_tested", 0)

    # Best filter per asset
    for asset in ["BTC", "ETH"]:
        filters = all_results.get(asset, {}).get("filters", {})
        if filters:
            best = max(filters.items(), key=lambda x: x[1]["delta_sharpe"])
            worst = min(filters.items(), key=lambda x: x[1]["delta_sharpe"])
            print(f"    {asset} best:  {best[0]} (Δ={best[1]['delta_sharpe']:+.2f})")
            print(f"    {asset} worst: {worst[0]} (Δ={worst[1]['delta_sharpe']:+.2f})")

    # R68/R73/R74 parallel check
    any_robust = False
    for asset in ["BTC", "ETH"]:
        for name, data in all_results.get(asset, {}).get("filters", {}).items():
            if data["delta_sharpe"] > 0.3:  # meaningful improvement
                any_robust = True

    verdict = "FILTERS MARGINAL" if not any_robust else "POTENTIAL IMPROVEMENT FOUND"

    print(f"""
    Overall: BTC {btc_improved}/{btc_total}, ETH {eth_improved}/{eth_total} filters improve

    VERDICT: {verdict}
    {'→ Consistent with R68/R73/R74: BF standalone is hard to beat' if not any_robust else '→ Investigate further with walk-forward validation'}
    {'→ 11th STATIC > DYNAMIC confirmation' if not any_robust else ''}
    """)

    all_results["verdict"] = {
        "btc_improved": btc_improved,
        "eth_improved": eth_improved,
        "any_robust": any_robust,
        "conclusion": verdict,
    }

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results saved: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
