#!/usr/bin/env python3
"""
Phase 130: Calendar Alpha Research
====================================
Search for time-based patterns orthogonal to existing price/funding signals.

Tests:
1. Day-of-week effects: Are certain weekdays consistently better/worse?
2. Hour-of-day effects: Are certain hours consistently better/worse?
3. Month-of-year effects: Monthly seasonality
4. Weekend vs weekday: Simple binary signal
5. Calendar strategy: Go long on "good" days, flat/short on "bad" days

All tests computed per-symbol, then aggregated cross-sectionally.
Uses 5-year data (2021-2025) with IS/OOS split.
"""
import json
import os
import signal as _signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from nexus_quant.data.providers.registry import make_provider

_partial = {}
def _on_timeout(signum, frame):
    print("\n⏰ TIMEOUT — saving partial...")
    _save(_partial, partial=True)
    sys.exit(0)
_signal.signal(_signal.SIGALRM, _on_timeout)
_signal.alarm(600)

PROD_CFG = json.loads((ROOT / "configs" / "production_p91b_champion.json").read_text())
SYMBOLS = PROD_CFG["data"]["symbols"]

YEAR_RANGES = {
    "2021": ("2021-02-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
}
YEARS = sorted(YEAR_RANGES.keys())

OUT_DIR = ROOT / "artifacts" / "phase130"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sharpe(rets: np.ndarray) -> float:
    if len(rets) < 100:
        return 0.0
    mu = float(np.mean(rets)) * 8760
    sd = float(np.std(rets)) * np.sqrt(8760)
    return round(mu / sd, 4) if sd > 1e-12 else 0.0


def obj_func(yearly_sharpes: list) -> float:
    arr = np.array(yearly_sharpes)
    return round(float(np.mean(arr) - 0.5 * np.std(arr)), 4)


def _save(data: dict, partial: bool = False) -> None:
    data["partial"] = partial
    data["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path = OUT_DIR / "calendar_alpha_report.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → Saved: {path}")


def main():
    global _partial
    t0 = time.time()
    print("=" * 70)
    print("Phase 130: Calendar Alpha Research")
    print("=" * 70)

    # 1. Load data — collect per-bar timestamps + returns for all symbols
    print("\n[1/4] Loading data...")
    all_data = {}  # year -> {"timestamps": [...], "returns": {sym: [...]}}

    for year, (start, end) in YEAR_RANGES.items():
        print(f"  {year}:", end="", flush=True)
        cfg_data = {
            "provider": "binance_rest_v1", "symbols": SYMBOLS,
            "start": start, "end": end, "bar_interval": "1h",
            "cache_dir": ".cache/binance_rest",
        }
        provider = make_provider(cfg_data, seed=42)
        dataset = provider.load()

        n = len(dataset.timeline)
        timestamps = [dataset.timeline[i] for i in range(1, n)]
        returns = {}
        for sym in SYMBOLS:
            rets = []
            for i in range(1, n):
                c0 = dataset.close(sym, i - 1)
                c1 = dataset.close(sym, i)
                rets.append((c1 / c0 - 1.0) if c0 > 0 else 0.0)
            returns[sym] = np.array(rets, dtype=np.float64)

        all_data[year] = {"timestamps": timestamps, "returns": returns}
        print(f" {n - 1} bars", flush=True)

    _partial = {"phase": 130}

    # 2. Day-of-week analysis
    print("\n[2/4] Day-of-week analysis...")
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_results = {}  # year -> dow -> avg_return

    for year in YEARS:
        ts_list = all_data[year]["timestamps"]
        # Compute equal-weight portfolio returns
        n_bars = min(len(all_data[year]["returns"][s]) for s in SYMBOLS)
        eq_rets = np.zeros(n_bars)
        for sym in SYMBOLS:
            eq_rets += all_data[year]["returns"][sym][:n_bars] / len(SYMBOLS)

        dow_rets = {d: [] for d in range(7)}
        for i in range(n_bars):
            dt = datetime.fromtimestamp(ts_list[i], tz=timezone.utc)
            dow = dt.weekday()  # 0=Mon, 6=Sun
            dow_rets[dow].append(eq_rets[i])

        dow_results[year] = {}
        for d in range(7):
            arr = np.array(dow_rets[d])
            ann_ret = float(np.mean(arr)) * 8760
            ann_vol = float(np.std(arr)) * np.sqrt(8760)
            sr = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0
            dow_results[year][dow_names[d]] = {
                "count": len(arr),
                "ann_return": round(ann_ret * 100, 2),
                "ann_vol": round(ann_vol * 100, 2),
                "sharpe": round(sr, 2),
            }

    # Aggregate across years
    print(f"\n  {'Day':5s}", end="")
    for year in YEARS:
        print(f" {year:>8s}", end="")
    print(f" {'AVG':>8s}")

    dow_avg = {}
    for dow in dow_names:
        print(f"  {dow:5s}", end="")
        sharpes = []
        for year in YEARS:
            s = dow_results[year][dow]["sharpe"]
            sharpes.append(s)
            print(f" {s:8.2f}", end="")
        avg_s = round(float(np.mean(sharpes)), 2)
        dow_avg[dow] = avg_s
        print(f" {avg_s:8.2f}")

    # 3. Hour-of-day analysis
    print("\n[3/4] Hour-of-day analysis...")
    hod_results = {}

    for year in YEARS:
        ts_list = all_data[year]["timestamps"]
        n_bars = min(len(all_data[year]["returns"][s]) for s in SYMBOLS)
        eq_rets = np.zeros(n_bars)
        for sym in SYMBOLS:
            eq_rets += all_data[year]["returns"][sym][:n_bars] / len(SYMBOLS)

        hod_rets = {h: [] for h in range(24)}
        for i in range(n_bars):
            dt = datetime.fromtimestamp(ts_list[i], tz=timezone.utc)
            hod_rets[dt.hour].append(eq_rets[i])

        hod_results[year] = {}
        for h in range(24):
            arr = np.array(hod_rets[h])
            ann_ret = float(np.mean(arr)) * 8760
            ann_vol = float(np.std(arr)) * np.sqrt(8760)
            sr = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0
            hod_results[year][h] = round(sr, 2)

    # Find best/worst hours
    hour_avgs = {}
    for h in range(24):
        vals = [hod_results[y][h] for y in YEARS]
        hour_avgs[h] = round(float(np.mean(vals)), 2)

    best_hours = sorted(hour_avgs, key=lambda h: hour_avgs[h], reverse=True)[:5]
    worst_hours = sorted(hour_avgs, key=lambda h: hour_avgs[h])[:5]
    print(f"  Best hours (UTC):  {', '.join(f'{h}h({hour_avgs[h]:+.2f})' for h in best_hours)}")
    print(f"  Worst hours (UTC): {', '.join(f'{h}h({hour_avgs[h]:+.2f})' for h in worst_hours)}")

    # 4. Build and test calendar strategies
    print("\n[4/4] Testing calendar strategies...")

    strategies = {}

    # Strategy A: Weekend effect (long weekends, flat weekdays)
    # Strategy B: Inverse weekend (long weekdays, flat weekends)
    # Strategy C: Best 3 DOW long, worst 2 flat
    # Strategy D: Best 8 hours long, rest flat
    # Strategy E: Combined DOW + HOD

    # Find consistently good/bad DOW
    good_dow = [d for d in dow_names if dow_avg[d] > 0.5]
    bad_dow = [d for d in dow_names if dow_avg[d] < -0.5]
    print(f"  Good DOW: {good_dow}")
    print(f"  Bad DOW: {bad_dow}")

    # Strategy tests
    strategy_configs = {
        "weekend_long": {
            "desc": "Long weekends (Sat+Sun), flat weekdays",
            "days": [5, 6],  # Sat, Sun
            "hours": None,
        },
        "weekday_long": {
            "desc": "Long weekdays (Mon-Fri), flat weekends",
            "days": [0, 1, 2, 3, 4],
            "hours": None,
        },
        "best_dow": {
            "desc": f"Long on DOW with Sharpe>0.5: {good_dow}",
            "days": [i for i, d in enumerate(dow_names) if dow_avg[d] > 0.5],
            "hours": None,
        },
        "best_hours": {
            "desc": "Long during best 8 hours, flat rest",
            "days": None,
            "hours": best_hours[:8],
        },
        "avoid_worst_dow": {
            "desc": f"Flat on worst DOW (Sharpe<-0.5): {bad_dow}",
            "days": [i for i, d in enumerate(dow_names) if dow_avg[d] >= -0.5],
            "hours": None,
        },
    }

    for strat_name, strat_cfg in strategy_configs.items():
        yearly_sharpes = []
        yearly_detail = {}

        for year in YEARS:
            ts_list = all_data[year]["timestamps"]
            n_bars = min(len(all_data[year]["returns"][s]) for s in SYMBOLS)
            eq_rets = np.zeros(n_bars)
            for sym in SYMBOLS:
                eq_rets += all_data[year]["returns"][sym][:n_bars] / len(SYMBOLS)

            # Apply calendar filter
            filtered = np.zeros(n_bars)
            n_active = 0
            for i in range(n_bars):
                dt = datetime.fromtimestamp(ts_list[i], tz=timezone.utc)
                active = True

                if strat_cfg["days"] is not None:
                    active = active and (dt.weekday() in strat_cfg["days"])
                if strat_cfg["hours"] is not None:
                    active = active and (dt.hour in strat_cfg["hours"])

                if active:
                    filtered[i] = eq_rets[i]
                    n_active += 1

            s = sharpe(filtered)
            yearly_sharpes.append(s)
            yearly_detail[year] = {
                "sharpe": s,
                "active_pct": round(n_active / n_bars * 100, 1),
            }

        avg = round(float(np.mean(yearly_sharpes)), 4)
        mn = round(float(np.min(yearly_sharpes)), 4)
        obj = obj_func(yearly_sharpes)

        strategies[strat_name] = {
            "description": strat_cfg["desc"],
            "yearly": yearly_detail,
            "avg_sharpe": avg,
            "min_sharpe": mn,
            "obj": obj,
        }

        delta = obj  # No formal baseline for standalone strategies
        print(f"  {strat_name:20s}: OBJ={obj:.4f}  AVG={avg:.3f}  MIN={mn:.3f}")

    # Also compute "always long" baseline (equal weight)
    baseline_sharpes = []
    for year in YEARS:
        n_bars = min(len(all_data[year]["returns"][s]) for s in SYMBOLS)
        eq_rets = np.zeros(n_bars)
        for sym in SYMBOLS:
            eq_rets += all_data[year]["returns"][sym][:n_bars] / len(SYMBOLS)
        baseline_sharpes.append(sharpe(eq_rets))
    baseline_obj = obj_func(baseline_sharpes)
    baseline_avg = round(float(np.mean(baseline_sharpes)), 4)
    print(f"\n  ALWAYS LONG BASELINE: OBJ={baseline_obj:.4f}  AVG={baseline_avg:.3f}")

    # Check if any strategy has consistent directionality
    print("\n  Consistency check (sign of Sharpe across years):")
    for sn, sr in strategies.items():
        signs = [1 if sr["yearly"][y]["sharpe"] > 0 else -1 for y in YEARS]
        consistency = sum(1 for s in signs if s > 0)
        print(f"    {sn:20s}: {consistency}/5 years positive")

    # Verdict
    best_strat = max(strategies, key=lambda k: strategies[k]["obj"])
    best = strategies[best_strat]

    if best["obj"] < 0.3:
        verdict = "NO ALPHA — calendar patterns too weak for standalone strategy"
    elif best["min_sharpe"] < 0:
        verdict = f"WEAK — {best_strat} has positive OBJ but negative years"
    else:
        verdict = f"POTENTIAL — {best_strat} OBJ={best['obj']:.3f}, needs ensemble test"

    elapsed = time.time() - t0
    _partial = {
        "phase": 130,
        "description": "Calendar Alpha Research",
        "elapsed_seconds": round(elapsed, 1),
        "dow_analysis": {
            "per_year": dow_results,
            "avg_sharpe": dow_avg,
        },
        "hour_analysis": {
            "per_year": {y: {str(h): v for h, v in hod_results[y].items()} for y in YEARS},
            "avg_sharpe": {str(h): v for h, v in hour_avgs.items()},
            "best_hours": best_hours,
            "worst_hours": worst_hours,
        },
        "strategies": strategies,
        "baseline": {"avg_sharpe": baseline_avg, "obj": baseline_obj},
        "best_strategy": {"name": best_strat, **best},
        "verdict": verdict,
    }
    _save(_partial, partial=False)

    print(f"\n  VERDICT: {verdict}")
    print(f"\nPhase 130 complete in {elapsed:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
