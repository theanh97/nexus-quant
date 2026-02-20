"""
Diversified CTA Backtest — Phase 138
======================================
Combines commodity futures + FX majors + Treasury bonds in one trend-following strategy.

This addresses the key finding from Phase 137:
- Commodity-only CTA fails in commodity bear markets (2015-2019 OOS1 Sharpe=-0.206)
- FX trends are INDEPENDENT of commodity cycles
- Bond trends are INVERSE to commodity crashes (flight to quality)
- Combined: much better Sharpe stability across all market regimes

Universe:
  Commodities (13): CL=F, NG=F, BZ=F, GC=F, SI=F, HG=F, PL=F, ZW=F, ZC=F, ZS=F, KC=F, SB=F, CT=F
  FX Majors  ( 6): EURUSD=X, GBPUSD=X, JPYUSD=X, AUDUSD=X, CADUSD=X, CHFUSD=X
  Bonds      ( 2): TLT (20Y Treasury), IEF (10Y Treasury)

Strategy: EMA(12/26 + 20/50) crossover + vol-targeting (same as commodity champion)
Rebalance: monthly (21 bars)
Costs: 5bps slippage + 1bp fee + 1bp spread = 7bps RT for all

Target: Sharpe > 0.5 (vs SG CTA Index ~0.5)
Usage:
    cd "/Users/qtmobile/Desktop/Nexus - Quant Trading "
    python3 scripts/run_diversified_cta.py
"""
from __future__ import annotations

import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("div_cta")

# ── Universe ─────────────────────────────────────────────────────────────────

COMMODITY_SYMBOLS = [
    "CL=F", "NG=F", "BZ=F", "GC=F", "SI=F", "HG=F", "PL=F",
    "ZW=F", "ZC=F", "ZS=F", "KC=F", "SB=F", "CT=F",
]
FX_SYMBOLS = [
    "EURUSD=X", "GBPUSD=X", "JPYUSD=X", "AUDUSD=X", "CADUSD=X", "CHFUSD=X",
]
BOND_SYMBOLS = ["TLT", "IEF"]

ALL_SYMBOLS = COMMODITY_SYMBOLS + FX_SYMBOLS + BOND_SYMBOLS


# ── Data download ────────────────────────────────────────────────────────────

def download_yfinance(symbol: str, start: str, end: str) -> Optional[Dict[str, List]]:
    """Download daily OHLCV via yfinance."""
    try:
        import yfinance as yf
        df = yf.Ticker(symbol).history(start=start, end=end)
        if df is None or len(df) == 0:
            return None

        col_map = {c.lower(): c for c in df.columns}
        close_col = col_map.get("close")
        if not close_col:
            return None

        result = {"dates": [], "timestamps": [], "open": [], "high": [], "low": [], "close": [], "volume": []}
        for idx_val, row in df.iterrows():
            try:
                c = float(row[close_col])
                if c != c or c <= 0:
                    continue
                date_str = str(idx_val)[:10]
                ts = int(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())

                def _g(col_name, fb=c):
                    mapped = col_map.get(col_name)
                    if not mapped or mapped not in row.index:
                        return fb
                    v = row[mapped]
                    try:
                        f = float(v)
                        return f if f == f and f > 0 else fb
                    except (TypeError, ValueError):
                        return fb

                result["dates"].append(date_str)
                result["timestamps"].append(ts)
                result["open"].append(_g("open"))
                result["high"].append(_g("high"))
                result["low"].append(_g("low"))
                result["close"].append(c)
                result["volume"].append(max(0.0, _g("volume", 0.0)))
            except Exception:
                continue

        return result if result["timestamps"] else None
    except Exception as e:
        logger.debug(f"  {symbol}: yfinance error: {e}")
        return None


def load_all_data(start="2005-01-01", end="2026-02-20", cache_dir="data/cache/diversified_cta"):
    """Download all symbols and align to common timeline."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    raw_data = {}
    valid_syms = []

    for sym in ALL_SYMBOLS:
        cache_file = cache_path / f"{sym.replace('=', '_').replace('/', '-')}_{start}_{end}.json"

        # Try cache first
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                if data and data.get("timestamps"):
                    logger.info(f"  ✓ {sym:12s}: {len(data['timestamps'])} bars (cached)")
                    raw_data[sym] = data
                    valid_syms.append(sym)
                    continue
            except Exception:
                pass

        # Download
        data = download_yfinance(sym, start, end)
        if data and len(data["timestamps"]) >= 200:
            with open(cache_file, "w") as f:
                json.dump(data, f, separators=(",", ":"))
            logger.info(f"  ✓ {sym:12s}: {len(data['timestamps'])} bars (downloaded)")
            raw_data[sym] = data
            valid_syms.append(sym)
        else:
            logger.warning(f"  ✗ {sym:12s}: insufficient data")
        time.sleep(0.2)

    if not valid_syms:
        raise RuntimeError("No valid symbols loaded")

    # Align to common timeline
    all_ts = sorted({ts for sym in valid_syms for ts in raw_data[sym]["timestamps"]})

    aligned = {}
    for sym in valid_syms:
        src = raw_data[sym]
        ts_to_i = {ts: i for i, ts in enumerate(src["timestamps"])}
        last = {"open": float("nan"), "high": float("nan"), "low": float("nan"),
                "close": float("nan"), "volume": 0.0}
        out = {k: [] for k in last}
        for ts in all_ts:
            if ts in ts_to_i:
                i = ts_to_i[ts]
                for k in last:
                    v = src[k][i]
                    if v == v:
                        last[k] = v
            for k in last:
                out[k].append(last[k])
        aligned[sym] = out

    # Trim to where ALL symbols have valid data
    trim_start = 0
    for sym in valid_syms:
        first_valid = next((i for i, v in enumerate(aligned[sym]["close"]) if v == v and v > 0), len(all_ts))
        trim_start = max(trim_start, first_valid)

    if trim_start > 0:
        logger.info(f"  Trimming {trim_start} bars (waiting for all symbols)")
        all_ts = all_ts[trim_start:]
        for sym in valid_syms:
            for k in aligned[sym]:
                aligned[sym][k] = aligned[sym][k][trim_start:]

    logger.info(f"  Aligned: {len(valid_syms)} symbols × {len(all_ts)} bars")
    return all_ts, aligned, valid_syms


# ── Features ────────────────────────────────────────────────────────────────

def compute_features(aligned, symbols, timeline):
    """Compute rv_20d (annualised realized vol) per symbol."""
    import math
    rv_20d = {}
    for sym in symbols:
        cl = aligned[sym]["close"]
        n = len(cl)
        rv = [0.20] * n  # default 20% vol
        buf = []
        for i in range(1, n):
            c0, c1 = cl[i-1], cl[i]
            if c0 > 0 and c1 > 0 and c0 == c0 and c1 == c1:
                r = math.log(c1 / c0)
                buf.append(r)
                if len(buf) > 20:
                    buf.pop(0)
                if len(buf) >= 5:
                    mn = sum(buf) / len(buf)
                    var = sum((x - mn)**2 for x in buf) / len(buf)
                    rv[i] = math.sqrt(var * 252) if var > 0 else 0.20
        rv_20d[sym] = rv
    return {"rv_20d": rv_20d}


# ── Backtest ─────────────────────────────────────────────────────────────────

def run_backtest(timeline, aligned, symbols, features, costs_bps=7.0):
    """Run EMA trend-following backtest with vol-targeting."""
    fast1, slow1 = 12, 26
    fast2, slow2 = 20, 50
    k1f = 2.0 / (fast1 + 1); k1s = 2.0 / (slow1 + 1)
    k2f = 2.0 / (fast2 + 1); k2s = 2.0 / (slow2 + 1)
    VOL_TARGET = 0.12
    MAX_GROSS = 2.0
    MAX_POS = 0.25
    REBAL_FREQ = 21
    WARMUP = 60

    # Initialize EMA state
    ema_state = {}
    for sym in symbols:
        p = aligned[sym]["close"][0]
        ema_state[sym] = {"ef1": p, "es1": p, "ef2": p, "es2": p, "ema1": 0.0, "ema2": 0.0, "bars": 0}

    rv20 = features["rv_20d"]
    n = len(timeline)
    equity = 1.0
    equity_curve = [equity]
    weights = {sym: 0.0 for sym in symbols}
    last_rebal = -1
    cost_rate = costs_bps / 10000.0

    for idx in range(1, n):
        # Update EMA state
        for sym in symbols:
            p = aligned[sym]["close"][idx]
            if p != p or p <= 0:
                continue
            st = ema_state[sym]
            st["ef1"] = st["ef1"] * (1 - k1f) + p * k1f
            st["es1"] = st["es1"] * (1 - k1s) + p * k1s
            st["ef2"] = st["ef2"] * (1 - k2f) + p * k2f
            st["es2"] = st["es2"] * (1 - k2s) + p * k2s
            st["bars"] += 1
            if st["bars"] >= slow2:
                st["ema1"] = (st["ef1"] - st["es1"]) / (st["es1"] + 1e-10)
                st["ema2"] = (st["ef2"] - st["es2"]) / (st["es2"] + 1e-10)

        # Price P&L
        port_ret = 0.0
        for sym in symbols:
            c0 = aligned[sym]["close"][idx - 1]
            c1 = aligned[sym]["close"][idx]
            if c0 > 0 and c1 > 0 and c0 == c0 and c1 == c1:
                port_ret += weights[sym] * (c1 / c0 - 1.0)
        equity += equity * port_ret

        # Rebalance
        if idx >= WARMUP and (last_rebal < 0 or idx - last_rebal >= REBAL_FREQ):
            last_rebal = idx
            raw = {}
            for sym in symbols:
                st = ema_state[sym]
                if st["bars"] < slow2:
                    continue
                s1 = math.copysign(1.0, st["ema1"]) if abs(st["ema1"]) > 0.001 else 0.0
                s2 = math.copysign(1.0, st["ema2"]) if abs(st["ema2"]) > 0.001 else 0.0
                combo = (s1 + s2) / 2.0  # just EMA for simplicity
                if abs(combo) < 0.3:  # require both EMAs to agree
                    continue
                vol = rv20[sym][idx]
                vol = vol if (vol and vol == vol and vol > 0) else 0.20
                w = math.copysign(VOL_TARGET / max(vol, 1e-10), combo)
                w = math.copysign(min(abs(w), MAX_POS), w)
                raw[sym] = w

            if raw:
                gross = sum(abs(v) for v in raw.values())
                if gross > MAX_GROSS:
                    f = MAX_GROSS / gross
                    raw = {k: v * f for k, v in raw.items()}

                # Transaction costs
                turnover = sum(abs(raw.get(sym, 0.0) - weights.get(sym, 0.0)) for sym in symbols)
                cost = equity * turnover * cost_rate
                equity -= cost
                weights = {sym: raw.get(sym, 0.0) for sym in symbols}

        equity_curve.append(equity)

    return equity_curve


def compute_metrics(equity_curve):
    rets = [equity_curve[i] / equity_curve[i-1] - 1 for i in range(1, len(equity_curve)) if equity_curve[i-1] > 0 and equity_curve[i] == equity_curve[i]]
    if not rets:
        return {"sharpe": 0, "cagr": 0, "max_drawdown": 1}
    n = len(rets)
    mn = sum(rets) / n
    std = math.sqrt(sum((r - mn)**2 for r in rets) / n)
    sharpe = (mn / (std + 1e-10)) * math.sqrt(252)
    cagr = equity_curve[-1] ** (252 / n) - 1
    peak = 1.0; max_dd = 0.0
    for e in equity_curve:
        if e > peak: peak = e
        dd = (peak - e) / peak
        if dd > max_dd: max_dd = dd
    return {"sharpe": sharpe, "cagr": cagr, "max_drawdown": max_dd}


def slice_equity(equity_curve, timeline, start_date, end_date):
    """Get equity slice for a date range."""
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    indices = [i for i, ts in enumerate(timeline) if start_ts <= ts <= end_ts]
    if not indices:
        return [1.0]
    i0 = indices[0]
    # Normalize to start at 1.0
    base = equity_curve[i0]
    return [equity_curve[i] / base for i in indices] if base > 0 else [1.0]


def main():
    logger.info("=" * 65)
    logger.info("NEXUS Diversified CTA — Phase 138")
    logger.info("13 Commodities + 6 FX Pairs + 2 Treasury Bonds = 21 instruments")
    logger.info("=" * 65)

    # Download data
    logger.info("\nDownloading/loading data...")
    timeline, aligned, symbols = load_all_data()

    # Compute features
    features = compute_features(aligned, symbols, timeline)

    logger.info(f"\nUniverse: {len(symbols)} symbols")
    logger.info(f"  Commodities: {[s for s in symbols if s in COMMODITY_SYMBOLS]}")
    logger.info(f"  FX:         {[s for s in symbols if s in FX_SYMBOLS]}")
    logger.info(f"  Bonds:      {[s for s in symbols if s in BOND_SYMBOLS]}")

    # Run full backtest
    logger.info("\nRunning full backtest (2005-2026, 7bps RT costs)...")
    eq_full = run_backtest(timeline, aligned, symbols, features, costs_bps=7.0)
    metrics_full = compute_metrics(eq_full)

    # Walk-forward analysis
    periods = [
        ("IS  ", "2007-07-01", "2015-12-31"),
        ("OOS1", "2016-01-01", "2020-12-31"),
        ("OOS2", "2021-01-01", "2026-02-20"),
        ("FULL", "2007-07-01", "2026-02-20"),
    ]

    logger.info("\nWalk-Forward Results:")
    logger.info(f"{'Period':6s}  {'Sharpe':7s}  {'CAGR':8s}  {'MaxDD':7s}")
    logger.info("-" * 40)

    results = {}
    for label, start, end in periods:
        eq_slice = slice_equity(eq_full, timeline, start, end)
        m = compute_metrics(eq_slice)
        results[label.strip()] = m
        logger.info(
            f"{label}  {m['sharpe']:+7.3f}  {m['cagr']*100:+7.2f}%  {m['max_drawdown']*100:6.1f}%"
        )

    # Verdict
    oos_min = min(results.get("OOS1", {}).get("sharpe", -99), results.get("OOS2", {}).get("sharpe", -99))
    oos_avg = (results.get("OOS1", {}).get("sharpe", 0) + results.get("OOS2", {}).get("sharpe", 0)) / 2

    logger.info("\n" + "─" * 40)
    logger.info("Diversified vs Commodity-Only:")
    logger.info("  Commodity-only: IS=+0.524, OOS1=-0.206, OOS2=+0.252, FULL=+0.338")
    logger.info(f"  Diversified:    IS={results.get('IS', {}).get('sharpe', 0):+.3f}, "
                f"OOS1={results.get('OOS1', {}).get('sharpe', 0):+.3f}, "
                f"OOS2={results.get('OOS2', {}).get('sharpe', 0):+.3f}, "
                f"FULL={results.get('FULL', {}).get('sharpe', 0):+.3f}")
    logger.info(f"\n  OOS MIN: {oos_min:+.3f}  (target > 0.3)")
    logger.info(f"  OOS AVG: {oos_avg:+.3f}  (target > 0.5)")

    if oos_min > 0.5:
        verdict = "STRONG PASS"
    elif oos_min > 0.3:
        verdict = "PASS"
    elif oos_min > 0.0:
        verdict = "WEAK PASS"
    else:
        verdict = "FAIL"
    logger.info(f"\n  Verdict: {verdict}")

    # Year-by-year
    logger.info("\nAnnual breakdown:")
    start_ts = int(datetime.strptime("2007-07-01", "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    by_yr = {}
    for i, ts in enumerate(timeline):
        if ts < start_ts:
            continue
        yr = datetime.utcfromtimestamp(ts).year
        if yr not in by_yr:
            by_yr[yr] = []
        by_yr[yr].append(eq_full[i])
    prev = eq_full[next(i for i, ts in enumerate(timeline) if ts >= start_ts)]
    for yr in sorted(by_yr):
        if len(by_yr[yr]) < 100:
            continue
        ann = by_yr[yr][-1] / prev - 1
        prev = by_yr[yr][-1]
        logger.info(f"  {yr}: {ann*100:+7.1f}%")

    # Save results
    out = {
        "strategy": "diversified_cta",
        "instruments": len(symbols),
        "commodities": len([s for s in symbols if s in COMMODITY_SYMBOLS]),
        "fx": len([s for s in symbols if s in FX_SYMBOLS]),
        "bonds": len([s for s in symbols if s in BOND_SYMBOLS]),
        "periods": {k: {m: round(v, 4) for m, v in met.items()} for k, met in results.items()},
        "oos_min": round(oos_min, 4),
        "oos_avg": round(oos_avg, 4),
        "verdict": verdict,
        "run_date": datetime.now().isoformat(),
    }
    out_path = ROOT / "artifacts" / "diversified_cta_wf.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
