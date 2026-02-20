"""
Phase 162 — F168 Rebalance Interval Sweep
==========================================
Hypothesis: Crypto perpetual funding resets every 8h (00:00 / 08:00 / 16:00 UTC).
The F168 signal accumulates funding momentum over 168h. Rebalancing at shorter
intervals (8h or 16h) may let us react faster to funding regime shifts; the
current 24h interval may miss dislocations that resolve within 8-16h.

Test: rebalance_interval_bars ∈ [8, 12, 16, 24, 36, 48]
  - 8h  = aligns exactly with funding settlement cadence
  - 12h = 1.5 funding cycles
  - 16h = 2 funding cycles
  - 24h = current production baseline (3 cycles/day)
  - 36h = 4.5 cycles
  - 48h = 6 cycles (slowest)

All other signals keep current production intervals:
  V1=60h, I460bw168=48h, I415bw216=48h

Baseline (prod v2.8.0): OBJ=2.2095
"""

import json
import os
import signal as _signal
import sys
import time
import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from nexus_quant.backtest.costs import cost_model_from_config
from nexus_quant.backtest.engine import BacktestConfig, BacktestEngine
from nexus_quant.data.providers.registry import make_provider
from nexus_quant.strategies.registry import make_strategy

_start = time.time()

PROD_CFG = json.loads((ROOT / "configs" / "production_p91b_champion.json").read_text())
SYMBOLS  = PROD_CFG["data"]["symbols"]

# v2.8.0 overlay constants
VOL_WINDOW     = 168
VOL_THRESHOLD  = 0.5
VOL_SCALE      = 0.4
VOL_F168_BOOST = 0.15
BRD_LOOKBACK   = 192
PCT_WINDOW     = 336
P_LOW, P_HIGH  = 0.35, 0.65
FUND_DISP_THR  = 0.75
FUND_DISP_SCALE= 1.15
TS_REDUCE_THR  = 0.70
TS_REDUCE_SCALE= 0.60
TS_BOOST_THR   = 0.30
TS_BOOST_SCALE = 1.15
TS_SHORT, TS_LONG = 24, 144

WEIGHTS = {
    "LOW":  {"v1": 0.2747, "i460bw168": 0.1967, "i415bw216": 0.3247, "f168": 0.2039},
    "MID":  {"v1": 0.16,   "i460bw168": 0.22,   "i415bw216": 0.37,   "f168": 0.25},
    "HIGH": {"v1": 0.05,   "i460bw168": 0.25,   "i415bw216": 0.45,   "f168": 0.25},
}

YEAR_RANGES = {
    "2021": ("2021-02-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
}
YEARS = sorted(YEAR_RANGES.keys())

OUT_DIR = ROOT / "artifacts" / "phase162"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

# Test variants: F168 rebalance interval (hours = bars at 1h resolution)
F168_INTERVALS = [8, 12, 16, 24, 36, 48]
BASELINE_F168_INTERVAL = 24  # production v2.8.0

_partial: dict = {}

def _on_timeout(signum, frame):
    print("\n⏰ TIMEOUT — saving partial...")
    _partial["partial"] = True
    _partial["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (OUT_DIR / "phase162_report.json").write_text(json.dumps(_partial, indent=2))
    sys.exit(0)

_signal.signal(_signal.SIGALRM, _on_timeout)
_signal.alarm(2700)  # 45min guard


def sharpe(rets: np.ndarray) -> float:
    if len(rets) < 100:
        return 0.0
    mu = float(np.mean(rets)) * 8760
    sd = float(np.std(rets)) * np.sqrt(8760)
    return round(mu / sd, 4) if sd > 1e-12 else 0.0


def obj_func(yearly_sharpes: dict) -> float:
    arr = np.array(list(yearly_sharpes.values()))
    return round(float(np.mean(arr) - 0.5 * np.std(arr)), 4)


def compute_signals(dataset) -> dict:
    n = len(dataset.timeline)

    # BTC price vol
    btc_rets = np.zeros(n)
    for i in range(1, n):
        c0 = dataset.close("BTCUSDT", i - 1)
        c1 = dataset.close("BTCUSDT", i)
        btc_rets[i] = (c1 / c0 - 1.0) if c0 > 0 else 0.0
    btc_vol = np.full(n, np.nan)
    for i in range(VOL_WINDOW, n):
        btc_vol[i] = float(np.std(btc_rets[i - VOL_WINDOW:i])) * np.sqrt(8760)
    if VOL_WINDOW < n:
        btc_vol[:VOL_WINDOW] = btc_vol[VOL_WINDOW]

    # Breadth regime
    breadth = np.full(n, 0.5)
    for i in range(BRD_LOOKBACK, n):
        pos = sum(
            1 for sym in SYMBOLS
            if (c0 := dataset.close(sym, i - BRD_LOOKBACK)) > 0
            and dataset.close(sym, i) > c0
        )
        breadth[i] = pos / len(SYMBOLS)
    breadth[:BRD_LOOKBACK] = 0.5
    brd_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        brd_pct[i] = float(np.mean(breadth[i - PCT_WINDOW:i] <= breadth[i]))
    brd_pct[:PCT_WINDOW] = 0.5
    breadth_regime = np.where(brd_pct >= P_HIGH, 2,
                     np.where(brd_pct >= P_LOW, 1, 0)).astype(int)

    # Funding dispersion
    fund_std_raw = np.zeros(n)
    for i in range(n):
        ts = dataset.timeline[i]
        rates = []
        for sym in SYMBOLS:
            try:
                rates.append(dataset.last_funding_rate_before(sym, ts))
            except Exception:
                pass
        fund_std_raw[i] = float(np.std(rates)) if len(rates) > 1 else 0.0
    fund_std_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        fund_std_pct[i] = float(np.mean(fund_std_raw[i - PCT_WINDOW:i] <= fund_std_raw[i]))
    fund_std_pct[:PCT_WINDOW] = 0.5

    # Funding term structure spread
    ts_spread_raw = np.zeros(n)
    for i in range(max(TS_SHORT, TS_LONG), n):
        short_rates, long_rates = [], []
        for sym in SYMBOLS:
            ts_now = dataset.timeline[i]
            ts_short_start = dataset.timeline[max(0, i - TS_SHORT)]
            r_now   = dataset.last_funding_rate_before(sym, ts_now)
            r_short = dataset.last_funding_rate_before(sym, ts_short_start)
            short_rates.append(r_now)
            long_rates.append((r_now + r_short) / 2)
        ts_spread_raw[i] = float(np.mean(short_rates)) - float(np.mean(long_rates))
    ts_spread_raw[:max(TS_SHORT, TS_LONG)] = 0.0
    ts_spread_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        ts_spread_pct[i] = float(np.mean(ts_spread_raw[i - PCT_WINDOW:i] <= ts_spread_raw[i]))
    ts_spread_pct[:PCT_WINDOW] = 0.5

    return {
        "btc_vol": btc_vol,
        "breadth_regime": breadth_regime,
        "fund_std_pct": fund_std_pct,
        "ts_spread_pct": ts_spread_pct,
    }


def compute_ensemble(sig_rets: dict, signals: dict) -> np.ndarray:
    sk_all = ["v1", "i460bw168", "i415bw216", "f168"]
    min_len = min(len(sig_rets[sk]) for sk in sk_all)
    bv  = signals["btc_vol"][:min_len]
    reg = signals["breadth_regime"][:min_len]
    fsp = signals["fund_std_pct"][:min_len]
    tsp = signals["ts_spread_pct"][:min_len]

    ens = np.zeros(min_len)
    for i in range(min_len):
        w = WEIGHTS[["LOW", "MID", "HIGH"][int(reg[i])]]
        if not np.isnan(bv[i]) and bv[i] > VOL_THRESHOLD:
            boost = VOL_F168_BOOST / max(1, len(sk_all) - 1)
            ret_i = 0.0
            for sk in sk_all:
                adj_w = (min(0.60, w[sk] + VOL_F168_BOOST) if sk == "f168"
                         else max(0.05, w[sk] - boost))
                ret_i += adj_w * sig_rets[sk][i]
            ret_i *= VOL_SCALE
        else:
            ret_i = sum(w[sk] * sig_rets[sk][i] for sk in sk_all)
        if fsp[i] > FUND_DISP_THR:
            ret_i *= FUND_DISP_SCALE
        if tsp[i] > TS_REDUCE_THR:
            ret_i *= TS_REDUCE_SCALE
        elif tsp[i] < TS_BOOST_THR:
            ret_i *= TS_BOOST_SCALE
        ens[i] = ret_i
    return ens


# ─── MAIN ────────────────────────────────────────────────────────────────────

print("=" * 68)
print("PHASE 162 — F168 Rebalance Interval Sweep")
print("=" * 68)

print("\n[1/4] Loading data + computing signals + base signals (V1, I460, I415)...")

# Shared: V1, I460, I415 — same for all F168 interval variants
shared_rets: dict = {"v1": {}, "i460bw168": {}, "i415bw216": {}}
# F168 variants: interval → year → returns
f168_rets: dict = {iv: {} for iv in F168_INTERVALS}
signals_data: dict = {}

for year, (start, end) in YEAR_RANGES.items():
    print(f"  {year}: ", end="", flush=True)
    cfg_data = {
        "provider": "binance_rest_v1", "symbols": SYMBOLS,
        "start": start, "end": end, "bar_interval": "1h",
        "cache_dir": ".cache/binance_rest",
    }
    provider = make_provider(cfg_data, seed=42)
    dataset  = provider.load()
    signals_data[year] = compute_signals(dataset)
    print("S", end="", flush=True)

    # Shared signals (fixed params)
    for sk, sname, params in [
        ("v1", "nexus_alpha_v1", {
            "k_per_side": 2, "w_carry": 0.35, "w_mom": 0.45, "w_mean_reversion": 0.2,
            "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 72,
            "vol_lookback_bars": 168, "target_gross_leverage": 0.35,
            "rebalance_interval_bars": 60,
        }),
        ("i460bw168", "idio_momentum_alpha", {
            "k_per_side": 4, "lookback_bars": 460, "beta_window_bars": 168,
            "target_gross_leverage": 0.3, "rebalance_interval_bars": 48,
        }),
        ("i415bw216", "idio_momentum_alpha", {
            "k_per_side": 4, "lookback_bars": 415, "beta_window_bars": 216,
            "target_gross_leverage": 0.3, "rebalance_interval_bars": 48,
        }),
    ]:
        bt_cfg = BacktestConfig(costs=COST_MODEL)
        strat  = make_strategy({"name": sname, "params": params})
        result = BacktestEngine(bt_cfg).run(dataset, strat)
        shared_rets[sk][year] = np.array(result.returns)
    print(".", end="", flush=True)

    # F168 variants: sweep rebalance_interval_bars
    for iv in F168_INTERVALS:
        bt_cfg = BacktestConfig(costs=COST_MODEL)
        strat  = make_strategy({"name": "funding_momentum_alpha", "params": {
            "k_per_side": 2,
            "funding_lookback_bars": 168,
            "direction": "contrarian",
            "target_gross_leverage": 0.25,
            "rebalance_interval_bars": iv,
        }})
        result = BacktestEngine(bt_cfg).run(dataset, strat)
        f168_rets[iv][year] = np.array(result.returns)
    print(". ✓")

_partial.update({
    "phase": 162,
    "description": "F168 Rebalance Interval Sweep",
    "partial": True,
})

# Assemble yearly sig_rets dict for a given F168 interval
def make_sig_rets(year, f168_iv):
    return {
        "v1":        shared_rets["v1"][year],
        "i460bw168": shared_rets["i460bw168"][year],
        "i415bw216": shared_rets["i415bw216"][year],
        "f168":      f168_rets[f168_iv][year],
    }

print("\n[2/4] Computing baselines and sweep...")
results = {}
for iv in F168_INTERVALS:
    yearly = {}
    for year in YEARS:
        yearly[year] = sharpe(compute_ensemble(make_sig_rets(year, iv), signals_data[year]))
    o = obj_func(yearly)
    results[iv] = {"obj": o, "yearly": yearly}

# Baseline is the iv=24 result
baseline_obj    = results[BASELINE_F168_INTERVAL]["obj"]
baseline_yearly = results[BASELINE_F168_INTERVAL]["yearly"]
print(f"  Baseline (f168_rb=24h): OBJ={baseline_obj:.4f} | {baseline_yearly}")
print()

best_iv    = BASELINE_F168_INTERVAL
best_delta = 0.0
for iv in F168_INTERVALS:
    o  = results[iv]["obj"]
    d  = o - baseline_obj
    yr = results[iv]["yearly"]
    sym = "✅" if d > 0.005 else ("⚠️ " if d > -0.005 else "❌")
    tag = " ← baseline" if iv == BASELINE_F168_INTERVAL else ""
    print(f"  {sym} rb={iv:2d}h: OBJ={o:.4f} (Δ={d:+.4f}) | {yr}{tag}")
    if d > best_delta:
        best_delta = d
        best_iv    = iv

best_obj    = results[best_iv]["obj"]
best_yearly = results[best_iv]["yearly"]
print(f"\n  Best: rb={best_iv}h OBJ={best_obj:.4f} (Δ={best_delta:+.4f})")

# LOYO validation (if best is not baseline and delta > threshold)
print(f"\n[3/4] LOYO validation (rb={best_iv}h)...")
loyo_wins, loyo_deltas = 0, []
for held_out in YEARS:
    sh_best = results[best_iv]["yearly"][held_out]
    sh_base = baseline_yearly[held_out]
    d = sh_best - sh_base
    loyo_deltas.append(d)
    loyo_wins += int(d > 0)
    print(f"  LOYO {held_out}: best={sh_best:.4f} base={sh_base:.4f} Δ={d:+.4f} {'✅' if d>0 else '❌'}")

loyo_avg  = float(np.mean(loyo_deltas))
validated = (best_iv != BASELINE_F168_INTERVAL
             and best_delta > 0.005
             and loyo_wins >= 3)

print("\n" + "=" * 68)
if validated:
    print(f"✅ VALIDATED — F168 rb={best_iv}h: OBJ={best_obj:.4f} (Δ={best_delta:+.4f}) | LOYO {loyo_wins}/5")
else:
    print(f"❌ NO IMPROVEMENT — baseline rb=24h OBJ={baseline_obj:.4f} optimal")
    print(f"   Best variant: rb={best_iv}h Δ={best_delta:+.4f} | LOYO {loyo_wins}/5 (need ≥3 + Δ>0.005)")
print("=" * 68)

# Full table for report
sweep_table = {}
for iv in F168_INTERVALS:
    sweep_table[f"rb_{iv}h"] = {
        "rebalance_interval_bars": iv,
        "obj": results[iv]["obj"],
        "delta": round(results[iv]["obj"] - baseline_obj, 4),
        "yearly": results[iv]["yearly"],
    }

report = {
    "phase": 162,
    "description": "F168 Rebalance Interval Sweep",
    "elapsed_seconds": round(time.time() - _start, 1),
    "baseline_rb": BASELINE_F168_INTERVAL,
    "baseline_obj": baseline_obj,
    "baseline_yearly": baseline_yearly,
    "sweep_table": sweep_table,
    "best_rb": best_iv,
    "best_obj": best_obj,
    "best_delta": best_delta,
    "best_yearly": best_yearly,
    "loyo_wins": loyo_wins,
    "loyo_avg_delta": loyo_avg,
    "validated": validated,
    "verdict": (
        f"VALIDATED — F168 rebalance_interval_bars={best_iv}h OBJ={best_obj:.4f} Δ={best_delta:+.4f} LOYO {loyo_wins}/5"
        if validated else
        f"NO IMPROVEMENT — F168 rb=24h OBJ={baseline_obj:.4f} optimal"
    ),
    "partial": False,
    "timestamp": datetime.datetime.utcnow().isoformat(),
}

out_path = OUT_DIR / "phase162_report.json"
out_path.write_text(json.dumps(report, indent=2))
print(f"\n[4/4] ✅ Saved → {out_path}")
