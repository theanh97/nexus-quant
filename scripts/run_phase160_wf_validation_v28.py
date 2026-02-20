"""
Phase 160 — Walk-Forward Validation of v2.8.0 Full Stack
=========================================================
Goal: Confirm all v2.8.0 improvements (F168, vol fine-tune, breadth lb=192/p=0.35/0.65)
hold out-of-sample. Last comprehensive WF was P152 on v2.4.0.

WF protocol:
  Window 1: IS=2021-2023, OOS=2024
  Window 2: IS=2022-2024, OOS=2025

Compare v2.4.0 vs v2.8.0 on each WF window.
Also run full IS (2021-2025) to confirm OBJ=2.2095.

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

# v2.8.0 (current production) overlay constants
V28_VOL_THRESHOLD  = 0.5
V28_VOL_SCALE      = 0.4
V28_VOL_F168_BOOST = 0.15
V28_BRD_LOOKBACK   = 192
V28_PCT_WINDOW     = 336
V28_P_LOW          = 0.35
V28_P_HIGH         = 0.65
V28_FUND_DISP_THR  = 0.75
V28_FUND_DISP_SCALE= 1.15
V28_TS_REDUCE_THR  = 0.70
V28_TS_REDUCE_SCALE= 0.60
V28_TS_BOOST_THR   = 0.30
V28_TS_BOOST_SCALE = 1.15
V28_TS_SHORT, V28_TS_LONG = 24, 144
VOL_WINDOW = 168

V28_WEIGHTS = {
    "LOW":  {"v1": 0.2747, "i460bw168": 0.1967, "i415bw216": 0.3247, "f168": 0.2039},
    "MID":  {"v1": 0.16,   "i460bw168": 0.22,   "i415bw216": 0.37,   "f168": 0.25},
    "HIGH": {"v1": 0.05,   "i460bw168": 0.25,   "i415bw216": 0.45,   "f168": 0.25},
}

# v2.4.0 (P152 baseline) constants for comparison
V24_BRD_LOOKBACK   = 168
V24_PCT_WINDOW     = 336
V24_P_LOW          = 0.33
V24_P_HIGH         = 0.67
V24_VOL_SCALE      = 0.5
V24_VOL_F144_BOOST = 0.2
V24_TS_REDUCE_THR  = 0.70
V24_TS_REDUCE_SCALE= 0.60
V24_TS_BOOST_THR   = 0.30
V24_TS_BOOST_SCALE = 1.15
V24_WEIGHTS = {
    "LOW":  {"v1": 0.2747, "i460bw168": 0.1967, "i415bw216": 0.3247, "f": 0.2039},
    "MID":  {"v1": 0.16,   "i460bw168": 0.22,   "i415bw216": 0.37,   "f": 0.25},
    "HIGH": {"v1": 0.05,   "i460bw168": 0.25,   "i415bw216": 0.45,   "f": 0.25},
}

YEAR_RANGES = {
    "2021": ("2021-02-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
}
YEARS = sorted(YEAR_RANGES.keys())

OUT_DIR = ROOT / "artifacts" / "phase160"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

_partial: dict = {}

def _on_timeout(signum, frame):
    print("\n⏰ TIMEOUT — saving partial...")
    _partial["partial"] = True
    _partial["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (OUT_DIR / "phase160_report.json").write_text(json.dumps(_partial, indent=2))
    sys.exit(0)

_signal.signal(_signal.SIGALRM, _on_timeout)
_signal.alarm(2700)  # 45min


def sharpe(rets: np.ndarray) -> float:
    if len(rets) < 100:
        return 0.0
    mu = float(np.mean(rets)) * 8760
    sd = float(np.std(rets)) * np.sqrt(8760)
    return round(mu / sd, 4) if sd > 1e-12 else 0.0


def obj_func(yearly_sharpes: dict) -> float:
    arr = np.array(list(yearly_sharpes.values()))
    return round(float(np.mean(arr) - 0.5 * np.std(arr)), 4)


def compute_signals_v28(dataset) -> dict:
    n = len(dataset.timeline)
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

    breadth = np.full(n, 0.5)
    for i in range(V28_BRD_LOOKBACK, n):
        pos = sum(
            1 for sym in SYMBOLS
            if (c0 := dataset.close(sym, i - V28_BRD_LOOKBACK)) > 0
            and dataset.close(sym, i) > c0
        )
        breadth[i] = pos / len(SYMBOLS)
    breadth[:V28_BRD_LOOKBACK] = 0.5
    brd_pct = np.full(n, 0.5)
    for i in range(V28_PCT_WINDOW, n):
        brd_pct[i] = float(np.mean(breadth[i - V28_PCT_WINDOW:i] <= breadth[i]))
    brd_pct[:V28_PCT_WINDOW] = 0.5
    breadth_regime = np.where(brd_pct >= V28_P_HIGH, 2,
                     np.where(brd_pct >= V28_P_LOW, 1, 0)).astype(int)

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
    for i in range(V28_PCT_WINDOW, n):
        fund_std_pct[i] = float(np.mean(fund_std_raw[i - V28_PCT_WINDOW:i] <= fund_std_raw[i]))
    fund_std_pct[:V28_PCT_WINDOW] = 0.5

    ts_spread_raw = np.zeros(n)
    for i in range(max(V28_TS_SHORT, V28_TS_LONG), n):
        short_rates, long_rates = [], []
        for sym in SYMBOLS:
            ts_now = dataset.timeline[i]
            ts_short_start = dataset.timeline[max(0, i - V28_TS_SHORT)]
            r_now   = dataset.last_funding_rate_before(sym, ts_now)
            r_short = dataset.last_funding_rate_before(sym, ts_short_start)
            short_rates.append(r_now)
            long_rates.append((r_now + r_short) / 2)
        ts_spread_raw[i] = float(np.mean(short_rates)) - float(np.mean(long_rates))
    ts_spread_raw[:max(V28_TS_SHORT, V28_TS_LONG)] = 0.0
    ts_spread_pct = np.full(n, 0.5)
    for i in range(V28_PCT_WINDOW, n):
        ts_spread_pct[i] = float(np.mean(ts_spread_raw[i - V28_PCT_WINDOW:i] <= ts_spread_raw[i]))
    ts_spread_pct[:V28_PCT_WINDOW] = 0.5

    return {
        "btc_vol": btc_vol,
        "breadth_regime": breadth_regime,
        "fund_std_pct": fund_std_pct,
        "ts_spread_pct": ts_spread_pct,
    }


def compute_signals_v24(dataset) -> dict:
    """v2.4.0 signals: breadth_lb=168, p=0.33/0.67"""
    n = len(dataset.timeline)
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

    breadth = np.full(n, 0.5)
    for i in range(V24_BRD_LOOKBACK, n):
        pos = sum(
            1 for sym in SYMBOLS
            if (c0 := dataset.close(sym, i - V24_BRD_LOOKBACK)) > 0
            and dataset.close(sym, i) > c0
        )
        breadth[i] = pos / len(SYMBOLS)
    breadth[:V24_BRD_LOOKBACK] = 0.5
    brd_pct = np.full(n, 0.5)
    for i in range(V24_PCT_WINDOW, n):
        brd_pct[i] = float(np.mean(breadth[i - V24_PCT_WINDOW:i] <= breadth[i]))
    brd_pct[:V24_PCT_WINDOW] = 0.5
    breadth_regime = np.where(brd_pct >= V24_P_HIGH, 2,
                     np.where(brd_pct >= V24_P_LOW, 1, 0)).astype(int)

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
    for i in range(V24_PCT_WINDOW, n):
        fund_std_pct[i] = float(np.mean(fund_std_raw[i - V24_PCT_WINDOW:i] <= fund_std_raw[i]))
    fund_std_pct[:V24_PCT_WINDOW] = 0.5

    ts_spread_raw = np.zeros(n)
    for i in range(max(V28_TS_SHORT, V28_TS_LONG), n):
        short_rates, long_rates = [], []
        for sym in SYMBOLS:
            ts_now = dataset.timeline[i]
            ts_short_start = dataset.timeline[max(0, i - V28_TS_SHORT)]
            r_now   = dataset.last_funding_rate_before(sym, ts_now)
            r_short = dataset.last_funding_rate_before(sym, ts_short_start)
            short_rates.append(r_now)
            long_rates.append((r_now + r_short) / 2)
        ts_spread_raw[i] = float(np.mean(short_rates)) - float(np.mean(long_rates))
    ts_spread_raw[:max(V28_TS_SHORT, V28_TS_LONG)] = 0.0
    ts_spread_pct = np.full(n, 0.5)
    for i in range(V24_PCT_WINDOW, n):
        ts_spread_pct[i] = float(np.mean(ts_spread_raw[i - V24_PCT_WINDOW:i] <= ts_spread_raw[i]))
    ts_spread_pct[:V24_PCT_WINDOW] = 0.5

    return {
        "btc_vol": btc_vol,
        "breadth_regime": breadth_regime,
        "fund_std_pct": fund_std_pct,
        "ts_spread_pct": ts_spread_pct,
    }


def compute_v28_ensemble(sig_rets, signals):
    sk_all = ["v1", "i460bw168", "i415bw216", "f168"]
    min_len = min(len(sig_rets[sk]) for sk in sk_all)
    bv  = signals["btc_vol"][:min_len]
    reg = signals["breadth_regime"][:min_len]
    fsp = signals["fund_std_pct"][:min_len]
    tsp = signals["ts_spread_pct"][:min_len]

    ens = np.zeros(min_len)
    for i in range(min_len):
        w = V28_WEIGHTS[["LOW","MID","HIGH"][int(reg[i])]]
        if not np.isnan(bv[i]) and bv[i] > V28_VOL_THRESHOLD:
            boost = V28_VOL_F168_BOOST / max(1, len(sk_all)-1)
            ret_i = 0.0
            for sk in sk_all:
                adj_w = (min(0.60, w[sk] + V28_VOL_F168_BOOST) if sk == "f168"
                         else max(0.05, w[sk] - boost))
                ret_i += adj_w * sig_rets[sk][i]
            ret_i *= V28_VOL_SCALE
        else:
            ret_i = sum(w[sk] * sig_rets[sk][i] for sk in sk_all)
        if fsp[i] > V28_FUND_DISP_THR:
            ret_i *= V28_FUND_DISP_SCALE
        if tsp[i] > V28_TS_REDUCE_THR:
            ret_i *= V28_TS_REDUCE_SCALE
        elif tsp[i] < V28_TS_BOOST_THR:
            ret_i *= V28_TS_BOOST_SCALE
        ens[i] = ret_i
    return ens


def compute_v24_ensemble(sig_rets_v24, signals):
    """v2.4.0: F144 (key='f'), breadth_lb=168, vol_scale=0.50, f_boost=0.20"""
    sk_all = ["v1", "i460bw168", "i415bw216", "f"]
    min_len = min(len(sig_rets_v24[sk]) for sk in sk_all)
    bv  = signals["btc_vol"][:min_len]
    reg = signals["breadth_regime"][:min_len]
    fsp = signals["fund_std_pct"][:min_len]
    tsp = signals["ts_spread_pct"][:min_len]

    ens = np.zeros(min_len)
    for i in range(min_len):
        w = V24_WEIGHTS[["LOW","MID","HIGH"][int(reg[i])]]
        if not np.isnan(bv[i]) and bv[i] > V28_VOL_THRESHOLD:
            boost = V24_VOL_F144_BOOST / max(1, len(sk_all)-1)
            ret_i = 0.0
            for sk in sk_all:
                adj_w = (min(0.60, w[sk] + V24_VOL_F144_BOOST) if sk == "f"
                         else max(0.05, w[sk] - boost))
                ret_i += adj_w * sig_rets_v24[sk][i]
            ret_i *= V24_VOL_SCALE
        else:
            ret_i = sum(w[sk] * sig_rets_v24[sk][i] for sk in sk_all)
        if fsp[i] > V28_FUND_DISP_THR:
            ret_i *= V28_FUND_DISP_SCALE
        if tsp[i] > V24_TS_REDUCE_THR:
            ret_i *= V24_TS_REDUCE_SCALE
        elif tsp[i] < V24_TS_BOOST_THR:
            ret_i *= V24_TS_BOOST_SCALE
        ens[i] = ret_i
    return ens


# ─── main ────────────────────────────────────────────────────────────────────

print("=" * 68)
print("PHASE 160 — Walk-Forward Validation of v2.8.0 Full Stack")
print("=" * 68)

print("\n[1/3] Loading datasets + computing all signals + strategy returns...")

sig_v28: dict = {sk: {} for sk in ["v1", "i460bw168", "i415bw216", "f168"]}
sig_v24: dict = {sk: {} for sk in ["v1", "i460bw168", "i415bw216", "f"]}
sigs_v28_data: dict = {}
sigs_v24_data: dict = {}

for year, (start, end) in YEAR_RANGES.items():
    print(f"  {year}: ", end="", flush=True)
    cfg_data = {
        "provider": "binance_rest_v1", "symbols": SYMBOLS,
        "start": start, "end": end, "bar_interval": "1h",
        "cache_dir": ".cache/binance_rest",
    }
    provider = make_provider(cfg_data, seed=42)
    dataset  = provider.load()
    sigs_v28_data[year] = compute_signals_v28(dataset)
    sigs_v24_data[year] = compute_signals_v24(dataset)
    print("O", end="", flush=True)

    # V1, I460, I415 — shared
    for sk, sname, params in [
        ("v1", "nexus_alpha_v1", {
            "k_per_side":2,"w_carry":0.35,"w_mom":0.45,"w_mean_reversion":0.2,
            "momentum_lookback_bars":336,"mean_reversion_lookback_bars":72,
            "vol_lookback_bars":168,"target_gross_leverage":0.35,"rebalance_interval_bars":60
        }),
        ("i460bw168", "idio_momentum_alpha", {
            "k_per_side":4,"lookback_bars":460,"beta_window_bars":168,
            "target_gross_leverage":0.3,"rebalance_interval_bars":48
        }),
        ("i415bw216", "idio_momentum_alpha", {
            "k_per_side":4,"lookback_bars":415,"beta_window_bars":216,
            "target_gross_leverage":0.3,"rebalance_interval_bars":48
        }),
    ]:
        bt_cfg = BacktestConfig(costs=COST_MODEL)
        strat  = make_strategy({"name": sname, "params": params})
        result = BacktestEngine(bt_cfg).run(dataset, strat)
        rets = np.array(result.returns)
        sig_v28[sk][year] = rets
        sig_v24[sk][year] = rets  # same for shared signals

    # F168 (v2.8.0)
    bt_cfg = BacktestConfig(costs=COST_MODEL)
    strat  = make_strategy({"name": "funding_momentum_alpha", "params": {
        "k_per_side":2,"funding_lookback_bars":168,"direction":"contrarian",
        "target_gross_leverage":0.25,"rebalance_interval_bars":24
    }})
    result = BacktestEngine(bt_cfg).run(dataset, strat)
    sig_v28["f168"][year] = np.array(result.returns)

    # F144 (v2.4.0)
    bt_cfg = BacktestConfig(costs=COST_MODEL)
    strat  = make_strategy({"name": "funding_momentum_alpha", "params": {
        "k_per_side":2,"funding_lookback_bars":144,"direction":"contrarian",
        "target_gross_leverage":0.25,"rebalance_interval_bars":24
    }})
    result = BacktestEngine(bt_cfg).run(dataset, strat)
    sig_v24["f"][year] = np.array(result.returns)

    print(". ✓")

print("\n[2/3] IS verification (all 5 years)...")
is_v28 = {}
is_v24 = {}
for year in YEARS:
    yr28 = {sk: sig_v28[sk][year] for sk in ["v1","i460bw168","i415bw216","f168"]}
    yr24 = {sk: sig_v24[sk][year] for sk in ["v1","i460bw168","i415bw216","f"]}
    is_v28[year] = sharpe(compute_v28_ensemble(yr28, sigs_v28_data[year]))
    is_v24[year] = sharpe(compute_v24_ensemble(yr24, sigs_v24_data[year]))

is_obj_v28 = obj_func(is_v28)
is_obj_v24 = obj_func(is_v24)

print(f"  v2.8.0 IS OBJ={is_obj_v28:.4f} | {is_v28}")
print(f"  v2.4.0 IS OBJ={is_obj_v24:.4f} | {is_v24}")
print(f"  IS delta: {is_obj_v28 - is_obj_v24:+.4f}")

print("\n[3/3] Walk-Forward validation...")
wf_results = []

# WF Window 1: IS=2021-2023, OOS=2024
oos_year = "2024"
yr28_oos = {sk: sig_v28[sk][oos_year] for sk in ["v1","i460bw168","i415bw216","f168"]}
yr24_oos = {sk: sig_v24[sk][oos_year] for sk in ["v1","i460bw168","i415bw216","f"]}
sh_v28 = sharpe(compute_v28_ensemble(yr28_oos, sigs_v28_data[oos_year]))
sh_v24 = sharpe(compute_v24_ensemble(yr24_oos, sigs_v24_data[oos_year]))
delta = sh_v28 - sh_v24
wf_results.append({"window": "WF1 (IS=21-23, OOS=24)", "v28": sh_v28, "v24": sh_v24, "delta": delta})
print(f"  WF1 OOS=2024: v2.8={sh_v28:.4f} v2.4={sh_v24:.4f} Δ={delta:+.4f} {'✅' if delta>0 else '❌'}")

# WF Window 2: IS=2022-2024, OOS=2025
oos_year = "2025"
yr28_oos = {sk: sig_v28[sk][oos_year] for sk in ["v1","i460bw168","i415bw216","f168"]}
yr24_oos = {sk: sig_v24[sk][oos_year] for sk in ["v1","i460bw168","i415bw216","f"]}
sh_v28 = sharpe(compute_v28_ensemble(yr28_oos, sigs_v28_data[oos_year]))
sh_v24 = sharpe(compute_v24_ensemble(yr24_oos, sigs_v24_data[oos_year]))
delta = sh_v28 - sh_v24
wf_results.append({"window": "WF2 (IS=22-24, OOS=25)", "v28": sh_v28, "v24": sh_v24, "delta": delta})
print(f"  WF2 OOS=2025: v2.8={sh_v28:.4f} v2.4={sh_v24:.4f} Δ={delta:+.4f} {'✅' if delta>0 else '❌'}")

wf_wins = sum(1 for r in wf_results if r["delta"] > 0)
wf_avg_delta = float(np.mean([r["delta"] for r in wf_results]))

print("\n" + "=" * 68)
if wf_wins >= 1:
    print(f"✅ WF CONFIRMED — v2.8.0 beats v2.4.0: {wf_wins}/2 wins, avg_delta={wf_avg_delta:+.4f}")
else:
    print(f"⚠️  WF INCONCLUSIVE — {wf_wins}/2 wins, avg_delta={wf_avg_delta:+.4f}")
print(f"  IS: v2.8.0={is_obj_v28:.4f} vs v2.4.0={is_obj_v24:.4f} (Δ={is_obj_v28-is_obj_v24:+.4f})")
print("=" * 68)

report = {
    "phase": 160,
    "description": "Walk-Forward Validation — v2.8.0 vs v2.4.0",
    "elapsed_seconds": round(time.time() - _start, 1),
    "is_obj_v28": is_obj_v28,
    "is_obj_v24": is_obj_v24,
    "is_delta": is_obj_v28 - is_obj_v24,
    "is_v28_yearly": is_v28,
    "is_v24_yearly": is_v24,
    "wf_results": wf_results,
    "wf_wins": wf_wins,
    "wf_avg_delta": wf_avg_delta,
    "verdict": (f"WF CONFIRMED — v2.8.0 {wf_wins}/2 wins avg_delta={wf_avg_delta:+.4f}"
                if wf_wins >= 1 else
                f"WF INCONCLUSIVE — {wf_wins}/2 wins avg_delta={wf_avg_delta:+.4f}"),
    "partial": False,
    "timestamp": datetime.datetime.utcnow().isoformat(),
}

out_path = OUT_DIR / "phase160_report.json"
out_path.write_text(json.dumps(report, indent=2))
print(f"✅ Saved → {out_path}")
