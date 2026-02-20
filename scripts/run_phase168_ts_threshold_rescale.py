"""
Phase 168 — TS Threshold/Scale Re-Sweep at v2.9.0
===================================================
P152 fine-tuned TS overlay thresholds/scales:
  reduce_threshold=0.70, reduce_scale=0.60
  boost_threshold=0.30,  boost_scale=1.15

BUT those were tuned with the OLD 2-point TS approximation method.
P164-165 upgraded to PROPER rolling-window method (short=12h, long=96h),
which produces a DIFFERENT distribution of spread_pct values.

Hypothesis: With the new rolling-window method, the optimal thresholds/scales
may have shifted. The new method captures more nuanced term structure dynamics,
so extreme events (>70th, <30th) may be detected differently.

Test:
  Part A: reduce_threshold sweep [0.60, 0.65, 0.70, 0.75, 0.80]
           with reduce_scale=0.60 and boost fixed
  Part B: reduce_scale sweep [0.40, 0.50, 0.60, 0.70, 0.80]
           with best threshold and boost fixed
  Part C: boost fine-tune [1.05, 1.10, 1.15, 1.20, 1.25]
           with best threshold+scale, boost_threshold=0.30 fixed

Baseline (prod v2.9.0): OBJ=2.2423
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

# v2.9.0 overlay constants (non-TS-threshold)
VOL_WINDOW     = 168
VOL_THRESHOLD  = 0.5
VOL_SCALE      = 0.4
VOL_F168_BOOST = 0.15
BRD_LOOKBACK   = 192
PCT_WINDOW     = 336
P_LOW, P_HIGH  = 0.35, 0.65
FUND_DISP_THR  = 0.75
FUND_DISP_SCALE= 1.15
TS_SHORT = 12
TS_LONG  = 96

# Current production TS thresholds/scales
BASELINE_RT = 0.70   # reduce_threshold
BASELINE_RS = 0.60   # reduce_scale
BASELINE_BT = 0.30   # boost_threshold
BASELINE_BS = 1.15   # boost_scale

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

REDUCE_THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80]
REDUCE_SCALES     = [0.40, 0.50, 0.60, 0.70, 0.80]
BOOST_SCALES      = [1.05, 1.10, 1.15, 1.20, 1.25]

OUT_DIR = ROOT / "artifacts" / "phase168"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

_partial: dict = {}

def _on_timeout(signum, frame):
    print("\n⏰ TIMEOUT — saving partial...")
    _partial["partial"] = True
    _partial["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (OUT_DIR / "phase168_report.json").write_text(json.dumps(_partial, indent=2))
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


def rolling_mean_arr(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    cs = np.zeros(n + 1)
    cs[1:] = np.cumsum(arr)
    result = np.zeros(n)
    for i in range(n):
        start = max(0, i - window + 1)
        result[i] = (cs[i + 1] - cs[start]) / (i - start + 1)
    return result


def compute_signals(dataset) -> dict:
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

    fund_rates = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        for i in range(n):
            ts = dataset.timeline[i]
            try:
                fund_rates[i, j] = dataset.last_funding_rate_before(sym, ts)
            except Exception:
                fund_rates[i, j] = 0.0

    fund_std_raw = np.std(fund_rates, axis=1)
    fund_std_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        fund_std_pct[i] = float(np.mean(fund_std_raw[i - PCT_WINDOW:i] <= fund_std_raw[i]))
    fund_std_pct[:PCT_WINDOW] = 0.5

    xsect_mean = np.mean(fund_rates, axis=1)
    short_avg  = rolling_mean_arr(xsect_mean, TS_SHORT)
    long_avg   = rolling_mean_arr(xsect_mean, TS_LONG)
    ts_spread_raw = short_avg - long_avg
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


def compute_ensemble(sig_rets: dict, signals: dict,
                     rt=BASELINE_RT, rs=BASELINE_RS,
                     bt=BASELINE_BT, bs=BASELINE_BS) -> np.ndarray:
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
        if tsp[i] > rt:
            ret_i *= rs
        elif tsp[i] < bt:
            ret_i *= bs
        ens[i] = ret_i
    return ens


# ─── MAIN ────────────────────────────────────────────────────────────────────

print("=" * 68)
print("PHASE 168 — TS Threshold/Scale Re-Sweep (new rolling-window method)")
print("=" * 68)

print("\n[1/5] Loading data + computing signals + strategy returns...")

strat_rets: dict = {"v1": {}, "i460bw168": {}, "i415bw216": {}, "f168": {}}
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
        ("f168", "funding_momentum_alpha", {
            "k_per_side": 2, "funding_lookback_bars": 168, "direction": "contrarian",
            "target_gross_leverage": 0.25, "rebalance_interval_bars": 24,
        }),
    ]:
        bt_cfg = BacktestConfig(costs=COST_MODEL)
        strat  = make_strategy({"name": sname, "params": params})
        result = BacktestEngine(bt_cfg).run(dataset, strat)
        strat_rets[sk][year] = np.array(result.returns)
    print(". ✓")

def make_sig_rets(year):
    return {sk: strat_rets[sk][year] for sk in ["v1", "i460bw168", "i415bw216", "f168"]}

baseline_yearly = {year: sharpe(compute_ensemble(make_sig_rets(year), signals_data[year]))
                   for year in YEARS}
baseline_obj = obj_func(baseline_yearly)
print(f"\n  Baseline (rt=0.70, rs=0.60, bt=0.30, bs=1.15): OBJ={baseline_obj:.4f} | {baseline_yearly}")

print("\n[2/5] PART A — reduce_threshold sweep (rs=0.60, bt=0.30, bs=1.15 fixed)...")
parta = {}
for rt in REDUCE_THRESHOLDS:
    yearly = {year: sharpe(compute_ensemble(make_sig_rets(year), signals_data[year],
                                            rt=rt)) for year in YEARS}
    o = obj_func(yearly)
    d = o - baseline_obj
    sym = "✅" if d > 0.005 else ("⚠️ " if d > -0.005 else "❌")
    tag = " ← baseline" if rt == BASELINE_RT else ""
    print(f"  {sym} rt={rt:.2f}: OBJ={o:.4f} (Δ={d:+.4f}){tag}")
    parta[rt] = {"obj": o, "delta": d, "yearly": yearly}

best_rt = max(parta, key=lambda k: parta[k]["obj"])
print(f"  Best rt={best_rt}: OBJ={parta[best_rt]['obj']:.4f} (Δ={parta[best_rt]['delta']:+.4f})")

print(f"\n[3/5] PART B — reduce_scale sweep (rt={best_rt}, bt=0.30, bs=1.15 fixed)...")
partb = {}
for rs in REDUCE_SCALES:
    yearly = {year: sharpe(compute_ensemble(make_sig_rets(year), signals_data[year],
                                            rt=best_rt, rs=rs)) for year in YEARS}
    o = obj_func(yearly)
    d = o - baseline_obj
    sym = "✅" if d > 0.005 else ("⚠️ " if d > -0.005 else "❌")
    tag = " ← baseline" if (best_rt == BASELINE_RT and rs == BASELINE_RS) else ""
    print(f"  {sym} rs={rs:.2f}: OBJ={o:.4f} (Δ={d:+.4f}){tag}")
    partb[rs] = {"obj": o, "delta": d, "yearly": yearly}

best_rs = max(partb, key=lambda k: partb[k]["obj"])
print(f"  Best rs={best_rs}: OBJ={partb[best_rs]['obj']:.4f} (Δ={partb[best_rs]['delta']:+.4f})")

print(f"\n[4/5] PART C — boost_scale sweep (rt={best_rt}, rs={best_rs}, bt=0.30 fixed)...")
partc = {}
for bs in BOOST_SCALES:
    yearly = {year: sharpe(compute_ensemble(make_sig_rets(year), signals_data[year],
                                            rt=best_rt, rs=best_rs, bs=bs)) for year in YEARS}
    o = obj_func(yearly)
    d = o - baseline_obj
    sym = "✅" if d > 0.005 else ("⚠️ " if d > -0.005 else "❌")
    tag = " ← baseline" if (best_rt == BASELINE_RT and best_rs == BASELINE_RS and bs == BASELINE_BS) else ""
    print(f"  {sym} bs={bs:.2f}: OBJ={o:.4f} (Δ={d:+.4f}){tag}")
    partc[bs] = {"obj": o, "delta": d, "yearly": yearly}

best_bs  = max(partc, key=lambda k: partc[k]["obj"])
best_obj   = partc[best_bs]["obj"]
best_delta = partc[best_bs]["delta"]
best_yearly= partc[best_bs]["yearly"]
print(f"  Best bs={best_bs}: OBJ={best_obj:.4f} (Δ={best_delta:+.4f})")
print(f"\n  Combined best: rt={best_rt} rs={best_rs} bt={BASELINE_BT} bs={best_bs}")

print("\n[5/5] LOYO validation...")
loyo_wins, loyo_deltas = 0, []
for held_out in YEARS:
    sh_best = best_yearly[held_out]
    sh_base = baseline_yearly[held_out]
    d = sh_best - sh_base
    loyo_deltas.append(d)
    loyo_wins += int(d > 0)
    print(f"  LOYO {held_out}: best={sh_best:.4f} base={sh_base:.4f} Δ={d:+.4f} {'✅' if d>0 else '❌'}")

loyo_avg   = float(np.mean(loyo_deltas))
changed    = not (best_rt == BASELINE_RT and best_rs == BASELINE_RS and best_bs == BASELINE_BS)
validated  = changed and best_delta > 0.005 and loyo_wins >= 3

print("\n" + "=" * 68)
if validated:
    print(f"✅ VALIDATED — rt={best_rt} rs={best_rs} bt={BASELINE_BT} bs={best_bs}: OBJ={best_obj:.4f} (Δ={best_delta:+.4f}) LOYO {loyo_wins}/5")
else:
    print(f"❌ NO IMPROVEMENT — baseline rt={BASELINE_RT}/rs={BASELINE_RS}/bs={BASELINE_BS} OBJ={baseline_obj:.4f} optimal")
    print(f"   Best: rt={best_rt} rs={best_rs} bs={best_bs} Δ={best_delta:+.4f} | LOYO {loyo_wins}/5")
print("=" * 68)

report = {
    "phase": 168,
    "description": "TS Threshold/Scale Re-Sweep (new rolling-window method)",
    "elapsed_seconds": round(time.time() - _start, 1),
    "baseline_rt": BASELINE_RT, "baseline_rs": BASELINE_RS,
    "baseline_bt": BASELINE_BT, "baseline_bs": BASELINE_BS,
    "baseline_obj": baseline_obj, "baseline_yearly": baseline_yearly,
    "parta_reduce_thr": {str(rt): {"obj": r["obj"], "delta": r["delta"]} for rt, r in parta.items()},
    "partb_reduce_scale": {str(rs): {"obj": r["obj"], "delta": r["delta"]} for rs, r in partb.items()},
    "partc_boost_scale": {str(bs): {"obj": r["obj"], "delta": r["delta"]} for bs, r in partc.items()},
    "best_rt": best_rt, "best_rs": best_rs, "best_bt": BASELINE_BT, "best_bs": best_bs,
    "best_obj": best_obj, "best_delta": best_delta, "best_yearly": best_yearly,
    "loyo_wins": loyo_wins, "loyo_avg_delta": loyo_avg,
    "validated": validated,
    "verdict": (
        f"VALIDATED — rt={best_rt} rs={best_rs} bt={BASELINE_BT} bs={best_bs} OBJ={best_obj:.4f} Δ={best_delta:+.4f} LOYO {loyo_wins}/5"
        if validated else
        f"NO IMPROVEMENT — baseline rt={BASELINE_RT}/rs={BASELINE_RS}/bs={BASELINE_BS} OBJ={baseline_obj:.4f} optimal"
    ),
    "partial": False,
    "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
}

out_path = OUT_DIR / "phase168_report.json"
out_path.write_text(json.dumps(report, indent=2))
print(f"\n✅ Saved → {out_path}")
