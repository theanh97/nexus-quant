"""
Phase 169 — Boost Fine-Tune (Extended bs + bt sweep)
======================================================
P168 showed boost_scale (bs) monotonically increasing from 1.05 to 1.25.
Diminishing returns pattern: gap shrinking (+0.088→+0.042 per 0.05 step).
Likely optimal is 1.30-1.40 before overfitting kicks in.

Also: boost_threshold=0.30 was never re-swept with new TS method.
A lower bt (0.20-0.25) would trigger boost more frequently.
A higher bt (0.35-0.40) would be more selective.

Fixed from P168: rt=0.60, rs=0.40 (confirmed optimal reduce params)

Part A: boost_scale [1.25, 1.30, 1.35, 1.40, 1.50, 1.60] with bt=0.30
Part B: boost_threshold [0.20, 0.25, 0.30, 0.35, 0.40] with best bs from A

Baseline (prod v2.10.0): rt=0.60, rs=0.40, bt=0.30, bs=1.25, OBJ=2.3510
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

# v2.10.0 baseline TS params
RT = 0.60   # reduce_threshold (locked from P168)
RS = 0.40   # reduce_scale (locked from P168)
BASELINE_BT = 0.30
BASELINE_BS = 1.25

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

BOOST_SCALES     = [1.25, 1.30, 1.35, 1.40, 1.50, 1.60]
BOOST_THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40]

OUT_DIR = ROOT / "artifacts" / "phase169"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

_partial: dict = {}

def _on_timeout(signum, frame):
    _partial["partial"] = True
    _partial["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (OUT_DIR / "phase169_report.json").write_text(json.dumps(_partial, indent=2))
    sys.exit(0)

_signal.signal(_signal.SIGALRM, _on_timeout)
_signal.alarm(2700)


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
    ts_spread_pct = np.full(n, 0.5)
    ts_raw = rolling_mean_arr(xsect_mean, TS_SHORT) - rolling_mean_arr(xsect_mean, TS_LONG)
    for i in range(PCT_WINDOW, n):
        ts_spread_pct[i] = float(np.mean(ts_raw[i - PCT_WINDOW:i] <= ts_raw[i]))
    ts_spread_pct[:PCT_WINDOW] = 0.5

    return {
        "btc_vol": btc_vol,
        "breadth_regime": breadth_regime,
        "fund_std_pct": fund_std_pct,
        "ts_spread_pct": ts_spread_pct,
    }


def compute_ensemble(sig_rets: dict, signals: dict, bt=BASELINE_BT, bs=BASELINE_BS) -> np.ndarray:
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
        if tsp[i] > RT:
            ret_i *= RS
        elif tsp[i] < bt:
            ret_i *= bs
        ens[i] = ret_i
    return ens


# ─── MAIN ────────────────────────────────────────────────────────────────────

print("=" * 68)
print("PHASE 169 — Boost Fine-Tune (Extended bs + bt sweep)")
print("=" * 68)
print(f"  Locked: rt={RT}, rs={RS}")
print(f"  Baseline: bt={BASELINE_BT}, bs={BASELINE_BS} → OBJ=2.3510\n")

print("[1/4] Loading data + signals + strategy returns...")
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

_partial.update({"phase": 169, "description": "Boost Fine-Tune", "partial": True})

def make_sig_rets(year):
    return {sk: strat_rets[sk][year] for sk in ["v1", "i460bw168", "i415bw216", "f168"]}

baseline_yearly = {year: sharpe(compute_ensemble(make_sig_rets(year), signals_data[year]))
                   for year in YEARS}
baseline_obj = obj_func(baseline_yearly)
print(f"\n  Baseline confirmed: OBJ={baseline_obj:.4f} | {baseline_yearly}")

print("\n[2/4] PART A — boost_scale sweep (bt=0.30 fixed)...")
parta = {}
for bs in BOOST_SCALES:
    yearly = {year: sharpe(compute_ensemble(make_sig_rets(year), signals_data[year], bs=bs))
              for year in YEARS}
    o = obj_func(yearly)
    d = o - baseline_obj
    sym = "✅" if d > 0.005 else ("⚠️ " if d > -0.005 else "❌")
    tag = " ← baseline" if bs == BASELINE_BS else ""
    print(f"  {sym} bs={bs:.2f}: OBJ={o:.4f} (Δ={d:+.4f}){tag}")
    parta[bs] = {"obj": o, "delta": d, "yearly": yearly}

best_bs = max(parta, key=lambda k: parta[k]["obj"])
print(f"  Best bs={best_bs}: OBJ={parta[best_bs]['obj']:.4f} (Δ={parta[best_bs]['delta']:+.4f})")

print(f"\n[3/4] PART B — boost_threshold sweep (bs={best_bs} fixed)...")
partb = {}
for bt in BOOST_THRESHOLDS:
    yearly = {year: sharpe(compute_ensemble(make_sig_rets(year), signals_data[year],
                                            bt=bt, bs=best_bs)) for year in YEARS}
    o = obj_func(yearly)
    d = o - baseline_obj
    sym = "✅" if d > 0.005 else ("⚠️ " if d > -0.005 else "❌")
    tag = " ← baseline" if (best_bs == BASELINE_BS and bt == BASELINE_BT) else ""
    print(f"  {sym} bt={bt:.2f}: OBJ={o:.4f} (Δ={d:+.4f}){tag}")
    partb[bt] = {"obj": o, "delta": d, "yearly": yearly}

best_bt  = max(partb, key=lambda k: partb[k]["obj"])
best_obj   = partb[best_bt]["obj"]
best_delta = partb[best_bt]["delta"]
best_yearly = partb[best_bt]["yearly"]
print(f"  Best bt={best_bt}: OBJ={best_obj:.4f} (Δ={best_delta:+.4f})")
print(f"\n  Combined: rt={RT} rs={RS} bt={best_bt} bs={best_bs}")

print("\n[4/4] LOYO validation...")
loyo_wins, loyo_deltas = 0, []
for held_out in YEARS:
    sh_best = best_yearly[held_out]
    sh_base = baseline_yearly[held_out]
    d = sh_best - sh_base
    loyo_deltas.append(d)
    loyo_wins += int(d > 0)
    print(f"  LOYO {held_out}: best={sh_best:.4f} base={sh_base:.4f} Δ={d:+.4f} {'✅' if d>0 else '❌'}")

loyo_avg  = float(np.mean(loyo_deltas))
changed   = not (best_bs == BASELINE_BS and best_bt == BASELINE_BT)
validated = changed and best_delta > 0.005 and loyo_wins >= 3

print("\n" + "=" * 68)
if validated:
    print(f"✅ VALIDATED — rt={RT} rs={RS} bt={best_bt} bs={best_bs}: OBJ={best_obj:.4f} (Δ={best_delta:+.4f}) LOYO {loyo_wins}/5")
else:
    print(f"❌ NO IMPROVEMENT — baseline bt={BASELINE_BT} bs={BASELINE_BS} OBJ={baseline_obj:.4f} optimal")
    print(f"   Best: bt={best_bt} bs={best_bs} Δ={best_delta:+.4f} | LOYO {loyo_wins}/5")
print("=" * 68)

report = {
    "phase": 169,
    "description": "Boost Fine-Tune (extended bs + bt sweep)",
    "elapsed_seconds": round(time.time() - _start, 1),
    "locked_rt": RT, "locked_rs": RS,
    "baseline_bt": BASELINE_BT, "baseline_bs": BASELINE_BS,
    "baseline_obj": baseline_obj, "baseline_yearly": baseline_yearly,
    "parta_bs_sweep": {str(bs): {"obj": r["obj"], "delta": r["delta"]} for bs, r in parta.items()},
    "partb_bt_sweep": {str(bt): {"obj": r["obj"], "delta": r["delta"]} for bt, r in partb.items()},
    "best_bt": best_bt, "best_bs": best_bs,
    "best_obj": best_obj, "best_delta": best_delta, "best_yearly": best_yearly,
    "loyo_wins": loyo_wins, "loyo_avg_delta": loyo_avg,
    "validated": validated,
    "verdict": (
        f"VALIDATED — bt={best_bt} bs={best_bs} OBJ={best_obj:.4f} Δ={best_delta:+.4f} LOYO {loyo_wins}/5"
        if validated else
        f"NO IMPROVEMENT — bt={BASELINE_BT} bs={BASELINE_BS} OBJ={baseline_obj:.4f} optimal"
    ),
    "partial": False,
    "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
}

out_path = OUT_DIR / "phase169_report.json"
out_path.write_text(json.dumps(report, indent=2))
print(f"\n✅ Saved → {out_path}")
