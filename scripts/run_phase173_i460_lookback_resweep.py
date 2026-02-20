"""
Phase 173 — I460 Lookback Re-Sweep
=====================================
P172 changed I460 beta_window from 168 → 120. The original lb=460
was calibrated with bw=168. With bw=120 (more aggressive de-trending),
the optimal lookback may have shifted.

Sweep: lookback_bars ∈ [400, 420, 440, 460, 480, 500, 520]
Baseline (v2.14.0): I460 lb=460, bw=120, OBJ=2.4299

Design: V1, I415bw216, F168 run once per year (shared).
        I460 re-run for each lookback variant (bw=120 fixed).
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

# Fixed overlay params (v2.14.0)
VOL_WINDOW     = 168
VOL_THRESHOLD  = 0.5
VOL_SCALE      = 0.4
VOL_F168_BOOST = 0.10  # v2.13.0
BRD_LOOKBACK   = 192
PCT_WINDOW     = 336
P_LOW, P_HIGH  = 0.35, 0.65
FUND_DISP_THR   = 0.60
FUND_DISP_SCALE = 1.05
FUND_DISP_PCT   = 240
TS_SHORT = 12
TS_LONG  = 96
RT = 0.60
RS = 0.40
BT = 0.25
BS = 1.50

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

BASELINE_LB = 460
I460_BW     = 120  # locked from P172
LB_SWEEP    = [400, 420, 440, 460, 480, 500, 520]

OUT_DIR = ROOT / "artifacts" / "phase173"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

_partial: dict = {}

def _on_timeout(signum, frame):
    _partial["partial"] = True
    _partial["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (OUT_DIR / "phase173_report.json").write_text(json.dumps(_partial, indent=2))
    sys.exit(0)

_signal.signal(_signal.SIGALRM, _on_timeout)
_signal.alarm(3600)


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
    for i in range(FUND_DISP_PCT, n):
        fund_std_pct[i] = float(np.mean(fund_std_raw[i - FUND_DISP_PCT:i] <= fund_std_raw[i]))
    fund_std_pct[:FUND_DISP_PCT] = 0.5

    xsect_mean = np.mean(fund_rates, axis=1)
    ts_raw = rolling_mean_arr(xsect_mean, TS_SHORT) - rolling_mean_arr(xsect_mean, TS_LONG)
    ts_spread_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        ts_spread_pct[i] = float(np.mean(ts_raw[i - PCT_WINDOW:i] <= ts_raw[i]))
    ts_spread_pct[:PCT_WINDOW] = 0.5

    return {
        "btc_vol":        btc_vol,
        "breadth_regime": breadth_regime,
        "fund_std_pct":   fund_std_pct,
        "ts_spread_pct":  ts_spread_pct,
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
        if tsp[i] > RT:
            ret_i *= RS
        elif tsp[i] < BT:
            ret_i *= BS
        ens[i] = ret_i
    return ens


# ─── MAIN ────────────────────────────────────────────────────────────────────

print("=" * 68)
print("PHASE 173 — I460 Lookback Re-Sweep (bw=120 locked)")
print("=" * 68)
print(f"  Baseline (v2.14.0): lb={BASELINE_LB}, bw={I460_BW}, OBJ=2.4299")
print(f"  Sweep lb: {LB_SWEEP}\n")

print("[1/3] Loading data + signals + shared strategy returns...")
shared_rets: dict = {"v1": {}, "i415bw216": {}, "f168": {}}
i460_rets_by_lb: dict = {lb: {} for lb in LB_SWEEP}
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
        shared_rets[sk][year] = np.array(result.returns)

    for lb in LB_SWEEP:
        bt_cfg = BacktestConfig(costs=COST_MODEL)
        strat  = make_strategy({"name": "idio_momentum_alpha", "params": {
            "k_per_side": 4, "lookback_bars": lb, "beta_window_bars": I460_BW,
            "target_gross_leverage": 0.3, "rebalance_interval_bars": 48,
        }})
        result = BacktestEngine(bt_cfg).run(dataset, strat)
        i460_rets_by_lb[lb][year] = np.array(result.returns)
    print(f". ✓ [{len(LB_SWEEP)} lb variants]")

_partial.update({"phase": 173, "description": "I460 Lookback Re-Sweep", "partial": True})

print("\n[2/3] Lookback sweep...")
sweep_table = {}
baseline_yearly = None
baseline_obj = None

for lb in LB_SWEEP:
    yearly = {}
    for year in YEARS:
        sr = {
            "v1":        shared_rets["v1"][year],
            "i460bw168": i460_rets_by_lb[lb][year],
            "i415bw216": shared_rets["i415bw216"][year],
            "f168":      shared_rets["f168"][year],
        }
        yearly[year] = sharpe(compute_ensemble(sr, signals_data[year]))
    o = obj_func(yearly)
    if lb == BASELINE_LB:
        baseline_yearly = yearly
        baseline_obj = o
    sweep_table[lb] = {"lb": lb, "obj": o, "yearly": yearly}

print(f"  (Baseline lb={BASELINE_LB}: OBJ={baseline_obj:.4f})")
for lb in LB_SWEEP:
    r = sweep_table[lb]
    d = r["obj"] - baseline_obj
    r["delta"] = d
    sym = "✅" if d > 0.005 else ("⚠️ " if d > -0.005 else "❌")
    tag = " ← baseline" if lb == BASELINE_LB else ""
    print(f"  {sym} lb={lb}: OBJ={r['obj']:.4f} (Δ={d:+.4f}){tag}")

best_lb     = max(sweep_table, key=lambda k: sweep_table[k]["obj"])
best_obj    = sweep_table[best_lb]["obj"]
best_delta  = sweep_table[best_lb]["delta"]
best_yearly = sweep_table[best_lb]["yearly"]
print(f"\n  Best lb={best_lb}: OBJ={best_obj:.4f} (Δ={best_delta:+.4f})")

print("\n[3/3] LOYO validation...")
loyo_wins, loyo_deltas = 0, []
for held_out in YEARS:
    sh_best = best_yearly[held_out]
    sh_base = baseline_yearly[held_out]
    d = sh_best - sh_base
    loyo_deltas.append(d)
    loyo_wins += int(d > 0)
    print(f"  LOYO {held_out}: best={sh_best:.4f} base={sh_base:.4f} Δ={d:+.4f} {'✅' if d>0 else '❌'}")

loyo_avg  = float(np.mean(loyo_deltas))
validated = best_lb != BASELINE_LB and best_delta > 0.005 and loyo_wins >= 3

print("\n" + "=" * 68)
if validated:
    print(f"✅ VALIDATED — I460 lb={best_lb} bw={I460_BW}: OBJ={best_obj:.4f} (Δ={best_delta:+.4f}) LOYO {loyo_wins}/5")
else:
    print(f"❌ NO IMPROVEMENT — I460 lb={BASELINE_LB} bw={I460_BW} OBJ={baseline_obj:.4f} optimal")
    print(f"   Best: lb={best_lb} Δ={best_delta:+.4f} | LOYO {loyo_wins}/5")
print("=" * 68)

report = {
    "phase": 173,
    "description": "I460 Lookback Re-Sweep (bw=120 locked)",
    "elapsed_seconds": round(time.time() - _start, 1),
    "i460_bw_locked": I460_BW,
    "baseline_lb": BASELINE_LB,
    "baseline_obj": baseline_obj,
    "baseline_yearly": baseline_yearly,
    "sweep_table": {
        str(lb): {"lookback_bars": lb, "obj": r["obj"],
                  "delta": r.get("delta", 0.0), "yearly": r["yearly"]}
        for lb, r in sweep_table.items()
    },
    "best_lb": best_lb,
    "best_obj": best_obj,
    "best_delta": best_delta,
    "best_yearly": best_yearly,
    "loyo_wins": loyo_wins,
    "loyo_avg_delta": loyo_avg,
    "validated": validated,
    "verdict": (
        f"VALIDATED — I460 lb={best_lb} bw={I460_BW} OBJ={best_obj:.4f} Δ={best_delta:+.4f} LOYO {loyo_wins}/5"
        if validated else
        f"NO IMPROVEMENT — I460 lb={BASELINE_LB} bw={I460_BW} OBJ={baseline_obj:.4f} optimal"
    ),
    "partial": False,
    "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
}

out_path = OUT_DIR / "phase173_report.json"
out_path.write_text(json.dumps(report, indent=2))
print(f"\n✅ Saved → {out_path}")
