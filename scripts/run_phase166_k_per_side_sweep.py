"""
Phase 166 — k_per_side Sweep for I460 and I415
================================================
k=4 for both idio momentum signals (I460bw168, I415bw216) was validated in P84
era with a simple 3-signal stack. With the full v2.9.0 ensemble (breadth regime
switching, vol overlay, dispersion, TS overlay), portfolio concentration may
benefit from a different k.

k_per_side controls: long top-k / short bottom-k symbols by idio alpha.
  k=2: very concentrated (4 positions total)
  k=3: moderately concentrated (6 positions)
  k=4: current production (8 positions)
  k=5: more diversified (10 positions = full universe long/short!)
  k=6: over-diversified (limited for 10-symbol universe)

Two-phase sweep:
  Part A: I460 k sweep [2,3,4,5] with I415 k=4 fixed
  Part B: I415 k sweep [2,3,4,5] with best I460 k

Baseline: I460 k=4, I415 k=4 → OBJ=2.2423 (v2.9.0)
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

# v2.9.0 overlay constants
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
# v2.9.0 TS windows
TS_SHORT = 12
TS_LONG  = 96

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

BASELINE_K460 = 4
BASELINE_K415 = 4
K_CANDIDATES  = [2, 3, 4, 5]

OUT_DIR = ROOT / "artifacts" / "phase166"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

_partial: dict = {}

def _on_timeout(signum, frame):
    print("\n⏰ TIMEOUT — saving partial...")
    _partial["partial"] = True
    _partial["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (OUT_DIR / "phase166_report.json").write_text(json.dumps(_partial, indent=2))
    sys.exit(0)

_signal.signal(_signal.SIGALRM, _on_timeout)
_signal.alarm(3600)  # 60min


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

    # BTC vol
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

    # Funding rates (shared fetch)
    fund_rates = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        for i in range(n):
            ts = dataset.timeline[i]
            try:
                fund_rates[i, j] = dataset.last_funding_rate_before(sym, ts)
            except Exception:
                fund_rates[i, j] = 0.0

    # Dispersion
    fund_std_raw = np.std(fund_rates, axis=1)
    fund_std_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        fund_std_pct[i] = float(np.mean(fund_std_raw[i - PCT_WINDOW:i] <= fund_std_raw[i]))
    fund_std_pct[:PCT_WINDOW] = 0.5

    # TS spread (v2.9.0 rolling window method, short=12/long=96)
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
print("PHASE 166 — k_per_side Sweep (I460 + I415)")
print("=" * 68)

print("\n[1/4] Loading data + computing signals + shared returns (V1, F168)...")

# Shared: V1 and F168 (fixed params)
shared_rets: dict = {"v1": {}, "f168": {}}
# I460/I415 variants: k → year → returns
i460_rets: dict = {k: {} for k in K_CANDIDATES}
i415_rets: dict = {k: {} for k in K_CANDIDATES}
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

    # Shared: V1 and F168
    for sk, sname, params in [
        ("v1", "nexus_alpha_v1", {
            "k_per_side": 2, "w_carry": 0.35, "w_mom": 0.45, "w_mean_reversion": 0.2,
            "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 72,
            "vol_lookback_bars": 168, "target_gross_leverage": 0.35,
            "rebalance_interval_bars": 60,
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
    print(".", end="", flush=True)

    # I460 variants
    for k in K_CANDIDATES:
        bt_cfg = BacktestConfig(costs=COST_MODEL)
        strat  = make_strategy({"name": "idio_momentum_alpha", "params": {
            "k_per_side": k, "lookback_bars": 460, "beta_window_bars": 168,
            "target_gross_leverage": 0.3, "rebalance_interval_bars": 48,
        }})
        result = BacktestEngine(bt_cfg).run(dataset, strat)
        i460_rets[k][year] = np.array(result.returns)

    # I415 variants
    for k in K_CANDIDATES:
        bt_cfg = BacktestConfig(costs=COST_MODEL)
        strat  = make_strategy({"name": "idio_momentum_alpha", "params": {
            "k_per_side": k, "lookback_bars": 415, "beta_window_bars": 216,
            "target_gross_leverage": 0.3, "rebalance_interval_bars": 48,
        }})
        result = BacktestEngine(bt_cfg).run(dataset, strat)
        i415_rets[k][year] = np.array(result.returns)
    print(". ✓")

_partial.update({"phase": 166, "description": "k_per_side Sweep", "partial": True})

def make_sig_rets(year, k460, k415):
    return {
        "v1":        shared_rets["v1"][year],
        "i460bw168": i460_rets[k460][year],
        "i415bw216": i415_rets[k415][year],
        "f168":      shared_rets["f168"][year],
    }

# Baseline
baseline_yearly = {}
for year in YEARS:
    baseline_yearly[year] = sharpe(compute_ensemble(
        make_sig_rets(year, BASELINE_K460, BASELINE_K415), signals_data[year]))
baseline_obj = obj_func(baseline_yearly)
print(f"\n  Baseline (I460 k=4, I415 k=4): OBJ={baseline_obj:.4f} | {baseline_yearly}")

print(f"\n[2/4] PART A — I460 k sweep (I415 k=4 fixed)...")
parta_results = {}
for k460 in K_CANDIDATES:
    yearly = {year: sharpe(compute_ensemble(make_sig_rets(year, k460, BASELINE_K415),
                                            signals_data[year])) for year in YEARS}
    o = obj_func(yearly)
    d = o - baseline_obj
    sym = "✅" if d > 0.005 else ("⚠️ " if d > -0.005 else "❌")
    tag = " ← baseline" if k460 == BASELINE_K460 else ""
    print(f"  {sym} I460 k={k460}: OBJ={o:.4f} (Δ={d:+.4f}) | {yearly}{tag}")
    parta_results[k460] = {"obj": o, "delta": d, "yearly": yearly}

best_k460       = max(parta_results, key=lambda k: parta_results[k]["obj"])
best_k460_obj   = parta_results[best_k460]["obj"]
best_k460_delta = parta_results[best_k460]["delta"]
print(f"\n  Part A best: I460 k={best_k460} OBJ={best_k460_obj:.4f} (Δ={best_k460_delta:+.4f})")

print(f"\n[3/4] PART B — I415 k sweep (I460 k={best_k460} fixed)...")
partb_results = {}
for k415 in K_CANDIDATES:
    yearly = {year: sharpe(compute_ensemble(make_sig_rets(year, best_k460, k415),
                                            signals_data[year])) for year in YEARS}
    o = obj_func(yearly)
    d = o - baseline_obj
    sym = "✅" if d > 0.005 else ("⚠️ " if d > -0.005 else "❌")
    tag = " ← baseline" if (best_k460 == BASELINE_K460 and k415 == BASELINE_K415) else ""
    print(f"  {sym} I415 k={k415}: OBJ={o:.4f} (Δ={d:+.4f}) | {yearly}{tag}")
    partb_results[k415] = {"obj": o, "delta": d, "yearly": yearly}

best_k415       = max(partb_results, key=lambda k: partb_results[k]["obj"])
best_k415_obj   = partb_results[best_k415]["obj"]
best_k415_delta = partb_results[best_k415]["delta"]
print(f"\n  Part B best: I415 k={best_k415} OBJ={best_k415_obj:.4f} (Δ={best_k415_delta:+.4f})")
print(f"  Combined (I460 k={best_k460}, I415 k={best_k415}): OBJ={best_k415_obj:.4f}")

# LOYO on combined best
print(f"\n[4/4] LOYO validation (I460 k={best_k460}, I415 k={best_k415})...")
loyo_wins, loyo_deltas = 0, []
for held_out in YEARS:
    sh_best = partb_results[best_k415]["yearly"][held_out]
    sh_base = baseline_yearly[held_out]
    d = sh_best - sh_base
    loyo_deltas.append(d)
    loyo_wins += int(d > 0)
    print(f"  LOYO {held_out}: best={sh_best:.4f} base={sh_base:.4f} Δ={d:+.4f} {'✅' if d>0 else '❌'}")

loyo_avg  = float(np.mean(loyo_deltas))
best_delta = best_k415_delta
best_obj   = best_k415_obj
validated  = (best_delta > 0.005 and loyo_wins >= 3
              and not (best_k460 == BASELINE_K460 and best_k415 == BASELINE_K415))

print("\n" + "=" * 68)
if validated:
    print(f"✅ VALIDATED — I460 k={best_k460} / I415 k={best_k415}: OBJ={best_obj:.4f} (Δ={best_delta:+.4f}) LOYO {loyo_wins}/5")
else:
    print(f"❌ NO IMPROVEMENT — I460 k=4 / I415 k=4 OBJ={baseline_obj:.4f} optimal")
    print(f"   Best: I460 k={best_k460}/I415 k={best_k415} Δ={best_delta:+.4f} | LOYO {loyo_wins}/5")
print("=" * 68)

parta_tab = {f"k{k}": {"k": k, "obj": r["obj"], "delta": r["delta"],
                        "yearly": r["yearly"]} for k, r in parta_results.items()}
partb_tab = {f"k{k}": {"k": k, "obj": r["obj"], "delta": r["delta"],
                        "yearly": r["yearly"]} for k, r in partb_results.items()}

report = {
    "phase": 166,
    "description": "k_per_side Sweep (I460 + I415)",
    "elapsed_seconds": round(time.time() - _start, 1),
    "baseline_k460": BASELINE_K460,
    "baseline_k415": BASELINE_K415,
    "baseline_obj": baseline_obj,
    "baseline_yearly": baseline_yearly,
    "parta_sweep_i460": parta_tab,
    "parta_best_k460": best_k460,
    "parta_best_obj": best_k460_obj,
    "parta_best_delta": best_k460_delta,
    "partb_sweep_i415": partb_tab,
    "partb_best_k415": best_k415,
    "partb_best_obj": best_k415_obj,
    "partb_best_delta": best_k415_delta,
    "best_k460": best_k460,
    "best_k415": best_k415,
    "best_obj": best_obj,
    "best_delta": best_delta,
    "best_yearly": partb_results[best_k415]["yearly"],
    "loyo_wins": loyo_wins,
    "loyo_avg_delta": loyo_avg,
    "validated": validated,
    "verdict": (
        f"VALIDATED — I460 k={best_k460} I415 k={best_k415} OBJ={best_obj:.4f} Δ={best_delta:+.4f} LOYO {loyo_wins}/5"
        if validated else
        f"NO IMPROVEMENT — I460 k=4 I415 k=4 OBJ={baseline_obj:.4f} optimal"
    ),
    "partial": False,
    "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
}

out_path = OUT_DIR / "phase166_report.json"
out_path.write_text(json.dumps(report, indent=2))
print(f"\n✅ Saved → {out_path}")
