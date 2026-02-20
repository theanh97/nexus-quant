"""
Phase 154 — Funding Signal Lookback Sweep
==========================================
Hypothesis: F144 (funding_lookback_bars=144) was fixed in early phases.
Shorter lookbacks (F72, F96) may better capture rapid funding regime changes
in 2025 (post-spot ETF era). Longer lookbacks (F168, F216) may smooth noise.

Test: Replace F144 with F72/F96/F120/F144/F168/F216 in full v2.5.0 ensemble.
Full LOYO for the best candidate.

Baseline (prod v2.5.0): OBJ=2.0851
OBJ = mean(yearly_sharpes) - 0.5 * std(yearly_sharpes)
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
SIG_DEFS = PROD_CFG["ensemble"]["signals"]
SIG_KEYS = list(SIG_DEFS.keys())

# v2.5.0 overlay constants
VOL_WINDOW     = 168
VOL_THRESHOLD  = 0.5
VOL_SCALE      = 0.5
VOL_F144_BOOST = 0.2
BREADTH_LOOKBACK = 168
PCT_WINDOW     = 336
P_LOW, P_HIGH  = 0.33, 0.67
FUND_DISP_THR  = 0.75
FUND_DISP_SCALE= 1.15
TS_REDUCE_THR  = 0.70
TS_REDUCE_SCALE= 0.60
TS_BOOST_THR   = 0.30
TS_BOOST_SCALE = 1.15
TS_SHORT, TS_LONG = 24, 144

WEIGHTS = {
    "LOW":  {"v1": 0.2747, "i460bw168": 0.1967, "i415bw216": 0.3247, "f144": 0.2039},
    "MID":  {"v1": 0.16,   "i460bw168": 0.22,   "i415bw216": 0.37,   "f144": 0.25},
    "HIGH": {"v1": 0.05,   "i460bw168": 0.25,   "i415bw216": 0.45,   "f144": 0.25},
}

YEAR_RANGES = {
    "2021": ("2021-02-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
}
YEARS = sorted(YEAR_RANGES.keys())

OUT_DIR = ROOT / "artifacts" / "phase154"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

# Funding lookbacks to test
F_LOOKBACKS = [72, 96, 120, 144, 168, 216]

_partial: dict = {}

def _on_timeout(signum, frame):
    print("\n⏰ TIMEOUT — saving partial...")
    _partial["partial"] = True
    _partial["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (OUT_DIR / "phase154_report.json").write_text(json.dumps(_partial, indent=2))
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


def compute_all_signals(dataset) -> dict:
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

    # Breadth
    breadth = np.full(n, 0.5)
    for i in range(BREADTH_LOOKBACK, n):
        pos = sum(
            1 for sym in SYMBOLS
            if (c0 := dataset.close(sym, i - BREADTH_LOOKBACK)) > 0
            and dataset.close(sym, i) > c0
        )
        breadth[i] = pos / len(SYMBOLS)
    breadth[:BREADTH_LOOKBACK] = 0.5
    brd_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        hist = breadth[i - PCT_WINDOW:i]
        brd_pct[i] = float(np.mean(hist <= breadth[i]))
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
        hist = fund_std_raw[i - PCT_WINDOW:i]
        fund_std_pct[i] = float(np.mean(hist <= fund_std_raw[i]))
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
        hist = ts_spread_raw[i - PCT_WINDOW:i]
        ts_spread_pct[i] = float(np.mean(hist <= ts_spread_raw[i]))
    ts_spread_pct[:PCT_WINDOW] = 0.5

    return {
        "btc_vol": btc_vol,
        "breadth_regime": breadth_regime,
        "fund_std_pct": fund_std_pct,
        "ts_spread_pct": ts_spread_pct,
    }


def compute_v25_ensemble(sig_rets: dict, signals: dict) -> np.ndarray:
    """v2.5.0 ensemble with all overlays. sig_rets['f'] is the funding signal."""
    sig_keys_local = ["v1", "i460bw168", "i415bw216", "f"]
    min_len = min(len(sig_rets[sk]) for sk in sig_keys_local)
    bv  = signals["btc_vol"][:min_len]
    reg = signals["breadth_regime"][:min_len]
    fsp = signals["fund_std_pct"][:min_len]
    tsp = signals["ts_spread_pct"][:min_len]

    ens = np.zeros(min_len)
    for i in range(min_len):
        w_raw = WEIGHTS[["LOW", "MID", "HIGH"][int(reg[i])]]
        # Map to local key 'f' instead of 'f144'
        w = {k: w_raw[k] for k in ["v1", "i460bw168", "i415bw216"]}
        w["f"] = w_raw["f144"]

        # Vol overlay
        if not np.isnan(bv[i]) and bv[i] > VOL_THRESHOLD:
            boost = VOL_F144_BOOST / max(1, len(sig_keys_local) - 1)
            ret_i = 0.0
            for sk in sig_keys_local:
                adj_w = (min(0.60, w[sk] + VOL_F144_BOOST) if sk == "f"
                         else max(0.05, w[sk] - boost))
                ret_i += adj_w * sig_rets[sk][i]
            ret_i *= VOL_SCALE
        else:
            ret_i = sum(w[sk] * sig_rets[sk][i] for sk in sig_keys_local)

        # Funding dispersion boost
        if fsp[i] > FUND_DISP_THR:
            ret_i *= FUND_DISP_SCALE

        # Funding term structure (v2.5.0 params)
        if tsp[i] > TS_REDUCE_THR:
            ret_i *= TS_REDUCE_SCALE
        elif tsp[i] < TS_BOOST_THR:
            ret_i *= TS_BOOST_SCALE

        ens[i] = ret_i

    return ens


# ─── main ────────────────────────────────────────────────────────────────────

print("=" * 68)
print("PHASE 154 — Funding Signal Lookback Sweep")
print("=" * 68)

print("\n[1/4] Loading data + computing base signals (2021-2025)...")

# Load base signals (V1, I460, I415) once per year — shared across F variants
base_rets: dict = {"v1": {}, "i460bw168": {}, "i415bw216": {}}
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
    signals_data[year] = compute_all_signals(dataset)
    print("O", end="", flush=True)

    # Base signals (V1, I460, I415) — loaded once
    for sk, strat_name, params in [
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
        strat  = make_strategy({"name": strat_name, "params": params})
        engine = BacktestEngine(bt_cfg)
        result = engine.run(dataset, strat)
        base_rets[sk][year] = np.array(result.returns)

    print(".", end="", flush=True)

    # Funding signals for all lookbacks
    for flb in F_LOOKBACKS:
        key = f"f{flb}"
        if key not in base_rets:
            base_rets[key] = {}
        bt_cfg = BacktestConfig(costs=COST_MODEL)
        strat  = make_strategy({"name": "funding_momentum_alpha", "params": {
            "k_per_side": 2,
            "funding_lookback_bars": flb,
            "direction": "contrarian",
            "target_gross_leverage": 0.25,
            "rebalance_interval_bars": 24,
        }})
        engine = BacktestEngine(bt_cfg)
        result = engine.run(dataset, strat)
        base_rets[key][year] = np.array(result.returns)

    print(" ✓")

print("\n[2/4] Computing yearly Sharpes for each F lookback...")
results = {}
for flb in F_LOOKBACKS:
    key = f"f{flb}"
    yearly = {}
    for year in YEARS:
        yr_sig = {
            "v1": base_rets["v1"][year],
            "i460bw168": base_rets["i460bw168"][year],
            "i415bw216": base_rets["i415bw216"][year],
            "f": base_rets[key][year],
        }
        combo = compute_v25_ensemble(yr_sig, signals_data[year])
        yearly[year] = sharpe(combo)
    o = obj_func(yearly)
    results[flb] = {"obj": o, "yearly": yearly}

# Baseline is F144
baseline_obj = results[144]["obj"]
baseline_yearly = results[144]["yearly"]
print(f"  Baseline F144: OBJ={baseline_obj:.4f} | {baseline_yearly}")
print()

best_lb   = max(results, key=lambda k: results[k]["obj"])
best_obj  = results[best_lb]["obj"]
best_delta = best_obj - baseline_obj

for flb in F_LOOKBACKS:
    o = results[flb]["obj"]
    d = o - baseline_obj
    is_best = " ← BEST" if flb == best_lb else ""
    sym = "✅" if d > 0.005 else ("⚠️ " if d > -0.005 else "❌")
    print(f"  {sym} F{flb}: OBJ={o:.4f} (Δ={d:+.4f}) | {results[flb]['yearly']}{is_best}")

print(f"\n[3/4] LOYO validation of best: F{best_lb}...")
best_key = f"f{best_lb}"
loyo_wins, loyo_deltas = 0, []
for held_out in YEARS:
    yr_sig_best = {
        "v1": base_rets["v1"][held_out],
        "i460bw168": base_rets["i460bw168"][held_out],
        "i415bw216": base_rets["i415bw216"][held_out],
        "f": base_rets[best_key][held_out],
    }
    yr_sig_base = {
        "v1": base_rets["v1"][held_out],
        "i460bw168": base_rets["i460bw168"][held_out],
        "i415bw216": base_rets["i415bw216"][held_out],
        "f": base_rets["f144"][held_out],
    }
    sh_best = sharpe(compute_v25_ensemble(yr_sig_best, signals_data[held_out]))
    sh_base = sharpe(compute_v25_ensemble(yr_sig_base, signals_data[held_out]))
    d = sh_best - sh_base
    loyo_deltas.append(d)
    loyo_wins += int(d > 0)
    print(f"  LOYO {held_out}: F{best_lb}={sh_best:.4f} F144={sh_base:.4f} Δ={d:+.4f} {'✅' if d>0 else '❌'}")

loyo_avg  = float(np.mean(loyo_deltas))
validated = loyo_wins >= 3 and best_delta > 0.005

print("\n" + "=" * 68)
if validated:
    print(f"✅ VALIDATED — F{best_lb}: OBJ={best_obj:.4f} (Δ={best_delta:+.4f}) | LOYO {loyo_wins}/5 avg={loyo_avg:+.4f}")
    print(f"   → Update production: funding_lookback_bars={best_lb}")
else:
    print(f"❌ NO IMPROVEMENT — best: F{best_lb} OBJ={best_obj:.4f} (Δ={best_delta:+.4f}) | LOYO {loyo_wins}/5")
    print(f"   Baseline F144 OBJ={baseline_obj:.4f} remains production.")
print("=" * 68)

report = {
    "phase": 154,
    "description": "Funding Signal Lookback Sweep",
    "hypothesis": "F144 was fixed early; shorter/longer lookback may better capture 2025 regime",
    "elapsed_seconds": round(time.time() - _start, 1),
    "baseline_lookback": 144,
    "baseline_obj": baseline_obj,
    "baseline_yearly": baseline_yearly,
    "results": {f"f{flb}": {"obj": results[flb]["obj"],
                              "delta": results[flb]["obj"] - baseline_obj,
                              "yearly": results[flb]["yearly"]}
                for flb in F_LOOKBACKS},
    "best_lookback": best_lb,
    "best_obj": best_obj,
    "best_delta": best_delta,
    "loyo_wins": loyo_wins,
    "loyo_avg_delta": loyo_avg,
    "validated": validated,
    "verdict": (f"VALIDATED — F{best_lb} OBJ={best_obj:.4f} LOYO {loyo_wins}/5"
                if validated else
                f"NO IMPROVEMENT — F144 OBJ={baseline_obj:.4f} remains production"),
    "partial": False,
    "timestamp": datetime.datetime.utcnow().isoformat(),
}

out_path = OUT_DIR / "phase154_report.json"
out_path.write_text(json.dumps(report, indent=2))
print(f"✅ Saved → {out_path}")
