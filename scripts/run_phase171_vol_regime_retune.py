"""
Phase 171 — Vol Regime Overlay Re-Tune
========================================
Vol regime overlay last tuned P156 (scale 0.50→0.40, boost 0.20→0.15).
Many overlays stacked since: TS method change (P165), TS threshold re-sweep
(P168-169), and dispersion fine-tune (P170). With v2.12.0 as new baseline
(OBJ=2.4148), re-sweep vol regime params to find new optimal.

Parameters:
  - VOL_THRESHOLD: ann_vol level that triggers the overlay (currently 0.50)
  - VOL_SCALE: leverage reduction factor when triggered (currently 0.40)
  - VOL_F168_BOOST: F168 weight boost when triggered (currently 0.15)
  - VOL_WINDOW: rolling window for BTC vol computation (currently 168)

Part A: VOL_THRESHOLD sweep [0.35, 0.40, 0.45, 0.50, 0.55, 0.60] (others fixed)
Part B: VOL_SCALE sweep     [0.25, 0.30, 0.35, 0.40, 0.45, 0.50] (best thr, others fixed)
Part C: VOL_F168_BOOST sweep [0.05, 0.10, 0.15, 0.20, 0.25]     (best thr+scale)

Baseline (prod v2.12.0): VOL_THR=0.50, VOL_SCALE=0.40, VOL_BOOST=0.15, OBJ=2.4148
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

# v2.12.0 baseline vol regime params
BASELINE_VOL_THR   = 0.50
BASELINE_VOL_SCALE = 0.40
BASELINE_VOL_BOOST = 0.15
VOL_WINDOW         = 168

# Fixed breadth params
BRD_LOOKBACK   = 192
PCT_WINDOW     = 336
P_LOW, P_HIGH  = 0.35, 0.65

# Fixed dispersion params (v2.12.0)
FUND_DISP_THR   = 0.60
FUND_DISP_SCALE = 1.05
FUND_DISP_PCT   = 240

# Fixed TS params (v2.11.0)
TS_SHORT = 12
TS_LONG  = 96
RT = 0.60   # reduce_threshold
RS = 0.40   # reduce_scale
BT = 0.25   # boost_threshold
BS = 1.50   # boost_scale

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

VOL_THRESHOLDS  = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
VOL_SCALES      = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
VOL_BOOSTS      = [0.05, 0.10, 0.15, 0.20, 0.25]

OUT_DIR = ROOT / "artifacts" / "phase171"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

_partial: dict = {}

def _on_timeout(signum, frame):
    _partial["partial"] = True
    _partial["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (OUT_DIR / "phase171_report.json").write_text(json.dumps(_partial, indent=2))
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

    # BTC price vol (raw — threshold compared per-call in compute_ensemble)
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

    # Funding rates matrix
    fund_rates = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        for i in range(n):
            ts = dataset.timeline[i]
            try:
                fund_rates[i, j] = dataset.last_funding_rate_before(sym, ts)
            except Exception:
                fund_rates[i, j] = 0.0

    # Funding dispersion (v2.12.0 params: pct_win=240)
    fund_std_raw = np.std(fund_rates, axis=1)
    fund_std_pct = np.full(n, 0.5)
    for i in range(FUND_DISP_PCT, n):
        fund_std_pct[i] = float(np.mean(fund_std_raw[i - FUND_DISP_PCT:i] <= fund_std_raw[i]))
    fund_std_pct[:FUND_DISP_PCT] = 0.5

    # TS overlay (rolling-window, cumsum)
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


def compute_ensemble(
    sig_rets: dict,
    signals: dict,
    vol_thr:   float = BASELINE_VOL_THR,
    vol_scale: float = BASELINE_VOL_SCALE,
    vol_boost: float = BASELINE_VOL_BOOST,
) -> np.ndarray:
    sk_all = ["v1", "i460bw168", "i415bw216", "f168"]
    min_len = min(len(sig_rets[sk]) for sk in sk_all)
    bv  = signals["btc_vol"][:min_len]
    reg = signals["breadth_regime"][:min_len]
    fsp = signals["fund_std_pct"][:min_len]
    tsp = signals["ts_spread_pct"][:min_len]

    ens = np.zeros(min_len)
    for i in range(min_len):
        w = WEIGHTS[["LOW", "MID", "HIGH"][int(reg[i])]]
        if not np.isnan(bv[i]) and bv[i] > vol_thr:
            # Vol regime active: boost F168, reduce others, scale leverage
            boost_per = vol_boost / max(1, len(sk_all) - 1)
            ret_i = 0.0
            for sk in sk_all:
                adj_w = (min(0.60, w[sk] + vol_boost) if sk == "f168"
                         else max(0.05, w[sk] - boost_per))
                ret_i += adj_w * sig_rets[sk][i]
            ret_i *= vol_scale
        else:
            ret_i = sum(w[sk] * sig_rets[sk][i] for sk in sk_all)
        # Funding dispersion boost (v2.12.0 locked)
        if fsp[i] > FUND_DISP_THR:
            ret_i *= FUND_DISP_SCALE
        # TS overlay (v2.11.0 locked)
        if tsp[i] > RT:
            ret_i *= RS
        elif tsp[i] < BT:
            ret_i *= BS
        ens[i] = ret_i
    return ens


# ─── MAIN ────────────────────────────────────────────────────────────────────

print("=" * 68)
print("PHASE 171 — Vol Regime Overlay Re-Tune")
print("=" * 68)
print(f"  Baseline (v2.12.0): vol_thr={BASELINE_VOL_THR}, vol_scale={BASELINE_VOL_SCALE}, vol_boost={BASELINE_VOL_BOOST}")
print(f"  Dispersion locked: scale={FUND_DISP_SCALE} thr={FUND_DISP_THR} pct={FUND_DISP_PCT}")
print(f"  TS locked: rt={RT} rs={RS} bt={BT} bs={BS}")
print(f"  Baseline OBJ=2.4148\n")

print("[1/5] Loading data + signals + strategy returns...")
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

_partial.update({"phase": 171, "description": "Vol Regime Overlay Re-Tune", "partial": True})

def make_sig_rets(year):
    return {sk: strat_rets[sk][year] for sk in ["v1", "i460bw168", "i415bw216", "f168"]}

baseline_yearly = {
    year: sharpe(compute_ensemble(make_sig_rets(year), signals_data[year]))
    for year in YEARS
}
baseline_obj = obj_func(baseline_yearly)
print(f"\n  Baseline confirmed: OBJ={baseline_obj:.4f} | {baseline_yearly}")

# ─── PART A: VOL_THRESHOLD sweep ─────────────────────────────────────────────
print("\n[2/5] PART A — vol_thr sweep (scale=0.40, boost=0.15 fixed)...")
parta = {}
for vt in VOL_THRESHOLDS:
    yearly = {year: sharpe(compute_ensemble(make_sig_rets(year), signals_data[year],
                                            vol_thr=vt))
              for year in YEARS}
    o = obj_func(yearly)
    d = o - baseline_obj
    sym = "✅" if d > 0.005 else ("⚠️ " if d > -0.005 else "❌")
    tag = " ← baseline" if vt == BASELINE_VOL_THR else ""
    print(f"  {sym} vol_thr={vt:.2f}: OBJ={o:.4f} (Δ={d:+.4f}){tag}")
    parta[vt] = {"obj": o, "delta": d, "yearly": yearly}

best_vt = max(parta, key=lambda k: parta[k]["obj"])
print(f"  Best vol_thr={best_vt}: OBJ={parta[best_vt]['obj']:.4f} (Δ={parta[best_vt]['delta']:+.4f})")

# ─── PART B: VOL_SCALE sweep ─────────────────────────────────────────────────
print(f"\n[3/5] PART B — vol_scale sweep (thr={best_vt}, boost=0.15 fixed)...")
partb = {}
for vs in VOL_SCALES:
    yearly = {year: sharpe(compute_ensemble(make_sig_rets(year), signals_data[year],
                                            vol_thr=best_vt, vol_scale=vs))
              for year in YEARS}
    o = obj_func(yearly)
    d = o - baseline_obj
    sym = "✅" if d > 0.005 else ("⚠️ " if d > -0.005 else "❌")
    tag = " ← baseline" if (best_vt == BASELINE_VOL_THR and vs == BASELINE_VOL_SCALE) else ""
    print(f"  {sym} vol_scale={vs:.2f}: OBJ={o:.4f} (Δ={d:+.4f}){tag}")
    partb[vs] = {"obj": o, "delta": d, "yearly": yearly}

best_vs = max(partb, key=lambda k: partb[k]["obj"])
print(f"  Best vol_scale={best_vs}: OBJ={partb[best_vs]['obj']:.4f} (Δ={partb[best_vs]['delta']:+.4f})")

# ─── PART C: VOL_BOOST sweep ─────────────────────────────────────────────────
print(f"\n[4/5] PART C — vol_boost sweep (thr={best_vt}, scale={best_vs} fixed)...")
partc = {}
for vb in VOL_BOOSTS:
    yearly = {year: sharpe(compute_ensemble(make_sig_rets(year), signals_data[year],
                                            vol_thr=best_vt, vol_scale=best_vs,
                                            vol_boost=vb))
              for year in YEARS}
    o = obj_func(yearly)
    d = o - baseline_obj
    sym = "✅" if d > 0.005 else ("⚠️ " if d > -0.005 else "❌")
    tag = " ← baseline" if (best_vt == BASELINE_VOL_THR and
                             best_vs == BASELINE_VOL_SCALE and
                             vb == BASELINE_VOL_BOOST) else ""
    print(f"  {sym} vol_boost={vb:.2f}: OBJ={o:.4f} (Δ={d:+.4f}){tag}")
    partc[vb] = {"obj": o, "delta": d, "yearly": yearly}

best_vb    = max(partc, key=lambda k: partc[k]["obj"])
best_obj   = partc[best_vb]["obj"]
best_delta = partc[best_vb]["delta"]
best_yearly = partc[best_vb]["yearly"]
print(f"  Best vol_boost={best_vb}: OBJ={best_obj:.4f} (Δ={best_delta:+.4f})")
print(f"\n  Combined: vol_thr={best_vt} vol_scale={best_vs} vol_boost={best_vb}")

# ─── LOYO VALIDATION ─────────────────────────────────────────────────────────
print("\n[5/5] LOYO validation...")
loyo_wins, loyo_deltas = 0, []
for held_out in YEARS:
    sh_best = best_yearly[held_out]
    sh_base = baseline_yearly[held_out]
    d = sh_best - sh_base
    loyo_deltas.append(d)
    loyo_wins += int(d > 0)
    print(f"  LOYO {held_out}: best={sh_best:.4f} base={sh_base:.4f} Δ={d:+.4f} {'✅' if d>0 else '❌'}")

loyo_avg  = float(np.mean(loyo_deltas))
changed   = not (best_vt == BASELINE_VOL_THR and
                 best_vs == BASELINE_VOL_SCALE and
                 best_vb == BASELINE_VOL_BOOST)
validated = changed and best_delta > 0.005 and loyo_wins >= 3

print("\n" + "=" * 68)
if validated:
    print(f"✅ VALIDATED — thr={best_vt} scale={best_vs} boost={best_vb}: OBJ={best_obj:.4f} (Δ={best_delta:+.4f}) LOYO {loyo_wins}/5")
else:
    print(f"❌ NO IMPROVEMENT — baseline thr={BASELINE_VOL_THR} scale={BASELINE_VOL_SCALE} boost={BASELINE_VOL_BOOST} OBJ={baseline_obj:.4f} optimal")
    print(f"   Best: thr={best_vt} scale={best_vs} boost={best_vb} Δ={best_delta:+.4f} | LOYO {loyo_wins}/5")
print("=" * 68)

report = {
    "phase": 171,
    "description": "Vol Regime Overlay Re-Tune",
    "elapsed_seconds": round(time.time() - _start, 1),
    "baseline_vol_thr":   BASELINE_VOL_THR,
    "baseline_vol_scale": BASELINE_VOL_SCALE,
    "baseline_vol_boost": BASELINE_VOL_BOOST,
    "baseline_obj":       baseline_obj,
    "baseline_yearly":    baseline_yearly,
    "parta_thr_sweep":   {str(vt): {"obj": r["obj"], "delta": r["delta"]} for vt, r in parta.items()},
    "partb_scale_sweep": {str(vs): {"obj": r["obj"], "delta": r["delta"]} for vs, r in partb.items()},
    "partc_boost_sweep": {str(vb): {"obj": r["obj"], "delta": r["delta"]} for vb, r in partc.items()},
    "best_vol_thr":   best_vt,
    "best_vol_scale": best_vs,
    "best_vol_boost": best_vb,
    "best_obj":       best_obj,
    "best_delta":     best_delta,
    "best_yearly":    best_yearly,
    "loyo_wins":      loyo_wins,
    "loyo_avg_delta": loyo_avg,
    "validated":      validated,
    "verdict": (
        f"VALIDATED — thr={best_vt} scale={best_vs} boost={best_vb} OBJ={best_obj:.4f} Δ={best_delta:+.4f} LOYO {loyo_wins}/5"
        if validated else
        f"NO IMPROVEMENT — baseline thr={BASELINE_VOL_THR} scale={BASELINE_VOL_SCALE} boost={BASELINE_VOL_BOOST} OBJ={baseline_obj:.4f} optimal"
    ),
    "partial": False,
    "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
}

out_path = OUT_DIR / "phase171_report.json"
out_path.write_text(json.dumps(report, indent=2))
print(f"\n✅ Saved → {out_path}")
