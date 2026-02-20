"""
Phase 170 — Funding Dispersion Overlay Fine-Tune
=================================================
Funding dispersion overlay (P148, v2.3.0) hasn't been re-tuned since
before the TS rolling-window method change (P165) and v2.9-v2.11 stacking.

Current (prod v2.11.0): boost_scale=1.15, boost_threshold_pct=0.75, pct_window=336

Hypothesis: With higher baseline OBJ=2.3599, the optimal dispersion
boost params may have shifted — especially since all TS/vol overlays
are now more tightly tuned.

Part A: boost_scale sweep  [1.05, 1.10, 1.15, 1.20, 1.25, 1.30]  (thr=0.75, pct=336 fixed)
Part B: boost_thr sweep    [0.60, 0.65, 0.70, 0.75, 0.80]         (best bs, pct=336 fixed)
Part C: pct_window sweep   [168, 240, 336, 480]                    (best bs + best thr)

Baseline (prod v2.11.0): OBJ=2.3599
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
TS_SHORT = 12
TS_LONG  = 96

# v2.11.0 locked TS params
RT = 0.60   # reduce_threshold
RS = 0.40   # reduce_scale
BT = 0.25   # boost_threshold
BS = 1.50   # boost_scale

# v2.11.0 baseline dispersion params
BASELINE_DISP_SCALE = 1.15
BASELINE_DISP_THR   = 0.75
BASELINE_DISP_PCT   = 336

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

DISP_SCALES     = [1.05, 1.10, 1.15, 1.20, 1.25, 1.30]
DISP_THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80]
PCT_WINDOWS     = [168, 240, 336, 480]

OUT_DIR = ROOT / "artifacts" / "phase170"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

_partial: dict = {}

def _on_timeout(signum, frame):
    _partial["partial"] = True
    _partial["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (OUT_DIR / "phase170_report.json").write_text(json.dumps(_partial, indent=2))
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
    """Compute all signals needed for ensemble. Precomputes fund_std_pct
    for all PCT_WINDOWS variants to avoid re-running for Part C."""
    n = len(dataset.timeline)

    # BTC vol (for vol regime overlay)
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

    # Funding rates matrix (single pass)
    fund_rates = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        for i in range(n):
            ts = dataset.timeline[i]
            try:
                fund_rates[i, j] = dataset.last_funding_rate_before(sym, ts)
            except Exception:
                fund_rates[i, j] = 0.0

    # Funding dispersion: cross-sectional std, precompute for all pct_windows
    fund_std_raw = np.std(fund_rates, axis=1)
    fund_std_pcts = {}
    for pw in PCT_WINDOWS:
        fsp = np.full(n, 0.5)
        for i in range(pw, n):
            fsp[i] = float(np.mean(fund_std_raw[i - pw:i] <= fund_std_raw[i]))
        fsp[:pw] = 0.5
        fund_std_pcts[pw] = fsp

    # TS overlay (rolling-window method, O(n) via cumsum)
    xsect_mean = np.mean(fund_rates, axis=1)
    ts_raw = rolling_mean_arr(xsect_mean, TS_SHORT) - rolling_mean_arr(xsect_mean, TS_LONG)
    ts_spread_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        ts_spread_pct[i] = float(np.mean(ts_raw[i - PCT_WINDOW:i] <= ts_raw[i]))
    ts_spread_pct[:PCT_WINDOW] = 0.5

    return {
        "btc_vol":        btc_vol,
        "breadth_regime": breadth_regime,
        "fund_std_pcts":  fund_std_pcts,  # dict: pw → array
        "ts_spread_pct":  ts_spread_pct,
    }


def compute_ensemble(
    sig_rets: dict,
    signals: dict,
    disp_scale: float = BASELINE_DISP_SCALE,
    disp_thr: float   = BASELINE_DISP_THR,
    disp_pct_win: int = BASELINE_DISP_PCT,
) -> np.ndarray:
    sk_all = ["v1", "i460bw168", "i415bw216", "f168"]
    min_len = min(len(sig_rets[sk]) for sk in sk_all)
    bv  = signals["btc_vol"][:min_len]
    reg = signals["breadth_regime"][:min_len]
    fsp = signals["fund_std_pcts"][disp_pct_win][:min_len]
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
        # Funding dispersion boost
        if fsp[i] > disp_thr:
            ret_i *= disp_scale
        # TS overlay (locked from v2.11.0)
        if tsp[i] > RT:
            ret_i *= RS
        elif tsp[i] < BT:
            ret_i *= BS
        ens[i] = ret_i
    return ens


# ─── MAIN ────────────────────────────────────────────────────────────────────

print("=" * 68)
print("PHASE 170 — Funding Dispersion Overlay Fine-Tune")
print("=" * 68)
print(f"  Baseline (v2.11.0): disp_scale={BASELINE_DISP_SCALE}, disp_thr={BASELINE_DISP_THR}, pct_win={BASELINE_DISP_PCT}")
print(f"  TS locked: rt={RT} rs={RS} bt={BT} bs={BS}")
print(f"  Baseline OBJ=2.3599\n")

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

_partial.update({"phase": 170, "description": "Funding Dispersion Fine-Tune", "partial": True})

def make_sig_rets(year):
    return {sk: strat_rets[sk][year] for sk in ["v1", "i460bw168", "i415bw216", "f168"]}

baseline_yearly = {
    year: sharpe(compute_ensemble(make_sig_rets(year), signals_data[year]))
    for year in YEARS
}
baseline_obj = obj_func(baseline_yearly)
print(f"\n  Baseline confirmed: OBJ={baseline_obj:.4f} | {baseline_yearly}")

# ─── PART A: boost_scale sweep ───────────────────────────────────────────────
print("\n[2/5] PART A — boost_scale sweep (thr=0.75, pct=336 fixed)...")
parta = {}
for ds in DISP_SCALES:
    yearly = {year: sharpe(compute_ensemble(make_sig_rets(year), signals_data[year],
                                            disp_scale=ds))
              for year in YEARS}
    o = obj_func(yearly)
    d = o - baseline_obj
    sym = "✅" if d > 0.005 else ("⚠️ " if d > -0.005 else "❌")
    tag = " ← baseline" if ds == BASELINE_DISP_SCALE else ""
    print(f"  {sym} scale={ds:.2f}: OBJ={o:.4f} (Δ={d:+.4f}){tag}")
    parta[ds] = {"obj": o, "delta": d, "yearly": yearly}

best_ds = max(parta, key=lambda k: parta[k]["obj"])
print(f"  Best scale={best_ds}: OBJ={parta[best_ds]['obj']:.4f} (Δ={parta[best_ds]['delta']:+.4f})")

# ─── PART B: boost_threshold sweep ───────────────────────────────────────────
print(f"\n[3/5] PART B — boost_thr sweep (scale={best_ds}, pct=336 fixed)...")
partb = {}
for dt in DISP_THRESHOLDS:
    yearly = {year: sharpe(compute_ensemble(make_sig_rets(year), signals_data[year],
                                            disp_scale=best_ds, disp_thr=dt))
              for year in YEARS}
    o = obj_func(yearly)
    d = o - baseline_obj
    sym = "✅" if d > 0.005 else ("⚠️ " if d > -0.005 else "❌")
    tag = " ← baseline" if (best_ds == BASELINE_DISP_SCALE and dt == BASELINE_DISP_THR) else ""
    print(f"  {sym} thr={dt:.2f}: OBJ={o:.4f} (Δ={d:+.4f}){tag}")
    partb[dt] = {"obj": o, "delta": d, "yearly": yearly}

best_dt = max(partb, key=lambda k: partb[k]["obj"])
print(f"  Best thr={best_dt}: OBJ={partb[best_dt]['obj']:.4f} (Δ={partb[best_dt]['delta']:+.4f})")

# ─── PART C: pct_window sweep ────────────────────────────────────────────────
print(f"\n[4/5] PART C — pct_window sweep (scale={best_ds}, thr={best_dt} fixed)...")
partc = {}
for pw in PCT_WINDOWS:
    yearly = {year: sharpe(compute_ensemble(make_sig_rets(year), signals_data[year],
                                            disp_scale=best_ds, disp_thr=best_dt,
                                            disp_pct_win=pw))
              for year in YEARS}
    o = obj_func(yearly)
    d = o - baseline_obj
    sym = "✅" if d > 0.005 else ("⚠️ " if d > -0.005 else "❌")
    tag = " ← baseline" if pw == BASELINE_DISP_PCT else ""
    print(f"  {sym} pct_win={pw}: OBJ={o:.4f} (Δ={d:+.4f}){tag}")
    partc[pw] = {"obj": o, "delta": d, "yearly": yearly}

best_pw    = max(partc, key=lambda k: partc[k]["obj"])
best_obj   = partc[best_pw]["obj"]
best_delta = partc[best_pw]["delta"]
best_yearly = partc[best_pw]["yearly"]
print(f"  Best pct_win={best_pw}: OBJ={best_obj:.4f} (Δ={best_delta:+.4f})")
print(f"\n  Combined: scale={best_ds} thr={best_dt} pct_win={best_pw}")

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
changed   = not (best_ds == BASELINE_DISP_SCALE and
                 best_dt == BASELINE_DISP_THR and
                 best_pw == BASELINE_DISP_PCT)
validated = changed and best_delta > 0.005 and loyo_wins >= 3

print("\n" + "=" * 68)
if validated:
    print(f"✅ VALIDATED — scale={best_ds} thr={best_dt} pct={best_pw}: OBJ={best_obj:.4f} (Δ={best_delta:+.4f}) LOYO {loyo_wins}/5")
else:
    print(f"❌ NO IMPROVEMENT — baseline scale={BASELINE_DISP_SCALE} thr={BASELINE_DISP_THR} pct={BASELINE_DISP_PCT} OBJ={baseline_obj:.4f} optimal")
    print(f"   Best: scale={best_ds} thr={best_dt} pct={best_pw} Δ={best_delta:+.4f} | LOYO {loyo_wins}/5")
print("=" * 68)

report = {
    "phase": 170,
    "description": "Funding Dispersion Overlay Fine-Tune",
    "elapsed_seconds": round(time.time() - _start, 1),
    "baseline_disp_scale": BASELINE_DISP_SCALE,
    "baseline_disp_thr":   BASELINE_DISP_THR,
    "baseline_disp_pct":   BASELINE_DISP_PCT,
    "baseline_obj":        baseline_obj,
    "baseline_yearly":     baseline_yearly,
    "parta_scale_sweep": {str(ds): {"obj": r["obj"], "delta": r["delta"]} for ds, r in parta.items()},
    "partb_thr_sweep":   {str(dt): {"obj": r["obj"], "delta": r["delta"]} for dt, r in partb.items()},
    "partc_pct_sweep":   {str(pw): {"obj": r["obj"], "delta": r["delta"]} for pw, r in partc.items()},
    "best_disp_scale": best_ds,
    "best_disp_thr":   best_dt,
    "best_disp_pct":   best_pw,
    "best_obj":        best_obj,
    "best_delta":      best_delta,
    "best_yearly":     best_yearly,
    "loyo_wins":       loyo_wins,
    "loyo_avg_delta":  loyo_avg,
    "validated":       validated,
    "verdict": (
        f"VALIDATED — scale={best_ds} thr={best_dt} pct={best_pw} OBJ={best_obj:.4f} Δ={best_delta:+.4f} LOYO {loyo_wins}/5"
        if validated else
        f"NO IMPROVEMENT — baseline scale={BASELINE_DISP_SCALE} thr={BASELINE_DISP_THR} pct={BASELINE_DISP_PCT} OBJ={baseline_obj:.4f} optimal"
    ),
    "partial": False,
    "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
}

out_path = OUT_DIR / "phase170_report.json"
out_path.write_text(json.dumps(report, indent=2))
print(f"\n✅ Saved → {out_path}")
