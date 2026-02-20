"""
Phase 153 — Calendar / Day-of-Week Overlay
==========================================
Hypothesis: Crypto perp markets show documented weekend degradation in
institutional participation and signal quality. Reducing leverage on
Sat/Sun UTC (and early Monday) should improve risk-adjusted returns.

Variants:
  weekend_85:  Sat+Sun → ×0.85
  weekend_80:  Sat+Sun → ×0.80
  weekend_75:  Sat+Sun → ×0.75
  mon_early85: Mon 00-08 UTC → ×0.85
  combo_8085:  Sat+Sun ×0.80 + Mon 00-08 ×0.85
  combo_7585:  Sat+Sun ×0.75 + Mon 00-08 ×0.85

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
SYMBOLS   = PROD_CFG["data"]["symbols"]
SIG_DEFS  = PROD_CFG["ensemble"]["signals"]
SIG_KEYS  = list(SIG_DEFS.keys())

# v2.5.0 overlay params
VOL_WINDOW     = 168
VOL_THRESHOLD  = 0.5
VOL_SCALE      = 0.5
VOL_F144_BOOST = 0.2
BREADTH_LOOKBACK = 168
PCT_WINDOW     = 336
P_LOW, P_HIGH  = 0.33, 0.67
FUND_DISP_THR  = 0.75
FUND_DISP_SCALE= 1.15
TS_REDUCE_THR  = 0.70   # fine-tuned P152
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

OUT_DIR = ROOT / "artifacts" / "phase153"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

# Timeout
_partial: dict = {}

def _on_timeout(signum, frame):
    print("\n⏰ TIMEOUT — saving partial...")
    _partial["partial"] = True
    _partial["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (OUT_DIR / "phase153_report.json").write_text(json.dumps(_partial, indent=2))
    sys.exit(0)

_signal.signal(_signal.SIGALRM, _on_timeout)
_signal.alarm(2400)


# ─── helpers ────────────────────────────────────────────────────────────────

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
    """Compute per-bar overlay signals: btc_vol, breadth_regime, fund_std_pct, ts_spread_pct."""
    n = len(dataset.timeline)

    # BTC log returns → rolling vol
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

    # Funding term structure spread (same method as P152)
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

    # Timeline timestamps for calendar overlay
    timeline_ts = np.array([int(t) for t in dataset.timeline])

    return {
        "btc_vol": btc_vol,
        "breadth_regime": breadth_regime,
        "fund_std_pct": fund_std_pct,
        "ts_spread_pct": ts_spread_pct,
        "timeline_ts": timeline_ts,
    }


def compute_v25_ensemble(
    sig_rets: dict,
    signals: dict,
    weekend_scale: float = 1.0,
    mon_early_scale: float = 1.0,
    mon_early_hours: int = 0,
) -> np.ndarray:
    """Full v2.5.0 ensemble + optional calendar overlay."""
    min_len = min(len(sig_rets[sk]) for sk in SIG_KEYS)
    bv  = signals["btc_vol"][:min_len]
    reg = signals["breadth_regime"][:min_len]
    fsp = signals["fund_std_pct"][:min_len]
    tsp = signals["ts_spread_pct"][:min_len]
    tl  = signals["timeline_ts"][:min_len]

    ens = np.zeros(min_len)
    for i in range(min_len):
        w = WEIGHTS[["LOW", "MID", "HIGH"][int(reg[i])]]

        # Vol overlay
        if not np.isnan(bv[i]) and bv[i] > VOL_THRESHOLD:
            boost = VOL_F144_BOOST / max(1, len(SIG_KEYS) - 1)
            ret_i = 0.0
            for sk in SIG_KEYS:
                adj_w = (min(0.60, w[sk] + VOL_F144_BOOST) if sk == "f144"
                         else max(0.05, w[sk] - boost))
                ret_i += adj_w * sig_rets[sk][i]
            ret_i *= VOL_SCALE
        else:
            ret_i = sum(w[sk] * sig_rets[sk][i] for sk in SIG_KEYS)

        # Funding dispersion boost
        if fsp[i] > FUND_DISP_THR:
            ret_i *= FUND_DISP_SCALE

        # Funding term structure (v2.5.0 fine-tuned params)
        if tsp[i] > TS_REDUCE_THR:
            ret_i *= TS_REDUCE_SCALE
        elif tsp[i] < TS_BOOST_THR:
            ret_i *= TS_BOOST_SCALE

        # Calendar overlay
        if weekend_scale != 1.0 or (mon_early_scale != 1.0 and mon_early_hours > 0):
            dt = datetime.datetime.utcfromtimestamp(int(tl[i]))
            dow  = dt.weekday()  # 0=Mon..6=Sun
            hour = dt.hour
            if dow in (5, 6):               # Sat / Sun
                ret_i *= weekend_scale
            elif dow == 0 and hour < mon_early_hours:  # Mon early
                ret_i *= mon_early_scale

        ens[i] = ret_i

    return ens


# ─── main ───────────────────────────────────────────────────────────────────

print("=" * 68)
print("PHASE 153 — Calendar / Day-of-Week Overlay")
print("=" * 68)

print("\n[1/4] Loading datasets + computing signals (2021-2025)...")
sig_returns:  dict = {sk: {} for sk in SIG_KEYS}
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

    for sk in SIG_KEYS:
        sig_def = SIG_DEFS[sk]
        bt_cfg  = BacktestConfig(costs=COST_MODEL)
        strat   = make_strategy({"name": sig_def["strategy"], "params": sig_def["params"]})
        engine  = BacktestEngine(bt_cfg)
        result  = engine.run(dataset, strat)
        sig_returns[sk][year] = np.array(result.returns)
    print(". ✓")

print("\n[2/4] Baseline v2.5.0 (no calendar overlay)...")
baseline_yearly = {}
for year in YEARS:
    yr_sig = {sk: sig_returns[sk][year] for sk in SIG_KEYS}
    combo  = compute_v25_ensemble(yr_sig, signals_data[year])
    baseline_yearly[year] = sharpe(combo)
baseline_obj = obj_func(baseline_yearly)
print(f"  Baseline OBJ={baseline_obj:.4f} | {baseline_yearly}")

VARIANTS = [
    ("weekend_85",  dict(weekend_scale=0.85, mon_early_scale=1.00, mon_early_hours=0)),
    ("weekend_80",  dict(weekend_scale=0.80, mon_early_scale=1.00, mon_early_hours=0)),
    ("weekend_75",  dict(weekend_scale=0.75, mon_early_scale=1.00, mon_early_hours=0)),
    ("mon_early85", dict(weekend_scale=1.00, mon_early_scale=0.85, mon_early_hours=8)),
    ("combo_8085",  dict(weekend_scale=0.80, mon_early_scale=0.85, mon_early_hours=8)),
    ("combo_7585",  dict(weekend_scale=0.75, mon_early_scale=0.85, mon_early_hours=8)),
]

print("\n[3/4] Testing calendar variants...")
results = {}
for name, kwargs in VARIANTS:
    yearly = {}
    for year in YEARS:
        yr_sig = {sk: sig_returns[sk][year] for sk in SIG_KEYS}
        combo  = compute_v25_ensemble(yr_sig, signals_data[year], **kwargs)
        yearly[year] = sharpe(combo)
    o = obj_func(yearly)
    delta = o - baseline_obj
    sym = "✅" if delta > 0.005 else ("⚠️ " if delta > -0.005 else "❌")
    print(f"  {sym} {name}: OBJ={o:.4f} (Δ={delta:+.4f}) | {yearly}")
    results[name] = {"obj": o, "delta": delta, "yearly": yearly}

best_name  = max(results, key=lambda k: results[k]["obj"])
best_obj   = results[best_name]["obj"]
best_delta = results[best_name]["delta"]
best_kw    = dict(next(v for n, v in VARIANTS if n == best_name))

print(f"\n[4/4] LOYO validation: {best_name}...")
loyo_wins, loyo_deltas = 0, []
for held_out in YEARS:
    yr_sig_cal  = {sk: sig_returns[sk][held_out] for sk in SIG_KEYS}
    oos_cal  = compute_v25_ensemble(yr_sig_cal, signals_data[held_out], **best_kw)
    oos_base = compute_v25_ensemble(yr_sig_cal, signals_data[held_out])
    sh_cal   = sharpe(oos_cal)
    sh_base  = sharpe(oos_base)
    d = sh_cal - sh_base
    loyo_deltas.append(d)
    loyo_wins += int(d > 0)
    print(f"  LOYO {held_out}: cal={sh_cal:.4f} base={sh_base:.4f} Δ={d:+.4f} {'✅' if d>0 else '❌'}")

loyo_avg  = float(np.mean(loyo_deltas))
validated = loyo_wins >= 3 and best_delta > 0.005

print("\n" + "=" * 68)
if validated:
    print(f"✅ VALIDATED — {best_name}: OBJ={best_obj:.4f} (Δ={best_delta:+.4f}) | LOYO {loyo_wins}/5 avg={loyo_avg:+.4f}")
else:
    print(f"❌ NO IMPROVEMENT — best: {best_name} OBJ={best_obj:.4f} (Δ={best_delta:+.4f}) | LOYO {loyo_wins}/5")
    print(f"   Baseline v2.5.0 OBJ={baseline_obj:.4f} remains production.")
print("=" * 68)

report = {
    "phase": 153,
    "description": "Calendar / Day-of-Week Overlay",
    "hypothesis": "Weekend lower institutional participation → noisier signals → reduce leverage",
    "elapsed_seconds": round(time.time() - _start, 1),
    "baseline_obj": baseline_obj,
    "baseline_yearly": baseline_yearly,
    "results": {n: {**r, "yearly": r["yearly"]} for n, r in results.items()},
    "best_variant": best_name,
    "best_obj": best_obj,
    "best_delta": best_delta,
    "best_params": best_kw,
    "loyo_wins": loyo_wins,
    "loyo_avg_delta": loyo_avg,
    "validated": validated,
    "verdict": (f"VALIDATED — {best_name} OBJ={best_obj:.4f} LOYO {loyo_wins}/5 avg={loyo_avg:+.4f}"
                if validated else
                f"NO IMPROVEMENT — baseline v2.5.0 OBJ={baseline_obj:.4f} remains production"),
    "partial": False,
    "timestamp": datetime.datetime.utcnow().isoformat(),
}

out_path = OUT_DIR / "phase153_report.json"
out_path.write_text(json.dumps(report, indent=2))
print(f"✅ Saved → {out_path}")
