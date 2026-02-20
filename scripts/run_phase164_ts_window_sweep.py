"""
Phase 164 — Funding Term Structure Window Sweep
================================================
The funding TS overlay was validated in P150b and fine-tuned in P152.
P152 optimized thresholds/scales: rt=0.70, rs=0.60, bt=0.30, bs=1.15.
The WINDOWS (short=24h, long=144h) have NEVER been swept.

Current logic: spread = mean_funding(last 24h) - mean_funding(last 144h)
  → spread_pct > 70th → scale×0.60 (crowded spike, reduce)
  → spread_pct < 30th → scale×1.15 (cooling, boost)

Hypothesis: Different window lengths capture different TS dynamics:
  - short=12h = exactly 1.5 funding cycles (very recent)
  - short=24h = 3 cycles (current production)
  - short=48h = 6 cycles (more stable recent average)
  - long=96h  = 12 cycles (4-day baseline)
  - long=144h = 18 cycles (6-day baseline, current production)
  - long=192h = 24 cycles (8-day baseline)
  - long=240h = 30 cycles (10-day baseline)

Test: 2-phase
  Part A: short ∈ [12, 24, 36, 48] with fixed long=144 — find best short
  Part B: long ∈ [96, 120, 144, 168, 192, 240] with best short — find best long

Baseline: short=24, long=144 → OBJ=2.2095
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

# v2.8.0 overlay constants (non-TS)
VOL_WINDOW     = 168
VOL_THRESHOLD  = 0.5
VOL_SCALE      = 0.4
VOL_F168_BOOST = 0.15
BRD_LOOKBACK   = 192
PCT_WINDOW     = 336
P_LOW, P_HIGH  = 0.35, 0.65
FUND_DISP_THR  = 0.75
FUND_DISP_SCALE= 1.15

# TS overlay params — thresholds/scales fixed from P152
TS_REDUCE_THR  = 0.70
TS_REDUCE_SCALE= 0.60
TS_BOOST_THR   = 0.30
TS_BOOST_SCALE = 1.15

# Production windows (baseline)
BASELINE_TS_SHORT = 24
BASELINE_TS_LONG  = 144

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

# Sweep windows
SHORT_WINDOWS = [12, 24, 36, 48]
LONG_WINDOWS  = [96, 120, 144, 168, 192, 240]

OUT_DIR = ROOT / "artifacts" / "phase164"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

_partial: dict = {}

def _on_timeout(signum, frame):
    print("\n⏰ TIMEOUT — saving partial...")
    _partial["partial"] = True
    _partial["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (OUT_DIR / "phase164_report.json").write_text(json.dumps(_partial, indent=2))
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


def compute_static_signals(dataset) -> dict:
    """Compute signals that don't depend on TS window (reused across all variants)."""
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
    fund_rates_cache = []
    for i in range(n):
        ts = dataset.timeline[i]
        rates = []
        for sym in SYMBOLS:
            try:
                rates.append(dataset.last_funding_rate_before(sym, ts))
            except Exception:
                rates.append(0.0)
        fund_rates_cache.append(rates)
        fund_std_raw[i] = float(np.std(rates)) if len(rates) > 1 else 0.0
    fund_std_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        fund_std_pct[i] = float(np.mean(fund_std_raw[i - PCT_WINDOW:i] <= fund_std_raw[i]))
    fund_std_pct[:PCT_WINDOW] = 0.5

    return {
        "btc_vol": btc_vol,
        "breadth_regime": breadth_regime,
        "fund_std_pct": fund_std_pct,
        "fund_rates_cache": fund_rates_cache,  # list[list[float]] for TS computation
        "timeline": dataset.timeline,
    }


def compute_ts_spread_pct(static: dict, ts_short: int, ts_long: int) -> np.ndarray:
    """Compute TS spread percentile for given window pair using cached funding rates."""
    n = len(static["timeline"])
    rates_cache = static["fund_rates_cache"]
    start_bar = max(ts_short, ts_long)

    ts_spread_raw = np.zeros(n)
    for i in range(start_bar, n):
        # short window: avg funding over [i-ts_short, i]
        short_rates = [np.mean([rates_cache[j][s] for s in range(len(SYMBOLS))])
                       for j in range(max(0, i - ts_short), i + 1)]
        # long window: avg funding over [i-ts_long, i]
        long_rates = [np.mean([rates_cache[j][s] for s in range(len(SYMBOLS))])
                      for j in range(max(0, i - ts_long), i + 1)]
        ts_spread_raw[i] = float(np.mean(short_rates)) - float(np.mean(long_rates))

    ts_spread_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        ts_spread_pct[i] = float(np.mean(ts_spread_raw[i - PCT_WINDOW:i] <= ts_spread_raw[i]))
    ts_spread_pct[:PCT_WINDOW] = 0.5
    return ts_spread_pct


def compute_ensemble(sig_rets: dict, static: dict, ts_spread_pct: np.ndarray) -> np.ndarray:
    sk_all = ["v1", "i460bw168", "i415bw216", "f168"]
    min_len = min(len(sig_rets[sk]) for sk in sk_all)
    bv  = static["btc_vol"][:min_len]
    reg = static["breadth_regime"][:min_len]
    fsp = static["fund_std_pct"][:min_len]
    tsp = ts_spread_pct[:min_len]

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
print("PHASE 164 — Funding TS Overlay Window Sweep")
print("=" * 68)

print("\n[1/5] Loading data + computing shared signals + strategy returns...")

shared_rets: dict = {"v1": {}, "i460bw168": {}, "i415bw216": {}, "f168": {}}
static_data: dict = {}

for year, (start, end) in YEAR_RANGES.items():
    print(f"  {year}: ", end="", flush=True)
    cfg_data = {
        "provider": "binance_rest_v1", "symbols": SYMBOLS,
        "start": start, "end": end, "bar_interval": "1h",
        "cache_dir": ".cache/binance_rest",
    }
    provider = make_provider(cfg_data, seed=42)
    dataset  = provider.load()
    static_data[year] = compute_static_signals(dataset)
    print("S", end="", flush=True)

    # Strategy returns (same for all window variants)
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
        shared_rets[sk][year] = np.array(result.returns)
    print(". ✓")

_partial.update({"phase": 164, "description": "TS Window Sweep", "partial": True})

def make_sig_rets(year):
    return {sk: shared_rets[sk][year] for sk in ["v1", "i460bw168", "i415bw216", "f168"]}

# Precompute baseline TS spread pct
baseline_ts_pct = {
    year: compute_ts_spread_pct(static_data[year], BASELINE_TS_SHORT, BASELINE_TS_LONG)
    for year in YEARS
}
baseline_yearly = {
    year: sharpe(compute_ensemble(make_sig_rets(year), static_data[year], baseline_ts_pct[year]))
    for year in YEARS
}
baseline_obj = obj_func(baseline_yearly)
print(f"\n  Baseline (short={BASELINE_TS_SHORT}, long={BASELINE_TS_LONG}): OBJ={baseline_obj:.4f} | {baseline_yearly}")

print(f"\n[2/5] PART A — short window sweep (fixed long={BASELINE_TS_LONG})...")
parta_results = {}
for sw in SHORT_WINDOWS:
    ts_pct_map = {
        year: compute_ts_spread_pct(static_data[year], sw, BASELINE_TS_LONG)
        for year in YEARS
    }
    yearly = {
        year: sharpe(compute_ensemble(make_sig_rets(year), static_data[year], ts_pct_map[year]))
        for year in YEARS
    }
    o = obj_func(yearly)
    d = o - baseline_obj
    sym = "✅" if d > 0.005 else ("⚠️ " if d > -0.005 else "❌")
    tag = " ← baseline" if sw == BASELINE_TS_SHORT else ""
    print(f"  {sym} short={sw:2d}h: OBJ={o:.4f} (Δ={d:+.4f}) | {yearly}{tag}")
    parta_results[sw] = {"obj": o, "delta": d, "yearly": yearly, "ts_pct": ts_pct_map}

best_short = max(parta_results, key=lambda k: parta_results[k]["obj"])
best_short_obj = parta_results[best_short]["obj"]
best_short_delta = parta_results[best_short]["delta"]
print(f"\n  Part A best: short={best_short}h OBJ={best_short_obj:.4f} (Δ={best_short_delta:+.4f})")

print(f"\n[3/5] PART B — long window sweep (using best short={best_short})...")
partb_results = {}
for lw in LONG_WINDOWS:
    ts_pct_map = {
        year: compute_ts_spread_pct(static_data[year], best_short, lw)
        for year in YEARS
    }
    yearly = {
        year: sharpe(compute_ensemble(make_sig_rets(year), static_data[year], ts_pct_map[year]))
        for year in YEARS
    }
    o = obj_func(yearly)
    d = o - baseline_obj
    sym = "✅" if d > 0.005 else ("⚠️ " if d > -0.005 else "❌")
    tag = " ← baseline" if (best_short == BASELINE_TS_SHORT and lw == BASELINE_TS_LONG) else ""
    print(f"  {sym} long={lw:3d}h: OBJ={o:.4f} (Δ={d:+.4f}) | {yearly}{tag}")
    partb_results[lw] = {"obj": o, "delta": d, "yearly": yearly, "ts_pct": ts_pct_map}

best_long = max(partb_results, key=lambda k: partb_results[k]["obj"])
best_long_obj = partb_results[best_long]["obj"]
best_long_delta = partb_results[best_long]["delta"]
print(f"\n  Part B best: long={best_long}h OBJ={best_long_obj:.4f} (Δ={best_long_delta:+.4f})")

# Best overall
best_obj   = best_long_obj
best_delta = best_long_delta
best_sw    = best_short
best_lw    = best_long
best_ts_pct = partb_results[best_long]["ts_pct"]
best_yearly = partb_results[best_long]["yearly"]

print(f"\n[4/5] LOYO validation (short={best_sw}h, long={best_lw}h)...")
loyo_wins, loyo_deltas = 0, []
for held_out in YEARS:
    sh_best = best_ts_pct[held_out]
    sh_best_val = sharpe(compute_ensemble(make_sig_rets(held_out), static_data[held_out],
                                          best_ts_pct[held_out]))
    sh_base = baseline_yearly[held_out]
    d = sh_best_val - sh_base
    loyo_deltas.append(d)
    loyo_wins += int(d > 0)
    print(f"  LOYO {held_out}: best={sh_best_val:.4f} base={sh_base:.4f} Δ={d:+.4f} {'✅' if d>0 else '❌'}")

loyo_avg  = float(np.mean(loyo_deltas))
validated = ((best_sw != BASELINE_TS_SHORT or best_lw != BASELINE_TS_LONG)
             and best_delta > 0.005
             and loyo_wins >= 3)

print("\n" + "=" * 68)
if validated:
    print(f"✅ VALIDATED — TS windows short={best_sw}h/long={best_lw}h: OBJ={best_obj:.4f} (Δ={best_delta:+.4f}) LOYO {loyo_wins}/5")
else:
    print(f"❌ NO IMPROVEMENT — baseline short={BASELINE_TS_SHORT}/long={BASELINE_TS_LONG} OBJ={baseline_obj:.4f} optimal")
    print(f"   Best: short={best_sw}h/long={best_lw}h Δ={best_delta:+.4f} | LOYO {loyo_wins}/5 (need ≥3 + Δ>0.005)")
print("=" * 68)

# Build sweep tables (strip ts_pct to keep report clean)
parta_table = {f"short_{sw}h": {"short": sw, "long": BASELINE_TS_LONG,
                                 "obj": r["obj"], "delta": r["delta"], "yearly": r["yearly"]}
               for sw, r in parta_results.items()}
partb_table = {f"short{best_short}h_long{lw}h": {"short": best_short, "long": lw,
                                                    "obj": r["obj"], "delta": r["delta"], "yearly": r["yearly"]}
               for lw, r in partb_results.items()}

report = {
    "phase": 164,
    "description": "Funding TS Overlay Window Sweep",
    "elapsed_seconds": round(time.time() - _start, 1),
    "baseline_short": BASELINE_TS_SHORT,
    "baseline_long": BASELINE_TS_LONG,
    "baseline_obj": baseline_obj,
    "baseline_yearly": baseline_yearly,
    "parta_sweep": parta_table,
    "parta_best": {"short": best_short, "long": BASELINE_TS_LONG,
                   "obj": best_short_obj, "delta": best_short_delta},
    "partb_sweep": partb_table,
    "partb_best": {"short": best_sw, "long": best_lw,
                   "obj": best_long_obj, "delta": best_long_delta},
    "best_short": best_sw,
    "best_long": best_lw,
    "best_obj": best_obj,
    "best_delta": best_delta,
    "best_yearly": best_yearly,
    "loyo_wins": loyo_wins,
    "loyo_avg_delta": loyo_avg,
    "validated": validated,
    "verdict": (
        f"VALIDATED — TS short={best_sw}h long={best_lw}h OBJ={best_obj:.4f} Δ={best_delta:+.4f} LOYO {loyo_wins}/5"
        if validated else
        f"NO IMPROVEMENT — TS short={BASELINE_TS_SHORT}/long={BASELINE_TS_LONG} OBJ={baseline_obj:.4f} optimal"
    ),
    "partial": False,
    "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
}

out_path = OUT_DIR / "phase164_report.json"
out_path.write_text(json.dumps(report, indent=2))
print(f"\n[5/5] ✅ Saved → {out_path}")
