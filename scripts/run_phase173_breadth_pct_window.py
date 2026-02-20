"""
Phase 173 — Breadth Percentile Window Re-Tune (v2.14.0 stack)
===============================================================
The breadth regime classifier uses rolling_percentile_window=336h (14 days)
to rank current breadth against recent history. This determines when we're
in LOW/MID/HIGH momentum regime and thus which weight set to apply.

The PCT_WINDOW=336 was set early (P158 sweep) and hasn't been re-tuned
with the evolved v2.14.0 stack (upgraded FTS, dispersion, vol regime).

Hypothesis: With more aggressive FTS boost (bs=1.50) and new regime weights
(LOW/MID f168=0.30), a faster breadth percentile (shorter window) might
better capture regime transitions and avoid whipsawing between regimes.

Alternatively, a longer window might provide more stable regime labels.

Test: breadth_pct_window ∈ [96, 168, 240, 336, 480, 672]
  (current: 336)

Validate: LOYO ≥ 3/5 AND delta > 0 → update breadth_regime_switching.rolling_percentile_window
  → PRODUCTION v2.15.0
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

# v2.14.0 overlay constants
VOL_WINDOW    = 168
VOL_SCALE     = 0.40
F168_BOOST    = 0.10
VOL_THR       = 0.50
BRD_LOOKBACK  = 192
P_LOW, P_HIGH = 0.35, 0.65
TS_SHORT      = 12
TS_LONG       = 96
RT, RS        = 0.60, 0.40
BT, BS        = 0.25, 1.50
DISP_SCALE    = 1.05
DISP_THR      = 0.60
DISP_PCT_WIN  = 240

# v2.14.0 regime weights
WEIGHTS = {
    "LOW":  {"v1": 0.2415, "i460bw168": 0.173,  "i415bw216": 0.2855, "f168": 0.30},
    "MID":  {"v1": 0.1493, "i460bw168": 0.2053, "i415bw216": 0.3453, "f168": 0.30},
    "HIGH": {"v1": 0.0567, "i460bw168": 0.2833, "i415bw216": 0.51,   "f168": 0.15},
}

YEAR_RANGES = {
    "2021": ("2021-02-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
}
YEARS = sorted(YEAR_RANGES.keys())

BASELINE_PCT_WIN = 336
PCT_WINDOWS = [96, 168, 240, 336, 480, 672]

OUT_DIR = ROOT / "artifacts" / "phase173"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

_partial: dict = {}

def _on_timeout(signum, frame):
    _partial["partial"] = True
    _partial["timestamp"] = datetime.datetime.utcnow().isoformat()
    (OUT_DIR / "phase173_report.json").write_text(json.dumps(_partial, indent=2, default=str))
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


def rolling_mean_arr(x: np.ndarray, w: int) -> np.ndarray:
    n = len(x)
    cs = np.zeros(n + 1)
    for i in range(n):
        cs[i + 1] = cs[i] + x[i]
    result = np.zeros(n)
    for i in range(n):
        s = max(0, i - w + 1)
        result[i] = (cs[i + 1] - cs[s]) / (i - s + 1)
    return result


def compute_static_part(dataset) -> dict:
    """Compute signals that don't depend on PCT_WINDOW (funding rates, BTC vol)."""
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

    # Raw breadth (doesn't depend on pct_window)
    breadth = np.full(n, 0.5)
    for i in range(BRD_LOOKBACK, n):
        pos = sum(
            1 for sym in SYMBOLS
            if (c0 := dataset.close(sym, i - BRD_LOOKBACK)) > 0
            and dataset.close(sym, i) > c0
        )
        breadth[i] = pos / len(SYMBOLS)
    breadth[:BRD_LOOKBACK] = 0.5

    # Funding rates matrix
    fund_rates = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        for i in range(n):
            ts = dataset.timeline[i]
            try:
                fund_rates[i, j] = dataset.last_funding_rate_before(sym, ts)
            except Exception:
                fund_rates[i, j] = 0.0

    # Funding dispersion (DISP_PCT_WIN=240 fixed)
    fund_std_raw = np.std(fund_rates, axis=1)
    fund_std_pct = np.full(n, 0.5)
    for i in range(DISP_PCT_WIN, n):
        fund_std_pct[i] = float(np.mean(fund_std_raw[i - DISP_PCT_WIN:i] <= fund_std_raw[i]))
    fund_std_pct[:DISP_PCT_WIN] = 0.5

    # FTS spread (fixed windows 12/96)
    xsect_mean = np.mean(fund_rates, axis=1)
    ts_raw = rolling_mean_arr(xsect_mean, TS_SHORT) - rolling_mean_arr(xsect_mean, TS_LONG)
    ts_spread_pct = np.full(n, 0.5)
    for i in range(BASELINE_PCT_WIN, n):
        ts_spread_pct[i] = float(np.mean(ts_raw[i - BASELINE_PCT_WIN:i] <= ts_raw[i]))
    ts_spread_pct[:BASELINE_PCT_WIN] = 0.5

    return {
        "btc_vol":      btc_vol,
        "breadth_raw":  breadth,          # raw breadth (pre-percentile)
        "fund_std_pct": fund_std_pct,
        "ts_spread_pct": ts_spread_pct,
        "n":            n,
    }


def get_breadth_regime(static: dict, pct_win: int) -> np.ndarray:
    """Compute breadth regime labels for a given percentile window."""
    n = static["n"]
    breadth = static["breadth_raw"]
    brd_pct = np.full(n, 0.5)
    for i in range(pct_win, n):
        brd_pct[i] = float(np.mean(breadth[i - pct_win:i] <= breadth[i]))
    brd_pct[:pct_win] = 0.5
    return np.where(brd_pct >= P_HIGH, 2, np.where(brd_pct >= P_LOW, 1, 0)).astype(int)


def compute_ensemble(sig_rets: dict, static: dict, breadth_regime: np.ndarray) -> np.ndarray:
    sk_all = ["v1", "i460bw168", "i415bw216", "f168"]
    min_len = min(len(sig_rets[sk]) for sk in sk_all)
    bv  = static["btc_vol"][:min_len]
    reg = breadth_regime[:min_len]
    fsp = static["fund_std_pct"][:min_len]
    tsp = static["ts_spread_pct"][:min_len]

    ens = np.zeros(min_len)
    for i in range(min_len):
        w = WEIGHTS[["LOW", "MID", "HIGH"][int(reg[i])]]
        if not np.isnan(bv[i]) and bv[i] > VOL_THR:
            boost_per = F168_BOOST / max(1, len(sk_all) - 1)
            ret_i = 0.0
            for sk in sk_all:
                if sk == "f168":
                    adj_w = min(0.60, w[sk] + F168_BOOST)
                else:
                    adj_w = max(0.05, w[sk] - boost_per)
                ret_i += adj_w * sig_rets[sk][i]
            ret_i *= VOL_SCALE
        else:
            ret_i = sum(w[sk] * sig_rets[sk][i] for sk in sk_all)
        if fsp[i] > DISP_THR:
            ret_i *= DISP_SCALE
        if tsp[i] > RT:
            ret_i *= RS
        elif tsp[i] < BT:
            ret_i *= BS
        ens[i] = ret_i
    return ens


# ─── MAIN ────────────────────────────────────────────────────────────────────

print("=" * 68)
print("PHASE 173 — Breadth PCT Window Re-Tune (v2.14.0)")
print("=" * 68)
print(f"  Baseline: pct_window={BASELINE_PCT_WIN}h  v2.14.0 OBJ=2.4817\n")

sig_specs = [
    ("v1",        "nexus_alpha_v1",        {"k_per_side": 2, "w_carry": 0.35, "w_mom": 0.45,
                                             "w_mean_reversion": 0.2, "momentum_lookback_bars": 336,
                                             "mean_reversion_lookback_bars": 72, "vol_lookback_bars": 168,
                                             "target_gross_leverage": 0.35, "rebalance_interval_bars": 60}),
    ("i460bw168", "idio_momentum_alpha",   {"k_per_side": 4, "lookback_bars": 460,
                                             "beta_window_bars": 168, "target_gross_leverage": 0.3,
                                             "rebalance_interval_bars": 48}),
    ("i415bw216", "idio_momentum_alpha",   {"k_per_side": 4, "lookback_bars": 415,
                                             "beta_window_bars": 216, "target_gross_leverage": 0.3,
                                             "rebalance_interval_bars": 48}),
    ("f168",      "funding_momentum_alpha", {"k_per_side": 2, "funding_lookback_bars": 168,
                                             "direction": "contrarian", "target_gross_leverage": 0.25,
                                             "rebalance_interval_bars": 24}),
]

print("[1/3] Loading data, static signals, strategy returns ...")
strat_by_yr: dict = {sk: {} for sk in ["v1", "i460bw168", "i415bw216", "f168"]}
static_by_yr: dict = {}

for year, (start, end) in YEAR_RANGES.items():
    print(f"  {year}: ", end="", flush=True)
    cfg_data = {"provider": "binance_rest_v1", "symbols": SYMBOLS,
                "start": start, "end": end, "bar_interval": "1h",
                "cache_dir": ".cache/binance_rest"}
    dataset = make_provider(cfg_data, seed=42).load()
    static_by_yr[year] = compute_static_part(dataset)
    print("S", end="", flush=True)
    for sk, sname, params in sig_specs:
        result = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
            dataset, make_strategy({"name": sname, "params": params}))
        strat_by_yr[sk][year] = np.array(result.returns)
    print(". ✓")

def get_rets(year):
    return {sk: strat_by_yr[sk][year] for sk in ["v1", "i460bw168", "i415bw216", "f168"]}

# Baseline (pct_win=336)
brd_regs_baseline = {y: get_breadth_regime(static_by_yr[y], BASELINE_PCT_WIN) for y in YEARS}
baseline_yearly = {y: sharpe(compute_ensemble(get_rets(y), static_by_yr[y], brd_regs_baseline[y])) for y in YEARS}
BASELINE_OBJ = obj_func(baseline_yearly)
_partial.update({"phase": 173, "description": "Breadth PCT Window Sweep", "baseline_obj": BASELINE_OBJ, "partial": True})
print(f"\n  Baseline OBJ={BASELINE_OBJ:.4f}  yearly={baseline_yearly}")

# ─── SWEEP ───────────────────────────────────────────────────────────────────

print("\n[2/3] Breadth PCT window sweep ...")

sweep_results = {}
best_obj  = BASELINE_OBJ
best_pw   = BASELINE_PCT_WIN

for pw in PCT_WINDOWS:
    brd_regs = {y: get_breadth_regime(static_by_yr[y], pw) for y in YEARS}
    yr = {y: sharpe(compute_ensemble(get_rets(y), static_by_yr[y], brd_regs[y])) for y in YEARS}
    o  = obj_func(yr)
    d  = round(o - BASELINE_OBJ, 4)
    sweep_results[f"pw{pw}"] = {"pct_window": pw, "obj": o, "delta": d, "yearly": yr}
    if o > best_obj:
        best_obj = o
        best_pw  = pw
    print(f"  pw={pw:3d}h → OBJ={o:.4f}  Δ={d:+.4f}")

best_delta = round(best_obj - BASELINE_OBJ, 4)
print(f"\n  Winner: pct_window={best_pw}h  OBJ={best_obj:.4f}  Δ={best_delta:+.4f}")

# ─── LOYO VALIDATION ──────────────────────────────────────────────────────

print("\n[3/3] LOYO validation of winner ...")
brd_regs_best = {y: get_breadth_regime(static_by_yr[y], best_pw) for y in YEARS}
best_yearly   = {y: sharpe(compute_ensemble(get_rets(y), static_by_yr[y], brd_regs_best[y])) for y in YEARS}
best_obj_chk  = obj_func(best_yearly)

loyo_wins   = 0
loyo_deltas = []
loyo_table  = {}
for y in YEARS:
    d = best_yearly[y] - baseline_yearly[y]
    loyo_deltas.append(d)
    if d > 0:
        loyo_wins += 1
    loyo_table[y] = {"baseline_sh": baseline_yearly[y], "best_sh": best_yearly[y],
                     "delta": round(d, 4), "win": bool(d > 0)}

loyo_avg  = round(float(np.mean(loyo_deltas)), 4)
validated = (loyo_wins >= 3) and (best_delta > 0)

print(f"  LOYO {loyo_wins}/5  avg_Δ={loyo_avg:+.4f}")
for y, row in loyo_table.items():
    icon = "✅" if row["win"] else "❌"
    print(f"    {y}: {icon}  Δ={row['delta']:+.4f}")

if validated:
    verdict = f"VALIDATED — pct_window={best_pw}h OBJ={best_obj:.4f} Δ={best_delta:+.4f} LOYO {loyo_wins}/5"
else:
    verdict = f"NO IMPROVEMENT — pct_window={BASELINE_PCT_WIN}h OBJ={BASELINE_OBJ} optimal (LOYO {loyo_wins}/5)"

print(f"\n  {verdict}")

report = {
    "phase": 173,
    "description": "Breadth PCT Window Re-Tune",
    "elapsed_seconds": round(time.time() - _start, 1),
    "baseline_pct_window": BASELINE_PCT_WIN, "baseline_obj": BASELINE_OBJ,
    "baseline_yearly": baseline_yearly,
    "sweep_results": sweep_results,
    "best_pct_window": best_pw, "best_obj": best_obj, "best_delta": best_delta,
    "best_yearly": best_yearly,
    "loyo_table": loyo_table, "loyo_wins": loyo_wins, "loyo_avg_delta": loyo_avg,
    "validated": validated, "verdict": verdict,
    "partial": False, "timestamp": datetime.datetime.utcnow().isoformat(),
}

(OUT_DIR / "phase173_report.json").write_text(json.dumps(report, indent=2, default=str))
print(f"\nReport → {OUT_DIR}/phase173_report.json")

if validated:
    print("\n✅ Updating production config to v2.15.0 ...")
    cfg = json.loads((ROOT / "configs" / "production_p91b_champion.json").read_text())
    prev = cfg["_version"]
    cfg["_version"] = "2.15.0"
    cfg["_created"] = datetime.date.today().isoformat()
    cfg["_validated"] += (
        f"; Breadth pct_window re-tune (P173): pw={best_pw}h "
        f"LOYO {loyo_wins}/5 Δ={best_delta:+} OBJ={best_obj} — PRODUCTION v2.15.0"
    )
    brd = cfg["breadth_regime_switching"]
    brd["rolling_percentile_window"] = best_pw
    cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = best_obj
    (ROOT / "configs" / "production_p91b_champion.json").write_text(json.dumps(cfg, indent=2))
    print(f"  {prev} → v2.15.0  pct_window: {BASELINE_PCT_WIN} → {best_pw}")
else:
    print("\n❌ NO IMPROVEMENT — pct_window=336h remains optimal.")

print("\n" + "=" * 68)
print(f"PHASE 173 COMPLETE — {verdict}")
print(f"Elapsed: {round(time.time() - _start, 1)}s")
print("=" * 68)
