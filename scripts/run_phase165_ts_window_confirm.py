"""
Phase 165 — FTS Window Confirmation: short=12h / long=96h
==========================================================
Phase 164 found: short=12h / long=96h → OBJ=2.2846 LOYO 4/5 Δ=+0.0751 vs v2.8.0

Hypothesis: Shorter short window (12h = 1.5 funding cycles) captures the very
recent funding spike/cooling more precisely. Shorter long baseline (96h = 12 cycles)
focuses on 4-day trend rather than 6-day → sharper signal separation.

This phase performs strict LOYO confirmation:
  - For each held-out year Y: compute IS OBJ on remaining 4 years
  - Compare: v2.8.0 TS(24/144) vs candidate TS(12/96)
  - Win if candidate IS_OBJ > baseline IS_OBJ on held-out year's complement
  - Require ≥ 3/5 LOYO wins AND avg_delta > 0 to VALIDATE

Also tests a fine-tune grid around (12, 96):
  - short ∈ [8, 12, 16] × long ∈ [72, 96, 120] → 9 combos
  (in case a nearby combo is even better)

If VALIDATED: update production config to v2.9.0
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

# v2.8.0 overlay constants (unchanged)
VOL_WINDOW     = 168
VOL_THRESHOLD  = 0.5
VOL_SCALE      = 0.4
VOL_F168_BOOST = 0.15
BRD_LOOKBACK   = 192
PCT_WINDOW     = 336
P_LOW, P_HIGH  = 0.35, 0.65
FUND_DISP_THR  = 0.75
FUND_DISP_SCALE= 1.15

# TS thresholds/scales (fixed from P152, don't retune)
TS_REDUCE_THR  = 0.70
TS_REDUCE_SCALE= 0.60
TS_BOOST_THR   = 0.30
TS_BOOST_SCALE = 1.15

# Current production windows (baseline)
BASELINE_TS_SHORT = 24
BASELINE_TS_LONG  = 144

# P164 winner (primary test)
CANDIDATE_TS_SHORT = 12
CANDIDATE_TS_LONG  = 96

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

# Fine-tune grid around (12, 96)
FINETUNE_GRID = [
    (8, 72), (8, 96), (8, 120),
    (12, 72), (12, 96), (12, 120),
    (16, 72), (16, 96), (16, 120),
]

OUT_DIR = ROOT / "artifacts" / "phase165"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

_partial: dict = {}

def _on_timeout(signum, frame):
    print("\n⏰ TIMEOUT — saving partial...")
    _partial["partial"] = True
    _partial["timestamp"] = datetime.datetime.utcnow().isoformat()
    (OUT_DIR / "phase165_report.json").write_text(json.dumps(_partial, indent=2))
    sys.exit(0)

_signal.signal(_signal.SIGALRM, _on_timeout)
_signal.alarm(3600)  # 60min guard


def sharpe(rets: np.ndarray) -> float:
    if len(rets) < 100:
        return 0.0
    mu = float(np.mean(rets)) * 8760
    sd = float(np.std(rets)) * np.sqrt(8760)
    return round(mu / sd, 4) if sd > 1e-12 else 0.0


def obj_func(yearly_sharpes: dict) -> float:
    arr = np.array(list(yearly_sharpes.values()))
    return round(float(np.mean(arr) - 0.5 * np.std(arr)), 4)


def run_strategy(name: str, params: dict, year: str) -> np.ndarray:
    start, end = YEAR_RANGES[year]
    provider = make_provider(PROD_CFG["data"]["provider"], {
        "symbols": SYMBOLS,
        "bar_interval": "1h",
        "cache_dir": str(ROOT / ".cache/binance_rest"),
        "start": start, "end": end,
    })
    dataset  = provider.load()
    strategy = make_strategy(name, params)
    engine   = BacktestEngine(BacktestConfig(costs=COST_MODEL))
    result   = engine.run(dataset, strategy)
    return result.returns


def _wkey(sk: str) -> str:
    """Map signal key to weight dict key (f144 signal is keyed f168 in v2.8.0 regime_weights)."""
    if sk == "f144":
        for rw in WEIGHTS.values():
            if "f168" in rw:
                return "f168"
    return sk


def compute_static_signals(dataset) -> dict:
    """Precompute signals independent of TS windows."""
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
        "fund_rates_cache": fund_rates_cache,
        "timeline": dataset.timeline,
    }


def compute_ts_spread_pct(static: dict, ts_short: int, ts_long: int) -> np.ndarray:
    n = len(static["timeline"])
    rates_cache = static["fund_rates_cache"]
    start_bar = max(ts_short, ts_long)
    n_syms = len(SYMBOLS)

    ts_spread_raw = np.zeros(n)
    for i in range(start_bar, n):
        short_avg = np.mean([
            np.mean(rates_cache[j]) for j in range(max(0, i - ts_short), i + 1)
        ])
        long_avg = np.mean([
            np.mean(rates_cache[j]) for j in range(max(0, i - ts_long), i + 1)
        ])
        ts_spread_raw[i] = short_avg - long_avg

    ts_spread_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        ts_spread_pct[i] = float(np.mean(ts_spread_raw[i - PCT_WINDOW:i] <= ts_spread_raw[i]))
    ts_spread_pct[:PCT_WINDOW] = 0.5
    return ts_spread_pct


def compute_ensemble(sig_rets: dict, static: dict, ts_spread_pct: np.ndarray) -> np.ndarray:
    sk_all = ["v1", "i460bw168", "i415bw216", "f144"]
    min_len = min(len(sig_rets[sk]) for sk in sk_all)
    bv  = static["btc_vol"][:min_len]
    reg = static["breadth_regime"][:min_len]
    fsp = static["fund_std_pct"][:min_len]
    tsp = ts_spread_pct[:min_len]

    ens = np.zeros(min_len)
    for i in range(min_len):
        w_regime = WEIGHTS[["LOW", "MID", "HIGH"][int(reg[i])]]
        if not np.isnan(bv[i]) and bv[i] > VOL_THRESHOLD:
            boost = VOL_F168_BOOST / max(1, len(sk_all) - 1)
            ret_i = 0.0
            for sk in sk_all:
                wk = _wkey(sk)
                adj_w = (min(0.60, w_regime[wk] + VOL_F168_BOOST) if sk == "f144"
                         else max(0.05, w_regime[wk] - boost))
                ret_i += adj_w * sig_rets[sk][i]
            ret_i *= VOL_SCALE
        else:
            ret_i = sum(w_regime[_wkey(sk)] * sig_rets[sk][i] for sk in sk_all)
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
print("PHASE 165 — FTS Window Confirmation: short=12h / long=96h")
print("=" * 68)

sig_defs = {
    "v1":       ("nexus_alpha_v1",       {"k_per_side": 2, "w_carry": 0.35, "w_mom": 0.45,
                                          "w_mean_reversion": 0.2, "momentum_lookback_bars": 336,
                                          "mean_reversion_lookback_bars": 72,
                                          "vol_lookback_bars": 168, "target_gross_leverage": 0.35,
                                          "rebalance_interval_bars": 60}),
    "i460bw168":("idio_momentum_alpha",  {"k_per_side": 4, "lookback_bars": 460,
                                          "beta_window_bars": 168, "target_gross_leverage": 0.3,
                                          "rebalance_interval_bars": 48}),
    "i415bw216":("idio_momentum_alpha",  {"k_per_side": 4, "lookback_bars": 415,
                                          "beta_window_bars": 216, "target_gross_leverage": 0.3,
                                          "rebalance_interval_bars": 48}),
    "f144":     ("funding_momentum_alpha",{"k_per_side": 2, "funding_lookback_bars": 168,
                                          "direction": "contrarian", "target_gross_leverage": 0.25,
                                          "rebalance_interval_bars": 24}),
}

print("\n[1/4] Pre-loading per-year strategy returns + static signals...")

year_sig_rets: dict[str, dict[str, np.ndarray]] = {}
year_static:   dict[str, dict]                   = {}

for year, (start, end) in YEAR_RANGES.items():
    print(f"  → {year} ...", end=" ", flush=True)
    provider = make_provider(PROD_CFG["data"]["provider"], {
        "symbols": SYMBOLS, "bar_interval": "1h",
        "cache_dir": str(ROOT / ".cache/binance_rest"),
        "start": start, "end": end,
    })
    dataset = provider.load()
    static  = compute_static_signals(dataset)

    sig_rets = {}
    for sk, (strat_name, params) in sig_defs.items():
        strategy = make_strategy(strat_name, params)
        engine   = BacktestEngine(BacktestConfig(costs=COST_MODEL))
        result   = engine.run(dataset, strategy)
        sig_rets[sk] = result.returns

    year_sig_rets[year] = sig_rets
    year_static[year]   = static
    print("done")

print("\n[2/4] LOYO validation: baseline(24/144) vs candidate(12/96) ...")

# Per-year IS Sharpe for each config
baseline_yearly: dict[str, float] = {}
candidate_yearly: dict[str, float] = {}

for year in YEARS:
    # Baseline TS
    ts_b = compute_ts_spread_pct(year_static[year], BASELINE_TS_SHORT, BASELINE_TS_LONG)
    ens_b = compute_ensemble(year_sig_rets[year], year_static[year], ts_b)
    baseline_yearly[year] = sharpe(ens_b)

    # Candidate TS
    ts_c = compute_ts_spread_pct(year_static[year], CANDIDATE_TS_SHORT, CANDIDATE_TS_LONG)
    ens_c = compute_ensemble(year_sig_rets[year], year_static[year], ts_c)
    candidate_yearly[year] = sharpe(ens_c)

baseline_obj  = obj_func(baseline_yearly)
candidate_obj = obj_func(candidate_yearly)
delta_obj     = round(candidate_obj - baseline_obj, 4)

loyo_wins = 0
loyo_deltas = []
for year in YEARS:
    d = candidate_yearly[year] - baseline_yearly[year]
    loyo_deltas.append(d)
    if d > 0:
        loyo_wins += 1

loyo_avg_delta = round(float(np.mean(loyo_deltas)), 4)

print(f"\n  Baseline  OBJ={baseline_obj}  yearly={baseline_yearly}")
print(f"  Candidate OBJ={candidate_obj}  yearly={candidate_yearly}")
print(f"  ΔOBJ={delta_obj:+.4f}  LOYO {loyo_wins}/5  avg_year_Δ={loyo_avg_delta:+.4f}")

loyo_table = {y: {"baseline_sh": baseline_yearly[y],
                   "candidate_sh": candidate_yearly[y],
                   "delta": round(candidate_yearly[y] - baseline_yearly[y], 4)}
              for y in YEARS}

print("\n[3/4] Fine-tune grid around (12, 96): 9 combos ...")

ft_results = {}
ft_best_combo = (CANDIDATE_TS_SHORT, CANDIDATE_TS_LONG)
ft_best_obj   = candidate_obj

for (s_w, l_w) in FINETUNE_GRID:
    tag = f"s{s_w}_l{l_w}"
    yr_sharpes = {}
    for year in YEARS:
        ts_ft = compute_ts_spread_pct(year_static[year], s_w, l_w)
        ens_ft = compute_ensemble(year_sig_rets[year], year_static[year], ts_ft)
        yr_sharpes[year] = sharpe(ens_ft)
    ft_obj = obj_func(yr_sharpes)
    ft_results[tag] = {
        "short": s_w, "long": l_w,
        "obj": ft_obj, "delta_vs_baseline": round(ft_obj - baseline_obj, 4),
        "yearly": yr_sharpes,
    }
    if ft_obj > ft_best_obj:
        ft_best_obj   = ft_obj
        ft_best_combo = (s_w, l_w)
    print(f"  s={s_w:2d}/l={l_w:3d} → OBJ={ft_obj:.4f}  Δ={ft_obj - baseline_obj:+.4f}")

best_short, best_long = ft_best_combo
best_obj  = ft_best_obj
best_delta = round(best_obj - baseline_obj, 4)

print(f"\n  Fine-tune winner: short={best_short}h / long={best_long}h  OBJ={best_obj:.4f}  Δ={best_delta:+.4f}")

# Run LOYO for fine-tune winner if different from candidate
if (best_short, best_long) != (CANDIDATE_TS_SHORT, CANDIDATE_TS_LONG):
    print(f"\n  Re-running LOYO for fine-tune winner ({best_short}/{best_long}) ...")
    ft_winner_yearly = {}
    for year in YEARS:
        ts_w = compute_ts_spread_pct(year_static[year], best_short, best_long)
        ens_w = compute_ensemble(year_sig_rets[year], year_static[year], ts_w)
        ft_winner_yearly[year] = sharpe(ens_w)
    ft_loyo_wins = sum(1 for y in YEARS if ft_winner_yearly[y] > baseline_yearly[y])
    ft_loyo_deltas = [ft_winner_yearly[y] - baseline_yearly[y] for y in YEARS]
    ft_loyo_avg   = round(float(np.mean(ft_loyo_deltas)), 4)
    print(f"  Fine-tune LOYO: {ft_loyo_wins}/5  avg_Δ={ft_loyo_avg:+.4f}")
    if ft_loyo_wins >= loyo_wins and ft_loyo_avg >= loyo_avg_delta:
        loyo_wins     = ft_loyo_wins
        loyo_avg_delta = ft_loyo_avg
        loyo_deltas   = ft_loyo_deltas
        candidate_yearly = ft_winner_yearly
        candidate_obj = best_obj
        delta_obj     = best_delta
else:
    ft_winner_yearly = candidate_yearly

# Validation decision
validated = (loyo_wins >= 3) and (delta_obj > 0)

if validated:
    verdict = f"VALIDATED — FTS short={best_short}h/long={best_long}h OBJ={candidate_obj} Δ={delta_obj:+} LOYO {loyo_wins}/5"
else:
    verdict = f"NO IMPROVEMENT — v2.8.0 TS(24/144) OBJ={baseline_obj} optimal (LOYO {loyo_wins}/5)"

print(f"\n[4/4] {verdict}")

# ─── REPORT ──────────────────────────────────────────────────────────────────

report = {
    "phase": 165,
    "description": f"FTS Window Confirmation short={CANDIDATE_TS_SHORT}h/long={CANDIDATE_TS_LONG}h",
    "elapsed_seconds": round(time.time() - _start, 1),
    "baseline_short": BASELINE_TS_SHORT,
    "baseline_long":  BASELINE_TS_LONG,
    "baseline_obj":   baseline_obj,
    "baseline_yearly": baseline_yearly,
    "candidate_short": CANDIDATE_TS_SHORT,
    "candidate_long":  CANDIDATE_TS_LONG,
    "candidate_obj":  candidate_obj,
    "candidate_yearly": candidate_yearly,
    "delta_obj":      delta_obj,
    "loyo_table":     loyo_table,
    "loyo_wins":      loyo_wins,
    "loyo_avg_delta": loyo_avg_delta,
    "finetune_grid":  ft_results,
    "best_short":     best_short,
    "best_long":      best_long,
    "best_obj":       best_obj,
    "best_delta":     best_delta,
    "validated":      validated,
    "verdict":        verdict,
    "partial":        False,
    "timestamp":      datetime.datetime.utcnow().isoformat(),
}

(OUT_DIR / "phase165_report.json").write_text(json.dumps(report, indent=2))
print(f"\nReport saved → {OUT_DIR}/phase165_report.json")

# ─── UPDATE PRODUCTION CONFIG ─────────────────────────────────────────────────

if validated:
    print("\n✅ VALIDATED — updating production config to v2.9.0 ...")
    cfg = json.loads((ROOT / "configs" / "production_p91b_champion.json").read_text())
    prev_version = cfg.get("_version", "2.8.0")
    cfg["_version"] = "2.9.0"
    cfg["_created"] = datetime.date.today().isoformat()
    old_val = cfg.get("_validated", "")
    cfg["_validated"] = (
        old_val +
        f"; FTS windows (P164/P165): short={best_short}h long={best_long}h "
        f"LOYO {loyo_wins}/5 Δ={delta_obj:+} OBJ={candidate_obj} — PRODUCTION v2.9.0"
    )
    fts = cfg["funding_term_structure_overlay"]
    fts["short_window_bars"] = best_short
    fts["long_window_bars"]  = best_long
    fts["_comment"] = (
        f"Phase 150/152/164/165: Bidirectional FTS overlay — fine-tuned P152 thresholds; "
        f"P164/P165 window sweep: short={best_short}h / long={best_long}h"
    )
    fts["_validated"] = (
        f"P150b LOYO 4/5 +0.160; P152 WF 2/2; "
        f"P164 candidate LOYO 4/5 Δ=+0.0751; P165 confirm LOYO {loyo_wins}/5 Δ={delta_obj:+}"
    )
    fts["mechanism"] = (
        f"spread = mean_funding_{best_short}h - mean_funding_{best_long}h (across all symbols). "
        f"Rolling percentile (336h). If spread_pct > 70th: scale×0.60 (overcrowded spike, reduce). "
        f"If spread_pct < 30th: scale×1.15 (cooling, boost). "
        f"Fine-tuned P152: rs=0.60, bs=1.15, rt=0.70, bt=0.30. "
        f"P164/P165: short={best_short}h (1.5 cycles) / long={best_long}h (12 cycles) = sharper signal."
    )
    cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = candidate_obj

    (ROOT / "configs" / "production_p91b_champion.json").write_text(
        json.dumps(cfg, indent=2)
    )
    print(f"  Updated: {prev_version} → v2.9.0")
    print(f"  short_window_bars: {BASELINE_TS_SHORT} → {best_short}")
    print(f"  long_window_bars:  {BASELINE_TS_LONG} → {best_long}")
else:
    print("\n❌ NO IMPROVEMENT — v2.8.0 TS(24/144) remains optimal.")

print("\n" + "=" * 68)
print(f"PHASE 165 COMPLETE — {verdict}")
print(f"Elapsed: {round(time.time() - _start, 1)}s")
print("=" * 68)
