#!/usr/bin/env python3
"""
Phase 155: 5th Signal Hunt — RS Acceleration + Lead-Lag + Dispersion
======================================================================
Production state: OBJ≈2.07 (v2.6.0: F168 + breadth + vol + fund_disp + fund_ts)
Phase 154: Taker buy alpha FAILS (LOYO 1/5). Volume-flow not orthogonal.

Hypothesis: Structural signals from price micro-dynamics are orthogonal.
  Current ensemble: carry (F168), idio-momentum (I415, I460), multi-factor (V1)
  ALL are medium/long horizon signals.
  Gap: SHORT-HORIZON price dynamics not captured.

Candidates (pure price, no additional data needed):
  1. rs_acceleration_alpha — RS_short(72h) - RS_long(336h) = momentum CHANGE
     Tests: 3 lookback combinations + k=2 or k=3
  2. lead_lag_alpha — BTC 12h return × altcoin beta → trade direction
     Tests: btc_lookback 4h/8h/12h
  3. dispersion_alpha — 72h cross-sectional momentum filtered by dispersion
     Tests: standard params + k=2/3

Blend: best variant at 10% weight (reduce all others proportionally)
Pass: IS OBJ > 2.0761 AND LOYO >= 3/5
"""
import json
import os
import signal as _signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from nexus_quant.backtest.costs import cost_model_from_config
from nexus_quant.backtest.engine import BacktestConfig, BacktestEngine
from nexus_quant.data.providers.registry import make_provider
from nexus_quant.strategies.registry import make_strategy

_partial: dict = {}
_start = time.time()


def _on_timeout(signum, frame):
    print("\n⏰ TIMEOUT — saving partial...")
    _save(_partial, partial=True)
    sys.exit(0)


_signal.signal(_signal.SIGALRM, _on_timeout)
_signal.alarm(3000)  # 50min

PROD_CFG = json.loads((ROOT / "configs" / "production_p91b_champion.json").read_text())
SYMBOLS = PROD_CFG["data"]["symbols"]
SIGNAL_DEFS = PROD_CFG["ensemble"]["signals"]

VOL_OV = PROD_CFG.get("vol_regime_overlay", {})
VOL_WINDOW     = VOL_OV.get("window_bars", 168)
VOL_THRESHOLD  = VOL_OV.get("threshold", 0.5)
VOL_SCALE      = VOL_OV.get("scale_factor", 0.5)
VOL_F144_BOOST = VOL_OV.get("f144_boost", 0.2)

BRS = PROD_CFG.get("breadth_regime_switching", {})
BREADTH_LB = BRS.get("breadth_lookback_bars", 168)
PCT_WINDOW = BRS.get("rolling_percentile_window", 336)
P_LOW      = BRS.get("p_low", 0.33)
P_HIGH     = BRS.get("p_high", 0.67)

FDO = PROD_CFG.get("funding_dispersion_overlay", {})
FUND_DISP_PCT   = FDO.get("boost_threshold_pct", 0.75)
FUND_DISP_SCALE = FDO.get("boost_scale", 1.15)

FTS = PROD_CFG.get("funding_term_structure_overlay", {})
FTS_SHORT_W = FTS.get("short_window_bars", 24)
FTS_LONG_W  = FTS.get("long_window_bars", 144)
FTS_PCT_WIN = FTS.get("rolling_percentile_window", 336)
FTS_RT      = FTS.get("reduce_threshold", 0.70)
FTS_RS      = FTS.get("reduce_scale", 0.60)
FTS_BT      = FTS.get("boost_threshold", 0.30)
FTS_BS      = FTS.get("boost_scale", 1.15)

WEIGHTS_4 = {
    "prod":  {"v1": 0.2747, "i460bw168": 0.1967, "i415bw216": 0.3247, "f144": 0.2039},
    "p143b": {"v1": 0.05,   "i460bw168": 0.25,   "i415bw216": 0.45,   "f144": 0.25},
    "mid":   {"v1": 0.16,   "i460bw168": 0.22,   "i415bw216": 0.37,   "f144": 0.25},
}

# 5-signal weights: add 5th at 10%, reduce others by 10/90 proportionally
def make_5sig_weights(base_weights: dict, fifth_key: str = "sig5") -> dict:
    result = {}
    for regime, w in base_weights.items():
        total = sum(w.values())
        new_w = {k: round(v * 0.9, 6) for k, v in w.items()}
        new_w[fifth_key] = 0.10
        result[regime] = new_w
    return result

WEIGHTS_5 = make_5sig_weights(WEIGHTS_4, "sig5")

YEAR_RANGES = {
    "2021": ("2021-02-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
}
YEARS = sorted(YEAR_RANGES.keys())

BASELINE_OBJ = 2.0761  # F168 on v2.5.0 from P153

OUT_DIR = ROOT / "artifacts" / "phase155"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

# Candidate 5th signals to test
CANDIDATES = [
    # RS Acceleration
    {"label": "rsa_72_336_k2", "strategy": "rs_acceleration_alpha",
     "params": {"k_per_side": 2, "rs_short_bars": 72, "rs_long_bars": 336,
                "vol_lookback_bars": 168, "target_gross_leverage": 0.25,
                "rebalance_interval_bars": 24}},
    {"label": "rsa_48_240_k2", "strategy": "rs_acceleration_alpha",
     "params": {"k_per_side": 2, "rs_short_bars": 48, "rs_long_bars": 240,
                "vol_lookback_bars": 168, "target_gross_leverage": 0.25,
                "rebalance_interval_bars": 24}},
    {"label": "rsa_96_336_k3", "strategy": "rs_acceleration_alpha",
     "params": {"k_per_side": 3, "rs_short_bars": 96, "rs_long_bars": 336,
                "vol_lookback_bars": 168, "target_gross_leverage": 0.25,
                "rebalance_interval_bars": 24}},
    # Lead-Lag
    {"label": "ll_btc12_k2", "strategy": "lead_lag_alpha",
     "params": {"k_per_side": 2, "btc_lookback_bars": 12, "beta_lookback_bars": 168,
                "vol_lookback_bars": 168, "target_gross_leverage": 0.25,
                "rebalance_interval_bars": 24}},
    {"label": "ll_btc8_k2", "strategy": "lead_lag_alpha",
     "params": {"k_per_side": 2, "btc_lookback_bars": 8, "beta_lookback_bars": 168,
                "vol_lookback_bars": 168, "target_gross_leverage": 0.25,
                "rebalance_interval_bars": 24}},
    {"label": "ll_btc4_k2", "strategy": "lead_lag_alpha",
     "params": {"k_per_side": 2, "btc_lookback_bars": 4, "beta_lookback_bars": 168,
                "vol_lookback_bars": 168, "target_gross_leverage": 0.25,
                "rebalance_interval_bars": 24}},
    # Dispersion
    {"label": "disp_72_k2", "strategy": "dispersion_alpha",
     "params": {"k_per_side": 2, "momentum_lookback_bars": 72,
                "dispersion_lookback_bars": 72, "dispersion_threshold": 0.0,
                "target_gross_leverage": 0.20, "rebalance_interval_bars": 24}},
    {"label": "disp_48_k2", "strategy": "dispersion_alpha",
     "params": {"k_per_side": 2, "momentum_lookback_bars": 48,
                "dispersion_lookback_bars": 72, "dispersion_threshold": 0.0,
                "target_gross_leverage": 0.20, "rebalance_interval_bars": 24}},
]


def sharpe(rets: np.ndarray) -> float:
    if len(rets) < 100:
        return 0.0
    mu = float(np.mean(rets)) * 8760
    sd = float(np.std(rets)) * np.sqrt(8760)
    return round(mu / sd, 4) if sd > 1e-12 else 0.0


def obj_func(yearly_sharpes: list) -> float:
    arr = np.array(yearly_sharpes)
    return round(float(np.mean(arr) - 0.5 * np.std(arr)), 4)


def _save(data: dict, partial: bool = False) -> None:
    data["partial"] = partial
    data["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out = OUT_DIR / "phase155_report.json"
    out.write_text(json.dumps(data, indent=2, default=str))
    print(f"✅ Saved → {out}")


def rolling_percentile(signal: np.ndarray, window: int) -> np.ndarray:
    n = len(signal)
    pct = np.full(n, 0.5)
    for i in range(window, n):
        hist = signal[i - window:i]
        pct[i] = float(np.mean(hist <= signal[i]))
    if window < n:
        pct[:window] = pct[window]
    return pct


def compute_btc_vol(dataset) -> np.ndarray:
    n = len(dataset.timeline)
    rets = np.zeros(n)
    for i in range(1, n):
        c0 = dataset.close("BTCUSDT", i - 1)
        c1 = dataset.close("BTCUSDT", i)
        rets[i] = (c1 / c0 - 1.0) if c0 > 0 else 0.0
    vol = np.full(n, np.nan)
    for i in range(VOL_WINDOW, n):
        vol[i] = float(np.std(rets[i - VOL_WINDOW:i])) * np.sqrt(8760)
    if VOL_WINDOW < n:
        vol[:VOL_WINDOW] = vol[VOL_WINDOW]
    return vol


def compute_breadth(dataset) -> np.ndarray:
    n = len(dataset.timeline)
    breadth = np.full(n, 0.5)
    for i in range(BREADTH_LB, n):
        pos = sum(
            1 for sym in SYMBOLS
            if (c0 := dataset.close(sym, i - BREADTH_LB)) > 0
            and dataset.close(sym, i) > c0
        )
        breadth[i] = pos / len(SYMBOLS)
    if BREADTH_LB < n:
        breadth[:BREADTH_LB] = breadth[BREADTH_LB]
    return breadth


def compute_fund_disp(dataset) -> np.ndarray:
    n = len(dataset.timeline)
    fund_std = np.zeros(n)
    for i in range(n):
        ts = dataset.timeline[i]
        rates = []
        for sym in SYMBOLS:
            try:
                r = dataset.last_funding_rate_before(sym, ts)
                if r is not None and not np.isnan(float(r)):
                    rates.append(float(r))
            except Exception:
                pass
        fund_std[i] = float(np.std(rates)) if len(rates) > 1 else 0.0
    return fund_std


def compute_fund_ts(dataset) -> np.ndarray:
    n = len(dataset.timeline)
    level = np.zeros(n)
    for i in range(n):
        ts = dataset.timeline[i]
        rates = []
        for sym in SYMBOLS:
            try:
                r = dataset.funding_rate_at(sym, ts)
                if r is not None and not np.isnan(float(r)):
                    rates.append(float(r))
            except Exception:
                pass
        level[i] = float(np.mean(rates)) if rates else 0.0
    spread = np.zeros(n)
    for i in range(FTS_LONG_W, n):
        s_avg = float(np.mean(level[max(0, i - FTS_SHORT_W):i]))
        l_avg = float(np.mean(level[i - FTS_LONG_W:i]))
        spread[i] = s_avg - l_avg
    if FTS_LONG_W < n:
        spread[:FTS_LONG_W] = spread[FTS_LONG_W]
    return spread


def blend4(sig_rets: dict, bv, brd_pct, fdo_pct, fts_pct) -> np.ndarray:
    """4-signal baseline ensemble with full v2.6.0 overlays."""
    sig_keys = ["v1", "i460bw168", "i415bw216", "f144"]
    min_len = min(len(sig_rets[sk]) for sk in sig_keys)
    bv_  = bv[:min_len];  brd_ = brd_pct[:min_len]
    fdo_ = fdo_pct[:min_len];  fts_ = fts_pct[:min_len]
    ens  = np.zeros(min_len)
    n_others = 3

    for i in range(min_len):
        if brd_[i] >= P_HIGH:
            w = WEIGHTS_4["p143b"]
        elif brd_[i] >= P_LOW:
            w = WEIGHTS_4["mid"]
        else:
            w = WEIGHTS_4["prod"]

        if not np.isnan(bv_[i]) and bv_[i] > VOL_THRESHOLD:
            boost_other = VOL_F144_BOOST / max(1, n_others)
            ret_i = 0.0
            for sk in sig_keys:
                adj_w = min(0.60, w[sk] + VOL_F144_BOOST) if sk == "f144" else max(0.05, w[sk] - boost_other)
                ret_i += adj_w * sig_rets[sk][i]
            ret_i *= VOL_SCALE
        else:
            ret_i = sum(w[sk] * sig_rets[sk][i] for sk in sig_keys)

        if fdo_[i] > FUND_DISP_PCT:
            ret_i *= FUND_DISP_SCALE
        sp = fts_[i]
        if sp >= FTS_RT:
            ret_i *= FTS_RS
        elif sp <= FTS_BT:
            ret_i *= FTS_BS

        ens[i] = ret_i

    return ens


def blend5(sig_rets: dict, sig5_rets: np.ndarray, bv, brd_pct, fdo_pct, fts_pct) -> np.ndarray:
    """5-signal ensemble (4 base + 1 candidate at 10%)."""
    base_keys = ["v1", "i460bw168", "i415bw216", "f144"]
    min_len = min(min(len(sig_rets[sk]) for sk in base_keys), len(sig5_rets))
    bv_  = bv[:min_len];  brd_ = brd_pct[:min_len]
    fdo_ = fdo_pct[:min_len];  fts_ = fts_pct[:min_len]
    ens  = np.zeros(min_len)
    n_others = 4  # now 5 signals total, 4 others vs f144

    for i in range(min_len):
        if brd_[i] >= P_HIGH:
            w = WEIGHTS_5["p143b"]
        elif brd_[i] >= P_LOW:
            w = WEIGHTS_5["mid"]
        else:
            w = WEIGHTS_5["prod"]

        if not np.isnan(bv_[i]) and bv_[i] > VOL_THRESHOLD:
            boost_other = VOL_F144_BOOST / max(1, n_others)
            ret_i = 0.0
            for sk in base_keys:
                adj_w = min(0.60, w[sk] + VOL_F144_BOOST) if sk == "f144" else max(0.0, w[sk] - boost_other)
                ret_i += adj_w * sig_rets[sk][i]
            ret_i += max(0.0, w.get("sig5", 0.0) - boost_other) * sig5_rets[i]
            ret_i *= VOL_SCALE
        else:
            ret_i = sum(w[sk] * sig_rets[sk][i] for sk in base_keys)
            ret_i += w.get("sig5", 0.0) * sig5_rets[i]

        if fdo_[i] > FUND_DISP_PCT:
            ret_i *= FUND_DISP_SCALE
        sp = fts_[i]
        if sp >= FTS_RT:
            ret_i *= FTS_RS
        elif sp <= FTS_BT:
            ret_i *= FTS_BS

        ens[i] = ret_i

    return ens


def main():
    global _partial

    print("=" * 72)
    print(f"PHASE 155: 5th Signal Hunt — {len(CANDIDATES)} candidates")
    print(f"  Baseline: 4-signal v2.6.0 stack, reference OBJ≈{BASELINE_OBJ}")
    print("=" * 72)

    # ── Load data ────────────────────────────────────────────────────────
    print("\n[1/3] Loading datasets + precomputing overlays + base signals...")
    datasets = {}
    btc_vols, breadth_raw, fund_disp_raw, fts_raw = {}, {}, {}, {}
    base_sig_rets: dict = {sk: {} for sk in ["v1", "i460bw168", "i415bw216", "f144"]}

    for year in YEARS:
        start, end = YEAR_RANGES[year]
        print(f"  {year}: ", end="", flush=True)
        cfg_data = {
            "provider": "binance_rest_v1", "symbols": SYMBOLS,
            "start": start, "end": end, "bar_interval": "1h",
            "cache_dir": ".cache/binance_rest",
        }
        ds = make_provider(cfg_data, seed=42).load()
        datasets[year]      = ds
        btc_vols[year]      = compute_btc_vol(ds)
        breadth_raw[year]   = compute_breadth(ds)
        fund_disp_raw[year] = compute_fund_disp(ds)
        fts_raw[year]       = compute_fund_ts(ds)

        for sk in ["v1", "i460bw168", "i415bw216"]:
            sig_def = SIGNAL_DEFS[sk]
            bt_cfg = BacktestConfig(costs=COST_MODEL)
            strat = make_strategy({"name": sig_def["strategy"], "params": dict(sig_def["params"])})
            result = BacktestEngine(bt_cfg).run(ds, strat)
            base_sig_rets[sk][year] = np.array(result.returns, dtype=np.float64)
            print(".", end="", flush=True)

        # F168 production
        f_params = dict(SIGNAL_DEFS["f144"]["params"])
        bt_cfg = BacktestConfig(costs=COST_MODEL)
        strat = make_strategy({"name": SIGNAL_DEFS["f144"]["strategy"], "params": f_params})
        result = BacktestEngine(bt_cfg).run(ds, strat)
        base_sig_rets["f144"][year] = np.array(result.returns, dtype=np.float64)
        print("f ✓")

    # ── Compute base OBJ ─────────────────────────────────────────────────
    print("\n[2/3] Computing base (4-signal) OBJ...")
    base_yr_sharpes = {}
    for year in YEARS:
        brd = rolling_percentile(breadth_raw[year], PCT_WINDOW)
        fdo = rolling_percentile(fund_disp_raw[year], PCT_WINDOW)
        fts = rolling_percentile(fts_raw[year], FTS_PCT_WIN)
        ens = blend4({sk: base_sig_rets[sk][year] for sk in base_sig_rets}, btc_vols[year], brd, fdo, fts)
        base_yr_sharpes[year] = sharpe(ens)
    base_obj = obj_func(list(base_yr_sharpes.values()))
    print(f"  Base OBJ={base_obj:.4f} | {base_yr_sharpes}")

    # ── Test each candidate ──────────────────────────────────────────────
    print("\n[3/3] Testing 5th signal candidates...")
    candidate_results = {}

    for cand in CANDIDATES:
        label    = cand["label"]
        strategy = cand["strategy"]
        params   = cand["params"]
        print(f"\n  [{label}] {strategy}...")

        yr_sharpes_5 = {}
        for year in YEARS:
            brd = rolling_percentile(breadth_raw[year], PCT_WINDOW)
            fdo = rolling_percentile(fund_disp_raw[year], PCT_WINDOW)
            fts = rolling_percentile(fts_raw[year], FTS_PCT_WIN)

            bt_cfg = BacktestConfig(costs=COST_MODEL)
            strat = make_strategy({"name": strategy, "params": params})
            try:
                result = BacktestEngine(bt_cfg).run(datasets[year], strat)
                sig5_rets = np.array(result.returns, dtype=np.float64)
            except Exception as e:
                print(f"    {year}: ERROR {e}")
                yr_sharpes_5[year] = base_yr_sharpes[year]  # fallback to base
                continue

            ens5 = blend5({sk: base_sig_rets[sk][year] for sk in base_sig_rets},
                          sig5_rets, btc_vols[year], brd, fdo, fts)
            yr_sharpes_5[year] = sharpe(ens5)

        obj5 = obj_func(list(yr_sharpes_5.values()))
        delta = round(obj5 - base_obj, 4)
        loyo_wins = sum(1 for yr in YEARS if yr_sharpes_5[yr] > base_yr_sharpes[yr])
        flag = " ✅" if obj5 > base_obj and loyo_wins >= 3 else ""
        print(f"    OBJ={obj5:.4f} Δ={delta:+.4f} LOYO={loyo_wins}/5{flag}")
        for yr in YEARS:
            d = round(yr_sharpes_5[yr] - base_yr_sharpes[yr], 4)
            print(f"    {yr}: base={base_yr_sharpes[yr]:.4f} 5sig={yr_sharpes_5[yr]:.4f} Δ={d:+.4f}")

        candidate_results[label] = {
            "strategy": strategy, "params": params,
            "yearly": yr_sharpes_5, "obj": obj5, "delta": delta, "loyo_wins": loyo_wins,
            "pass": bool(obj5 > base_obj and loyo_wins >= 3),
        }
        _partial.update({"base_obj": base_obj, "candidates": candidate_results})
        _save(_partial, partial=True)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("5TH SIGNAL HUNT SUMMARY:")
    print(f"  Base OBJ={base_obj:.4f} | {base_yr_sharpes}")
    print()

    winners = [(l, d) for l, d in candidate_results.items() if d["pass"]]
    ranked = sorted(candidate_results.items(), key=lambda x: -x[1]["delta"])

    for label, d in ranked:
        flag = " ✅ WINNER" if d["pass"] else ""
        print(f"  {label:20s} OBJ={d['obj']:.4f} Δ={d['delta']:+.4f} LOYO={d['loyo_wins']}/5{flag}")

    if winners:
        best = sorted(winners, key=lambda x: -x[1]["delta"])[0]
        best_label, best_data = best
        verdict = (
            f"WINNER FOUND: {best_label} OBJ={best_data['obj']:.4f} "
            f"(+{best_data['delta']:.4f} vs base), LOYO={best_data['loyo_wins']}/5"
        )
        next_phase = (
            f"Phase 156: LOYO+WF validation of {best_label}. "
            f"If passes → add as 5th signal at 10% weight, update prod config."
        )
    else:
        best_label = ranked[0][0] if ranked else None
        best_data  = ranked[0][1] if ranked else {}
        verdict = (
            f"NO 5TH SIGNAL FOUND — all candidates fail LOYO>=3 threshold. "
            f"Best: {best_label} Δ={best_data.get('delta',0):+.4f} LOYO={best_data.get('loyo_wins',0)}/5. "
            f"v2.6.0 is the signal floor — diminishing returns on 5th signal."
        )
        next_phase = (
            "Phase 156: Accept v2.6.0 as production-ready. "
            "Deploy focus: (a) live paper trading, (b) risk management tuning, "
            "(c) post-live performance monitoring vs backtest expectation. "
            "OR explore regime-specific signal activation (only run 5th signal in certain regimes)."
        )

    print(f"\nVERDICT: {verdict}")
    print(f"NEXT: {next_phase}")
    print("=" * 72)

    report = {
        "phase": 155,
        "description": "5th signal hunt: RS acceleration + lead-lag + dispersion",
        "elapsed_seconds": round(time.time() - _start, 1),
        "base_yearly": base_yr_sharpes,
        "base_obj": base_obj,
        "candidates_tested": len(CANDIDATES),
        "candidates": candidate_results,
        "winners": [l for l, _ in winners],
        "verdict": verdict,
        "next_phase_notes": next_phase,
    }
    _save(report, partial=False)
    return report


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\n❌ ERROR: {e}")
        traceback.print_exc()
        _partial["error"] = str(e)
        _partial["traceback"] = traceback.format_exc()
        _save(_partial, partial=True)
        sys.exit(1)
